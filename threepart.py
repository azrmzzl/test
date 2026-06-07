import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse, to_dense_adj

# 设置随机种子，保证结果可复现
torch.manual_seed(2025)
np.random.seed(2025)


# ==========================================
# 1. 论文特定数据模拟器 (Bus-Line Graph Simulator)
# ==========================================
class ThesisGraphData:
    def __init__(self, num_buses=14, num_lines=20):
        self.num_buses = num_buses
        self.num_lines = num_lines
        self.num_nodes = num_buses + num_lines

        # --- 构建基础拓扑 (Base Topology) ---
        # 模拟 IEEE 14 节点的连接关系
        # 邻接矩阵 A: [N, N], 包含 Bus-Bus, Bus-Line, Line-Line
        # 这里简化：重点构建 Bus-Line 连接，用于计算 KCL 物理损失
        self.adj = torch.eye(self.num_nodes)  # 自环

        # 随机生成连接关系：每个 Line 连接两个 Bus
        self.line_connections = []  # 记录 (Line_idx, Bus_from, Bus_to)

        for l in range(num_lines):
            line_node_idx = num_buses + l
            # 随机选择两个母线连接
            b1 = np.random.randint(0, num_buses)
            b2 = np.random.randint(0, num_buses)
            while b1 == b2: b2 = np.random.randint(0, num_buses)

            self.line_connections.append((line_node_idx, b1, b2))

            # Bus 与 Line 相连 (双向)
            self.adj[b1, line_node_idx] = 1
            self.adj[line_node_idx, b1] = 1
            self.adj[b2, line_node_idx] = 1
            self.adj[line_node_idx, b2] = 1

    def get_data(self, num_samples, case='normal'):
        """
        [升级版] 生成全维度物理特征数据 X
        X 维度: [Batch, N, 4]

        特征定义 (根据你的论文):
        - Bus节点 (前 num_buses 个): [Index 0: P, Index 1: Q, Index 2: V, Index 3: theta]
          (注意：为了方便矩阵计算，我把 P, Q 放在前两位，和你论文略有不同，只需在 Loss 里对应即可)
        - Line节点 (后 num_lines 个): [Index 0: P, Index 1: Q, Index 2: I_re, Index 3: I_im]
        """
        # 1. 处理拓扑 (N-1 故障模拟)
        current_adj = self.adj.clone()

        # if case == 'N-1':
        #     # 假设第 0 条支路断开 (Line Node 索引 = num_buses)
        #     fault_line_node = self.num_buses + 0
        #     # 物理动作：切断连接
        #     current_adj[fault_line_node, :] = 0
        #     current_adj[:, fault_line_node] = 0
        #     current_adj[fault_line_node, fault_line_node] = 1
        #     print(f">>> [场景: {case}] 支路节点 {fault_line_node} 已断开物理连接。")

        # [修改后代码] 替换上面那段为：
        if case == 'N-1':
            # 假设断开第 0 条线 (索引 num_buses + 0)
            fault_line_idx = 0
            fault_node = self.num_buses + fault_line_idx

            # 1. 物理动作：切断连接
            current_adj[fault_node, :] = 0
            current_adj[:, fault_node] = 0
            current_adj[fault_node, fault_node] = 1

            # 2. [新增] 模拟潮流转移 (Redistribution)
            # 现实中：Line 0 断了，它的流量会分摊到并行或附近的 Line 1, Line 2 上
            # 我们获取 Line 0原本的基础流量值
            lost_flow = line_p_base[0, fault_line_idx]

            # 强制让 Line 1 和 Line 2 分担这个流量 (简单模拟)
            # 对应节点索引：num_buses+1, num_buses+2
            x[:, self.num_buses + 1, 0] += lost_flow * 0.6  # Line 1 分担 60%
            x[:, self.num_buses + 2, 0] += lost_flow * 0.4  # Line 2 分担 40%

            print(f">>> [场景: {case}] 支路 {fault_node} 断开，潮流已物理转移至邻近线路。")

        edge_index, _ = dense_to_sparse(current_adj)

        # =========================
        # 额外输出：Incidence Matrix B_inc (带符号)
        # B_inc: [num_buses, num_lines]
        # 对每条线路：from bus = -1, to bus = +1
        # =========================
        B_inc = torch.zeros(self.num_buses, self.num_lines)

        for l_rel in range(self.num_lines):
            line_node_idx = self.num_buses + l_rel
            # 从初始化保存的连接中读取 (line_node_idx, b1, b2)
            _, b1, b2 = self.line_connections[l_rel]

            # 约定方向：b1 -> b2
            B_inc[b1, l_rel] = -1.0
            B_inc[b2, l_rel] = +1.0


        # 2. 初始化数据容器
        x = torch.zeros(num_samples, self.num_nodes, 4)
        t = torch.linspace(0, 10, num_samples).view(-1, 1)  # 时间轴

        # =========================================
        # A. 模拟 Line 特征 [P, Q, I_re, I_im]
        # =========================================
        # 模拟 Line P (有功): 正弦波动
        line_p_base = torch.randn(1, self.num_lines)
        x[:, self.num_buses:, 0] = line_p_base + 0.5 * torch.sin(t)

        # 模拟 Line Q (无功): 余弦波动 (模拟与 P 的不同步)
        line_q_base = torch.randn(1, self.num_lines) * 0.2  # 无功通常比有功小
        x[:, self.num_buses:, 1] = line_q_base + 0.1 * torch.cos(t)

        # 模拟 I_re, I_im (简单模拟数值，保持非零)
        x[:, self.num_buses:, 2] = x[:, self.num_buses:, 0] / 1.0  # 粗略 I ~ P/V
        x[:, self.num_buses:, 3] = x[:, self.num_buses:, 1] / 1.0

        # =========================================
        # B. 模拟 Bus 特征 [P, Q, V, theta]
        # =========================================
        # 模拟 Bus V (电压): 在 1.0 p.u. 附近微小波动
        x[:, :self.num_buses, 2] = 1.0 + 0.02 * torch.randn(num_samples, self.num_buses)

        # 模拟 Bus theta (相角): 在 0 附近微小波动
        x[:, :self.num_buses, 3] = 0.0 + 0.05 * torch.randn(num_samples, self.num_buses)

        # [核心物理约束] 模拟 Bus P/Q: 与 B_inc 完全一致的方向
        # 方向唯一来源：self.line_connections (b_from -> b_to)
        for l_idx_rel in range(self.num_lines):
            l_node_idx = self.num_buses + l_idx_rel

            # ✅ 不要再用 nonzero() 找连接母线
            _, b_from, b_to = self.line_connections[l_idx_rel]

            # 获取该 Line 的 P 和 Q
            val_p = x[:, l_node_idx, 0]
            val_q = x[:, l_node_idx, 1]

            # 如果是 N-1 且该线路断了，则流量归零
            if case == 'N-1' and l_idx_rel == 0:
                val_p = torch.zeros_like(val_p)
                val_q = torch.zeros_like(val_q)
                x[:, l_node_idx, :] = 0  # 故障线路全维度归零

            # ✅ 与 B_inc 完全一致：from=-1, to=+1
            x[:, b_from, 0] -= val_p
            x[:, b_to, 0] += val_p
            x[:, b_from, 1] -= val_q
            x[:, b_to, 1] += val_q

        # 加上微量测量噪声
        x += 0.001 * torch.randn_like(x)

        return x, edge_index, current_adj, B_inc



# ==========================================
# 2. 核心模型: PI-GraphMAE (修正版：GNN Decoder + Re-masking)
# ==========================================
class PI_GraphMAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4):
        super(PI_GraphMAE, self).__init__()

        # [Token 1] 编码器掩码令牌 (用于输入层)
        self.enc_mask_token = nn.Parameter(torch.randn(1, 1, in_channels))
        # [Token 2] 解码器掩码令牌 (用于 Re-masking)
        self.dec_mask_token = nn.Parameter(torch.randn(1, 1, hidden_channels))

        # --- Encoder: 多头 GAT ---
        # 第一层: 提取特征
        self.enc1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        # 第二层: 聚合特征 (输出维度调整回 hidden_channels)
        self.enc2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False)

        # --- Decoder: GAT (修正点) ---
        # 之前是 MLP，现在改为 GAT，利用图拓扑进行物理推断
        # 这是一个单层 GAT，专门用于从邻居聚合信息来填补空缺
        self.decoder_gat = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=True)

        # 输出头: 将 GAT 的输出映射回物理量纲 (P, Q, V, theta)
        self.decoder_head = nn.Linear(hidden_channels * num_heads, out_channels)

    def forward(self, x, edge_index, mask=None):
        """
        前向传播包含: Mask -> Encode -> Re-mask -> Decode
        """
        """
            x: [B, N, F]
            edge_index: [2, E] (单图)
            mask: [B, N, 1]
            """
        B, N, Fdim = x.shape
        device = x.device

        x_in = x.clone()

        # 1) Encoder masking
        if mask is not None:
            x_in = x_in * (1 - mask) + self.enc_mask_token.to(device) * mask  # [B,N,F]

        # 2) 把 batch 摊平为一个大图：节点索引偏移
        x_flat = x_in.reshape(B * N, Fdim)  # [B*N, F]
        E = edge_index.shape[1]

        # 构造 batch 版 edge_index：复制 B 份并偏移
        offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)  # [B,1,1]
        edge_rep = edge_index.view(1, 2, E).repeat(B, 1, 1)  # [B,2,E]
        edge_rep = edge_rep + offsets  # 偏移
        edge_big = edge_rep.reshape(2, B * E)  # [2, B*E]

        # 3) Encoding
        h = F.elu(self.enc1(x_flat, edge_big))
        h = F.elu(self.enc2(h, edge_big))  # [B*N, H]
        Hdim = h.shape[1]
        h = h.view(B, N, Hdim)

        # 4) Re-masking (latent)
        if mask is not None:
            h = h * (1 - mask) + self.dec_mask_token.to(device) * mask  # [B,N,H]

        # 5) Decoder GAT
        h_flat = h.reshape(B * N, Hdim)
        h_rec = F.elu(self.decoder_gat(h_flat, edge_big))  # [B*N, H*heads]
        out = self.decoder_head(h_rec)  # [B*N, out_channels]
        out = out.view(B, N, -1)
        return out


# ==========================================
# 3. 物理损失函数 (KCL Loss) - 核心干货 [修正版]
# ==========================================
# def physics_kcl_loss(recon_x, B_inc, num_buses, mask=None):
#     """
#     线性化 KCL（带符号 incidence matrix）
#     recon_x: [B, N, 4]
#     B_inc:   [nb, nl]   (from=-1, to=+1)
#
#     关键修正：
#     - KCL 聚合时，必须把“被 mask 的线路”的 P/Q 排除掉，否则物理项用的是瞎猜的 line flow
#     - 同时，KCL 只在“未被 mask 的 bus”上计算（可选，但更稳定）
#     """
#     B, N, Fdim = recon_x.shape
#     nb = num_buses
#     nl = N - nb
#
#     bus = recon_x[:, :nb, :]      # [B, nb, 4]
#     line = recon_x[:, nb:, :]     # [B, nl, 4]
#
#     bus_p = bus[:, :, 0]          # [B, nb]
#     bus_q = bus[:, :, 1]          # [B, nb]
#     line_p = line[:, :, 0]        # [B, nl]
#     line_q = line[:, :, 1]        # [B, nl]
#
#     if mask is not None:
#         bus_mask = mask[:, :nb, 0]     # [B, nb]
#         line_mask = mask[:, nb:, 0]    # [B, nl]
#
#         keep_bus = (1.0 - bus_mask)    # 1=参与KCL
#         keep_line = (1.0 - line_mask)  # 1=线路可信
#
#         # ✅ 关键：把被 mask 的线路流量置零，不参与聚合
#         line_p_eff = line_p * keep_line
#         line_q_eff = line_q * keep_line
#     else:
#         keep_bus = None
#         line_p_eff = line_p
#         line_q_eff = line_q
#
#     # 聚合： [B,nl] @ [nl,nb] -> [B,nb]
#     agg_p = torch.matmul(line_p_eff, B_inc.T)
#     agg_q = torch.matmul(line_q_eff, B_inc.T)
#
#     if keep_bus is not None:
#         denom = keep_bus.sum().clamp(min=1.0)
#         loss_p = (((bus_p - agg_p) ** 2) * keep_bus).sum() / denom
#         loss_q = (((bus_q - agg_q) ** 2) * keep_bus).sum() / denom
#     else:
#    都7     loss_p = F.mse_loss(bus_p, agg_p)
#         loss_q = F.mse_loss(bus_q, agg_q)
#
#     return loss_p + loss_q

def physics_kcl_loss(recon_x, B_inc, num_buses, mask=None):
    """
    最简线性化 KCL：
    - 对所有 Bus 计算
    - 不区分 mask / 非 mask
    - 所有 Line（即便被 mask）都参与聚合

    recon_x: [B, N, 4]
    B_inc:   [nb, nl] (from=-1, to=+1)
    """
    B, N, _ = recon_x.shape
    nb = num_buses
    nl = N - nb

    bus = recon_x[:, :nb, :]      # [B, nb, 4]
    line = recon_x[:, nb:, :]     # [B, nl, 4]

    # P / Q
    bus_p = bus[:, :, 0]          # [B, nb]
    bus_q = bus[:, :, 1]          # [B, nb]
    line_p = line[:, :, 0]        # [B, nl]
    line_q = line[:, :, 1]        # [B, nl]

    # KCL 聚合（所有线路都算）
    agg_p = torch.matmul(line_p, B_inc.T)   # [B, nb]
    agg_q = torch.matmul(line_q, B_inc.T)   # [B, nb]

    # [修改前]
    # loss_p = F.mse_loss(bus_p, agg_p)

    # >>> [修改后] 使用归一化 MSE (Relative Error-like)
    # 分母加一个极小值防止除以0，使用 detach() 不让梯度传给分母
    scale_p = bus_p.abs().detach().mean(dim=1, keepdim=True) + 0.1
    scale_q = bus_q.abs().detach().mean(dim=1, keepdim=True) + 0.1

    loss_p = ((bus_p - agg_p) / scale_p).pow(2).mean()
    loss_q = ((bus_q - agg_q) / scale_q).pow(2).mean()

    return loss_p + loss_q


def make_mask(B, N, nb, device, bus_ratio=0.3, line_ratio=0.05):
    """
    bus_ratio:  bus 节点mask比例（可以大）
    line_ratio: line 节点mask比例（必须小，否则KCL用不到真实line flow）
    """
    nl = N - nb
    bus_mask = (torch.rand(B, nb, 1, device=device) < bus_ratio).float()
    line_mask = (torch.rand(B, nl, 1, device=device) < line_ratio).float()
    return torch.cat([bus_mask, line_mask], dim=1)  # [B,N,1]

# ==========================================
# 4. 完整的迁移学习与恢复流程
# ==========================================
def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化数据模拟器
    N_BUS = 14
    N_LINE = 20
    simulator = ThesisGraphData(num_buses=N_BUS, num_lines=N_LINE)

    # ---------------------------------------------------------
    # 阶段一：源域预训练 (Source Pre-training)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> 阶段一：源域预训练 (Pre-training on Source Domain)")
    print("=" * 50)

    # 1. 获取源域数据 (正常拓扑)
    x_src, edge_index_src, adj_src, B_src = simulator.get_data(num_samples=2000, case='normal')
    x_src, edge_index_src, adj_src, B_src = x_src.to(device), edge_index_src.to(device), adj_src.to(device), B_src.to(device)

    with torch.no_grad():
        bus_p = x_src[:, :N_BUS, 0]  # [B, nb]
        line_p = x_src[:, N_BUS:, 0]  # [B, nl]
        agg_p = torch.matmul(line_p, B_src.T)
        print("[CHECK] KCL(P) MSE on generated data =", F.mse_loss(bus_p, agg_p).item())

    # 2. 初始化模型
    model = PI_GraphMAE(in_channels=4, hidden_channels=64, out_channels=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 3. 训练循环
    model.train()
    loss_history_src = []

    for epoch in range(1000):
        optimizer.zero_grad()

        # # 随机 Mask (自监督训练，遮挡 30% 节点)
        mask = torch.rand(x_src.shape[0], x_src.shape[1], 1).to(device) > 0.7
        mask = mask.float()
        # mask = make_mask(x_src.shape[0], x_src.shape[1], N_BUS, device, bus_ratio=0.3, line_ratio=0.05)

        # 前向传播
        recon = model(x_src, edge_index_src, mask)

        # 计算 Loss
        # a. 重构 Loss (只看被 Mask 的部分，这是 MAE 的精髓)
        loss_mse = F.mse_loss(recon * mask, x_src * mask)
        # b. 物理 Loss (KCL 约束)
        loss_phy = physics_kcl_loss(recon, B_src, N_BUS, mask=mask)

        # 物理权重 warm-up：前200轮从0线性涨到0.1
        warm_epochs = 200
        w_phy = 0.1 * min(1.0, epoch / warm_epochs)

        total_loss = loss_mse + w_phy * loss_phy

        total_loss.backward()
        optimizer.step()
        loss_history_src.append(total_loss.item())

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | Loss: {total_loss.item():.5f} (MSE: {loss_mse.item():.5f} + Phy: {loss_phy.item():.5f})")

    # 保存预训练权重
    torch.save(model.state_dict(), '../pi_graphmae_pretrained.pth')
    print(">>> 预训练完成，模型已保存。")

    # ---------------------------------------------------------
    # 阶段二：目标域微调 (Target Fine-tuning)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> 阶段二：目标域微调 (Fine-tuning on Target N-1)")
    print("=" * 50)

    # 1. 获取目标域数据 (N-1 故障, 样本极少, 只有 200 个!)
    x_tgt, edge_index_tgt, adj_tgt, B_tgt = simulator.get_data(num_samples=500, case='N-1')
    x_tgt, edge_index_tgt, adj_tgt, B_tgt = x_tgt.to(device), edge_index_tgt.to(device), adj_tgt.to(device), B_tgt.to(device)

    # 2. 加载预训练模型
    model_ft = PI_GraphMAE(in_channels=4, hidden_channels=64, out_channels=4).to(device)
    model_ft.load_state_dict(torch.load('../pi_graphmae_pretrained.pth'))

    # 3. 冻结 Encoder (迁移策略核心)
    print(">>> [策略] 冻结 Encoder 参数，只微调 Decoder 和 MaskToken...")
    for name, param in model_ft.named_parameters():
        if 'enc' in name:
            param.requires_grad = False

    # 4. 微调训练 (低学习率)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001)
    loss_history_tgt = []

    for epoch in range(200):
        optimizer_ft.zero_grad()
        # # 依然随机 Mask，让模型适应新拓扑下的填空
        mask = torch.rand(x_tgt.shape[0], x_tgt.shape[1], 1).to(device) > 0.7
        mask = mask.float()
        # mask = make_mask(x_tgt.shape[0], x_tgt.shape[1], N_BUS, device, bus_ratio=0.3, line_ratio=0.05)

        # !!! 关键: 输入的是 edge_index_tgt (断线后的拓扑) !!!
        recon = model_ft(x_tgt, edge_index_tgt, mask)

        loss_mse = F.mse_loss(recon * mask, x_tgt * mask)
        # !!! 关键: 物理约束基于 adj_tgt (断线后的KCL) !!!
        loss_phy = physics_kcl_loss(recon, B_tgt, N_BUS, mask=mask)

        total_loss = loss_mse + 0.1 * loss_phy
        total_loss.backward()
        optimizer_ft.step()
        loss_history_tgt.append(total_loss.item())

        if epoch % 10 == 0:
            print(f"FT Epoch {epoch:03d} | Loss: {total_loss.item():.5f}")

    # ---------------------------------------------------------
    # 阶段三：FDIA 数据恢复测试 (Inference)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> 阶段三：FDIA 数据恢复 (Data Recovery)")
    print("=" * 50)

    model_ft.eval()

    # 1) 取一个真实样本
    true_sample = x_tgt[0:1].clone()  # [1, N, 4]

    # 2) 假设第二章定位结果：节点5(Bus) 和 节点18(Line) 被攻击
    attack_indices = [5, 18]

    # 3) 构建输入：把“受攻击节点的全维特征”当作缺失
    #    论文叙事更像“删除/置空”，所以这里直接置0（或任意值都行，因为会被 token 替换）
    input_sample = true_sample.clone()
    input_sample[0, attack_indices, :] = 0.0  # <-- 等价于“删除后占位”

    # 4) 构建定向 mask：节点级mask（整节点全维）
    target_mask = torch.zeros_like(input_sample)  # [1,N,4]
    target_mask[0, attack_indices, :] = 1.0
    target_mask = target_mask[:, :, 0:1]  # [1,N,1] 节点级mask

    # 5) 恢复
    with torch.no_grad():
        recovered = model_ft(input_sample, edge_index_tgt, target_mask)  # [1,N,4]

    # 6) 全维结果打印 & 误差
    feat_names_bus = ["P", "Q", "V", "theta"]
    feat_names_line = ["P", "Q", "I_re", "I_im"]

    print(f"攻击节点: {attack_indices}")
    for idx in attack_indices:
        is_bus = (idx < N_BUS)
        names = feat_names_bus if is_bus else feat_names_line
        node_type = "Bus" if is_bus else "Line"

        gt = true_sample[0, idx, :].detach().cpu().numpy()
        inp = input_sample[0, idx, :].detach().cpu().numpy()
        rec = recovered[0, idx, :].detach().cpu().numpy()

        mse_all = float(np.mean((rec - gt) ** 2))

        print("\n" + "-" * 60)
        print(f"[{node_type} Node {idx}]  (全维恢复)  MSE={mse_all:.6e}")
        for d in range(4):
            print(
                f"  {names[d]:>5s}:  GT={gt[d]:>10.4f} | IN={inp[d]:>10.4f} | REC={rec[d]:>10.4f} | SE={(rec[d] - gt[d]) ** 2:.3e}")

    # 7) 如果你还想给一个“整体攻击区域的平均MSE”
    attack_tensor_gt = true_sample[0, attack_indices, :]
    attack_tensor_rec = recovered[0, attack_indices, :]
    mse_attack_avg = F.mse_loss(attack_tensor_rec, attack_tensor_gt).item()
    print("\n" + "=" * 60)
    print(f"[Summary] 攻击节点集合(含Bus+Line)的全维平均 MSE = {mse_attack_avg:.6e}")
    print("=" * 60)


    # 可视化 Loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history_src)
    plt.title("Source Pre-training Loss")
    plt.xlabel("Epoch")
    plt.subplot(1, 2, 2)
    plt.plot(loss_history_tgt, color='orange')
    plt.title("Target Fine-tuning Loss (N-1)")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()