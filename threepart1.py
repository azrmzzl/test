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

        if case == 'N-1':
            # 假设第 0 条支路断开 (Line Node 索引 = num_buses)
            fault_line_node = self.num_buses + 0
            # 物理动作：切断连接
            current_adj[fault_line_node, :] = 0
            current_adj[:, fault_line_node] = 0
            current_adj[fault_line_node, fault_line_node] = 1
            print(f">>> [场景: {case}] 支路节点 {fault_line_node} 已断开物理连接。")

        edge_index, _ = dense_to_sparse(current_adj)

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

        # [核心物理约束] 模拟 Bus P/Q: 根据 KCL 聚合 Line P/Q
        # Bus P = Sum(Line P), Bus Q = Sum(Line Q)
        for l_idx_rel in range(self.num_lines):
            l_node_idx = self.num_buses + l_idx_rel

            # 找到该 Line 连接的两个 Bus (从 self.line_connections 获取)
            # 注意：这里需要稍微修改一下 __init__ 把 line_connections 存下来
            # 为了代码独立性，这里重新遍历 adj 找连接 (低效但通用)
            connected_buses = (self.adj[:self.num_buses, l_node_idx] == 1).nonzero(as_tuple=True)[0]

            if len(connected_buses) == 2:
                b1, b2 = connected_buses[0], connected_buses[1]

                # 获取该 Line 的 P 和 Q
                val_p = x[:, l_node_idx, 0]
                val_q = x[:, l_node_idx, 1]

                # 如果是 N-1 且该线路断了，则流量归零
                if case == 'N-1' and l_idx_rel == 0:
                    val_p = torch.zeros_like(val_p)
                    val_q = torch.zeros_like(val_q)
                    x[:, l_node_idx, :] = 0  # 故障线路全维度归零

                # 简单模拟流向：流出 b1, 流入 b2
                # Bus P
                x[:, b1, 0] -= val_p
                x[:, b2, 0] += val_p
                # Bus Q
                x[:, b1, 1] -= val_q
                x[:, b2, 1] += val_q

        # 加上微量测量噪声
        x += 0.001 * torch.randn_like(x)

        return x, edge_index, current_adj


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
        batch_size, num_nodes, _ = x.shape
        outputs = []

        for i in range(batch_size):
            x_i = x[i].clone()

            # ------------------------------------------------
            # 1. Encoder Masking (输入端掩码)
            # ------------------------------------------------
            if mask is not None:
                m = mask[i]  # [N, 1]
                # 将被攻击/缺失的节点替换为 enc_token
                x_i = x_i * (1 - m) + self.enc_mask_token.squeeze(0) * m

            # ------------------------------------------------
            # 2. Encoding (GAT 聚合)
            # ------------------------------------------------
            h = F.elu(self.enc1(x_i, edge_index))
            h = F.elu(self.enc2(h, edge_index))  # 得到潜在表征 Latent H

            # ------------------------------------------------
            # 3. Re-masking (核心修正：再次掩码)
            # ------------------------------------------------
            # 在进入解码器之前，再次把受损节点的 Latent Feature 抹掉
            # 强迫解码器 GAT 必须去邻居节点抓取信息，而不能依赖 Encoder 传过来的残留信息
            if mask is not None:
                m = mask[i]
                h = h * (1 - m) + self.dec_mask_token.squeeze(0) * m

            # ------------------------------------------------
            # 4. Decoding (GAT 解码)
            # ------------------------------------------------
            # 利用 GAT 聚合邻居信息来填补 [DECODER_TOKEN]
            h_rec = F.elu(self.decoder_gat(h, edge_index))

            # 映射回物理数值
            out = self.decoder_head(h_rec)

            outputs.append(out.unsqueeze(0))

        return torch.cat(outputs, dim=0)


# ==========================================
# 3. 物理损失函数 (KCL Loss) - 核心干货 [修正版]
# ==========================================
def physics_kcl_loss(recon_x, adj, num_buses):
    """
    升级版物理损失：同时约束有功(P)和无功(Q)的平衡。
    对应 get_data 中生成的全维度物理特征。
    """
    batch_size = recon_x.shape[0]
    loss = 0

    # 1. 提取连接关系 [Bus, Line]
    # adj 的前 num_buses 行是 Bus，后 num_lines 列是 Line
    # bus_line_adj[i, j] = 1 表示 Bus i 与 Line j 相连
    bus_line_adj = adj[:num_buses, num_buses:]

    for i in range(batch_size):
        x = recon_x[i]

        # --- 1. 有功功率 P 的平衡 (修正索引) ---
        # 根据 get_data 定义: P 都在索引 0
        bus_p = x[:num_buses, 0]
        line_p = x[num_buses:, 0]

        # 计算 Bus 聚合的有功 (Sum of connected Line P)
        # 注意：这里简化了流向问题，强迫模型学习注入功率与线路功率幅值的强相关性
        agg_p = torch.matmul(bus_line_adj, line_p)

        # P 的残差 (MSE)
        loss_p = torch.mean((bus_p - agg_p) ** 2)

        # --- 2. 无功功率 Q 的平衡 (修正索引) ---
        # 根据 get_data 定义: Q 都在索引 1
        bus_q = x[:num_buses, 1]
        line_q = x[num_buses:, 1]

        # 计算 Bus 聚合的无功
        agg_q = torch.matmul(bus_line_adj, line_q)

        # Q 的残差 (MSE)
        loss_q = torch.mean((bus_q - agg_q) ** 2)

        # 总物理损失 = P损失 + Q损失
        loss += (loss_p + loss_q)

    return loss / batch_size


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
    x_src, edge_index_src, adj_src = simulator.get_data(num_samples=2000, case='normal')
    x_src, edge_index_src, adj_src = x_src.to(device), edge_index_src.to(device), adj_src.to(device)

    # 2. 初始化模型
    model = PI_GraphMAE(in_channels=4, hidden_channels=64, out_channels=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 3. 训练循环
    model.train()
    loss_history_src = []

    for epoch in range(1000):
        optimizer.zero_grad()

        # 随机 Mask (自监督训练，遮挡 30% 节点)
        mask = torch.rand(x_src.shape[0], x_src.shape[1], 1).to(device) > 0.7
        mask = mask.float()

        # 前向传播
        recon = model(x_src, edge_index_src, mask)

        # 计算 Loss
        # a. 重构 Loss (只看被 Mask 的部分，这是 MAE 的精髓)
        loss_mse = F.mse_loss(recon * mask, x_src * mask)
        # b. 物理 Loss (KCL 约束)
        loss_phy = physics_kcl_loss(recon, adj_src, N_BUS)

        total_loss = loss_mse + 0.1 * loss_phy

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
    x_tgt, edge_index_tgt, adj_tgt = simulator.get_data(num_samples=500, case='N-1')
    x_tgt, edge_index_tgt, adj_tgt = x_tgt.to(device), edge_index_tgt.to(device), adj_tgt.to(device)

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
        # 依然随机 Mask，让模型适应新拓扑下的填空
        mask = torch.rand(x_tgt.shape[0], x_tgt.shape[1], 1).to(device) > 0.7
        mask = mask.float()

        # !!! 关键: 输入的是 edge_index_tgt (断线后的拓扑) !!!
        recon = model_ft(x_tgt, edge_index_tgt, mask)

        loss_mse = F.mse_loss(recon * mask, x_tgt * mask)
        # !!! 关键: 物理约束基于 adj_tgt (断线后的KCL) !!!
        loss_phy = physics_kcl_loss(recon, adj_tgt, N_BUS)

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

    # 1. 模拟攻击场景
    # 取一个真实样本
    true_sample = x_tgt[0:1].clone()
    # 假设 Chapter 2 定位出：节点 5 (Bus) 和 节点 18 (Line) 被攻击
    attack_indices = [5, 18]

    # 2. 构建输入
    input_sample = true_sample.clone()
    # 制造攻击数据 (比如注入很大的值)
    input_sample[0, attack_indices, :] = 999.9

    # 3. 构建定向 Mask (Targeted Masking)
    # 1 表示被攻击/需要恢复，0 表示正常
    target_mask = torch.zeros_like(input_sample)
    target_mask[0, attack_indices, :] = 1
    target_mask = target_mask[:, :, 0:1]  # 形状 [1, N, 1]

    print(f"攻击节点: {attack_indices}")
    print(f"攻击前真实值 (Bus 5 P): {true_sample[0, 5, 1].item():.4f}")
    print(f"注入攻击值 (Bus 5 P): {input_sample[0, 5, 1].item():.4f}")

    # 4. 模型恢复
    with torch.no_grad():
        # 输入：被攻击数据 + 掩码 + 目标域拓扑
        # 模型会自动用 mask_token 替换掉 999.9，然后推理
        recovered = model_ft(input_sample, edge_index_tgt, target_mask)

    # 5. 结果验证
    rec_val = recovered[0, 5, 1].item()
    print(f"模型恢复值 (Bus 5 P): {rec_val:.4f}")
    print(f"恢复误差 (MSE): {(rec_val - true_sample[0, 5, 1].item()) ** 2:.6f}")

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