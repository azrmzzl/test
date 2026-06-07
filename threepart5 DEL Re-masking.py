import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

# 设置随机种子
torch.manual_seed(2025)
np.random.seed(2025)


# ==========================================
# 1. 论文特定数据模拟器
# ==========================================
class ThesisGraphData:
    def __init__(self, num_buses=14, num_lines=20):
        self.num_buses = num_buses
        self.num_lines = num_lines
        self.num_nodes = num_buses + num_lines

        self.adj = torch.eye(self.num_nodes)
        self.line_connections = []

        for l in range(num_lines):
            line_node_idx = num_buses + l
            b1 = np.random.randint(0, num_buses)
            b2 = np.random.randint(0, num_buses)
            while b1 == b2: b2 = np.random.randint(0, num_buses)

            self.line_connections.append((line_node_idx, b1, b2))
            self.adj[b1, line_node_idx] = 1
            self.adj[line_node_idx, b1] = 1
            self.adj[b2, line_node_idx] = 1
            self.adj[line_node_idx, b2] = 1

    def get_data(self, num_samples, case='normal'):
        current_adj = self.adj.clone()
        x = torch.zeros(num_samples, self.num_nodes, 4)
        t = torch.linspace(0, 10, num_samples).view(-1, 1)

        line_p_base = torch.randn(1, self.num_lines)
        line_q_base = torch.randn(1, self.num_lines) * 0.2

        if case == 'N-1':
            fault_line_idx = 0
            fault_node = self.num_buses + fault_line_idx
            current_adj[fault_node, :] = 0
            current_adj[:, fault_node] = 0
            current_adj[fault_node, fault_node] = 1

            p_transfer = line_p_base[0, fault_line_idx]
            q_transfer = line_q_base[0, fault_line_idx]
            line_p_base[0, 1] += p_transfer * 0.6
            line_p_base[0, 2] += p_transfer * 0.4
            line_q_base[0, 1] += q_transfer * 0.6
            line_q_base[0, 2] += q_transfer * 0.4
            line_p_base[0, fault_line_idx] = 0
            line_q_base[0, fault_line_idx] = 0
            print(f">>> [场景: {case}] 支路 {fault_node} 断开，潮流已模拟转移。")

        edge_index, _ = dense_to_sparse(current_adj)

        B_inc = torch.zeros(self.num_buses, self.num_lines)
        for l_rel in range(self.num_lines):
            _, b1, b2 = self.line_connections[l_rel]
            B_inc[b1, l_rel] = -1.0
            B_inc[b2, l_rel] = +1.0

        x[:, self.num_buses:, 0] = line_p_base + 0.5 * torch.sin(t)
        x[:, self.num_buses:, 1] = line_q_base + 0.1 * torch.cos(t)
        x[:, self.num_buses:, 2] = x[:, self.num_buses:, 0] / 1.0
        x[:, self.num_buses:, 3] = x[:, self.num_buses:, 1] / 1.0

        x[:, :self.num_buses, 2] = 1.0 + 0.02 * torch.randn(num_samples, self.num_buses)
        x[:, :self.num_buses, 3] = 0.0 + 0.05 * torch.randn(num_samples, self.num_buses)

        for l_idx_rel in range(self.num_lines):
            l_node_idx = self.num_buses + l_idx_rel
            _, b_from, b_to = self.line_connections[l_idx_rel]
            val_p = x[:, l_node_idx, 0]
            val_q = x[:, l_node_idx, 1]
            x[:, b_from, 0] -= val_p
            x[:, b_to, 0] += val_p
            x[:, b_from, 1] -= val_q
            x[:, b_to, 1] += val_q

        x += 0.001 * torch.randn_like(x)
        return x, edge_index, current_adj, B_inc


# ==========================================
# 2. 核心模型: PI-GraphMAE (移除 Re-masking 以求稳)
# ==========================================
class PI_GraphMAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8):
        super(PI_GraphMAE, self).__init__()

        self.enc_mask_token = nn.Parameter(torch.randn(1, 1, in_channels))
        # [修改] 移除了 dec_mask_token，简化训练难度

        self.enc1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.enc2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False)
        self.decoder_gat = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=True)
        self.decoder_head = nn.Linear(hidden_channels * num_heads, out_channels)

    def forward(self, x, edge_index, mask=None):
        B, N, Fdim = x.shape
        device = x.device
        x_in = x.clone()

        # 1) Input Masking
        if mask is not None:
            x_in = x_in * (1 - mask) + self.enc_mask_token.to(device) * mask

        # 2) Graph Conv
        x_flat = x_in.reshape(B * N, Fdim)
        E = edge_index.shape[1]
        offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
        edge_rep = edge_index.view(1, 2, E).repeat(B, 1, 1) + offsets
        edge_big = edge_rep.reshape(2, B * E)

        h = F.elu(self.enc1(x_flat, edge_big))
        h = F.elu(self.enc2(h, edge_big))
        Hdim = h.shape[1]

        # [修改] 移除了 Re-masking，直接进入解码器
        # 这对于小图来说至关重要，保证梯度流通

        h_rec = F.elu(self.decoder_gat(h, edge_big))
        out = self.decoder_head(h_rec)
        out = out.view(B, N, -1)
        return out


# ==========================================
# 3. 物理损失函数 (标准化残差版)
# ==========================================
def physics_kcl_loss(recon_x, B_inc, num_buses, data_std, mask=None):
    """
    recon_x: 真实物理值
    data_std: 全局标准差，用于将物理残差拉回 Normal Scale
    """
    nb = num_buses
    bus = recon_x[:, :nb, :]
    line = recon_x[:, nb:, :]

    # 获取全局 Std
    std_p = data_std[0, 0, 0].item() + 1e-6
    std_q = data_std[0, 0, 1].item() + 1e-6

    agg_p = torch.matmul(line[:, :, 0], B_inc.T)
    agg_q = torch.matmul(line[:, :, 1], B_inc.T)

    # 计算标准化后的物理误差
    loss_p = ((bus[:, :, 0] - agg_p) / std_p).pow(2).mean()
    loss_q = ((bus[:, :, 1] - agg_q) / std_q).pow(2).mean()

    return loss_p + loss_q


# ==========================================
# 4. 主程序
# ==========================================
def main():
    torch.cuda.empty_cache()  # [新增] 清理显存
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    simulator = ThesisGraphData()

    # --- 阶段一 ---
    print("\n>>> 阶段一：源域预训练")
    x_src, edge_index_src, _, B_src = simulator.get_data(num_samples=2000, case='normal')
    x_src, edge_index_src, B_src = x_src.to(device), edge_index_src.to(device), B_src.to(device)

    # 计算全局统计量
    data_mean = x_src.mean(dim=(0, 1), keepdim=True)
    data_std = x_src.std(dim=(0, 1), keepdim=True)

    def normalize(data):
        return (data - data_mean) / (data_std + 1e-6)

    def denormalize(data):
        return data * (data_std + 1e-6) + data_mean

    x_src_norm = normalize(x_src)

    # 稍微增大 Hidden Dim 到 256
    model = PI_GraphMAE(in_channels=4, hidden_channels=128, out_channels=4, num_heads=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 降低 LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)

    model.train()
    for epoch in range(2000):
        optimizer.zero_grad()
        mask = torch.rand(x_src.shape[0], x_src.shape[1], 1).to(device) > 0.75  # 25% Mask
        mask = mask.float()

        recon_norm = model(x_src_norm, edge_index_src, mask)

        # MSE Loss (只计算 Mask 部分)
        loss_mse = ((recon_norm - x_src_norm) * mask).pow(2).sum() / (mask.sum() + 1e-6)

        # Physics Loss (Denormalized -> Calculated -> Standardized Error)
        recon_real = denormalize(recon_norm)
        loss_phy = physics_kcl_loss(recon_real, B_src, simulator.num_buses, data_std, mask)

        # 权重
        warm_epochs = 500
        w_phy = 0.1 * min(1.0, epoch / warm_epochs)
        total_loss = loss_mse + w_phy * loss_phy

        total_loss.backward()
        # 梯度裁剪 (防止爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step(total_loss)

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:03d} | Loss: {total_loss.item():.4f} (MSE: {loss_mse.item():.4f}, Phy: {loss_phy.item():.4f})")

    torch.save(model.state_dict(), '../pi_graphmae_pretrained.pth')

    # --- 阶段二 ---
    print("\n>>> 阶段二：目标域微调")
    x_tgt, edge_index_tgt, _, B_tgt = simulator.get_data(num_samples=500, case='N-1')
    x_tgt, edge_index_tgt, B_tgt = x_tgt.to(device), edge_index_tgt.to(device), B_tgt.to(device)
    x_tgt_norm = normalize(x_tgt)

    # 注意: 这里 num_heads 和 hidden_channels 必须和上面一致
    model_ft = PI_GraphMAE(in_channels=4, hidden_channels=128, out_channels=4, num_heads=8).to(device)
    model_ft.load_state_dict(torch.load('../pi_graphmae_pretrained.pth', weights_only=True))

    for name, param in model_ft.named_parameters():
        if 'enc' in name: param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.0005)

    for epoch in range(400):
        optimizer_ft.zero_grad()
        mask = torch.rand(x_tgt.shape[0], x_tgt.shape[1], 1).to(device) > 0.75
        mask = mask.float()

        recon_norm = model_ft(x_tgt_norm, edge_index_tgt, mask)
        loss_mse = ((recon_norm - x_tgt_norm) * mask).pow(2).sum() / (mask.sum() + 1e-6)

        recon_real = denormalize(recon_norm)
        loss_phy = physics_kcl_loss(recon_real, B_tgt, simulator.num_buses, data_std)

        total_loss = loss_mse + 0.1 * loss_phy
        total_loss.backward()
        optimizer_ft.step()

        if epoch % 50 == 0:
            print(f"FT Epoch {epoch:03d} | Loss: {total_loss.item():.4f}")

    # --- 阶段三 ---
    print("\n>>> 阶段三：恢复测试")
    model_ft.eval()

    true_sample_norm = x_tgt_norm[0:1].clone()
    input_sample_norm = true_sample_norm.clone()

    attack_indices = [5, 18]
    input_sample_norm[0, attack_indices, :] = 0.0  # Attack/Missing
    target_mask = torch.zeros_like(input_sample_norm)
    target_mask[0, attack_indices, :] = 1.0

    with torch.no_grad():
        rec_norm = model_ft(input_sample_norm, edge_index_tgt, target_mask)
        rec_real = denormalize(rec_norm)
        gt_real = denormalize(true_sample_norm)

    print(f"攻击节点: {attack_indices}")
    for idx in attack_indices:
        gt = gt_real[0, idx, 1].item()
        rec = rec_real[0, idx, 1].item()
        print(f"Node {idx} (Q) | GT: {gt:.4f} | Rec: {rec:.4f} | MSE: {(gt - rec) ** 2:.6f}")


if __name__ == "__main__":
    main()