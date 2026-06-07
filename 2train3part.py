import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
import os
import time
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

# 设置随机种子
torch.manual_seed(2025)
np.random.seed(2025)


# ==========================================
# 1. IEEE 57 数据加载器
# ==========================================
class IEEE57GraphDataset:
    def __init__(self, data_dir='dataset_split'):
        print(f">>> 正在加载 IEEE 57 数据集: {data_dir} ...")
        self.samples = sio.loadmat(os.path.join(data_dir, 'Samples_x_y_normal_operation_all.mat'))['Samples']
        self.branch_data = sio.loadmat(os.path.join(data_dir, 'Topology_Branch.mat'))['branch_data']
        pmu_raw = sio.loadmat(os.path.join(data_dir, 'PMU_Position.mat'))['pmu_position'][0]
        self.pmu_pos = pmu_raw.astype(np.int64) - 1
        self.branch_data = self.branch_data.astype(np.int64) - 1
        self.num_buses = 57
        self.num_lines = self.branch_data.shape[0]
        self.num_nodes = self.num_buses + self.num_lines
        self.edge_index, self.B_inc = self._build_graph_structure()
        self.x, self.data_mean, self.data_std = self._process_features()
        print(f"数据加载完成: {self.x.shape}")

    def _build_graph_structure(self):
        adj = torch.eye(self.num_nodes)
        B_inc = torch.zeros(self.num_buses, self.num_lines)
        for l_idx in range(self.num_lines):
            b_from = int(self.branch_data[l_idx, 0])
            b_to = int(self.branch_data[l_idx, 1])
            l_node = self.num_buses + l_idx
            adj[b_from, l_node] = 1;
            adj[l_node, b_from] = 1
            adj[b_to, l_node] = 1;
            adj[l_node, b_to] = 1
            B_inc[b_from, l_idx] = 1.0;
            B_inc[b_to, l_idx] = -1.0
        edge_index, _ = dense_to_sparse(adj)
        return edge_index, B_inc

    def _process_features(self):
        num_samples = self.samples.shape[0]
        x_graph = torch.zeros(num_samples, self.num_nodes, 4)
        idx = 0
        idx_V_scada = slice(0, 57);
        idx += 57
        idx_P_inj = slice(idx, idx + 57);
        idx += 57
        idx_Q_inj = slice(idx, idx + 57);
        idx += 57
        idx_P_flow = slice(idx, idx + 80);
        idx += 80
        idx_Q_flow = slice(idx, idx + 80);
        idx += 80
        n_pmu = len(self.pmu_pos)
        idx_Del_pmu = slice(idx, idx + n_pmu);
        idx += n_pmu
        idx_V_pmu = slice(idx, idx + n_pmu);
        idx += n_pmu
        i_from_mask = np.isin(self.branch_data[:, 0], self.pmu_pos)
        i_to_mask = np.isin(self.branch_data[:, 1], self.pmu_pos)
        n_if = np.sum(i_from_mask);
        n_it = np.sum(i_to_mask)
        idx_Ire_from = slice(idx, idx + n_if);
        idx += n_if
        idx_Iim_from = slice(idx, idx + n_if);
        idx += n_if
        idx_Ire_to = slice(idx, idx + n_it);
        idx += n_it
        idx_Iim_to = slice(idx, idx + n_it);
        idx += n_it

        samples_t = torch.tensor(self.samples, dtype=torch.float32)
        x_graph[:, :57, 0] = samples_t[:, idx_V_scada]
        x_graph[:, :57, 1] = samples_t[:, idx_P_inj]
        x_graph[:, :57, 2] = samples_t[:, idx_Q_inj]
        x_graph[:, self.pmu_pos, 3] = samples_t[:, idx_Del_pmu]
        x_graph[:, 57:, 0] = samples_t[:, idx_P_flow]
        x_graph[:, 57:, 1] = samples_t[:, idx_Q_flow]
        branch_indices_from = np.where(i_from_mask)[0]
        branch_indices_to = np.where(i_to_mask)[0]
        x_graph[:, 57 + branch_indices_from, 2] = samples_t[:, idx_Ire_from]
        x_graph[:, 57 + branch_indices_from, 3] = samples_t[:, idx_Iim_from]
        x_graph[:, 57 + branch_indices_to, 2] = samples_t[:, idx_Ire_to]
        x_graph[:, 57 + branch_indices_to, 3] = samples_t[:, idx_Iim_to]

        mean = x_graph.mean(dim=(0, 1), keepdim=True)
        std = x_graph.std(dim=(0, 1), keepdim=True) + 1e-6
        x_norm = (x_graph - mean) / std
        valid_mask = (x_graph != 0).float()
        x_norm = x_norm * valid_mask
        return x_norm, mean, std


# ==========================================
# 2. 模型定义 (保持不变)
# ==========================================
class PI_GraphMAE(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=4, num_heads=4):
        super(PI_GraphMAE, self).__init__()
        self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_channels))
        self.enc1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.enc2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dec1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=True)
        self.dec_head = nn.Linear(hidden_channels * num_heads, out_channels)

    def forward(self, x, edge_index, mask=None):
        B, N, Fdim = x.shape
        x_in = x.clone()
        if mask is not None:
            token = self.enc_mask_token.expand(B, N, Fdim)
            x_in = x_in * (1 - mask) + token * mask
        x_flat = x_in.reshape(B * N, Fdim)
        E = edge_index.shape[1]
        offsets = (torch.arange(B, device=x.device) * N).view(B, 1, 1)
        edge_batch = (edge_index.view(1, 2, E) + offsets).reshape(2, B * E)
        h = self.enc1(x_flat, edge_batch)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.enc2(h, edge_batch)
        h = self.bn2(h)
        h = F.elu(h)
        h = h.view(B, N, -1)
        if mask is not None:
            token_dec = self.dec_mask_token.expand(B, N, -1)
            h = h * (1 - mask) + token_dec * mask
        h_flat = h.reshape(B * N, -1)
        h_rec = self.dec1(h_flat, edge_batch)
        h_rec = F.elu(h_rec)
        out = self.dec_head(h_rec)
        return out.view(B, N, -1)


# ==========================================
# 3. [核心修复] 物理 Loss (先反归一化再计算)
# ==========================================
def physics_loss_ieee57_corrected(recon_norm, B_inc, mean, std):
    """
    recon_norm: 归一化后的输出
    mean, std: 全局均值和方差，用于反归一化
    """
    # 1. 反归一化 (恢复到真实的物理数值)
    recon_phys = recon_norm * std + mean

    num_buses = 57
    bus_p = recon_phys[:, :num_buses, 1]
    bus_q = recon_phys[:, :num_buses, 2]
    branch_p = recon_phys[:, num_buses:, 0]
    branch_q = recon_phys[:, num_buses:, 1]

    # 2. 计算 KCL (在真实物理空间计算)
    agg_p = torch.matmul(branch_p, B_inc.T)
    agg_q = torch.matmul(branch_q, B_inc.T)

    loss_p = F.mse_loss(bus_p, agg_p)
    loss_q = F.mse_loss(bus_q, agg_q)

    return loss_p + loss_q


# ==========================================
# 4. 训练主程序
# ==========================================
def main_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    dataset = IEEE57GraphDataset(data_dir='dataset_split')
    x_all = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    B_inc = dataset.B_inc.to(device)

    # 获取 mean 和 std 用于物理 Loss
    data_mean = dataset.data_mean.to(device)
    data_std = dataset.data_std.to(device)

    # ==========================================
    # 🔥 正式训练设置
    # ==========================================
    FIXED_MASK_MODE = True  # 正式训练请设为 False
    IS_SHUFFLE = False # 正式训练请设为 True
    BATCH_SIZE = 128  # 增大 Batch Size 可以让梯度更稳，训练更快
    EPOCHS = 1000
    # ==========================================

    num_train = int(x_all.shape[0] * 1)
    train_loader = torch.utils.data.DataLoader(x_all[:num_train], batch_size=BATCH_SIZE, shuffle=IS_SHUFFLE)

    # [优化 1] 增大模型容量：Hidden 64->256, Heads 4->8
    print(">>> 初始化增强版模型 (Hidden=256, Heads=8)...")
    model = PI_GraphMAE(in_channels=4, hidden_channels=256, out_channels=4, num_heads=8).to(device)

    # [优化 2] 调整优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=0.002)  # 初始 LR 加倍
    # CosineAnnealingLR：让 LR 像余弦曲线一样下降，最后收敛得更紧
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"\n>>> 开始训练...")
    model.train()

    fixed_mask = None

    for epoch in range(EPOCHS):
        total_loss = 0
        total_mse = 0
        total_phy = 0

        for batch_x in train_loader:
            optimizer.zero_grad()
            B, N, Fdim = batch_x.shape

            # 掩码生成
            if FIXED_MASK_MODE:
                if fixed_mask is None or fixed_mask.shape[0] != B:
                    fixed_mask = (torch.rand(B, N, 1, device=device) < 0.2).float()
                mask = fixed_mask
            else:
                mask = (torch.rand(B, N, 1, device=device) < 0.2).float()

            recon = model(batch_x, edge_index, mask)

            # Loss 计算
            loss_mse = F.mse_loss(recon * mask, batch_x * mask)
            loss_phy = physics_loss_ieee57_corrected(recon, B_inc, data_mean, data_std)

            # 组合
            loss = loss_mse + 0.01 * loss_phy

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_phy += loss_phy.item()

        # 每个 Epoch 结束后更新 LR
        scheduler.step()

        if (epoch + 1) % 5 == 0:  # 每20轮打印一次，减少刷屏
            avg_loss = total_loss / len(train_loader)
            avg_mse = total_mse / len(train_loader)
            avg_phy = total_phy / len(train_loader)
            lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1:04d} | Loss: {avg_loss:.6f} | MSE: {avg_mse:.6f} | Phy: {avg_phy:.6f} | LR: {lr:.6f}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = f'pi_graphmae_ieee57_large_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存为 {model_path}")


if __name__ == "__main__":
    main_train()