import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
import os
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

# 设置随机种子
torch.manual_seed(2025)
np.random.seed(2025)


# ==========================================
# 1. IEEE 57 数据加载器 (严格适配 4维特征)
# ==========================================
class IEEE57GraphDataset:
    def __init__(self, data_dir='dataset_split'):
        print(f">>> 正在加载 IEEE 57 数据集 (4维特征版): {data_dir} ...")

        # 1. 加载 MATLAB 数据
        self.samples = sio.loadmat(os.path.join(data_dir, 'Samples_正常运行初始10天精选线路.mat'))['Samples']  # (N, 427)
        self.branch_data = sio.loadmat(os.path.join(data_dir, 'Topology_Branch.mat'))['branch_data']  # (80, 2)

        # [核心修复] 强制转换为 int64，避免 uint8 导致 PyTorch 索引报错
        pmu_raw = sio.loadmat(os.path.join(data_dir, 'PMU_Position.mat'))['pmu_position'][0]
        self.pmu_pos = pmu_raw.astype(np.int64) - 1  # 变为 int64 并转为 0-based 索引

        # 索引修正 (MATLAB -> Python)
        self.branch_data = self.branch_data.astype(np.int64) - 1

        self.num_buses = 57
        self.num_lines = self.branch_data.shape[0]  # 80
        self.num_nodes = self.num_buses + self.num_lines  # 137

        # 2. 构建图结构
        self.edge_index, self.B_inc = self._build_graph_structure()

        # 3. 处理特征 (核心修改：映射到 4维)
        self.x, self.data_mean, self.data_std = self._process_features()

        print(f"数据加载完成: {self.x.shape} (样本数, 节点数=137, 特征数=4)")

    def _build_graph_structure(self):
        """构建母线-支路联合图"""
        adj = torch.eye(self.num_nodes)
        B_inc = torch.zeros(self.num_buses, self.num_lines)

        for l_idx in range(self.num_lines):
            b_from = int(self.branch_data[l_idx, 0])
            b_to = int(self.branch_data[l_idx, 1])
            l_node = self.num_buses + l_idx

            # Bus <-> Branch 连接
            adj[b_from, l_node] = 1;
            adj[l_node, b_from] = 1
            adj[b_to, l_node] = 1;
            adj[l_node, b_to] = 1

            # Incidence Matrix (母线为行，支路为列)
            B_inc[b_from, l_idx] = 1.0
            B_inc[b_to, l_idx] = -1.0

        edge_index, _ = dense_to_sparse(adj)
        return edge_index, B_inc

    def _process_features(self):
        """
        将 427维向量映射到 (137, 4) 矩阵
        """
        num_samples = self.samples.shape[0]
        x_graph = torch.zeros(num_samples, self.num_nodes, 4)

        # === 1. 解析 MATLAB Samples 列索引 ===
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
        n_if = np.sum(i_from_mask)
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

        # === 2. 填充特征 ===
        # A. 母线节点 (0~56)
        x_graph[:, :57, 0] = samples_t[:, idx_V_scada]
        x_graph[:, :57, 1] = samples_t[:, idx_P_inj]
        x_graph[:, :57, 2] = samples_t[:, idx_Q_inj]

        # [修复点] 使用 numpy.int64 数组进行索引，PyTorch 就不会把它当成 mask 了
        x_graph[:, self.pmu_pos, 3] = samples_t[:, idx_Del_pmu]

        # B. 支路节点 (57~136)
        x_graph[:, 57:, 0] = samples_t[:, idx_P_flow]
        x_graph[:, 57:, 1] = samples_t[:, idx_Q_flow]

        branch_indices_from = np.where(i_from_mask)[0]
        branch_indices_to = np.where(i_to_mask)[0]

        x_graph[:, 57 + branch_indices_from, 2] = samples_t[:, idx_Ire_from]
        x_graph[:, 57 + branch_indices_from, 3] = samples_t[:, idx_Iim_from]

        x_graph[:, 57 + branch_indices_to, 2] = samples_t[:, idx_Ire_to]
        x_graph[:, 57 + branch_indices_to, 3] = samples_t[:, idx_Iim_to]

        # === 3. 归一化 ===
        mean = x_graph.mean(dim=(0, 1), keepdim=True)
        std = x_graph.std(dim=(0, 1), keepdim=True) + 1e-6
        x_norm = (x_graph - mean) / std
        valid_mask = (x_graph != 0).float()
        x_norm = x_norm * valid_mask

        return x_norm, mean, std


# ==========================================
# 2. PI-GraphMAE 模型 (输入4维)
# ==========================================
class PI_GraphMAE(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=4, num_heads=4):
        super(PI_GraphMAE, self).__init__()

        # [Mask] Token 维度改为 4
        self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        # [DMask] Token
        self.dec_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_channels))

        # Encoder
        self.enc1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.enc2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Decoder (对称结构)
        self.dec1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=True)
        self.dec_head = nn.Linear(hidden_channels * num_heads, out_channels)

    def forward(self, x, edge_index, mask=None):
        B, N, Fdim = x.shape
        x_in = x.clone()

        # 1. 输入掩码 (Input Masking)
        if mask is not None:
            token = self.enc_mask_token.expand(B, N, Fdim)
            x_in = x_in * (1 - mask) + token * mask

        # 2. 编码器 (GAT Encoder)
        x_flat = x_in.reshape(B * N, Fdim)
        E = edge_index.shape[1]
        offsets = (torch.arange(B, device=x.device) * N).view(B, 1, 1)
        edge_batch = (edge_index.view(1, 2, E) + offsets).reshape(2, B * E)

        h = self.enc1(x_flat, edge_batch)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.enc2(h, edge_batch)
        h = self.bn2(h)
        h = F.elu(h)  # (B*N, Hidden)
        h = h.view(B, N, -1)

        # 3. 潜在重掩码 (Latent Re-masking) [论文图1核心步骤]
        if mask is not None:
            token_dec = self.dec_mask_token.expand(B, N, -1)
            h = h * (1 - mask) + token_dec * mask

        # 4. 解码器 (GAT Decoder)
        h_flat = h.reshape(B * N, -1)
        h_rec = self.dec1(h_flat, edge_batch)
        h_rec = F.elu(h_rec)
        out = self.dec_head(h_rec)  # (B*N, 4)

        return out.view(B, N, -1)


# ==========================================
# 3. 物理 Loss (适配 4维混合特征)
# ==========================================
def physics_loss_ieee57(recon_x, B_inc):
    """
    recon_x: (Batch, 137, 4)
    根据定义：
      Bus  (0~56) : [V, P, Q, Theta]
      Line (57~end): [P, Q, Ire, Iim]

    KCL 约束: Bus P/Q = sum(Line P/Q)
    """
    num_buses = 57

    # === 提取 Bus 的 P, Q ===
    # Bus特征中，P在通道1，Q在通道2
    bus_p = recon_x[:, :num_buses, 1]
    bus_q = recon_x[:, :num_buses, 2]

    # === 提取 Branch 的 P, Q ===
    # Line特征中，P在通道0，Q在通道1
    branch_p = recon_x[:, num_buses:, 0]
    branch_q = recon_x[:, num_buses:, 1]

    # === 计算 KCL 聚合值 ===
    # matmul: (Batch, N_line) x (N_line, N_bus) -> (Batch, N_bus)
    agg_p = torch.matmul(branch_p, B_inc.T)
    agg_q = torch.matmul(branch_q, B_inc.T)

    # === 计算 Loss ===
    loss_p = F.mse_loss(bus_p, agg_p)
    loss_q = F.mse_loss(bus_q, agg_q)

    return loss_p + loss_q


# ==========================================
# 4. 训练主程序
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载数据
    dataset = IEEE57GraphDataset(data_dir='dataset_split')
    x_all = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    B_inc = dataset.B_inc.to(device)

    num_train = int(x_all.shape[0] * 0.8)
    train_loader = torch.utils.data.DataLoader(x_all[:num_train], batch_size=64, shuffle=True)

    # 2. 模型 (特征数=4)
    model = PI_GraphMAE(in_channels=4, hidden_channels=64, out_channels=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n>>> 开始训练 PI-GraphMAE (Feature=4)...")
    model.train()

    for epoch in range(200):
        total_loss = 0
        for batch_x in train_loader:
            optimizer.zero_grad()
            # [修改点] 将 F 改名为 Fdim，避免覆盖 torch.nn.functional
            B, N, Fdim = batch_x.shape

            # 随机掩码
            mask = (torch.rand(B, N, 1, device=device) < 0.8).float()

            # 前向
            recon = model(batch_x, edge_index, mask)

            # MSE Loss (仅计算掩码位置)
            loss_mse = F.mse_loss(recon * mask, batch_x * mask)

            # KCL Physics Loss
            loss_phy = physics_loss_ieee57(recon, B_inc)

            loss = loss_mse + 0.1 * loss_phy
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} | Avg Loss: {total_loss / len(train_loader):.6f}")

    print("训练完成！")
    torch.save(model.state_dict(), '../pi_graphmae_ieee57_4dim.pth')


if __name__ == "__main__":
    main()