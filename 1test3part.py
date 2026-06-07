import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from torch_geometric.utils import dense_to_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ==========================================
# 1. Dataset (必须与训练代码一致：无归一化)
# ==========================================
class IEEE57GraphDataset:
    def __init__(self, data_dir='dataset_split'):
        # 这里的 mat 文件名请确认与您本地一致
        self.samples = sio.loadmat(os.path.join(data_dir, 'Samples_x_y_normal_operation_all.mat'))['Samples']
        self.branch_data = sio.loadmat(os.path.join(data_dir, 'Topology_Branch.mat'))['branch_data']
        self.pmu_pos = sio.loadmat(os.path.join(data_dir, 'PMU_Position.mat'))['pmu_position'][0].astype(np.int64) - 1
        self.branch_data = self.branch_data.astype(np.int64) - 1

        self.num_buses = 57
        self.num_lines = self.branch_data.shape[0]
        self.num_nodes = self.num_buses + self.num_lines

        self.edge_index, self.B_inc = self._build_graph_structure()
        self.x = self._process_features()  # 无 mean/std 返回

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

        # 直接返回，无归一化
        return x_graph


# ==========================================
# 2. Model (必须与训练时参数一致)
# ==========================================
class PI_GraphMAE(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=256, out_channels=4, num_heads=8):
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
# 3. Test Logic
# ==========================================
def main_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    dataset = IEEE57GraphDataset(data_dir='dataset_split')
    x_all = dataset.x
    edge_index = dataset.edge_index.to(device)

    # 2. 加载模型 (注意：这里 hidden_channels=256, num_heads=8)
    model = PI_GraphMAE(in_channels=4, hidden_channels=256, out_channels=4, num_heads=8).to(device)

    # [请修改] 这里替换为您训练生成的 .pth 文件名
    model_filename = 'pi_graphmae_ieee57_no_norm_20260207-123456.pth'

    # 自动搜索最新的模型文件 (方便测试)
    import glob
    pth_files = glob.glob('pi_graphmae_ieee57_best_*.pth')
    if pth_files:
        model_filename = max(pth_files, key=os.path.getctime)  # 找最新的
        print(f">>> 自动找到最新模型: {model_filename}")

    if os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename, map_location=device, weights_only=True))
        print(">>> 模型加载成功！")
    else:
        print(f"❌ 找不到模型文件 {model_filename}，请检查路径。")
        return

    model.eval()

    # 3. 挑选测试样本 (Bus 12, Index 11)
    sample_idx = 100
    x_true = x_all[sample_idx:sample_idx + 1].to(device)  # (1, 137, 4)

    # 4. 设置掩码
    target_node = 11  # Bus 12
    mask = torch.zeros((x_true.shape[0], x_true.shape[1], 1)).to(device)
    mask[:, target_node, :] = 1.0

    # 5. 推理
    with torch.no_grad():
        x_recon = model(x_true, edge_index, mask)  # 直接输出物理值

    # 6. 打印结果 (无需 denormalize)
    print(f"\n>>> 节点 {target_node + 1} (Bus {target_node + 1}) 重构结果:")
    feats = ['Voltage (pu)', 'P (pu)', 'Q (pu)', 'Theta (rad)']

    vec_true = x_true[0, target_node, :].cpu().numpy()
    vec_recon = x_recon[0, target_node, :].cpu().numpy()

    print(f"{'Feature':<15} | {'Real Value':<12} | {'Reconstructed':<12} | {'Error':<10}")
    print("-" * 55)
    for i in range(4):
        err = abs(vec_true[i] - vec_recon[i])
        print(f"{feats[i]:<15} | {vec_true[i]:.4f}       | {vec_recon[i]:.4f}       | {err:.4f}")

    # 7. 画图 (全网有功 P 对比)
    plt.figure(figsize=(12, 5))
    bus_indices = np.arange(57)
    # 通道 1 是 P
    p_true = x_true[0, :57, 1].cpu().numpy()
    p_recon = x_recon[0, :57, 1].cpu().numpy()

    plt.plot(bus_indices, p_true, 'b-', label='Ground Truth', alpha=0.7)
    plt.plot(bus_indices, p_recon, 'r--', label='Reconstructed', alpha=0.7)

    # 标记被掩码的节点
    plt.scatter([target_node], [p_true[target_node]], color='orange', s=100, label='Masked Node', zorder=5)

    plt.title(f'Bus Active Power Injection (Sample {sample_idx}) - No Norm')
    plt.xlabel('Bus Index')
    plt.ylabel('P (p.u.)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main_test()