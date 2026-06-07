import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from torch_geometric.utils import dense_to_sparse

# 复用之前的 Dataset 和 Model 类 (为了方便，这里直接粘贴定义，或者你可以 import)
# -----------------------------------------------------------------------------
# 请确保这里定义的类和 data3part.py 里完全一致
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class IEEE57GraphDataset:
    def __init__(self, data_dir='dataset_split'):
        # 简化版加载，只为获取 mean/std 和图结构
        self.samples = sio.loadmat(os.path.join(data_dir, 'Samples_正常运行初始10天精选线路.mat'))['Samples']
        self.branch_data = sio.loadmat(os.path.join(data_dir, 'Topology_Branch.mat'))['branch_data']
        self.pmu_pos = sio.loadmat(os.path.join(data_dir, 'PMU_Position.mat'))['pmu_position'][0].astype(np.int64) - 1
        self.branch_data = self.branch_data.astype(np.int64) - 1

        self.num_buses = 57
        self.num_lines = self.branch_data.shape[0]
        self.num_nodes = self.num_buses + self.num_lines

        self.edge_index, self.B_inc = self._build_graph_structure()
        self.x, self.mean, self.std = self._process_features()

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
        # 重新执行一遍特征构建以获取正确的 mean/std 用于反归一化
        # (代码逻辑同 data3part.py，此处省略部分重复细节，保证逻辑一致即可)
        # 为了代码简洁，这里假设你已经把 Dataset 封装好，可以直接调用
        # 如果没有封装，请把 data3part.py 里的 _process_features 复制过来

        # ... (此处应复制 _process_features 的完整逻辑) ...
        # 为节省篇幅，我直接使用 data3part.py 的逻辑复现核心部分:

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


# -----------------------------------------------------------------------------
# 主测试逻辑
# -----------------------------------------------------------------------------
def denormalize(x_norm, mean, std):
    """反归一化：把模型输出变回真实物理值"""
    return x_norm * std + mean


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    dataset = IEEE57GraphDataset(data_dir='dataset_split')
    x_all = dataset.x
    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    edge_index = dataset.edge_index.to(device)

    # 2. 加载模型
    model = PI_GraphMAE(in_channels=4, hidden_channels=64, out_channels=4).to(device)

    # [修复警告] 添加 weights_only=True
    model.load_state_dict(torch.load('../pi_graphmae_ieee57_4dim.pth', map_location=device, weights_only=True))
    model.eval()
    print(">>> 模型加载成功！")

    # 3. 挑选一个测试样本
    sample_idx = 100
    x_true = x_all[sample_idx:sample_idx + 1].to(device)  # (1, 137, 4)

    # 4. 设置攻击/掩码 (模拟 FDIA 或 数据丢失)
    target_node = 3  # 第4号母线

    # ================= [关键修改] =================
    # 必须保证 mask 最后一维是 1，即 (Batch, Nodes, 1)
    # 这样才能广播适配 64维的 hidden state
    mask = torch.zeros((x_true.shape[0], x_true.shape[1], 1)).to(device)
    mask[:, target_node, :] = 1.0
    # =============================================

    # 5. 模型推理
    with torch.no_grad():
        x_recon_norm = model(x_true, edge_index, mask)

    # 6. 反归一化
    x_true_phys = denormalize(x_true, mean, std)
    x_recon_phys = denormalize(x_recon_norm, mean, std)

    # 7. 打印对比结果
    print(f"\n>>> 节点 {target_node + 1} 重构结果对比:")
    feats = ['Voltage (pu)', 'P (pu)', 'Q (pu)', 'Theta (rad)']

    vec_true = x_true_phys[0, target_node, :].cpu().numpy()
    vec_recon = x_recon_phys[0, target_node, :].cpu().numpy()

    print(f"{'Feature':<15} | {'Real Value':<12} | {'Reconstructed':<12} | {'Error':<10}")
    print("-" * 55)
    for i in range(4):
        err = abs(vec_true[i] - vec_recon[i])
        print(f"{feats[i]:<15} | {vec_true[i]:.4f}       | {vec_recon[i]:.4f}       | {err:.4f}")

    # 8. 画图
    plt.figure(figsize=(12, 5))
    bus_indices = np.arange(57)
    p_true = x_true_phys[0, :57, 1].cpu().numpy()
    p_recon = x_recon_phys[0, :57, 1].cpu().numpy()

    plt.plot(bus_indices, p_true, 'b-', label='Ground Truth', alpha=0.7)
    plt.plot(bus_indices, p_recon, 'r--', label='Reconstructed', alpha=0.7)
    plt.title(f'Bus Active Power Injection (Sample {sample_idx})')
    plt.xlabel('Bus Index')
    plt.ylabel('P (p.u.)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()