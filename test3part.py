import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from torch_geometric.utils import dense_to_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from visualization_utils import plot_single_sample_comparison, plot_time_series_comparison, set_random_seed
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# ==========================================
# 1. 定义类 (必须与训练代码完全一致)
# ==========================================
class IEEE57GraphDataset:
    def __init__(self, data_dir='dataset_split'):
        # 加载数据用于获取 mean/std
        self.samples = sio.loadmat(os.path.join(data_dir, 'Samples_x_y_normal_operation_all.mat'))['Samples']
        self.branch_data = sio.loadmat(os.path.join(data_dir, 'Topology_Branch.mat'))['branch_data']
        # [修复] 索引类型转换
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
            adj[b_from, l_node] = 1
            adj[l_node, b_from] = 1
            adj[b_to, l_node] = 1
            adj[l_node, b_to] = 1
            B_inc[b_from, l_idx] = 1.0
            B_inc[b_to, l_idx] = -1.0
        edge_index, _ = dense_to_sparse(adj)
        return edge_index, B_inc

    def _process_features(self):
        # 简化版特征处理，仅为了获取 mean/std
        num_samples = self.samples.shape[0]
        x_graph = torch.zeros(num_samples, self.num_nodes, 4)

        idx = 0
        idx_V_scada = slice(0, 57)
        idx += 57
        idx_P_inj = slice(idx, idx + 57)
        idx += 57
        idx_Q_inj = slice(idx, idx + 57)
        idx += 57
        idx_P_flow = slice(idx, idx + 80)
        idx += 80
        idx_Q_flow = slice(idx, idx + 80)
        idx += 80
        n_pmu = len(self.pmu_pos)
        idx_Del_pmu = slice(idx, idx + n_pmu)
        idx += n_pmu
        idx_V_pmu = slice(idx, idx + n_pmu)
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


def denormalize(x_norm, mean, std):
    return x_norm * std + mean


# ==========================================
# 2. 测试主程序
# ==========================================
def main_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    dataset = IEEE57GraphDataset(data_dir='dataset_split')
    x_all = dataset.x
    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    edge_index = dataset.edge_index.to(device)

    # 2. 加载模型 (必须与训练时结构一致)
    model = PI_GraphMAE(in_channels=4, hidden_channels=64, out_channels=4).to(device)

    # [修复] 在当前目录查找模型文件
    import glob
    model_files = glob.glob('pi_graphmae_ieee57_best_*.pth')
    if not model_files:
        print("❌ 错误：找不到模型文件 'pi_graphmae_ieee57_best_*.pth'。请先运行训练代码！")
        return
    
    # 选择最新的模型文件（按文件名排序）
    latest_model = sorted(model_files)[-1]
    print(f">>> 加载模型：{latest_model}")
    model.load_state_dict(torch.load(latest_model, map_location=device, weights_only=True))

    model.eval()

    # [关键] 固定随机种子，确保与其他方法使用相同的对比基准
    set_random_seed(2025)
        
    # 3. 挑选测试样本 (Bus 12 负载较大，数据不为 0，适合观察)
    sample_idx = 100
    x_true = x_all[sample_idx:sample_idx + 1].to(device)  # (1, 137, 4)

    # 4. 设置攻击/掩码
    target_node = 11  # Bus 12 (MATLAB索引12 -> Python索引11)

    # [修复] 强制 mask 最后一维为 1，解决 64 vs 4 报错
    mask = torch.zeros((x_true.shape[0], x_true.shape[1], 1)).to(device)
    mask[:, target_node, :] = 1.0

    # 5. 推理
    with torch.no_grad():
        x_recon_norm = model(x_true, edge_index, mask)

    # 6. 反归一化
    x_true_phys = denormalize(x_true, mean, std)
    x_recon_phys = denormalize(x_recon_norm, mean, std)

    # 7. 打印结果
    print(f"\n>>> 节点 {target_node + 1} (Bus 12) 重构结果对比:")
    feats = ['Voltage (pu)', 'P (pu)', 'Q (pu)', 'Theta (rad)']

    vec_true = x_true_phys[0, target_node, :].cpu().numpy()
    vec_recon = x_recon_phys[0, target_node, :].cpu().numpy()

    print(f"{'Feature':<15} | {'Real Value':<12} | {'Reconstructed':<12} | {'Error':<10}")
    print("-" * 55)
    for i in range(4):
        err = abs(vec_true[i] - vec_recon[i])
        print(f"{feats[i]:<15} | {vec_true[i]:.4f}       | {vec_recon[i]:.4f}       | {err:.4f}")
    
    # [修改] 使用统一可视化工具函数
    # 1. 准备数据（展平为 2D 矩阵）
    x_true_2d = x_true_phys[0].reshape(-1).cpu().numpy()  # (137*4,)
    x_recon_2d = x_recon_phys[0].reshape(-1).cpu().numpy()  # (137*4,)
    
    # 2. 计算评估指标（与其他方法一致）
    print(f"\n📊 PI-GAT 重构性能评估:")
    
    # 计算 RMSE 和 MAE（全局所有特征）
    rmse = np.sqrt(mean_squared_error(x_true_2d, x_recon_2d))
    mae = mean_absolute_error(x_true_2d, x_recon_2d)
    
    # 过滤掉接近 0 的值，防止 MAPE 计算异常
    epsilon = 1e-8
    valid_indices = np.abs(x_true_2d) > epsilon
    x_true_valid = x_true_2d[valid_indices]
    x_recon_valid = x_recon_2d[valid_indices]
    
    if len(x_true_valid) > 0:
        mape = mean_absolute_percentage_error(x_true_valid, x_recon_valid) * 100
    else:
        mape = np.nan
    
    print(f"   传统统计指标:")
    print(f"      MSE  : {rmse ** 2:.6f}")
    print(f"      RMSE : {rmse:.6f}")
    print(f"      MAE  : {mae:.6f}")
    print(f"      MAPE : {mape:.4f}%")
    
    # [新增] 计算物理一致性指标 PID
    A_inc, num_buses, num_lines = load_grid_topology()
    A_inc = A_inc.astype(np.float32)
    
    pid_true = calculate_pid_metric(x_true_phys.cpu().numpy(), A_inc, num_buses, num_lines)
    pid_recovered = calculate_pid_metric(x_recon_phys.cpu().numpy(), A_inc, num_buses, num_lines)
    pid_improvement = (pid_true - pid_recovered) / pid_true * 100 if pid_true > 0 else 0
    
    print(f"\n   物理一致性指标 (PID):")
    print(f"      真实数据 PID:    {pid_true:.6f}")
    print(f"      重构数据 PID:    {pid_recovered:.6f}")
    print(f"      PID 变化率：     {pid_improvement:.2f}%")
    
    # 3. 时序对比图（取所有母线的 P 特征）
    p_true_all_buses = x_true_phys[0, :57, 1].cpu().numpy()  # (57,)
    p_recon_all_buses = x_recon_phys[0, :57, 1].cpu().numpy()  # (57,)
    
    plot_time_series_comparison(
        y_true=p_true_all_buses,
        y_pred=p_recon_all_buses,
        method_name='PI-GAT',
        feature_idx=0,
        plot_len=57
    )
    
    # 4. 单样本全特征对比图
    plot_single_sample_comparison(
        pred_matrix=x_recon_2d.reshape(1, -1),
        true_matrix=x_true_2d.reshape(1, -1),
        sample_idx=0,
        method_name='PI-GAT'
    )


# ==========================================
# 辅助函数：PID计算（与其他方法一致）
# ==========================================
def load_grid_topology(data_dir='dataset_split'):
    """
    加载电网拓扑结构（母线 - 支路关联矩阵）
    """
    print(f">>> 加载电网拓扑结构...")
    try:
        branch_data = sio.loadmat(os.path.join(data_dir, 'Topology_Branch.mat'))['branch_data']

        num_buses = 57
        num_lines = branch_data.shape[0]

        # 构建母线 - 支路关联矩阵 A_inc (57 × 支路数)
        A_inc = np.zeros((num_buses, num_lines))
        for l_idx in range(num_lines):
            b_from = int(branch_data[l_idx, 0]) - 1  # MATLAB 索引转 Python
            b_to = int(branch_data[l_idx, 1]) - 1
            A_inc[b_from, l_idx] = 1.0
            A_inc[b_to, l_idx] = -1.0

        print(f"   母线数：{num_buses}, 支路数：{num_lines}")
        return A_inc, num_buses, num_lines
    except Exception as e:
        print(f"拓扑加载失败：{e}，使用随机矩阵代替")
        return np.random.randn(57, 80), 57, 80


def calculate_pid_metric(reconstructed_data, A_inc, num_buses, num_lines):
    """
    计算物理一致性指标 PID (Power Imbalance Degree)
    
    数据结构 (3D): (N_samples, N_nodes, 4)
    - 节点 0-56: 57 个母线
    - 节点 57-136: 80 个支路
    - 每个节点 4 个特征：[V, P, Q, Theta]
    """
    if reconstructed_data.ndim == 3:
        N_samples = reconstructed_data.shape[0]
        N_nodes = reconstructed_data.shape[1]
        features_per_node = reconstructed_data.shape[2]
    else:
        print(f"⚠️  警告：数据维度不正确！期望 3D 数组，实际 {reconstructed_data.ndim}D")
        print(f"   跳过 PID计算...")
        return np.nan

    pid_values = []

    for i in range(N_samples):
        sample = reconstructed_data[i]

        # 提取母线有功/无功注入 (前 57 个节点，通道 1=P, 通道 2=Q)
        bus_p = sample[:num_buses, 1]  # P_inj
        bus_q = sample[:num_buses, 2]  # Q_inj

        # 提取支路潮流 (后 80 个支路，通道 0=P, 通道 1=Q)
        branch_p = sample[num_buses:, 0]  # P_flow
        branch_q = sample[num_buses:, 1]  # Q_flow

        # 计算 KCL 残差：P_bus - A_inc @ P_branch
        p_residual = bus_p - A_inc @ branch_p
        q_residual = bus_q - A_inc @ branch_q

        # L1 范数（绝对值之和）
        p_imbalance = np.sum(np.abs(p_residual))
        q_imbalance = np.sum(np.abs(q_residual))

        # PID = (P 不平衡 + Q 不平衡) / 母线数
        pid = (p_imbalance + q_imbalance) / num_buses
        pid_values.append(pid)

    return np.mean(pid_values)

if __name__ == "__main__":
    main_test()