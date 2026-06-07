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
# 1. IEEE 57 数据加载器 (无归一化)
# ==========================================
class IEEE57GraphDataset:
    def __init__(self, data_dir='dataset_split'):
        print(f">>> 正在加载 IEEE 57 数据集 (原始标幺值): {data_dir} ...")

        # 1. 加载 MATLAB 数据
        self.samples = sio.loadmat(os.path.join(data_dir, 'Samples_x_y_normal_operation_all.mat'))['Samples']
        self.branch_data = sio.loadmat(os.path.join(data_dir, 'Topology_Branch.mat'))['branch_data']

        # 索引修正
        pmu_raw = sio.loadmat(os.path.join(data_dir, 'PMU_Position.mat'))['pmu_position'][0]
        self.pmu_pos = pmu_raw.astype(np.int64) - 1
        self.branch_data = self.branch_data.astype(np.int64) - 1

        self.num_buses = 57
        self.num_lines = self.branch_data.shape[0]
        self.num_nodes = self.num_buses + self.num_lines

        # 2. 构建图结构
        self.edge_index, self.B_inc = self._build_graph_structure()

        # 3. 处理特征 (直接使用原始值)
        self.x = self._process_features()

        print(f"数据加载完成: {self.x.shape}")

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
        num_samples = self.samples.shape[0]
        # 初始化全 0 矩阵 (物理意义上的 0)
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

        # 填充 Bus
        x_graph[:, :57, 0] = samples_t[:, idx_V_scada]
        x_graph[:, :57, 1] = samples_t[:, idx_P_inj]
        x_graph[:, :57, 2] = samples_t[:, idx_Q_inj]
        x_graph[:, self.pmu_pos, 3] = samples_t[:, idx_Del_pmu]

        # 填充 Line
        x_graph[:, 57:, 0] = samples_t[:, idx_P_flow]
        x_graph[:, 57:, 1] = samples_t[:, idx_Q_flow]

        branch_indices_from = np.where(i_from_mask)[0]
        branch_indices_to = np.where(i_to_mask)[0]
        x_graph[:, 57 + branch_indices_from, 2] = samples_t[:, idx_Ire_from]
        x_graph[:, 57 + branch_indices_from, 3] = samples_t[:, idx_Iim_from]
        x_graph[:, 57 + branch_indices_to, 2] = samples_t[:, idx_Ire_to]
        x_graph[:, 57 + branch_indices_to, 3] = samples_t[:, idx_Iim_to]

        # [修改] 直接返回原始数值，不进行归一化
        return x_graph


# ==========================================
# 2. 模型定义 (保持增强版参数)
# ==========================================
class GAT_AE(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=8, bottleneck_channels=16, out_channels=4, num_heads=4):
        super(GAT_AE, self).__init__()

        # Encoder
        self.enc1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)   # 4 -> 256
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)

        self.enc2 = GATConv(hidden_channels * num_heads, bottleneck_channels, heads=1, concat=False)  # 256 -> 16
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)

        # Decoder
        self.dec1 = GATConv(bottleneck_channels, hidden_channels, heads=num_heads, concat=True)  # 16 -> 256
        self.bn3 = nn.BatchNorm1d(hidden_channels * num_heads)

        self.dec2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False)  # 256 -> 64
        self.bn4 = nn.BatchNorm1d(hidden_channels)

        self.dec_head = nn.Linear(hidden_channels, out_channels)  # 64 -> 4

    def forward(self, x, edge_index):
        B, N, Fdim = x.shape

        x_flat = x.reshape(B * N, Fdim)
        E = edge_index.shape[1]
        offsets = (torch.arange(B, device=x.device) * N).view(B, 1, 1)
        edge_batch = (edge_index.view(1, 2, E) + offsets).reshape(2, B * E)

        # Encoder
        h = self.enc1(x_flat, edge_batch)
        h = self.bn1(h)
        h = F.elu(h)

        h = self.enc2(h, edge_batch)
        h = self.bn2(h)
        h = F.elu(h)

        h = h.view(B, N, -1)

        h_flat = h.reshape(B * N, -1)

        # Decoder
        h_rec = self.dec1(h_flat, edge_batch)
        h_rec = self.bn3(h_rec)
        h_rec = F.elu(h_rec)

        h_rec = self.dec2(h_rec, edge_batch)
        h_rec = self.bn4(h_rec)
        h_rec = F.elu(h_rec)

        out = self.dec_head(h_rec)
        return out.view(B, N, -1)


# ==========================================
# 3. [简化] 物理 Loss (直接计算，无需反归一化)
# ==========================================
def physics_loss_ieee57_direct(recon_x, B_inc):
    """
    recon_x: 模型的直接输出 (已经是标幺值)
    """
    num_buses = 57
    bus_p = recon_x[:, :num_buses, 1]
    bus_q = recon_x[:, :num_buses, 2]
    branch_p = recon_x[:, num_buses:, 0]
    branch_q = recon_x[:, num_buses:, 1]

    # 直接计算 KCL
    agg_p = torch.matmul(branch_p, B_inc.T)
    agg_q = torch.matmul(branch_q, B_inc.T)

    loss_p = F.mse_loss(bus_p, agg_p)
    loss_q = F.mse_loss(bus_q, agg_q)

    return loss_p + loss_q


# ==========================================
# 4. 训练主程序 (添加早停机制)
# ==========================================
def main_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    dataset = IEEE57GraphDataset(data_dir='dataset_split')
    x_all = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    B_inc = dataset.B_inc.to(device)

    # ==========================================
    # 🔥 训练设置
    # ==========================================
    IS_SHUFFLE = False
    BATCH_SIZE = 16
    EPOCHS = 400
    PATIENCE = 100  # 早停耐心值：允许连续多少个 epoch 没有改善
    # ==========================================

    # 划分训练集和验证集
    num_samples = x_all.shape[0]
    num_train = int(num_samples * 0.9)  # 90% 用于训练
    num_val = num_samples - num_train  # 10% 用于验证

    train_loader = torch.utils.data.DataLoader(x_all[:num_train], batch_size=BATCH_SIZE, shuffle=IS_SHUFFLE)
    val_loader = torch.utils.data.DataLoader(x_all[num_train:], batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = GAT_AE(in_channels=4, hidden_channels=8, bottleneck_channels=16, out_channels=4, num_heads=4).to(device)

    # 优化器和调度器
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=10,
        min_lr=1e-6
    )

    print(f"\n>>> 开始训练 (无归一化模式)...")
    model.train()

    fixed_mask = None
    best_avg_loss = float('inf')  # 记录最佳验证损失
    best_val_loss = float('inf')  # 记录最佳验证损失
    best_model_state = None  # 保存最优模型的状态字典
    patience_counter = 0  # 早停计数器

    for epoch in range(EPOCHS):
        total_loss = 0
        total_mse = 0
        total_phy = 0

        # 训练阶段
        for batch_idx, batch_x in enumerate(train_loader):
            optimizer.zero_grad()
            B, N, Fdim = batch_x.shape

            # 普通GAT-AE，不使用掩码
            recon = model(batch_x, edge_index)

            # Loss 计算（普通 MSE，不使用掩码）
            loss_mse = F.mse_loss(recon, batch_x)
            loss_phy = physics_loss_ieee57_direct(recon, B_inc)
            loss = loss_mse + 0.1 * loss_phy

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_phy += loss_phy.item()

        # scheduler.step()


        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x in val_loader:
                B, N, Fdim = batch_x.shape
                # 验证集，不使用掩码
                recon = model(batch_x, edge_index)
                loss_mse = F.mse_loss(recon, batch_x)
                loss_phy = physics_loss_ieee57_direct(recon, B_inc)
                loss = loss_mse + 0.1 * loss_phy
                val_loss += loss.item()

        val_loss /= len(val_loader)

        model.train()
        scheduler.step(val_loss)
        # 打印日志
        avg_loss = total_loss / len(train_loader)
        avg_mse = total_mse / len(train_loader)
        avg_phy = total_phy / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1:04d} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"MSE: {avg_mse:.6f} | Phy: {avg_phy:.6f} | LR: {lr:.6f}"
        )

        # # 早停逻辑
        # if avg_loss < best_avg_loss:
        #     best_avg_loss = avg_loss
        #     patience_counter = 0
        #     # 更新最优模型状态
        #     best_model_state = model.state_dict()
        # else:
        #     patience_counter += 1
        #     if patience_counter >= PATIENCE:
        #         print(f"早停触发：验证损失已连续 {PATIENCE} 个 epoch 未改善，停止训练。")
        #         break

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 更新最优模型状态
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停触发：验证损失已连续 {PATIENCE} 个 epoch 未改善，停止训练。")
                break

    # 训练结束后保存最优模型
    if best_model_state is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = f'gat_ae_ieee57_best_{timestamp}.pth'
        torch.save(best_model_state, model_path)
        print(f"最优模型已保存为 {model_path}")

    print("训练结束！")


# ==========================================
# 5. 测试和评估代码
# ==========================================
def main_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n>>> 开始测试评估...")

    # 1. 加载数据
    dataset = IEEE57GraphDataset(data_dir='dataset_split')
    x_all = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    B_inc = dataset.B_inc.to(device)

    # 2. 加载最优模型
    import glob
    model_files = glob.glob('gat_ae_ieee57_best_*.pth')
    if not model_files:
        print("❌ 错误：找不到模型文件 'gat_ae_ieee57_best_*.pth'")
        return

    latest_model = sorted(model_files)[-1]
    print(f">>> 加载模型：{latest_model}")

    model = GAT_AE(in_channels=4, hidden_channels=8, bottleneck_channels=16, out_channels=4, num_heads=4).to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device, weights_only=True))
    model.eval()

    # 3. 准备测试集（后 10%）
    num_samples = x_all.shape[0]
    num_train = int(num_samples * 0.9)
    test_data = x_all[num_train:]  # 后 10% 作为测试集

    print(f"测试集样本数：{test_data.shape[0]}")

    # 4. 推理评估（不使用掩码）
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for i in range(len(test_data)):
            x_sample = test_data[i:i + 1]  # (1, 137, 4)

            # 前向推理（不使用掩码）
            recon = model(x_sample, edge_index)

            # 收集结果
            all_preds.append(recon.cpu().numpy())
            all_trues.append(x_sample.cpu().numpy())

    # 拼接结果
    pred_matrix = np.concatenate(all_preds, axis=0)  # (N_test, 137, 4)
    true_matrix = np.concatenate(all_trues, axis=0)  # (N_test, 137, 4)

    print(f"\n测试集总样本数：{pred_matrix.shape[0]}")
    print(f"特征维度：{pred_matrix.shape[1]} × {pred_matrix.shape[2]}")

    # 5. 计算评估指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

    # 展平为 2D
    pred_2d = pred_matrix.reshape(pred_matrix.shape[0], -1)
    true_2d = true_matrix.reshape(true_matrix.shape[0], -1)

    # 计算所有数据的误差（不使用掩码）
    true_values = true_2d.flatten()
    recovered_values = pred_2d.flatten()

    # 过滤接近 0 的值
    epsilon = 1e-8
    valid_indices = np.abs(true_values) > epsilon
    true_valid = true_values[valid_indices]
    recovered_valid = recovered_values[valid_indices]

    rmse = np.sqrt(mean_squared_error(true_values, recovered_values))
    mae = mean_absolute_error(true_values, recovered_values)

    if len(true_valid) > 0:
        mape = mean_absolute_percentage_error(true_valid, recovered_valid) * 100
    else:
        mape = np.nan

    print(f"\n📊 GAT-AE 重构性能 (全部数据，共 {len(true_values)} 个数据点):")
    print(f"   传统统计指标:")
    print(f"      MSE  : {rmse ** 2:.6f}")
    print(f"      RMSE : {rmse:.6f}")
    print(f"      MAE  : {mae:.6f}")
    print(f"      MAPE : {mape:.4f}%")

    # 6. 计算物理一致性指标 PID
    A_inc, num_buses, num_lines = load_grid_topology()
    A_inc = A_inc.astype(np.float32)

    pid_true = calculate_pid_metric(true_matrix, A_inc, num_buses, num_lines)
    pid_recovered = calculate_pid_metric(pred_matrix, A_inc, num_buses, num_lines)
    pid_improvement = (pid_true - pid_recovered) / pid_true * 100 if pid_true > 0 else 0

    print(f"\n   物理一致性指标 (PID):")
    print(f"      真实数据 PID:    {pid_true:.6f}")
    print(f"      重构数据 PID:    {pid_recovered:.6f}")
    print(f"      PID 变化率：     {pid_improvement:.2f}%")

    # 7. 可视化对比
    from visualization_utils import plot_single_sample_comparison, plot_time_series_comparison, set_random_seed

    # 固定随机种子
    set_random_seed(2025)

    # 时序对比图（取第一个测试样本的所有母线 P 特征）
    sample_idx = 0
    p_true_all_buses = true_matrix[sample_idx, :57, 1]
    p_recon_all_buses = pred_matrix[sample_idx, :57, 1]

    plot_time_series_comparison(
        y_true=p_true_all_buses,
        y_pred=p_recon_all_buses,
        method_name='GAT-AE',
        feature_idx=0,
        plot_len=57
    )

    # 单样本全特征对比图
    plot_single_sample_comparison(
        pred_matrix=pred_2d[sample_idx:sample_idx + 1],
        true_matrix=true_2d[sample_idx:sample_idx + 1],
        sample_idx=0,
        method_name='GAT-AE'
    )


# ==========================================
# 辅助函数：PID计算（与其他方法一致）
# ==========================================
def load_grid_topology(data_dir='dataset_split'):
    """
    加载电网拓扑结构（母线-支路关联矩阵）
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
    main_train()  # 先训练
    main_test()  # 再测试
