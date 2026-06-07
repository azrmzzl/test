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
# 1. IEEE 57 数据加载器 (仅P/Q特征)
# ==========================================
class IEEE57PQDataset:
    def __init__(self, data_dir='dataset_split'):
        print(f">>> 正在加载 IEEE 57 数据集 (仅P/Q特征): {data_dir} ...")

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

        # 3. 处理特征 (仅保留P和Q)
        self.x = self._process_features_pq_only()

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

    def _process_features_pq_only(self):
        """
        仅提取P和Q特征：
        - 母线节点: [P_inj, Q_inj] (2维)
        - 支路节点: [P_flow, Q_flow] (2维)
        """
        num_samples = self.samples.shape[0]
        # 初始化全 0 矩阵 (2个特征: P, Q)
        x_graph = torch.zeros(num_samples, self.num_nodes, 2)

        idx = 0
        idx_V_scada = slice(0, 57); idx += 57
        idx_P_inj = slice(idx, idx + 57); idx += 57
        idx_Q_inj = slice(idx, idx + 57); idx += 57
        idx_P_flow = slice(idx, idx + 80); idx += 80
        idx_Q_flow = slice(idx, idx + 80); idx += 80
        
        # 跳过PMU相关特征（不需要）
        n_pmu = len(self.pmu_pos)
        idx += n_pmu  # Del_pmu
        idx += n_pmu  # V_pmu
        
        i_from_mask = np.isin(self.branch_data[:, 0], self.pmu_pos)
        i_to_mask = np.isin(self.branch_data[:, 1], self.pmu_pos)
        n_if = np.sum(i_from_mask)
        n_it = np.sum(i_to_mask)
        idx += n_if  # Ire_from
        idx += n_if  # Iim_from
        idx += n_it  # Ire_to
        idx += n_it  # Iim_to

        samples_t = torch.tensor(self.samples, dtype=torch.float32)

        # 填充母线节点的P和Q
        x_graph[:, :57, 0] = samples_t[:, idx_P_inj]   # P_inj
        x_graph[:, :57, 1] = samples_t[:, idx_Q_inj]   # Q_inj

        # 填充支路节点的P和Q
        x_graph[:, 57:, 0] = samples_t[:, idx_P_flow]  # P_flow
        x_graph[:, 57:, 1] = samples_t[:, idx_Q_flow]  # Q_flow

        return x_graph


# ==========================================
# 2. 模型定义 (2维特征 + 残差连接)
# ==========================================
class PI_GraphMAE_PQ(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=16, bottleneck_channels=8, out_channels=2, num_heads=4):
        super(PI_GraphMAE_PQ, self).__init__()

        self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, 1, bottleneck_channels))

        # Encoder - 添加残差连接
        self.enc1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.residual_proj1 = nn.Linear(in_channels, hidden_channels * num_heads)

        self.enc2 = GATConv(hidden_channels * num_heads, bottleneck_channels, heads=1, concat=False)
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)
        self.residual_proj2 = nn.Linear(hidden_channels * num_heads, bottleneck_channels)

        # Decoder - 添加残差连接
        self.dec1 = GATConv(bottleneck_channels, hidden_channels, heads=num_heads, concat=True)
        self.bn3 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.residual_proj3 = nn.Linear(bottleneck_channels, hidden_channels * num_heads)

        self.dec2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.residual_proj4 = nn.Linear(hidden_channels * num_heads, hidden_channels)

        self.dec_head = nn.Linear(hidden_channels, out_channels)

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

        # Encoder Layer 1 with Residual
        h = self.enc1(x_flat, edge_batch)
        h = self.bn1(h)
        h_res = self.residual_proj1(x_flat)
        h = F.elu(h + h_res)

        # Encoder Layer 2 with Residual
        h_before_enc2 = h.clone()
        h = self.enc2(h, edge_batch)
        h = self.bn2(h)
        h_res2 = self.residual_proj2(h_before_enc2)
        h = F.elu(h + h_res2)

        h = h.view(B, N, -1)

        if mask is not None:
            token_dec = self.dec_mask_token.expand(B, N, h.size(-1))
            h = h * (1 - mask) + token_dec * mask

        h_flat = h.reshape(B * N, -1)

        # Decoder Layer 1 with Residual
        h_before_dec1 = h_flat.clone()
        h_rec = self.dec1(h_flat, edge_batch)
        h_rec = self.bn3(h_rec)
        h_rec_res = self.residual_proj3(h_before_dec1)
        h_rec = F.elu(h_rec + h_rec_res)

        # Decoder Layer 2 with Residual
        h_before_dec2 = h_rec.clone()
        h_rec = self.dec2(h_rec, edge_batch)
        h_rec = self.bn4(h_rec)
        h_rec_res2 = self.residual_proj4(h_before_dec2)
        h_rec = F.elu(h_rec + h_rec_res2)

        out = self.dec_head(h_rec)
        return out.view(B, N, -1)


# ==========================================
# 3. 物理 Loss (仅P/Q)
# ==========================================
def physics_loss_pq_direct(recon_x, B_inc, mask=None):
    """
    recon_x: (B, N, 2) - 仅包含P和Q
    B_inc: 母线-支路关联矩阵
    mask: (B, N, 1) - 掩码矩阵，用于只计算未掩码节点的物理损失
    """
    num_buses = 57
    bus_p = recon_x[:, :num_buses, 0]  # P_inj
    bus_q = recon_x[:, :num_buses, 1]  # Q_inj
    branch_p = recon_x[:, num_buses:, 0]  # P_flow
    branch_q = recon_x[:, num_buses:, 1]  # Q_flow

    # KCL约束: P_bus = A_inc @ P_branch
    agg_p = torch.matmul(branch_p, B_inc.T)
    agg_q = torch.matmul(branch_q, B_inc.T)

    if mask is not None:
        # [修复] 只计算未掩码母线的物理损失
        bus_mask = mask[:, :num_buses, 0]  # (B, 57)
        unmasked_weight = 1.0 - bus_mask  # 未掩码=1, 掩码=0
        
        # 加权MSE：未掩码节点权重高，掩码节点权重低
        loss_p = torch.mean(unmasked_weight * (bus_p - agg_p) ** 2)
        loss_q = torch.mean(unmasked_weight * (bus_q - agg_q) ** 2)
    else:
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
    dataset = IEEE57PQDataset(data_dir='dataset_split')
    x_all = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    B_inc = dataset.B_inc.to(device)

    # ==========================================
    # 🔥 训练设置
    # ==========================================
    FIXED_MASK_MODE = False
    IS_SHUFFLE = False
    BATCH_SIZE = 64
    EPOCHS = 1000
    PATIENCE = 100
    MASK_RATIO = 0.15
    # ==========================================

    # 划分训练集和验证集
    num_samples = x_all.shape[0]
    num_train = int(num_samples * 0.9)
    num_val = num_samples - num_train

    train_loader = torch.utils.data.DataLoader(x_all[:num_train], batch_size=BATCH_SIZE, shuffle=IS_SHUFFLE)
    val_loader = torch.utils.data.DataLoader(x_all[num_train:], batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = PI_GraphMAE_PQ(in_channels=2, hidden_channels=16, bottleneck_channels=8, out_channels=2, num_heads=4).to(device)

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=15,
        min_lr=1e-6
    )

    print(f"\n>>> 开始训练 (仅P/Q特征)...")
    model.train()

    fixed_mask = None
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        total_loss = 0
        total_mse = 0
        total_phy = 0

        # 训练阶段
        for batch_idx, batch_x in enumerate(train_loader):
            optimizer.zero_grad()
            B, N, Fdim = batch_x.shape

            # 掩码生成
            if FIXED_MASK_MODE:
                if fixed_mask is None:
                    fixed_mask = (torch.rand(1, N, 1, device=device) < MASK_RATIO).float()
                mask = fixed_mask.expand(B, N, 1)
            else:
                mask = (torch.rand(B, N, 1, device=device) < MASK_RATIO).float()

            recon = model(batch_x, edge_index, mask)

            # 损失计算 - [优化] 对P给予更高权重
            loss_mse_masked = F.mse_loss(recon * mask, batch_x * mask)
            
            # [新增] P/Q分别加权：P的权重设为2.0，Q为1.0
            mask_expanded = mask.expand_as(batch_x)
            feature_weights = torch.tensor([2.0, 1.0], device=batch_x.device)  # P权重更高
            weighted_mask = mask_expanded * feature_weights.view(1, 1, -1)
            loss_mse_weighted = torch.mean(weighted_mask * (recon - batch_x) ** 2) / torch.mean(weighted_mask)
            
            loss_mse_all = F.mse_loss(recon, batch_x)
            # [优化] 平衡三者权重：掩码40% + 加权20% + 全局40%
            loss_mse = 0.4 * loss_mse_masked + 0.2 * loss_mse_weighted + 0.4 * loss_mse_all
            
            loss_phy = physics_loss_pq_direct(recon, B_inc, mask)
            
            # 特征平滑正则化
            loss_smooth = torch.mean(torch.abs(recon[:, 1:, :] - recon[:, :-1, :]))
            
            loss = loss_mse + 0.1 * loss_phy + 0.02 * loss_smooth

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_phy += loss_phy.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        val_mse_masked = 0
        val_mse_all = 0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                B, N, Fdim = batch_x.shape
                if FIXED_MASK_MODE:
                    mask = fixed_mask.expand(B, N, 1) if fixed_mask is not None else \
                        (torch.rand(1, N, 1, device=device) < MASK_RATIO).float().expand(B, N, 1)
                else:
                    mask = (torch.rand(B, N, 1, device=device) < MASK_RATIO).float()

                recon = model(batch_x, edge_index, mask)
                loss_mse_masked_val = F.mse_loss(recon * mask, batch_x * mask)
                
                # P/Q分别加权
                mask_expanded_val = mask.expand_as(batch_x)
                feature_weights_val = torch.tensor([2.0, 1.0], device=batch_x.device)
                weighted_mask_val = mask_expanded_val * feature_weights_val.view(1, 1, -1)
                loss_mse_weighted_val = torch.mean(weighted_mask_val * (recon - batch_x) ** 2) / torch.mean(weighted_mask_val)
                
                loss_mse_all_val = F.mse_loss(recon, batch_x)
                # [优化] 平衡三者权重：掩码40% + 加权20% + 全局40%
                loss_mse_val = 0 * loss_mse_masked_val + 0 * loss_mse_weighted_val + 1 * loss_mse_all_val
                loss_phy_val = physics_loss_pq_direct(recon, B_inc, mask)
                loss_smooth_val = torch.mean(torch.abs(recon[:, 1:, :] - recon[:, :-1, :]))
                loss = loss_mse_val + 0.1 * loss_phy_val + 0.02 * loss_smooth_val
                
                val_loss += loss.item()
                val_mse_masked += loss_mse_masked_val.item()
                val_mse_all += loss_mse_all_val.item()
        
        val_loss /= len(val_loader)
        val_mse_masked /= len(val_loader)
        val_mse_all /= len(val_loader)

        model.train()
        scheduler.step(val_loss)
        
        # 打印日志
        avg_loss = total_loss / len(train_loader)
        avg_mse = total_mse / len(train_loader)
        avg_phy = total_phy / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1:04d} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val MSE(masked): {val_mse_masked:.6f} | Val MSE(all): {val_mse_all:.6f} | "
            f"Train MSE: {avg_mse:.6f} | Phy: {avg_phy:.6f} | LR: {lr:.6f}"
        )

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"早停触发：验证损失已连续 {PATIENCE} 个 epoch 未改善，停止训练。")
                break

    # 保存最优模型
    if best_model_state is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = f'pi_graphmae_pq_best_{timestamp}.pth'
        torch.save(best_model_state, model_path)
        print(f"最优模型已保存为 {model_path}")

    print("训练结束！")


# ==========================================
# 5. 测试和评估
# ==========================================
def main_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n>>> 开始测试评估...")

    # 1. 加载数据
    dataset = IEEE57PQDataset(data_dir='dataset_split')
    x_all = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    B_inc = dataset.B_inc.to(device)
    
    # 数据分布诊断
    num_samples = x_all.shape[0]
    num_train = int(num_samples * 0.9)
    test_data = x_all[num_train:]
    
    print(f"\n=== 数据分布诊断 ===")
    print(f"训练集均值: {x_all[:num_train].mean():.4f}, 标准差: {x_all[:num_train].std():.4f}")
    print(f"测试集均值: {test_data.mean():.4f}, 标准差: {test_data.std():.4f}")
    print(f"测试集最小值: {test_data.min():.4f}, 最大值: {test_data.max():.4f}")
    zero_ratio = (test_data.abs() < 1e-6).sum() / test_data.numel() * 100
    print(f"零值比例: {zero_ratio:.2f}%")
    print(f"各特征通道统计:")
    for i in range(2):
        feat_name = ['P', 'Q'][i]
        feat_data = test_data[:, :, i]
        print(f"  {feat_name}: 均值={feat_data.mean():.4f}, 标准差={feat_data.std():.4f}, "
              f"范围=[{feat_data.min():.4f}, {feat_data.max():.4f}]")

    # 2. 加载最优模型
    import glob
    model_files = glob.glob('pi_graphmae_pq_best_*.pth')
    if not model_files:
        print("❌ 错误：找不到模型文件 'pi_graphmae_pq_best_*.pth'")
        return

    latest_model = sorted(model_files)[-1]
    print(f">>> 加载模型：{latest_model}")

    model = PI_GraphMAE_PQ(in_channels=2, hidden_channels=16, bottleneck_channels=8, out_channels=2, num_heads=4).to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device, weights_only=True))
    model.eval()
    
    # 模型参数检查
    print(f"\n=== 模型参数检查 ===")
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")

    print(f"测试集样本数：{test_data.shape[0]}")

    # 3. 批量推理
    print(f"\n=== 开始批量推理 ===")
    torch.manual_seed(2025)
    np.random.seed(2025)
    
    all_preds = []
    all_trues = []
    all_masks = []
    
    test_batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    with torch.no_grad():
        for batch_idx, batch_x in enumerate(test_loader):
            batch_x = batch_x.to(device)
            B, N, Fdim = batch_x.shape
            mask = (torch.rand(B, N, 1, device=device) < 0.15).float()
            recon = model(batch_x, edge_index, mask)
            
            all_preds.append(recon.cpu().numpy())
            all_trues.append(batch_x.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"   已处理 {batch_idx + 1}/{len(test_loader)} 个batch")

    # 拼接结果
    pred_matrix = np.concatenate(all_preds, axis=0)
    true_matrix = np.concatenate(all_trues, axis=0)
    mask_matrix = np.concatenate(all_masks, axis=0)

    print(f"\n测试集总样本数：{pred_matrix.shape[0]}")
    print(f"特征维度：{pred_matrix.shape[1]} × {pred_matrix.shape[2]}")
    
    # 预测值分布诊断
    print(f"\n=== 预测值分布诊断 ===")
    print(f"预测值均值: {pred_matrix.mean():.4f}, 标准差: {pred_matrix.std():.4f}")
    print(f"预测值范围: [{pred_matrix.min():.4f}, {pred_matrix.max():.4f}]")
    print(f"真实值均值: {true_matrix.mean():.4f}, 标准差: {true_matrix.std():.4f}")
    
    for i in range(2):
        feat_name = ['P', 'Q'][i]
        pred_feat = pred_matrix[:, :, i].flatten()
        true_feat = true_matrix[:, :, i].flatten()
        mask_feat = mask_matrix[:, :, 0].flatten() > 0.5
        
        masked_pred = pred_feat[mask_feat]
        masked_true = true_feat[mask_feat]
        
        if len(masked_true) > 0:
            channel_mse = np.mean((masked_pred - masked_true) ** 2)
            channel_mae = np.mean(np.abs(masked_pred - masked_true))
            print(f"  {feat_name} (掩码区域): MSE={channel_mse:.6f}, MAE={channel_mae:.6f}")

    # 4. 计算评估指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    pred_2d = pred_matrix.reshape(pred_matrix.shape[0], -1)
    true_2d = true_matrix.reshape(true_matrix.shape[0], -1)
    
    mask_matrix_expanded = np.repeat(mask_matrix, 2, axis=2)
    mask_flat = mask_matrix_expanded.reshape(mask_matrix_expanded.shape[0], -1) > 0.5
    
    true_masked = true_2d[mask_flat]
    pred_masked = pred_2d[mask_flat]
    
    print(f"\n=== 掩码统计 ===")
    print(f"总数据点数: {true_2d.size}")
    print(f"掩码数据点数: {true_masked.size} ({true_masked.size / true_2d.size * 100:.1f}%)")
    
    mse = mean_squared_error(true_masked, pred_masked)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_masked, pred_masked)
    
    # SMAPE计算
    epsilon = max(np.percentile(np.abs(true_masked), 5), 1e-4)
    valid_mask = np.abs(true_masked) > epsilon
    true_valid = true_masked[valid_mask]
    pred_valid = pred_masked[valid_mask]
    
    if len(true_valid) > 0:
        smape = np.mean(2.0 * np.abs(pred_valid - true_valid) / (np.abs(pred_valid) + np.abs(true_valid) + 1e-8)) * 100
    else:
        smape = np.nan

    print(f"\n📊 PI-GAT-PQ 重构性能 (掩码区域，共 {len(true_masked)} 个数据点):")
    print(f"   传统统计指标:")
    print(f"      MSE  : {mse:.6f}")
    print(f"      RMSE : {rmse:.6f}")
    print(f"      MAE  : {mae:.6f}")
    print(f"      SMAPE: {smape:.4f}%")
    
    # 未掩码区域误差
    unmask_flat = ~mask_flat
    true_unmasked = true_2d[unmask_flat]
    pred_unmasked = pred_2d[unmask_flat]
    mse_unmasked = mean_squared_error(true_unmasked, pred_unmasked)
    mae_unmasked = mean_absolute_error(true_unmasked, pred_unmasked)
    print(f"\n   未掩码区域误差:")
    print(f"      MSE  : {mse_unmasked:.6f}")
    print(f"      MAE  : {mae_unmasked:.6f}")

    # 5. 计算PID
    A_inc, num_buses, num_lines = load_grid_topology()
    A_inc = A_inc.astype(np.float32)
    
    pid_true = calculate_pid_metric_pq(true_matrix, A_inc, num_buses, num_lines)
    pid_recovered = calculate_pid_metric_pq(pred_matrix, A_inc, num_buses, num_lines)
    pid_improvement = (pid_true - pid_recovered) / pid_true * 100 if pid_true > 0 else 0
    
    print(f"\n   物理一致性指标 (PID):")
    print(f"      真实数据 PID:    {pid_true:.6f}")
    print(f"      重构数据 PID:    {pid_recovered:.6f}")
    print(f"      PID 变化率：     {pid_improvement:.2f}%")


# ==========================================
# 辅助函数
# ==========================================
def load_grid_topology(data_dir='dataset_split'):
    print(f">>> 加载电网拓扑结构...")
    try:
        branch_data = sio.loadmat(os.path.join(data_dir, 'Topology_Branch.mat'))['branch_data']
        num_buses = 57
        num_lines = branch_data.shape[0]
        
        A_inc = np.zeros((num_buses, num_lines))
        for l_idx in range(num_lines):
            b_from = int(branch_data[l_idx, 0]) - 1
            b_to = int(branch_data[l_idx, 1]) - 1
            A_inc[b_from, l_idx] = 1.0
            A_inc[b_to, l_idx] = -1.0
        
        print(f"   母线数：{num_buses}, 支路数：{num_lines}")
        return A_inc, num_buses, num_lines
    except Exception as e:
        print(f"拓扑加载失败：{e}，使用随机矩阵代替")
        return np.random.randn(57, 80), 57, 80


def calculate_pid_metric_pq(reconstructed_data, A_inc, num_buses, num_lines):
    """
    计算PID (仅使用P和Q)
    reconstructed_data: (N_samples, N_nodes, 2)
    """
    if reconstructed_data.ndim != 3:
        print(f"⚠️  警告：数据维度不正确！期望 3D 数组")
        return np.nan

    pid_values = []
    for i in range(reconstructed_data.shape[0]):
        sample = reconstructed_data[i]
        
        bus_p = sample[:num_buses, 0]
        bus_q = sample[:num_buses, 1]
        branch_p = sample[num_buses:, 0]
        branch_q = sample[num_buses:, 1]
        
        p_residual = bus_p - A_inc @ branch_p
        q_residual = bus_q - A_inc @ branch_q
        
        p_imbalance = np.sum(np.abs(p_residual))
        q_imbalance = np.sum(np.abs(q_residual))
        
        pid = (p_imbalance + q_imbalance) / num_buses
        pid_values.append(pid)

    return np.mean(pid_values)


if __name__ == "__main__":
    main_train()
    main_test()
