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
class PI_GraphMAE(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, bottleneck_channels=16, out_channels=4, num_heads=4):
        super(PI_GraphMAE, self).__init__()

        self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, 1, bottleneck_channels))

        # Encoder - [优化] 添加残差连接
        self.enc1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.residual_proj1 = nn.Linear(in_channels, hidden_channels * num_heads)

        self.enc2 = GATConv(hidden_channels * num_heads, bottleneck_channels, heads=1, concat=False)
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)
        self.residual_proj2 = nn.Linear(hidden_channels * num_heads, bottleneck_channels)

        # Decoder - [优化] 添加残差连接
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
        h = F.elu(h + h_res)  # 残差连接

        # Encoder Layer 2 with Residual
        h_before_enc2 = h.clone()  # 保存enc2前的特征
        h = self.enc2(h, edge_batch)
        h = self.bn2(h)
        h_res2 = self.residual_proj2(h_before_enc2)  # [修复] 使用enc2前的特征做残差
        h = F.elu(h + h_res2)  # 残差连接

        h = h.view(B, N, -1)

        if mask is not None:
            token_dec = self.dec_mask_token.expand(B, N, h.size(-1))
            h = h * (1 - mask) + token_dec * mask

        h_flat = h.reshape(B * N, -1)

        # Decoder Layer 1 with Residual
        h_before_dec1 = h_flat.clone()  # 保存dec1前的特征
        h_rec = self.dec1(h_flat, edge_batch)
        h_rec = self.bn3(h_rec)
        h_rec_res = self.residual_proj3(h_before_dec1)  # [修复] 使用dec1前的特征做残差
        h_rec = F.elu(h_rec + h_rec_res)  # 残差连接

        # Decoder Layer 2 with Residual
        h_before_dec2 = h_rec.clone()  # 保存dec2前的特征
        h_rec = self.dec2(h_rec, edge_batch)
        h_rec = self.bn4(h_rec)
        h_rec_res2 = self.residual_proj4(h_before_dec2)  # [修复] 使用dec2前的特征做残差
        h_rec = F.elu(h_rec + h_rec_res2)  # 残差连接

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
    FIXED_MASK_MODE = False
    IS_SHUFFLE = False
    BATCH_SIZE = 16
    EPOCHS = 1000
    PATIENCE = 100  # 早停耐心值：允许连续多少个 epoch 没有改善
    MASK_RATIO = 0.15  # [调整] 降低掩码率从0.2到0.15，因为数据有29.56%零值
    # ==========================================

    # 划分训练集和验证集
    num_samples = x_all.shape[0]
    num_train = int(num_samples * 0.9)  # 90% 用于训练
    num_val = num_samples - num_train  # 10% 用于验证

    train_loader = torch.utils.data.DataLoader(x_all[:num_train], batch_size=BATCH_SIZE, shuffle=IS_SHUFFLE)
    val_loader = torch.utils.data.DataLoader(x_all[num_train:], batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型 - [优化] 使用小模型+残差连接
    model = PI_GraphMAE(in_channels=4, hidden_channels=64, bottleneck_channels=16, out_channels=4, num_heads=4).to(device)

    # 优化器和调度器 - [调整] 使用标准学习率
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=15,
        min_lr=1e-6
    )

    print(f"\n>>> 开始训练 (无归一化模式)...")
    model.train()

    fixed_mask = None
    best_avg_loss = float('inf')  # 记录最佳验证损失
    best_val_loss = float('inf')  # 记录最佳验证损失
    best_model_state = None  # 保存最优模型的状态字典
    patience_counter = 0  # 早停计数器
    # 初始化 batch 计数器
    batch_counter = 0

    for epoch in range(EPOCHS):
        total_loss = 0
        total_mse = 0
        total_phy = 0

        # 训练阶段
        for batch_idx, batch_x in enumerate(train_loader):
            optimizer.zero_grad()
            B, N, Fdim = batch_x.shape

            # 掩码生成
            # 训练集
            if FIXED_MASK_MODE:
                if fixed_mask is None:
                    fixed_mask = (torch.rand(1, N, 1, device=device) < MASK_RATIO).float()
                mask = fixed_mask.expand(B, N, 1)  # ✅ 每次都 expand 到当前 B
            else:
                # 同样为整个 batch 生成统一掩码
                mask = (torch.rand(B, N, 1, device=device) < MASK_RATIO).float()


            '''
            # 打印掩码节点位置（每5个epoch打印一次，展示第一个和最后一个样本）
            if epoch % 1 == 0 and batch_idx == 0:
                B, N, _ = batch_x.shape  # 获取batch大小和节点数

                # 获取第一个样本的掩码位置
                first_sample_mask = mask[0].squeeze()  # shape: [N]
                first_masked_positions = torch.where(first_sample_mask == 1)[0]  # 掩码位置索引
                first_masked_count = len(first_masked_positions)

                # 获取第二个样本的掩码位置
                second_sample_mask = mask[1].squeeze()
                second_masked_positions = torch.where(second_sample_mask == 1)[0]
                second_masked_count = len(second_masked_positions)

                # 获取最后一个样本的掩码位置
                last_sample_mask = mask[-1].squeeze()  # shape: [N]
                last_masked_positions = torch.where(last_sample_mask == 1)[0]  # 掩码位置索引
                last_masked_count = len(last_masked_positions)

                # 打印第一个样本的掩码信息
                print(
                    f"Epoch {epoch + 1}: 第一个样本掩码节点数: {first_masked_count}/{N}, 位置: {first_masked_positions.tolist()[:10]}{'...' if first_masked_count > 10 else ''}")
                print(
                    f"Epoch {epoch + 1}: 第二个样本掩码节点数: {second_masked_count}/{N}, 位置: {second_masked_positions.tolist()[:10]}{'...' if second_masked_count > 10 else ''}")
                # 打印最后一个样本的掩码信息
                print(
                    f"Epoch {epoch + 1}: 最后一个样本掩码节点数: {last_masked_count}/{N}, 位置: {last_masked_positions.tolist()[:10]}{'...' if last_masked_count > 10 else ''}")

                # 新增：处理第二个 batch 的第一个和第二个样本
            if epoch % 1 == 0 and batch_idx == 1:  # 判断是否为第二个 batch
                B, N, _ = batch_x.shape  # 获取batch大小和节点数

                # 获取第一个样本的掩码位置
                first_sample_mask = mask[0].squeeze()  # shape: [N]
                first_masked_positions = torch.where(first_sample_mask == 1)[0]  # 掩码位置索引
                first_masked_count = len(first_masked_positions)

                # 获取第二个样本的掩码位置
                second_sample_mask = mask[1].squeeze()
                second_masked_positions = torch.where(second_sample_mask == 1)[0]
                second_masked_count = len(second_masked_positions)

                last_sample_mask = mask[-1].squeeze()
                last_masked_positions = torch.where(last_sample_mask == 1)[0]
                last_masked_count = len(last_masked_positions)

                # 打印第二个 batch 的第一个和第二个样本掩码信息
                print(
                    f"Epoch {epoch + 1}, Batch 2: 第一个样本掩码节点数: {first_masked_count}/{N}, 位置: {first_masked_positions.tolist()[:10]}{'...' if first_masked_count > 10 else ''}")
                print(
                    f"Epoch {epoch + 1}, Batch 2: 第二个样本掩码节点数: {second_masked_count}/{N}, 位置: {second_masked_positions.tolist()[:10]}{'...' if second_masked_count > 10 else ''}")
                print(
                        f"Epoch {epoch + 1}, Batch 2: 最后一个样本掩码节点数: {last_masked_count}/{N}, 位置: {last_masked_positions.tolist()[:10]}{'...' if last_masked_count > 10 else ''}")
                '''

                # 增加 batch 计数器
            batch_counter += 1


            recon = model(batch_x, edge_index, mask)

            # [核心修复] 损失函数重新设计 - [优化] 添加特征加权
            loss_mse_masked = F.mse_loss(recon * mask, batch_x * mask)
            
            # [新增] 对特征0（电压）给予额外权重，因为它最难预测
            mask_expanded = mask.expand_as(batch_x)
            feature_weights = torch.tensor([2.0, 1.0, 1.0, 1.0], device=batch_x.device)
            weighted_mask = mask_expanded * feature_weights.view(1, 1, -1)
            loss_mse_feature_weighted = torch.mean(weighted_mask * (recon - batch_x) ** 2) / torch.mean(weighted_mask)
            
            loss_mse_all = F.mse_loss(recon, batch_x)
            
            # 组合MSE损失：三重加权
            loss_mse = 0.5 * loss_mse_masked + 0.3 * loss_mse_feature_weighted + 0.2 * loss_mse_all
            
            # 物理约束损失
            loss_phy = physics_loss_ieee57_direct(recon, B_inc)
            
            # 特征平滑正则化
            loss_smooth = torch.mean(torch.abs(recon[:, 1:, :] - recon[:, :-1, :]))
            
            # 最终组合损失
            loss = loss_mse + 0.08 * loss_phy + 0.02 * loss_smooth

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_phy += loss_phy.item()

        # scheduler.step()


        # 验证阶段
        model.eval()
        val_loss = 0
        val_mse_masked = 0
        val_mse_all = 0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)  # [修复] 确保验证数据也在device上
                B, N, Fdim = batch_x.shape
                # 验证集
                if FIXED_MASK_MODE:
                    mask = fixed_mask.expand(B, N, 1) if fixed_mask is not None else \
                        (torch.rand(1, N, 1, device=device) < MASK_RATIO).float().expand(B, N, 1)
                else:
                    mask = (torch.rand(B, N, 1, device=device) < MASK_RATIO).float()

                recon = model(batch_x, edge_index, mask)
                
                # 验证时使用与训练相同的损失计算方式
                loss_mse_masked_val = F.mse_loss(recon * mask, batch_x * mask)
                
                # 特征加权MSE
                mask_expanded_val = mask.expand_as(batch_x)
                feature_weights_val = torch.tensor([2.0, 1.0, 1.0, 1.0], device=batch_x.device)
                weighted_mask_val = mask_expanded_val * feature_weights_val.view(1, 1, -1)
                loss_mse_feature_weighted_val = torch.mean(weighted_mask_val * (recon - batch_x) ** 2) / torch.mean(weighted_mask_val)
                
                loss_mse_all_val = F.mse_loss(recon, batch_x)
                loss_mse_val = 0.5 * loss_mse_masked_val + 0.3 * loss_mse_feature_weighted_val + 0.2 * loss_mse_all_val
                
                loss_phy_val = physics_loss_ieee57_direct(recon, B_inc)
                loss_smooth_val = torch.mean(torch.abs(recon[:, 1:, :] - recon[:, :-1, :]))
                loss = loss_mse_val + 0.08 * loss_phy_val + 0.02 * loss_smooth_val
                
                val_loss += loss.item()
                val_mse_masked += loss_mse_masked_val.item()
                val_mse_all += loss_mse_all_val.item()
        
        val_loss /= len(val_loader)
        val_mse_masked /= len(val_loader)
        val_mse_all /= len(val_loader)

        model.train()
        scheduler.step(val_loss)
        
        # 打印日志 - [增强] 显示更多验证集指标
        avg_loss = total_loss / len(train_loader)
        avg_mse = total_mse / len(train_loader)
        avg_phy = total_phy / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1:04d} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val MSE(masked): {val_mse_masked:.6f} | Val MSE(all): {val_mse_all:.6f} | "
            f"Train MSE: {avg_mse:.6f} | Phy: {avg_phy:.6f} | LR: {lr:.6f}"
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
        model_path = f'pi_graphmae_ieee57_best_{timestamp}.pth'
        torch.save(best_model_state, model_path)
        print(f"最优模型已保存为 {model_path}")

    print("训练结束！")


# ==========================================
# 5. 测试和评估代码（使用与训练时相同的掩码）
# ==========================================
def main_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n>>> 开始测试评估...")

    # 1. 加载数据
    dataset = IEEE57GraphDataset(data_dir='dataset_split')
    x_all = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    B_inc = dataset.B_inc.to(device)
    
    # [新增] 数据分布诊断
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
    for i in range(4):
        feat_data = test_data[:, :, i]
        print(f"  特征{i}: 均值={feat_data.mean():.4f}, 标准差={feat_data.std():.4f}, "
              f"范围=[{feat_data.min():.4f}, {feat_data.max():.4f}]")

    # 2. 加载最优模型
    import glob
    model_files = glob.glob('pi_graphmae_ieee57_best_*.pth')
    if not model_files:
        print("❌ 错误：找不到模型文件 'pi_graphmae_ieee57_best_*.pth'")
        return

    latest_model = sorted(model_files)[-1]
    print(f">>> 加载模型：{latest_model}")

    model = PI_GraphMAE(in_channels=4, hidden_channels=64, bottleneck_channels=16, out_channels=4, num_heads=4).to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device, weights_only=True))
    model.eval()
    
    # [新增] 模型参数检查
    print(f"\n=== 模型参数检查 ===")
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")

    # 3. 准备测试集（后 10%）- 已在前面定义
    print(f"测试集样本数：{test_data.shape[0]}")

    # 4. 推理评估 - [核心修复] 使用批量推理+固定种子确保可复现性
    print(f"\n=== 开始批量推理 ===")
    
    # [修复] 设置测试时的随机种子，确保每次运行结果一致
    torch.manual_seed(2025)
    np.random.seed(2025)
    
    all_preds = []
    all_trues = []
    all_masks = []
    
    # [优化] 使用更大的batch size加速推理
    test_batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    with torch.no_grad():
        for batch_idx, batch_x in enumerate(test_loader):
            batch_x = batch_x.to(device)
            B, N, Fdim = batch_x.shape

            # [关键修复] 为当前batch生成随机掩码（与训练时策略一致）
            mask = (torch.rand(B, N, 1, device=device) < 0.15).float()  # 测试时使用相同的掩码率

            # 前向推理
            recon = model(batch_x, edge_index, mask)

            # 收集结果
            all_preds.append(recon.cpu().numpy())
            all_trues.append(batch_x.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"   已处理 {batch_idx + 1}/{len(test_loader)} 个batch")

    # 拼接结果
    pred_matrix = np.concatenate(all_preds, axis=0)  # (N_test, 137, 4)
    true_matrix = np.concatenate(all_trues, axis=0)  # (N_test, 137, 4)
    mask_matrix = np.concatenate(all_masks, axis=0)  # (N_test, 137, 1)

    print(f"\n测试集总样本数：{pred_matrix.shape[0]}")
    print(f"特征维度：{pred_matrix.shape[1]} × {pred_matrix.shape[2]}")
    
    # [新增] 诊断：检查预测值的分布
    print(f"\n=== 预测值分布诊断 ===")
    print(f"预测值均值: {pred_matrix.mean():.4f}, 标准差: {pred_matrix.std():.4f}")
    print(f"预测值范围: [{pred_matrix.min():.4f}, {pred_matrix.max():.4f}]")
    print(f"真实值均值: {true_matrix.mean():.4f}, 标准差: {true_matrix.std():.4f}")
    print(f"真实值范围: [{true_matrix.min():.4f}, {true_matrix.max():.4f}]")
    
    # 检查每个通道的误差
    for i in range(4):
        pred_feat = pred_matrix[:, :, i].flatten()
        true_feat = true_matrix[:, :, i].flatten()
        mask_feat = mask_matrix[:, :, 0].flatten() > 0.5
        
        masked_pred = pred_feat[mask_feat]
        masked_true = true_feat[mask_feat]
        
        if len(masked_true) > 0:
            channel_mse = np.mean((masked_pred - masked_true) ** 2)
            channel_mae = np.mean(np.abs(masked_pred - masked_true))
            print(f"  特征{i} (掩码区域): MSE={channel_mse:.6f}, MAE={channel_mae:.6f}, "
                  f"预测均值={masked_pred.mean():.4f}, 真实均值={masked_true.mean():.4f}")

    # 5. 计算评估指标 - [核心修复] 正确处理零值和MAPE
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # [重要] GraphMAE的标准评估：只计算被掩码位置的误差
    # 展平为 2D
    pred_2d = pred_matrix.reshape(pred_matrix.shape[0], -1)
    true_2d = true_matrix.reshape(true_matrix.shape[0], -1)
    
    # 扩展掩码到所有特征维度
    mask_matrix_expanded = np.repeat(mask_matrix, 4, axis=2)  # (N_test, 137, 4)
    mask_flat = mask_matrix_expanded.reshape(mask_matrix_expanded.shape[0], -1) > 0.5
    
    # 只提取被掩码位置的值
    true_masked = true_2d[mask_flat]
    pred_masked = pred_2d[mask_flat]
    
    print(f"\n=== 掩码统计 ===")
    print(f"总数据点数: {true_2d.size}")
    print(f"掩码数据点数: {true_masked.size} ({true_masked.size / true_2d.size * 100:.1f}%)")
    print(f"掩码位置的真实值 - 均值: {true_masked.mean():.4f}, 标准差: {true_masked.std():.4f}")
    print(f"掩码位置的预测值 - 均值: {pred_masked.mean():.4f}, 标准差: {pred_masked.std():.4f}")
    
    # 计算基础指标
    mse = mean_squared_error(true_masked, pred_masked)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_masked, pred_masked)
    
    # [关键修复] MAPE计算 - 过滤掉接近0的值和异常值
    # 使用相对阈值而非绝对阈值
    epsilon_relative = np.percentile(np.abs(true_masked), 5)  # 取第5百分位数作为阈值
    epsilon_absolute = 1e-4
    epsilon = max(epsilon_relative, epsilon_absolute)
    
    valid_mask = np.abs(true_masked) > epsilon
    true_valid = true_masked[valid_mask]
    pred_valid = pred_masked[valid_mask]
    
    print(f"\nMAPE计算过滤:")
    print(f"  阈值epsilon: {epsilon:.6f}")
    print(f"  有效数据点: {len(true_valid)} / {len(true_masked)} ({len(true_valid)/len(true_masked)*100:.1f}%)")
    
    if len(true_valid) > 0:
        # [改进] 使用对称MAPE避免除零问题
        smape = np.mean(2.0 * np.abs(pred_valid - true_valid) / (np.abs(pred_valid) + np.abs(true_valid) + 1e-8)) * 100
        mape = np.mean(np.abs((true_valid - pred_valid) / (true_valid + 1e-8))) * 100
    else:
        mape = np.nan
        smape = np.nan

    print(f"\n📊 PI-GAT 重构性能 (掩码区域，共 {len(true_masked)} 个数据点):")
    print(f"   传统统计指标:")
    print(f"      MSE  : {mse:.6f}")
    print(f"      RMSE : {rmse:.6f}")
    print(f"      MAE  : {mae:.6f}")
    if not np.isnan(mape):
        print(f"      MAPE : {mape:.4f}%")
        print(f"      SMAPE: {smape:.4f}% (对称MAPE，更稳定)")
    else:
        print(f"      MAPE : N/A (过多零值)")
        print(f"      SMAPE: {smape:.4f}% (对称MAPE)")
    
    # [新增] 计算未掩码区域的误差（应该接近0）
    unmask_flat = ~mask_flat
    true_unmasked = true_2d[unmask_flat]
    pred_unmasked = pred_2d[unmask_flat]
    mse_unmasked = mean_squared_error(true_unmasked, pred_unmasked)
    mae_unmasked = mean_absolute_error(true_unmasked, pred_unmasked)
    print(f"\n   未掩码区域误差 (应接近0，检验恒等映射能力):")
    print(f"      MSE  : {mse_unmasked:.6f}")
    print(f"      MAE  : {mae_unmasked:.6f}")

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
        method_name='PI-GAT',
        feature_idx=0,
        plot_len=57
    )

    # 单样本全特征对比图
    plot_single_sample_comparison(
        pred_matrix=pred_2d[sample_idx:sample_idx + 1],
        true_matrix=true_2d[sample_idx:sample_idx + 1],
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
    main_train()  # 先训练
    main_test()  # 再测试
