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

            adj[b_from, l_node] = 1;
            adj[l_node, b_from] = 1
            adj[b_to, l_node] = 1;
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
    def __init__(self, in_channels=4, hidden_channels=256, out_channels=4, num_heads=8):
        super(PI_GraphMAE, self).__init__()
        # Mask Token 初始值设为 0 比较合理，因为物理上缺失通常意味着信息归零
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
    BATCH_SIZE = 32
    EPOCHS = 1000
    PATIENCE = 50  # 早停耐心值：允许连续多少个 epoch 没有改善
    # ==========================================

    # 划分训练集和验证集
    num_samples = x_all.shape[0]
    num_train = int(num_samples * 0.9)  # 90% 用于训练
    num_val = num_samples - num_train  # 10% 用于验证

    train_loader = torch.utils.data.DataLoader(x_all[:num_train], batch_size=BATCH_SIZE, shuffle=IS_SHUFFLE)
    val_loader = torch.utils.data.DataLoader(x_all[num_train:], batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = PI_GraphMAE(in_channels=4, hidden_channels=64, out_channels=4, num_heads=4).to(device)

    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"\n>>> 开始训练 (无归一化模式)...")
    model.train()

    fixed_mask = None
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
                    fixed_mask = (torch.rand(1, N, 1, device=device) < 0.2).float()
                mask = fixed_mask.expand(B, N, 1)  # ✅ 每次都 expand 到当前 B
            else:
                # 同样为整个 batch 生成统一掩码
                mask = (torch.rand(B, N, 1, device=device) < 0.2).float()



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

                # 增加 batch 计数器
            batch_counter += 1


            recon = model(batch_x, edge_index, mask)

            # Loss 计算
            loss_mse = F.mse_loss(recon * mask, batch_x * mask)
            loss_phy = physics_loss_ieee57_direct(recon, B_inc)
            loss = loss_mse + 0.1 * loss_phy

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_phy += loss_phy.item()

        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x in val_loader:
                B, N, Fdim = batch_x.shape
                # 验证集
                if FIXED_MASK_MODE:
                    mask = fixed_mask.expand(B, N, 1) if fixed_mask is not None else \
                        (torch.rand(1, N, 1, device=device) < 0.2).float().expand(B, N, 1)
                else:
                    mask = (torch.rand(B, N, 1, device=device) < 0.2).float()

                recon = model(batch_x, edge_index, mask)
                loss_mse = F.mse_loss(recon * mask, batch_x * mask)
                loss_phy = physics_loss_ieee57_direct(recon, B_inc)
                loss = loss_mse + 0.1 * loss_phy
                val_loss += loss.item()
        val_loss /= len(val_loader)
        model.train()

        # 打印日志
        avg_loss = total_loss / len(train_loader)
        avg_mse = total_mse / len(train_loader)
        avg_phy = total_phy / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1:04d} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"MSE: {avg_mse:.6f} | Phy: {avg_phy:.6f} | LR: {lr:.6f}"
        )

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



if __name__ == "__main__":
    main_train()
