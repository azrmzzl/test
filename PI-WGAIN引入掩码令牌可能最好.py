import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from visualization_utils import plot_single_sample_comparison, plot_time_series_comparison, set_random_seed
import pandas as pd
import time

# 设置随机种子

# ==========================================
# 【数据路径配置区】- 在这里修改数据文件路径
# ==========================================

# --- 预训练阶段（源域）数据配置 ---
SOURCE_DATA_DIR = 'dataset_split'                              # 源域数据文件夹
SOURCE_SAMPLES_FILE = 'Samples_x_y_normal_operation_all.mat'   # 源域样本文件名
SOURCE_TOPOLOGY_FILE = 'Topology_Branch_正常运行.mat'                   # 源域拓扑文件名

# --- 微调阶段（目标域）数据配置 ---
TARGET_DATA_DIR = 'dataset_split_target'                       # 目标域数据文件夹
TARGET_SAMPLES_FILE = 'Samples_x_y_normal_operation_all.mat'   # 目标域样本文件名
TARGET_TOPOLOGY_FILE = 'Topology_Branch_正常运行.mat'                   # 目标域拓扑文件名
TARGET_TRAIN_SIZE = 200                                        # 目标域用于微调的样本数量

# ==========================================


class GAIN_Generator(nn.Module):
    """
    【创新点1】融合全局自注意力机制的生成器
    不再将全网数据简单粗暴地全连接，而是将每个量测及其掩码视为一个独立的Token，
    通过 Multi-Head Self-Attention 动态挖掘量测之间的隐式电气关联。
    """

    def __init__(self, input_dim, embed_dim=16, num_heads=4):
        super(GAIN_Generator, self).__init__()
        self.input_dim = input_dim

        # 【核心修改】定义可学习的掩码令牌 (Learnable Mask Token)
        # 初始化为0，形状为 (1, input_dim)，允许模型为每个物理量测学习专属的最优占位特征
        self.mask_token = nn.Parameter(torch.zeros(1, input_dim))

        # 1. 特征嵌入层：将每个量测的 (数值, 掩码状态) 映射为高维特征向量
        self.feature_embedding = nn.Linear(2, embed_dim)

        # 2. 全局多头自注意力层：挖掘拓扑变化下的隐式电气关联
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # 3. 展平后的全连接重构网络
        flat_dim = input_dim * embed_dim
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # 归一化在 [0, 1]，保留 Sigmoid
        )

    def forward(self, x_raw, mask):
        B, D = x_raw.shape

        # 【核心修改】使用可学习的令牌替换受损量测，而不是外部传入的随机噪声
        # 掩码中 1 表示正常保留，0 表示受损替换为 mask_token
        x_corrupted = x_raw * mask + self.mask_token * (1 - mask)

        # 步骤 A：组合特征与掩码，shape变为 (B, D, 2)
        inputs = torch.stack([x_corrupted, mask], dim=-1)

        # 步骤 B：独立特征嵌入，shape变为 (B, D, embed_dim)
        emb = self.feature_embedding(inputs)

        # 步骤 C：全局自注意力交互
        attn_out, attn_weights = self.self_attention(emb, emb, emb)

        # 步骤 D：残差连接与展平
        out_features = (emb + attn_out).reshape(B, -1)

        # 步骤 E：非线性映射重构缺失数据
        recovered_x = self.fc(out_features)

        return recovered_x


class WGAIN_Critic(nn.Module):
    """
    【创新点】WGAN-GP 的评估器 (取代原 Discriminator)
    不再使用 Sigmoid 输出概率，而是输出无界的实数 (Wasserstein Score)
    """

    def __init__(self, input_dim):
        super(WGAIN_Critic, self).__init__()
        # 输入: Imputed Data + Hint Vector = input_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)  # 注意：这里去掉了原有的 Sigmoid
        )

    def forward(self, x, hint):
        inputs = torch.cat([x, hint], dim=1)
        return self.net(inputs)


# ==========================================
# 2. 物理一致性损失函数 (可导) & 梯度惩罚
# ==========================================
def differentiable_physics_loss(x_imputed, A_inc_tensor, scaler_min, scaler_scale):
    """
    【创新点】在计算图中对网络输出进行反归一化，并计算可导的物理一致性损失
    """
    # 1. 动态可导反归一化: x = (x_scaled - min) / scale
    x_unscaled = (x_imputed - scaler_min) / scaler_scale

    # 2. 提取物理量 (基于你的427维数据结构)
    if x_unscaled.shape[1] >= 331:
        bus_p = x_unscaled[:, 57:114]  # 注入有功
        bus_q = x_unscaled[:, 114:171]  # 注入无功
        branch_p = x_unscaled[:, 171:251]  # 有功潮流
        branch_q = x_unscaled[:, 251:331]  # 无功潮流

        # 3. 计算 KCL 残差：P_bus_calc = A_inc @ P_branch
        agg_p = torch.matmul(branch_p, A_inc_tensor.T)
        agg_q = torch.matmul(branch_q, A_inc_tensor.T)

        # 4. 计算 MSE 损失
        loss_p = torch.mean((bus_p - agg_p) ** 2)
        loss_q = torch.mean((bus_q - agg_q) ** 2)

        return loss_p + loss_q
    else:
        return torch.tensor(0.0, device=x_imputed.device)


def calc_gradient_penalty(netC, real_data, fake_data, hint, device):
    """
    【创新点】计算 WGAN-GP 的梯度惩罚项，强制满足 1-Lipschitz 约束
    """
    alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand_as(real_data)

    # 构建真实数据与生成数据的随机插值
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)

    # 评估器对插值数据打分
    disc_interpolates = netC(interpolates, hint)

    # 计算相对于插值数据的梯度
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates, device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    # 惩罚梯度范数偏离 1 的部分
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ==========================================
# 2.5 [新增] 自适应多任务损失权重模块
# ==========================================
class AdaptiveLossWeight(nn.Module):
    """
    【创新点2】自适应多任务损失模块 (基于同方差不确定性)
    自动平衡数据重构损失(MSE)与物理一致性损失(Phy)，解决固定超参数导致的梯度冲突问题。
    """

    def __init__(self, num_tasks=2):
        super(AdaptiveLossWeight, self).__init__()
        # 使用 log(sigma^2) 来保证数值稳定性，避免除以0或产生负数权重
        # 初始化为0，意味着初始权重 precision = exp(0) = 1
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        losses: 包含多个任务损失的列表，例如 [loss_mse, loss_phy]
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            # 权重精度: precision = 1 / sigma^2 = exp(-log(sigma^2))
            precision = torch.exp(-self.log_vars[i])
            # 公式: (1 / 2*sigma^2) * Loss + log(sigma)
            # 等价于: 0.5 * exp(-s) * Loss + 0.5 * s
            total_loss += 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return total_loss


# ==========================================
# 3. 数据加载 (展平 + 归一化) - 保持不变
# ==========================================
def load_and_normalize_data(data_dir=None, samples_file=None):
    """
    加载并归一化数据
    
    参数:
        data_dir: 数据文件夹路径，如果为None则使用全局配置
        samples_file: 样本文件名，如果为None则使用全局配置
    """
    if data_dir is None:
        data_dir = SOURCE_DATA_DIR
    if samples_file is None:
        samples_file = SOURCE_SAMPLES_FILE
    
    print(f">>> 正在加载 IEEE 57 数据 (用于 PG-WGAIN)...")
    print(f"    数据文件夹: {data_dir}")
    print(f"    样本文件: {samples_file}")
    
    try:
        mat_path = os.path.join(data_dir, samples_file)
        if not os.path.exists(mat_path):
            print(f"    ⚠️ 文件不存在: {mat_path}，使用随机数据替代")
            raw_data = np.random.rand(2000, 137 * 4)
        else:
            samples = sio.loadmat(mat_path)['Samples']
            if samples.ndim == 3:
                N, Nodes, Feats = samples.shape
                raw_data = samples.reshape(N, Nodes * Feats)
            else:
                raw_data = samples
            print(f"    ✅ 成功加载数据，维度: {raw_data.shape}")
    except Exception as e:
        print(f"    ❌ 加载失败: {e}，使用随机数据替代")
        raw_data = np.random.rand(2000, 137 * 4)

    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(raw_data)
    return data_norm.astype(np.float32), scaler


def load_grid_topology(data_dir=None, topology_file=None):
    """
    加载电网拓扑结构
    
    参数:
        data_dir: 数据文件夹路径，如果为None则使用全局配置
        topology_file: 拓扑文件名，如果为None则使用全局配置
    """
    if data_dir is None:
        data_dir = SOURCE_DATA_DIR
    if topology_file is None:
        topology_file = SOURCE_TOPOLOGY_FILE
    
    try:
        topo_path = os.path.join(data_dir, topology_file)
        print(f">>> 加载拓扑文件: {topo_path}")
        
        branch_data = sio.loadmat(topo_path)['branch_data']
        num_buses = 57
        num_lines = branch_data.shape[0]
        A_inc = np.zeros((num_buses, num_lines), dtype=np.float32)
        
        for l_idx in range(num_lines):
            # 【关键改进】检查支路状态字段，只包含运行中的支路
            status = branch_data[l_idx, 2] if branch_data.shape[1] > 2 else 1.0
            if status == 1.0:
                b_from = int(branch_data[l_idx, 0]) - 1
                b_to = int(branch_data[l_idx, 1]) - 1
                A_inc[b_from, l_idx] = 1.0
                A_inc[b_to, l_idx] = -1.0
        
        print(f"    ✅ 拓扑加载成功: {num_buses} 个节点, {num_lines} 条支路")
        return A_inc, num_buses, num_lines
    except Exception as e:
        print(f"    ❌ 拓扑加载失败: {e}，使用随机拓扑替代")
        return np.random.randn(57, 80).astype(np.float32), 57, 80


def calculate_pid_metric(reconstructed_data, A_inc, num_buses, num_lines):
    if reconstructed_data.ndim == 1:
        reconstructed_data = reconstructed_data.reshape(1, -1)

    N_samples = reconstructed_data.shape[0]
    total_features = reconstructed_data.shape[1]
    pid_values = []

    for i in range(N_samples):
        sample = reconstructed_data[i]
        if total_features >= 331:
            bus_p = sample[57:114]
            bus_q = sample[114:171]
            branch_p = sample[171:251]
            branch_q = sample[251:331]

            p_residual = bus_p - A_inc @ branch_p
            q_residual = bus_q - A_inc @ branch_q

            p_imbalance = np.sum(np.abs(p_residual))
            q_imbalance = np.sum(np.abs(q_residual))
            pid = (p_imbalance + q_imbalance) / num_buses
        else:
            return np.nan
        pid_values.append(pid)
    return np.mean(pid_values)


def sample_hint(batch_mask, hint_rate=0.9):
    B, D = batch_mask.shape
    hint_mask = (torch.rand(B, D).to(batch_mask.device) < hint_rate).float()
    hint = batch_mask * hint_mask + 0.5 * (1 - hint_mask)
    return hint


# ==========================================
# 4. 训练主程序 (引入 WGAN-GP 机制)
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 参数
    # [新增] 渐进式预热策略参数 (公式 3-23, 3-24)
    ALPHA_MAX = 10.0  # MSE损失的最大权重
    BETA_MAX = 1.0  # 物理损失的最大权重
    K = 0.05  # Sigmoid陡峭系数 (控制过渡速度)
    T0 = 200  # Sigmoid中心点 (在第200个epoch达到50%过渡)
    PATIENCE = 400  # [新增] 早停耐心值：允许连续多少个epoch没有改善
    # 参数 (注意：这里移除了固定的 ALPHA 和 BETA，交由网络自适应学习)
    BATCH_SIZE = 128
    EPOCHS = 500
    LR_G = 0.001
    LR_D = 0.001
    N_CRITIC = 3
    LAMBDA_GP = 10
    MASK_RATE = 0.3

    # 1. 准备数据
    data_all, scaler = load_and_normalize_data(
        data_dir=SOURCE_DATA_DIR, 
        samples_file=SOURCE_SAMPLES_FILE
    )
    input_dim = data_all.shape[1]
    X_tensor = torch.FloatTensor(data_all).to(device)

    # 提取 Scaler 参数放入张量，以便后续做可导反归一化计算
    scaler_min = torch.tensor(scaler.min_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale[scaler_scale == 0] = 1.0  # 防止除0异常

    # 加载拓扑矩阵放入张量
    A_inc_np, _, _ = load_grid_topology(
        data_dir=SOURCE_DATA_DIR, 
        topology_file=SOURCE_TOPOLOGY_FILE
    )
    A_inc_tensor = torch.tensor(A_inc_np, dtype=torch.float32, device=device)

    # 划分
    first_sample = X_tensor[0:1]
    remaining_data = X_tensor[1:]

    train_size = int(len(remaining_data) * 0.9)
    val_size = len(remaining_data) - train_size

    train_dataset = remaining_data[:train_size]
    val_dataset = remaining_data[train_size:]
    test_dataset = torch.cat([first_sample, val_dataset], dim=0)

    train_loader = DataLoader(TensorDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_dataset), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_dataset), batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型 (采用 WGAN 推荐的 betas 参数)
    netG = GAIN_Generator(input_dim).to(device)
    netC = WGAIN_Critic(input_dim).to(device)

    # # 【修改点 A】初始化自适应权重模块
    # adaptive_weight = AdaptiveLossWeight(num_tasks=2).to(device)
    # # 【修改点 B】生成器的优化器需要同时更新 netG 的参数 和 adaptive_weight 的参数
    # optG = optim.Adam(list(netG.parameters()) + list(adaptive_weight.parameters()), lr=LR_G, betas=(0.5, 0.9))
    # optC = optim.Adam(netC.parameters(), lr=LR_D, betas=(0.5, 0.9))

    optG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.9))
    optC = optim.Adam(netC.parameters(), lr=LR_D, betas=(0.5, 0.9))

    # [新增] 打印渐进式策略说明
    print(f"\n>>> 开始训练 PG-WGAIN (Sigmoid渐进式预热 + 全局自注意力 + WGAN-GP + 早停)...")
    print(f"    渐进式策略参数: ALPHA_MAX={ALPHA_MAX}, BETA_MAX={BETA_MAX}, K={K}, T0={T0}")
    print(f"    公式: α2(t) = α2_max / (1 + exp(-k*(t-t0))), α1(t) = 1 - α2(t)")

    # [新增] 早停相关变量
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_netG_state = None
    best_adaptive_weight_state = None

    for epoch in range(EPOCHS):
        netG.train()
        netC.train()

        # [新增] 计算当前epoch的动态权重 (公式 3-23, 3-24) - 反向策略：先物理后MSE
        t = epoch + 1  # 当前epoch (从1开始)
        # 反向Sigmoid：物理损失权重从高到低，MSE权重从低到高
        alpha2_t = BETA_MAX / (1 + np.exp(K * (t - T0)))  # 物理损失权重（反向：exp符号变正）
        alpha1_t = 1.0 - alpha2_t / BETA_MAX  # MSE权重归一化系数
        current_alpha = ALPHA_MAX * alpha1_t  # 实际MSE权重
        current_beta = alpha2_t  # 实际物理权重

        loss_c_log, loss_g_log, loss_phy_log = 0, 0, 0
        batches_processed = 0

        # 手动控制 DataLoader，以实现 N_CRITIC 步的错频训练
        train_iter = iter(train_loader)
        while True:
            try:
                # ----------------------------------------
                # 步骤一：训练 Critic (评估器) N_CRITIC 次
                # ----------------------------------------
                for _ in range(N_CRITIC):
                    batch_x = next(train_iter)[0]
                    B, Dim = batch_x.shape

                    mask = (torch.rand(B, Dim, device=device) > MASK_RATE).float()
                    noise = torch.rand(B, Dim, device=device)
                    x_corrupted = batch_x * mask + noise * (1 - mask)
                    hint = sample_hint(mask)

                    # G前向传播 (不计算梯度)
                    with torch.no_grad():
                        x_generated = netG(x_corrupted, mask)
                        x_imputed = batch_x * mask + x_generated * (1 - mask)

                    # 评估器打分
                    c_real = netC(batch_x, hint)
                    c_fake = netC(x_imputed.detach(), hint)

                    # Wasserstein 距离损失
                    loss_c_wasserstein = torch.mean(c_fake) - torch.mean(c_real)

                    # 计算梯度惩罚
                    gp = calc_gradient_penalty(netC, batch_x, x_imputed.detach(), hint, device)

                    # 评估器总损失
                    loss_c = loss_c_wasserstein + LAMBDA_GP * gp

                    optC.zero_grad()
                    loss_c.backward()
                    optC.step()

                    loss_c_log += loss_c.item()

                # ----------------------------------------
                # 步骤二：训练 Generator (生成器) 1 次
                # ----------------------------------------
                batch_x = next(train_iter)[0]
                B, Dim = batch_x.shape

                mask = (torch.rand(B, Dim, device=device) > MASK_RATE).float()
                hint = sample_hint(mask)

                x_generated = netG(batch_x, mask)  # 直接传入原始 batch_x
                x_imputed = batch_x * mask + x_generated * (1 - mask)

                # 生成器的对抗损失: 骗过 Critic (让 Critic 对生成数据打低分，由于前面有负号，即极小化)
                c_fake = netC(x_imputed, hint)
                loss_g_adv = -torch.mean(c_fake)

                # 重构损失 (MSE) - 观测部分必须重建得准
                loss_g_mse = torch.mean(((1 - mask) * batch_x - (1 - mask) * x_generated) ** 2) / (
                            torch.mean(1 - mask) + 1e-8)

                # 物理一致性损失 (利用 PyTorch 张量直接计算)
                loss_g_phy = differentiable_physics_loss(x_imputed, A_inc_tensor, scaler_min, scaler_scale)

                # [修改] 使用渐进式动态权重 (公式 3-23, 3-24)
                loss_g = loss_g_adv + current_alpha * loss_g_mse + current_beta * loss_g_phy

                # # 【修改点 C】利用自适应模块动态加权 (替代原有的 ALPHA*MSE + BETA*Phy)
                # # 将需自适应平衡的损失打包传入
                # loss_g_task = adaptive_weight([loss_g_mse, loss_g_phy])
                # # 最终生成器总损失 = 对抗基准损失 + 动态加权的多任务损失
                # loss_g = loss_g_adv + loss_g_task

                optG.zero_grad()
                loss_g.backward()
                optG.step()

                loss_g_log += loss_g.item()
                loss_phy_log += loss_g_phy.item()
                batches_processed += 1

            except StopIteration:
                break  # 该 Epoch 所有的 Batch 迭代完毕

        if (epoch + 1) % 10 == 0:
            avg_c = loss_c_log / (batches_processed * N_CRITIC)
            avg_g = loss_g_log / batches_processed
            avg_phy = loss_phy_log / batches_processed
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | C Loss: {avg_c:.4f} | G Loss: {avg_g:.4f} | α(MSE): {current_alpha:.2f} | β(Phy): {current_beta:.2f}")

        # [新增] 每个epoch结束后进行验证集评估
        netG.eval()
        val_loss_total = 0
        val_batches = 0

        with torch.no_grad():
            for val_batch_x, in val_loader:
                B, Dim = val_batch_x.shape
                mask = (torch.rand(B, Dim, device=device) > MASK_RATE).float()
                hint = sample_hint(mask)

                x_generated = netG(val_batch_x, mask)  # 注意这里传入的是 val_batch_x
                x_imputed = val_batch_x * mask + x_generated * (1 - mask)

                # [修改] 验证损失看全部损失（MSE + 物理损失）
                val_mse = torch.mean(((1 - mask) * val_batch_x - (1 - mask) * x_generated) ** 2) / (
                            torch.mean(1 - mask) + 1e-8)
                val_phy = differentiable_physics_loss(x_imputed, A_inc_tensor, scaler_min, scaler_scale)
                val_loss = current_alpha * val_mse + current_beta * val_phy

                val_loss_total += val_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_total / val_batches if val_batches > 0 else float('inf')
        netG.train()

        # [新增] 早停逻辑
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最优模型状态
            best_netG_state = netG.state_dict().copy()
            # best_adaptive_weight_state = adaptive_weight.state_dict().copy()
            if (epoch + 1) % 10 == 0:
                print(f"   ✅ 验证损失改善: {avg_val_loss:.6f} (最优epoch: {best_epoch})")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f"   ⏳ 验证损失未改善: {avg_val_loss:.6f} (patience: {patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                print(f"\n🛑 早停触发！验证损失已连续 {PATIENCE} 个epoch未改善")
                print(f"   最优epoch: {best_epoch}, 最优验证损失: {best_val_loss:.6f}")
                break

    # [新增] 恢复最优模型
    if best_netG_state is not None:
        netG.load_state_dict(best_netG_state)
        # adaptive_weight.load_state_dict(best_adaptive_weight_state)
        print(f"\n✨ 已恢复最优模型 (epoch {best_epoch})")

    # 保存最优模型到文件
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = f'pg_wgain_best_{timestamp}.pth'
    torch.save({
        'netG_state_dict': best_netG_state,
        'netC_state_dict': netC.state_dict(),  # [新增] 保存 Critic 权重供微调使用
        'adaptive_weight_state_dict': best_adaptive_weight_state,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }, model_path)
    print(f"💾 最优模型已保存为 {model_path}")

    # 4. 可视化对比 (全维度评估)
    visualize_gain_result(netG, test_loader, scaler, device, MASK_RATE)

    return model_path  # [新增] 返回模型路径供微调函数调用


# ==========================================
# 4.5 [新增] 目标域变拓扑微调程序 (Frozen Fine-Tuning)
# ==========================================
def main_finetune(pretrained_model_path, target_data_dir=None, target_samples_file=None, target_topology_file=None):
    """
    目标域变拓扑场景冻结微调
    
    参数:
        pretrained_model_path: 预训练模型路径
        target_data_dir: 目标域数据文件夹，如果为None则使用全局配置
        target_samples_file: 目标域样本文件名，如果为None则使用全局配置
        target_topology_file: 目标域拓扑文件名，如果为None则使用全局配置
    """
    if target_data_dir is None:
        target_data_dir = TARGET_DATA_DIR
    if target_samples_file is None:
        target_samples_file = TARGET_SAMPLES_FILE
    if target_topology_file is None:
        target_topology_file = TARGET_TOPOLOGY_FILE
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==========================================")
    print(f"     阶段 2：目标域变拓扑场景冻结微调")
    print(f"==========================================")
    print(f"    目标域文件夹: {target_data_dir}")
    print(f"    目标域样本: {target_samples_file}")
    print(f"    目标域拓扑: {target_topology_file}")

    # 微调参数：学习率降低十倍，Epoch减少
    BATCH_SIZE = 32
    EPOCHS = 100
    LR_FINETUNE = 5e-4
    N_CRITIC = 3
    MASK_RATE = 0.3

    # 1. 加载目标域(新拓扑)的数据与新关联矩阵
    data_all, scaler = load_and_normalize_data(
        data_dir=target_data_dir, 
        samples_file=target_samples_file
    )
    input_dim = data_all.shape[1]
    X_tensor = torch.FloatTensor(data_all).to(device)

    scaler_min = torch.tensor(scaler.min_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale[scaler_scale == 0] = 1.0

    # [重点] 加载断线后的新 Topology_Branch.mat
    A_inc_np_target, num_buses, num_lines = load_grid_topology(
        data_dir=target_data_dir, 
        topology_file=target_topology_file
    )
    A_inc_tensor_target = torch.tensor(A_inc_np_target, dtype=torch.float32, device=device)

    # 目标域数量是源域样本数量的十分之一
    target_train_size = TARGET_TRAIN_SIZE  # 使用全局配置

    # 从目标域数据中截取前 target_train_size 个样本用于微调训练
    train_tensor = X_tensor[:target_train_size]

    # 目标域剩余的样本，全部留作测试集，用来验证微调后的泛化效果
    test_tensor = X_tensor[target_train_size:]

    # 如果测试集为空（说明目标域总数据刚好只有200个），则用全部数据测试
    if len(test_tensor) == 0:
        test_tensor = X_tensor

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=BATCH_SIZE, shuffle=False)
    # ---------------------------------------------------------

    # 2. 初始化网络并加载预训练权重
    netG = GAIN_Generator(input_dim).to(device)
    netC = WGAIN_Critic(input_dim).to(device)

    checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=True)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netC.load_state_dict(checkpoint['netC_state_dict'])
    print("✅ 已成功加载源域预训练模型权重。")

    # 3. 实施冻结 (Frozen)
    # 冻结特征嵌入层和自注意力层
    for param in netG.feature_embedding.parameters():
        param.requires_grad = False
    for param in netG.self_attention.parameters():
        param.requires_grad = False
    print("🔒 已冻结生成器底层的 [特征嵌入层] 和 [自注意力层]。")

    # 仅将未冻结的参数 (netG.fc) 传入优化器
    optG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=LR_FINETUNE, betas=(0.5, 0.9))
    optC = optim.Adam(netC.parameters(), lr=LR_FINETUNE, betas=(0.5, 0.9))

    # 微调阶段直接固定较高物理权重，快速对齐新拓扑
    current_alpha = 10.0
    current_beta = 1.0

    print(f">>> 开始微调 (Target Domain) ...")
    for epoch in range(EPOCHS):
        netG.train()
        netC.train()
        loss_c_log, loss_g_log = 0, 0
        batches_processed = 0
        train_iter = iter(train_loader)

        while True:
            try:
                for _ in range(N_CRITIC):
                    batch_x = next(train_iter)[0]
                    B, Dim = batch_x.shape
                    mask = (torch.rand(B, Dim, device=device) > MASK_RATE).float()
                    hint = sample_hint(mask)

                    with torch.no_grad():
                        x_generated = netG(batch_x, mask)  # 直接传入 batch_x
                        x_imputed = batch_x * mask + x_generated * (1 - mask)

                    c_real = netC(batch_x, hint)
                    c_fake = netC(x_imputed.detach(), hint)
                    loss_c = (torch.mean(c_fake) - torch.mean(c_real)) + 10 * calc_gradient_penalty(netC, batch_x,
                                                                                                    x_imputed.detach(),
                                                                                                    hint, device)

                    optC.zero_grad()
                    loss_c.backward()
                    optC.step()
                    loss_c_log += loss_c.item()

                batch_x = next(train_iter)[0]
                B, Dim = batch_x.shape
                mask = (torch.rand(B, Dim, device=device) > MASK_RATE).float()
                hint = sample_hint(mask)
                x_generated = netG(batch_x, mask)  # 直接传入 batch_x
                x_imputed = batch_x * mask + x_generated * (1 - mask)

                loss_g_adv = -torch.mean(netC(x_imputed, hint))
                loss_g_mse = torch.mean(((1 - mask) * batch_x - (1 - mask) * x_generated) ** 2) / (
                            torch.mean(1 - mask) + 1e-8)

                # [关键] 使用目标域的新拓扑 A_inc_tensor_target 计算物理损失
                loss_g_phy = differentiable_physics_loss(x_imputed, A_inc_tensor_target, scaler_min, scaler_scale)
                loss_g = loss_g_adv + current_alpha * loss_g_mse + current_beta * loss_g_phy

                optG.zero_grad()
                loss_g.backward()
                optG.step()

                loss_g_log += loss_g.item()
                batches_processed += 1
            except StopIteration:
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"Finetune Epoch {epoch + 1}/{EPOCHS} | C Loss: {loss_c_log / (batches_processed * N_CRITIC):.4f} | G Loss: {loss_g_log / batches_processed:.4f}")

    print("\n✨ 目标域微调完成！开始测试...")
    # [修改] 调用可视化代码时，必须传入新的拓扑，否则 PID 计算会出错
    visualize_gain_result_target(netG, test_loader, scaler, device, MASK_RATE, A_inc_np_target, num_buses, num_lines)


# 为了微调测试专门弄一个精简的可视化接口，避免改动原有的可视化函数
def visualize_gain_result_target(netG, test_loader, scaler, device, mask_rate, A_inc, num_buses, num_lines):
    netG.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, in test_loader:
            B, Dim = batch_x.shape
            mask = (torch.rand(B, Dim).to(device) > mask_rate).float()

            x_generated = netG(batch_x, mask)  # 直接传入 batch_x
            # 填补：保留已知，仅替换掩码部分
            x_imputed = batch_x * mask + x_generated * (1 - mask)

            all_preds.append(x_imputed.cpu().numpy())
            all_trues.append(batch_x.cpu().numpy())

    pred_mat = scaler.inverse_transform(np.concatenate(all_preds, axis=0))
    true_mat = scaler.inverse_transform(np.concatenate(all_trues, axis=0))

    # ----------------------------------------------------
    # 补全统计学评价指标 (MSE, RMSE, MAE, MAPE)
    # ----------------------------------------------------
    # 提取被掩码（受损）部分的数据进行误差计算
    mask_flat = np.random.rand(*true_mat.shape) > mask_rate
    true_values = true_mat[mask_flat]
    recovered_values = pred_mat[mask_flat]

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

    # 计算物理一致性指标 PID
    pid_true = calculate_pid_metric(true_mat, A_inc, num_buses, num_lines)
    pid_recovered = calculate_pid_metric(pred_mat, A_inc, num_buses, num_lines)
    pid_improvement = (pid_true - pid_recovered) / pid_true * 100 if pid_true > 0 else 0

    print(f"\n📊 变拓扑微调重构性能 (Target Domain, 共 {len(true_values)} 个数据点):")
    print(f"   传统统计指标:")
    print(f"      MSE  : {rmse ** 2:.6f}")
    print(f"      RMSE : {rmse:.6f}")
    print(f"      MAE  : {mae:.6f}")
    print(f"      MAPE : {mape:.4f}%")
    print(f"\n   物理一致性指标 (PID):")
    print(f"      真实数据 PID: {pid_true:.6f}")
    print(f"      重构数据 PID: {pid_recovered:.6f}")
    print(f"      PID 改善率  : {pid_improvement:.2f}%")

    # 继续调用原有的画图与保存代码
    plot_single_sample_comparison(pred_matrix=pred_mat, true_matrix=true_mat, sample_idx=0,
                                  method_name='PG-WGAIN-Target')
    save_single_sample_to_excel(pred_matrix=pred_mat, true_matrix=true_mat, sample_idx=0, method_name='PG_WGAIN_Target')

# ==========================================
# 5. 可视化函数 (全维度评估 + 反归一化) - 保持不变
# ==========================================
def visualize_gain_result(netG, test_loader, scaler, device, mask_rate=0.2):
    print(f"\n>>> 正在生成 PG-WGAIN 对比图 (全维度评估)...")
    netG.eval()

    set_random_seed(2025)

    A_inc, num_buses, num_lines = load_grid_topology()
    A_inc = A_inc.astype(np.float32)

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_x, in test_loader:
            B, Dim = batch_x.shape
            mask = (torch.rand(B, Dim).to(device) > mask_rate).float()

            x_generated = netG(batch_x, mask)  # 直接传入 batch_x
            x_imputed = batch_x * mask + x_generated * (1 - mask)

            all_preds.append(x_imputed.cpu().numpy())
            all_trues.append(batch_x.cpu().numpy())

    pred_mat = scaler.inverse_transform(np.concatenate(all_preds, axis=0))
    true_mat = scaler.inverse_transform(np.concatenate(all_trues, axis=0))

    print(f"\n测试集总样本数：{pred_mat.shape[0]}")
    print(f"特征维度：{pred_mat.shape[1]}")
    print(f"ℹ️  注意：测试集包含第 1 个样本（索引 0），该样本未参与训练")

    mask_flat = np.random.rand(*true_mat.shape) > mask_rate
    true_values = true_mat[mask_flat]
    recovered_values = pred_mat[mask_flat]

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

    pid_true = calculate_pid_metric(true_mat, A_inc, num_buses, num_lines)
    pid_recovered = calculate_pid_metric(pred_mat, A_inc, num_buses, num_lines)
    pid_improvement = (pid_true - pid_recovered) / pid_true * 100 if pid_true > 0 else 0

    print(f"\n📊 PG-WGAIN 重构性能 (全维度，共 {len(true_values)} 个数据点):")
    print(f"   传统统计指标:")
    print(f"      MSE  : {rmse ** 2:.6f}")
    print(f"      RMSE : {rmse:.6f}")
    print(f"      MAE  : {mae:.6f}")
    print(f"      MAPE : {mape:.4f}%")
    print(f"\n   物理一致性指标 (PID):")
    print(f"      真实数据 PID:    {pid_true:.6f}")
    print(f"      重构数据 PID:    {pid_recovered:.6f}")
    print(f"      PID 变化率：     {pid_improvement:.2f}%")

    plot_time_series_comparison(y_true=true_mat, y_pred=pred_mat, method_name='PG-WGAIN', feature_idx=0, plot_len=200)
    plot_single_sample_comparison(pred_matrix=pred_mat, true_matrix=true_mat, sample_idx=0, method_name='PG-WGAIN')
    save_single_sample_to_excel(pred_matrix=pred_mat, true_matrix=true_mat, sample_idx=0, method_name='PG-WGAIN')


def save_single_sample_to_excel(pred_matrix, true_matrix, sample_idx=0, method_name='PG_WGAIN'):
    print(f"\n>>> 正在保存 {method_name} 单样本 (测试集索引={sample_idx}) 详细数据到 Excel...")
    y_true_sample = true_matrix[sample_idx]
    y_pred_sample = pred_matrix[sample_idx]
    n_features = len(y_true_sample)

    df = pd.DataFrame({
        'Feature_Index': range(n_features),
        'Ground_Truth': y_true_sample,
        'Recovered': y_pred_sample,
        'Absolute_Error': np.abs(y_true_sample - y_pred_sample),
        'Relative_Error_Percent': np.abs((y_true_sample - y_pred_sample) / (y_true_sample + 1e-8)) * 100
    })

    stats = pd.DataFrame({
        'Feature_Index': ['MEAN', 'STD', 'MAX', 'MIN'],
        'Ground_Truth': [df['Ground_Truth'].mean(), df['Ground_Truth'].std(), df['Ground_Truth'].max(),
                         df['Ground_Truth'].min()],
        'Recovered': [df['Recovered'].mean(), df['Recovered'].std(), df['Recovered'].max(), df['Recovered'].min()],
        'Absolute_Error': [df['Absolute_Error'].mean(), df['Absolute_Error'].std(), df['Absolute_Error'].max(),
                           df['Absolute_Error'].min()],
        'Relative_Error_Percent': [df['Relative_Error_Percent'].mean(), df['Relative_Error_Percent'].std(),
                                   df['Relative_Error_Percent'].max(), df['Relative_Error_Percent'].min()]
    })

    df_with_stats = pd.concat([df, stats], ignore_index=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'{method_name}_单样本数据_{sample_idx}_{timestamp}.xlsx'
    df_with_stats.to_excel(filename, index=False, sheet_name='单样本详细数据')
    print(f"   数据已保存至：{filename}")
    return filename


if __name__ == "__main__":
    print("="*60)
    print("📊 PG-WGAIN 训练流程")
    print("="*60)
    print(f"\n【预训练配置】")
    print(f"  数据文件夹: {SOURCE_DATA_DIR}")
    print(f"  样本文件: {SOURCE_SAMPLES_FILE}")
    print(f"  拓扑文件: {SOURCE_TOPOLOGY_FILE}")
    print(f"\n【微调配置】")
    print(f"  数据文件夹: {TARGET_DATA_DIR}")
    print(f"  样本文件: {TARGET_SAMPLES_FILE}")
    print(f"  拓扑文件: {TARGET_TOPOLOGY_FILE}")
    print(f"  训练样本数: {TARGET_TRAIN_SIZE}")
    print("="*60)
    
    # 1. 基准场景预训练 (将返回最优模型的文件路径)
    best_model_path = main()

    # 2. 目标域变拓扑场景微调
    # 注意：运行前请确保存在 dataset_split_target 文件夹，且里面有断线后的新拓扑数据
    if os.path.exists(TARGET_DATA_DIR):
        main_finetune(
            pretrained_model_path=best_model_path, 
            target_data_dir=TARGET_DATA_DIR,
            target_samples_file=TARGET_SAMPLES_FILE,
            target_topology_file=TARGET_TOPOLOGY_FILE
        )
    else:
        print(f"\n⚠️ 未检测到 '{TARGET_DATA_DIR}' 文件夹，跳过变拓扑微调阶段。")