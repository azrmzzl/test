import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import scipy.io as sio
import os
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from visualization_utils import plot_single_sample_comparison, plot_time_series_comparison, set_random_seed
import pandas as pd
import time
import copy

# 设置随机种子
torch.manual_seed(2025)
np.random.seed(2025)

# ==========================================
# 0. 数据路径配置
# ==========================================
# 训练阶段只放正常样本。文件夹内至少包含 Samples_*.mat，建议同时放 Topology_Branch.mat。
SOURCE_NORMAL_DIR = 'dataset_split_source_normal'

# 目标域少量正常样本，用于变拓扑冻结微调。没有该文件夹时会自动跳过微调。
TARGET_NORMAL_DIR = 'dataset_split_target_normal'

# 攻击测试阶段数据，由 segment_*.m 输出，包含 Samples、Clean_Samples、Labels_Meas 或 Recovery_Mask。
ATTACK_TEST_DIR = 'dataset_split_attack_test'

# 如果你的正常样本仍放在旧的 dataset_split 文件夹，可以把 SOURCE_NORMAL_DIR 改成 'dataset_split'。



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
# 3. 数据加载、文件匹配与归一化
# ==========================================
def _sorted_mat_files(data_dir, pattern):
    files = glob.glob(os.path.join(data_dir, pattern))
    return sorted(files, key=os.path.getmtime, reverse=True)


def _derive_tag_from_file(file_path, prefix):
    name = os.path.splitext(os.path.basename(file_path))[0]
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def _find_required_file(data_dir, prefix, tag=None):
    if tag is None:
        files = _sorted_mat_files(data_dir, f'{prefix}*.mat')
    else:
        files = [os.path.join(data_dir, f'{prefix}{tag}.mat')]
        files = [f for f in files if os.path.exists(f)]

    if not files:
        raise FileNotFoundError(f'在 {data_dir} 中未找到 {prefix}*.mat')
    return files[0]


def _find_optional_file(data_dir, prefix, tag=None):
    if tag is None:
        files = _sorted_mat_files(data_dir, f'{prefix}*.mat')
    else:
        files = [os.path.join(data_dir, f'{prefix}{tag}.mat')]
        files = [f for f in files if os.path.exists(f)]
    return files[0] if files else None


def _load_mat_variable(file_path, preferred_names):
    mat = sio.loadmat(file_path)
    for name in preferred_names:
        if name in mat:
            return mat[name]
    public_keys = [k for k in mat.keys() if not k.startswith('__')]
    raise KeyError(f'{file_path} 中未找到变量 {preferred_names}。当前可用变量: {public_keys}')


def _flatten_samples(samples):
    samples = np.asarray(samples)
    if samples.ndim == 3:
        n, nodes, feats = samples.shape
        samples = samples.reshape(n, nodes * feats)
    if samples.ndim != 2:
        raise ValueError(f'Samples 必须是二维或三维数组，当前维度为 {samples.shape}')
    return samples.astype(np.float32)


def load_samples_from_dir(data_dir, tag=None):
    sample_file = _find_required_file(data_dir, 'Samples_', tag)
    tag = _derive_tag_from_file(sample_file, 'Samples_')
    samples = _flatten_samples(_load_mat_variable(sample_file, ['Samples']))
    print(f'>>> 读取 Samples: {sample_file}，维度 {samples.shape}')
    return samples, tag, sample_file


def load_and_normalize_data(data_dir='dataset_split', scaler=None, fit_scaler=True):
    """
    训练或微调阶段只读取正常样本 Samples_*.mat。
    fit_scaler=True 时拟合新的 MinMaxScaler；fit_scaler=False 时复用外部 scaler。
    """
    raw_data, tag, sample_file = load_samples_from_dir(data_dir)

    if scaler is None:
        scaler = MinMaxScaler()
        data_norm = scaler.fit_transform(raw_data)
    else:
        if fit_scaler:
            data_norm = scaler.fit_transform(raw_data)
        else:
            data_norm = scaler.transform(raw_data)

    return data_norm.astype(np.float32), scaler, tag


def load_grid_topology(data_dir='dataset_split'):
    """
    读取 Topology_Branch.mat。
    branch_data 第1列为 from bus，第2列为 to bus，第3列可选，表示线路状态。
    如果没有第三列，默认全部线路连通。
    """
    topo_file = os.path.join(data_dir, 'Topology_Branch.mat')
    if not os.path.exists(topo_file):
        raise FileNotFoundError(f'未找到拓扑文件: {topo_file}')

    branch_data = sio.loadmat(topo_file)['branch_data']
    num_buses = 57
    num_lines = branch_data.shape[0]
    A_inc = np.zeros((num_buses, num_lines), dtype=np.float32)

    for l_idx in range(num_lines):
        status = branch_data[l_idx, 2] if branch_data.shape[1] > 2 else 1.0
        if status == 1.0:
            b_from = int(branch_data[l_idx, 0]) - 1
            b_to = int(branch_data[l_idx, 1]) - 1
            A_inc[b_from, l_idx] = 1.0
            A_inc[b_to, l_idx] = -1.0

    return A_inc, num_buses, num_lines


def _load_attack_test_data(data_dir):
    """
    从攻击测试文件夹读取同一个 tag 的数据，避免 Samples、Clean_Samples 和标签错配。
    优先读取 Recovery_Mask；没有时读取 Pred_Labels_Meas；仍没有时读取 Labels_Meas。
    """
    samples, tag, sample_file = load_samples_from_dir(data_dir)

    clean_file = _find_optional_file(data_dir, 'Clean_Samples_', tag)
    clean_samples = None
    if clean_file is not None:
        clean_samples = _flatten_samples(_load_mat_variable(clean_file, ['Clean_Samples']))
        if clean_samples.shape != samples.shape:
            raise ValueError(f'Clean_Samples 维度 {clean_samples.shape} 与 Samples 维度 {samples.shape} 不一致')

    recovery_mask_file = _find_optional_file(data_dir, 'Recovery_Mask_', tag)
    pred_label_file = _find_optional_file(data_dir, 'Pred_Labels_Meas_', tag)
    label_file = _find_optional_file(data_dir, 'Labels_Meas_', tag)

    label_mat = None
    if recovery_mask_file is not None:
        recovery_mask = _load_mat_variable(recovery_mask_file, ['Recovery_Mask', 'mask', 'Mask'])
        recovery_mask = np.asarray(recovery_mask, dtype=np.float32)
        mask_source = recovery_mask_file
        if label_file is not None:
            label_mat = np.asarray(_load_mat_variable(label_file, ['Labels_Meas']), dtype=np.float32)
        else:
            label_mat = 1.0 - recovery_mask
    elif pred_label_file is not None:
        label_mat = np.asarray(_load_mat_variable(pred_label_file, ['Pred_Labels_Meas', 'Labels_Meas', 'pred_labels']), dtype=np.float32)
        recovery_mask = 1.0 - label_mat
        mask_source = pred_label_file
    elif label_file is not None:
        label_mat = np.asarray(_load_mat_variable(label_file, ['Labels_Meas']), dtype=np.float32)
        recovery_mask = 1.0 - label_mat
        mask_source = label_file
    else:
        raise FileNotFoundError(f'在 {data_dir} 中未找到 Recovery_Mask_{tag}.mat、Pred_Labels_Meas_{tag}.mat 或 Labels_Meas_{tag}.mat')

    if recovery_mask.shape != samples.shape:
        raise ValueError(f'恢复掩码维度 {recovery_mask.shape} 与 Samples 维度 {samples.shape} 不一致')
    if label_mat is not None and label_mat.shape != samples.shape:
        raise ValueError(f'量测标签维度 {label_mat.shape} 与 Samples 维度 {samples.shape} 不一致')

    meta_file = _find_optional_file(data_dir, 'Sample_Meta_', tag)
    sample_meta = None
    if meta_file is not None:
        sample_meta = _load_mat_variable(meta_file, ['Sample_Meta'])

    print(f'>>> 攻击测试 tag: {tag}')
    print(f'>>> 恢复掩码来源: {mask_source}')
    if clean_file is not None:
        print(f'>>> 干净样本来源: {clean_file}')

    return {
        'tag': tag,
        'samples': samples,
        'clean_samples': clean_samples,
        'label_mat': label_mat,
        'recovery_mask': recovery_mask.astype(np.float32),
        'sample_meta': sample_meta,
        'sample_file': sample_file,
        'clean_file': clean_file,
        'mask_source': mask_source,
    }


def masked_regression_metrics(y_true, y_pred, mask_bool):
    if mask_bool is None or np.sum(mask_bool) == 0:
        return None
    true_values = y_true[mask_bool]
    pred_values = y_pred[mask_bool]
    rmse = np.sqrt(mean_squared_error(true_values, pred_values))
    mae = mean_absolute_error(true_values, pred_values)
    eps = 1e-8
    valid = np.abs(true_values) > eps
    mape = mean_absolute_percentage_error(true_values[valid], pred_values[valid]) * 100 if np.any(valid) else np.nan
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


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
def main(data_dir=SOURCE_NORMAL_DIR):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 参数
    ALPHA_MAX = 10.0
    BETA_MAX = 1.0
    K = 0.05
    T0 = 200
    PATIENCE = 400
    BATCH_SIZE = 128
    EPOCHS = 500
    LR_G = 0.001
    LR_D = 0.001
    N_CRITIC = 3
    LAMBDA_GP = 10
    MASK_RATE = 0.3

    # 1. 准备数据
    data_all, scaler, source_tag = load_and_normalize_data(data_dir, scaler=None, fit_scaler=True)
    input_dim = data_all.shape[1]
    X_tensor = torch.FloatTensor(data_all).to(device)

    scaler_min = torch.tensor(scaler.min_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale[scaler_scale == 0] = 1.0

    A_inc_np, _, _ = load_grid_topology(data_dir)
    A_inc_tensor = torch.tensor(A_inc_np, dtype=torch.float32, device=device)

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

    netG = GAIN_Generator(input_dim).to(device)
    netC = WGAIN_Critic(input_dim).to(device)

    optG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.9))
    optC = optim.Adam(netC.parameters(), lr=LR_D, betas=(0.5, 0.9))

    print(f"\n>>> 开始训练 PG-WGAIN...")

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_netG_state = None

    for epoch in range(EPOCHS):
        netG.train()
        netC.train()

        t = epoch + 1
        alpha2_t = BETA_MAX / (1 + np.exp(K * (t - T0)))
        alpha1_t = 1.0 - alpha2_t / BETA_MAX
        current_alpha = ALPHA_MAX * alpha1_t
        current_beta = alpha2_t

        loss_c_log, loss_g_log, loss_phy_log = 0, 0, 0
        batches_processed = 0

        train_iter = iter(train_loader)
        while True:
            try:
                for _ in range(N_CRITIC):
                    batch_x = next(train_iter)[0]
                    B, Dim = batch_x.shape

                    mask = (torch.rand(B, Dim, device=device) > MASK_RATE).float()
                    noise = torch.rand(B, Dim, device=device)
                    x_corrupted = batch_x * mask + noise * (1 - mask)
                    hint = sample_hint(mask)

                    with torch.no_grad():
                        x_generated = netG(x_corrupted, mask)
                        x_imputed = batch_x * mask + x_generated * (1 - mask)

                    c_real = netC(batch_x, hint)
                    c_fake = netC(x_imputed.detach(), hint)

                    loss_c_wasserstein = torch.mean(c_fake) - torch.mean(c_real)
                    gp = calc_gradient_penalty(netC, batch_x, x_imputed.detach(), hint, device)
                    loss_c = loss_c_wasserstein + LAMBDA_GP * gp

                    optC.zero_grad()
                    loss_c.backward()
                    optC.step()

                    loss_c_log += loss_c.item()

                batch_x = next(train_iter)[0]
                B, Dim = batch_x.shape

                mask = (torch.rand(B, Dim, device=device) > MASK_RATE).float()
                hint = sample_hint(mask)

                x_generated = netG(batch_x, mask)
                x_imputed = batch_x * mask + x_generated * (1 - mask)

                c_fake = netC(x_imputed, hint)
                loss_g_adv = -torch.mean(c_fake)
                # 修改建议：只计算 mask=0 部分的损失
                loss_g_mse = torch.mean(((1 - mask) * batch_x - (1 - mask) * x_generated) ** 2) / (
                            torch.mean(1 - mask) + 1e-8)
                loss_g_phy = differentiable_physics_loss(x_imputed, A_inc_tensor, scaler_min, scaler_scale)
                loss_g = loss_g_adv + current_alpha * loss_g_mse + current_beta * loss_g_phy

                optG.zero_grad()
                loss_g.backward()
                optG.step()

                loss_g_log += loss_g.item()
                loss_phy_log += loss_g_phy.item()
                batches_processed += 1

            except StopIteration:
                break

        if (epoch + 1) % 10 == 0:
            avg_c = loss_c_log / (batches_processed * N_CRITIC)
            avg_g = loss_g_log / batches_processed
            avg_phy = loss_phy_log / batches_processed
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | C Loss: {avg_c:.4f} | G Loss: {avg_g:.4f} | Phy Loss: {avg_phy:.6f} | α(MSE): {current_alpha:.2f} | β(Phy): {current_beta:.2f}")

        netG.eval()
        val_loss_total = 0
        val_batches = 0

        with torch.no_grad():
            for val_batch_x, in val_loader:
                B, Dim = val_batch_x.shape
                mask = (torch.rand(B, Dim, device=device) > MASK_RATE).float()
                x_generated = netG(val_batch_x, mask)
                x_imputed = val_batch_x * mask + x_generated * (1 - mask)
                val_mse = torch.mean(((1 - mask) * val_batch_x - (1 - mask) * x_generated) ** 2) / (torch.mean(1 - mask) + 1e-8)
                val_phy = differentiable_physics_loss(x_imputed, A_inc_tensor, scaler_min, scaler_scale)
                val_loss = current_alpha * val_mse + current_beta * val_phy

                val_loss_total += val_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_total / val_batches if val_batches > 0 else float('inf')
        netG.train()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_netG_state = copy.deepcopy(netG.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n🛑 早停触发！验证损失已连续 {PATIENCE} 个epoch未改善")
                break

    if best_netG_state is not None:
        netG.load_state_dict(best_netG_state)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = f'pg_wgain_best_{timestamp}.pth'
    torch.save({
        'netG_state_dict': best_netG_state,
        'netC_state_dict': netC.state_dict(),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'source_tag': source_tag,
        'input_dim': input_dim
    }, model_path)

    visualize_gain_result(netG, test_loader, scaler, device, MASK_RATE, data_dir=data_dir)

    return model_path, scaler, netG


# ==========================================
# 4.5 目标域变拓扑微调程序 (Frozen Fine-Tuning)
# ==========================================
def main_finetune(pretrained_model_path, target_data_dir=TARGET_NORMAL_DIR, scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==========================================")
    print(f"     阶段 2：目标域变拓扑场景冻结微调")
    print(f"==========================================")

    BATCH_SIZE = 32
    EPOCHS = 100
    LR_FINETUNE = 5e-4
    N_CRITIC = 3
    MASK_RATE = 0.3

    data_all, scaler, target_tag = load_and_normalize_data(target_data_dir, scaler=scaler, fit_scaler=(scaler is None))
    input_dim = data_all.shape[1]
    X_tensor = torch.FloatTensor(data_all).to(device)

    scaler_min = torch.tensor(scaler.min_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale[scaler_scale == 0] = 1.0

    A_inc_np_target, num_buses, num_lines = load_grid_topology(target_data_dir)
    A_inc_tensor_target = torch.tensor(A_inc_np_target, dtype=torch.float32, device=device)

    target_train_size = 200
    train_tensor = X_tensor[:target_train_size]
    test_tensor = X_tensor[target_train_size:]
    if len(test_tensor) == 0:
        test_tensor = X_tensor

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=BATCH_SIZE, shuffle=False)

    netG = GAIN_Generator(input_dim).to(device)
    netC = WGAIN_Critic(input_dim).to(device)

    try:
        checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(pretrained_model_path, map_location=device)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netC.load_state_dict(checkpoint['netC_state_dict'])
    print("✅ 已成功加载源域预训练模型权重。")

    for param in netG.feature_embedding.parameters():
        param.requires_grad = False
    for param in netG.self_attention.parameters():
        param.requires_grad = False
    print("🔒 已冻结生成器底层的 [特征嵌入层] 和 [自注意力层]。")

    optG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=LR_FINETUNE, betas=(0.5, 0.9))
    optC = optim.Adam(netC.parameters(), lr=LR_FINETUNE, betas=(0.5, 0.9))

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
                        x_generated = netG(batch_x, mask)
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
                x_generated = netG(batch_x, mask)
                x_imputed = batch_x * mask + x_generated * (1 - mask)

                loss_g_adv = -torch.mean(netC(x_imputed, hint))
                # 修改建议：只计算 mask=0 部分的损失
                loss_g_mse = torch.mean(((1 - mask) * batch_x - (1 - mask) * x_generated) ** 2) / (
                            torch.mean(1 - mask) + 1e-8)

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
    visualize_gain_result_target(netG, test_loader, scaler, device, MASK_RATE, A_inc_np_target, num_buses, num_lines)

    # 【改动 2】：返回微调后的生成器，供后续恢复阶段测试使用
    return netG, scaler


# ==========================================
# 【改动 3】新增：定向攻击恢复阶段 (测试微调后模型的性能)
# ==========================================
def evaluate_attack_recovery(netG, scaler, target_data_dir=ATTACK_TEST_DIR):
    device = next(netG.parameters()).device
    print(f"\n==========================================")
    print(f"     阶段 3：攻击样本定向恢复测试")
    print(f"==========================================")

    attack_data = _load_attack_test_data(target_data_dir)
    sample_mat = attack_data['samples']
    clean_mat = attack_data['clean_samples']
    label_mat = attack_data['label_mat']
    recovery_mask = attack_data['recovery_mask']
    tag = attack_data['tag']

    # Recovery_Mask 中 1 表示可信保留，0 表示待恢复
    mask_tensor = torch.FloatTensor(recovery_mask).to(device)

    norm_samples = scaler.transform(sample_mat)
    norm_samples_tensor = torch.FloatTensor(norm_samples).to(device)

    A_inc_np, num_buses, num_lines = load_grid_topology(target_data_dir)

    netG.eval()
    BATCH_SIZE = 128
    num_samples = norm_samples_tensor.shape[0]
    x_imputed_list = []

    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            batch_x = norm_samples_tensor[i:i + BATCH_SIZE]
            batch_mask = mask_tensor[i:i + BATCH_SIZE]
            x_generated = netG(batch_x, batch_mask)
            batch_imputed = batch_x * batch_mask + x_generated * (1.0 - batch_mask)
            x_imputed_list.append(batch_imputed)

    x_imputed = torch.cat(x_imputed_list, dim=0)
    pred_mat = scaler.inverse_transform(x_imputed.cpu().numpy())

    pid_attacked = calculate_pid_metric(sample_mat, A_inc_np, num_buses, num_lines)
    pid_recovered = calculate_pid_metric(pred_mat, A_inc_np, num_buses, num_lines)
    pid_improvement = (pid_attacked - pid_recovered) / pid_attacked * 100 if pid_attacked > 0 else 0

    print(f"📊 定向恢复性能测试结果")
    print(f"   测试文件标识: {tag}")
    print(f"   总测试样本数: {sample_mat.shape[0]}")

    if label_mat is not None:
        attacked_pos = label_mat.astype(bool)
        print(f"   平均每个样本被篡改的量测数: {np.mean(label_mat.sum(axis=1)):.2f} 个")
    else:
        attacked_pos = recovery_mask < 0.5
        print(f"   平均每个样本待恢复量测数: {np.mean(np.sum(attacked_pos, axis=1)):.2f} 个")

    print(f"   [攻击后] PID: {pid_attacked:.6f}")
    print(f"   [恢复后] PID: {pid_recovered:.6f}")
    print(f"   PID 改善率: {pid_improvement:.2f}%")

    metrics_to_save = {
        'PID_Attacked': pid_attacked,
        'PID_Recovered': pid_recovered,
        'PID_Improvement_Percent': pid_improvement,
    }

    if clean_mat is not None:
        pid_clean = calculate_pid_metric(clean_mat, A_inc_np, num_buses, num_lines)
        metrics_attack = masked_regression_metrics(clean_mat, sample_mat, attacked_pos)
        metrics_recover = masked_regression_metrics(clean_mat, pred_mat, attacked_pos)
        metrics_full = masked_regression_metrics(clean_mat, pred_mat, np.ones_like(clean_mat, dtype=bool))

        print(f"   [干净样本] PID: {pid_clean:.6f}")
        if metrics_attack is not None:
            print(f"\n   攻击位置恢复前误差:")
            print(f"      RMSE: {metrics_attack['RMSE']:.6f}")
            print(f"      MAE : {metrics_attack['MAE']:.6f}")
            print(f"      MAPE: {metrics_attack['MAPE']:.4f}%")
        if metrics_recover is not None:
            print(f"\n   攻击位置恢复后误差:")
            print(f"      RMSE: {metrics_recover['RMSE']:.6f}")
            print(f"      MAE : {metrics_recover['MAE']:.6f}")
            print(f"      MAPE: {metrics_recover['MAPE']:.4f}%")
        if metrics_full is not None:
            print(f"\n   全量测恢复后误差:")
            print(f"      RMSE: {metrics_full['RMSE']:.6f}")
            print(f"      MAE : {metrics_full['MAE']:.6f}")
            print(f"      MAPE: {metrics_full['MAPE']:.4f}%")

        metrics_to_save.update({
            'PID_Clean': pid_clean,
            'Attack_Pos_RMSE_Before': metrics_attack['RMSE'] if metrics_attack else np.nan,
            'Attack_Pos_MAE_Before': metrics_attack['MAE'] if metrics_attack else np.nan,
            'Attack_Pos_MAPE_Before': metrics_attack['MAPE'] if metrics_attack else np.nan,
            'Attack_Pos_RMSE_After': metrics_recover['RMSE'] if metrics_recover else np.nan,
            'Attack_Pos_MAE_After': metrics_recover['MAE'] if metrics_recover else np.nan,
            'Attack_Pos_MAPE_After': metrics_recover['MAPE'] if metrics_recover else np.nan,
            'Full_RMSE_After': metrics_full['RMSE'] if metrics_full else np.nan,
            'Full_MAE_After': metrics_full['MAE'] if metrics_full else np.nan,
            'Full_MAPE_After': metrics_full['MAPE'] if metrics_full else np.nan,
        })
    else:
        print("\n   未读取到 Clean_Samples，跳过 MAE、RMSE 和 MAPE。")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(target_data_dir, f'Recovered_Data_{tag}_{timestamp}.mat')
    save_dict = {
        'Recovered_Samples': pred_mat,
        'Original_Attacked': sample_mat,
        'Recovery_Mask': recovery_mask,
        'Metrics': metrics_to_save,
    }
    if clean_mat is not None:
        save_dict['Clean_Samples'] = clean_mat
    if label_mat is not None:
        save_dict['Attack_Label'] = label_mat
    if attack_data['sample_meta'] is not None:
        save_dict['Sample_Meta'] = attack_data['sample_meta']

    sio.savemat(out_file, save_dict)
    print(f"\n恢复结果已保存至: {out_file}")



def visualize_gain_result_target(netG, test_loader, scaler, device, mask_rate, A_inc, num_buses, num_lines):
    netG.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, in test_loader:
            B, Dim = batch_x.shape
            mask = (torch.rand(B, Dim).to(device) > mask_rate).float()

            x_generated = netG(batch_x, mask)
            x_imputed = batch_x * mask + x_generated * (1 - mask)

            all_preds.append(x_imputed.cpu().numpy())
            all_trues.append(batch_x.cpu().numpy())

    pred_mat = scaler.inverse_transform(np.concatenate(all_preds, axis=0))
    true_mat = scaler.inverse_transform(np.concatenate(all_trues, axis=0))

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

    print(f"\n📊 变拓扑微调重构性能 (Target Domain, 共 {len(true_values)} 个随机遮掩点):")
    print(f"   传统统计指标:")
    print(f"      MSE  : {rmse ** 2:.6f}")
    print(f"      RMSE : {rmse:.6f}")
    print(f"      MAE  : {mae:.6f}")
    print(f"      MAPE : {mape:.4f}%")
    print(f"\n   物理一致性指标 (PID):")
    print(f"      真实数据 PID: {pid_true:.6f}")
    print(f"      重构数据 PID: {pid_recovered:.6f}")
    print(f"      PID 改善率  : {pid_improvement:.2f}%")

    plot_single_sample_comparison(pred_matrix=pred_mat, true_matrix=true_mat, sample_idx=0,
                                  method_name='PG-WGAIN-Target')
    save_single_sample_to_excel(pred_matrix=pred_mat, true_matrix=true_mat, sample_idx=0, method_name='PG_WGAIN_Target')


def visualize_gain_result(netG, test_loader, scaler, device, mask_rate=0.2, data_dir=SOURCE_NORMAL_DIR):
    print(f"\n>>> 正在生成 PG-WGAIN 对比图 (全维度评估)...")
    netG.eval()

    set_random_seed(2025)

    A_inc, num_buses, num_lines = load_grid_topology(data_dir)
    A_inc = A_inc.astype(np.float32)

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_x, in test_loader:
            B, Dim = batch_x.shape
            mask = (torch.rand(B, Dim).to(device) > mask_rate).float()

            x_generated = netG(batch_x, mask)
            x_imputed = batch_x * mask + x_generated * (1 - mask)

            all_preds.append(x_imputed.cpu().numpy())
            all_trues.append(batch_x.cpu().numpy())

    pred_mat = scaler.inverse_transform(np.concatenate(all_preds, axis=0))
    true_mat = scaler.inverse_transform(np.concatenate(all_trues, axis=0))

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
    # 1. 源域正常样本预训练
    best_model_path, source_scaler, netG_current = main(data_dir=SOURCE_NORMAL_DIR)

    # 2. 目标域少量正常样本微调。没有该文件夹时，直接使用源域预训练模型。
    if os.path.exists(TARGET_NORMAL_DIR):
        netG_current, scaler_current = main_finetune(
            pretrained_model_path=best_model_path,
            target_data_dir=TARGET_NORMAL_DIR,
            scaler=source_scaler
        )
    else:
        print(f"\n未检测到 {TARGET_NORMAL_DIR} 文件夹，跳过目标域微调，直接使用源域预训练模型。")
        scaler_current = source_scaler

    # 3. 攻击和定位数据只用于测试阶段
    if os.path.exists(ATTACK_TEST_DIR):
        evaluate_attack_recovery(netG_current, scaler_current, target_data_dir=ATTACK_TEST_DIR)
    else:
        print(f"\n未检测到 {ATTACK_TEST_DIR} 文件夹，跳过攻击恢复测试。")
