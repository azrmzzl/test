import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import scipy.io as sio
import os
import glob
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import time
import copy


# ==========================================
# 0. 数据路径配置
# ==========================================
# ==========================================
# 【新增】运行模式控制
# ==========================================
# 运行模式选项：
# - 'all': 运行全部三个阶段（预训练 → 微调 → 攻击恢复）
# - 'stage1': 只运行第一阶段（源域预训练）
# - 'stage2': 只运行第二阶段（目标域微调），需要指定 STAGE2_PRETRAINED_MODEL_PATH
# - 'stage3': 只运行第三阶段（攻击恢复测试），需要指定 STAGE3_MODEL_PATH
RUN_MODE = 'stage3'  # 默认只运行第三阶段

# 【新增】第二阶段模型加载配置
# None: 自动查找最新的预训练模型
# 字符串: 指定具体的 .pth 文件路径，例如 'pg_wgain_best_20260520-184436.pth'
STAGE2_PRETRAINED_MODEL_PATH = 'pg_wgain_best_20260524-160844.pth'  # 第二阶段加载的预训练模型文件

# 【新增】第三阶段模型加载配置
# None: 自动查找最新的微调或预训练模型
# 字符串: 指定具体的 .pth 文件路径，例如 'pg_wgain_finetuned_20260520-180442.pth'
STAGE3_MODEL_PATH = 'pg_wgain_finetuned_20260602-161330.pth'  # 指定第三阶段加载的模型文件

# 【新增】训练样本数量控制（可选）
# None: 使用所有样本；整数N: 只使用前N个样本进行训练
# 例如：SOURCE_TRAIN_SAMPLES = 10000 表示只使用前10000个样本
SOURCE_TRAIN_SAMPLES = 10000  # 源域预训练使用的样本数
TARGET_TRAIN_SAMPLES = 1000  # 目标域微调使用的样本数
ATTACK_TEST_SAMPLES = 1000  # 攻击恢复测试使用的样本数

# 【新增】动态掩码率配置
# USE_DYNAMIC_MASK_RATE: True=使用动态掩码率，False=使用固定掩码率
# MASK_RATE_MIN/MAX: 动态掩码率的范围（仅在USE_DYNAMIC_MASK_RATE=True时有效）
# 例如：0.2-0.5表示随机选择20%-50%的缺失率
USE_DYNAMIC_MASK_RATE = True  # 是否启用动态掩码率
MASK_RATE_MIN = 0.25  # 【修改】提高最小掩码率，从0.2提高到0.3，缩小范围
MASK_RATE_MAX = 0.45  # 【修改】降低最大掩码率，从0.5降低到0.45，缩小范围

# 训练阶段只放正常样本。文件夹内至少包含 Samples_*.mat，建议同时放 Topology_Branch.mat。
SOURCE_NORMAL_DIR = 'dataset_split_source_normal'

# 目标域少量正常样本，用于变拓扑冻结微调。没有该文件夹时会自动跳过微调。
TARGET_NORMAL_DIR = 'dataset_split_target_normal'

# 攻击测试阶段数据，由 segment_*.m 输出，包含 Samples、Clean_Samples、Labels_Meas 或 Recovery_Mask。
ATTACK_TEST_DIR = 'dataset_split_attack_test'

# 【新增】恢复数据保存目录
RECOVERY_OUTPUT_DIR = 'recovery_results'

# 如果你的正常样本仍放在旧的 dataset_split 文件夹，可以把 SOURCE_NORMAL_DIR 改成 'dataset_split'。

# 精确指定读取哪个 Samples_*.mat。
# 例如 SOURCE_TAG = '正常运行_2001_2880时刻' 对应 Samples_正常运行_2001_2880时刻.mat。
# 如果设为 None 或 ''，则默认读取对应文件夹中修改时间最新的 Samples_*.mat。
SOURCE_TAG = '正常运行_1_35000时刻_正常数据波动'
TARGET_TAG = '支路12-13断路_1_2880时刻_正常数据波动'
ATTACK_TAG = '支路12-13断路_1_1000时刻波动'



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
        # 【新增】除以特征维度，避免损失值过大
        num_features = bus_p.shape[1]
        return (loss_p + loss_q) / num_features
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


def _normalize_tag(tag):
    if tag is None:
        return None
    tag = str(tag).strip()
    return tag if tag else None


def _find_required_file(data_dir, prefix, tag=None):
    tag = _normalize_tag(tag)
    if tag is None:
        files = _sorted_mat_files(data_dir, f'{prefix}*.mat')
        if len(files) > 1:
            print(f'⚠️ {data_dir} 中存在多个 {prefix}*.mat，未指定 tag，默认读取修改时间最新的文件：')
            for f in files[:10]:
                print(f'   {os.path.basename(f)}')
    else:
        target_file = os.path.join(data_dir, f'{prefix}{tag}.mat')
        files = [target_file] if os.path.exists(target_file) else []

    if not files:
        if tag is None:
            raise FileNotFoundError(f'在 {data_dir} 中未找到 {prefix}*.mat')
        raise FileNotFoundError(f'在 {data_dir} 中未找到指定文件: {prefix}{tag}.mat')
    return files[0]


def _find_optional_file(data_dir, prefix, tag=None):
    tag = _normalize_tag(tag)
    if tag is None:
        files = _sorted_mat_files(data_dir, f'{prefix}*.mat')
    else:
        target_file = os.path.join(data_dir, f'{prefix}{tag}.mat')
        files = [target_file] if os.path.exists(target_file) else []
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


def load_and_normalize_data(data_dir='dataset_split', tag=None, max_samples=None):
    """
    训练或微调阶段只读取正常样本 Samples_*.mat。
    始终进行归一化处理，返回归一化后的数据和拟合的 Scaler。
    
    参数:
        data_dir: 数据目录
        tag: 文件标签
        max_samples: 最大样本数，None表示使用所有样本，整数N表示只使用前N个样本
    """
    raw_data, tag, sample_file = load_samples_from_dir(data_dir, tag=tag)
    
    # 【新增】如果指定了最大样本数，则截取前N个样本
    if max_samples is not None and max_samples < len(raw_data):
        print(f"⚠️  样本数量限制: {len(raw_data)} → {max_samples} (使用前{max_samples}个样本)")
        raw_data = raw_data[:max_samples]

    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(raw_data)

    return data_norm.astype(np.float32), scaler, tag


def load_grid_topology(data_dir='dataset_split', tag=None):
    """
    读取 Topology_Branch。
    优先读取带 tag 的文件，例如：
    Topology_Branch_支路12-13断路_2001_2880时刻.mat

    如果 tag 为空，则仍然读取固定文件名：
    Topology_Branch.mat
    """
    tag = _normalize_tag(tag)

    if tag is not None:
        topo_file = os.path.join(data_dir, f'Topology_Branch_{tag}.mat')
        if not os.path.exists(topo_file):
            raise FileNotFoundError(f'未找到指定拓扑文件: {topo_file}')
    else:
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

    print(f'>>> 读取拓扑文件: {topo_file}')
    return A_inc, num_buses, num_lines


def _load_attack_test_data(data_dir, tag=None):
    """
    从攻击测试文件夹读取同一个 tag 的数据，避免 Samples、Clean_Samples 和标签错配。
    优先读取 Recovery_Mask；没有时读取 Pred_Labels_Meas；仍没有时读取 Labels_Meas。
    """
    samples, tag, sample_file = load_samples_from_dir(data_dir, tag=tag)

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
    return {'RMSE': rmse, 'MAE': mae}


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


def generate_mask(B, Dim, device, use_dynamic=True, mask_rate=0.4, mask_rate_min=0.2, mask_rate_max=0.6):
    """
    【新增】生成动态或固定掩码率的掩码
    
    参数:
        B: batch size
        Dim: 特征维度
        device: 设备
        use_dynamic: 是否使用动态掩码率
        mask_rate: 固定掩码率（当use_dynamic=False时使用）
        mask_rate_min: 动态掩码率最小值
        mask_rate_max: 动态掩码率最大值
    
    返回:
        mask: 掩码矩阵，1表示保留，0表示遮蔽
        current_mask_rate: 当前使用的掩码率
    """
    if use_dynamic:
        # 动态掩码率：在[min, max]范围内随机选择
        current_mask_rate = np.random.uniform(mask_rate_min, mask_rate_max)
    else:
        # 固定掩码率
        current_mask_rate = mask_rate
    
    # 生成掩码：mask中1表示保留，0表示遮蔽
    mask = (torch.rand(B, Dim, device=device) > current_mask_rate).float()
    
    return mask, current_mask_rate


# ==========================================
# 4. 训练主程序 (引入 WGAN-GP 机制)
# ==========================================
def main(data_dir=SOURCE_NORMAL_DIR, tag=SOURCE_TAG):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 参数
    ALPHA_MAX = 10.0
    BETA_MAX = 1
    K = 0.05
    T0 = 200
    PATIENCE = 200
    BATCH_SIZE = 128
    EPOCHS = 1000
    LR_G = 0.001
    LR_D = 0.001
    N_CRITIC = 3
    LAMBDA_GP = 10
    MASK_RATE = 0.4

    # 1. 准备数据
    data_all, scaler, source_tag = load_and_normalize_data(data_dir, tag=tag, max_samples=SOURCE_TRAIN_SAMPLES)
    input_dim = data_all.shape[1]
    X_tensor = torch.FloatTensor(data_all).to(device)

    scaler_min = torch.tensor(scaler.min_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale[scaler_scale == 0] = 1.0

    A_inc_np, _, _ = load_grid_topology(data_dir, tag=source_tag)
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

    best_val_mse = float('inf')  # 【修改】只看 MSE 判断最优模型
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

        loss_c_log, loss_g_log, loss_phy_log, loss_mse_log = 0, 0, 0, 0
        batches_processed = 0

        train_iter = iter(train_loader)
        while True:
            try:
                for _ in range(N_CRITIC):
                    batch_x = next(train_iter)[0]
                    B, Dim = batch_x.shape

                    # 【修改】使用动态或固定掩码率
                    mask, current_mr = generate_mask(B, Dim, device, 
                                                      use_dynamic=USE_DYNAMIC_MASK_RATE,
                                                      mask_rate=MASK_RATE,
                                                      mask_rate_min=MASK_RATE_MIN,
                                                      mask_rate_max=MASK_RATE_MAX)
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

                # 【修改】使用动态或固定掩码率
                mask, current_mr = generate_mask(B, Dim, device,
                                                  use_dynamic=USE_DYNAMIC_MASK_RATE,
                                                  mask_rate=MASK_RATE,
                                                  mask_rate_min=MASK_RATE_MIN,
                                                  mask_rate_max=MASK_RATE_MAX)
                hint = sample_hint(mask)

                x_generated = netG(batch_x, mask)
                x_imputed = batch_x * mask + x_generated * (1 - mask)

                c_fake = netC(x_imputed, hint)
                loss_g_adv = -torch.mean(c_fake)
                # 根在可信位置(m=1)计算重构损失
                loss_g_mse = torch.mean((mask * batch_x - mask * x_generated) ** 2) / (
                            torch.mean(mask) + 1e-8)
                loss_g_phy = differentiable_physics_loss(x_imputed, A_inc_tensor, scaler_min, scaler_scale)
                loss_g = loss_g_adv + current_alpha * loss_g_mse + current_beta * loss_g_phy

                optG.zero_grad()
                loss_g.backward()
                optG.step()

                loss_g_log += loss_g.item()
                loss_phy_log += loss_g_phy.item()
                loss_mse_log += loss_g_mse.item()
                batches_processed += 1

            except StopIteration:
                break

        if (epoch + 1) % 10 == 0:
            avg_c = loss_c_log / (batches_processed * N_CRITIC)
            avg_g = loss_g_log / batches_processed
            avg_phy = loss_phy_log / batches_processed
            avg_mse = loss_mse_log / batches_processed
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | C Loss: {avg_c:.4f} | G Loss: {avg_g:.4f} | α(MSE): {current_alpha:.2f} | β(Phy): {current_beta:.2f}")
            print(f"   [Debug] MSE={avg_mse:.6f}, Phy={avg_phy:.6f}, α*MSE={current_alpha*avg_mse:.4f}, β*Phy={current_beta*avg_phy:.4f}")
            if USE_DYNAMIC_MASK_RATE:
                print(f"   🎲 动态掩码率范围: [{MASK_RATE_MIN:.1%}, {MASK_RATE_MAX:.1%}]")

        netG.eval()
        val_mse_total = 0  # 【修改】只累积 MSE 损失
        val_batches = 0

        with torch.no_grad():
            for val_batch_x, in val_loader:
                B, Dim = val_batch_x.shape
                # 【修改】验证时也使用动态或固定掩码率
                mask, _ = generate_mask(B, Dim, device,
                                        use_dynamic=USE_DYNAMIC_MASK_RATE,
                                        mask_rate=MASK_RATE,
                                        mask_rate_min=MASK_RATE_MIN,
                                        mask_rate_max=MASK_RATE_MAX)
                x_generated = netG(val_batch_x, mask)
                x_imputed = val_batch_x * mask + x_generated * (1 - mask)
                # 在可信位置(m=1)计算重构损失
                val_mse = torch.mean((mask * val_batch_x - mask * x_generated) ** 2) / torch.mean(mask)

                val_mse_total += val_mse.item()  # 【修改】只累积 MSE
                val_batches += 1

        avg_val_mse = val_mse_total / val_batches if val_batches > 0 else float('inf')  # 【修改】平均 MSE
        netG.train()

        # 【修改】只根据 MSE 判断是否改善
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            best_epoch = epoch + 1
            patience_counter = 0
            best_netG_state = copy.deepcopy(netG.state_dict())
            if (epoch + 1) % 10 == 0:
                print(f"   ✅ 验证 MSE 改善: {avg_val_mse:.6f} (最优epoch: {best_epoch})")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f"   ⏳ 验证 MSE 未改善: {avg_val_mse:.6f} (patience: {patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                print(f"\n🛑 早停触发！验证 MSE 已连续 {PATIENCE} 个epoch未改善")
                print(f"   最优epoch: {best_epoch}, 最优验证 MSE: {best_val_mse:.6f}")
                break

    if best_netG_state is not None:
        netG.load_state_dict(best_netG_state)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = f'pg_wgain_best_{timestamp}.pth'
    torch.save({
        'netG_state_dict': best_netG_state,
        'netC_state_dict': netC.state_dict(),
        'best_epoch': best_epoch,
        'best_val_mse': best_val_mse,  # 【修改】保存最优 MSE
        'source_tag': source_tag,
        'input_dim': input_dim
    }, model_path)
    print(f"\n💾 预训练模型已保存至: {model_path}")
    print(f"   最优epoch: {best_epoch}, 最优验证 MSE: {best_val_mse:.6f}")

    # 【注释】不生成图像
    visualize_gain_result(netG, test_loader, scaler, device, MASK_RATE, data_dir=data_dir, tag=source_tag,
                           max_abs_error_threshold=None)

    return model_path, scaler, netG


# ==========================================
# 4.5 目标域变拓扑微调程序 (Frozen Fine-Tuning)
# ==========================================
def main_finetune(pretrained_model_path, target_data_dir=TARGET_NORMAL_DIR, tag=TARGET_TAG, scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==========================================")
    print(f"     阶段 2：目标域变拓扑场景冻结微调")
    print(f"==========================================")

    BATCH_SIZE = 64
    EPOCHS = 500
    LR_FINETUNE = 5e-4  # 【修改】降低学习率，从1e-3降到5e-4，减少震荡
    N_CRITIC = 3
    MASK_RATE = 0.4
    PATIENCE = 100
    data_all, scaler, target_tag = load_and_normalize_data(target_data_dir, tag=tag, max_samples=TARGET_TRAIN_SAMPLES)
    input_dim = data_all.shape[1]
    X_tensor = torch.FloatTensor(data_all).to(device)

    scaler_min = torch.tensor(scaler.min_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).view(1, -1)
    scaler_scale[scaler_scale == 0] = 1.0
    
    # 【调试信息】打印 Scaler 参数，检查是否与源域差异过大
    print(f"   📊 目标域 Scaler 参数统计:")
    print(f"      min 范围: [{scaler.min_.min():.4f}, {scaler.min_.max():.4f}]")
    print(f"      scale 范围: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
    print(f"      ⚠️  如果与源域差异很大，物理损失可能不准确")

    A_inc_np_target, num_buses, num_lines = load_grid_topology(target_data_dir, tag=target_tag)
    A_inc_tensor_target = torch.tensor(A_inc_np_target, dtype=torch.float32, device=device)

    # 【修改】与预训练阶段保持一致的数据划分方式
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
    
    print(f"   📊 目标域数据划分:")
    print(f"      总样本数: {len(X_tensor)}")
    print(f"      训练集: {len(train_dataset)} 样本")
    print(f"      验证集: {len(val_dataset)} 样本")
    print(f"      测试集: {len(test_dataset)} 样本 (首样本 + 验证集)")

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
    
    # 【新增】学习率调度器：当验证损失不再改善时降低学习率
    schedulerG = optim.lr_scheduler.ReduceLROnPlateau(optG, mode='min', factor=0.5, patience=30, verbose=True)
    schedulerC = optim.lr_scheduler.ReduceLROnPlateau(optC, mode='min', factor=0.5, patience=30, verbose=True)

    current_alpha = 10.0
    current_beta = 1

    # 【新增】用于保存最优模型的变量
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    best_netG_state = None
    best_netC_state = None

    print(f">>> 开始微调 (Target Domain) ...")
    for epoch in range(EPOCHS):
        netG.train()
        netC.train()
        loss_c_log, loss_g_log, loss_phy_log, loss_mse_log = 0, 0, 0, 0
        batches_processed = 0
        train_iter = iter(train_loader)

        while True:
            try:
                for _ in range(N_CRITIC):
                    batch_x = next(train_iter)[0]
                    B, Dim = batch_x.shape
                    # 【修改】使用动态或固定掩码率
                    mask, current_mr = generate_mask(B, Dim, device,
                                                      use_dynamic=USE_DYNAMIC_MASK_RATE,
                                                      mask_rate=MASK_RATE,
                                                      mask_rate_min=MASK_RATE_MIN,
                                                      mask_rate_max=MASK_RATE_MAX)
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
                # 【修改】使用动态或固定掩码率
                mask, current_mr = generate_mask(B, Dim, device,
                                                  use_dynamic=USE_DYNAMIC_MASK_RATE,
                                                  mask_rate=MASK_RATE,
                                                  mask_rate_min=MASK_RATE_MIN,
                                                  mask_rate_max=MASK_RATE_MAX)
                hint = sample_hint(mask)
                x_generated = netG(batch_x, mask)
                x_imputed = batch_x * mask + x_generated * (1 - mask)

                loss_g_adv = -torch.mean(netC(x_imputed, hint))
                # 在可信位置(m=1)计算重构损失
                loss_g_mse = torch.mean((mask * batch_x - mask * x_generated) ** 2) / (
                            torch.mean(mask) + 1e-8)

                loss_g_phy = differentiable_physics_loss(x_imputed, A_inc_tensor_target, scaler_min, scaler_scale)
                loss_g = loss_g_adv + current_alpha * loss_g_mse + current_beta * loss_g_phy

                optG.zero_grad()
                loss_g.backward()
                optG.step()

                loss_g_log += loss_g.item()
                loss_phy_log += loss_g_phy.item()
                loss_mse_log += loss_g_mse.item()
                batches_processed += 1
            except StopIteration:
                break

        # 【新增】计算验证损失并保存最优模型
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_phy = loss_phy_log / batches_processed if batches_processed > 0 else 0
            avg_mse = loss_mse_log / batches_processed if batches_processed > 0 else 0
            avg_loss = loss_g_log / batches_processed if batches_processed > 0 else 0
            
            # 在验证集上评估（使用val_loader而非test_loader）
            netG.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_x = val_batch[0].to(device)
                    B, Dim = val_x.shape
                    # 【修改】验证时也使用动态或固定掩码率
                    val_mask, _ = generate_mask(B, Dim, device,
                                                use_dynamic=USE_DYNAMIC_MASK_RATE,
                                                mask_rate=MASK_RATE,
                                                mask_rate_min=MASK_RATE_MIN,
                                                mask_rate_max=MASK_RATE_MAX)
                    val_hint = sample_hint(val_mask)
                    val_generated = netG(val_x, val_mask)
                    val_imputed = val_x * val_mask + val_generated * (1 - val_mask)
                    
                    # 计算验证损失（仅MSE）
                    val_mse = torch.mean((val_mask * val_x - val_mask * val_generated) ** 2) / (
                                torch.mean(val_mask) + 1e-8)
                    val_loss += val_mse.item()
                    val_batches += 1
            
            if val_batches > 0:
                val_loss /= val_batches
            
            # 【新增】更新学习率调度器
            schedulerG.step(val_loss)
            schedulerC.step(val_loss)
            
            netG.train()
            
            print(
                f"Finetune Epoch {epoch + 1}/{EPOCHS} | C Loss: {loss_c_log / (batches_processed * N_CRITIC):.4f} | "
                f"G Loss: {loss_g_log / batches_processed:.4f} | Val MSE: {val_loss:.6f}")
            print(f"   [Debug] MSE={avg_mse:.6f}, Phy={avg_phy:.6f}, α*MSE={current_alpha*avg_mse:.4f}, β*Phy={current_beta*avg_phy:.4f}")
            if USE_DYNAMIC_MASK_RATE:
                print(f"   🎲 动态掩码率范围: [{MASK_RATE_MIN:.1%}, {MASK_RATE_MAX:.1%}]")
            
            # 【新增】早停逻辑和最优模型保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_netG_state = netG.state_dict().copy()
                best_netC_state = netC.state_dict().copy()
                print(f"   ✨ 发现更优模型！Val MSE: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n🛑 早停触发！验证损失已连续 {PATIENCE} 个epoch未改善")
                    break
        else:
            # 非评估epoch也打印基本信息
            if (epoch + 1) % 10 == 0:
                avg_phy = loss_phy_log / batches_processed if batches_processed > 0 else 0
                avg_mse = loss_mse_log / batches_processed if batches_processed > 0 else 0
                print(
                    f"Finetune Epoch {epoch + 1}/{EPOCHS} | C Loss: {loss_c_log / (batches_processed * N_CRITIC):.4f} | "
                    f"G Loss: {loss_g_log / batches_processed:.4f}")
                print(f"   [Debug] MSE={avg_mse:.6f}, Phy={avg_phy:.6f}, α*MSE={current_alpha*avg_mse:.4f}, β*Phy={current_beta*avg_phy:.4f}")

    # 【新增】恢复最优模型
    if best_netG_state is not None:
        netG.load_state_dict(best_netG_state)
        netC.load_state_dict(best_netC_state)
        print(f"\n✨ 已恢复最优微调模型 (epoch {best_epoch}, Val MSE: {best_val_loss:.6f})")

    print("\n✨ 目标域微调完成！开始测试...")
    
    # 【新增】保存微调后的模型
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    finetuned_model_path = f'pg_wgain_finetuned_{timestamp}.pth'
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netC_state_dict': netC.state_dict(),
        'target_tag': target_tag,
        'input_dim': input_dim,
        'scaler': scaler  # 保存目标域的 scaler
    }, finetuned_model_path)
    print(f"💾 微调模型已保存至: {finetuned_model_path}")
    
    # 【注释】不生成图像
    visualize_gain_result_target(netG, test_loader, scaler, device, MASK_RATE, A_inc_np_target, num_buses, num_lines,max_abs_error_threshold=0.1)

    # 【改动 2】：返回微调后的生成器和模型路径，供后续恢复阶段测试使用
    return netG, scaler, finetuned_model_path


# ==========================================
# 【改动 3】新增：定向攻击恢复阶段 (测试微调后模型的性能)
# ==========================================
def evaluate_attack_recovery(netG, scaler, target_data_dir=ATTACK_TEST_DIR, tag=ATTACK_TAG, max_abs_error_threshold=None):
    device = next(netG.parameters()).device
    print(f"\n==========================================")
    print(f"     阶段 3：攻击样本定向恢复测试")
    print(f"==========================================")

    attack_data = _load_attack_test_data(target_data_dir, tag=tag)
    sample_mat = attack_data['samples']
    clean_mat = attack_data['clean_samples']
    label_mat = attack_data['label_mat']
    recovery_mask = attack_data['recovery_mask']
    tag = attack_data['tag']
    
    # 【新增】样本数量限制
    if ATTACK_TEST_SAMPLES is not None and ATTACK_TEST_SAMPLES < len(sample_mat):
        print(f"⚠️  攻击测试样本数量限制: {len(sample_mat)} → {ATTACK_TEST_SAMPLES} (使用前{ATTACK_TEST_SAMPLES}个样本)")
        sample_mat = sample_mat[:ATTACK_TEST_SAMPLES]
        if clean_mat is not None:
            clean_mat = clean_mat[:ATTACK_TEST_SAMPLES]
        if label_mat is not None:
            label_mat = label_mat[:ATTACK_TEST_SAMPLES]
        if recovery_mask is not None:
            recovery_mask = recovery_mask[:ATTACK_TEST_SAMPLES]

    # Recovery_Mask 中 1 表示可信保留，0 表示待恢复
    mask_tensor = torch.FloatTensor(recovery_mask).to(device)

    norm_samples = scaler.transform(sample_mat)
    norm_samples_tensor = torch.FloatTensor(norm_samples).to(device)

    A_inc_np, num_buses, num_lines = load_grid_topology(target_data_dir, tag=tag)

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


    if clean_mat is not None and max_abs_error_threshold is not None:
        abs_errors = np.abs(pred_mat - clean_mat)
        max_abs_err_per_feature = np.max(abs_errors, axis=0)
        valid_features = max_abs_err_per_feature <= max_abs_error_threshold
        filtered_count = np.sum(~valid_features)
        
        if filtered_count > 0:

            pred_mat_for_metrics = pred_mat[:, valid_features]
            clean_mat_for_metrics = clean_mat[:, valid_features]
            sample_mat_for_metrics = sample_mat[:, valid_features]
            if label_mat is not None:
                label_mat_for_metrics = label_mat[:, valid_features]
            else:
                label_mat_for_metrics = None
        else:
            pred_mat_for_metrics = pred_mat
            clean_mat_for_metrics = clean_mat
            sample_mat_for_metrics = sample_mat
            label_mat_for_metrics = label_mat
    else:
        pred_mat_for_metrics = pred_mat
        clean_mat_for_metrics = clean_mat
        sample_mat_for_metrics = sample_mat
        label_mat_for_metrics = label_mat

    # 恢复数据已得到 pred_mat

    # 计算 PID 指标
    # 根据公式 (4-37) 计算 RPID，以干净样本 (Clean) 为基准
    if clean_mat is not None:
        pid_true = calculate_pid_metric(clean_mat, A_inc_np, num_buses, num_lines)
        pid_rec = calculate_pid_metric(pred_mat, A_inc_np, num_buses, num_lines)
        rpid = (abs(pid_rec - pid_true) / pid_true * 100) if pid_true > 0 else 0.0
        
        # 保留攻击后的 PID 供参考，但不作为 RPID 计算基准
        pid_attacked = calculate_pid_metric(sample_mat, A_inc_np, num_buses, num_lines)
    else:
        # 如果没有干净样本，退化为使用攻击后样本或无法计算 RPID
        pid_true = None
        pid_rec = calculate_pid_metric(pred_mat, A_inc_np, num_buses, num_lines)
        pid_attacked = calculate_pid_metric(sample_mat, A_inc_np, num_buses, num_lines)
        rpid = None

    print(f"📊 定向恢复性能测试结果")
    print(f"   测试文件标识: {tag}")
    print(f"   总测试样本数: {sample_mat.shape[0]}")

    if label_mat_for_metrics is not None:
        attacked_pos = label_mat_for_metrics.astype(bool)
        print(f"   平均每个样本被篡改的量测数: {np.mean(label_mat_for_metrics.sum(axis=1)):.2f} 个")
    else:
        attacked_pos = recovery_mask < 0.5
        print(f"   平均每个样本待恢复量测数: {np.mean(np.sum(attacked_pos, axis=1)):.2f} 个")

    print(f"   [攻击后] PID: {pid_attacked:.6f}")
    if pid_true is not None:
        print(f"   [原始正常] PID (基准): {pid_true:.6f}")
        print(f"   [恢复后] PID: {pid_rec:.6f}")
        print(f"   RPID (相对变化率): {rpid:.2f}%")
    metrics_to_save = {
        'PID_Attacked': pid_attacked,
        'PID_True': pid_true if pid_true is not None else np.nan,
        'PID_Recovered': pid_rec,
        'RPID_Percent': rpid if rpid is not None else np.nan,
    }

    if clean_mat_for_metrics is not None:
        metrics_attack = masked_regression_metrics(clean_mat_for_metrics, sample_mat_for_metrics, attacked_pos)
        metrics_recover = masked_regression_metrics(clean_mat_for_metrics, pred_mat_for_metrics, attacked_pos)
        metrics_full = masked_regression_metrics(clean_mat_for_metrics, pred_mat_for_metrics, np.ones_like(clean_mat_for_metrics, dtype=bool))

        if metrics_attack is not None:
            print(f"\n   恢复前误差:")
            print(f"      RMSE: {metrics_attack['RMSE']:.6f}")
            print(f"      MAE : {metrics_attack['MAE']:.6f}")


        if metrics_full is not None:
            print(f"\n   恢复后误差:")
            print(f"      RMSE: {metrics_full['RMSE']:.6f}")
            print(f"      MAE : {metrics_full['MAE']:.6f}")

        # 【新增】计算掩码节点小于30%的样本的RMSE和MAE
        if label_mat_for_metrics is not None:
            # 计算每个样本的掩码比例
            mask_ratio_per_sample = label_mat_for_metrics.sum(axis=1) / label_mat_for_metrics.shape[1]
            low_mask_samples = mask_ratio_per_sample < 0.3  # 掩码比例小于30%
            
            if np.any(low_mask_samples):
                low_mask_count = np.sum(low_mask_samples)
                print(f"\n📊 掩码节点<30%的样本统计 (共{low_mask_count}个样本):")
                
                # 提取这些样本的数据
                clean_low_mask = clean_mat_for_metrics[low_mask_samples]
                pred_low_mask = pred_mat_for_metrics[low_mask_samples]
                attacked_pos_low_mask = attacked_pos[low_mask_samples]
                
                # 计算攻击位置恢复后的误差
                metrics_low_mask_recover = masked_regression_metrics(clean_low_mask, pred_low_mask, attacked_pos_low_mask)
                # 计算全量测恢复后的误差
                metrics_low_mask_full = masked_regression_metrics(clean_low_mask, pred_low_mask, np.ones_like(clean_low_mask, dtype=bool))

                
                if metrics_low_mask_full is not None:
                    print(f"   恢复后误差:")
                    print(f"      RMSE: {metrics_low_mask_full['RMSE']:.6f}")
                    print(f"      MAE : {metrics_low_mask_full['MAE']:.6f}")
                
                # 保存到 metrics_to_save
                metrics_to_save.update({
                    'LowMask_Count': int(low_mask_count),
                    'LowMask_AttackPos_RMSE': metrics_low_mask_recover['RMSE'] if metrics_low_mask_recover else np.nan,
                    'LowMask_AttackPos_MAE': metrics_low_mask_recover['MAE'] if metrics_low_mask_recover else np.nan,
                    'LowMask_Full_RMSE': metrics_low_mask_full['RMSE'] if metrics_low_mask_full else np.nan,
                    'LowMask_Full_MAE': metrics_low_mask_full['MAE'] if metrics_low_mask_full else np.nan,
                })
            else:
                print(f"\n⚠️  没有掩码节点<30%的样本")
                metrics_to_save.update({
                    'LowMask_Count': 0,
                    'LowMask_AttackPos_RMSE': np.nan,
                    'LowMask_AttackPos_MAE': np.nan,
                    'LowMask_Full_RMSE': np.nan,
                    'LowMask_Full_MAE': np.nan,
                })

        # 【新增】打印单样本详细统计
        # if len(pred_mat_for_metrics) > 0:
        #     sample_idx = 0
        #     y_true_sample = clean_mat_for_metrics[sample_idx]
        #     y_pred_sample = pred_mat_for_metrics[sample_idx]
        #     abs_errors_sample = np.abs(y_true_sample - y_pred_sample)
        #
        #     max_abs_error = np.max(abs_errors_sample)
        #     max_error_idx = np.argmax(abs_errors_sample)
        #     min_abs_error = np.min(abs_errors_sample)
        #     mean_abs_error = np.mean(abs_errors_sample)
        #
        #     print(f"\n📋 攻击恢复单样本详细统计 (样本索引={sample_idx}):")
        #     print(f"   最大绝对误差：{max_abs_error:.6f} (特征索引：{max_error_idx})")
        #     print(f"   最小绝对误差：{min_abs_error:.6f}")
        #     print(f"   平均绝对误差：{mean_abs_error:.6f}")
        #     print(f"   误差标准差：  {np.std(abs_errors_sample):.6f}")
        #     print(f"   误差中位数：  {np.median(abs_errors_sample):.6f}")
        #
        #     # 【新增】打印被攻击位置的误差分布
        #     # attacked_pos 是二维布尔数组 (samples, features)，取第一个样本的被攻击位置
        #     attacked_mask_sample = attacked_pos[sample_idx]  # 一维布尔数组
        #     if np.any(attacked_mask_sample):
        #         attacked_errors_sample = np.abs(y_true_sample[attacked_mask_sample] -
        #                                        y_pred_sample[attacked_mask_sample])
        #
        #         print(f"\n📊 被攻击位置误差分布 (样本{sample_idx}, 共{attacked_mask_sample.sum()}个特征):")
        #         print(f"   最小误差: {attacked_errors_sample.min():.6f}")
        #         print(f"   25%分位: {np.percentile(attacked_errors_sample, 25):.6f}")
        #         print(f"   中位数:  {np.median(attacked_errors_sample):.6f}")
        #         print(f"   75%分位: {np.percentile(attacked_errors_sample, 75):.6f}")
        #         print(f"   最大误差: {attacked_errors_sample.max():.6f}")
        #         print(f"   平均值:  {attacked_errors_sample.mean():.6f}")
        #         print(f"   标准差:  {attacked_errors_sample.std():.6f}")

        metrics_to_save.update({
            'PID_True': pid_true if pid_true is not None else np.nan,
            'Attack_Pos_RMSE_Before': metrics_attack['RMSE'] if metrics_attack else np.nan,
            'Attack_Pos_MAE_Before': metrics_attack['MAE'] if metrics_attack else np.nan,
            'Attack_Pos_RMSE_After': metrics_recover['RMSE'] if metrics_recover else np.nan,
            'Attack_Pos_MAE_After': metrics_recover['MAE'] if metrics_recover else np.nan,
            'Full_RMSE_After': metrics_full['RMSE'] if metrics_full else np.nan,
            'Full_MAE_After': metrics_full['MAE'] if metrics_full else np.nan,
        })
    else:
        print("\n   未读取到 Clean_Samples，跳过 MAE、RMSE 和 MAPE。")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 【修改】保存到新的恢复结果文件夹
    if not os.path.exists(RECOVERY_OUTPUT_DIR):
        os.makedirs(RECOVERY_OUTPUT_DIR)
        print(f"✅ 创建恢复结果目录: {RECOVERY_OUTPUT_DIR}")
    out_file = os.path.join(RECOVERY_OUTPUT_DIR, f'Recovered_Data_{tag}_{timestamp}.mat')
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
    print(f"\n💾 恢复结果已保存至: {out_file}")



def visualize_gain_result_target(netG, test_loader, scaler, device, mask_rate, A_inc, num_buses, num_lines, max_abs_error_threshold=None):
    netG.eval()
    all_preds, all_trues, all_masks = [], [], []  # 【新增】保存掩码
    with torch.no_grad():
        for batch_x, in test_loader:
            B, Dim = batch_x.shape
            mask = (torch.rand(B, Dim).to(device) > mask_rate).float()

            x_generated = netG(batch_x, mask)
            x_imputed = batch_x * mask + x_generated * (1 - mask)

            all_preds.append(x_imputed.cpu().numpy())
            all_trues.append(batch_x.cpu().numpy())
            all_masks.append(mask.cpu().numpy())  # 【新增】保存掩码

    pred_mat = scaler.inverse_transform(np.concatenate(all_preds, axis=0))
    true_mat = scaler.inverse_transform(np.concatenate(all_trues, axis=0))
    mask_mat = np.concatenate(all_masks, axis=0)  # 【新增】合并掩码


    abs_errors = np.abs(pred_mat - true_mat)
    max_abs_err_per_feature = np.max(abs_errors, axis=0)
    
    if max_abs_error_threshold is not None:
        valid_features = max_abs_err_per_feature <= max_abs_error_threshold
        filtered_count = np.sum(~valid_features)

        
        pred_mat_filtered = pred_mat[:, valid_features]
        true_mat_filtered = true_mat[:, valid_features]
        mask_mat_filtered = mask_mat[:, valid_features]  # 【新增】过滤掩码
    else:
        pred_mat_filtered = pred_mat
        true_mat_filtered = true_mat
        mask_mat_filtered = mask_mat

    # 【修改】计算所有量测的误差
    true_all = true_mat_filtered.flatten()
    pred_all = pred_mat_filtered.flatten()
    
    rmse_full = np.sqrt(mean_squared_error(true_all, pred_all))
    mae_full = mean_absolute_error(true_all, pred_all)
    
    print(f"\n📊 变拓扑微调重构性能 (共 {len(true_all)} 个数据点):")
    print(f"   传统统计指标:")
    print(f"      MSE  : {rmse_full ** 2:.6f}")
    print(f"      RMSE : {rmse_full:.6f}")
    print(f"      MAE  : {mae_full:.6f}")

    pid_true = calculate_pid_metric(true_mat, A_inc, num_buses, num_lines)
    pid_recovered = calculate_pid_metric(pred_mat, A_inc, num_buses, num_lines)
    pid_improvement = (pid_true - pid_recovered) / pid_true * 100 if pid_true > 0 else 0

    print(f"\n   物理一致性指标 (PID):")
    print(f"      真实数据 PID: {pid_true:.6f}")
    print(f"      重构数据 PID: {pid_recovered:.6f}")
    print(f"      PID 改善率  : {pid_improvement:.2f}%")

    # 【新增】打印单样本详细统计
    if len(pred_mat_filtered) > 0:
        sample_idx = 0
        y_true_sample = true_mat_filtered[sample_idx]
        y_pred_sample = pred_mat_filtered[sample_idx]
        abs_errors_sample = np.abs(y_true_sample - y_pred_sample)
        
        max_abs_error = np.max(abs_errors_sample)
        max_error_idx = np.argmax(abs_errors_sample)
        min_abs_error = np.min(abs_errors_sample)
        mean_abs_error = np.mean(abs_errors_sample)
        
        print(f"\n📋 PG-WGAIN-Target 单样本详细统计 (样本索引={sample_idx}):")
        print(f"   最大绝对误差：{max_abs_error:.6f} (特征索引：{max_error_idx})")
        print(f"   最小绝对误差：{min_abs_error:.6f}")
        print(f"   平均绝对误差：{mean_abs_error:.6f}")
        print(f"   误差标准差：  {np.std(abs_errors_sample):.6f}")
        print(f"   误差中位数：  {np.median(abs_errors_sample):.6f}")

    # 【注释】不生成图像，不保存Excel
    # save_single_sample_to_excel(pred_matrix=pred_mat_filtered, true_matrix=true_mat_filtered, sample_idx=0, method_name='PG_WGAIN_Target')


def visualize_gain_result(netG, test_loader, scaler, device, mask_rate=0.2, data_dir=SOURCE_NORMAL_DIR, tag=None, max_abs_error_threshold=None):
    print(f"\n>>> 正在生成 PG-WGAIN 对比图 (全维度评估)...")
    netG.eval()


    A_inc, num_buses, num_lines = load_grid_topology(data_dir, tag=tag)
    A_inc = A_inc.astype(np.float32)

    all_preds = []
    all_trues = []
    all_masks = []  # 【新增】保存掩码信息

    with torch.no_grad():
        for batch_x, in test_loader:
            B, Dim = batch_x.shape
            mask = (torch.rand(B, Dim).to(device) > mask_rate).float()

            x_generated = netG(batch_x, mask)
            x_imputed = batch_x * mask + x_generated * (1 - mask)

            all_preds.append(x_imputed.cpu().numpy())
            all_trues.append(batch_x.cpu().numpy())
            all_masks.append(mask.cpu().numpy())  # 【新增】保存掩码

    pred_mat = scaler.inverse_transform(np.concatenate(all_preds, axis=0))
    true_mat = scaler.inverse_transform(np.concatenate(all_trues, axis=0))
    mask_mat = np.concatenate(all_masks, axis=0)  # 【新增】合并掩码

    abs_errors = np.abs(pred_mat - true_mat)
    max_abs_err_per_feature = np.max(abs_errors, axis=0)
    
    if max_abs_error_threshold is not None:
        valid_features = max_abs_err_per_feature <= max_abs_error_threshold
        filtered_count = np.sum(~valid_features)

        pred_mat_filtered = pred_mat[:, valid_features]
        true_mat_filtered = true_mat[:, valid_features]
        mask_mat_filtered = mask_mat[:, valid_features]  # 【新增】过滤掩码
    else:
        pred_mat_filtered = pred_mat
        true_mat_filtered = true_mat
        mask_mat_filtered = mask_mat

    # 【修改】计算所有量测的误差
    true_all = true_mat_filtered.flatten()
    pred_all = pred_mat_filtered.flatten()
    
    rmse_full = np.sqrt(mean_squared_error(true_all, pred_all))
    mae_full = mean_absolute_error(true_all, pred_all)
    
    print(f"\n PG-WGAIN 重构性能 (共 {len(true_all)} 个数据点):")
    print(f"   传统统计指标:")
    print(f"      MSE  : {rmse_full ** 2:.6f}")
    print(f"      RMSE : {rmse_full:.6f}")
    print(f"      MAE  : {mae_full:.6f}")

    pid_true = calculate_pid_metric(true_mat, A_inc, num_buses, num_lines)
    pid_recovered = calculate_pid_metric(pred_mat, A_inc, num_buses, num_lines)
    pid_improvement = (pid_true - pid_recovered) / pid_true * 100 if pid_true > 0 else 0

    print(f"\n   物理一致性指标 (PID):")
    print(f"      真实数据 PID:    {pid_true:.6f}")
    print(f"      重构数据 PID:    {pid_recovered:.6f}")
    print(f"      PID 变化率：     {pid_improvement:.2f}%")

    # 【新增】打印单样本详细统计
    if len(pred_mat_filtered) > 0:
        sample_idx = 0
        y_true_sample = true_mat_filtered[sample_idx]
        y_pred_sample = pred_mat_filtered[sample_idx]
        abs_errors_sample = np.abs(y_true_sample - y_pred_sample)
        
        max_abs_error = np.max(abs_errors_sample)
        max_error_idx = np.argmax(abs_errors_sample)
        min_abs_error = np.min(abs_errors_sample)
        mean_abs_error = np.mean(abs_errors_sample)
        
        print(f"\n📋 PG-WGAIN 单样本详细统计 (样本索引={sample_idx}):")
        print(f"   最大绝对误差：{max_abs_error:.6f} (特征索引：{max_error_idx})")
        print(f"   最小绝对误差：{min_abs_error:.6f}")
        print(f"   平均绝对误差：{mean_abs_error:.6f}")
        print(f"   误差标准差：  {np.std(abs_errors_sample):.6f}")
        print(f"   误差中位数：  {np.median(abs_errors_sample):.6f}")

    # 【注释】不生成图像，不保存Excel
    # save_single_sample_to_excel(pred_matrix=pred_mat_filtered, true_matrix=true_mat_filtered, sample_idx=0, method_name='PG-WGAIN')


# 【已删除】save_single_sample_to_excel 函数已移除


if __name__ == "__main__":
    # ==================== 根据 RUN_MODE 选择运行模式 ====================
    
    if RUN_MODE == 'stage1':
        # ==================== 只运行第一阶段（源域预训练）====================
        print("\n" + "="*60)
        print("     仅运行第一阶段：源域预训练")
        print("="*60)
        
        best_model_path, source_scaler, netG_current = main(data_dir=SOURCE_NORMAL_DIR, tag=SOURCE_TAG)
        print(f"\n✅ 第一阶段完成！模型已保存至: {best_model_path}")
    
    elif RUN_MODE == 'stage2':
        # ==================== 只运行第二阶段（目标域微调）====================
        print("\n" + "="*60)
        print("     仅运行第二阶段：目标域微调")
        print("="*60)
        
        # 确定要加载的预训练模型
        if STAGE2_PRETRAINED_MODEL_PATH is not None:
            # 用户指定了预训练模型
            if not os.path.exists(STAGE2_PRETRAINED_MODEL_PATH):
                raise FileNotFoundError(f"指定的预训练模型文件不存在: {STAGE2_PRETRAINED_MODEL_PATH}")
            pretrained_model_path = STAGE2_PRETRAINED_MODEL_PATH
            print(f"✅ 加载指定预训练模型: {pretrained_model_path}")
        else:
            # 自动查找最新的预训练模型
            pretrained_files = glob.glob('pg_wgain_best_*.pth')
            if not pretrained_files:
                raise FileNotFoundError(
                    f"未找到预训练模型文件 (pg_wgain_best_*.pth)！\n"
                    f"请先运行第一阶段（设置 RUN_MODE='stage1'）进行预训练。"
                )
            pretrained_model_path = max(pretrained_files, key=os.path.getmtime)
            print(f"✅ 加载最新预训练模型: {pretrained_model_path}")
        
        # 执行第二阶段：目标域微调
        if os.path.exists(TARGET_NORMAL_DIR):
            netG_current, scaler_current, finetuned_model_path = main_finetune(
                pretrained_model_path=pretrained_model_path,
                target_data_dir=TARGET_NORMAL_DIR,
                tag=TARGET_TAG,
                scaler=None  # 微调阶段会自己加载 scaler
            )
            print(f"\n✅ 第二阶段完成！微调模型已保存至: {finetuned_model_path}")
        else:
            print(f"\n❌ 未检测到 {TARGET_NORMAL_DIR} 文件夹，无法进行微调。")
    
    elif RUN_MODE == 'stage3':
        # ==================== 只运行第三阶段（攻击恢复测试）====================
        print("\n" + "="*60)
        print("     仅运行第三阶段：攻击恢复测试")
        print("="*60)
        
        # 确定要加载的模型文件
        if STAGE3_MODEL_PATH is not None:
            # 用户指定了模型文件
            if not os.path.exists(STAGE3_MODEL_PATH):
                raise FileNotFoundError(f"指定的模型文件不存在: {STAGE3_MODEL_PATH}")
            model_path = STAGE3_MODEL_PATH
            print(f"✅ 加载指定模型: {model_path}")
            # 根据文件名判断是否是微调模型
            use_finetuned = 'finetuned' in os.path.basename(model_path)
        else:
            # 自动查找最新的模型
            finetuned_files = glob.glob('pg_wgain_finetuned_*.pth')
            pretrained_files = glob.glob('pg_wgain_best_*.pth')
            
            if finetuned_files:
                # 有微调模型，使用微调模型
                model_path = max(finetuned_files, key=os.path.getmtime)
                print(f"✅ 加载最新微调模型: {model_path}")
                use_finetuned = True
            elif pretrained_files:
                # 没有微调模型，使用预训练模型
                model_path = max(pretrained_files, key=os.path.getmtime)
                print(f"⚠️  未找到微调模型，使用最新预训练模型: {model_path}")
                use_finetuned = False
            else:
                raise FileNotFoundError(
                    f"未找到模型文件 (pg_wgain_best_*.pth 或 pg_wgain_finetuned_*.pth)！\n"
                    f"请先运行完整流程（设置 RUN_MODE='all'）进行训练。"
                )
        
        # 加载模型
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        except (TypeError, Exception) as e:
            # 如果 weights_only=True 失败（例如包含 sklearn 对象），回退到 weights_only=False
            if 'weights_only' in str(e) or 'UnpicklingError' in str(type(e).__name__):
                print(f"⚠️  使用 weights_only=True 加载失败，回退到 weights_only=False")
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            else:
                raise
        
        input_dim = checkpoint['input_dim']
        scaler_current = checkpoint.get('scaler', None)
        
        # 如果模型中没有保存 scaler，需要从数据重新加载
        if scaler_current is None:
            if use_finetuned and os.path.exists(TARGET_NORMAL_DIR):
                print("⚠️  模型文件中未保存 Scaler，正在从目标域数据重新加载...")
                _, scaler_current, _ = load_and_normalize_data(TARGET_NORMAL_DIR, tag=TARGET_TAG)
            else:
                print("⚠️  模型文件中未保存 Scaler，正在从源域数据重新加载...")
                _, scaler_current, _ = load_and_normalize_data(SOURCE_NORMAL_DIR, tag=SOURCE_TAG)
        
        # 创建模型实例并加载权重
        netG_current = GAIN_Generator(input_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        netG_current.load_state_dict(checkpoint['netG_state_dict'])
        
        # 执行第三阶段：攻击恢复测试
        if os.path.exists(ATTACK_TEST_DIR):
            evaluate_attack_recovery(netG_current, scaler_current, target_data_dir=ATTACK_TEST_DIR, tag=ATTACK_TAG, max_abs_error_threshold=None)
        else:
            print(f"\n❌ 未检测到 {ATTACK_TEST_DIR} 文件夹，无法进行攻击恢复测试。")
    
    elif RUN_MODE == 'all':
        # ==================== 运行全部三个阶段 ====================
        print("\n" + "="*60)
        print("     运行完整流程：预训练 → 微调 → 攻击恢复")
        print("="*60)
        
        # 1. 源域正常样本预训练
        best_model_path, source_scaler, netG_current = main(data_dir=SOURCE_NORMAL_DIR, tag=SOURCE_TAG)

        # 2. 目标域少量正常样本微调。没有该文件夹时，直接使用源域预训练模型。
        if os.path.exists(TARGET_NORMAL_DIR):
            netG_current, scaler_current, finetuned_model_path = main_finetune(
                pretrained_model_path=best_model_path,
                target_data_dir=TARGET_NORMAL_DIR,
                tag=TARGET_TAG,
                scaler=source_scaler
            )
        else:
            print(f"\n未检测到 {TARGET_NORMAL_DIR} 文件夹，跳过目标域微调，直接使用源域预训练模型。")
            scaler_current = source_scaler

        # 3. 攻击和定位数据只用于测试阶段
        if os.path.exists(ATTACK_TEST_DIR):
            evaluate_attack_recovery(netG_current, scaler_current, target_data_dir=ATTACK_TEST_DIR, tag=ATTACK_TAG, max_abs_error_threshold=None)
        else:
            print(f"\n未检测到 {ATTACK_TEST_DIR} 文件夹，跳过攻击恢复测试。")
    
    else:
        raise ValueError(f"未知的运行模式: {RUN_MODE}。请使用 'all', 'stage1', 'stage2', 或 'stage3'")
