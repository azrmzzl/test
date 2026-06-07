import os
import glob
import time
import copy
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 0. 与主代码保持一致的路径、标签和运行模式配置
# ==========================================
RUN_MODE = 'all'  # 'all': 训练 + 攻击恢复测试；'train': 只训练；'test': 只测试

SOURCE_NORMAL_DIR = 'dataset_split_source_normal'
TARGET_NORMAL_DIR = 'dataset_split_target_normal'
ATTACK_TEST_DIR = 'dataset_split_attack_test'

SOURCE_TAG = '正常运行_1_35000时刻_正常数据波动'
TARGET_TAG = '支路12-13断路_1_35000时刻_正常数据波动'
ATTACK_TAG = '支路12-13断路_1_1000时刻波动'

# 对比模型不设置“源域预训练—目标域微调”流程。默认直接用源域正常样本训练。
# 如果希望对比模型直接使用少量目标域正常样本训练，只需改为：
# TRAIN_NORMAL_DIR = TARGET_NORMAL_DIR
# TRAIN_TAG = TARGET_TAG
TRAIN_NORMAL_DIR = SOURCE_NORMAL_DIR
TRAIN_TAG = SOURCE_TAG
TRAIN_SAMPLES = 20000  # None 表示使用全部样本

# 测试阶段使用与主代码一致的攻击样本、Clean_Samples 和 Recovery_Mask / Labels_Meas。
TEST_DIR = ATTACK_TEST_DIR
TEST_TAG = ATTACK_TAG

MODEL_PATH = None  # RUN_MODE='test' 时，None 表示自动读取当前目录下最新模型

SEED = 2025
NUM_BUSES = 57


def set_random_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_random_seed(SEED)


# ==========================================
# 1. 文件匹配、数据读取与归一化：保持与主代码一致
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


def load_and_normalize_data(data_dir, tag=None, max_samples=None):
    raw_data, tag, sample_file = load_samples_from_dir(data_dir, tag=tag)
    if max_samples is not None and max_samples < len(raw_data):
        print(f'⚠️  样本数量限制: {len(raw_data)} → {max_samples} (使用前{max_samples}个样本)')
        raw_data = raw_data[:max_samples]
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(raw_data)
    return data_norm.astype(np.float32), scaler, tag


def load_grid_topology(data_dir, tag=None, num_buses=NUM_BUSES):
    tag = _normalize_tag(tag)
    topo_file = None
    if tag is not None:
        tagged_file = os.path.join(data_dir, f'Topology_Branch_{tag}.mat')
        if os.path.exists(tagged_file):
            topo_file = tagged_file
        else:
            fallback = os.path.join(data_dir, 'Topology_Branch.mat')
            if os.path.exists(fallback):
                topo_file = fallback
                print(f'⚠️ 未找到指定拓扑文件 {tagged_file}，改用 {fallback}')
    if topo_file is None:
        topo_file = os.path.join(data_dir, 'Topology_Branch.mat')
    if not os.path.exists(topo_file):
        raise FileNotFoundError(f'未找到拓扑文件: {topo_file}')

    branch_data = sio.loadmat(topo_file)['branch_data']
    num_lines = branch_data.shape[0]
    A_inc = np.zeros((num_buses, num_lines), dtype=np.float32)
    for l_idx in range(num_lines):
        status = branch_data[l_idx, 2] if branch_data.shape[1] > 2 else 1.0
        if status == 1.0:
            b_from = int(branch_data[l_idx, 0]) - 1
            b_to = int(branch_data[l_idx, 1]) - 1
            if 0 <= b_from < num_buses and 0 <= b_to < num_buses:
                A_inc[b_from, l_idx] = 1.0
                A_inc[b_to, l_idx] = -1.0
    print(f'>>> 读取拓扑文件: {topo_file}')
    return A_inc, num_buses, num_lines


def _load_attack_test_data(data_dir, tag=None):
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
    if reconstructed_data.ndim != 2:
        reconstructed_data = reconstructed_data.reshape(reconstructed_data.shape[0], -1)

    total_features = reconstructed_data.shape[1]
    need_dim = 171 + 2 * num_lines
    if total_features < need_dim:
        print(f'⚠️  数据维度 {total_features} 小于 PID 计算所需维度 {need_dim}，跳过 PID 计算')
        return np.nan

    pid_values = []
    for i in range(reconstructed_data.shape[0]):
        sample = reconstructed_data[i]
        bus_p = sample[57:114]
        bus_q = sample[114:171]
        branch_p = sample[171:171 + num_lines]
        branch_q = sample[171 + num_lines:171 + 2 * num_lines]
        p_residual = bus_p - A_inc @ branch_p
        q_residual = bus_q - A_inc @ branch_q
        pid = (np.sum(np.abs(p_residual)) + np.sum(np.abs(q_residual))) / num_buses
        pid_values.append(pid)
    return float(np.mean(pid_values))


def save_attack_recovery_result(method_name, pred_mat, attack_data, target_data_dir, tag):
    sample_mat = attack_data['samples']
    clean_mat = attack_data['clean_samples']
    label_mat = attack_data['label_mat']
    recovery_mask = attack_data['recovery_mask']

    A_inc_np, num_buses, num_lines = load_grid_topology(target_data_dir, tag=tag)

    if clean_mat is not None:
        pid_true = calculate_pid_metric(clean_mat, A_inc_np, num_buses, num_lines)
        pid_rec = calculate_pid_metric(pred_mat, A_inc_np, num_buses, num_lines)
        rpid = (abs(pid_rec - pid_true) / pid_true * 100) if pid_true and pid_true > 0 else 0.0
        pid_attacked = calculate_pid_metric(sample_mat, A_inc_np, num_buses, num_lines)
    else:
        pid_true = None
        pid_rec = calculate_pid_metric(pred_mat, A_inc_np, num_buses, num_lines)
        pid_attacked = calculate_pid_metric(sample_mat, A_inc_np, num_buses, num_lines)
        rpid = None

    print(f'📊 定向恢复性能测试结果')
    print(f'   方法名称: {method_name}')
    print(f'   测试文件标识: {tag}')
    print(f'   总测试样本数: {sample_mat.shape[0]}')

    if label_mat is not None:
        attacked_pos = label_mat.astype(bool)
        print(f'   平均每个样本被篡改的量测数: {np.mean(label_mat.sum(axis=1)):.2f} 个')
    else:
        attacked_pos = recovery_mask < 0.5
        print(f'   平均每个样本待恢复量测数: {np.mean(np.sum(attacked_pos, axis=1)):.2f} 个')

    print(f'   [攻击后] PID: {pid_attacked:.6f}')
    if pid_true is not None:
        print(f'   [原始正常] PID (基准): {pid_true:.6f}')
        print(f'   [恢复后] PID: {pid_rec:.6f}')
        print(f'   RPID (相对变化率): {rpid:.2f}%')

    metrics_to_save = {
        'PID_Attacked': pid_attacked,
        'PID_True': pid_true if pid_true is not None else np.nan,
        'PID_Recovered': pid_rec,
        'RPID_Percent': rpid if rpid is not None else np.nan,
    }

    if clean_mat is not None:
        metrics_attack = masked_regression_metrics(clean_mat, sample_mat, attacked_pos)
        metrics_recover = masked_regression_metrics(clean_mat, pred_mat, attacked_pos)
        metrics_full = masked_regression_metrics(clean_mat, pred_mat, np.ones_like(clean_mat, dtype=bool))

        if metrics_attack is not None:
            print(f'\n   恢复前误差:')
            print(f"      RMSE: {metrics_attack['RMSE']:.6f}")
            print(f"      MAE : {metrics_attack['MAE']:.6f}")
        if metrics_full is not None:
            print(f'\n   恢复后误差:')
            print(f"      RMSE: {metrics_full['RMSE']:.6f}")
            print(f"      MAE : {metrics_full['MAE']:.6f}")

        if label_mat is not None:
            mask_ratio_per_sample = label_mat.sum(axis=1) / label_mat.shape[1]
            low_mask_samples = mask_ratio_per_sample < 0.3
            if np.any(low_mask_samples):
                low_mask_count = int(np.sum(low_mask_samples))
                print(f'\n📊 掩码节点<30%的样本统计 (共{low_mask_count}个样本):')
                clean_low = clean_mat[low_mask_samples]
                pred_low = pred_mat[low_mask_samples]
                attacked_low = attacked_pos[low_mask_samples]
                metrics_low_attack = masked_regression_metrics(clean_low, pred_low, attacked_low)
                metrics_low_full = masked_regression_metrics(clean_low, pred_low, np.ones_like(clean_low, dtype=bool))
                if metrics_low_full is not None:
                    print(f'   恢复后误差:')
                    print(f"      RMSE: {metrics_low_full['RMSE']:.6f}")
                    print(f"      MAE : {metrics_low_full['MAE']:.6f}")
                metrics_to_save.update({
                    'LowMask_Count': low_mask_count,
                    'LowMask_AttackPos_RMSE': metrics_low_attack['RMSE'] if metrics_low_attack else np.nan,
                    'LowMask_AttackPos_MAE': metrics_low_attack['MAE'] if metrics_low_attack else np.nan,
                    'LowMask_Full_RMSE': metrics_low_full['RMSE'] if metrics_low_full else np.nan,
                    'LowMask_Full_MAE': metrics_low_full['MAE'] if metrics_low_full else np.nan,
                })
            else:
                print(f'\n⚠️  没有掩码节点<30%的样本')
                metrics_to_save.update({
                    'LowMask_Count': 0,
                    'LowMask_AttackPos_RMSE': np.nan,
                    'LowMask_AttackPos_MAE': np.nan,
                    'LowMask_Full_RMSE': np.nan,
                    'LowMask_Full_MAE': np.nan,
                })

        metrics_to_save.update({
            'Attack_Pos_RMSE_Before': metrics_attack['RMSE'] if metrics_attack else np.nan,
            'Attack_Pos_MAE_Before': metrics_attack['MAE'] if metrics_attack else np.nan,
            'Attack_Pos_RMSE_After': metrics_recover['RMSE'] if metrics_recover else np.nan,
            'Attack_Pos_MAE_After': metrics_recover['MAE'] if metrics_recover else np.nan,
            'Full_RMSE_After': metrics_full['RMSE'] if metrics_full else np.nan,
            'Full_MAE_After': metrics_full['MAE'] if metrics_full else np.nan,
        })
    else:
        print('\n   未读取到 Clean_Samples，跳过 MAE 和 RMSE。')

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    safe_method = method_name.replace('-', '_').replace(' ', '_')
    out_file = os.path.join(target_data_dir, f'Recovered_Data_{safe_method}_{tag}_{timestamp}.mat')
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
    print(f'\n恢复结果已保存至: {out_file}')
    return out_file, metrics_to_save


def find_latest_model(prefix):
    files = glob.glob(f'{prefix}_*.pth')
    if not files:
        raise FileNotFoundError(f'未找到模型文件: {prefix}_*.pth，请先运行 RUN_MODE="train" 或 RUN_MODE="all"')
    return max(files, key=os.path.getmtime)


def load_checkpoint(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


METHOD_NAME = 'DAE'
MODEL_PREFIX = 'dae_baseline'

# ==========================================
# 2. DAE 模型定义
# ==========================================
class DenoisingAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128):
        super(DenoisingAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def evaluate_normal_reconstruction(model, data_loader, scaler, device, mask_rate, data_dir, tag):
    print(f'\n>>> 正在评估 {METHOD_NAME} 正常样本随机遮掩重构性能...')
    model.eval()
    set_random_seed(SEED)
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, in data_loader:
            mask = (torch.rand_like(batch_x) > mask_rate).float()
            outputs = model(batch_x * mask)
            x_imputed = batch_x * mask + outputs * (1.0 - mask)
            all_preds.append(x_imputed.cpu().numpy())
            all_trues.append(batch_x.cpu().numpy())
    pred_mat = scaler.inverse_transform(np.concatenate(all_preds, axis=0))
    true_mat = scaler.inverse_transform(np.concatenate(all_trues, axis=0))
    mask_eval = np.random.rand(*true_mat.shape) > mask_rate
    rmse = np.sqrt(mean_squared_error(true_mat[mask_eval], pred_mat[mask_eval]))
    mae = mean_absolute_error(true_mat[mask_eval], pred_mat[mask_eval])
    A_inc, num_buses, num_lines = load_grid_topology(data_dir, tag=tag)
    pid_true = calculate_pid_metric(true_mat, A_inc, num_buses, num_lines)
    pid_rec = calculate_pid_metric(pred_mat, A_inc, num_buses, num_lines)
    print(f'\n📊 {METHOD_NAME} 重构性能 (正常样本随机遮掩):')
    print(f'   MSE  : {rmse ** 2:.6f}')
    print(f'   RMSE : {rmse:.6f}')
    print(f'   MAE  : {mae:.6f}')
    print(f'   真实数据 PID: {pid_true:.6f}')
    print(f'   重构数据 PID: {pid_rec:.6f}')


def train_model(data_dir=TRAIN_NORMAL_DIR, tag=TRAIN_TAG):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    BATCH_SIZE = 128
    EPOCHS = 1000
    LR = 1e-3
    MASK_RATE = 0.4
    PATIENCE = 100

    data_all, scaler, train_tag = load_and_normalize_data(data_dir, tag=tag, max_samples=TRAIN_SAMPLES)
    input_dim = data_all.shape[1]
    X_tensor = torch.FloatTensor(data_all).to(device)

    first_sample = X_tensor[0:1]
    remaining_data = X_tensor[1:]
    train_size = int(len(remaining_data) * 0.9)
    train_dataset = remaining_data[:train_size]
    val_dataset = remaining_data[train_size:]
    test_dataset = torch.cat([first_sample, val_dataset], dim=0)

    train_loader = DataLoader(TensorDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_dataset), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_dataset), batch_size=BATCH_SIZE, shuffle=False)

    model = DenoisingAE(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f'\n>>> 开始训练 {METHOD_NAME} Baseline...')
    best_state = None
    best_val = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, in train_loader:
            mask = (torch.rand_like(batch_x) > MASK_RATE).float()
            outputs = model(batch_x * mask)
            loss = torch.mean(((1.0 - mask) * (outputs - batch_x)) ** 2) / (torch.mean(1.0 - mask) + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_x, in val_loader:
                mask = (torch.rand_like(val_x) > MASK_RATE).float()
                outputs = model(val_x * mask)
                val_loss = torch.mean(((1.0 - mask) * (outputs - val_x)) ** 2) / (torch.mean(1.0 - mask) + 1e-8)
                val_total += val_loss.item()
                val_batches += 1
        avg_val = val_total / val_batches if val_batches else float('inf')

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{EPOCHS} | Train Loss: {total_loss / len(train_loader):.6f} | Val MSE: {avg_val:.6f}')
        if patience_counter >= PATIENCE:
            print(f'\n🛑 早停触发！最优epoch: {best_epoch}, 最优验证MSE: {best_val:.6f}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model_path = f'{MODEL_PREFIX}_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'scaler': scaler,
        'train_tag': train_tag,
        'best_epoch': best_epoch,
        'best_val_mse': best_val,
    }, model_path)
    print(f'\n💾 模型已保存至: {model_path}')
    evaluate_normal_reconstruction(model, test_loader, scaler, device, MASK_RATE, data_dir, train_tag)
    return model_path, model, scaler


def load_model_for_test(model_path, device):
    checkpoint = load_checkpoint(model_path, device)
    model = DenoisingAE(checkpoint['input_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['scaler']


def evaluate_attack_recovery(model, scaler, target_data_dir=TEST_DIR, tag=TEST_TAG):
    device = next(model.parameters()).device
    print(f'\n==========================================')
    print(f'     测试阶段：攻击样本定向恢复测试（{METHOD_NAME}）')
    print(f'==========================================')

    attack_data = _load_attack_test_data(target_data_dir, tag=tag)
    sample_mat = attack_data['samples']
    recovery_mask = attack_data['recovery_mask']
    tag = attack_data['tag']

    norm_samples = scaler.transform(sample_mat)
    x_tensor = torch.FloatTensor(norm_samples).to(device)
    mask_tensor = torch.FloatTensor(recovery_mask).to(device)

    model.eval()
    BATCH_SIZE = 128
    pred_batches = []
    with torch.no_grad():
        for i in range(0, x_tensor.shape[0], BATCH_SIZE):
            batch_x = x_tensor[i:i + BATCH_SIZE]
            batch_mask = mask_tensor[i:i + BATCH_SIZE]
            outputs = model(batch_x * batch_mask)
            imputed = batch_x * batch_mask + outputs * (1.0 - batch_mask)
            pred_batches.append(imputed)
    pred_norm = torch.cat(pred_batches, dim=0).cpu().numpy()
    pred_mat = scaler.inverse_transform(pred_norm)
    return save_attack_recovery_result(METHOD_NAME, pred_mat, attack_data, target_data_dir, tag)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if RUN_MODE in ['all', 'train']:
        model_path, model, scaler = train_model(TRAIN_NORMAL_DIR, TRAIN_TAG)
    else:
        model_path = MODEL_PATH if MODEL_PATH is not None else find_latest_model(MODEL_PREFIX)
        print(f'✅ 加载模型: {model_path}')
        model, scaler = load_model_for_test(model_path, device)
    if RUN_MODE in ['all', 'test']:
        if RUN_MODE == 'all':
            model, scaler = load_model_for_test(model_path, device)
        evaluate_attack_recovery(model, scaler, TEST_DIR, TEST_TAG)


if __name__ == '__main__':
    main()
