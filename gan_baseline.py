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
torch.manual_seed(2025)
np.random.seed(2025)
class GAN_Generator(nn.Module):
    """
    传统GAN生成器：从随机噪声生成完整数据
    """
    def __init__(self, latent_dim, output_dim):
        super(GAN_Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )

    def forward(self, z):
        return self.net(z)


class GAN_Critic(nn.Module):
    """
    WGAN的Critic（判别器）：输出无界实数（Wasserstein分数）
    """
    def __init__(self, input_dim):
        super(GAN_Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 去掉Sigmoid，输出无界实数
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 1. 数据加载 (展平 + 归一化)
# ==========================================
def load_and_normalize_data(data_dir='dataset_split'):
    print(f">>> 正在加载 IEEE 57 数据 (用于 GAN)...")
    try:
        mat_path = os.path.join(data_dir, 'Samples_x_y_normal_operation_all.mat')
        if not os.path.exists(mat_path):
            raw_data = np.random.rand(2000, 137 * 4)
        else:
            samples = sio.loadmat(mat_path)['Samples']
            # 展平
            if samples.ndim == 3:
                N, Nodes, Feats = samples.shape
                raw_data = samples.reshape(N, Nodes * Feats)
            else:
                raw_data = samples
    except:
        raw_data = np.random.rand(2000, 137 * 4)

    # GAN 通常也使用 [0,1] 归一化
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(raw_data)

    return data_norm.astype(np.float32), scaler


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

    数据结构 (427 维):
    - [0:57]    : 57 个节点电压 V
    - [57:114]  : 57 个注入有功 P_inj
    - [114:171] : 57 个注入无功 Q_inj
    - [171:251] : 80 个线路有功潮流 P_flow
    - [251:331] : 80 个线路无功潮流 Q_flow
    """
    if reconstructed_data.ndim == 1:
        reconstructed_data = reconstructed_data.reshape(1, -1)

    N_samples = reconstructed_data.shape[0]
    total_features = reconstructed_data.shape[1]

    pid_values = []

    for i in range(N_samples):
        sample = reconstructed_data[i]

        # 根据实际维度解析数据
        if total_features >= 331:
            # 标准结构：至少包含 V, P_inj, Q_inj, P_flow, Q_flow
            bus_p = sample[57:114]  # 注入有功
            bus_q = sample[114:171]  # 注入无功
            branch_p = sample[171:251]  # 线路有功潮流
            branch_q = sample[251:331]  # 线路无功潮流

            # 计算 KCL 残差：P_bus - A_inc @ P_branch
            p_residual = bus_p - A_inc @ branch_p
            q_residual = bus_q - A_inc @ branch_q

            # L1 范数（绝对值之和）
            p_imbalance = np.sum(np.abs(p_residual))
            q_imbalance = np.sum(np.abs(q_residual))

            # PID = (P 不平衡 + Q 不平衡) / 母线数
            pid = (p_imbalance + q_imbalance) / num_buses

        else:
            if i == 0:
                print(f"⚠️  警告：无法解析的数据维度 {total_features}，跳过 PID 计算")
            return np.nan

        pid_values.append(pid)

    return np.mean(pid_values)


# ==========================================
# 2. 训练主程序
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 参数
    BATCH_SIZE = 128
    EPOCHS = 500  # WGAN 需要训练久一点
    PATIENCE = 100  # 早停耐心值
    LR_G = 0.0001  # WGAN建议使用较小学习率
    LR_D = 0.0001
    LATENT_DIM = 100  # 噪声维度
    N_CRITIC = 5  # Critic训练频率（每5步C训练1步G）
    LAMBDA_GP = 10  # 梯度惩罚系数

    # 1. 准备数据
    data_all, scaler = load_and_normalize_data()
    input_dim = data_all.shape[1]

    X_tensor = torch.FloatTensor(data_all).to(device)

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

    # 2. 初始化模型
    netG = GAN_Generator(LATENT_DIM, input_dim).to(device)
    netC = GAN_Critic(input_dim).to(device)  # WGAN使用Critic

    optG = optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.9))
    optC = optim.Adam(netC.parameters(), lr=LR_D, betas=(0.5, 0.9))

    print(f"\n>>> 开始训练 WGAN (Baseline + 早停)...")
    print(f"    噪声维度: {LATENT_DIM}")
    print(f"    N_CRITIC: {N_CRITIC}, LAMBDA_GP: {LAMBDA_GP}")
    
    # 早停相关变量
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_netG_state = None

    # 梯度惩罚计算函数
    def calc_gradient_penalty(critic, real_data, fake_data):
        alpha = torch.rand(real_data.size(0), 1, device=device)
        alpha = alpha.expand_as(real_data)
        
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates.requires_grad_(True)
        
        critic_interpolates = critic(interpolates)
        
        gradients = autograd.grad(
            outputs=critic_interpolates, 
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates, device=device),
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    for epoch in range(EPOCHS):
        netG.train()
        netC.train()

        loss_c_log = 0
        loss_g_log = 0
        batches_processed = 0

        # 手动控制DataLoader实现N_CRITIC步的错频训练
        train_iter = iter(train_loader)
        while True:
            try:
                # ----------------------------------------
                # 步骤1：训练Critic N_CRITIC次
                # ----------------------------------------
                for _ in range(N_CRITIC):
                    batch_x = next(train_iter)[0]
                    B, Dim = batch_x.shape

                    # 生成假数据
                    z = torch.randn(B, LATENT_DIM).to(device)
                    fake_data = netG(z).detach()

                    # Critic打分
                    c_real = netC(batch_x)
                    c_fake = netC(fake_data)

                    # Wasserstein距离损失
                    loss_c_wasserstein = torch.mean(c_fake) - torch.mean(c_real)

                    # 梯度惩罚
                    gp = calc_gradient_penalty(netC, batch_x, fake_data)

                    # Critic总损失
                    loss_c = loss_c_wasserstein + LAMBDA_GP * gp

                    optC.zero_grad()
                    loss_c.backward()
                    optC.step()

                    loss_c_log += loss_c.item()

                # ----------------------------------------
                # 步骤2：训练Generator 1次
                # ----------------------------------------
                batch_x = next(train_iter)[0]
                B, Dim = batch_x.shape

                z = torch.randn(B, LATENT_DIM).to(device)
                fake_data = netG(z)

                c_fake = netC(fake_data)

                # Generator损失：骗过Critic（让Critic打低分）
                loss_g = -torch.mean(c_fake)

                optG.zero_grad()
                loss_g.backward()
                optG.step()

                loss_g_log += loss_g.item()
                batches_processed += 1

            except StopIteration:
                break

        if (epoch + 1) % 10 == 0:
            avg_c = loss_c_log / (batches_processed * N_CRITIC)
            avg_g = loss_g_log / batches_processed
            print(f"Epoch {epoch + 1}/{EPOCHS} | C Loss: {avg_c:.4f} | G Loss: {avg_g:.4f}")
        
        # 验证集评估
        netG.eval()
        netC.eval()
        val_loss_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for val_batch_x, in val_loader:
                z = torch.randn(val_batch_x.shape[0], LATENT_DIM).to(device)
                fake_data = netG(z)
                
                # 验证损失：WGAN的Generator损失
                c_fake = netC(fake_data)
                val_loss = -torch.mean(c_fake)
                
                val_loss_total += val_loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss_total / val_batches if val_batches > 0 else float('inf')
        netG.train()
        netC.train()
        
        # 早停逻辑
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_netG_state = netG.state_dict().copy()
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
    
    # 恢复最优模型
    if best_netG_state is not None:
        netG.load_state_dict(best_netG_state)
        print(f"\n✨ 已恢复最优模型 (epoch {best_epoch})")

    # 4. 可视化对比
    visualize_gan_result(netG, test_loader, scaler, device, LATENT_DIM)


# ==========================================
# 3. 可视化函数
# ==========================================
def visualize_gan_result(netG, test_loader, scaler, device, latent_dim):
    print(f"\n>>> 正在生成 GAN 对比图 (全维度评估)...")
    netG.eval()

    set_random_seed(2025)

    A_inc, num_buses, num_lines = load_grid_topology()
    A_inc = A_inc.astype(np.float32)

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_x, in test_loader:
            B, Dim = batch_x.shape
            # GAN生成数据（不使用掩码）
            z = torch.randn(B, latent_dim).to(device)
            x_generated = netG(z)

            all_preds.append(x_generated.cpu().numpy())
            all_trues.append(batch_x.cpu().numpy())

    # 反归一化
    pred_mat = scaler.inverse_transform(np.concatenate(all_preds, axis=0))
    true_mat = scaler.inverse_transform(np.concatenate(all_trues, axis=0))

    print(f"\n测试集总样本数：{pred_mat.shape[0]}")
    print(f"特征维度：{pred_mat.shape[1]}")
    print(f"ℹ️  注意：测试集包含第 1 个样本（索引 0），该样本未参与训练")

    # 计算评估指标
    # 对于GAN，直接比较所有特征
    true_values = true_mat.flatten()
    recovered_values = pred_mat.flatten()

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

    # 计算PID
    pid_true = calculate_pid_metric(true_mat, A_inc, num_buses, num_lines)
    pid_recovered = calculate_pid_metric(pred_mat, A_inc, num_buses, num_lines)
    pid_improvement = (pid_true - pid_recovered) / pid_true * 100 if pid_true > 0 else 0

    print(f"\n📊 GAN 重构性能 (全维度，共 {len(true_values)} 个数据点):")
    print(f"   传统统计指标:")
    print(f"      MSE  : {rmse ** 2:.6f}")
    print(f"      RMSE : {rmse:.6f}")
    print(f"      MAE  : {mae:.6f}")
    print(f"      MAPE : {mape:.4f}%")
    print(f"\n   物理一致性指标 (PID):")
    print(f"      真实数据 PID:    {pid_true:.6f}")
    print(f"      重构数据 PID:    {pid_recovered:.6f}")
    print(f"      PID 变化率：     {pid_improvement:.2f}%")

    # 可视化
    plot_time_series_comparison(
        y_true=true_mat, 
        y_pred=pred_mat, 
        method_name='GAN', 
        feature_idx=0, 
        plot_len=200
    )

    plot_single_sample_comparison(
        pred_matrix=pred_mat,
        true_matrix=true_mat,
        sample_idx=0,
        method_name='GAN'
    )

    save_single_sample_to_excel(
        pred_matrix=pred_mat,
        true_matrix=true_mat,
        sample_idx=0,
        method_name='GAN'
    )


def save_single_sample_to_excel(pred_matrix, true_matrix, sample_idx=0, method_name='GAN'):
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
        'Ground_Truth': [df['Ground_Truth'].mean(), df['Ground_Truth'].std(),
                         df['Ground_Truth'].max(), df['Ground_Truth'].min()],
        'Recovered': [df['Recovered'].mean(), df['Recovered'].std(),
                      df['Recovered'].max(), df['Recovered'].min()],
        'Absolute_Error': [df['Absolute_Error'].mean(), df['Absolute_Error'].std(),
                           df['Absolute_Error'].max(), df['Absolute_Error'].min()],
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
    main()
