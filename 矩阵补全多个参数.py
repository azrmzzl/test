import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from fancyimpute import SoftImpute, IterativeSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
# 忽略 FutureWarning (未来版本弃用警告)
warnings.simplefilter(action='ignore', category=FutureWarning)
from visualization_utils import plot_single_sample_comparison, plot_time_series_comparison, set_random_seed

# ==========================================
# 1. 数据加载类 (复用你之前的逻辑)
# ==========================================
def load_ieee57_data(data_dir='dataset_split'):
    print(f">>> 正在加载 IEEE 57 数据...")
    # 加载 Samples
    mat_data = sio.loadmat(os.path.join(data_dir, 'Samples_x_y_normal_operation_all.mat'))
    samples = mat_data['Samples']  # Shape: (Num_Samples, 427) 或者你的原始维度

    # 这里的 Samples 应该是 (样本数, 特征数) 的 2D 矩阵
    # 矩阵补全要求输入必须是 2D 矩阵
    return samples


# ==========================================
# 2. 矩阵补全求解器（增强版）
# ==========================================
class MatrixCompletionSolver:
    def __init__(self, method='SoftImpute', rank=None, max_iter=100, tol=1e-3, lambda_=None):
        self.method = method
        self.rank = rank  # 目标秩，None 表示自动估计
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛容差
        self.lambda_ = lambda_  # 正则化参数

    def solve(self, X_incomplete):
        """
        X_incomplete: 缺失数据用 np.nan 填充的矩阵
        """
        print(f">>> 开始矩阵补全 (Method: {self.method})...")

        if self.method == 'SoftImpute':
            # [性能提升关键参数]
            # lambda_: 正则化参数，控制低秩程度（越小拟合越好，但可能过拟合）
            # max_iter: 增加迭代次数可以提高精度
            # min_value/max_value: 限制输出范围，防止异常值

            lambda_param = self.lambda_ if self.lambda_ else 0.1 * np.nanmax(np.abs(X_incomplete))

            print(f"   Lambda: {lambda_param:.6f}, Max Iter: {self.max_iter}")

            solver = SoftImpute(
                verbose=True,
                min_value=np.nanmin(X_incomplete),
                max_value=np.nanmax(X_incomplete),
                max_iters=self.max_iter,
                shrinkage_value=lambda_param  # 🔴 关键修复：将 lambda_ 替换为 shrinkage_value
            )
            X_filled = solver.fit_transform(X_incomplete)

        elif self.method == 'IterativeSVD':
            # [性能提升关键参数]
            # rank: 低秩近似的目标秩（需要根据数据特性调整）
            # 通常选择矩阵维度的 1/5 到 1/10

            rank_param = self.rank if self.rank else min(20, min(X_incomplete.shape) // 3)

            print(f"   Target Rank: {rank_param}, Max Iter: {self.max_iter}")

            solver = IterativeSVD(
                verbose=True,
                rank=rank_param,
                max_iter=self.max_iter
            )
            X_filled = solver.fit_transform(X_incomplete)

        else:
            raise ValueError("Unknown method")

        return X_filled
# ==========================================
# 2. 辅助函数：PID计算
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
            bus_v = sample[:57]  # 节点电压
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

            # [调试] 打印第一个样本的详细信息
            if i == 0:
                print(f"\n🔍 PID 调试信息 (样本 0):")
                print(f"   母线 P 注入总和：{np.sum(np.abs(bus_p)):.6f}")
                print(f"   支路 P 潮流总和：{np.sum(np.abs(branch_p)):.6f}")
                print(f"   A_inc @ branch_p 总和：{np.sum(np.abs(A_inc @ branch_p)):.6f}")
                print(f"   P 残差总和：{np.sum(np.abs(p_residual)):.6f}")
                print(
                    f"   最大 P 残差节点：{np.argmax(np.abs(p_residual))}, 值：{p_residual[np.argmax(np.abs(p_residual))]:.6f}")

                # 检查是否方向反了
                p_check = bus_p + A_inc @ branch_p  # 如果是加法，说明方向反了
                print(f"   [检查] bus_p + A_inc@branch_p = {np.sum(np.abs(p_check)):.6f}")

        elif total_features == 228:
            # 简化结构：只有母线特征 (57×4)
            bus_data = sample[:228].reshape(57, 4)
            bus_p = bus_data[:, 1]
            bus_q = bus_data[:, 2]

            p_imbalance = np.sum(np.abs(bus_p))
            q_imbalance = np.sum(np.abs(bus_q))
            pid = (p_imbalance + q_imbalance) / num_buses

        else:
            if i == 0:
                print(f"⚠️  警告：无法解析的数据维度 {total_features}，跳过 PID 计算")
            return np.nan

        pid_values.append(pid)

    return np.mean(pid_values)


# ==========================================
# 3. 主程序
# ==========================================
def main():
    # 1. 加载完整数据 (Ground Truth)
    # 假设数据在这个路径
    try:
        X_true = load_ieee57_data()
    except FileNotFoundError:
        # 如果没有文件，生成假数据演示
        print("未找到数据文件，生成随机低秩矩阵演示...")
        U = np.random.randn(1000, 10)
        V = np.random.randn(10, 137)
        X_true = np.dot(U, V)  # 生成秩为 10 的矩阵

    print(f"原始数据维度：{X_true.shape}")

    # 2. 构造缺失/攻击 (Masking)
    # [性能提升关键参数 1] 缺失率设置
    # 建议测试多个缺失率：0.1, 0.2, 0.3, 0.5
    missing_rate = 0.2  # 可以改为 0.1, 0.3, 0.5 等进行对比
    mask = np.random.rand(*X_true.shape) < missing_rate

    # [关键步骤] 矩阵补全库通常要求缺失值必须是 np.nan，而不是 0
    X_incomplete = X_true.copy()
    X_incomplete[mask] = np.nan

    # 3. 运行矩阵补全
    # [性能提升关键参数 2] 方法选择和参数调优
    # 可选方案:
    #   - 'SoftImpute': 精度最高，适合小规模数据
    #   - 'IterativeSVD': 速度较快，适合大规模数据

    # 【推荐配置】根据不同场景调整以下参数:
    # 场景 1: 高精度需求 → SoftImpute + 小 lambda + 多迭代
    # 场景 2: 快速计算 → IterativeSVD + 合适 rank
    # 场景 3: 平衡性能 → SoftImpute + 中等参数

    solver = MatrixCompletionSolver(
        method='SoftImpute',  # 或 'IterativeSVD'
        rank=15,  # IterativeSVD 专用：目标秩 (10-30 之间)
        max_iter=200,  # 增加迭代次数 (100-500)
        tol=1e-7,  # 更严格的收敛条件 (1e-3 ~ 1e-5) # 越小收敛标准越严格
        lambda_=0.5  # SoftImpute 专用：正则化参数 (0.01-0.2) # 越小 → 拟合越好，但可能过拟合
        # 越大 → 更平滑，但可能欠拟合
    )
    X_recovered = solver.solve(X_incomplete)

    # 4. 评估结果 (只计算缺失部分的误差)
    # 提取缺失部分的真实值和恢复值
    true_values = X_true[mask]
    recovered_values = X_recovered[mask]

    # [新增] 过滤掉接近 0 的值，防止 MAPE 计算异常
    epsilon = 1e-8
    valid_indices = np.abs(true_values) > epsilon
    true_valid = true_values[valid_indices]
    recovered_valid = recovered_values[valid_indices]

    rmse = np.sqrt(mean_squared_error(true_values, recovered_values))
    mae = mean_absolute_error(true_values, recovered_values)

    # [新增] 计算 MAPE (平均绝对百分比误差)
    if len(true_valid) > 0:
        mape = mean_absolute_percentage_error(true_valid, recovered_valid) * 100
    else:
        mape = np.nan

    # [新增] 计算物理一致性指标 PID
    # 加载电网拓扑
    A_inc, num_buses, num_lines = load_grid_topology()
    A_inc = A_inc.astype(np.float32)

    pid_true = calculate_pid_metric(X_true, A_inc, num_buses, num_lines)
    pid_recovered = calculate_pid_metric(X_recovered, A_inc, num_buses, num_lines)
    pid_improvement = (pid_true - pid_recovered) / pid_true * 100 if pid_true > 0 else 0

    print(f"\n>>> 补全结果评估:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"\n   物理一致性指标 (PID):")
    print(f"      真实数据 PID:    {pid_true:.6f}")
    print(f"      重构数据 PID:    {pid_recovered:.6f}")
    print(f"      PID 变化率：     {pid_improvement:.2f}%")

    # [修改] 使用统一可视化工具函数
    # 1. 时序对比图
    plot_time_series_comparison(
        y_true=X_true, 
        y_pred=X_recovered, 
        method_name='Matrix Complet·ion',
        feature_idx=0, 
        plot_len=200
    )
    
    # 2. 单样本全特征对比图
    plot_single_sample_comparison(
        pred_matrix=X_recovered, 
        true_matrix=X_true, 
        sample_idx=0, 
        method_name='Matrix Completion'
    )


if __name__ == "__main__":
    main()