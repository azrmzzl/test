import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from fancyimpute import SoftImpute, IterativeSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
# 忽略 FutureWarning (未来版本弃用警告)
warnings.simplefilter(action='ignore', category=FutureWarning)

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
# 2. 矩阵补全求解器
# ==========================================
class MatrixCompletionSolver:
    def __init__(self, method='SoftImpute'):
        self.method = method

    def solve(self, X_incomplete):
        """
        X_incomplete: 缺失数据用 np.nan 填充的矩阵
        """
        print(f">>> 开始矩阵补全 (Method: {self.method})...")

        if self.method == 'SoftImpute':
            # SoftImpute 是最经典的矩阵补全算法
            # verbose=False 关闭啰嗦的日志
            solver = SoftImpute(verbose=True, min_value=np.min(X_incomplete), max_value=np.max(X_incomplete))
            X_filled = solver.fit_transform(X_incomplete)

        elif self.method == 'IterativeSVD':
            # 另一种基于 SVD 的迭代方法，速度通常比 SoftImpute 快
            solver = IterativeSVD(verbose=True)
            X_filled = solver.fit_transform(X_incomplete)

        else:
            raise ValueError("Unknown method")

        return X_filled


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
        X_true = np.dot(U, V)  # 生成秩为10的矩阵

    print(f"原始数据维度: {X_true.shape}")

    # 2. 构造缺失/攻击 (Masking)
    # 模拟 20% 的数据丢失
    missing_rate = 0.2
    mask = np.random.rand(*X_true.shape) < missing_rate

    # [关键步骤] 矩阵补全库通常要求缺失值必须是 np.nan，而不是 0
    X_incomplete = X_true.copy()
    X_incomplete[mask] = np.nan

    # 3. 运行矩阵补全
    # 你可以选 'SoftImpute' (精度高) 或 'IterativeSVD' (速度快)
    solver = MatrixCompletionSolver(method='SoftImpute')
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

    print(f"\n>>> 补全结果评估:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"MAPE: {mape:.4f}%")

    # 5. 可视化对比 (取前 100 个缺失点)
    plt.figure(figsize=(12, 5))
    plt.plot(true_values[:100], 'b.-', label='Ground Truth', alpha=0.7)
    plt.plot(recovered_values[:100], 'r.--', label='Matrix Completion (SoftImpute)', alpha=0.7)
    plt.legend()
    plt.title('Matrix Completion Recovery Result (First 100 Missing Points)')
    plt.xlabel('Sample Index')
    plt.ylabel('Value (p.u.)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()