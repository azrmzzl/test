"""
可视化统一工具函数
用于 IEEE 57 节点系统数据恢复方法的性能评估和可视化对比

功能：
1. 绘制单样本全特征对比图（4 合 1）
2. 绘制时序对比图（统一风格）
3. 计算并打印详细统计信息
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings

# 忽略 mpld3 的警告
warnings.filterwarnings('ignore', category=UserWarning, module='mpld3')


def plot_single_sample_comparison(pred_matrix, true_matrix, sample_idx=0, method_name='Method', 
                                   save_path=None, show=True):
    """
    绘制单个样本的所有特征对比图（4 合 1）
    
    Parameters:
    - pred_matrix: 预测值矩阵 (N_samples, N_features)
    - true_matrix: 真实值矩阵 (N_samples, N_features)
    - sample_idx: 选择哪个样本（默认第 0 个）
    - method_name: 方法名称（用于标题显示）
    - save_path: 图片保存路径（可选）
    - show: 是否显示图片（默认 True）
    """
    print(f"\n>>> 正在生成 {method_name} 单样本全特征对比图...")
    
    # 选择样本
    if sample_idx >= len(true_matrix):
        sample_idx = 0
    
    y_true_sample = true_matrix[sample_idx]
    y_pred_sample = pred_matrix[sample_idx]
    
    n_features = len(y_true_sample)
    
    # 计算每个特征的误差
    feature_errors = np.abs(y_true_sample - y_pred_sample)
    rmse_per_feature = np.sqrt(np.mean((y_true_sample - y_pred_sample) ** 2))
    
    # 创建图形（2x2 布局）
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # === 子图 1: 所有特征的真实值 vs 恢复值对比 ===
    ax1 = axes[0, 0]
    x_axis = np.arange(n_features)
    ax1.plot(x_axis, y_true_sample, 'bo-', label='Ground Truth', markersize=4, linewidth=1.5, alpha=0.7)
    ax1.plot(x_axis, y_pred_sample, 'rs--', label=f'{method_name} Recovered', markersize=4, linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Feature Index', fontsize=12)
    ax1.set_ylabel('Value (p.u.)', fontsize=12)
    ax1.set_title(f'{method_name}: Single Sample ({sample_idx}) - All Features Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim([0, n_features - 1])
    
    # === 子图 2: 绝对误差分布 ===
    ax2 = axes[0, 1]
    colors = plt.cm.RdYlGn_r(feature_errors / np.max(feature_errors + 1e-8))
    ax2.bar(x_axis, feature_errors, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=np.mean(feature_errors), color='r', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(feature_errors):.6f}')
    ax2.set_xlabel('Feature Index', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title(f'Absolute Error per Feature (Sample {sample_idx})', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim([0, n_features - 1])
    
    # === 子图 3: 散点对比图 (真实值 vs 预测值) ===
    ax3 = axes[1, 0]
    ax3.scatter(y_true_sample, y_pred_sample, c='blue', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    min_val = min(np.min(y_true_sample), np.min(y_pred_sample))
    max_val = max(np.max(y_true_sample), np.max(y_pred_sample))
    padding = (max_val - min_val) * 0.05
    ax3.set_xlim([min_val - padding, max_val + padding])
    ax3.set_ylim([min_val - padding, max_val + padding])
    ax3.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Ground Truth', fontsize=12)
    ax3.set_ylabel(f'{method_name} Prediction', fontsize=12)
    ax3.set_title(f'True vs Predicted Values (Sample {sample_idx})', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_aspect('equal', 'box')
    
    # === 子图 4: 特征误差统计直方图 ===
    ax4 = axes[1, 1]
    ax4.hist(feature_errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.axvline(x=np.mean(feature_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(feature_errors):.6f}')
    ax4.axvline(x=np.std(feature_errors), color='g', linestyle='--', linewidth=2, label=f'Std: {np.std(feature_errors):.6f}')
    ax4.set_xlabel('Absolute Error', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title(f'Error Distribution Histogram (Sample {sample_idx})', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   图片已保存至：{save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # 打印该样本的详细统计信息
    print(f"\n📊 {method_name} 单样本 (索引={sample_idx}) 详细统计:")
    print(f"   特征数量：{n_features}")
    print(f"   平均绝对误差：{np.mean(feature_errors):.6f}")
    print(f"   最大绝对误差：{np.max(feature_errors):.6f} (特征索引：{np.argmax(feature_errors)})")
    print(f"   最小绝对误差：{np.min(feature_errors):.6f}")
    print(f"   误差标准差：{np.std(feature_errors):.6f}")
    print(f"   RMSE (全特征): {rmse_per_feature:.6f}")
    
    return {
        'mae': np.mean(feature_errors),
        'max_error': np.max(feature_errors),
        'min_error': np.min(feature_errors),
        'std': np.std(feature_errors),
        'rmse': rmse_per_feature,
        'n_features': n_features
    }


def plot_time_series_comparison(y_true, y_pred, method_name='Method', feature_idx=0, 
                                 plot_len=200, save_path=None, show=True):
    """
    绘制时序对比图（前 plot_len 个样本的指定特征）
    
    Parameters:
    - y_true: 真实值序列 (N_samples,) 或 (N_samples, N_features)
    - y_pred: 预测值序列 (N_samples,) 或 (N_samples, N_features)
    - method_name: 方法名称
    - feature_idx: 特征索引（如果是 2D 数据）
    - plot_len: 绘制前多少个样本
    - save_path: 图片保存路径
    - show: 是否显示图片
    """
    print(f"\n>>> 正在生成 {method_name} 时序对比图 (特征 {feature_idx})...")
    
    # 处理 2D 数据
    if y_true.ndim == 2:
        y_true = y_true[:, feature_idx]
        y_pred = y_pred[:, feature_idx]
    
    # 截取指定长度
    plot_len = min(plot_len, len(y_true))
    y_true_plot = y_true[:plot_len]
    y_pred_plot = y_pred[:plot_len]
    
    # 计算 RMSE
    rmse = np.sqrt(np.mean((y_true_plot - y_pred_plot) ** 2))
    
    # 创建图形
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_plot, 'b.-', label='Ground Truth', alpha=0.7, linewidth=1.5)
    plt.plot(y_pred_plot, 'r.--', label=f'{method_name} Reconstructed', alpha=0.7, linewidth=1.5)
    plt.title(f'{method_name} Time-Series Reconstruction (Feature {feature_idx}) - RMSE: {rmse:.4f}')
    plt.xlabel('Sample Index')
    plt.ylabel('Value (p.u.)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   图片已保存至：{save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"   时序 RMSE: {rmse:.6f}")
    return rmse


def calculate_comprehensive_metrics(pred_matrix, true_matrix, mask_flat=None, method_name='Method'):
    """
    计算综合评估指标（RMSE, MAE, MAPE, PID 等）
    
    Parameters:
    - pred_matrix: 预测值矩阵 (N_samples, N_features)
    - true_matrix: 真实值矩阵 (N_samples, N_features)
    - mask_flat: 掩码数组（只计算缺失部分的误差，如果为 None 则计算全局）
    - method_name: 方法名称
    
    Returns:
    - metrics: 包含所有指标的字典
    """
    print(f"\n>>> 正在计算 {method_name} 的综合评估指标...")
    
    # 如果提供了掩码，只计算缺失部分
    if mask_flat is not None:
        true_values = true_matrix[mask_flat]
        recovered_values = pred_matrix[mask_flat]
    else:
        true_values = true_matrix.flatten()
        recovered_values = pred_matrix.flatten()
    
    # 过滤掉接近 0 的值，防止 MAPE 计算异常
    epsilon = 1e-8
    valid_indices = np.abs(true_values) > epsilon
    true_valid = true_values[valid_indices]
    recovered_valid = recovered_values[valid_indices]
    
    # 计算传统统计指标
    rmse = np.sqrt(mean_squared_error(true_values, recovered_values))
    mae = mean_absolute_error(true_values, recovered_values)
    
    if len(true_valid) > 0:
        mape = mean_absolute_percentage_error(true_valid, recovered_valid) * 100
    else:
        mape = np.nan
    
    print(f"\n📊 {method_name} 重构性能:")
    print(f"   传统统计指标:")
    print(f"      MSE  : {rmse ** 2:.6f}")
    print(f"      RMSE : {rmse:.6f}")
    print(f"      MAE  : {mae:.6f}")
    print(f"      MAPE : {mape:.4f}%")
    
    metrics = {
        'mse': rmse ** 2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'n_points': len(true_values),
        'n_valid': len(true_valid)
    }
    
    return metrics


# 统一的随机种子设置
def set_random_seed(seed=2025):
    """
    设置统一的随机种子，确保不同方法之间的公平对比
    
    Parameters:
    - seed: 随机种子（默认 2025）
    """
    import torch
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f">>> 已设置统一随机种子：{seed}")


if __name__ == "__main__":
    # 测试示例
    print("=== 可视化统一工具函数测试 ===\n")
    
    # 生成假数据
    n_samples = 100
    n_features = 427
    true_data = np.random.randn(n_samples, n_features)
    pred_data = true_data + np.random.randn(n_samples, n_features) * 0.1
    
    # 测试单样本对比图
    plot_single_sample_comparison(pred_data, true_data, sample_idx=0, method_name='Test Method')
    
    # 测试时序对比图
    plot_time_series_comparison(true_data, pred_data, method_name='Test Method', feature_idx=0)
    
    # 测试综合指标计算
    calculate_comprehensive_metrics(pred_data, true_data, method_name='Test Method')
    
    print("\n=== 测试完成 ===")
