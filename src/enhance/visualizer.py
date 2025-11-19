# visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_gtnnwr_results(gtnnwr_model, save_path=None, show_plot=True):
    """
    为GTNNWR模型生成真实值vs预测值的散点图

    Parameters
    ----------
    gtnnwr_model : GTNNWR
        已经训练好并调用过result()方法的GTNNWR模型对象
    save_path : str, optional
        图片保存路径，如果为None则不保存
    show_plot : bool, optional
        是否显示图片，默认为True

    Returns
    -------
    dict
        包含各种评估指标的字典
    """

    # 检查模型是否已经训练并评估过
    if not hasattr(gtnnwr_model, '_test_diagnosis'):
        raise ValueError("请先调用 gtnnwr_model.result() 方法来获取诊断结果")

    # 从诊断对象中获取真实值和预测值
    test_diagnosis = gtnnwr_model._test_diagnosis

    # 获取真实值和预测值
    y_true = test_diagnosis._DIAGNOSIS__y_data.cpu().numpy().flatten()
    y_pred = test_diagnosis._DIAGNOSIS__y_pred.cpu().numpy().flatten()

    # 计算评估指标
    r2 = test_diagnosis.R2().item()
    rmse = test_diagnosis.RMSE().item()
    mae = test_diagnosis.MAE().item()
    aicc = test_diagnosis.AICc()

    # 创建散点图
    plt.figure(figsize=(10, 8))

    # 绘制散点
    plt.scatter(y_true, y_pred, alpha=0.6, s=30, c='royalblue',
                edgecolors='black', linewidth=0.5, label='预测点')

    # 绘制y=x参考线
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--',
             linewidth=2, label='y = x (完美预测线)')

    # 设置坐标轴标签和标题
    plt.xlabel('真实值', fontsize=14)
    plt.ylabel('预测值', fontsize=14)
    plt.title(f'{gtnnwr_model._modelName} 预测结果', fontsize=16, fontweight='bold')

    # 在右上角添加指标值文本框
    metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nAICc = {aicc:.4f}\n样本数 = {len(y_true)}'
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       alpha=0.8, edgecolor='gray'))

    # 设置网格和图例
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"散点图已保存至: {save_path}")

    # 显示图片
    if show_plot:
        plt.show()
    else:
        plt.close()

    # 返回评估指标
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'AICc': aicc,
        'n_samples': len(y_true)
    }

    return metrics


def plot_multiple_models_results(model_results_dict, save_path=None, show_plot=True):
    """
    绘制多个模型的对比散点图

    Parameters
    ----------
    model_results_dict : dict
        字典，键为模型名称，值为GTNNWR模型对象
    save_path : str, optional
        图片保存路径
    show_plot : bool, optional
        是否显示图片

    Returns
    -------
    dict
        包含所有模型评估指标的字典
    """

    n_models = len(model_results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    all_metrics = {}

    for idx, (model_name, model) in enumerate(model_results_dict.items()):
        if not hasattr(model, '_test_diagnosis'):
            raise ValueError(f"模型 {model_name} 请先调用 result() 方法")

        test_diagnosis = model._test_diagnosis
        y_true = test_diagnosis._DIAGNOSIS__y_data.cpu().numpy().flatten()
        y_pred = test_diagnosis._DIAGNOSIS__y_pred.cpu().numpy().flatten()

        r2 = test_diagnosis.R2().item()
        rmse = test_diagnosis.RMSE().item()
        mae = test_diagnosis.MAE().item()

        all_metrics[model_name] = {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'n_samples': len(y_true)
        }

        ax = axes[idx]
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, c='royalblue',
                   edgecolors='black', linewidth=0.5)

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        ax.set_xlabel('真实值', fontsize=12)
        ax.set_ylabel('预测值', fontsize=12)
        ax.set_title(f'{model_name}\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return all_metrics
