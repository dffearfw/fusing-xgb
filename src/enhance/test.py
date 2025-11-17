import os
import sys
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from gnnwr import models, datasets, utils
import time
import psutil
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


def monitor_performance(step_name):
    """简单的性能监控"""
    memory = psutil.virtual_memory()
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"[性能监控] {step_name} - 内存使用: {memory_usage:.1f}MB, 系统内存: {memory.percent}%")


def debug_data_issues(data, x_column, y_column, spatial_column, station_column='station_id'):
    """详细的数据问题调试"""
    print("=== 数据调试信息 ===")
    print(f"原始数据形状: {data.shape}")

    # 1. 检查列是否存在
    all_required_columns = x_column + y_column + spatial_column + [station_column]
    missing_columns = [col for col in all_required_columns if col not in data.columns]
    if missing_columns:
        print(f"❌ 缺失列: {missing_columns}")
        print(f"可用列: {list(data.columns)}")
        return False

    print("✅ 所有必需列都存在")

    # 2. 检查站点数量
    unique_stations = data[station_column].nunique()
    print(f"站点数量: {unique_stations}")

    # 3. 检查缺失值
    print("\n=== 缺失值分析 ===")
    missing_info = data[all_required_columns].isnull().sum()
    total_missing = missing_info.sum()
    print(f"总缺失值: {total_missing}")

    if total_missing > 0:
        print("缺失值详情:")
        for col, missing_count in missing_info[missing_info > 0].items():
            print(f"  {col}: {missing_count} 个缺失值 ({missing_count / len(data):.1%})")

    # 4. 检查无穷值
    print("\n=== 无穷值检查 ===")
    numeric_cols = data[all_required_columns].select_dtypes(include=[np.number]).columns
    inf_count = 0
    for col in numeric_cols:
        if np.isinf(data[col]).any():
            inf_count += 1
            print(f"❌ 列 '{col}' 包含无穷值")

    if inf_count == 0:
        print("✅ 没有无穷值")

    # 5. 检查每个站点的数据量
    print("\n=== 站点数据量分布 ===")
    station_counts = data[station_column].value_counts()
    print(f"每个站点平均数据量: {station_counts.mean():.1f}")
    print(f"最小数据量: {station_counts.min()}")
    print(f"最大数据量: {station_counts.max()}")
    print(f"数据量少于5条的站点数: {(station_counts < 5).sum()}")

    return True


def robust_data_cleaning(data, x_column, y_column, spatial_column, station_column):
    """鲁棒的数据清洗"""
    print("开始数据清洗...")
    clean_data = data.copy()

    # 1. 检查缺失值
    all_columns = x_column + y_column + spatial_column + [station_column]
    missing_rates = clean_data[all_columns].isnull().mean()

    print("各列缺失率:")
    for col in all_columns:
        # 确保获取的是标量值
        rate = float(missing_rates[col])  # 转换为float确保是标量
        print(f"  {col}: {rate:.2%}")

    # 修复缺失值处理逻辑
    for col in all_columns:
        # 确保获取标量值
        rate = float(missing_rates[col])

        if rate > 0 and rate < 0.3:  # 缺失率低于30%
            if col in ['elevation', 'slope', 'aspect', 'X', 'Y']:  # 数值型特征
                median_val = clean_data[col].median()
                if not pd.isna(median_val):
                    clean_data[col].fillna(median_val, inplace=True)
                else:
                    clean_data[col].fillna(0, inplace=True)
            elif col in ['doy', 'year', 'month']:  # 时间特征
                mode_vals = clean_data[col].mode()
                if len(mode_vals) > 0 and not pd.isna(mode_vals.iloc[0]):
                    clean_data[col].fillna(mode_vals.iloc[0], inplace=True)
                else:
                    clean_data[col].fillna(0, inplace=True)
            else:  # 其他特征
                median_val = clean_data[col].median()
                if not pd.isna(median_val):
                    clean_data[col].fillna(median_val, inplace=True)
                else:
                    clean_data[col].fillna(0, inplace=True)
        elif rate >= 0.3:  # 缺失率过高
            print(f"⚠️ 列 {col} 缺失率过高 ({rate:.2%})，考虑删除")

    # 2. 移除仍有缺失值的行
    initial_rows = len(clean_data)
    clean_data = clean_data.dropna(subset=all_columns)
    removed_rows = initial_rows - len(clean_data)
    print(f"移除 {removed_rows} 个仍有缺失值的行")

    # 3. 检查并处理无穷大值
    numeric_columns = clean_data[x_column + y_column].select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        inf_mask = np.isinf(clean_data[numeric_columns]).any(axis=1)
        if inf_mask.any():
            print(f"移除 {inf_mask.sum()} 个包含无穷大值的行")
            clean_data = clean_data[~inf_mask]

    # 4. 检查站点数据量
    station_counts = clean_data[station_column].value_counts()
    valid_stations = station_counts[station_counts >= 3].index  # 至少3个样本
    clean_data = clean_data[clean_data[station_column].isin(valid_stations)]
    print(f"移除数据量少于3的站点，剩余 {len(valid_stations)} 个站点")

    # 5. 检查特征值范围
    print("\n特征值范围:")
    for col in x_column + y_column:
        if col in clean_data.columns:
            min_val = float(clean_data[col].min())  # 确保是标量
            max_val = float(clean_data[col].max())  # 确保是标量
            print(f"  {col}: [{min_val:.4f}, {max_val:.4f}]")

    print(f"清洗后数据: {clean_data.shape}")
    return clean_data


def safe_dataset_initialization(train_data, val_data, x_column, y_column, spatial_column):
    """安全的数据集初始化"""
    print("初始化数据集...")
    monitor_performance("数据集初始化前")

    # 验证每个数据集都不为空
    for name, dataset in [("训练集", train_data), ("验证集", val_data)]:
        if len(dataset) == 0:
            raise ValueError(f"{name} 为空")
        print(f"{name}: {len(dataset)} 行")

    try:
        start_time = time.time()
        train_set, val_set, _ = datasets.init_dataset_split(
            train_data=train_data,
            val_data=val_data,
            test_data=val_data,  # 使用验证集作为测试集占位
            x_column=x_column,
            y_column=y_column,
            spatial_column=spatial_column,
            batch_size=64,
            use_model="gnnwr"
        )
        init_time = time.time() - start_time
        print(f"✅ 数据集初始化成功 - 耗时: {init_time:.2f}秒")
        monitor_performance("数据集初始化后")
        return train_set, val_set
    except Exception as e:
        print(f"❌ 数据集初始化失败: {e}")
        raise


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)

    return {
        'RMSE': rmse,
        'MSE': mse,
        'R': r,
        'R2': r2
    }


def plot_aggregated_scatter(all_true, all_pred, metrics, save_path="result/cross_validation_results"):
    """绘制聚合散点图"""
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(12, 10))

    # 散点图
    plt.subplot(2, 2, 1)
    plt.scatter(all_true, all_pred, alpha=0.6, s=10)

    # 添加1:1线
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('站点级交叉验证结果')
    plt.grid(True, alpha=0.3)

    # 在右上角添加指标文本
    metrics_text = f"RMSE: {metrics['RMSE']:.4f}\nR²: {metrics['R2']:.4f}\nR: {metrics['R']:.4f}\nMSE: {metrics['MSE']:.4f}"
    plt.text(0.95, 0.05, metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 残差图
    plt.subplot(2, 2, 2)
    residuals = all_pred - all_true
    plt.scatter(all_pred, residuals, alpha=0.6, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差图')
    plt.grid(True, alpha=0.3)

    # 分布图
    plt.subplot(2, 2, 3)
    plt.hist(all_true, bins=50, alpha=0.7, label='真实值', density=True)
    plt.hist(all_pred, bins=50, alpha=0.7, label='预测值', density=True)
    plt.xlabel('值')
    plt.ylabel('密度')
    plt.title('真实值与预测值分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 残差分布图（新增）
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=50, alpha=0.7, color='green', density=True)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('残差')
    plt.ylabel('密度')
    plt.title('残差分布')
    plt.grid(True, alpha=0.3)

    # 在残差分布图中添加统计信息
    residual_stats = f"残差统计:\n均值: {residuals.mean():.4f}\n标准差: {residuals.std():.4f}"
    plt.text(0.95, 0.95, residual_stats, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_path}/aggregated_scatter_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}/aggregated_scatter_plot.pdf", bbox_inches='tight')
    plt.close()

    print(f"✅ 散点图已保存至: {save_path}/aggregated_scatter_plot.png")


def station_level_cross_validation(data, x_column, y_column, spatial_column, station_column='station_id'):
    """站点级交叉验证"""
    print("开始站点级交叉验证...")

    # 获取所有唯一站点
    unique_stations = data[station_column].unique()
    n_stations = len(unique_stations)
    print(f"总站点数: {n_stations}")

    # 存储所有预测结果
    all_true = []
    all_pred = []
    fold_results = []

    # 创建结果目录
    os.makedirs("result/cross_validation_results", exist_ok=True)

    # 数据标准化（使用全体数据）
    print("数据标准化...")
    scaler = StandardScaler()
    data_standardized = data.copy()
    data_standardized[x_column] = scaler.fit_transform(data_standardized[x_column])

    total_start_time = time.time()

    for i, test_station in enumerate(unique_stations):
        print(f"\n--- 折 {i + 1}/{n_stations}: 验证站点 {test_station} ---")

        try:
            # 分割数据：一个站点作为验证集，其余作为训练集
            train_data = data_standardized[data_standardized[station_column] != test_station]
            val_data = data_standardized[data_standardized[station_column] == test_station]

            if len(train_data) == 0 or len(val_data) == 0:
                print(f"跳过站点 {test_station}: 训练集或验证集为空")
                continue

            print(f"训练集: {len(train_data)} 行, 验证集: {len(val_data)} 行")

            # 初始化数据集
            train_set, val_set = safe_dataset_initialization(
                train_data, val_data, x_column, y_column, spatial_column
            )

            # 配置模型参数
            optimizer_params = {
                "scheduler": "MultiStepLR",
                "scheduler_milestones": [100, 200, 300],
                "scheduler_gamma": 0.75,
            }

            # 初始化模型
            model_name = f"GNNWR_Fold_{i + 1}"
            gnnwr = models.GNNWR(
                train_dataset=train_set,
                valid_dataset=val_set,
                test_dataset=val_set,  # 使用验证集作为测试集
                dense_layers=[256, 128, 64],
                activate_func=nn.PReLU(),
                start_lr=0.001,
                optimizer="Adam",
                model_name=model_name,
                model_save_path="result/cross_validation_models",
                log_path="result/cross_validation_logs",
                write_path="result/cross_validation_runs",
                optimizer_params=optimizer_params
            )

            # 创建目录
            os.makedirs("result/cross_validation_models", exist_ok=True)
            os.makedirs("result/cross_validation_logs", exist_ok=True)
            os.makedirs("result/cross_validation_runs", exist_ok=True)

            # 训练模型（较少的epoch，因为有很多折）
            gnnwr.add_graph()
            gnnwr.run(max_epoch=50, early_stop=20, print_frequency=10)

            # 加载模型并进行预测
            gnnwr.load_model(f'result/cross_validation_models/{model_name}.pkl')

            # 获取验证集预测结果
            val_predictions = gnnwr.predict(val_set)
            val_true = val_data[y_column[0]].values

            # 存储结果
            all_true.extend(val_true)
            all_pred.extend(val_predictions)

            # 计算当前折的指标
            fold_metrics = calculate_metrics(val_true, val_predictions)
            fold_results.append({
                'station_id': test_station,
                'fold': i + 1,
                'n_train': len(train_data),
                'n_val': len(val_data),
                **fold_metrics
            })

            print(f"折 {i + 1} 完成 - RMSE: {fold_metrics['RMSE']:.4f}, R²: {fold_metrics['R2']:.4f}")

            # 清理内存
            del gnnwr, train_set, val_set
            gc.collect()

        except Exception as e:
            print(f"折 {i + 1} 失败: {e}")
            continue

    total_time = time.time() - total_start_time
    print(f"\n=== 交叉验证完成 ===")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每折耗时: {total_time / len(fold_results):.2f}秒")
    print(f"成功完成的折数: {len(fold_results)}/{n_stations}")

    # 计算总体指标
    if len(all_true) > 0:
        overall_metrics = calculate_metrics(all_true, all_pred)

        # 保存详细结果
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv("result/cross_validation_results/detailed_results.csv", index=False)

        # 保存总体结果
        overall_results = {
            'total_stations': n_stations,
            'successful_folds': len(fold_results),
            'total_samples': len(all_true),
            **overall_metrics
        }
        pd.DataFrame([overall_results]).to_csv("result/cross_validation_results/overall_results.csv", index=False)

        # 绘制聚合散点图
        plot_aggregated_scatter(np.array(all_true), np.array(all_pred), overall_metrics)

        # 打印结果
        print("\n=== 总体评估结果 ===")
        for metric, value in overall_metrics.items():
            print(f"{metric}: {value:.4f}")

        return overall_metrics, results_df
    else:
        print("❌ 没有成功的交叉验证折")
        return None, None


def main():
    """主函数 - 站点级交叉验证版本"""
    try:
        # 1. 加载数据
        print("加载数据...")
        monitor_performance("程序开始")
        if not os.path.exists('lu_onehot.xlsx'):
            raise FileNotFoundError("数据文件 'lu_onehot.xlsx' 不存在")

        data = pd.read_excel('lu_onehot.xlsx')
        print(f"原始数据: {data.shape}")
        monitor_performance("数据加载后")


        # 2. 定义特征
        x_column = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation',
                    'std_slope', 'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2',
                    'std_high', 'std_aspect', 'glsnow', 'cswe', 'snow_depth_snow_depth',
                    'ERA5温度_ERA5温度', 'era5_swe', 'doy', 'gldas', 'year', 'month', 'scp_start',
                    'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da', 'db', 'dc', 'dd', 'landuse_11',
                    'landuse_12', 'landuse_21', 'landuse_22', 'landuse_23', 'landuse_24',
                    'landuse_31', 'landuse_32', 'landuse_33', 'landuse_41', 'landuse_42',
                    'landuse_43', 'landuse_46', 'landuse_51', 'landuse_52', 'landuse_53',
                    'landuse_62', 'landuse_63', 'landuse_64']
        y_column = ['swe']
        spatial_column = ['X', 'Y']
        station_column = 'station_id'

        # 3. 数据调试
        if not debug_data_issues(data, x_column, y_column, spatial_column, station_column):
            raise ValueError("数据调试发现问题，请检查数据")

        # 4. 数据清洗
        clean_data = robust_data_cleaning(data, x_column, y_column, spatial_column, station_column)

        # 5. 执行站点级交叉验证
        overall_metrics, detailed_results = station_level_cross_validation(
            clean_data, x_column, y_column, spatial_column, station_column
        )

        if overall_metrics is not None:
            print("\n🎉 站点级交叉验证成功完成!")
            print(f"最终结果保存在: result/cross_validation_results/")

            # 打印最佳和最差站点
            if detailed_results is not None:
                best_station = detailed_results.loc[detailed_results['R2'].idxmax()]
                worst_station = detailed_results.loc[detailed_results['R2'].idxmin()]

                print(f"\n最佳预测站点: {best_station['station_id']} (R²: {best_station['R2']:.4f})")
                print(f"最差预测站点: {worst_station['station_id']} (R²: {worst_station['R2']:.4f})")

        monitor_performance("程序结束")

    except Exception as e:
        print(f"❌ 主程序失败: {e}")
        import traceback
        traceback.print_exc()


def simple_station_cv_version():
    """简化版本的站点级交叉验证"""
    try:
        print("尝试简化版本站点级交叉验证...")

        # 1. 加载数据
        data = pd.read_excel('lu_onehot.xlsx')
        print(f"原始数据: {data.shape}")

        # 2. 定义特征（简化版）
        x_column = ['elevation', 'slope', 'aspect', 'X', 'Y', 'doy', 'year', 'month']
        y_column = ['swe']
        spatial_column = ['X', 'Y']
        station_column = 'station_id'

        # 3. 简化数据清洗
        clean_data = data.copy()
        clean_data = clean_data.dropna(subset=x_column + y_column + [station_column])

        # 移除数据量过少的站点
        station_counts = clean_data[station_column].value_counts()
        valid_stations = station_counts[station_counts >= 3].index
        clean_data = clean_data[clean_data[station_column].isin(valid_stations)]

        print(f"简化清洗后数据: {clean_data.shape}")
        print(f"可用站点数: {clean_data[station_column].nunique()}")

        # 4. 执行简化的交叉验证（只运行前10个站点作为测试）
        unique_stations = clean_data[station_column].unique()[:10]
        print(f"测试运行前 {len(unique_stations)} 个站点...")

        all_true = []
        all_pred = []

        for i, test_station in enumerate(unique_stations):
            print(f"折 {i + 1}/{len(unique_stations)}: 站点 {test_station}")

            try:
                # 分割数据
                train_data = clean_data[clean_data[station_column] != test_station]
                val_data = clean_data[clean_data[station_column] == test_station]

                if len(train_data) == 0 or len(val_data) == 0:
                    continue

                # 初始化数据集
                train_set, val_set, _ = datasets.init_dataset_split(
                    train_data=train_data,
                    val_data=val_data,
                    test_data=val_data,
                    x_column=x_column,
                    y_column=y_column,
                    spatial_column=spatial_column,
                    batch_size=32,  # 更小的batch size
                    use_model="gnnwr"
                )

                # 简化模型
                model_name = f"GNNWR_Simple_Fold_{i + 1}"
                gnnwr = models.GNNWR(
                    train_dataset=train_set,
                    valid_dataset=val_set,
                    test_dataset=val_set,
                    dense_layers=[128, 64],  # 更简单的网络
                    activate_func=nn.ReLU(),
                    start_lr=0.001,
                    optimizer="Adam",
                    model_name=model_name,
                    model_save_path="result/simple_cv_models",
                    log_path="result/simple_cv_logs",
                    write_path="result/simple_cv_runs"
                )

                # 创建目录
                os.makedirs("result/simple_cv_models", exist_ok=True)

                # 快速训练
                gnnwr.add_graph()
                gnnwr.run(max_epoch=30, early_stop=10, print_frequency=5)

                # 预测
                gnnwr.load_model(f'result/simple_cv_models/{model_name}.pkl')
                val_predictions = gnnwr.predict(val_set)
                val_true = val_data[y_column[0]].values

                all_true.extend(val_true)
                all_pred.extend(val_predictions)

                print(f"折 {i + 1} 完成")

            except Exception as e:
                print(f"折 {i + 1} 失败: {e}")
                continue

        # 计算总体指标
        if len(all_true) > 0:
            overall_metrics = calculate_metrics(all_true, all_pred)
            print("\n简化版本结果:")
            for metric, value in overall_metrics.items():
                print(f"{metric}: {value:.4f}")

            # 绘制散点图
            plot_aggregated_scatter(np.array(all_true), np.array(all_pred), overall_metrics,
                                    "result/simple_cv_results")

            return overall_metrics
        else:
            print("❌ 简化版本没有成功预测")
            return None

    except Exception as e:
        print(f"简化版本失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 先尝试完整版本
    try:
        main()
    except Exception as e:
        print(f"完整版本失败: {e}")
        print("\n尝试简化版本...")
        # 如果完整版本失败，尝试简化版本
        simple_station_cv_version()

