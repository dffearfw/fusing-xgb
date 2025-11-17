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


def fix_timestamp_issues(data, x_columns, y_columns):
    """修复时间戳数据类型问题"""
    print("🔧 修复时间戳问题...")

    data_fixed = data.copy()
    updated_x_columns = x_columns.copy()

    for col in x_columns + y_columns:
        if col in data_fixed.columns:
            # 检查是否是时间戳类型或字符串类型
            if data_fixed[col].dtype == 'object' or 'time' in str(col).lower() or 'date' in str(col).lower():
                try:
                    # 尝试转换为数值
                    data_fixed[col] = pd.to_numeric(data_fixed[col], errors='coerce')
                    print(f"   - 已转换列 {col} 为数值类型")
                except:
                    # 如果是时间字符串，转换为时间戳再提取特征
                    try:
                        timestamp_col = pd.to_datetime(data_fixed[col], errors='coerce')
                        if not timestamp_col.isna().all():
                            # 成功转换为时间戳，提取数值特征
                            data_fixed[f'{col}_year'] = timestamp_col.dt.year
                            data_fixed[f'{col}_month'] = timestamp_col.dt.month
                            data_fixed[f'{col}_day'] = timestamp_col.dt.day
                            data_fixed[f'{col}_dayofweek'] = timestamp_col.dt.dayofweek
                            data_fixed[f'{col}_hour'] = timestamp_col.dt.hour

                            # 更新特征列
                            updated_x_columns.extend([f'{col}_year', f'{col}_month', f'{col}_day',
                                                      f'{col}_dayofweek', f'{col}_hour'])
                            print(f"   - 已从时间列 {col} 提取特征")

                            # 移除原始列
                            if col in updated_x_columns:
                                updated_x_columns.remove(col)
                            data_fixed = data_fixed.drop(columns=[col])
                    except:
                        print(f"   ⚠️ 无法处理列 {col}，将删除")
                        if col in updated_x_columns:
                            updated_x_columns.remove(col)
                        data_fixed = data_fixed.drop(columns=[col])

    # 确保所有列都是数值类型
    for col in updated_x_columns + y_columns:
        if col in data_fixed.columns:
            data_fixed[col] = pd.to_numeric(data_fixed[col], errors='coerce')

    print(f"   - 修复后特征列数: {len(updated_x_columns)}")
    return data_fixed, updated_x_columns


def create_10_fold_by_station(data, station_column='station_id', random_state=42):
    """将站点分成10折"""
    print("🎯 创建10折交叉验证（按站点划分）...")

    # 获取所有唯一站点
    stations = data[station_column].unique()
    print(f"   - 总站点数: {len(stations)}")

    # 随机打乱站点
    np.random.seed(random_state)
    np.random.shuffle(stations)

    # 分成10折
    n_folds = 10
    fold_size = len(stations) // n_folds
    station_folds = []

    for i in range(n_folds):
        if i < n_folds - 1:
            fold_stations = stations[i * fold_size:(i + 1) * fold_size]
        else:
            fold_stations = stations[i * fold_size:]  # 最后一折包含剩余站点
        station_folds.append(fold_stations)
        print(f"   - 折 {i + 1}: {len(fold_stations)} 个站点")

    return station_folds


def run_10_fold_cross_validation(data, x_columns, y_columns, spatial_columns, station_column='station_id'):
    """运行10折交叉验证"""
    print("🚀 开始10折交叉验证...")

    # 修复时间戳问题
    data_fixed, x_columns_fixed = fix_timestamp_issues(data, x_columns, y_columns)
    print(f"   - 修复后特征列: {x_columns_fixed}")
    print(f"   - 目标列: {y_columns}")

    # 数据清洗
    clean_data = robust_data_cleaning(data_fixed, x_columns_fixed, y_columns, spatial_columns, station_column)

    # 创建10折
    station_folds = create_10_fold_by_station(clean_data, station_column)

    all_true = []
    all_pred = []
    all_results = []
    fold_metrics_list = []

    total_start_time = time.time()

    for fold_idx, val_stations in enumerate(station_folds):
        print(f"\n=== 折 {fold_idx + 1}/10 ===")

        try:
            # 训练集：其他9折的站点
            train_stations = []
            for i, stations in enumerate(station_folds):
                if i != fold_idx:
                    train_stations.extend(stations)

            # 分割数据
            train_data = clean_data[clean_data[station_column].isin(train_stations)]
            val_data = clean_data[clean_data[station_column].isin(val_stations)]

            print(f"   - 训练集: {len(train_data)} 样本, {len(train_stations)} 站点")
            print(f"   - 验证集: {len(val_data)} 样本, {len(val_stations)} 站点")

            if len(train_data) == 0 or len(val_data) == 0:
                print("⚠️ 训练集或验证集为空，跳过该折")
                continue

            # 数据标准化
            data_standardized = standardize_data(pd.concat([train_data, val_data]), x_columns_fixed, y_columns)
            train_data_std = data_standardized[data_standardized[station_column].isin(train_stations)]
            val_data_std = data_standardized[data_standardized[station_column].isin(val_stations)]

            # 创建数据集
            train_set, val_set = safe_dataset_initialization(
                train_data_std, val_data_std, x_columns_fixed, y_columns, spatial_columns
            )

            # 配置模型
            model_name = f"GNNWR_10fold_{fold_idx + 1}"
            gnnwr = models.GNNWR(
                train_dataset=train_set,
                valid_dataset=val_set,
                test_dataset=val_set,
                dense_layers=[128, 64],
                activate_func=nn.ReLU(),
                start_lr=0.0005,
                optimizer="Adam",
                model_name=model_name,
                model_save_path="result/10fold_cv_models",
                log_path="result/10fold_cv_logs",
                write_path="result/10fold_cv_runs",
                optimizer_params={
                    "scheduler": "MultiStepLR",
                    "scheduler_milestones": [20, 40],
                    "scheduler_gamma": 0.8,
                }
            )

            # 创建目录
            os.makedirs("result/10fold_cv_models", exist_ok=True)
            os.makedirs("result/10fold_cv_logs", exist_ok=True)
            os.makedirs("result/10fold_cv_runs", exist_ok=True)

            # 训练模型（减少epoch数）
            gnnwr.add_graph()
            gnnwr.run(max_epoch=20, early_stop=8, print_frequency=5)

            # 预测
            gnnwr.load_model(f'result/10fold_cv_models/{model_name}.pkl')
            val_predictions = gnnwr.predict(val_set)

            if len(val_predictions) == 0:
                print(f"⚠️ 无预测结果，跳过该折")
                continue

            val_true = val_data_std[y_columns[0]].values

            # 存储结果
            all_true.extend(val_true)
            all_pred.extend(val_predictions)

            # 计算当前折的指标
            fold_metrics = calculate_metrics(val_true, val_predictions)
            fold_results = {
                'fold': fold_idx + 1,
                'train_stations': len(train_stations),
                'val_stations': len(val_stations),
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                **fold_metrics
            }
            all_results.append(fold_results)
            fold_metrics_list.append(fold_metrics)

            print(f"✅ 折 {fold_idx + 1} 完成 - RMSE: {fold_metrics['RMSE']:.4f}, R²: {fold_metrics['R2']:.4f}")

            # 清理内存
            del gnnwr, train_set, val_set
            gc.collect()

        except Exception as e:
            print(f"❌ 折 {fold_idx + 1} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总结果
    total_time = time.time() - total_start_time
    print(f"\n=== 10折交叉验证完成 ===")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"成功完成的折数: {len(all_results)}/10")

    if len(all_true) > 0:
        overall_metrics = calculate_metrics(all_true, all_pred)

        # 计算各折平均指标
        if fold_metrics_list:
            avg_r2 = np.mean([m['R2'] for m in fold_metrics_list])
            avg_rmse = np.mean([m['RMSE'] for m in fold_metrics_list])
            std_r2 = np.std([m['R2'] for m in fold_metrics_list])
            std_rmse = np.std([m['RMSE'] for m in fold_metrics_list])

            print(f"\n📊 10折交叉验证统计:")
            print(f"  平均 R²: {avg_r2:.4f} ± {std_r2:.4f}")
            print(f"  平均 RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
            print(f"  总体 R²: {overall_metrics['R2']:.4f}")
            print(f"  总体 RMSE: {overall_metrics['RMSE']:.4f}")

        # 保存结果
        results_df = pd.DataFrame({
            'True': all_true,
            'Predicted': all_pred
        })
        results_df.to_csv('result/10fold_cv_results.csv', index=False)

        detailed_results = pd.DataFrame(all_results)
        detailed_results.to_csv('result/10fold_cv_detailed.csv', index=False)

        # 绘制结果图
        plot_aggregated_scatter(all_true, all_pred, overall_metrics, "result/10fold_cv_results")

        print("\n总体评估指标:")
        for metric, value in overall_metrics.items():
            print(f"{metric}: {value:.4f}")

        return overall_metrics, detailed_results
    else:
        print("❌ 没有成功的交叉验证折")
        return None, None


def quick_2_fold_test(data, x_columns, y_columns, spatial_columns, station_column='station_id'):
    """修复的快速2折测试"""
    print("⚡ 执行修复的快速2折测试...")

    # 使用已经清洗过的数据，跳过时间戳修复
    data_fixed, x_fixed = fix_timestamp_issues(data, x_columns, y_columns)

    # 再次确保数据清洗
    clean_data = enhanced_robust_data_cleaning(data_fixed, x_fixed, y_columns, spatial_columns, station_column)

    # 只取前10个站点测试（更少的数据用于快速测试）
    stations = clean_data[station_column].unique()[:10]
    test_data = clean_data[clean_data[station_column].isin(stations)]

    if len(test_data) < 10:
        print("⚠️ 测试数据过少，跳过快速测试")
        return []

    # 分成2折
    np.random.seed(42)
    np.random.shuffle(stations)
    fold1_stations = stations[:5]  # 每折5个站点
    fold2_stations = stations[5:10]

    results = []
    all_true = []
    all_pred = []

    for fold_idx, (train_stations, val_stations) in enumerate(
            [(fold2_stations, fold1_stations), (fold1_stations, fold2_stations)]):
        print(f"\n快速测试折 {fold_idx + 1}:")

        train_data = test_data[test_data[station_column].isin(train_stations)]
        val_data = test_data[test_data[station_column].isin(val_stations)]

        if len(train_data) < 5 or len(val_data) < 2:
            print("⚠️ 训练集或验证集样本过少，跳过")
            continue

        print(f"  训练集: {len(train_data)} 样本, {len(train_stations)} 站点")
        print(f"  验证集: {len(val_data)} 样本, {len(val_stations)} 站点")

        try:
            # 数据标准化
            data_standardized = standardize_data(pd.concat([train_data, val_data]), x_fixed, y_columns)
            train_data_std = data_standardized[data_standardized[station_column].isin(train_stations)]
            val_data_std = data_standardized[data_standardized[station_column].isin(val_stations)]

            # 数据集初始化
            train_set, val_set = enhanced_safe_dataset_initialization(
                train_data_std, val_data_std, x_fixed, y_columns, spatial_columns
            )

            # 简化模型配置
            model_name = f"quick_test_{fold_idx}"
            gnnwr = models.GNNWR(
                train_dataset=train_set,
                valid_dataset=val_set,
                test_dataset=val_set,
                dense_layers=[32, 16],  # 更小的网络
                start_lr=0.001,
                optimizer="Adam",
                model_name=model_name,
                model_save_path="result/quick_test"
            )

            # 创建目录
            os.makedirs("result/quick_test", exist_ok=True)

            # 快速训练
            gnnwr.add_graph()
            gnnwr.run(max_epoch=3, early_stop=1, print_frequency=1)  # 更少的epoch

            # 预测
            model_path = f'result/quick_test/{model_name}.pkl'
            if os.path.exists(model_path):
                gnnwr.load_model(model_path)
                val_predictions = gnnwr.predict(val_set)
                val_true = val_data_std[y_columns[0]].values

                if len(val_predictions) > 0:
                    all_true.extend(val_true)
                    all_pred.extend(val_predictions)

                    fold_metrics = calculate_metrics(val_true, val_predictions)
                    results.append({
                        'fold': fold_idx + 1,
                        'r2': fold_metrics['R2'],
                        'rmse': fold_metrics['RMSE']
                    })

                    print(f"✅ 快速测试折 {fold_idx + 1} 完成: R² = {fold_metrics['R2']:.4f}")
                else:
                    print(f"⚠️ 快速测试折 {fold_idx + 1} 无预测结果")
            else:
                print(f"⚠️ 快速测试折 {fold_idx + 1} 模型文件不存在")

        except Exception as e:
            print(f"❌ 快速测试折 {fold_idx + 1} 失败: {e}")
            continue

        # 清理内存
        try:
            del gnnwr, train_set, val_set
            gc.collect()
        except:
            pass

    # 计算总体指标
    if len(all_true) > 0:
        overall_metrics = calculate_metrics(all_true, all_pred)
        print(f"\n快速测试总体结果: R² = {overall_metrics['R2']:.4f}, RMSE = {overall_metrics['RMSE']:.4f}")

    return results


# 修改主函数以使用10折交叉验证
def main():
    """主函数 - 修复版本"""
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
        x_columns = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation',
                     'std_slope', 'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2',
                     'std_high', 'std_aspect', 'glsnow', 'cswe', 'snow_depth_snow_depth',
                     'ERA5温度_ERA5温度', 'era5_swe', 'doy', 'gldas', 'year', 'month', 'scp_start',
                     'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da', 'db', 'dc', 'dd', 'landuse_11',
                     'landuse_12', 'landuse_21', 'landuse_22', 'landuse_23', 'landuse_24',
                     'landuse_31', 'landuse_32', 'landuse_33', 'landuse_41', 'landuse_42',
                     'landuse_43', 'landuse_46', 'landuse_51', 'landuse_52', 'landuse_53',
                     'landuse_62', 'landuse_63', 'landuse_64']
        y_columns = ['swe']
        spatial_columns = ['X', 'Y']
        station_column = 'station_id'

        # 3. 使用增强的数据调试
        print("\n=== 执行增强数据调试 ===")
        if not enhanced_debug_data_issues(data, x_columns, y_columns, spatial_columns, station_column):
            print("❌ 数据调试发现问题，将尝试修复...")

        # 4. 使用增强的数据清洗
        print("\n=== 执行增强数据清洗 ===")
        clean_data = enhanced_robust_data_cleaning(
            data, x_columns, y_columns, spatial_columns, station_column
        )

        # 5. 重新调试清洗后的数据
        print("\n=== 检查清洗后数据 ===")
        if not enhanced_debug_data_issues(clean_data, x_columns, y_columns, spatial_columns, station_column):
            raise ValueError("数据清洗后仍然存在问题")

        print("✅ 数据准备完成，开始模型训练...")

        # 6. 先运行快速测试
        print("\n=== 开始快速2折测试 ===")
        quick_test_results = quick_2_fold_test(
            clean_data, x_columns, y_columns, spatial_columns, station_column
        )

        if quick_test_results and len(quick_test_results) > 0:
            print("✅ 快速测试通过，开始完整10折交叉验证...")

            # 7. 执行10折交叉验证
            overall_metrics, detailed_results = improved_10_fold_cross_validation(
                clean_data, x_columns, y_columns, spatial_columns, station_column
            )

            if overall_metrics is not None:
                print("\n🎉 10折交叉验证成功完成!")
                # ... 其余代码保持不变
            else:
                print("❌ 10折交叉验证失败，回退到简化版本...")
                simple_station_cv_version()
        else:
            print("❌ 快速测试失败！请先修复问题再继续")
            return

        monitor_performance("程序结束")

    except Exception as e:
        print(f"❌ 主程序失败: {e}")
        import traceback
        traceback.print_exc()


# 添加缺失的函数（保持原有实现）
def enhanced_debug_data_issues(data, x_columns, y_columns, spatial_columns, station_column='station_id'):
    """增强的数据问题调试"""
    print("=== 增强数据调试 ===")
    print(f"原始数据形状: {data.shape}")

    # 1. 检查列是否存在
    all_required_columns = x_columns + y_columns + spatial_columns + [station_column]
    missing_columns = [col for col in all_required_columns if col not in data.columns]
    if missing_columns:
        print(f"❌ 缺失列: {missing_columns}")
        print(f"可用列: {list(data.columns)}")
        return False

    print("✅ 所有必需列都存在")

    # 2. 检查数据类型
    print("检查数据类型...")
    for col in all_required_columns:
        dtype = data[col].dtype
        print(f"   {col}: {dtype}")

    # 3. 检查缺失值
    print("检查缺失值...")
    missing_stats = data[all_required_columns].isnull().sum()
    if missing_stats.sum() > 0:
        print("❌ 发现缺失值:")
        for col, count in missing_stats.items():
            if count > 0:
                print(f"   {col}: {count} 个缺失值 ({count / len(data):.2%})")
    else:
        print("✅ 无缺失值")

    # 4. 检查无穷大值
    print("检查无穷大值...")
    numeric_columns = data[all_required_columns].select_dtypes(include=[np.number]).columns
    inf_found = False
    for col in numeric_columns:
        inf_count = np.isinf(data[col]).sum()
        if inf_count > 0:
            print(f"❌ {col}: {inf_count} 个无穷大值")
            inf_found = True

    if not inf_found:
        print("✅ 无无穷大值")

    # 5. 检查零方差特征
    print("检查零方差特征...")
    for col in x_columns:
        if col in data.columns and data[col].dtype in [np.number]:
            variance = data[col].var()
            if variance == 0:
                print(f"⚠️ {col}: 零方差特征")

    # 6. 检查数据范围
    print("检查数据范围...")
    for col in y_columns:
        if col in data.columns:
            print(f"   {col}: min={data[col].min():.4f}, max={data[col].max():.4f}, mean={data[col].mean():.4f}")

    return not (missing_stats.sum() > 0 or inf_found)


def enhanced_robust_data_cleaning(data, x_columns, y_columns, spatial_columns, station_column='station_id'):
    """增强版本的数据清洗，专门处理inf和NaN值"""
    print("开始增强数据清洗...")
    clean_data = data.copy()

    # 检查必需列
    all_required_columns = x_columns + y_columns + spatial_columns + [station_column]
    missing_columns = [col for col in all_required_columns if col not in clean_data.columns]
    if missing_columns:
        raise ValueError(f"缺少列: {missing_columns}")

    # 第一步：处理无穷大值
    print("处理无穷大值...")
    numeric_columns = clean_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in clean_data.columns:
            # 替换inf为NaN
            clean_data[col] = clean_data[col].replace([np.inf, -np.inf], np.nan)
            # 统计inf数量
            inf_count = np.isinf(clean_data[col]).sum()
            if inf_count > 0:
                print(f"   - 列 {col}: 替换 {inf_count} 个无穷大值为NaN")

    # 第二步：处理缺失值
    print("处理缺失值...")
    required_data_columns = x_columns + y_columns + spatial_columns + [station_column]

    # 检查每列的缺失率
    missing_stats = clean_data[required_data_columns].isnull().sum()
    high_missing_cols = missing_stats[missing_stats > 0].index.tolist()

    if high_missing_cols:
        print(f"   - 有缺失值的列: {high_missing_cols}")
        for col in high_missing_cols:
            missing_rate = missing_stats[col] / len(clean_data)
            print(f"     {col}: {missing_stats[col]} 个缺失值 ({missing_rate:.2%})")

    # 删除在必需列中有缺失值的行
    initial_count = len(clean_data)
    clean_data = clean_data.dropna(subset=required_data_columns)
    removed_count = initial_count - len(clean_data)
    print(f"   - 删除 {removed_count} 个有缺失值的行")
    print(f"   - 剩余数据量: {len(clean_data)}")

    if len(clean_data) == 0:
        raise ValueError("数据清洗后无有效数据")

    # 第三步：检查并修复零方差特征
    print("检查零方差特征...")
    zero_variance_cols = []
    for col in x_columns:
        if col in clean_data.columns:
            variance = clean_data[col].var()
            if variance == 0:
                zero_variance_cols.append(col)
                # 添加微小噪声修复零方差
                noise = np.random.normal(0, 1e-6, len(clean_data))
                clean_data[col] = clean_data[col] + noise
                print(f"   - 修复零方差列 {col}")

    if zero_variance_cols:
        print(f"   - 修复的零方差列: {zero_variance_cols}")

    # 第四步：处理异常值（可选，根据需求调整）
    print("处理异常值...")
    for col in y_columns:
        if col in clean_data.columns:
            Q1 = clean_data[col].quantile(0.25)
            Q3 = clean_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = clean_data[(clean_data[col] < lower_bound) | (clean_data[col] > upper_bound)]
            if len(outliers) > 0:
                print(f"   - {col}: 发现 {len(outliers)} 个异常值")
                # 可以选择删除或缩尾处理
                clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]

    # 第五步：筛选有效站点
    print("筛选有效站点...")
    station_counts = clean_data[station_column].value_counts()
    valid_stations = station_counts[station_counts >= 3].index
    clean_data = clean_data[clean_data[station_column].isin(valid_stations)]

    print(f"   - 有效站点数: {clean_data[station_column].nunique()}")
    print(f"   - 最终数据量: {len(clean_data)}")

    # 最终检查
    final_missing = clean_data[required_data_columns].isnull().sum().sum()
    final_inf = np.isinf(clean_data[required_data_columns].select_dtypes(include=[np.number])).sum().sum()

    print(f"数据清洗完成:")
    print(f"   - 剩余缺失值: {final_missing}")
    print(f"   - 无穷大值: {final_inf}")

    return clean_data


def standardize_data(data, x_column, y_column):
    """数据标准化"""
    print("标准化数据...")
    standardized_data = data.copy()

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    standardized_data[x_column] = scaler.fit_transform(standardized_data[x_column])

    print("数据标准化完成")
    return standardized_data


def safe_dataset_initialization(train_data, val_data, x_column, y_column, spatial_column):
    """修复版本的数据集初始化"""
    print("初始化数据集...")
    monitor_performance("数据集初始化前")

    try:
        start_time = time.time()

        # 关键修复：在初始化前检查并修复数据
        train_data_fixed = train_data.copy()
        val_data_fixed = val_data.copy()

        # 修复零方差问题 - 对每个特征列进行检查
        for col in x_column:
            if col in train_data_fixed.columns:
                # 如果训练集该列方差为零，添加微小噪声
                if train_data_fixed[col].var() == 0:
                    print(f"⚠️ 修复零方差列: {col}")
                    noise = np.random.normal(0, 1e-6, len(train_data_fixed))
                    train_data_fixed[col] = train_data_fixed[col] + noise

                # 同样修复验证集
                if col in val_data_fixed.columns and val_data_fixed[col].var() == 0:
                    noise = np.random.normal(0, 1e-6, len(val_data_fixed))
                    val_data_fixed[col] = val_data_fixed[col] + noise

        train_set, val_set, _ = datasets.init_dataset_split(
            train_data=train_data_fixed,
            val_data=val_data_fixed,
            test_data=val_data_fixed,
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
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('10折交叉验证结果')
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

    # 残差分布图
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=50, alpha=0.7, color='green', density=True)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('残差')
    plt.ylabel('密度')
    plt.title('残差分布')
    plt.grid(True, alpha=0.3)

    # 在残差分布图中添加统计信息
    residual_stats = f"残差统计:\n均值: {np.mean(residuals):.4f}\n标准差: {np.std(residuals):.4f}"
    plt.text(0.95, 0.95, residual_stats, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_path}/aggregated_scatter_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}/aggregated_scatter_plot.pdf", bbox_inches='tight')
    plt.close()

    print(f"✅ 散点图已保存至: {save_path}/aggregated_scatter_plot.png")


def simple_station_cv_version():
    """简化版本的站点级交叉验证"""
    try:
        print("尝试简化版本站点级交叉验证...")

        # 1. 加载数据
        data = pd.read_excel('lu_onehot.xlsx')
        print(f"原始数据: {data.shape}")

        # 2. 定义特征（简化版）
        x_columns = ['elevation', 'slope', 'aspect', 'X', 'Y', 'doy', 'year', 'month']
        y_columns = ['swe']
        spatial_columns = ['X', 'Y']
        station_column = 'station_id'

        # 3. 修复时间戳问题
        data_fixed, x_columns_fixed = fix_timestamp_issues(data, x_columns, y_columns)

        # 4. 简化数据清洗
        clean_data = data_fixed.copy()
        clean_data = clean_data.dropna(subset=x_columns_fixed + y_columns + [station_column])

        # 移除数据量过少的站点
        station_counts = clean_data[station_column].value_counts()
        valid_stations = station_counts[station_counts >= 3].index
        clean_data = clean_data[clean_data[station_column].isin(valid_stations)]

        print(f"简化清洗后数据: {clean_data.shape}")
        print(f"可用站点数: {clean_data[station_column].nunique()}")

        # 5. 执行简化的交叉验证（只运行前10个站点作为测试）
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

                # 数据标准化
                data_standardized = standardize_data(pd.concat([train_data, val_data]), x_columns_fixed, y_columns)
                train_data_std = data_standardized[
                    data_standardized[station_column].isin(train_data[station_column].unique())]
                val_data_std = data_standardized[data_standardized[station_column].isin([test_station])]

                # 初始化数据集
                train_set, val_set, _ = datasets.init_dataset_split(
                    train_data=train_data_std,
                    val_data=val_data_std,
                    test_data=val_data_std,
                    x_column=x_columns_fixed,
                    y_column=y_columns,
                    spatial_column=spatial_columns,
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
                val_true = val_data_std[y_columns[0]].values

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
    # 先尝试10折交叉验证版本
    try:
        main()
    except Exception as e:
        print(f"10折交叉验证版本失败: {e}")
        print("\n尝试简化版本...")
        # 如果10折版本失败，尝试简化版本
        simple_station_cv_version()