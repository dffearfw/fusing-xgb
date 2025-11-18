import gc
import os
import sys
import traceback

import torch
from gnnwr import models, datasets, utils
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# 保存原始方法
original_randperm = torch.randperm


def patched_randperm(n, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None,
                     requires_grad=False):
    """修复设备不匹配的randperm"""
    if generator is not None:
        current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if hasattr(generator, 'device') and generator.device != current_device:
            # 创建新的生成器
            generator = torch.Generator(device=current_device)
            generator.manual_seed(torch.randint(0, 1000000, (1,)).item())

    return original_randperm(n, generator=generator, out=out, dtype=dtype, layout=layout, device=device,
                             requires_grad=requires_grad)


# 应用补丁
torch.randperm = patched_randperm

# 立即优化设置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()


def quick_gpu_fix():
    """快速GPU修复"""
    # 设置GPU设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return device


def setup_device(device_id=0):
    """设置GPU设备"""
    if torch.cuda.is_available():
        # 检查可用的GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个GPU")

        # 确保设备ID有效
        if device_id < gpu_count:
            device = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(device_id)  # 使用整数索引
            print(f"使用GPU: {torch.cuda.get_device_name(device_id)}")
        else:
            device = torch.device('cpu')
            print(f"设备ID {device_id} 无效，使用CPU")
    else:
        device = torch.device('cpu')
        print("CUDA不可用，使用CPU")

    return device


def station_based_kfold_cross_validation():
    """基于站点的10折交叉验证 - GPU优化版本"""
    device = quick_gpu_fix()

    print("=== 基于站点的10折交叉验证 (GPU优化版) ===")

    # 读取数据
    data = pd.read_excel('lu_onehot.xlsx')
    print(f"原始数据形状: {data.shape}")

    # 定义特征列和目标列
    x_column = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation', 'std_slope',
                'std_eastness', 'std_tpi', 'std_curvature1',
                'std_curvature2', 'std_high', 'std_aspect', 'glsnow', 'cswe', 'snow_depth_snow_depth',
                'ERA5温度_ERA5温度', 'era5_swe', 'doy', 'gldas',
                'year', 'month', 'scp_start', 'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da', 'db', 'dc', 'dd',
                'landuse_11', 'landuse_12', 'landuse_21', 'landuse_22',
                'landuse_23', 'landuse_24', 'landuse_31', 'landuse_32', 'landuse_33', 'landuse_41', 'landuse_42',
                'landuse_43', 'landuse_46',
                'landuse_51', 'landuse_52', 'landuse_53', 'landuse_62', 'landuse_63', 'landuse_64']
    y_column = ['swe']
    spatial_column = ['longitude', 'latitude']

    # 移除标准差为零的特征
    safe_x_columns = []
    for col in x_column:
        if col in data.columns and data[col].std() > 0:
            safe_x_columns.append(col)
        else:
            print(f"跳过特征 {col}")

    print(f"使用 {len(safe_x_columns)} 个有效特征")

    # 识别唯一站点（基于经纬度）
    print("\n识别站点中...")
    unique_stations = data[spatial_column].drop_duplicates()
    print(f"识别到 {len(unique_stations)} 个唯一站点")

    # 为每个站点分配ID
    station_ids = {}
    for idx, (_, row) in enumerate(unique_stations.iterrows()):
        station_ids[(row[spatial_column[0]], row[spatial_column[1]])] = idx

    # 为数据添加站点ID
    data_with_station = data.copy()
    data_with_station['station_id'] = data_with_station.apply(
        lambda row: station_ids.get((row[spatial_column[0]], row[spatial_column[1]]), -1), axis=1
    )

    # 检查站点数据分布
    station_counts = data_with_station['station_id'].value_counts()
    print(f"\n站点数据分布统计:")
    print(f"平均每个站点的样本数: {station_counts.mean():.2f}")
    print(f"最少样本的站点: {station_counts.min()}")
    print(f"最多样本的站点: {station_counts.max()}")
    print(f"样本数少于10的站点数: {len(station_counts[station_counts < 10])}")

    # 10折交叉验证 - 基于站点划分
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    unique_station_ids = sorted(data_with_station['station_id'].unique())

    print(f"\n开始10折交叉验证，共 {len(unique_station_ids)} 个站点...")

    fold_results = []
    detailed_predictions = []

    for fold, (train_val_station_idx, test_station_idx) in enumerate(kf.split(unique_station_ids)):
        print(f"\n{'=' * 60}")
        print(f"第 {fold + 1}/10 折")
        print(f"{'=' * 60}")

        # 获取当前折的站点ID
        train_val_stations = [unique_station_ids[i] for i in train_val_station_idx]
        test_stations = [unique_station_ids[i] for i in test_station_idx]

        # 从训练验证集中划分训练集和验证集站点
        train_stations, val_stations = train_test_split(
            train_val_stations, test_size=0.2, random_state=42
        )

        # 根据站点ID获取对应的数据
        train_data = data_with_station[data_with_station['station_id'].isin(train_stations)].drop('station_id', axis=1)
        val_data = data_with_station[data_with_station['station_id'].isin(val_stations)].drop('station_id', axis=1)
        test_data = data_with_station[data_with_station['station_id'].isin(test_stations)].drop('station_id', axis=1)

        print(f"训练集: {len(train_data)} 样本, {len(train_stations)} 站点")
        print(f"验证集: {len(val_data)} 样本, {len(val_stations)} 站点")
        print(f"测试集: {len(test_data)} 样本, {len(test_stations)} 站点")

        # 检查数据平衡性
        if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
            print("⚠️ 警告: 某个集合为空，跳过该折")
            continue

        try:
            # 创建数据集（内存优化版本）
            print("创建数据集中...")
            train_set, val_set, test_set = create_memory_efficient_dataset(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                safe_x_columns=safe_x_columns,
                y_column=y_column,
                spatial_column=spatial_column
            )

            # 初始化模型
            print("初始化模型中...")
            gnnwr = models.GNNWR(
                train_dataset=train_set,
                valid_dataset=val_set,
                test_dataset=test_set,
                dense_layers=[256, 128, 64],
                activate_func=nn.ReLU(),
                start_lr=0.001,
                optimizer="Adam",
                model_name=f"GNNWR_Station_Fold_{fold + 1}",
                model_save_path=f"result/station_kfold/fold_{fold + 1}",
                log_path=f"result/station_kfold/logs_fold_{fold + 1}",
                write_path=f"result/station_kfold/runs_fold_{fold + 1}",
                optimizer_params={}
            )

            # 训练模型（带内存管理）
            print("开始训练...")
            # gnnwr.add_graph()

            # 使用安全的GPU训练
            training_success = safe_gnnwr_training(
                gnnwr,
                max_epoch=300,
                early_stop=50,
                print_frequency=50
            )

            if training_success:
                # 评估模型
                model_path = f'result/station_kfold/fold_{fold + 1}/GNNWR_Station_Fold_{fold + 1}.pkl'
                if os.path.exists(model_path):
                    print("加载模型进行评估...")
                    gnnwr.load_model(model_path)
                    results = gnnwr.result(return_metrics=True)

                    # 获取详细预测结果
                    predictions = gnnwr.predict(return_result=True)
                    if predictions is not None:
                        test_with_pred = test_data.copy()
                        if hasattr(predictions, 'shape') and len(predictions) == len(test_data):
                            test_with_pred['predicted_swe'] = predictions
                            test_with_pred['fold'] = fold + 1
                            detailed_predictions.append(test_with_pred)

                    fold_result = {
                        'fold': fold + 1,
                        'train_stations': len(train_stations),
                        'val_stations': len(val_stations),
                        'test_stations': len(test_stations),
                        'train_samples': len(train_data),
                        'val_samples': len(val_data),
                        'test_samples': len(test_data),
                        'metrics': results
                    }
                    fold_results.append(fold_result)

                    print(f"✅ 第 {fold + 1} 折完成")
                    print(f"   测试集指标: {results}")

                else:
                    print(f"❌ 第 {fold + 1} 折模型文件未找到")
            else:
                print(f"❌ 第 {fold + 1} 折训练失败")

        except Exception as e:
            print(f"❌ 第 {fold + 1} 折训练失败: {e}")
            traceback.print_exc()
            continue

        finally:
            # 每个折结束后清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    # 汇总结果
    print("\n" + "=" * 80)
    print("基于站点的10折交叉验证汇总结果")
    print("=" * 80)

    if fold_results:
        # 计算各指标的平均值和标准差
        metrics_summary = {}
        for result in fold_results:
            print(f"\n第 {result['fold']} 折:")
            print(
                f"  站点数 - 训练: {result['train_stations']}, 验证: {result['val_stations']}, 测试: {result['test_stations']}")
            print(
                f"  样本数 - 训练: {result['train_samples']}, 验证: {result['val_samples']}, 测试: {result['test_samples']}")
            print(f"  指标: {result['metrics']}")

            for metric, value in result['metrics'].items():
                if metric not in metrics_summary:
                    metrics_summary[metric] = []
                metrics_summary[metric].append(value)

        print(f"\n{'=' * 50}")
        print("平均结果 (± 标准差):")
        print(f"{'=' * 50}")
        for metric, values in metrics_summary.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric.upper():<8}: {mean_val:.4f} ± {std_val:.4f}")

        # 保存详细结果
        os.makedirs('result/station_kfold', exist_ok=True)

        # 保存指标结果
        summary_df = pd.DataFrame(fold_results)

        # 展开metrics列
        metrics_expanded = pd.json_normalize(summary_df['metrics'])
        summary_expanded = pd.concat([summary_df.drop('metrics', axis=1), metrics_expanded], axis=1)
        summary_expanded.to_csv('result/station_kfold/cross_validation_summary.csv', index=False)

        # 保存详细预测结果
        if detailed_predictions:
            all_predictions = pd.concat(detailed_predictions, ignore_index=True)
            all_predictions.to_csv('result/station_kfold/detailed_predictions.csv', index=False)
            print(f"\n详细预测结果已保存到: result/station_kfold/detailed_predictions.csv")

        print(f"\n汇总结果已保存到: result/station_kfold/cross_validation_summary.csv")
        print(f"成功完成的折数: {len(fold_results)}/10")

        return summary_expanded, all_predictions if detailed_predictions else None

    else:
        print("所有折的训练都失败了")
        return None, None


def create_memory_efficient_dataset(train_data, val_data, test_data, safe_x_columns, y_column, spatial_column):
    """创建内存高效的数据集"""
    try:
        # 使用较小的批量大小
        batch_size = 16

        train_set, val_set, test_set = datasets.init_dataset_split(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            x_column=safe_x_columns,
            y_column=y_column,
            spatial_column=spatial_column,
            batch_size=batch_size,  # 减少批量大小
            use_model="gnnwr"
        )

        print(f"📊 数据集创建完成 - 批量大小: {batch_size}")
        return train_set, val_set, test_set

    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        raise

class GPUMemoryManager:
    """GPU内存管理器"""

    def __init__(self, safety_margin_gb=1.0):
        self.safety_margin_gb = safety_margin_gb

    def get_available_memory(self):
        """获取可用GPU内存"""
        if not torch.cuda.is_available():
            return 0
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return total - allocated - self.safety_margin_gb

    def can_allocate(self, estimated_size_gb):
        """检查是否可以分配指定大小的内存"""
        return self.get_available_memory() >= estimated_size_gb

    def optimize_memory(self):
        """优化内存使用"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def handle_oom_error(self, gnnwr_instance, current_batch_size):
        """处理OOM错误"""
        print("🔄 处理OOM错误...")
        self.optimize_memory()

        # 减少批量大小
        new_batch_size = max(1, current_batch_size // 2)
        if hasattr(gnnwr_instance, 'batch_size'):
            gnnwr_instance.batch_size = new_batch_size
            print(f"🔽 批量大小调整为: {new_batch_size}")

        # 等待内存稳定
        import time
        time.sleep(2)

        return new_batch_size

def safe_gnnwr_training(gnnwr_instance, max_epoch=300, early_stop=50, print_frequency=50):
    """安全的GNNWR训练，带内存管理"""
    memory_manager = GPUMemoryManager(safety_margin_gb=1.0)

    # 初始内存优化
    memory_manager.optimize_memory()

    # 设置较小的批量大小
    current_batch_size = getattr(gnnwr_instance, 'batch_size', 64)
    if current_batch_size > 16:
        gnnwr_instance.batch_size = 16
        print(f"📊 初始批量大小设置为: 16")

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # 训练前检查内存
            if not memory_manager.can_allocate(2.0):  # 预估需要2GB
                print("⚠️ 内存紧张，清理缓存...")
                memory_manager.optimize_memory()

            # 训练模型
            gnnwr_instance.run(max_epoch=max_epoch, early_stop=early_stop, print_frequency=print_frequency)
            return True

        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                retry_count += 1
                print(f"💥 OOM错误 (尝试 {retry_count}/{max_retries})")

                if retry_count >= max_retries:
                    print("❌ 达到最大重试次数，切换到CPU模式")
                    # return force_cpu_training(gnnwr_instance, max_epoch, early_stop, print_frequency)

                # 处理OOM错误
                current_batch_size = memory_manager.handle_oom_error(gnnwr_instance, current_batch_size)

            else:
                print(f"❌ 训练错误: {e}")
                return False

    return False

def analyze_station_distribution():
    """分析站点数据分布"""
    print("=== 站点数据分布分析 ===")

    data = pd.read_excel('lu_onehot.xlsx')
    spatial_column = ['longitude', 'latitude']

    # 识别站点
    unique_stations = data[spatial_column].drop_duplicates()
    print(f"总站点数: {len(unique_stations)}")

    # 为每个站点分配ID并统计样本数
    station_stats = []
    for idx, (_, row) in enumerate(unique_stations.iterrows()):
        lon, lat = row[spatial_column[0]], row[spatial_column[1]]
        station_data = data[(data[spatial_column[0]] == lon) & (data[spatial_column[1]] == lat)]
        station_stats.append({
            'station_id': idx,
            'longitude': lon,
            'latitude': lat,
            'sample_count': len(station_data),
            'mean_swe': station_data['swe'].mean() if 'swe' in station_data.columns else 0
        })

    station_df = pd.DataFrame(station_stats)

    print(f"\n站点数据分布:")
    print(f"平均每个站点样本数: {station_df['sample_count'].mean():.2f}")
    print(f"样本数统计:")
    print(station_df['sample_count'].describe())

    # 保存站点分布信息
    os.makedirs('result', exist_ok=True)
    station_df.to_csv('result/station_distribution.csv', index=False)
    print(f"\n站点分布信息已保存到: result/station_distribution.csv")

    return station_df


if __name__ == "__main__":
    # 首先分析站点分布
    device = setup_device(0)

    station_info = analyze_station_distribution()

    # 执行基于站点的10折交叉验证
    print("\n开始基于站点的10折交叉验证...")
    results, predictions = station_based_kfold_cross_validation()

    if results is not None:
        print("\n🎉 基于站点的10折交叉验证完成！")
        print("结果文件保存在: result/station_kfold/")
    else:
        print("\n❌ 交叉验证失败")