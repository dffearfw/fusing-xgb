# 尝试提取堆叠后的空间权重
import os
import sys
from gnnwr import models, datasets, utils
import pandas as pd
import torch.nn as nn
import numpy as np
import torch

# 加载数据
data = pd.read_excel('D:/pyworkspace/fusing xgb/src/pre-process/aggregated_station_data.xlsx')

# 打乱数据并划分
data = data.sample(frac=1, random_state=42)
indices = data.index.tolist()
train_idx = indices[:int(0.7 * len(data))]
val_idx = indices[int(0.7 * len(data)):int(0.8 * len(data))]
test_idx = indices[int(0.8 * len(data)):]

train_data = data.loc[train_idx]
val_data = data.loc[val_idx]
test_data = data.loc[test_idx]

# 定义列名
x_column = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation', 'std_slope',
            'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high', 'std_aspect',
            'glsnow', 'cswe', 'snow_depth_snow_depth', 'ERA5温度_ERA5温度', 'era5_swe', 'doy', 'gldas',
            'year', 'month', 'scp_start', 'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da', 'db', 'dc', 'dd']
y_column = ['swe']
spatial_column = ['longitude', 'latitude']

# 初始化数据集
train_set, val_set, test_set = datasets.init_dataset_split(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    x_column=x_column,
    y_column=y_column,
    spatial_column=spatial_column,
    batch_size=128,
    use_model="gnnwr"
)

# 创建权重提取目录
os.makedirs('result/weights', exist_ok=True)

# 定义优化器参数
optimizer_params = {
    "scheduler": "MultiStepLR",
    "scheduler_milestones": [500, 1000, 1500, 2000],
    "scheduler_gamma": 0.75,
}

# 初始化GNNWR模型
gnnwr = models.GNNWR(
    train_dataset=train_set,
    valid_dataset=val_set,
    test_dataset=test_set,
    dense_layers=[1024, 512, 256],
    activate_func=nn.PReLU(init=0.4),
    start_lr=0.1,
    optimizer="Adadelta",
    model_name="GNNWR_PM25",
    model_save_path="result/gnnwr_models",
    log_path="result/gnnwr_logs",
    write_path="result/gnnwr_runs",
    optimizer_params=optimizer_params
)

# 添加图结构
gnnwr.add_graph()

# 训练模型
gnnwr.run(max_epoch=4000, early_stop=1000, print_frequency=500)

# 显示结果
gnnwr.result()


# ==========================================
# 新增代码：提取特征维度的空间变权重矩阵
# ==========================================

def extract_feature_space_weights(gnnwr_instance, dataset, dataset_name="dataset"):
    """
    提取特征维度的空间变权重矩阵

    参数:
    gnnwr_instance: 训练好的GNNWR实例
    dataset: 数据集（train_set, val_set, test_set）
    dataset_name: 数据集名称，用于输出信息

    返回:
    feature_weights: (n_samples, n_features+1) 特征空间变权重矩阵
    ids: 对应的样本ID
    """
    print(f"\n=== 提取 {dataset_name} 的特征空间变权重矩阵 ===")

    # 获取模型
    model = gnnwr_instance._model
    model.eval()

    # 获取设备信息
    device = gnnwr_instance._device
    use_gpu = gnnwr_instance._use_gpu

    all_feature_weights = []
    all_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataset.dataloader):
            if len(batch) == 4:
                distances, features, labels, ids = batch
            else:
                distances, features = batch[:2]
                ids = batch[-1] if len(batch) > 2 else None

            # 移动到设备
            distances = distances.to(device)

            # 获取特征空间变权重（SWNN的输出）
            # 这就是每个样本的特征权重向量
            feature_weights = model(distances)

            # 存储结果
            all_feature_weights.append(feature_weights.cpu().numpy())
            if ids is not None:
                all_ids.append(ids.cpu().numpy())

            # 打印第一批信息
            if batch_idx == 0:
                print(f"第一批数据:")
                print(f"  输入距离矩阵形状: {distances.shape}")
                print(f"  输入特征形状: {features.shape}")
                print(f"  输出特征权重形状: {feature_weights.shape}")
                print(f"  特征权重范围: [{feature_weights.min().item():.4f}, {feature_weights.max().item():.4f}]")
                print(f"  特征权重均值: {feature_weights.mean().item():.4f}")

                # 检查权重矩阵的特性
                print(f"\n  第一个样本的特征权重（前5个）:")
                sample_weights = feature_weights[0].cpu().numpy()
                for i in range(min(5, len(sample_weights))):
                    print(f"    权重{i + 1}: {sample_weights[i]:.4f}")

    # 合并所有batch的结果
    if all_feature_weights:
        feature_weights_matrix = np.concatenate(all_feature_weights, axis=0)
        if all_ids:
            ids_array = np.concatenate(all_ids, axis=0)
        else:
            ids_array = np.arange(len(feature_weights_matrix))
    else:
        feature_weights_matrix = np.array([])
        ids_array = np.array([])

    print(f"\n{dataset_name} 特征空间变权重矩阵:")
    print(f"  形状: {feature_weights_matrix.shape}")
    print(f"  总样本数: {len(feature_weights_matrix)}")
    print(f"  每个样本的特征权重数: {feature_weights_matrix.shape[1] if len(feature_weights_matrix.shape) > 1 else 0}")

    return feature_weights_matrix, ids_array


def save_feature_weights_analysis(feature_weights, dataset_name, output_dir='result/weights'):
    """
    保存特征权重矩阵并进行分析

    参数:
    feature_weights: 特征权重矩阵
    dataset_name: 数据集名称
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(feature_weights) == 0:
        print(f"警告: {dataset_name} 的特征权重矩阵为空")
        return

    # 1. 保存原始权重矩阵
    npy_path = os.path.join(output_dir, f'{dataset_name}_feature_weights.npy')
    np.save(npy_path, feature_weights)
    print(f"  原始权重矩阵已保存: {npy_path}")

    # 2. 保存为CSV（便于查看）
    n_features = feature_weights.shape[1]
    columns = [f'weight_feature_{i}' for i in range(n_features)]

    weights_df = pd.DataFrame(feature_weights, columns=columns)
    csv_path = os.path.join(output_dir, f'{dataset_name}_feature_weights.csv')
    weights_df.to_csv(csv_path, index=False)
    print(f"  CSV格式已保存: {csv_path}")

    # 3. 创建统计分析报告
    stats = {}
    for i in range(n_features):
        col_name = f'weight_feature_{i}'
        col_data = feature_weights[:, i]
        stats[col_name] = {
            'mean': np.mean(col_data),
            'std': np.std(col_data),
            'min': np.min(col_data),
            'max': np.max(col_data),
            'median': np.median(col_data),
            'q25': np.percentile(col_data, 25),
            'q75': np.percentile(col_data, 75)
        }

    # 转换为DataFrame
    stats_df = pd.DataFrame(stats).T
    stats_path = os.path.join(output_dir, f'{dataset_name}_feature_weights_statistics.csv')
    stats_df.to_csv(stats_path)
    print(f"  统计分析已保存: {stats_path}")

    # 4. 打印摘要信息
    print(f"\n  {dataset_name} 特征权重统计摘要:")
    print(f"    整体均值: {np.mean(feature_weights):.4f}")
    print(f"    整体标准差: {np.std(feature_weights):.4f}")
    print(f"    整体最小值: {np.min(feature_weights):.4f}")
    print(f"    整体最大值: {np.max(feature_weights):.4f}")

    # 检查权重分布
    negative_ratio = np.sum(feature_weights < 0) / feature_weights.size
    zero_ratio = np.sum(np.abs(feature_weights) < 0.01) / feature_weights.size
    print(f"    负值比例: {negative_ratio:.2%}")
    print(f"    接近零值比例: {zero_ratio:.2%}")


# ==========================================
# 执行特征权重提取
# ==========================================
print("\n" + "=" * 60)
print("开始提取特征维度的空间变权重矩阵")
print("=" * 60)

# 提取训练集特征权重
feature_weights_train, train_ids = extract_feature_space_weights(gnnwr, train_set, "训练集")
save_feature_weights_analysis(feature_weights_train, "train")

# 提取验证集特征权重
feature_weights_val, val_ids = extract_feature_space_weights(gnnwr, val_set, "验证集")
save_feature_weights_analysis(feature_weights_val, "val")

# 提取测试集特征权重
feature_weights_test, test_ids = extract_feature_space_weights(gnnwr, test_set, "测试集")
save_feature_weights_analysis(feature_weights_test, "test")

# ==========================================
# 验证权重矩阵的数学性质
# ==========================================
print("\n" + "=" * 60)
print("验证特征权重矩阵的数学性质")
print("=" * 60)

# 1. 验证与OLS系数的关系
print("\n1. OLS系数验证:")
# 将列表转换为numpy数组
if isinstance(gnnwr._coefficient, list):
    coefficient_array = np.array(gnnwr._coefficient)
    print(f"   OLS系数类型: 列表 -> 已转换为numpy数组")
else:
    coefficient_array = gnnwr._coefficient
    print(f"   OLS系数类型: {type(coefficient_array).__name__}")

print(f"   OLS系数形状: {coefficient_array.shape}")
print(f"   OLS系数值（前5个）:")
for i in range(min(5, len(coefficient_array))):
    print(f"    系数{i + 1}: {coefficient_array[i]:.4f}")

# 检查维度是否匹配
if len(feature_weights_train.shape) > 1:
    n_features = feature_weights_train.shape[1]
    if n_features == len(coefficient_array):
        print(f"   ✅ 特征权重维度({n_features})与OLS系数维度({len(coefficient_array)})匹配")
    else:
        print(f"   ⚠️  特征权重维度({n_features})与OLS系数维度({len(coefficient_array)})不匹配")


# 2. 验证预测一致性（可选）
def verify_prediction_consistency(gnnwr_instance, dataset, feature_weights, dataset_name):
    """
    验证使用提取的权重是否能重现原始预测
    """
    print(f"\n2. {dataset_name} 预测一致性验证:")

    # 获取OLS系数
    if isinstance(gnnwr_instance._coefficient, list):
        coefficient_array = np.array(gnnwr_instance._coefficient)
    else:
        coefficient_array = gnnwr_instance._coefficient

    # 获取原始预测
    original_pred = gnnwr_instance.predict(dataset)
    original_y = original_pred['pred_result'].values

    # 使用提取的权重计算预测
    reconstructed_y = []
    start_idx = 0

    for batch_idx, batch in enumerate(dataset.dataloader):
        if len(batch) == 4:
            distances, features, labels, ids = batch
            batch_size = features.shape[0]

            # 获取对应的特征权重
            end_idx = start_idx + batch_size
            batch_weights = feature_weights[start_idx:end_idx]
            start_idx = end_idx  # 更新索引

            # 转换为tensor
            features_tensor = torch.tensor(features).float()
            weights_tensor = torch.tensor(batch_weights).float()
            ols_tensor = torch.tensor(coefficient_array).float()

            # 计算预测值: y = sum(w_ij * β_ols_j * x_ij)
            # 注意：features_tensor包含了偏置项（最后一列全1）
            weighted_coeff = weights_tensor * ols_tensor
            predictions = torch.sum(weighted_coeff * features_tensor, dim=1)

            reconstructed_y.extend(predictions.numpy())

    reconstructed_y = np.array(reconstructed_y)

    # 比较预测结果
    if len(original_y) == len(reconstructed_y):
        mse = np.mean((original_y - reconstructed_y) ** 2)
        mae = np.mean(np.abs(original_y - reconstructed_y))
        corr = np.corrcoef(original_y, reconstructed_y)[0, 1]

        print(f"   原始预测样本数: {len(original_y)}")
        print(f"   重建预测样本数: {len(reconstructed_y)}")
        print(f"   均方误差(MSE): {mse:.6f}")
        print(f"   平均绝对误差(MAE): {mae:.6f}")
        print(f"   相关系数: {corr:.6f}")

        if mse < 1e-4:
            print("   ✅ 预测一致性验证通过！")
        else:
            print(f"   ⚠️  预测存在差异，MSE={mse:.6f}")
    else:
        print(f"   样本数不匹配: 原始={len(original_y)}, 重建={len(reconstructed_y)}")
        print(f"   差异: {abs(len(original_y) - len(reconstructed_y))} 个样本")


# 执行验证
verify_prediction_consistency(gnnwr, train_set, feature_weights_train, "训练集")

# ==========================================
# 为GNNW-XGBoost准备数据
# ==========================================
print("\n" + "=" * 60)
print("为GNNW-XGBoost准备数据")
print("=" * 60)


def prepare_gnnw_xgboost_data(gnnwr_instance, dataset, feature_weights, dataset_name):
    """
    为GNNW-XGBoost准备输入数据

    返回:
    X_combined: 组合特征 [原始特征, 特征权重]
    y: 目标变量
    """
    print(f"\n准备 {dataset_name} 的GNNW-XGBoost数据:")

    all_features = []
    all_labels = []

    for batch_idx, batch in enumerate(dataset.dataloader):
        if len(batch) == 4:
            distances, features, labels, ids = batch

            # 获取对应的特征权重
            start_idx = batch_idx * features.shape[0]
            end_idx = start_idx + features.shape[0]
            batch_weights = feature_weights[start_idx:end_idx]

            # 原始特征（去除偏置项，因为偏置已包含在特征权重中）
            # features形状: (batch, 35)，其中最后一列是偏置（全1）
            original_features = features[:, :-1].numpy()  # (batch, 34)

            # 组合特征: [原始特征, 特征权重]
            combined_features = np.hstack([original_features, batch_weights])

            all_features.append(combined_features)
            all_labels.append(labels.numpy())

    if all_features:
        X_combined = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)

        print(f"  原始特征维度: {original_features.shape[1]}")
        print(f"  特征权重维度: {batch_weights.shape[1]}")
        print(f"  组合特征总维度: {X_combined.shape[1]}")
        print(f"  总样本数: {X_combined.shape[0]}")

        return X_combined, y
    else:
        return np.array([]), np.array([])


# 准备训练数据
X_train_gnnw, y_train = prepare_gnnw_xgboost_data(gnnwr, train_set, feature_weights_train, "训练集")
X_val_gnnw, y_val = prepare_gnnw_xgboost_data(gnnwr, val_set, feature_weights_val, "验证集")
X_test_gnnw, y_test = prepare_gnnw_xgboost_data(gnnwr, test_set, feature_weights_test, "测试集")

# 保存GNNW-XGBoost数据
if len(X_train_gnnw) > 0:
    np.save('result/weights/X_train_gnnw_xgboost.npy', X_train_gnnw)
    np.save('result/weights/y_train.npy', y_train)
    np.save('result/weights/X_val_gnnw_xgboost.npy', X_val_gnnw)
    np.save('result/weights/y_val.npy', y_val)
    np.save('result/weights/X_test_gnnw_xgboost.npy', X_test_gnnw)
    np.save('result/weights/y_test.npy', y_test)

    print("\nGNNW-XGBoost数据已保存:")
    print(f"  训练集: X={X_train_gnnw.shape}, y={y_train.shape}")
    print(f"  验证集: X={X_val_gnnw.shape}, y={y_val.shape}")
    print(f"  测试集: X={X_test_gnnw.shape}, y={y_test.shape}")

# ==========================================
# 总结报告
# ==========================================
print("\n" + "=" * 60)
print("特征权重提取完成总结")
print("=" * 60)

print(f"\n1. 提取的特征权重矩阵维度:")
print(f"   训练集: {feature_weights_train.shape} (样本数×特征权重数)")
print(f"   验证集: {feature_weights_val.shape}")
print(f"   测试集: {feature_weights_test.shape}")

print(f"\n2. 特征权重统计:")
print(f"   训练集 - 均值: {np.mean(feature_weights_train):.4f}, 标准差: {np.std(feature_weights_train):.4f}")
print(f"   验证集 - 均值: {np.mean(feature_weights_val):.4f}, 标准差: {np.std(feature_weights_val):.4f}")
print(f"   测试集 - 均值: {np.mean(feature_weights_test):.4f}, 标准差: {np.std(feature_weights_test):.4f}")

print(f"\n3. GNNW-XGBoost输入特征维度:")
if len(X_train_gnnw) > 0:
    print(f"   原始特征数: {len(x_column)} = {len(x_column)}")
    print(f"   特征权重数: {feature_weights_train.shape[1]}")
    print(f"   总特征数: {X_train_gnnw.shape[1]}")
    print(f"   GNNW-XGBoost特征 = 原始特征 + GNNWR特征权重")

print(f"\n4. 输出文件:")
print(f"   特征权重矩阵: result/weights/*_feature_weights.npy")
print(f"   统计分析: result/weights/*_feature_weights_statistics.csv")
print(f"   GNNW-XGBoost数据: result/weights/*_gnnw_xgboost.npy")

print("\n" + "=" * 60)
print("完成！现在可以基于这些特征权重构建GNNW-XGBoost模型")
print("=" * 60)