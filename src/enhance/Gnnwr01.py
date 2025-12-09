import os
import sys
from gnnwr import models, datasets, utils
import pandas as pd
import torch.nn as nn
import numpy as np
import torch

# 加载数据
data = pd.read_excel('D:/pyworkspace/fusing xgb/src/pre-process/aggregated_station_data.xlsx')
data.head(5)

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


# 定义权重提取函数
def extract_weights_from_model(gnnwr_instance, dataset):
    """
    从训练好的GNNWR模型中提取权重矩阵

    参数:
    gnnwr_instance: GNNWR模型实例
    dataset: 数据集（train_set, val_set或test_set）

    返回:
    weights: 空间权重矩阵
    ids: 对应的ID数组
    """
    # 从GNNWR实例中获取模型
    model = gnnwr_instance._model
    model.eval()

    # 获取设备信息
    device = gnnwr_instance._device
    use_gpu = gnnwr_instance._use_gpu

    all_weights = []
    all_ids = []

    with torch.no_grad():
        for batch in dataset.dataloader:
            if len(batch) == 4:  # (distances, x_data, y_data, id_data)
                distances, coef, label, ids = batch
            else:  # 预测数据集的情况
                distances, coef = batch[:2]
                ids = batch[-1] if len(batch) > 2 else None

            # 将数据移动到设备
            distances = distances.to(device)

            # 提取权重
            weights = model(distances)

            all_weights.append(weights.cpu().numpy())
            if ids is not None:
                all_ids.append(ids.cpu().numpy())

    if all_weights:
        weights_matrix = np.concatenate(all_weights, axis=0)
    else:
        weights_matrix = np.array([])

    if all_ids:
        ids_array = np.concatenate(all_ids, axis=0)
    else:
        ids_array = np.array([])

    return weights_matrix, ids_array


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

# 提取权重矩阵
print("\n正在提取权重矩阵...")
weights_train, ids_train = extract_weights_from_model(gnnwr, train_set)
weights_val, ids_val = extract_weights_from_model(gnnwr, val_set)
weights_test, ids_test = extract_weights_from_model(gnnwr, test_set)

# 打印权重矩阵信息
print(f"训练集权重矩阵形状: {weights_train.shape}")
print(f"训练集ID形状: {ids_train.shape}")
print(f"验证集权重矩阵形状: {weights_val.shape}")
print(f"验证集ID形状: {ids_val.shape}")
print(f"测试集权重矩阵形状: {weights_test.shape}")
print(f"测试集ID形状: {ids_test.shape}")

# 保存权重矩阵
np.save('result/weights/weights_train.npy', weights_train)
np.save('result/weights/weights_val.npy', weights_val)
np.save('result/weights/weights_test.npy', weights_test)
np.save('result/weights/ids_train.npy', ids_train)
np.save('result/weights/ids_val.npy', ids_val)
np.save('result/weights/ids_test.npy', ids_test)


# 将权重保存为CSV格式，便于查看
def save_weights_to_csv(weights, ids, filename):
    """将权重保存为CSV文件"""
    # 创建列名
    n_features = weights.shape[1]
    columns = [f'weight_{i}' for i in range(n_features)]

    # 创建DataFrame
    weights_df = pd.DataFrame(weights, columns=columns)
    weights_df.insert(0, 'id', ids)

    # 保存
    weights_df.to_csv(filename, index=False)
    print(f"权重已保存到: {filename}")

    return weights_df


# 保存为CSV
train_weights_df = save_weights_to_csv(weights_train, ids_train, 'result/weights/weights_train.csv')
val_weights_df = save_weights_to_csv(weights_val, ids_val, 'result/weights/weights_val.csv')
test_weights_df = save_weights_to_csv(weights_test, ids_test, 'result/weights/weights_test.csv')

# 显示部分权重数据
print("\n训练集权重前5行:")
print(train_weights_df.head())
print("\n权重描述性统计:")
print(train_weights_df.describe())


# 创建汇总报告
def create_weight_summary(weights_df, dataset_name):
    """创建权重摘要报告"""
    weight_cols = [col for col in weights_df.columns if col.startswith('weight_')]

    if not weight_cols:
        return None

    summary = {
        'dataset': dataset_name,
        'n_samples': len(weights_df),
        'n_weights': len(weight_cols),
        'mean_weights': weights_df[weight_cols].mean().mean(),
        'std_weights': weights_df[weight_cols].std().mean(),
        'min_weights': weights_df[weight_cols].min().min(),
        'max_weights': weights_df[weight_cols].max().max(),
    }

    return summary


# 生成摘要
summary_train = create_weight_summary(train_weights_df, 'train')
summary_val = create_weight_summary(val_weights_df, 'val')
summary_test = create_weight_summary(test_weights_df, 'test')

# 保存摘要
summary_list = []
if summary_train: summary_list.append(summary_train)
if summary_val: summary_list.append(summary_val)
if summary_test: summary_list.append(summary_test)

if summary_list:
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv('result/weights/weight_summary.csv', index=False)

    print("\n权重摘要:")
    print(summary_df)

print("\n所有权重矩阵已成功提取并保存！")


# 可选：分析权重的额外功能
def analyze_weights_extra(weights_df, dataset_name, original_data=None):
    """
    额外的权重分析功能
    """
    print(f"\n=== {dataset_name} 权重分析 ===")

    # 基本统计
    weight_cols = [col for col in weights_df.columns if col.startswith('weight_')]

    if weight_cols:
        print(f"权重数量: {len(weight_cols)}")
        print(f"权重值范围: [{weights_df[weight_cols].min().min():.4f}, {weights_df[weight_cols].max().max():.4f}]")
        print(f"权重均值: {weights_df[weight_cols].mean().mean():.4f}")
        print(f"权重标准差: {weights_df[weight_cols].std().mean():.4f}")

        # 检查权重是否有负值
        has_negative = (weights_df[weight_cols] < 0).any().any()
        if has_negative:
            print("警告: 权重中包含负值！")

        # 检查权重是否接近0或1
        near_zero = ((weights_df[weight_cols].abs() < 0.01).sum().sum()) / (len(weights_df) * len(weight_cols))
        near_one = ((weights_df[weight_cols].abs() - 1).abs() < 0.01).sum().sum() / (len(weights_df) * len(weight_cols))

        print(f"接近0的权重比例: {near_zero:.2%}")
        print(f"接近1的权重比例: {near_one:.2%}")

    # 如果提供了原始数据，可以分析权重与特征的关系
    if original_data is not None and 'id' in weights_df.columns:
        # 合并权重和原始数据
        merged = pd.merge(weights_df, original_data, on='id', how='left')
        print(f"成功合并权重和原始数据，合并后数据形状: {merged.shape}")

    return weights_df


# 分析各数据集的权重
print("\n进行权重详细分析...")
analyze_weights_extra(train_weights_df, '训练集', data)
analyze_weights_extra(val_weights_df, '验证集', data)
analyze_weights_extra(test_weights_df, '测试集', data)

# 检查权重矩阵的特性
print("\n=== 权重矩阵特性检查 ===")
print(f"权重数据类型: {weights_train.dtype}")
print(f"权重是否包含NaN值: {np.isnan(weights_train).any()}")
print(f"权重是否包含Inf值: {np.isinf(weights_train).any()}")
print(f"权重矩阵稀疏度: {(np.abs(weights_train) < 1e-6).sum() / weights_train.size:.2%}")

# 保存训练好的模型以供后续使用
print("\n=== 保存模型和配置 ===")
# 模型已经通过gnnwr自动保存，这里可以保存额外的配置信息

config_info = {
    'x_columns': x_column,
    'y_column': y_column,
    'spatial_columns': spatial_column,
    'model_name': 'GNNWR_PM25',
    'weights_shape': {
        'train': weights_train.shape,
        'val': weights_val.shape,
        'test': weights_test.shape
    }
}

import json

with open('result/weights/model_config.json', 'w') as f:
    json.dump(config_info, f, indent=2, ensure_ascii=False)

print("模型配置已保存到: result/weights/model_config.json")
print("\n所有任务完成！")