# 既然swnn是torch.nn.Module的子类，那么尝试使用state_dict()方法
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
# 方法1：验证模型类型并获取state_dict
# ==========================================
print("\n=== 验证模型类型 ===")
print(f"gnnwr 类型: {type(gnnwr)}")
print(f"gnnwr._model 类型: {type(gnnwr._model)}")
print(f"Is gnnwr._model a nn.Module? {isinstance(gnnwr._model, torch.nn.Module)}")

# 获取模型的state_dict
print("\n=== 获取模型参数 ===")
model_state_dict = gnnwr._model.state_dict()
print(f"模型参数数量: {len(model_state_dict)}")
print("模型参数键名:")
for key in model_state_dict.keys():
    print(f"  {key}: {model_state_dict[key].shape}")

# 保存模型参数
torch.save(model_state_dict, 'result/weights/model_state_dict.pth')
print("模型参数已保存到: result/weights/model_state_dict.pth")




# ==========================================
# 方法2：使用state_dict提取权重
# ==========================================
def extract_weights_using_model(gnnwr_instance, dataset):
    """
    使用训练好的模型提取权重矩阵
    """
    # 获取模型
    model = gnnwr_instance._model
    model.eval()

    # 获取设备
    device = gnnwr_instance._device

    all_weights = []
    all_ids = []

    with torch.no_grad():
        for batch in dataset.dataloader:
            distances, coef, label, ids = batch

            distances = distances.to(device)

            # 使用模型计算权重
            weights = model(distances)

            all_weights.append(weights.cpu().numpy())
            all_ids.append(ids.cpu().numpy())

    if all_weights:
        weights_matrix = np.concatenate(all_weights, axis=0)
        ids_array = np.concatenate(all_ids, axis=0)
    else:
        weights_matrix = np.array([])
        ids_array = np.array([])

    return weights_matrix, ids_array


# ==========================================
def debug_model_dimensions(gnnwr_instance, dataset):
    """
    调试模型的实际输入输出维度
    """
    gnnwr_instance._model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataset.dataloader):
            if i >= 1:  # 只看第一个batch
                break

            distances, coef, label, ids = batch

            print(f"\n=== Batch {i + 1} 维度分析 ===")
            print(f"1. 输入数据维度:")
            print(f"   distances形状: {distances.shape}")
            print(f"   coef形状: {coef.shape}")
            print(f"   label形状: {label.shape}")
            print(f"   ids形状: {ids.shape}")

            # 查看模型参数
            print(f"\n2. 模型第一层参数:")
            for name, param in gnnwr_instance._model.named_parameters():
                if 'layer.weight' in name and '0' in name:
                    print(f"   {name}: {param.shape}")
                    break

            # 前向传播
            distances = distances.to(gnnwr_instance._device)
            weight_output = gnnwr_instance._model(distances)

            print(f"\n3. 模型输出:")
            print(f"   weight_output形状: {weight_output.shape}")

            # 计算最终输出
            coef = coef.to(gnnwr_instance._device)
            final_output = gnnwr_instance._out(weight_output.mul(coef))
            print(f"   final_output形状: {final_output.shape}")

            # OLS系数
            print(f"\n4. OLS系数:")
            print(f"   OLS系数形状: {gnnwr_instance._coefficient.shape}")
            print(f"   OLS系数值: {gnnwr_instance._coefficient}")

            return distances.shape, weight_output.shape


# 方法3：更直接的方式（推荐）
# ==========================================
def extract_spatial_weights_directly(gnnwr_instance, dataset):
    """
    直接使用模型提取空间权重
    """
    # 检查模型状态
    print(f"模型是否在训练模式: {gnnwr_instance._model.training}")

    # 设置为评估模式
    gnnwr_instance._model.eval()

    device = gnnwr_instance._device
    use_gpu = gnnwr_instance._use_gpu

    all_spatial_weights = []
    all_sample_weights = []  # 每个样本的权重
    all_ids = []

    print(f"使用设备: {device}")
    print(f"是否使用GPU: {use_gpu}")

    with torch.no_grad():
        for i, batch in enumerate(dataset.dataloader):
            if len(batch) == 4:
                distances, features, labels, ids = batch
            else:
                # 可能是预测模式
                distances, features = batch[:2]
                ids = batch[-1] if len(batch) > 2 else None
                labels = None

            # 移动到设备
            distances = distances.to(device)

            # 提取权重
            spatial_weights = gnnwr_instance._model(distances)

            # 转换到CPU并存储
            spatial_weights_cpu = spatial_weights.cpu().numpy()
            all_spatial_weights.append(spatial_weights_cpu)

            if ids is not None:
                all_ids.append(ids.cpu().numpy())

            # 打印第一批信息用于调试
            if i == 0:
                print(f"第一批数据形状:")
                print(f"  distances形状: {distances.shape}")
                print(f"  spatial_weights形状: {spatial_weights.shape}")
                print(f"  权重范围: [{spatial_weights.min().item():.4f}, {spatial_weights.max().item():.4f}]")
                print(f"  权重均值: {spatial_weights.mean().item():.4f}")

    # 合并结果
    if all_spatial_weights:
        spatial_weights_matrix = np.concatenate(all_spatial_weights, axis=0)
    else:
        spatial_weights_matrix = np.array([])

    if all_ids:
        ids_array = np.concatenate(all_ids, axis=0)
    else:
        ids_array = np.arange(len(spatial_weights_matrix))

    return spatial_weights_matrix, ids_array


# ==========================================
# 方法4：通过预测系数反向计算权重
# ==========================================
def extract_weights_from_coefficients(gnnwr_instance):
    """
    通过回归系数反向计算权重
    权重 = 空间系数 / OLS系数
    """
    # 获取OLS系数
    ols_coefficients = gnnwr_instance._coefficient.flatten()
    print(f"OLS系数形状: {ols_coefficients.shape}")
    print(f"OLS系数值: {ols_coefficients}")

    # 获取回归结果
    coef_data = gnnwr_instance.getCoefs()

    # 提取系数列
    coef_columns = [col for col in coef_data.columns if col.startswith('coef_')]
    print(f"找到的系数列: {coef_columns}")

    # 计算权重
    weights_data = coef_data.copy()
    for i, col in enumerate(coef_columns):
        if i < len(ols_coefficients):
            weight_col = f'weight_{col[5:]}' if col.startswith('coef_') else f'weight_{i}'
            weights_data[weight_col] = coef_data[col] / ols_coefficients[i]

    return weights_data


# ==========================================
# 执行权重提取
# ==========================================
print("\n" + "=" * 60)
print("开始提取权重矩阵...")
print("=" * 60)

# 方法A：直接提取空间权重
print("\n[方法A] 直接提取空间权重")
try:
    spatial_weights_train, train_ids = extract_spatial_weights_directly(gnnwr, train_set)
    spatial_weights_val, val_ids = extract_spatial_weights_directly(gnnwr, val_set)
    spatial_weights_test, test_ids = extract_spatial_weights_directly(gnnwr, test_set)

    print(f"训练集空间权重形状: {spatial_weights_train.shape}")
    print(f"验证集空间权重形状: {spatial_weights_val.shape}")
    print(f"测试集空间权重形状: {spatial_weights_test.shape}")

    # 保存空间权重
    np.save('result/weights/spatial_weights_train.npy', spatial_weights_train)
    np.save('result/weights/spatial_weights_val.npy', spatial_weights_val)
    np.save('result/weights/spatial_weights_test.npy', spatial_weights_test)

    print("空间权重已保存")

except Exception as e:
    print(f"方法A失败: {e}")
    import traceback

    traceback.print_exc()

# 方法B：通过系数反向计算权重
print("\n[方法B] 通过系数反向计算权重")
try:
    weights_data = extract_weights_from_coefficients(gnnwr)

    print(f"权重数据形状: {weights_data.shape}")
    print(f"权重数据列: {list(weights_data.columns)}")

    # 保存权重数据
    weights_data.to_csv('result/weights/weights_from_coefficients.csv', index=False)

    # 按数据集分离
    if 'dataset_belong' in weights_data.columns:
        train_weights = weights_data[weights_data['dataset_belong'] == 'train']
        val_weights = weights_data[weights_data['dataset_belong'] == 'valid']
        test_weights = weights_data[weights_data['dataset_belong'] == 'test']

        print(f"训练集权重: {len(train_weights)} 个样本")
        print(f"验证集权重: {len(val_weights)} 个样本")
        print(f"测试集权重: {len(test_weights)} 个样本")

        # 保存分开的数据集
        train_weights.to_csv('result/weights/train_weights_from_coef.csv', index=False)
        val_weights.to_csv('result/weights/val_weights_from_coef.csv', index=False)
        test_weights.to_csv('result/weights/test_weights_from_coef.csv', index=False)

    print("系数权重已保存")

except Exception as e:
    print(f"方法B失败: {e}")
    import traceback

    traceback.print_exc()

# ==========================================
# 分析提取的权重
# ==========================================
print("\n" + "=" * 60)
print("权重分析")
print("=" * 60)

# 加载并分析权重
try:
    if 'spatial_weights_train' in locals():
        print(f"\n空间权重分析:")
        print(f"训练集 - 最小值: {spatial_weights_train.min():.6f}, 最大值: {spatial_weights_train.max():.6f}")
        print(f"训练集 - 均值: {spatial_weights_train.mean():.6f}, 标准差: {spatial_weights_train.std():.6f}")

        # 创建权重统计
        weight_stats = {
            'dataset': ['train', 'val', 'test'],
            'min': [
                spatial_weights_train.min(),
                spatial_weights_val.min() if 'spatial_weights_val' in locals() else None,
                spatial_weights_test.min() if 'spatial_weights_test' in locals() else None
            ],
            'max': [
                spatial_weights_train.max(),
                spatial_weights_val.max() if 'spatial_weights_val' in locals() else None,
                spatial_weights_test.max() if 'spatial_weights_test' in locals() else None
            ],
            'mean': [
                spatial_weights_train.mean(),
                spatial_weights_val.mean() if 'spatial_weights_val' in locals() else None,
                spatial_weights_test.mean() if 'spatial_weights_test' in locals() else None
            ],
            'std': [
                spatial_weights_train.std(),
                spatial_weights_val.std() if 'spatial_weights_val' in locals() else None,
                spatial_weights_test.std() if 'spatial_weights_test' in locals() else None
            ]
        }

        stats_df = pd.DataFrame(weight_stats)
        stats_df.to_csv('result/weights/weight_statistics.csv', index=False)
        print("\n权重统计已保存:")
        print(stats_df)

except Exception as e:
    print(f"权重分析失败: {e}")

# ==========================================
# 保存模型架构信息
# ==========================================
print("\n" + "=" * 60)
print("保存模型信息")
print("=" * 60)

# 保存模型架构
try:
    # 获取模型架构
    model_str = str(gnnwr._model)

    with open('result/weights/model_architecture.txt', 'w') as f:
        f.write("GNNWR模型架构:\n")
        f.write("=" * 50 + "\n")
        f.write(model_str)

    # 保存参数信息
    param_info = []
    total_params = 0
    for name, param in gnnwr._model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            param_info.append(f"{name}: {param.shape} | {param_count} 参数")

    with open('result/weights/model_parameters.txt', 'w') as f:
        f.write("模型参数信息:\n")
        f.write("=" * 50 + "\n")
        f.write(f"总参数数量: {total_params:,}\n")
        f.write("=" * 50 + "\n")
        for info in param_info:
            f.write(info + "\n")

    print(f"模型架构已保存到: result/weights/model_architecture.txt")
    print(f"模型参数信息已保存到: result/weights/model_parameters.txt")
    print(f"模型总参数: {total_params:,}")

except Exception as e:
    print(f"保存模型信息失败: {e}")

print("\n" + "=" * 60)
print("权重提取完成！")
print("=" * 60)
print("开始调试维度...")
input_shape, weight_shape = debug_model_dimensions(gnnwr, train_set)