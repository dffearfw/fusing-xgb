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
gnnwr.run(max_epoch=11, early_stop=1000, print_frequency=500)

# 显示结果
gnnwr.result()


## ==========================================
# 修复和验证部分
# ==========================================

# 首先，让我们检查模型输出的实际内容
print("\n" + "=" * 60)
print("检查模型输出类型")
print("=" * 60)

# 获取一个批次的数据进行检查
for batch_idx, batch in enumerate(train_set.dataloader):
    if len(batch) >= 2:
        distances, features = batch[:2]
        break

# 获取模型输出
model = gnnwr._model
model.eval()
device = gnnwr._device
distances_device = distances.to(device)

with torch.no_grad():
    model_output = model(distances_device)

print(f"输入距离矩阵形状: {distances.shape}")
print(f"输入特征形状: {features.shape}")
print(f"模型输出形状: {model_output.shape}")
print(f"模型输出范围: [{model_output.min().item():.4f}, {model_output.max().item():.4f}]")
print(f"模型输出均值: {model_output.mean().item():.4f}")
print(f"模型输出标准差: {model_output.std().item():.4f}")

# 检查模型输出是否可能是权重（应该每个样本有35个值）
if model_output.shape[1] == 35:
    print("✅ 模型输出维度与特征数匹配（35）")
    # 检查权重特性
    weights_sample = model_output[0].cpu().numpy()
    print(f"第一个样本的权重统计:")
    print(f"  权重和: {weights_sample.sum():.4f}")
    print(f"  权重均值的绝对值: {np.abs(weights_sample).mean():.4f}")
    print(f"  权重标准差: {weights_sample.std():.4f}")

    # 检查前5个权重值
    print(f"  前5个权重值:")
    for i in range(5):
        print(f"    w_{i}: {weights_sample[i]:.4f}")
else:
    print(f"⚠️ 模型输出维度({model_output.shape[1]})不等于特征数(35)")

# ==========================================
# 方法1：尝试直接验证预测公式
# ==========================================
print("\n" + "=" * 60)
print("方法1：直接验证预测公式")
print("=" * 60)

# 使用提取的"权重"计算预测
with torch.no_grad():
    # 获取模型输出的"权重"
    weights = model(distances_device)

    # 将数据移到CPU进行计算
    features_tensor = features.float()
    weights_cpu = weights.cpu()

    # 获取OLS系数
    if isinstance(gnnwr._coefficient, list):
        ols_coeff = np.array(gnnwr._coefficient)
    else:
        ols_coeff = gnnwr._coefficient

    ols_tensor = torch.tensor(ols_coeff).float()

    # 计算预测值：y = sum(w_ij * β_ols_j * x_ij)
    # 注意：特征矩阵的最后一列是偏置项（全1）
    weighted_coeff = weights_cpu * ols_tensor
    predictions = torch.sum(weighted_coeff * features_tensor, dim=1)

    print(f"计算得到的预测值形状: {predictions.shape}")
    print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"预测值均值: {predictions.mean():.4f}")

    # 获取该批次的实际标签
    if len(batch) >= 3:
        labels = batch[2]
        print(f"实际标签范围: [{labels.min():.4f}, {labels.max():.4f}]")
        print(f"实际标签均值: {labels.mean():.4f}")

        # 计算误差
        mse = torch.mean((predictions - labels) ** 2).item()
        mae = torch.mean(torch.abs(predictions - labels)).item()
        print(f"该批次预测误差:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")

# ==========================================
# 方法2：检查GNNWR原始预测
# ==========================================
print("\n" + "=" * 60)
print("方法2：检查GNNWR原始预测")
print("=" * 60)

# 获取原始预测
original_pred = gnnwr.predict(train_set)
print(f"原始预测结果形状: {original_pred.shape}")
print(f"原始预测值统计:")
print(f"  均值: {original_pred['pred_result'].mean():.4f}")
print(f"  标准差: {original_pred['pred_result'].std():.4f}")
print(f"  范围: [{original_pred['pred_result'].min():.4f}, {original_pred['pred_result'].max():.4f}]")

# ==========================================
# 方法3：重新设计验证函数
# ==========================================
def verify_prediction_with_debug(gnnwr_instance, dataset, dataset_name="dataset"):
    """
    带调试信息的预测验证
    """
    print(f"\n=== 调试验证：{dataset_name} ===")

    model = gnnwr_instance._model
    model.eval()
    device = gnnwr_instance._device

    # 获取OLS系数
    if isinstance(gnnwr_instance._coefficient, list):
        ols_coeff = np.array(gnnwr_instance._coefficient)
    else:
        ols_coeff = gnnwr_instance._coefficient

    all_predictions = []
    all_labels = []
    all_weights = []
    all_features = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataset.dataloader):
            if len(batch) >= 3:
                distances, features, labels = batch[:3]

                # 移动到设备
                distances_device = distances.to(device)
                features_tensor = features.float()

                # 获取模型输出
                model_output = model(distances_device)
                weights = model_output.cpu()

                # 存储数据用于分析
                all_weights.append(weights.numpy())
                all_features.append(features.numpy())

                # 使用提取的权重计算预测
                ols_tensor = torch.tensor(ols_coeff).float()
                weighted_coeff = weights * ols_tensor
                batch_predictions = torch.sum(weighted_coeff * features_tensor, dim=1)

                all_predictions.append(batch_predictions.numpy())
                all_labels.append(labels.numpy())

                # 只打印第一批的详细信息
                if batch_idx == 0:
                    print(f"第一批数据:")
                    print(f"  距离矩阵形状: {distances.shape}")
                    print(f"  特征形状: {features.shape}")
                    print(f"  模型输出形状: {model_output.shape}")
                    print(f"  模型输出均值: {model_output.mean().item():.4f}")

                    # 检查一个样本的计算过程
                    sample_idx = 0
                    print(f"\n  样本{sample_idx}的详细计算:")
                    print(f"    特征值（前5个）: {features_tensor[sample_idx, :5].numpy()}")
                    print(f"    权重值（前5个）: {weights[sample_idx, :5].numpy()}")
                    print(f"    OLS系数（前5个）: {ols_coeff[:5]}")

                    # 计算加权系数
                    weighted = weights[sample_idx].numpy() * ols_coeff
                    print(f"    加权系数（前5个）: {weighted[:5]}")

                    # 计算贡献
                    contributions = weighted * features_tensor[sample_idx].numpy()
                    print(f"    特征贡献（前5个）: {contributions[:5]}")

                    # 预测值
                    pred_value = batch_predictions[sample_idx].item()
                    true_value = labels[sample_idx].item() if len(labels) > sample_idx else None
                    print(f"    计算预测值: {pred_value:.4f}")
                    if true_value is not None:
                        print(f"    实际标签值: {true_value:.4f}")
                        print(f"    误差: {abs(pred_value - true_value):.4f}")

    # 合并所有批次
    if all_predictions:
        predictions_array = np.concatenate(all_predictions, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)

        print(f"\n整体统计:")
        print(f"  总样本数: {len(predictions_array)}")
        print(f"  预测值均值: {np.mean(predictions_array):.4f}")
        print(f"  预测值标准差: {np.std(predictions_array):.4f}")
        print(f"  标签均值: {np.mean(labels_array):.4f}")
        print(f"  标签标准差: {np.std(labels_array):.4f}")

        # 计算误差
        mse = np.mean((predictions_array - labels_array) ** 2)
        mae = np.mean(np.abs(predictions_array - labels_array))
        corr = np.corrcoef(predictions_array, labels_array)[0, 1]

        print(f"\n误差分析:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {np.sqrt(mse):.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  相关系数: {corr:.6f}")

        # 检查权重统计
        if all_weights:
            weights_array = np.concatenate(all_weights, axis=0)
            print(f"\n权重统计:")
            print(f"  权重均值: {np.mean(weights_array):.4f}")
            print(f"  权重标准差: {np.std(weights_array):.4f}")
            print(f"  权重范围: [{np.min(weights_array):.4f}, {np.max(weights_array):.4f}]")

            # 检查每个特征的权重统计
            print(f"\n各特征权重统计（前5个特征）:")
            for i in range(min(5, weights_array.shape[1])):
                w_col = weights_array[:, i]
                print(f"  特征{i}: 均值={np.mean(w_col):.4f}, 标准差={np.std(w_col):.4f}, 范围=[{np.min(w_col):.4f}, {np.max(w_col):.4f}]")

        return predictions_array, labels_array
    else:
        print("没有数据")
        return None, None

# ==========================================
# 执行验证
# ==========================================
print("\n" + "=" * 60)
print("执行综合验证")
print("=" * 60)

# 对训练集进行验证
train_pred, train_labels = verify_prediction_with_debug(gnnwr, train_set, "训练集")

# 获取原始预测进行比较
if train_pred is not None:
    original_train_pred = gnnwr.predict(train_set)['pred_result'].values

    print("\n" + "=" * 60)
    print("比较提取权重预测 vs 原始GNNWR预测")
    print("=" * 60)

    print(f"提取权重预测样本数: {len(train_pred)}")
    print(f"原始GNNWR预测样本数: {len(original_train_pred)}")

    # 确保长度一致
    min_len = min(len(train_pred), len(original_train_pred))
    if min_len > 0:
        train_pred_trimmed = train_pred[:min_len]
        original_pred_trimmed = original_train_pred[:min_len]

        mse = np.mean((train_pred_trimmed - original_pred_trimmed) ** 2)
        mae = np.mean(np.abs(train_pred_trimmed - original_pred_trimmed))
        corr = np.corrcoef(train_pred_trimmed, original_pred_trimmed)[0, 1]

        print(f"\n比较结果:")
        print(f"  MSE (与原始预测): {mse:.6f}")
        print(f"  RMSE: {np.sqrt(mse):.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  相关系数: {corr:.6f}")

        # 分析差异
        diff = train_pred_trimmed - original_pred_trimmed
        print(f"\n差异分析:")
        print(f"  差异均值: {np.mean(diff):.6f}")
        print(f"  差异标准差: {np.std(diff):.6f}")
        print(f"  最大正向差异: {np.max(diff):.6f}")
        print(f"  最大负向差异: {np.min(diff):.6f}")

        if mse < 1e-4:
            print("✅ 预测一致性验证通过！")
        else:
            print(f"❌ 预测存在显著差异，MSE={mse:.6f}")

# ==========================================
# 诊断可能的问题
# ==========================================
print("\n" + "=" * 60)
print("问题诊断")
print("=" * 60)

# 检查几个可能性：

# 1. 检查权重是否可能是经过变换的
print("\n1. 检查权重特性:")
if 'all_weights' in locals() and len(all_weights) > 0:
    weights_sample = all_weights[0][0]  # 第一个批次的第一个样本
    print(f"  样本权重和: {weights_sample.sum():.4f}")
    print(f"  样本权重均值的绝对值: {np.abs(weights_sample).mean():.4f}")

    # 检查权重是否接近1（如果是乘法权重）
    abs_diff_from_one = np.abs(weights_sample - 1.0)
    print(f"  与1的平均绝对偏差: {np.mean(abs_diff_from_one):.4f}")

    # 检查权重是否可能是加法项而不是乘法项
    print(f"  权重平方和: {np.sum(weights_sample**2):.4f}")

# 2. 尝试不同的解释
print("\n2. 尝试不同的解释:")
print("  可能性1: model(distances) 输出的是 w_ij * β_ols_j 的乘积")
print("  可能性2: model(distances) 输出的是经过某种变换的权重")
print("  可能性3: 预测公式可能不是简单的 w_ij * β_ols_j * x_ij")

# 3. 建议的解决方案
print("\n3. 建议的解决方案:")
print("  方案A: 检查GNNWR论文中的确切公式")
print("  方案B: 查看GNNWR库的源代码，了解model(distances)的具体含义")
print("  方案C: 尝试 model(distances) * ols_coefficient * features 的不同组合")
print("  方案D: 可能权重需要归一化或经过激活函数")

# ==========================================
# 尝试方案C：不同的组合
# ==========================================
print("\n" + "=" * 60)
print("尝试不同的预测公式组合")
print("=" * 60)

# 用第一批数据测试
for batch_idx, batch in enumerate(train_set.dataloader):
    if len(batch) >= 3:
        distances, features, labels = batch[:3]
        break

with torch.no_grad():
    distances_device = distances.to(device)
    model_output = model(distances_device).cpu()
    features_tensor = features.float()
    ols_tensor = torch.tensor(ols_coeff).float()

    print(f"测试不同公式组合:")

    # 组合1: y = sum(model_output * features)  # 假设model_output已经是加权系数
    pred1 = torch.sum(model_output * features_tensor, dim=1)
    error1 = torch.mean((pred1 - labels) ** 2).item()
    print(f"  公式1: y = sum(model_output * features)")
    print(f"    MSE: {error1:.4f}")

    # 组合2: y = sum((model_output * ols_coeff) * features)
    pred2 = torch.sum((model_output * ols_tensor) * features_tensor, dim=1)
    error2 = torch.mean((pred2 - labels) ** 2).item()
    print(f"  公式2: y = sum((model_output * β_ols) * features)")
    print(f"    MSE: {error2:.4f}")

    # 组合3: y = sum(model_output) + sum(β_ols * features)  # 加法形式
    pred3 = torch.sum(model_output, dim=1) + torch.sum(ols_tensor * features_tensor, dim=1)
    error3 = torch.mean((pred3 - labels) ** 2).item()
    print(f"  公式3: y = sum(model_output) + sum(β_ols * features)")
    print(f"    MSE: {error3:.4f}")

    # 组合4: y = sum(model_output) * sum(β_ols * features)  # 乘法形式
    pred4 = torch.sum(model_output, dim=1) * torch.sum(ols_tensor * features_tensor, dim=1)
    error4 = torch.mean((pred4 - labels) ** 2).item()
    print(f"  公式4: y = sum(model_output) * sum(β_ols * features)")
    print(f"    MSE: {error4:.4f}")

    # 找出最佳组合
    errors = [error1, error2, error3, error4]
    best_idx = np.argmin(errors)
    best_formula = [1, 2, 3, 4][best_idx]
    print(f"\n  最佳公式: 公式{best_formula}, MSE={errors[best_idx]:.4f}")

    # 如果最佳公式是公式2，说明我们的原始理解可能是正确的
    if best_idx == 1:
        print("  ✅ 公式2表现最好，说明原始理解可能正确")
    else:
        print(f"  ⚠️  公式{best_formula}表现最好，可能需要调整理解")