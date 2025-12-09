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

# ==========================================
# 核心验证：检查权重提取是否正确
# ==========================================
print("\n" + "=" * 60)
print("核心验证：检查权重提取是否正确")
print("=" * 60)


def verify_weight_extraction_core(gnnwr_instance, dataset, dataset_name="dataset"):
    """
    核心验证：检查提取的权重是否能重现模型预测

    逻辑：
    1. 用模型直接预测一批数据得到 y_model
    2. 提取该批数据的权重矩阵 W
    3. 计算 y_calc = Σ(W_ij × β_ols_j × x_ij)
    4. 比较 y_model 和 y_calc
    """
    print(f"\n=== 核心验证：{dataset_name} ===")

    model = gnnwr_instance._model
    model.eval()
    device = gnnwr_instance._device

    # 获取OLS系数
    if isinstance(gnnwr_instance._coefficient, list):
        ols_coeff = np.array(gnnwr_instance._coefficient)
    else:
        ols_coeff = gnnwr_instance._coefficient

    # 收集所有批次的验证结果
    all_model_preds = []
    all_calc_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataset.dataloader):
            if len(batch) >= 3:
                distances, features, labels = batch[:3]

                # 1. 用模型直接预测
                distances_device = distances.to(device)
                model_output = model(distances_device)  # 这就是权重矩阵 W
                model_output_cpu = model_output.cpu()

                # 2. 计算模型预测的y（假设model(distances)返回的就是权重）
                # 我们需要先确认模型是如何预测的
                # 暂时先跳过这步，直接计算验证

                # 3. 用提取的权重计算预测
                features_tensor = features.float()
                ols_tensor = torch.tensor(ols_coeff).float()

                # 计算 y_calc = Σ(W_ij × β_ols_j × x_ij)
                weighted_coeff = model_output_cpu * ols_tensor  # W × β
                y_calc = torch.sum(weighted_coeff * features_tensor, dim=1)

                # 存储结果
                all_calc_preds.append(y_calc.numpy())
                all_labels.append(labels.numpy())

                # 4. 获取模型的实际预测值（通过gnnwr.predict）
                # 我们稍后统一比较

                # 只分析第一批
                if batch_idx == 0:
                    print(f"第一批数据验证:")
                    print(f"  距离矩阵形状: {distances.shape}")
                    print(f"  特征矩阵形状: {features.shape}")
                    print(f"  提取的权重W形状: {model_output.shape}")

                    # 检查第一个样本
                    sample_idx = 0
                    print(f"\n  样本{sample_idx}详细计算:")

                    # 获取数据
                    w_sample = model_output_cpu[sample_idx].numpy()  # 权重向量
                    x_sample = features_tensor[sample_idx].numpy()  # 特征向量
                    beta = ols_coeff  # OLS系数

                    print(f"    权重W范围: [{w_sample.min():.4f}, {w_sample.max():.4f}]")
                    print(f"    特征X范围: [{x_sample.min():.4f}, {x_sample.max():.4f}]")
                    print(f"    系数β范围: [{beta.min():.4f}, {beta.max():.4f}]")

                    # 计算各项
                    w_beta = w_sample * beta  # W × β
                    contributions = w_beta * x_sample  # (W × β) × X

                    print(f"\n    计算过程:")
                    print(f"      sum(W): {w_sample.sum():.4f}")
                    print(f"      sum(β): {beta.sum():.4f}")
                    print(f"      sum(X): {x_sample.sum():.4f}")
                    print(f"      sum(W×β): {w_beta.sum():.4f}")
                    print(f"      sum((W×β)×X): {contributions.sum():.4f}")

                    y_calc_sample = y_calc[sample_idx].item()
                    y_true_sample = labels[sample_idx].item()
                    print(f"\n    结果:")
                    print(f"      计算预测值 y_calc: {y_calc_sample:.4f}")
                    print(f"      实际标签 y_true: {y_true_sample:.4f}")
                    print(f"      绝对误差: {abs(y_calc_sample - y_true_sample):.4f}")

    # 合并所有批次
    if all_calc_preds:
        y_calc_all = np.concatenate(all_calc_preds, axis=0)
        y_labels_all = np.concatenate(all_labels, axis=0)

        # 计算与真实标签的误差
        mse_vs_true = np.mean((y_calc_all - y_labels_all) ** 2)
        mae_vs_true = np.mean(np.abs(y_calc_all - y_labels_all))

        print(f"\n整体统计（计算预测 vs 真实标签）:")
        print(f"  总样本数: {len(y_calc_all)}")
        print(f"  MSE: {mse_vs_true:.6f}")
        print(f"  RMSE: {np.sqrt(mse_vs_true):.6f}")
        print(f"  MAE: {mae_vs_true:.6f}")

        return y_calc_all
    else:
        print("没有数据")
        return None


# ==========================================
# 关键验证：与模型原始预测比较
# ==========================================
print("\n" + "=" * 60)
print("关键验证：计算预测 vs 模型原始预测")
print("=" * 60)

# 对训练集进行验证
print("\n1. 对训练集验证:")
y_calc_train = verify_weight_extraction_core(gnnwr, train_set, "训练集")

# 获取模型对训练集的原始预测
print("\n2. 获取模型原始预测:")
original_train_pred = gnnwr.predict(train_set)
original_y_train = original_train_pred['pred_result'].values

print(f"原始模型预测样本数: {len(original_y_train)}")
print(f"原始预测统计:")
print(f"  均值: {original_y_train.mean():.4f}")
print(f"  标准差: {original_y_train.std():.4f}")
print(f"  范围: [{original_y_train.min():.4f}, {original_y_train.max():.4f}]")

# 比较计算预测和模型预测
if y_calc_train is not None and len(y_calc_train) == len(original_y_train):
    print("\n3. 比较结果:")

    mse = np.mean((y_calc_train - original_y_train) ** 2)
    mae = np.mean(np.abs(y_calc_train - original_y_train))
    corr = np.corrcoef(y_calc_train, original_y_train)[0, 1]

    print(f"  MSE（计算预测 vs 模型预测）: {mse:.8f}")
    print(f"  RMSE: {np.sqrt(mse):.8f}")
    print(f"  MAE: {mae:.8f}")
    print(f"  相关系数: {corr:.8f}")

    # 检查差异
    diff = y_calc_train - original_y_train
    print(f"\n  差异分析:")
    print(f"    差异均值: {diff.mean():.8f}")
    print(f"    差异标准差: {diff.std():.8f}")
    print(f"    最大正向差异: {diff.max():.8f}")
    print(f"    最大负向差异: {diff.min():.8f}")

    # 判断标准
    if mse < 1e-6:
        print("\n✅ 权重提取验证通过！MSE < 1e-6")
        print("  说明：提取的权重矩阵 W 是正确的")
        print("  公式：y = Σ(W_ij × β_ols_j × x_ij) 成立")
    elif mse < 1e-4:
        print("\n⚠️ 权重提取基本正确，但有轻微误差")
        print(f"  MSE = {mse:.8f}，可能存在数值精度问题")
    else:
        print("\n❌ 权重提取不正确！")
        print(f"  MSE = {mse:.8f} 太大")
        print("  可能原因：")
        print("    1. model(distances) 返回的不是权重矩阵 W")
        print("    2. 预测公式不是 y = Σ(W_ij × β_ols_j × x_ij)")
        print("    3. 需要额外的处理（如激活函数、归一化）")
else:
    print(f"\n⚠️ 样本数不匹配：")
    print(f"  计算预测样本数: {len(y_calc_train) if y_calc_train is not None else 'None'}")
    print(f"  模型预测样本数: {len(original_y_train)}")

# ==========================================
# 诊断：检查几个关键样本
# ==========================================
print("\n" + "=" * 60)
print("诊断：检查关键样本")
print("=" * 60)

# 找到差异最大的样本
if y_calc_train is not None and len(y_calc_train) == len(original_y_train):
    diff_abs = np.abs(y_calc_train - original_y_train)
    max_diff_idx = np.argmax(diff_abs)

    print(f"\n差异最大的样本（索引 {max_diff_idx}）:")
    print(f"  计算预测值: {y_calc_train[max_diff_idx]:.8f}")
    print(f"  模型预测值: {original_y_train[max_diff_idx]:.8f}")
    print(f"  绝对差异: {diff_abs[max_diff_idx]:.8f}")
    print(f"  相对差异: {diff_abs[max_diff_idx] / abs(original_y_train[max_diff_idx]):.2%}")

    # 检查几个随机样本
    print(f"\n随机检查几个样本:")
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_calc_train), min(5, len(y_calc_train)), replace=False)

    for idx in sample_indices:
        calc_val = y_calc_train[idx]
        model_val = original_y_train[idx]
        abs_diff = abs(calc_val - model_val)
        rel_diff = abs_diff / abs(model_val) if model_val != 0 else float('inf')

        print(
            f"  样本 {idx}: 计算={calc_val:.6f}, 模型={model_val:.6f}, 绝对差异={abs_diff:.6f}, 相对差异={rel_diff:.2%}")

# ==========================================
# 结论
# ==========================================
print("\n" + "=" * 60)
print("验证结论")
print("=" * 60)

if y_calc_train is not None and len(y_calc_train) == len(original_y_train):
    mse = np.mean((y_calc_train - original_y_train) ** 2)

    if mse < 1e-6:
        print("✅ 验证成功！")
        print("提取的权重矩阵 W 是正确的。")
        print("可以安全地使用这些权重进行后续分析。")
    else:
        print("❌ 验证失败！")
        print(f"MSE = {mse:.8f} 太大，权重提取可能有问题。")
        print("\n建议：")
        print("1. 检查GNNWR论文中的确切预测公式")
        print("2. 查看GNNWR源代码中model(distances)的返回值")
        print("3. 可能需要调整公式：")
        print("   - y = Σ(model_output * features)  # 假设model_output已是W×β")
        print("   - y = Σ(model_output) + Σ(β×X)    # 加法形式")
        print("   - y = f(Σ(W×β×X))                 # 有激活函数")

        # 提供调试建议
        print("\n调试建议：")
        print("1. 检查第一批数据的详细计算")
        print("2. 查看model_output的值是否合理")
        print("3. 尝试不同的公式组合")
else:
    print("⚠️ 验证无法完成（样本数不匹配）")