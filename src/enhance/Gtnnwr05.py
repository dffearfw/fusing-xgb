import os
import sys
import warnings
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from torch import nn
from scipy.spatial import distance
from sklearn.model_selection import GroupKFold
import optuna  # 🔥【新增】导入 Optuna

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR
from visualizer import plot_gtnnwr_results, plot_multiple_models_results


# ----------------------------------------------------------------------
# --- 🔥【封装的修复补丁】修复 gnnwr 库的内部 bug ---
# ----------------------------------------------------------------------
# (这里的 patched_reg_result 函数保持不变)
def patched_reg_result(self, filename=None, model_path=None, use_dict=False, only_return=False, map_location=None):
    if model_path is None:
        model_path = self._modelSavePath + "/" + self._modelName + ".pkl"
    if use_dict:
        data = torch.load(model_path, map_location=map_location, weights_only=False)
        self._model.load_state_dict(data)
    else:
        self._model = torch.load(model_path, map_location=map_location, weights_only=False)
    if self._use_gpu:
        self._model = nn.DataParallel(module=self._model)
        self._model, self._out = self._model.cuda(), self._out.cuda()
    else:
        self._model, self._out = self._model.cpu(), self._out.cpu()
    device = torch.device('cuda') if self._use_gpu else torch.device('cpu')
    result = torch.tensor([]).to(torch.float32).to(device)
    train_data_size = valid_data_size = 0
    with torch.no_grad():
        for data, coef, label, data_index in self._train_dataset.dataloader:
            data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(device)
            output = self._out(self._model(data).mul(coef.to(torch.float32)))
            coefficient = self._model(data).mul(torch.tensor(self._coefficient).to(torch.float32).to(device))
            output = torch.cat((coefficient, output, data_index), dim=1)
            result = torch.cat((result, output), 0)
        train_data_size = len(result)
        for data, coef, label, data_index in self._valid_dataset.dataloader:
            data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(device)
            output = self._out(self._model(data).mul(coef.to(torch.float32)))
            coefficient = self._model(data).mul(torch.tensor(self._coefficient).to(torch.float32).to(device))
            output = torch.cat((coefficient, output, data_index), dim=1)
            result = torch.cat((result, output), 0)
        valid_data_size = len(result) - train_data_size
        for data, coef, label, data_index in self._test_dataset.dataloader:
            data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(device)
            output = self._out(self._model(data).mul(coef.to(torch.float32)))
            coefficient = self._model(data).mul(torch.tensor(self._coefficient).to(torch.float32).to(device))
            output = torch.cat((coefficient, output, data_index), dim=1)
            result = torch.cat((result, output), 0)
    result = result.cpu().detach().numpy()
    columns = list(self._train_dataset.x)
    for i in range(len(columns)):
        columns[i] = "coef_" + columns[i]
    columns.append("bias")
    columns = columns + ["Pred_" + self._train_dataset.y[0]] + self._train_dataset.id
    result = pd.DataFrame(result, columns=columns)
    result[self._train_dataset.id] = result[self._train_dataset.id].astype(np.int32)
    result["Pred_" + self._train_dataset.y[0]] = result["Pred_" + self._train_dataset.y[0]].astype(np.float32)
    result['dataset_belong'] = np.concatenate([
        np.full(train_data_size, 'train'),
        np.full(valid_data_size, 'valid'),
        np.full(len(result) - train_data_size - valid_data_size, 'test')
    ])
    pred_col_name = "Pred_" + self._train_dataset.y[0]

    if self._train_dataset.y[0] == 'swe_log':
        print("检测到对数变换目标，正在对预测结果进行反向变换...")
        result['denormalized_pred_result'] = np.expm1(result[pred_col_name])
    elif self._train_dataset.y_scale_info:
        _, denormalized_pred = self._train_dataset.rescale(None, result[pred_col_name].to_frame())
        result['denormalized_pred_result'] = denormalized_pred.iloc[:, 0]
    else:
        result['denormalized_pred_result'] = result[pred_col_name]

    if only_return:
        return result
    if filename is not None:
        result.to_csv(filename, index=False)
    else:
        warnings.warn(
            "Warning! The input write file path is not set. Result is returned by function but not saved as file.",
            RuntimeWarning)
    return result


# ----------------------------------------------------------------------
# --- 主流程 ---
# ----------------------------------------------------------------------
# 🔥【关键】在创建任何模型之前，应用修复补丁
GTNNWR.reg_result = patched_reg_result

# --- 【前置步骤：数据准备、基于 zone_id 的空间分区和站点划分】---
# ----------------------------------------------------------------------
print("=== 1. 加载并预处理数据 ===")
data = pd.read_excel('lu_onehot.xlsx')
# 确保数据中包含 zone_id 列
if 'zone_id' not in data.columns:
    raise ValueError("数据文件 'lu_onehot.xlsx' 中未找到 'zone_id' 列。")

data["id"] = np.arange(len(data))
data['station_id'] = data['X'].astype(str) + '_' + data['Y'].astype(str)
data['swe_log'] = np.log1p(data['swe'])

# 🔥【核心修改】直接使用 zone_id 作为空间分区（cluster）的依据
# 将 zone_id 转换为从0开始的整数类别，方便后续处理
data['cluster'] = data['zone_id'].astype('category').cat.codes
print(f"数据已根据 'zone_id' 划分为 {data['cluster'].nunique()} 个空间分区。")

# 定义特征和目标列
x_columns = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation', 'std_slope', 'std_eastness',
             'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high', 'std_aspect', 'glsnow', 'cswe',
             'snow_depth_snow_depth', 'ERA5温度_ERA5温度', 'era5_swe', 'gldas', 'scp_start', 'scp_end', 'd1', 'd2',
             'da', 'db', 'dc', 'dd', 'landuse_11', 'landuse_12', 'landuse_21', 'landuse_22', 'landuse_23', 'landuse_24',
             'landuse_31', 'landuse_32', 'landuse_33', 'landuse_41', 'landuse_43', 'landuse_46', 'landuse_51',
             'landuse_52', 'landuse_53', 'landuse_62', 'landuse_64']
y_column_transformed = ['swe_log']

print("\n=== 2. 基于 zone_id (cluster) 划分训练/验证集和测试集 ===")
# 🔥【核心修改】按空间分区（cluster）来划分测试集，以评估空间外推能力
all_clusters = sorted(data['cluster'].unique())
np.random.seed(42) # 设置随机种子以保证结果可复现
np.random.shuffle(all_clusters)

# 选择 20% 的分区作为测试区域
test_clusters = all_clusters[:int(len(all_clusters) * 0.2)]
train_val_clusters = all_clusters[int(len(all_clusters) * 0.2):]

print(f"测试分区: {sorted(test_clusters)}")
print(f"训练/验证分区: {sorted(train_val_clusters)}")

train_val_data_full = data[data['cluster'].isin(train_val_clusters)].copy()
test_data_full = data[data['cluster'].isin(test_clusters)].copy()

print(f"训练/验证集样本数: {len(train_val_data_full)}")
print(f"测试集样本数: {len(test_data_full)}")

print("\n=== 3. 在训练/验证集和测试集内部按时间划分 ===")
# 训练/验证集的时间划分：取最后20%的时间作为验证集
train_val_df_sorted = train_val_data_full.sort_values(by=['year', 'month', 'doy'])
val_sample_count = int(len(train_val_df_sorted) * 0.2)
val_data_full = train_val_df_sorted.iloc[-val_sample_count:].copy() # 使用最后一段时间作为验证集
train_data_full = train_val_df_sorted.iloc[:-val_sample_count].copy()

# 测试集的时间划分：确保测试集的时间范围与验证集有重叠，以进行公平比较
test_df_sorted = test_data_full.sort_values(by=['year', 'month', 'doy'])
# 筛选出与验证集年份相同的测试数据
min_year, max_year = val_data_full['year'].min(), val_data_full['year'].max()
test_data = test_df_sorted[(test_df_sorted['year'] >= min_year) & (test_df_sorted['year'] <= max_year)].copy()

print(f"最终训练集样本数: {len(train_data_full)}")
print(f"最终验证集样本数: {len(val_data_full)}")
print(f"最终测试集样本数 (时间筛选后): {len(test_data)}")

# 合并训练集和验证集的站点信息，为 GroupKFold 交叉验证做准备
train_val_stations = list(set(train_data_full['station_id'].unique()) | set(val_data_full['station_id'].unique()))
train_val_data_full = data[data['station_id'].isin(train_val_stations)].copy()
# --- 前置步骤结束 ---


# ----------------------------------------------------------------------
# --- 🔥【优化版】使用 Optuna 进行超参数搜索 ---
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# --- 🔥【优化版】使用 Optuna 进行超参数搜索 ---
# ----------------------------------------------------------------------
def objective(trial):
    """
    一个更健壮的 Optuna 目标函数，包含更广泛的搜索空间、剪枝和错误处理。
    """
    # 1. 定义更广泛的超参数搜索空间
    # 🔥【优化器选择】让 Optuna 选择优化器
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'Adadelta', 'AdamW'])

    # 🔥【学习率选择】为不同优化器设置不同的合理范围
    if optimizer_name == 'Adam':
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    elif optimizer_name == 'AdamW':
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    else:  # Adadelta
        lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)

    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

    # 🔥【网络结构】动态定义网络层数和单元数
    n_layers = trial.suggest_int('n_layers', 1, 4)
    layers = []
    for i in range(n_layers):
        num_units = trial.suggest_int(f'n_units_l{i}', 32, 512, step=32)
        layers.append(num_units)
    hidden_dims = [[3], layers]  # STPNN 和 SWNN 的结构

    # 🔥【调度器选择】让 Optuna 选择学习率调度器
    scheduler_name = trial.suggest_categorical('scheduler', ['MultiStepLR', 'CosineAnnealingLR'])

    # 🔥【调度器参数】根据选择的调度器动态配置参数
    if scheduler_name == 'MultiStepLR':
        # 动态建议 2 到 4 个里程碑点
        n_milestones = trial.suggest_int('n_milestones', 2, 4)

        # 🔥【修复】分两步生成 milestones
        # 第一步：收集所有建议的浮点数
        milestone_floats = [trial.suggest_float(f'milestone_{i}', 0.2, 0.8, step=0.2) for i in range(n_milestones)]
        # 第二步：将浮点数转换为整数并排序
        milestones = sorted([int(m * 200) for m in milestone_floats])

        scheduler_gamma = trial.suggest_float('scheduler_gamma', 0.5, 0.9)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler_T_max = trial.suggest_int('scheduler_T_max', 100, 500)
        scheduler_eta_min = trial.suggest_float('scheduler_eta_min', 1e-4, 1e-2, log=True)

    print(f"\n--- Trial {trial.number}: Testing params ---")
    print(f"  Optimizer: {optimizer_name}, LR: {lr:.5f}, Dropout: {dropout:.2f}, Weight Decay: {weight_decay:.2e}")
    print(f"  Layers: {hidden_dims}, Scheduler: {scheduler_name}")
    if scheduler_name == 'MultiStepLR':
        print(f"  Milestones: {milestones}, Gamma: {scheduler_gamma:.2f}")
    else:
        print(f"  T_max: {scheduler_T_max}, Eta_min: {scheduler_eta_min:.5f}")

    # 2. 设置交叉验证
    N_SPLITS = 5  # 增加折数，使评估更稳定

    # 🔥【修复】GroupKFold 不支持 shuffle 和 random_state，直接移除
    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_scores = []

    # 3. 遍历每一折
    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(train_val_data_full, groups=train_val_data_full['station_id'])):
        print(f"  Fold {fold + 1}/{N_SPLITS}...")

        fold_train_data = train_val_data_full.iloc[train_idx]
        fold_val_data = train_val_data_full.iloc[val_idx]

        # 🔥【关键修复】确保每次试验的数据划分都是基于该试验的随机种子
        fold_train_dataset, fold_val_dataset, _ = init_dataset_split(
            train_data=fold_train_data, val_data=fold_val_data, test_data=fold_val_data,
            x_column=x_columns, y_column=y_column_transformed,
            spatial_column=['X', 'Y', 'Z'], temp_column=['doy', 'year', 'month'],
            id_column=['id'], use_model="gtnnwr", batch_size=128, dropna=True
        )

        # 🔥【关键修复】构建 optimizer_params 字典
        optimizer_params_cv = {
            "weight_decay": weight_decay,
            "scheduler": scheduler_name,
        }
        if scheduler_name == 'MultiStepLR':
            optimizer_params_cv["scheduler_milestones"] = milestones
            optimizer_params_cv["scheduler_gamma"] = scheduler_gamma
        elif scheduler_name == 'CosineAnnealingLR':
            optimizer_params_cv["scheduler_T_max"] = scheduler_T_max
            optimizer_params_cv["scheduler_eta_min"] = scheduler_eta_min

        # 🔥【关键修复】通过 start_lr=lr 传递学习率
        model_cv = GTNNWR(
            fold_train_dataset, fold_val_dataset, fold_val_dataset,
            hidden_dims, drop_out=dropout,
            optimizer=optimizer_name, start_lr=lr, optimizer_params=optimizer_params_cv,
            write_path=f"../demo_result/optuna_runs/trial_{trial.number}",
            model_name=f"fold_{fold + 1}"
        )

        try:
            # 运行模型，设置早停
            model_cv.run(max_epoch=200, early_stop=30)  # 增加早停的耐心值

            # 🔥【剪枝核心】向 Optuna 报告中间结果
            # 我们使用验证损失作为剪枝的依据
            current_val_loss = model_cv._validLossList[-1]
            trial.report(current_val_loss, fold + 1)

            # 🔥【剪枝判断】检查是否应该剪枝
            if trial.should_prune():
                print(f"    !!! Fold {fold + 1} was pruned.")
                raise optuna.exceptions.TrialPruned()

        except torch._C._LinAlgError:
            print(f"    !!! Fold {fold + 1} failed due to a singular matrix. Pruning this trial.")
            raise optuna.exceptions.TrialPruned()
        except RuntimeError as e:
            # 捕获如 CUDA OOM 等运行时错误
            print(f"    !!! Fold {fold + 1} failed with a RuntimeError: {e}. Pruning this trial.")
            raise optuna.exceptions.TrialPruned()
        except Exception as e:
            print(f"    !!! Fold {fold + 1} failed with an unexpected error: {e}. Pruning this trial.")
            raise optuna.exceptions.TrialPruned()

        # 记录当前折的最终验证损失
        score = model_cv._validLossList[-1]
        fold_scores.append(score)
        print(f"    >> Fold {fold + 1} finished with validation loss: {score:.6f}")

        # 🔥【资源清理】彻底清理模型和缓存
        del model_cv
        torch.cuda.empty_cache()

    # 4. 返回当前超参数组合的平均验证损失
    # 如果有折被剪枝，fold_scores 可能不完整，需要处理
    if len(fold_scores) < N_SPLITS:
        print(f"  >> Trial {trial.number} was pruned early.")
        raise optuna.exceptions.TrialPruned()

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"  >> Trial {trial.number} finished. Mean Loss: {mean_score:.6f}, Std: {std_score:.6f}")

    return mean_score


# ----------------------------------------------------------------------
# --- 运行 Optuna 研究 ---
# ----------------------------------------------------------------------
print("\n=== 开始 Optuna 超参数搜索 ===")
# 创建一个 study 对象，目标是 'minimize' (最小化验证损失)
study = optuna.create_study(direction='minimize')
# 运行优化，例如尝试 50 次不同的超参数组合
study.optimize(objective, n_trials=50, timeout=3600) # 增加一个超时时间（秒）

print("\n=== Optuna 搜索完成 ===")
print(f"最优参数: {study.best_params}")
print(f"最优验证损失: {study.best_value:.6f}")

# 可以可视化优化过程
# try:
#     import optuna.visualization as vis
#     fig = vis.plot_optimization_history(study)
#     fig.show()
# except ImportError:
#     print("请安装 plotly 以可视化 Optuna 结果: pip install plotly")


# ----------------------------------------------------------------------
# --- 🔥【最终训练】使用 Optuna 找到的最优超参数训练最终模型 ---
# ----------------------------------------------------------------------
print("\n=== 使用 Optuna 最优超参数训练最终模型 ===")

best_params = study.best_params

# 🔥【关键】将 Optuna 的参数转换为 GTNNWR 需要的格式
final_hidden_dims = [[3], []]
for i in range(best_params['n_layers']):
    final_hidden_dims[1].append(best_params[f'n_units_l{i}'])

# 为了在最终训练时监控性能，我们从 train_val_data_full 中再划分一个小的验证集
# 注意：这里的划分方式需要与交叉验证时一致，即按站点分组
# 但为了简单，我们直接使用之前划分好的 train_data_full 和 val_data_full
final_train_dataset, final_val_dataset, final_test_dataset = init_dataset_split(
    train_data=train_data_full, val_data=val_data_full, test_data=test_data,
    x_column=x_columns, y_column=y_column_transformed,
    spatial_column=['X', 'Y', 'Z'], temp_column=['doy', 'year', 'month'],
    id_column=['id'], use_model="gtnnwr", batch_size=128, dropna=True
)

# 🔥【关键】根据 Optuna 找到的最佳调度器类型来构建参数
optimizer_params_final = {
    "weight_decay": best_params['weight_decay'],
    "scheduler": best_params['scheduler'],
}
if best_params['scheduler'] == 'MultiStepLR':
    # 在最终训练中，可以使用更通用的milestones
    optimizer_params_final["scheduler_milestones"] = [200, 400, 600, 800]
    optimizer_params_final["scheduler_gamma"] = 0.7
elif best_params['scheduler'] == 'CosineAnnealingLR':
    optimizer_params_final["scheduler_T_max"] = 500
    optimizer_params_final["scheduler_eta_min"] = 1e-5


final_model = GTNNWR(
    final_train_dataset, final_val_dataset, final_test_dataset,
    final_hidden_dims, drop_out=best_params['dropout'],
    optimizer=best_params['optimizer'], optimizer_params=optimizer_params_final,
    # 🔥【关键修复】通过 start_lr=best_params['lr'] 传递学习率
    start_lr=best_params['lr'],
    write_path="../demo_result/final_model_optuna",
    model_name="GTNNWR_Optuna_Best"
)
final_model.add_graph()
final_model.run(max_epoch=1000, early_stop=100) # 使用更多的epoch进行充分训练

# ----------------------------------------------------------------------
print("\n=== 在独立测试集上评估最终模型 ===")

# 🔥【关键修复4】使用 reg_result 来获取所有预测结果，而不是访问不存在的 _test_dataset.pred
final_results_df = final_model.reg_result(only_return=True)

# 从结果DataFrame中筛选出测试集的预测和真实值
test_results_df = final_results_df[final_results_df['dataset_belong'] == 'test'].copy()

# 🔥【关键修复5】使用正确的列名 'Pred_swe_log' 来获取预测值
# 你的 y_column_transformed 是 ['swe_log']，所以预测列名是 'Pred_' + 'swe_log'
pred_log = test_results_df['Pred_swe_log'].values
# 🔥【关键修复6】从原始数据中获取真实值，而不是 DataFrame
# 假设你的原始数据 test_data 中有 'swe' 列
true_original_scale = test_data['swe'].values

# 将对数预测结果还原为原始尺度
pred_original_scale = np.expm1(pred_log)

print("已获取测试集的真实值和预测值，并还原为原始尺度，准备进行最终评估。")

# 使用 visualizer 进行评估和绘图
# 注意：plot_gtnnwr_results 函数可能需要调整，因为它期望的是一个模型对象
# 如果不能直接修改，我们可以手动计算指标
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(true_original_scale, pred_original_scale)
rmse = np.sqrt(mean_squared_error(true_original_scale, pred_original_scale))
mae = mean_absolute_error(true_original_scale, pred_original_scale)

print("\n=== 最终评估指标 ===")
print(f"R2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# 如果你仍然想用 plot_gtnnwr_results，可能需要创建一个虚拟的模型对象来传递数据
# 或者修改该函数以接受 DataFrame 作为输入。这里我们先手动计算并绘图。
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.scatter(true_original_scale, pred_original_scale, alpha=0.5, label='Data Points')
plt.plot([min(true_original_scale), max(true_original_scale)], [min(true_original_scale), max(true_original_scale)], '--', color='red', label='Ideal Fit')
plt.xlabel('True Values (Original Scale)')
plt.ylabel('Predictions (Original Scale)')
plt.title('True vs. Predicted Values on Test Set')
plt.legend()
plt.grid(True)
save_path = "../demo_result/final_model_optuna/GTNNWR_Optuna_Best_Results.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()
