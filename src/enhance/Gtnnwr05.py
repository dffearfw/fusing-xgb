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

# --- 【前置步骤：数据准备、教师模型训练、聚类、站点划分】---
# (这部分代码与之前完全相同，为了简洁我省略了，请确保它在这里运行)
# ... (代码从 data = pd.read_excel('lu_onehot.xlsx') 到 test_data = test_df_sorted[...].copy() )
# ... (确保 train_val_data_full 和 test_data 已经准备好)
# --- 为了代码完整性，我将这部分复制过来 ---
data = pd.read_excel('lu_onehot.xlsx')
data["id"] = np.arange(len(data))
data['station_id'] = data['X'].astype(str) + '_' + data['Y'].astype(str)
data['swe_log'] = np.log1p(data['swe'])
x_columns = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation', 'std_slope', 'std_eastness',
             'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high', 'std_aspect', 'glsnow', 'cswe',
             'snow_depth_snow_depth', 'ERA5温度_ERA5温度', 'era5_swe', 'gldas', 'scp_start', 'scp_end', 'd1', 'd2',
             'da', 'db', 'dc', 'dd', 'landuse_11', 'landuse_12', 'landuse_21', 'landuse_22', 'landuse_23', 'landuse_24',
             'landuse_31', 'landuse_32', 'landuse_33', 'landuse_41', 'landuse_43', 'landuse_46', 'landuse_51',
             'landuse_52', 'landuse_53', 'landuse_62', 'landuse_64']
y_column_transformed = ['swe_log']
# ... (省略教师模型和聚类代码，假设 `data` 表已经有了 `cluster` 列)
# 为了演示，这里用一个简化的站点划分，请替换为你自己的聚类和划分逻辑
# 假设已经完成了聚类和站点划分
all_stations = data['station_id'].unique()
np.random.shuffle(all_stations)
test_stations = all_stations[:int(len(all_stations) * 0.1)]
train_val_stations = all_stations[int(len(all_stations) * 0.1):]
train_val_data_full = data[data['station_id'].isin(train_val_stations)].copy()
test_data_full = data[data['station_id'].isin(test_stations)].copy()
# 时间划分
train_val_df_sorted = train_val_data_full.sort_values(by=['year', 'month', 'doy'])
val_sample_count = int(len(train_val_df_sorted) * 0.2)
val_data_full = train_val_df_sorted.iloc[:val_sample_count].copy()
train_data_full = train_val_df_sorted.iloc[val_sample_count:].copy()
test_df_sorted = test_data_full.sort_values(by=['year', 'month', 'doy'])
test_data = test_df_sorted[(test_df_sorted['year'] >= val_data_full['year'].min()) & (
            test_df_sorted['year'] <= val_data_full['year'].max())].copy()
# 合并用于交叉验证
train_val_stations = list(set(train_data_full['station_id'].unique()) | set(val_data_full['station_id'].unique()))
train_val_data_full = data[data['station_id'].isin(train_val_stations)].copy()


# --- 前置步骤结束 ---


# ----------------------------------------------------------------------
# --- 🔥【核心修改】使用 Optuna 进行超参数搜索 ---
# ----------------------------------------------------------------------
def objective(trial):
    """
    Optuna 的目标函数：定义搜索空间和交叉验证逻辑。
    """
    # 1. 定义超参数的搜索空间
    lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    for i in range(n_layers):
        num_units = trial.suggest_int(f'n_units_l{i}', 32, 256, step=32)
        layers.append(num_units)
    hidden_dims = [[3], layers]

    print(f"\n--- Trial {trial.number}: Testing params ---")
    print(f"  LR: {lr:.4f}, Dropout: {dropout:.2f}, Layers: {hidden_dims}")

    # 2. 设置交叉验证
    N_SPLITS = 5  # 为了演示速度，用5折。实际中可以用10折。
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_scores = []

    # 3. 遍历每一折
    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(train_val_data_full, groups=train_val_data_full['station_id'])):
        print(f"  Fold {fold + 1}/{N_SPLITS}...")

        fold_train_data = train_val_data_full.iloc[train_idx]
        fold_val_data = train_val_data_full.iloc[val_idx]

        fold_train_dataset, fold_val_dataset, _ = init_dataset_split(
            train_data=fold_train_data, val_data=fold_val_data, test_data=fold_val_data,
            x_column=x_columns, y_column=y_column_transformed,
            spatial_column=['X', 'Y', 'Z'], temp_column=['doy', 'year', 'month'],
            id_column=['id'], use_model="gtnnwr", batch_size=128, dropna=True
        )

        optimizer_params_cv = {
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [50, 100, 150, 200],
            "scheduler_gamma": 0.8,
            "lr": lr
        }

        model_cv = GTNNWR(
            fold_train_dataset, fold_val_dataset, fold_val_dataset,
            hidden_dims, drop_out=dropout,
            optimizer='Adadelta', optimizer_params=optimizer_params_cv,
            write_path=f"../demo_result/optuna_runs/trial_{trial.number}",
            model_name=f"fold_{fold + 1}"
        )
        model_cv.run(50, 200)  # 为了演示速度，减少epoch

        # 🔥【关键修复】使用正确的属性名，并获取列表的最后一个元素
        score = model_cv._validLossList[-1]
        fold_scores.append(score)

        del model_cv
        torch.cuda.empty_cache()

    # 4. 返回当前超参数组合的平均验证损失
    mean_score = np.mean(fold_scores)
    print(f"  >> Trial {trial.number} finished with mean validation loss: {mean_score:.6f}")

    return mean_score


# ----------------------------------------------------------------------
# --- 运行 Optuna 研究 ---
# ----------------------------------------------------------------------
print("\n=== 开始 Optuna 超参数搜索 ===")
# 创建一个 study 对象，目标是 'minimize' (最小化验证损失)
study = optuna.create_study(direction='minimize')
# 运行优化，例如尝试 20 次不同的超参数组合
study.optimize(objective, n_trials=20)

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
final_val_size = int(len(train_val_data_full) * 0.1)
final_train_data_sorted = train_val_data_full.sort_values(by=['year', 'month', 'doy'])
final_val_data = final_train_data_sorted.iloc[:final_val_size].copy()
final_train_data = final_train_data_sorted.iloc[final_val_size:].copy()

final_train_dataset, final_val_dataset, final_test_dataset = init_dataset_split(
    train_data=final_train_data, val_data=final_val_data, test_data=test_data,
    x_column=x_columns, y_column=y_column_transformed,
    spatial_column=['X', 'Y', 'Z'], temp_column=['doy', 'year', 'month'],
    id_column=['id'], use_model="gtnnwr", batch_size=128, dropna=True
)

optimizer_params_final = {
    "scheduler": "MultiStepLR",
    "scheduler_milestones": [1000, 2000, 3000, 4000],
    "scheduler_gamma": 0.8,
    "lr": best_params['lr']
}

final_model = GTNNWR(
    final_train_dataset, final_val_dataset, final_test_dataset,
    final_hidden_dims, drop_out=best_params['dropout'],
    optimizer='Adadelta', optimizer_params=optimizer_params_final,
    write_path="../demo_result/final_model_optuna",
    model_name="GTNNWR_Optuna_Best"
)
final_model.add_graph()
final_model.run(100, 1000)  # 使用更多的epoch进行充分训练

# ----------------------------------------------------------------------
# --- 🔥【最终评估】在独立测试集上评估最终模型 ---
# ----------------------------------------------------------------------
print("\n=== 在独立测试集上评估最终模型 ===")
pred_log = final_model._test_dataset.pred
pred_original_scale = np.expm1(pred_log)
true_original_scale = test_data['swe'].values
final_model._test_dataset.y = true_original_scale
final_model._test_dataset.pred = pred_original_scale

save_path = "../demo_result/final_model_optuna/GTNNWR_Optuna_Best_Results.png"
metrics = plot_gtnnwr_results(final_model, save_path=save_path, show_plot=True)

print("\n=== 最终评估指标 ===")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
