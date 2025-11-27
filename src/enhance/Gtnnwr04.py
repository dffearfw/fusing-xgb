import os
import sys
import warnings
from sklearn.cluster import AgglomerativeClustering
import numpy as np  # 🔥【对数变换】确保 numpy 已导入
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from torch import nn
from scipy.spatial import distance

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR
from visualizer import plot_gtnnwr_results, plot_multiple_models_results


# ----------------------------------------------------------------------
# --- 🔥【封装的修复补丁】修复 gnnwr 库的内部 bug ---
# ----------------------------------------------------------------------
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

    # 🔥【对数变换】修改：在保存结果时，如果目标是swe_log，则将预测结果反向变换回原始尺度
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

data = pd.read_excel('lu_onehot.xlsx')
data["id"] = np.arange(len(data))
data['station_id'] = data['X'].astype(str) + '_' + data['Y'].astype(str)

# 🔥【对数变换】修改1：创建对数变换后的目标变量列
data['swe_log'] = np.log1p(data['swe'])

# 定义特征列，以便教师模型和最终模型共用
x_columns = [
    'aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation', 'std_slope',
    'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high', 'std_aspect', 'glsnow',
    'cswe', 'snow_depth_snow_depth', 'ERA5温度_ERA5温度', 'era5_swe', 'gldas',
    'scp_start', 'scp_end', 'd1', 'd2', 'da', 'db', 'dc', 'dd',
    'landuse_11', 'landuse_12', 'landuse_21', 'landuse_22', 'landuse_23', 'landuse_24',
    'landuse_31', 'landuse_32', 'landuse_33', 'landuse_41', 'landuse_43',
    'landuse_46', 'landuse_51', 'landuse_52', 'landuse_53', 'landuse_62', 'landuse_64'
]

# 🔥【对数变换】修改2：定义变换后的y列名
y_column_transformed = ['swe_log']

# --- 【新增】第0.5步：训练教师模型，为聚类提供高质量特征 ---
print("=== 步骤0.5: 训练教师模型以生成聚类特征 ===")
# 0.5.1 🔥【修改】使用全体数据作为教师模型的数据源
teacher_data = data.copy()

# 0.5.2 🔥【修改】教师模型数据集初始化 (仅按时间划分，不进行空间划分)
# 对全体数据按时间排序
teacher_data_sorted = teacher_data.sort_values(by=['year', 'month', 'doy'])

# 按时间顺序划分训练集和验证集（例如，用最后20%的时间段作为验证集）
val_size = int(len(teacher_data_sorted) * 0.2)
teacher_train = teacher_data_sorted.iloc[:-val_size].copy()
teacher_val = teacher_data_sorted.iloc[-val_size:].copy()

teacher_train_dataset, teacher_val_dataset, _ = init_dataset_split(
    train_data=teacher_train, val_data=teacher_val, test_data=teacher_val,
    x_column=x_columns, y_column=y_column_transformed,  # 🔥【对数变换】修改3：使用变换后的y列
    spatial_column=['X', 'Y', 'Z'],
    temp_column=['doy', 'year', 'month'], id_column=['id'], use_model="gtnnwr",
    batch_size=128, process_fn="minmax_scale", process_var=["x", "y"], dropna=True
)

# 0.5.3 训练教师模型
print("开始训练教师模型...")
optimizer_params_teacher = {"scheduler": "MultiStepLR", "scheduler_milestones": [200, 400, 600, 800],
                            "scheduler_gamma": 0.8}
teacher_model = GTNNWR(
    teacher_train_dataset, teacher_val_dataset, teacher_val_dataset,
    [[3], [128, 64]], drop_out=0.3, optimizer='Adadelta',
    optimizer_params=optimizer_params_teacher,
    write_path="../demo_result/teacher_model",
    model_name="Teacher_Model"
)
teacher_model.run(100, 500)

# 0.5.4 提取模型系数作为聚类特征
print("提取模型学习到的空间系数作为聚类特征...")
# 使用修复后的 result 方法，并直接获取返回值
teacher_results = teacher_model.reg_result(only_return=True)

coef_columns = [col for col in teacher_results.columns if col.startswith('coef_')]

# --- 🔥【关键修复1】获取完整的 id -> station_id 映射并聚合到站点级别 ---
# 1. 从教师模型使用的完整数据中获取映射
id_to_station_full = teacher_data[['id', 'station_id']].drop_duplicates()

# 2. 将 station_id 合并到模型结果中
results_with_station_id = pd.merge(teacher_results, id_to_station_full, on='id', how='left')

# 3. 按 station_id 聚合系数，得到每个站点的代表性系数
station_level_coefs = results_with_station_id.groupby('station_id')[coef_columns].mean().reset_index()

# 4. 清理可能因合并产生的 NaN 行（如果有的话）
station_level_coefs.dropna(inplace=True)

print(f"成功为 {len(station_level_coefs)} 个站点聚合了系数特征。")

# --- 第1步：基于【教师模型特征】对站点进行聚类 ---
# 1.1 🔥【修改】使用聚合后的站点系数作为聚类特征
clustering_features = coef_columns
station_features_for_clustering = station_level_coefs[['station_id'] + clustering_features].copy()

# 1.2 🔥【关键】对特征进行标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(station_features_for_clustering[clustering_features])

# --- 【绕过方案】使用 AgglomerativeClustering 替代 DBSCAN ---
from sklearn.cluster import AgglomerativeClustering

# 🔥【关键】您需要预先指定簇的数量
N_CLUSTERS = 5

print(f"\n使用 AgglomerativeClustering 进行聚类，预设簇数量为: {N_CLUSTERS}")

# 1.3 使用层次聚类
agglo = AgglomerativeClustering(n_clusters=N_CLUSTERS)
station_features_for_clustering['cluster'] = agglo.fit_predict(features_scaled)

# 1.4 统计结果
n_clusters = station_features_for_clustering['cluster'].nunique()
print(f"\n站点已聚类为 {n_clusters} 类。")
print("各簇站点数量：")
print(station_features_for_clustering['cluster'].value_counts().sort_index())

# 1.5 🔥【关键修复2】安全地将聚类标签合并回原始数据
data = pd.merge(data, station_features_for_clustering[['station_id', 'cluster']], on='station_id', how='left')
if data['cluster'].isnull().any():
    max_cluster_id = data['cluster'].max()
    data['cluster'].fillna(max_cluster_id + 1, inplace=True)
    print(f"将 {data['cluster'].isnull().sum()} 个未聚类站点归入新簇 {int(max_cluster_id + 1)}。")

# --- 🔥【修改】将采样和训练放入循环，进行10次实验 ---
all_predictions = []
all_true_values_list = []  # 每次测试集的真实值可能不同，所以也保存起来
successful_runs = 0
failed_runs = 0
total_attempts = 0
max_attempts = 50  # 设置最大尝试次数，防止无限循环

optimizer_params = {
    "scheduler": "MultiStepLR",
    "scheduler_milestones": [1000, 2000, 3000, 4000],
    "scheduler_gamma": 0.8,
}

print("\n=== 开始10次独立的采样和训练实验（带错误恢复） ===")
while successful_runs < 10 and total_attempts < max_attempts:
    total_attempts += 1
    print(f"\n--- 尝试第 {total_attempts} 次实验 (已成功 {successful_runs}/10) ---")

    try:
        # --- 第2步：在每个簇内进行分层空间采样 (每次循环都重新采样) ---
        train_stations, val_stations, test_stations = [], [], []
        clustered_stations_df = station_features_for_clustering.copy()

        for cluster_id in clustered_stations_df['cluster'].unique():
            cluster_stations = clustered_stations_df[clustered_stations_df['cluster'] == cluster_id][
                'station_id'].unique()
            np.random.shuffle(cluster_stations)  # 🔥【关键】每次循环都重新打乱

            n = len(cluster_stations)
            if n < 10:
                train_stations.extend(cluster_stations)
                continue

            test_set = cluster_stations[:int(n * 0.1)]
            val_set = cluster_stations[int(n * 0.1):int(n * 0.2)]
            train_set = cluster_stations[int(n * 0.2):]

            train_stations.extend(train_set)
            val_stations.extend(val_set)
            test_stations.extend(test_set)

        # --- 第3步：根据划分好的站点创建数据集 ---
        train_data_full = data[data['station_id'].isin(train_stations)].copy()
        val_data_full = data[data['station_id'].isin(val_stations)].copy()
        test_data_full = data[data['station_id'].isin(test_stations)].copy()

        # --- 第4步：在空间划分的基础上，进行时间划分 ---
        train_val_df_sorted = train_data_full.sort_values(by=['year', 'month', 'doy'])
        val_sample_count = int(len(val_data_full))
        val_data = train_val_df_sorted.iloc[:val_sample_count].copy()
        train_data = train_val_df_sorted.iloc[val_sample_count:].copy()

        val_start_time = val_data['year'].min()
        val_end_time = val_data['year'].max()
        test_df_sorted = test_data_full.sort_values(by=['year', 'month', 'doy'])
        test_data = test_df_sorted[
            (test_df_sorted['year'] >= val_start_time) & (test_df_sorted['year'] <= val_end_time)
            ].copy()

        # 初始化数据集
        train_dataset, val_dataset, test_dataset = init_dataset_split(train_data=train_data,
                                                                      val_data=val_data,
                                                                      test_data=test_data,
                                                                      x_column=x_columns,
                                                                      y_column=y_column_transformed,
                                                                      spatial_column=['X', 'Y', 'Z'],
                                                                      temp_column=['doy', 'year', 'month'],
                                                                      id_column=['id'],
                                                                      use_model="gtnnwr",
                                                                      batch_size=128)

        # 训练学生模型
        gtnnwr = GTNNWR(train_dataset, val_dataset, test_dataset, [[3], [256, 128, 64]],
                        drop_out=0.4, optimizer='Adadelta', optimizer_params=optimizer_params,
                        write_path=f"../demo_result/gtnnwr_runs/run_{successful_runs + 1}",
                        model_name=f"GTNNWR_Run_{successful_runs + 1}")

        gtnnwr.add_graph()
        gtnnwr.run(100, 1000)

        # 获取预测结果
        results_df = gtnnwr.reg_result(only_return=True)
        test_results = results_df[results_df['dataset_belong'] == 'test']

        # 还原到原始尺度
        pred_log = test_results['Pred_swe_log'].values
        pred_original_scale = np.expm1(pred_log)

        # 保存本次实验的预测和真实值
        all_predictions.append(pred_original_scale)
        all_true_values_list.append(test_data['swe'].values)

        successful_runs += 1
        print(f"✅ 第 {successful_runs} 次实验成功完成，测试集样本数: {len(pred_original_scale)}")

    except Exception as e:
        failed_runs += 1
        print(f"❌ 第 {total_attempts} 次实验失败: {str(e)}")
        print("   将跳过此次失败，继续下一次尝试...")
        # 清理GPU内存（如果使用）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

# 检查是否成功完成了10次实验
if successful_runs < 10:
    print(f"\n⚠️ 警告：仅成功完成了 {successful_runs}/10 次实验（尝试了 {total_attempts} 次）")
else:
    print(f"\n✅ 成功完成了 {successful_runs} 次实验（共尝试了 {total_attempts} 次，失败 {failed_runs} 次）")

# 只有当至少有一次成功时才继续
if successful_runs > 0:
    # --- 🔥【新增】绘制成功实验的散点图 ---
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, successful_runs))

    # 绘制每次运行的结果
    for i, (preds, trues) in enumerate(zip(all_predictions, all_true_values_list)):
        plt.scatter(trues, preds, alpha=0.6, color=colors[i], s=20, label=f'Run {i + 1}')

    # 绘制理想线
    all_trues_flat = np.concatenate(all_true_values_list)
    all_preds_flat = np.concatenate(all_predictions)
    max_val = max(all_trues_flat.max(), all_preds_flat.max())
    min_val = min(all_trues_flat.min(), all_preds_flat.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')

    plt.xlabel('True SWE', fontsize=12)
    plt.ylabel('Predicted SWE', fontsize=12)
    plt.title(f'{successful_runs} GTNNWR Experiments (Different Splits): True vs Predicted SWE', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图像
    save_path = f"../demo_result/gtnnwr_runs/{successful_runs}_experiments_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n散点图已保存至: {save_path}")

    # 显示统计信息
    print("\n=== 实验统计信息 ===")
    r2_scores = []
    mae_scores = []
    rmse_scores = []

    for i, (preds, trues) in enumerate(zip(all_predictions, all_true_values_list)):
        r2 = np.corrcoef(trues, preds)[0, 1] ** 2
        mae = np.mean(np.abs(trues - preds))
        rmse = np.sqrt(np.mean((trues - preds) ** 2))

        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print(f"Run {i + 1}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    print(f"\n平均性能: R²={np.mean(r2_scores):.4f}±{np.std(r2_scores):.4f}, "
          f"MAE={np.mean(mae_scores):.4f}±{np.std(mae_scores):.4f}, "
          f"RMSE={np.mean(rmse_scores):.4f}±{np.std(rmse_scores):.4f}")

    plt.show()
else:
    print("\n❌ 没有成功的实验，无法生成结果。")
