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
    if self._train_dataset.y_scale_info:
        _, denormalized_pred = self._train_dataset.rescale(None, result[pred_col_name].to_frame())
        result['denormalized_pred_result'] = denormalized_pred.iloc[:, 0]
    else:
        result['denormalized_pred_result'] = result[pred_col_name]
    if only_return:
        return result
    if filename is not None:
        result.to_csv(filename, index=False)
    else:
        warnings.warn("Warning! The input write file path is not set. Result is returned by function but not saved as file.", RuntimeWarning)
    return result

# ----------------------------------------------------------------------
# --- 主流程 ---
# ----------------------------------------------------------------------
# 🔥【关键】在创建任何模型之前，应用修复补丁
GTNNWR.reg_result = patched_reg_result

data = pd.read_excel('lu_onehot.xlsx')
data["id"] = np.arange(len(data))
data['station_id'] = data['X'].astype(str) + '_' + data['Y'].astype(str)

# 定义特征列，以便教师模型和最终模型共用
x_columns = [
    'aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation', 'std_slope',
    'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high', 'std_aspect', 'glsnow',
    'cswe', 'snow_depth_snow_depth', 'ERA5温度_ERA5温度', 'era5_swe', 'gldas',
    'scp_start', 'scp_end', 'd1', 'd2','da', 'db', 'dc', 'dd',
    'landuse_11', 'landuse_12', 'landuse_21', 'landuse_22', 'landuse_23', 'landuse_24',
    'landuse_31', 'landuse_32', 'landuse_33', 'landuse_41',  'landuse_43',
    'landuse_46', 'landuse_51', 'landuse_52', 'landuse_53', 'landuse_62', 'landuse_64'
]

# --- 【新增】第0.5步：训练教师模型，为聚类提供高质量特征 ---
print("=== 步骤0.5: 训练教师模型以生成聚类特征 ===")
# 0.5.1 划分教师模型数据 (随机抽取80%的站点)
unique_stations = data['station_id'].unique()
np.random.shuffle(unique_stations)
teacher_stations = unique_stations[:int(0.8 * len(unique_stations))]
teacher_data = data[data['station_id'].isin(teacher_stations)].copy()

# 0.5.2 教师模型数据集初始化 (简单按时间划分)
teacher_data_sorted = teacher_data.sort_values(by=['year', 'month', 'doy'])
val_size = int(len(teacher_data) * 0.2)
teacher_train = teacher_data_sorted.iloc[:-val_size]
teacher_val = teacher_data_sorted.iloc[-val_size:]

teacher_train_dataset, teacher_val_dataset, _ = init_dataset_split(
    train_data=teacher_train, val_data=teacher_val, test_data=teacher_val,
    x_column=x_columns, y_column=['swe'], spatial_column=['X', 'Y', 'Z'],
    temp_column=['doy', 'year', 'month'], id_column=['id'], use_model="gtnnwr",
    batch_size=128, process_fn="minmax_scale", process_var=["x", "y"], dropna=True
)

# 0.5.3 训练教师模型
print("开始训练教师模型...")
optimizer_params_teacher = {"scheduler": "MultiStepLR", "scheduler_milestones": [200, 400, 600, 800], "scheduler_gamma": 0.8}
teacher_model = GTNNWR(
    teacher_train_dataset, teacher_val_dataset, teacher_val_dataset,
    [[3], [128, 64]], drop_out=0.3, optimizer='Adadelta',
    optimizer_params=optimizer_params_teacher,
    write_path="../demo_result/teacher_model",
    model_name="Teacher_Model"
)
teacher_model.run(5, 500)

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

# 1.3 使用DBSCAN进行聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
station_features_for_clustering['cluster'] = dbscan.fit_predict(features_scaled)

# 1.4 处理噪声点并统计结果
station_features_for_clustering['cluster'] = station_features_for_clustering['cluster'].apply(
    lambda x: x if x != -1 else station_features_for_clustering['cluster'].max() + 1)
n_clusters = station_features_for_clustering['cluster'].nunique()

print(f"\n站点已聚类为 {n_clusters} 类。")
print("各簇站点数量：")
print(station_features_for_clustering['cluster'].value_counts().sort_index())

# 1.5 🔥【关键修复2】安全地将聚类标签合并回原始数据，避免笛卡尔积
# station_features_for_clustering 的 'station_id' 现在是唯一的，可以安全 merge
data = pd.merge(data, station_features_for_clustering[['station_id', 'cluster']], on='station_id', how='left')

# 处理未被聚类的站点（例如，在教师模型中未出现的20%的站点）
# 这里我们将它们归为一个新的簇
if data['cluster'].isnull().any():
    max_cluster_id = data['cluster'].max()
    data['cluster'].fillna(max_cluster_id + 1, inplace=True)
    print(f"将 {data['cluster'].isnull().sum()} 个未聚类站点归入新簇 {int(max_cluster_id + 1)}。")


# --- 第2步：在每个簇内进行分层空间采样 ---
train_stations, val_stations, test_stations = [], [], []

# 🔥【修改】从正确的聚类结果中获取唯一的站点和簇
clustered_stations_df = station_features_for_clustering

for cluster_id in clustered_stations_df['cluster'].unique():
    cluster_stations = clustered_stations_df[clustered_stations_df['cluster'] == cluster_id]['station_id'].unique()
    np.random.shuffle(cluster_stations)

    n = len(cluster_stations)
    # 确保有足够的数据点进行划分
    if n < 10: # 如果簇太小，可以全部放入训练集
        train_stations.extend(cluster_stations)
        print(f"簇 {cluster_id} 太小 ({n}个站点)，已全部放入训练集。")
        continue

    test_set = cluster_stations[:int(n * 0.1)]
    val_set = cluster_stations[int(n * 0.1):int(n * 0.2)]
    train_set = cluster_stations[int(n * 0.2):]

    train_stations.extend(train_set)
    val_stations.extend(val_set)
    test_stations.extend(test_set)

print(f"\n分层采样后：")
print(f"训练集站点数: {len(train_stations)}")
print(f"验证集站点数: {len(val_stations)}")
print(f"测试集站点数: {len(test_stations)}")

# --- 第3步：根据划分好的站点创建数据集 ---
# (这里的代码保持不变，因为它现在基于正确的站点列表)
train_data_full = data[data['station_id'].isin(train_stations)].copy()
val_data_full = data[data['station_id'].isin(val_stations)].copy()
test_data_full = data[data['station_id'].isin(test_stations)].copy()

# --- 第4步：在空间划分的基础上，进行时间划分 ---
# 训练/验证集：时间划分
train_val_df_sorted = train_data_full.sort_values(by=['year', 'month', 'doy'])
val_sample_count = int(len(val_data_full))  # 使用验证集的样本数作为划分标准，保持比例
val_data = train_val_df_sorted.iloc[:val_sample_count].copy()
train_data = train_val_df_sorted.iloc[val_sample_count:].copy()

# 测试集：使用与验证集相同的时间窗口，防止泄露
val_start_time = val_data['year'].min()
val_end_time = val_data['year'].max()
test_df_sorted = test_data_full.sort_values(by=['year', 'month', 'doy'])
test_data = test_df_sorted[
    (test_df_sorted['year'] >= val_start_time) & (test_df_sorted['year'] <= val_end_time)
    ].copy()

print(f"\n最终数据集样本数：")
print(f"训练集样本数: {len(train_data)}")
print(f"验证集样本数: {len(val_data)}")
print(f"测试集样本数: {len(test_data)}")

train_dataset, val_dataset, test_dataset = init_dataset_split(train_data=train_data,
                                                              val_data=val_data,
                                                              test_data=test_data,
                                                              x_column=x_columns,
                                                              y_column=['swe'],
                                                              spatial_column=['X', 'Y','Z'],
                                                              temp_column=['doy','year','month'],
                                                              id_column=['id'],
                                                              use_model="gtnnwr",
                                                              batch_size=128)


optimizer_params = {
    "scheduler":"MultiStepLR",
    "scheduler_milestones":[1000, 2000, 3000, 4000],
    "scheduler_gamma":0.8,
}
gtnnwr = GTNNWR(train_dataset, val_dataset, test_dataset, [[3], [256,128,64]],drop_out=0.4,optimizer='Adadelta',optimizer_params=optimizer_params,
                write_path = "../demo_result/gtnnwr_runs",
                model_name="GTNNWR_Final")
gtnnwr.add_graph()

gtnnwr.run(100,1000)

gtnnwr.result()
save_path = "../demo_result/gtnnwr_runs/GTNNWR_Final_results.png"

metrics = plot_gtnnwr_results(gtnnwr, save_path=save_path, show_plot=True)
