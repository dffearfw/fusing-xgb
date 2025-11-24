import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR
from visualizer import plot_gtnnwr_results, plot_multiple_models_results


data = pd.read_excel('lu_onehot.xlsx')
data["id"] = np.arange(len(data))
# 添加混合分割策略
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
teacher_model.run(50, 500)

# 0.5.4 提取模型系数作为聚类特征
print("提取模型学习到的空间系数作为聚类特征...")
# 假设 teacher_model.result() 返回一个包含系数的 DataFrame
teacher_results = teacher_model.result()
# 如果 result() 是保存文件，你需要先读取它，例如：
# teacher_results = pd.read_csv("../demo_result/teacher_model/Teacher_Model_results.csv")

coef_columns = [col for col in teacher_results.columns if col.startswith('coef_')]
station_coefs = teacher_results.groupby('id')[coef_columns].mean().reset_index()
# 将 id 映射回 station_id
id_to_station = teacher_train[['id', 'station_id']].drop_duplicates()
station_coefs = pd.merge(station_coefs, id_to_station, on='id', how='left')
print(f"成功为 {len(station_coefs)} 个站点提取了系数特征。")


# --- 第1步：基于【教师模型特征】对站点进行聚类 ---
# 1.1 🔥【修改】使用教师模型的系数作为聚类特征
clustering_features = coef_columns
station_features = station_coefs[['station_id'] + clustering_features].copy()

# 1.2 🔥【关键】对特征进行标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(station_features[clustering_features])

# 1.3 使用DBSCAN进行聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
station_features['cluster'] = dbscan.fit_predict(features_scaled)

# 1.4 处理噪声点并统计结果
station_features['cluster'] = station_features['cluster'].apply(
    lambda x: x if x != -1 else station_features['cluster'].max() + 1)
n_clusters = station_features['cluster'].nunique()

print(f"\n站点已聚类为 {n_clusters} 类。")
print("各簇站点数量：")
print(station_features['cluster'].value_counts().sort_index())

# 1.5 将聚类标签合并回原始数据
data = pd.merge(data, station_features[['station_id', 'cluster']], on='station_id', how='left')

# --- 第2步：在每个簇内进行分层空间采样 ---
train_stations, val_stations, test_stations = [], [], []

# 对每个簇进行独立的随机采样
for cluster_id in station_features['cluster'].unique():
    cluster_stations = station_features[station_features['cluster'] == cluster_id]['station_id'].unique()
    np.random.shuffle(cluster_stations)  # 打乱顺序

    # 按比例划分 (可以调整为 8:1:1)
    n = len(cluster_stations)
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
                write_path = "../demo_result/gtnnwr_runs", # 这里需要修改
                model_name="GTNNWR_Di")
gtnnwr.add_graph()

gtnnwr.run(100,1000)

gtnnwr.result()
save_path = "../demo_result/gtnnwr_runs/GTNNWR_DSi_results.png"

metrics = plot_gtnnwr_results(gtnnwr, save_path=save_path, show_plot=True)
