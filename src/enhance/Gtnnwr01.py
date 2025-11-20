import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from gnnwr.datasets import init_dataset_split
from gnnwr.models import GTNNWR
from visualizer import plot_gtnnwr_results, plot_multiple_models_results


data = pd.read_excel('lu_onehot.xlsx')
data["id"] = np.arange(len(data))
# 添加混合分割策略
data['station_id'] = data['X'].astype(str) + '_' + data['Y'].astype(str)

# --- 第1步：站点划分 (保证空间完全独立) ---
unique_stations = data['station_id'].unique()

np.random.shuffle(unique_stations)

test_stations = unique_stations[:int(len(unique_stations) * 0.1)]
train_val_stations = unique_stations[int(len(unique_stations) * 0.1):]

# --- 第2步：为训练/验证集准备数据 ---
train_val_df = data[data['station_id'].isin(train_val_stations)].copy()
train_val_df_sorted = train_val_df.sort_values(by=['year', 'month', 'doy'])

# --- 第3步：时间划分 (保证验证集的时间外推能力) ---
valid_sample_count = int(len(train_val_df_sorted) * 0.15)
val_data = train_val_df_sorted.iloc[:valid_sample_count].copy() # 最早的数据
train_data = train_val_df_sorted.iloc[valid_sample_count:].copy() # 最晚的数据

# --- 第4步：为测试集准备数据 (🔥 最终修复：使用与验证集相同的时间窗口) ---
test_df = data[data['station_id'].isin(test_stations)].copy()
test_df_sorted = test_df.sort_values(by=['year', 'month', 'doy'])

# 🔥 获取验证集的时间范围
val_start_time = val_data['year'].min()
val_end_time = val_data['year'].max()

# 🔥 从测试站点中筛选出与验证集时间窗口相同的数据
# 这保证了测试集不为空，且时间上不泄露训练集的信息
test_data = test_df_sorted[
    (test_df_sorted['year'] >= val_start_time) & (test_df_sorted['year'] <= val_end_time)
].copy()

# 打印一下数据集大小，确认非空
print(f"训练集样本数: {len(train_data)}")
print(f"验证集样本数: {len(val_data)}")
print(f"测试集样本数: {len(test_data)}") # 这个数现在应该 > 0 了


train_dataset, val_dataset, test_dataset = init_dataset_split(train_data=train_data,
                                                              val_data=val_data,
                                                              test_data=test_data,
                                                              x_column=[
                                                                  'aspect', 'slope', 'eastness', 'tpi', 'curvature1',
                                                                  'curvature2', 'elevation', 'std_slope',
                                                                  'std_eastness', 'std_tpi', 'std_curvature1',
                                                                  'std_curvature2', 'std_high', 'std_aspect', 'glsnow',
                                                                  'cswe', 'snow_depth_snow_depth',
                                                                  'ERA5温度_ERA5温度', 'era5_swe', 'gldas',
                                                                  'scp_start', 'scp_end',
                                                                    'd1', 'd2','da', 'db', 'dc', 'dd',
                                                                  'landuse_11', 'landuse_12', 'landuse_21',
                                                                  'landuse_22', 'landuse_23', 'landuse_24',
                                                                  'landuse_31', 'landuse_32', 'landuse_33',
                                                                  'landuse_41',  'landuse_43',
                                                                  'landuse_46', 'landuse_51', 'landuse_52',
                                                                  'landuse_53', 'landuse_62',
                                                                  'landuse_64'
                                                              ],
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
gtnnwr = GTNNWR(train_dataset, val_dataset, test_dataset, [[3], [512,256,64]],drop_out=0.4,optimizer='Adadelta',optimizer_params=optimizer_params,
                write_path = "../demo_result/gtnnwr_runs", # 这里需要修改
                model_name="GTNNWR_Di")
gtnnwr.add_graph()

gtnnwr.run(100,1000)

gtnnwr.result()
save_path = "../demo_result/gtnnwr_runs/GTNNWR_DSi_results.png"

metrics = plot_gtnnwr_results(gtnnwr, save_path=save_path, show_plot=True)
