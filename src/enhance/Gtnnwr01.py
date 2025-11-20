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

# 第一阶段：站点分割，划出测试集
unique_stations = data['station_id'].unique()
np.random.seed(48)
np.random.shuffle(unique_stations)

test_stations = unique_stations[:int(len(unique_stations) * 0.1)]
remaining_stations = unique_stations[int(len(unique_stations) * 0.1):]

test_data = data[data['station_id'].isin(test_stations)].copy()
remaining_data = data[data['station_id'].isin(remaining_stations)].copy()

# 第二阶段：时间分割，划出训练和验证集
remaining_data_sorted = remaining_data.sort_values(by=['year', 'month', 'doy'])
valid_sample_count = int(len(remaining_data_sorted) * 0.15)
valid_data = remaining_data_sorted.iloc[:valid_sample_count].copy()
train_data = remaining_data_sorted.iloc[valid_sample_count:].copy()

train_dataset, val_dataset, test_dataset = init_dataset_split(train_data=train_data,
                                                              val_data=valid_data,
                                                              test_data=test_data,
                                                              x_column=[
                                                                  'aspect', 'slope', 'eastness', 'tpi', 'curvature1',
                                                                  'curvature2', 'elevation', 'std_slope',
                                                                  'std_eastness', 'std_tpi', 'std_curvature1',
                                                                  'std_curvature2', 'std_high', 'std_aspect', 'glsnow',
                                                                  'cswe', 'snow_depth_snow_depth',
                                                                  'ERA5温度_ERA5温度', 'era5_swe', 'gldas',
                                                                  'scp_start', 'scp_end', 'd1', 'd2',
                                                                  'Z', 'da', 'db', 'dc', 'dd',
                                                                  'landuse_11', 'landuse_12', 'landuse_21',
                                                                  'landuse_22', 'landuse_23', 'landuse_24',
                                                                  'landuse_31', 'landuse_32', 'landuse_33',
                                                                  'landuse_41', 'landuse_42', 'landuse_43',
                                                                  'landuse_46', 'landuse_51', 'landuse_52',
                                                                  'landuse_53', 'landuse_62', 'landuse_63',
                                                                  'landuse_64'
                                                              ],
                                                              y_column=['swe'],
                                                              spatial_column=['X', 'Y'],
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

gtnnwr.run(10,1000)

gtnnwr.result()
save_path = "../demo_result/gtnnwr_runs/GTNNWR_DSi_results.png"

metrics = plot_gtnnwr_results(gtnnwr, save_path=save_path, show_plot=True)
