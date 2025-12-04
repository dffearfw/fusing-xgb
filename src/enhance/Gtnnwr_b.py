import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from gnnwr.datasets import init_dataset, baseDataset, _init_gtnnwr_distance
from gnnwr.models import GTNNWR

# -----------------------------
# 1. 加载数据
# -----------------------------
data = pd.read_excel('../lu_onehot.xlsx')

# 确保有 station_id 列（请根据你的实际列名修改！）
if 'station_id' not in data.columns:
    raise ValueError("请确认你的数据中有 'station_id' 列（或修改为实际站点ID列名）")

# -----------------------------
# 2. 按 station_id 划分：10% 独立站点作测试
# -----------------------------
np.random.seed(48)  # 保证可复现

unique_stations = data['station_id'].unique()
n_test_stations = max(1, int(0.1 * len(unique_stations)))  # 至少1个站点

test_station_ids = np.random.choice(unique_stations, size=n_test_stations, replace=False)
train_val_stations = [sid for sid in unique_stations if sid not in test_station_ids]

# 划分数据
test_data_independent = data[data['station_id'].isin(test_station_ids)].copy()
train_val_data = data[data['station_id'].isin(train_val_stations)].copy()

print(f"总站点数: {len(unique_stations)}")
print(f"测试站点数: {len(test_station_ids)} ({len(test_data_independent)} 条记录)")
print(f"训练+验证站点数: {len(train_val_stations)} ({len(train_val_data)} 条记录)")

# -----------------------------
# 3. 为 train_val_data 添加 id 列（init_dataset 需要）
# -----------------------------
train_val_data = train_val_data.reset_index(drop=True)
train_val_data["id"] = np.arange(len(train_val_data))

# -----------------------------
# 4. 用 init_dataset 初始化 train/val（test_ratio=0）
# -----------------------------
x_cols = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation', 'std_slope', 'std_eastness',
          'std_tpi', 'std_curvature1',
          'std_curvature2', 'std_high', 'std_aspect', 'glsnow', 'cswe', 'snow_depth_snow_depth', 'ERA5温度_ERA5温度',
          'era5_swe', 'gldas',
          'scp_start', 'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da', 'db', 'dc', 'dd']
y_cols = ['swe']
spatial_cols = ['longitude', 'latitude']
temp_cols = ['year', 'month', 'doy']

train_dataset, val_dataset, _ = init_dataset(
    data=train_val_data,
    test_ratio=0.0,  # 不从 train_val_data 中分测试集
    valid_ratio=0.1,
    x_column=x_cols,
    y_column=y_cols,
    spatial_column=spatial_cols,
    temp_column=temp_cols,
    id_column=['id'],
    use_model="gtnnwr",
    sample_seed=48,
    batch_size=1024,
    is_need_STNN=True
)

# -----------------------------
# 5. 手动构建 test_dataset（复用 train_dataset 的预处理）
# -----------------------------
test_data_independent = test_data_independent.reset_index(drop=True)
test_data_independent["id"] = np.arange(len(test_data_independent)) + 1000000  # 避免ID冲突

test_dataset = baseDataset(
    data=test_data_independent,
    x_column=train_dataset.x,
    y_column=train_dataset.y,
    id_column=train_dataset.id,
    is_need_STNN=train_dataset.is_need_STNN
)

# 应用相同的 scaling
test_dataset.scale(
    scale_fn=train_dataset.scale_fn,
    scale_params=[train_dataset.x_scale_info, train_dataset.y_scale_info]
)

# -----------------------------
# 6. 计算 test 的距离矩阵（使用 train_dataset.reference 作为 reference）
# -----------------------------
reference_data = train_dataset.reference  # 这是在 init_dataset 中设置的

# 调用内部距离函数（需定义或导入 _init_gtnnwr_distance）
test_distances = _init_gtnnwr_distance(
    ref_points=[
        reference_data[spatial_cols].values,
        reference_data[temp_cols].values
    ],
    train_points=None,
    val_points=None,
    test_points=[
        test_data_independent[spatial_cols].values,
        test_data_independent[temp_cols].values
    ],
    spatial_fun=train_dataset.spatial_fun if hasattr(train_dataset, 'spatial_fun') else None,
    temporal_fun=train_dataset.temporal_fun if hasattr(train_dataset, 'temporal_fun') else None
)[2]  # 第三个返回值是 test_distances

test_dataset.distances = test_distances

# -----------------------------
# 7. 对 distances 应用相同的 distance scaler
# -----------------------------
if train_dataset.distances_scale_param is not None:
    orig_shape = test_dataset.distances.shape
    flat_dist = test_dataset.distances.reshape(-1, orig_shape[-1])

    if train_dataset.scale_fn == "minmax_scale":
        d_min = train_dataset.distances_scale_param["min"]
        d_max = train_dataset.distances_scale_param["max"]
        scaled_flat = (flat_dist - d_min) / (d_max - d_min + 1e-8)
    elif train_dataset.scale_fn == "standard_scale":
        d_mean = train_dataset.distances_scale_param["mean"]
        d_var = train_dataset.distances_scale_param["var"]
        scaled_flat = (flat_dist - d_mean) / (np.sqrt(d_var) + 1e-8)
    else:
        raise ValueError("Unknown distance scaler")

    test_dataset.distances = scaled_flat.reshape(orig_shape)

# -----------------------------
# 8. 设置 DataLoader（可选，GTNNWR 可能自己处理）
# -----------------------------
test_dataset.batch_size = 1024
test_dataset.shuffle = False

# -----------------------------
# 9. 训练模型
# -----------------------------
optimizer_params = {
    "scheduler": "MultiStepLR",
    "scheduler_milestones": [1000, 2000, 3000, 4000],
    "scheduler_gamma": 0.8,
}

gtnnwr = GTNNWR(
    train_dataset, val_dataset, test_dataset,
    dense_layers=[[3], [512, 256, 64]],
    drop_out=0.4,
    optimizer='Adadelta',
    optimizer_params=optimizer_params,
    write_path="../demo_result/gtnnwr_runs",
    model_name="GTNNWR_DSi"
)

gtnnwr.add_graph()
gtnnwr.run(15000, 1000)
gtnnwr.result()