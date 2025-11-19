import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from gnnwr.datasets import init_dataset
from gnnwr.models import GTNNWR

# 导入我们的可视化模块
from visualizer import plot_gtnnwr_results, plot_multiple_models_results

os.environ['PYTORCH_UNIFIED'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# ----------------------------------------------------------------------
# 1. 加载原始数据
try:
    data = pd.read_excel('lu_onehot.xlsx')
except FileNotFoundError:
    print("错误：找不到文件 'lu_onehot.xlsx'。请检查文件路径。")
    sys.exit(1)

# 2. 创建一个唯一的站点标识符
data['station_id'] = data['X'].astype(str) + '_' + data['Y'].astype(str)

# 3. 确保有id列（如果没有则创建）
if 'id' not in data.columns:
    data['id'] = np.arange(len(data))

# 4. 定义所有需要的列名
# 为了清晰，我们在这里明确定义所有列
x_columns = [
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
]
y_column = ['swe']
spatial_columns = ['X', 'Y']
temporal_columns = ['year', 'month', 'doy']
id_column = ['id']  # 明确使用列表

# ----------------------------------------------------------------------
# --- 2. 按站点分割数据（空间独立性分割） ---

# 1. 获取所有唯一的站点ID并分割
unique_stations = data['station_id'].unique()
np.random.seed(48)  # 使用一个固定的随机种子，确保结果可复现
np.random.shuffle(unique_stations)

total_stations = len(unique_stations)
test_ratio = 0.15
valid_ratio = 0.1

# 2. 计算每个集合的站点数量
train_stations_count = int(total_stations * (1 - test_ratio - valid_ratio))
valid_stations_count = int(total_stations * valid_ratio)

# 3. 根据站点ID进行分割
train_stations = unique_stations[:train_stations_count]
valid_stations = unique_stations[train_stations_count : train_stations_count + valid_stations_count]
test_stations = unique_stations[train_stations_count + valid_stations_count:]

# 4. 根据分割的站点ID，创建训练、验证、测试DataFrame
# 使用 .copy() 来避免 pandas 的 SettingWithCopyWarning
train_data = data[data['station_id'].isin(train_stations)].copy()
valid_data = data[data['station_id'].isin(valid_stations)].copy()
test_data = data[data['station_id'].isin(test_stations)].copy()

# ----------------------------------------------------------------------
# --- 3. 调用库函数创建数据集和模型 ---

# 1. 导入必要的库和函数
from gnnwr.datasets import init_dataset_split, baseDataset, BasicDistance, ManhattanDistance
from gnnwr.models import GTNNWR

# 2. 定义模型参数
# 注意：这里我们使用显式导入的函数，而不是依赖默认值
optimizer_params = {
    'lr': 0.1,
    'rho': 0.95,
    'eps': 1e-8
}
dense_layers = [[3], [128, 64, 32]]  # 网络层结构

# ----------------------------------------------------------------------
# --- 4. 初始化数据集和模型 ---

# 1. 初始化数据集
# 使用 try-except 块来捕获任何在初始化过程中可能发生的错误
try:
    print("\n--- 正在初始化数据集 ---")
    train_dataset, val_dataset, test_dataset = init_dataset_split(
        train_data=train_data,
        val_data=valid_data,
        test_data=test_data,
        x_column=x_columns,
        y_column=y_column,
        spatial_column=spatial_columns,
        temp_column=temporal_columns,
        id_column=id_column,  # 使用明确定义的列表
        process_fn="minmax_scale",
        process_var=["x", "y"],
        batch_size=64,
        shuffle=True,
        use_model="gtnnwr",
        spatial_fun=BasicDistance,      # 明确传入函数
        temporal_fun=ManhattanDistance,  # 明确传入函数
        simple_distance=True,
        dropna=True
    )
    print("数据集初始化成功！")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

# 2. 初始化模型
print("\n--- 正在初始化 GTNNWR 模型 ---")
gtnnwr = GTNNWR(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    dense_layers=dense_layers,
    drop_out=0.4,
    optimizer='Adadelta',
    optimizer_params=optimizer_params
)
print("GTNNWR 模型初始化成功！")

gtnnwr.add_graph()

gtnnwr.run(400,1000)

gtnnwr.load_model('../demo_result/gtnnwr_models/GTNNWR_DSi.pkl')

# 调用result()来计算诊断结果
gtnnwr.result()

# 使用可视化模块生成散点图
print("\n=== 生成可视化结果 ===")
save_path = "../demo_result/gtnnwr_runs/GTNNWR_DSi_results.png"
metrics = plot_gtnnwr_results(gtnnwr, save_path=save_path, show_plot=True)

print("\n=== 最终评估指标 ===")
for key, value in metrics.items():
    print(f"{key}: {value}")
