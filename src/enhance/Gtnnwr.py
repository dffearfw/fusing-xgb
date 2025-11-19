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
data = pd.read_excel('lu_onehot.xlsx')

# 2. 创建一个唯一的站点标识符
data['station_id'] = data['X'].astype(str) + '_' + data['Y'].astype(str)

# 3. 确保有id列（如果没有则创建）
if 'id' not in data.columns:
    data['id'] = np.arange(len(data))

# 4. 获取所有唯一的站点ID并分割
unique_stations = data['station_id'].unique()
np.random.seed(48)
np.random.shuffle(unique_stations)

total_stations = len(unique_stations)
test_ratio = 0.15
valid_ratio = 0.1

train_stations_count = int(total_stations * (1 - test_ratio - valid_ratio))
valid_stations_count = int(total_stations * valid_ratio)

train_stations = unique_stations[:train_stations_count]
valid_stations = unique_stations[train_stations_count : train_stations_count + valid_stations_count]
test_stations = unique_stations[train_stations_count + valid_stations_count:]

# 5. 根据分割的站点ID，创建训练、验证、测试DataFrame
# 使用 .copy() 来避免 pandas 的 SettingWithCopyWarning
train_data = data[data['station_id'].isin(train_stations)].copy()
valid_data = data[data['station_id'].isin(valid_stations)].copy()
test_data = data[data['station_id'].isin(test_stations)].copy()

# --- 【核心修复】检查数据集是否为空 ---
print("--- 检查分割后的数据集大小 ---")
print(f"训练集站点数: {len(train_stations)}, 样本数: {len(train_data)}")
print(f"验证集站点数: {len(valid_stations)}, 样本数: {len(valid_data)}")
print(f"测试集站点数: {len(test_stations)}, 样本数: {len(test_data)}")

# 如果任何一个数据集为空，则报错并退出
if len(train_data) == 0 or len(valid_data) == 0 or len(test_data) == 0:
    print("错误：数据分割后，至少有一个数据集为空。这通常是由于该数据集中的所有行都包含NaN值，被dropna=True清空了。")
    print("请尝试：")
    print("1. 检查原始数据中NaN的分布。")
    print(data.isnull().sum())
    print("2. 调整 test_ratio 或 valid_ratio，或检查随机种子。")
    # 直接退出，避免后续的 OLS 错误
    sys.exit(1)
print("---------------------------------\n")

# 6. 使用 init_dataset_split 来处理已经分割好的数据
from gnnwr.datasets import init_dataset_split, baseDataset, BasicDistance, ManhattanDistance

train_dataset, val_dataset, test_dataset = init_dataset_split(
    train_data=train_data,
    val_data=valid_data,
    test_data=test_data,
    x_column=[...], # 你的x_column列表
    y_column=['swe'],
    spatial_column=['X', 'Y'],
    temp_column=['year', 'month','doy'],
    id_column=['id'],
    process_fn="minmax_scale",
    process_var=["x", "y"],
    batch_size=64,
    shuffle=True,
    use_model="gtnnwr",
    spatial_fun=BasicDistance,
    temporal_fun=ManhattanDistance,
    is_need_STNN=False,
    simple_distance=True,
    dropna=False  # <--- 暂时关闭，看是否与警告有关
)

print("数据集创建成功！")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
optimizer_params = {
    "scheduler":"MultiStepLR",
    "scheduler_milestones":[1000, 2000, 3000, 4000],
    "scheduler_gamma":0.8,
}
gtnnwr = GTNNWR(train_dataset, val_dataset, test_dataset, [[3], [128,64,32]], drop_out=0.4, optimizer='Adadelta', optimizer_params=optimizer_params,
                write_path = "../demo_result/gtnnwr_runs",
                model_name="GTNNWR_DSi")
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
