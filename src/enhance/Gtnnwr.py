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

# 6. 使用 init_dataset_split 来处理已经分割好的数据
from gnnwr.datasets import init_dataset_split, baseDataset, BasicDistance, ManhattanDistance

# --- 【核心修复】确保 id_column 是一个列表 ---
# 确保 id_column 是一个包含正确列名的列表
# 例如，如果 'id' 是你的ID列，那么就应该是 ['id']
id_column_for_split = ['id']

train_dataset, val_dataset, test_dataset = init_dataset_split(
    train_data=train_data,
    val_data=valid_data,
    test_data=test_data,
    x_column=[...], # 填入你实际的x_column列表
    y_column=['swe'],
    spatial_column=['X', 'Y'],
    temp_column=['year', 'month','doy'],
    id_column=id_column_for_split, # <--- 修正这里
    # ... 其他参数保持不变
)

# 7. 添加数据质量检查，以帮助诊断 RuntimeWarning
print("--- 检查分割后的数据集大小 ---")
print(f"训练集站点数: {len(train_stations)}, 样本数: {len(train_data)}")
print(f"验证集站点数: {len(valid_stations)}, 样本数: {len(valid_data)}")
print(f"测试集站点数: {len(test_stations)}, 样本数: {len(test_data)}")

# 8. 检查每个数据子集的 'swe' 列是否为常数
# 这可以帮助诊断 RuntimeWarning 的来源
print("\n--- 检查 'swe' 列的唯一值数量 ---")
print(f"训练集 'swe' 的唯一值数量: {train_data['swe'].nunique()}")
print(f"验证集 'swe' 的唯一值数量: {valid_data['swe'].nunique()}")
print(f"测试集 'swe' 的唯一值数量: {test_data['swe'].nunique()}")
print("---------------------\n")

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
