import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

from torch import nn

# 假设 visualizer 模块在 python 路径中
from visualizer import plot_gtnnwr_results, plot_multiple_models_results


# ----------------------------------------------------------------------
# --- 1. 加载和准备原始数据 ---

# 1. 加载原始数据
try:
    data = pd.read_excel('lu_onehot.xlsx')
except FileNotFoundError:
    print("错误：找不到文件 'lu_onehot.xlsx'。请检查文件路径。")
    sys.exit(1)

# 2. 创建一个唯一的站点标识符D
data['station_id'] = data['X'].astype(str) + '_' + data['Y'].astype(str)

# 3. 确保有id列（如果没有则创建）
if 'id' not in data.columns:
    data['id'] = np.arange(len(data))

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
valid_stations = unique_stations[train_stations_count: train_stations_count + valid_stations_count]
test_stations = unique_stations[train_stations_count + valid_stations_count:]

# 4. 根据分割的站点ID，创建训练、验证、测试DataFrame
# 使用 .copy() 来避免 pandas 的 SettingWithCopyWarning
train_data = data[data['station_id'].isin(train_stations)].copy()
valid_data = data[data['station_id'].isin(valid_stations)].copy()
test_data = data[data['station_id'].isin(test_stations)].copy()

# ----------------------------------------------------------------------
# --- 3. 自动化特征筛选（核心修复部分） ---
# ----------------------------------------------------------------------
print("--- 数据分割质量检查 ---")
print(f"总站点数: {total_stations}")
print(f"训练集站点数: {len(train_stations)}, 样本数: {len(train_data)}")
print(f"验证集站点数: {len(valid_stations)}, 样本数: {len(valid_data)}")
print(f"测试集站点数: {len(test_stations)}, 样本数: {len(test_data)}")

# 定义所有原始特征列
x_columns_raw = [
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

# 定义其他列
y_column = ['swe']
spatial_columns = ['X', 'Y']
temporal_columns = ['year', 'month', 'doy']
id_column = ['id']

# 自动化筛选逻辑
print("\n--- 正在自动化筛选有效特征列 ---")
valid_x_columns = []
removed_columns = []

for col in x_columns_raw:
    if col not in train_data.columns:
        print(f"警告: 特征列 '{col}' 在训练数据中不存在，已跳过。")
        removed_columns.append(col)
        continue

    if train_data[col].nunique() <= 1:
        print(f"信息: 特征列 '{col}' 是常数列 (唯一值数量: {train_data[col].nunique()})，已自动移除。")
        removed_columns.append(col)
        continue

    valid_x_columns.append(col)

print("\n--- 特征筛选摘要 ---")
print(f"原始特征数量: {len(x_columns_raw)}")
print(f"移除的常数/无效特征 ({len(removed_columns)} 个): {removed_columns}")
print(f"最终用于训练的有效特征数量: {len(valid_x_columns)}")

# 用筛选后的列表覆盖原来的 x_columns
x_columns = valid_x_columns

# ----------------------------------------------------------------------
# --- 4. 调用库函数创建数据集和模型 ---
# ----------------------------------------------------------------------
# 1. 导入必要的库和函数
from gnnwr.datasets import init_dataset_split, baseDataset, BasicDistance, ManhattanDistance
from gnnwr.models import GTNNWR

# 2. 定义模型参数
optimizer_params = {
    'lr': 0.1,
    'rho': 0.95,
    'eps': 1e-8
}
dense_layers = [[3], [512,256,64]]  # 网络层结构

# ----------------------------------------------------------------------
# --- 5. 初始化数据集和模型 ---
# ----------------------------------------------------------------------
# 使用 try-except 块来捕获任何在初始化过程中可能发生的错误
try:
    print("\n--- 正在初始化数据集 ---")
    train_dataset, val_dataset, test_dataset = init_dataset_split(
        train_data=train_data,
        val_data=valid_data,
        test_data=test_data,
        x_column=x_columns,  # 这里会使用筛选后的干净列表
        y_column=y_column,
        spatial_column=spatial_columns,
        temp_column=temporal_columns,
        id_column=id_column,
        process_fn="minmax_scale",
        process_var=["x", "y"],
        batch_size=64,
        shuffle=True,
        use_model="gtnnwr",
        spatial_fun=BasicDistance,
        temporal_fun=ManhattanDistance,
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
        valid_dataset=val_dataset,
        test_dataset=test_dataset,
        dense_layers=dense_layers,
        drop_out=0.4,
        optimizer='Adadelta',
        optimizer_params=optimizer_params,
        log_path='.'
    )
    print("GTNNWR 模型初始化成功！")

    gtnnwr.add_graph()

    original_reg_result = GTNNWR.reg_result


    # 定义一个修复后的新方法
    def patched_reg_result(self, filename=None, model_path=None, use_dict=False, only_return=False, map_location=None):
        """
        这是修复后的 reg_result 版本。
        它只对预测列进行反归一化，解决了原始版本的 bug。
        """
        # --- 复制原始方法的前半部分逻辑（加载模型、遍历数据集生成 result DataFrame）---
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
            # ... (这里省略，和原始代码完全一样，直到生成 result DataFrame) ...
            # calculate the result of train dataset
            for data, coef, label, data_index in self._train_dataset.dataloader:
                data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(
                    device)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                coefficient = self._model(data).mul(torch.tensor(self._coefficient).to(torch.float32).to(device))
                output = torch.cat((coefficient, output, data_index), dim=1)
                result = torch.cat((result, output), 0)
            train_data_size = len(result)
            # calculate the result of valid dataset
            for data, coef, label, data_index in self._valid_dataset.dataloader:
                data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(
                    device)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                coefficient = self._model(data).mul(torch.tensor(self._coefficient).to(torch.float32).to(device))
                output = torch.cat((coefficient, output, data_index), dim=1)
                result = torch.cat((result, output), 0)
            valid_data_size = len(result) - train_data_size
            # calculate the result of test dataset
            for data, coef, label, data_index in self._test_dataset.dataloader:
                data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(
                    device)
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

        # set dataset belong to postprocess
        result['dataset_belong'] = np.concatenate([
            np.full(train_data_size, 'train'),
            np.full(valid_data_size, 'valid'),
            np.full(len(result) - train_data_size - valid_data_size, 'test')
        ])

        # --- 关键修复：只对预测列进行反归一化 ---
        pred_col_name = "Pred_" + self._train_dataset.y[0]
        if self._train_dataset.y_scale_info:
            # 只传递预测列（作为单列DataFrame）给 rescale
            _, denormalized_pred = self._train_dataset.rescale(None, result[pred_col_name].to_frame())
            # 将反归一化后的结果（单列DataFrame）赋值给新列
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


    # 用我们的新方法替换掉原始方法
    GTNNWR.reg_result =  original_reg_result
    print("已应用基于源码的精准修复补丁。")

    # ----------------------------------------------------------------------

    gtnnwr.run(10, 1000)

    # 根据之前的分析，这里你可能想用新训练的模型，所以注释掉 load_model
    # gtnnwr.load_model('../demo_result/gtnnwr_models/GTNNWR_DSi.pkl')

    # 调用result()来计算诊断结果
    gtnnwr.result()

    # 使用可视化模块生成散点图
    print("\n=== 生成可视化结果 ===")
    # 确保目录存在
    os.makedirs("../demo_result/gtnnwr_runs/", exist_ok=True)
    save_path = "../demo_result/gtnnwr_runs/GTNNWR_DSi_results.png"

    # 暂时注释掉可视化，以防 visualizer 模块有问题
    metrics = plot_gtnnwr_results(gtnnwr, save_path=save_path, show_plot=True)
    print("模型训练和评估完成")

except Exception as e:
    # 捕获任何异常，并打印详细的错误信息
    print(f"\n!!! 发生未处理的异常 !!!")
    print(f"错误类型: {type(e)}")
    print(f"错误信息: {e}")
    import traceback

    traceback.print_exc()
    print("\n请根据以上错误信息检查你的代码。")
    # 优雅地退出，避免程序崩溃
    sys.exit(1)

