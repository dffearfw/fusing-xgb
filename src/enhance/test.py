import os
import sys
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from gnnwr import models, datasets, utils


def debug_data_issues(data, x_column, y_column, spatial_column):
    """详细的数据问题调试"""
    print("=== 数据调试信息 ===")
    print(f"原始数据形状: {data.shape}")

    # 1. 检查列是否存在
    all_required_columns = x_column + y_column + spatial_column
    missing_columns = [col for col in all_required_columns if col not in data.columns]
    if missing_columns:
        print(f"❌ 缺失列: {missing_columns}")
        print(f"可用列: {list(data.columns)}")
        return False

    print("✅ 所有必需列都存在")

    # 2. 检查缺失值
    print("\n=== 缺失值分析 ===")
    missing_info = data[all_required_columns].isnull().sum()
    total_missing = missing_info.sum()
    print(f"总缺失值: {total_missing}")

    if total_missing > 0:
        print("缺失值详情:")
        for col, missing_count in missing_info[missing_info > 0].items():
            print(f"  {col}: {missing_count} 个缺失值 ({missing_count / len(data):.1%})")

    # 3. 检查无穷值
    print("\n=== 无穷值检查 ===")
    numeric_cols = data[all_required_columns].select_dtypes(include=[np.number]).columns
    inf_count = 0
    for col in numeric_cols:
        if np.isinf(data[col]).any():
            inf_count += 1
            print(f"❌ 列 '{col}' 包含无穷值")

    if inf_count == 0:
        print("✅ 没有无穷值")

    # 4. 检查数据有效性
    print("\n=== 数据有效性检查 ===")
    valid_data = data[all_required_columns].copy()

    # 移除包含缺失值的行
    original_size = len(valid_data)
    valid_data = valid_data.dropna()
    print(f"移除缺失值后: {len(valid_data)} 行 (保留 {len(valid_data) / original_size:.1%})")

    # 移除包含无穷值的行
    for col in numeric_cols:
        valid_data = valid_data[np.isfinite(valid_data[col])]
    print(f"移除无穷值后: {len(valid_data)} 行 (保留 {len(valid_data) / original_size:.1%})")

    if len(valid_data) == 0:
        print("❌ 警告: 数据清洗后为空!")
        return False

    print(f"✅ 最终有效数据: {len(valid_data)} 行")
    return True


def robust_data_cleaning(data, x_column, y_column, spatial_column):
    """鲁棒的数据清洗"""
    print("开始数据清洗...")

    # 1. 创建数据副本
    clean_data = data.copy()

    # 2. 处理缺失值 - 使用更智能的方法
    all_columns = x_column + y_column + spatial_column

    # 检查每列的缺失率
    missing_rates = clean_data[all_columns].isnull().sum() / len(clean_data)

    # 对低缺失率的列进行填充
    for col in all_columns:
        if missing_rates[col] > 0 and missing_rates[col] < 0.3:  # 缺失率低于30%
            if col in y_column:  # 目标变量严格处理
                clean_data = clean_data.dropna(subset=[col])
            else:  # 特征变量可以填充
                if clean_data[col].dtype in ['float64', 'int64']:
                    clean_data[col].fillna(clean_data[col].median(), inplace=True)
                else:
                    clean_data[col].fillna(clean_data[col].mode()[0] if len(clean_data[col].mode()) > 0 else 0,
                                           inplace=True)
        elif missing_rates[col] >= 0.3:  # 高缺失率列
            print(f"警告: 列 '{col}' 缺失率过高 ({missing_rates[col]:.1%})")

    # 3. 处理无穷值
    numeric_cols = clean_data[all_columns].select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(clean_data[col]).any():
            # 将无穷值替换为列的最大值/最小值
            finite_values = clean_data[col][np.isfinite(clean_data[col])]
            if len(finite_values) > 0:
                col_max = finite_values.max()
                col_min = finite_values.min()
                clean_data[col] = clean_data[col].replace([np.inf], col_max)
                clean_data[col] = clean_data[col].replace([-np.inf], col_min)

    # 4. 最终清理：移除任何剩余的无效值
    clean_data = clean_data.dropna()
    for col in numeric_cols:
        clean_data = clean_data[np.isfinite(clean_data[col])]

    print(f"清洗后数据形状: {clean_data.shape}")
    print(f"数据保留率: {len(clean_data) / len(data):.1%}")

    if len(clean_data) == 0:
        raise ValueError("数据清洗后为空，请检查原始数据质量")

    return clean_data


def safe_dataset_initialization(train_data, val_data, test_data, x_column, y_column, spatial_column):
    """安全的数据集初始化"""
    print("初始化数据集...")

    # 验证每个数据集都不为空
    for name, dataset in [("训练集", train_data), ("验证集", val_data), ("测试集", test_data)]:
        if len(dataset) == 0:
            raise ValueError(f"{name} 为空")
        print(f"{name}: {len(dataset)} 行")

    try:
        train_set, val_set, test_set = datasets.init_dataset_split(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            x_column=x_column,
            y_column=y_column,
            spatial_column=spatial_column,
            batch_size=64,  # 从较小的批次开始
            use_model="gnnwr"
        )
        print("✅ 数据集初始化成功")
        return train_set, val_set, test_set
    except Exception as e:
        print(f"❌ 数据集初始化失败: {e}")
        raise


def should_use_stratify(y_data):
    """判断是否应该使用分层抽样"""
    # 检查目标变量是否适合分层抽样
    unique_values = y_data.unique()
    n_unique = len(unique_values)

    # 如果是连续变量或类别过多，不使用分层抽样
    if n_unique > 10 or (y_data.dtype in ['float64', 'float32'] and n_unique > 0.1 * len(y_data)):
        print(f"目标变量为连续变量或类别过多 ({n_unique} 个唯一值)，不使用分层抽样")
        return None

    # 检查类别分布是否均衡
    value_counts = y_data.value_counts()
    min_count = value_counts.min()
    max_count = value_counts.max()

    if min_count < 5:  # 如果某个类别的样本数太少
        print(f"某些类别样本数过少 (最少 {min_count} 个)，不使用分层抽样")
        return None

    imbalance_ratio = max_count / min_count
    if imbalance_ratio > 20:  # 如果类别极度不均衡
        print(f"类别不均衡严重 (比例 {imbalance_ratio:.1f}:1)，不使用分层抽样")
        return None

    print(f"使用分层抽样，目标变量有 {n_unique} 个类别")
    return y_data


def main():
    """主函数 - 修复版本"""
    try:
        # 1. 加载数据
        print("加载数据...")
        if not os.path.exists('lu_onehot.xlsx'):
            raise FileNotFoundError("数据文件 'lu_onehot.xlsx' 不存在")

        data = pd.read_excel('lu_onehot.xlsx')
        print(f"原始数据: {data.shape}")

        # 2. 定义特征
        x_column = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation',
                    'std_slope', 'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2',
                    'std_high', 'std_aspect', 'glsnow', 'cswe', 'snow_depth_snow_depth',
                    'ERA5温度_ERA5温度', 'era5_swe', 'doy', 'gldas', 'year', 'month', 'scp_start',
                    'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da', 'db', 'dc', 'dd', 'landuse_11',
                    'landuse_12', 'landuse_21', 'landuse_22', 'landuse_23', 'landuse_24',
                    'landuse_31', 'landuse_32', 'landuse_33', 'landuse_41', 'landuse_42',
                    'landuse_43', 'landuse_46', 'landuse_51', 'landuse_52', 'landuse_53',
                    'landuse_62', 'landuse_63', 'landuse_64']

        y_column = ['swe']
        spatial_column = ['longitude', 'latitude']

        # 3. 调试数据问题
        if not debug_data_issues(data, x_column, y_column, spatial_column):
            print("发现数据问题，尝试修复...")

        # 4. 数据清洗
        clean_data = robust_data_cleaning(data, x_column, y_column, spatial_column)

        # 5. 数据标准化
        print("数据标准化...")
        scaler = StandardScaler()
        clean_data[x_column] = scaler.fit_transform(clean_data[x_column])

        # 6. 数据分割 - 修复 stratify 参数问题
        print("数据分割...")
        if len(clean_data) < 100:
            raise ValueError(f"数据量过少 ({len(clean_data)} 行)，无法有效分割")

        # 修复：正确使用 stratify 参数
        y_data = clean_data[y_column[0]]  # 获取目标变量的 Series
        stratify_param = should_use_stratify(y_data)  # 返回 None 或数组

        train_data, temp_data = train_test_split(
            clean_data,
            test_size=0.3,
            random_state=42,
            stratify=stratify_param  # 直接使用 None 或数组
        )

        # 对验证/测试集分割也使用相同的逻辑
        if stratify_param is not None:
            temp_stratify = stratify_param.loc[temp_data.index]
        else:
            temp_stratify = None

        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.33,
            random_state=42,
            stratify=temp_stratify
        )

        print(f"分割结果 - 训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

        # 7. 安全初始化数据集
        train_set, val_set, test_set = safe_dataset_initialization(
            train_data, val_data, test_data, x_column, y_column, spatial_column
        )

        # 8. 配置模型参数（使用更保守的参数）
        optimizer_params = {
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [500, 1000, 1500, 2000],
            "scheduler_gamma": 0.75,
        }

        # 9. 初始化模型
        print("初始化 GNNWR 模型...")
        gnnwr = models.GNNWR(
            train_dataset=train_set,
            valid_dataset=val_set,
            test_dataset=test_set,
            dense_layers=[512, 256, 128],  # 使用更小的网络
            activate_func=nn.PReLU(),
            start_lr=0.001,  # 更小的学习率
            optimizer="Adam",
            model_name="GNNWR_SWE_Fixed",
            model_save_path="result/gnnwr_models",
            log_path="result/gnnwr_logs",
            write_path="result/gnnwr_runs",
            optimizer_params=optimizer_params
        )

        # 10. 创建输出目录
        os.makedirs("result/gnnwr_models", exist_ok=True)
        os.makedirs("result/gnnwr_logs", exist_ok=True)
        os.makedirs("result/gnnwr_runs", exist_ok=True)

        # 11. 训练模型
        print("开始训练模型...")
        gnnwr.add_graph()
        gnnwr.run(max_epoch=500, early_stop=500, print_frequency=100)  # 更少的epoch

        # 12. 评估模型
        gnnwr.load_model('result/gnnwr_models/GNNWR_SWE_Fixed.pkl')
        gnnwr.result()

    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()


# 简化版本 - 完全不使用 stratify
def simple_no_stratify_version():
    """完全不使用分层抽样的简化版本"""
    try:
        # 1. 加载数据
        data = pd.read_excel('lu_onehot.xlsx')
        print(f"原始数据: {data.shape}")

        # 2. 定义特征
        x_column = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation',
                    'std_slope', 'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2',
                    'std_high', 'std_aspect', 'glsnow', 'cswe', 'snow_depth_snow_depth',
                    'ERA5温度_ERA5温度', 'era5_swe', 'doy', 'gldas', 'year', 'month', 'scp_start',
                    'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da', 'db', 'dc', 'dd', 'landuse_11',
                    'landuse_12', 'landuse_21', 'landuse_22', 'landuse_23', 'landuse_24',
                    'landuse_31', 'landuse_32', 'landuse_33', 'landuse_41', 'landuse_42',
                    'landuse_43', 'landuse_46', 'landuse_51', 'landuse_52', 'landuse_53',
                    'landuse_62', 'landuse_63', 'landuse_64']

        y_column = ['swe']
        spatial_column = ['longitude', 'latitude']

        # 3. 简单数据清洗
        clean_data = data[x_column + y_column + spatial_column].dropna()
        print(f"清洗后数据: {clean_data.shape}")

        if len(clean_data) == 0:
            print("数据清洗后为空")
            return

        # 4. 数据标准化
        scaler = StandardScaler()
        clean_data[x_column] = scaler.fit_transform(clean_data[x_column])

        # 5. 简单数据分割（完全不使用 stratify）
        train_data, temp_data = train_test_split(clean_data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.33, random_state=42)

        print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

        # 6. 初始化数据集
        train_set, val_set, test_set = datasets.init_dataset_split(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            x_column=x_column,
            y_column=y_column,
            spatial_column=spatial_column,
            batch_size=64,
            use_model="gnnwr"
        )

        print("✅ 简化版本成功!")
        return train_set, val_set, test_set

    except Exception as e:
        print(f"简化版本失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # 先尝试完整版本
    try:
        main()
    except Exception as e:
        print(f"完整版本失败: {e}")
        print("\n尝试简化版本...")
        # 如果完整版本失败，尝试简化版本
        train_set, val_set, test_set = simple_no_stratify_version()