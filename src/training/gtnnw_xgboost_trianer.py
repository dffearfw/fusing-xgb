from datetime import datetime  # 正确：导入了datetime类
# 使用时：datetime.now() 就可以
import json
import os
from venv import logger

import gnnwr
import joblib
import numpy as np
import pandas as pd
import torch
from gnnwr import models, datasets
from torch import nn

# ⭐⭐⭐ 猴子补丁修复STPNN ⭐⭐⭐
original_STPNN_forward = gnnwr.networks.STPNN.forward


def patched_STPNN_forward(self, x):
    x = x.to(torch.float32)
    batch = x.shape[0]
    height = x.shape[1]

    # 检查并修复维度
    actual_input_dim = x.shape[2]

    # 检查第一层权重
    first_layer = self.fc[0]
    if hasattr(first_layer, 'layer'):
        weight_shape = first_layer.layer.weight.shape

        # 如果权重维度不匹配，修正它
        if weight_shape[1] != actual_input_dim:
            print(f"⚠️ 自动修复STPNN维度: {weight_shape[1]} -> {actual_input_dim}")

            # 创建新层
            new_layer = nn.Linear(actual_input_dim, weight_shape[0])
            nn.init.kaiming_uniform_(new_layer.weight, a=0, mode='fan_in')
            if new_layer.bias is not None:
                new_layer.bias.data.fill_(0)

            # 替换
            first_layer.layer = new_layer
            self.insize = actual_input_dim
            if self.dense_layer and len(self.dense_layer) > 0:
                self.dense_layer[0] = actual_input_dim

    # 展平并继续
    x = torch.reshape(x, shape=(batch * height, x.shape[2]))
    output = self.fc(x)
    output = torch.reshape(output, shape=(batch, height * self.outsize))
    return output


# 应用补丁
gnnwr.networks.STPNN.forward = patched_STPNN_forward

print("✅ STPNN猴子补丁已应用")

class GTNNW_XGBoostTrainer:
    """GTNNW-XGBoost训练器 - 集成GTNNWR权重矩阵与XGBoost"""

    # 默认XGBoost参数
    DEFAULT_PARAMS = {
        'n_estimators': 60,
        'learning_rate': 0.17,
        'max_depth': 5,
        'min_child_weight': 5,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.05,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    # GTNNWR参数
    DEFAULT_GTNNWR_PARAMS = {
        'dense_layers': [[3, 64, 32], [128, 64, 32]],
        'drop_out': 0.4,
        'optimizer': "Adadelta",
        'optimizer_params': {
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [1000, 2000, 3000, 4000],
            "scheduler_gamma": 0.8,
        },
        'max_epoch': 3000,
        'early_stop': 1000,
        'print_frequency': 100
    }

    def __init__(self, params=None, gtnnwr_params=None, use_gtnnwr=True,
                 nan_strategy='median', nan_fill_value=0.0,
                 # 新增参数
                 use_feature_mahalanobis=True,
                 feature_columns_for_distance=None):
        """初始化训练器

        Args:
            use_feature_mahalanobis: 是否使用特征马氏距离
            feature_columns_for_distance: 用于马氏距离计算的特征列
        """
        self.logger = logger
        self.model = None
        self.feature_columns = None
        self.target_column = 'swe'
        self.use_gtnnwr = use_gtnnwr
        self.nan_strategy = nan_strategy
        self.nan_fill_value = nan_fill_value

        # 新增：特征马氏距离相关参数
        self.use_feature_mahalanobis = use_feature_mahalanobis
        self.feature_columns_for_distance = feature_columns_for_distance

        # 存储填充值用于后续预测
        self.nan_fill_values = {}
        self.nan_fill_stats = {}

        # 定义GTNNWR特征列
        self.gtnnwr_x_columns = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation',
                                 'std_slope', 'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high',
                                 'std_aspect', 'glsnow', 'cswe', 'snow_depth_snow_depth', 'ERA5温度_ERA5温度',
                                 'era5_swe', 'doy',
                                 'gldas', 'year', 'month', 'scp_start', 'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da',
                                 'db', 'dc',
                                 'dd']

        # GTNNWR需要空间列和时间列
        self.gtnnwr_spatial_columns = ['X', 'Y']
        self.gtnnwr_temp_columns = ['year', 'month', 'doy']
        self.gtnnwr_id_column = 'id'
        self.gtnnwr_y_column = ['swe']

        # 更新参数
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        self.gtnnwr_params = self.DEFAULT_GTNNWR_PARAMS.copy()
        if gtnnwr_params:
            self.gtnnwr_params.update(gtnnwr_params)

        self.logger.info(f"初始化GTNNW-XGBoost训练器")
        self.logger.info(f"XGBoost参数: {self.params}")
        self.logger.info(f"使用GTNNWR权重增强: {self.use_gtnnwr}")
        self.logger.info(f"使用特征马氏距离: {self.use_feature_mahalanobis}")
        self.logger.info(f"NaN处理策略: {self.nan_strategy}")

    def _handle_nan_values(self, df, is_training=True, fill_values=None):
        """处理NaN值

        Args:
            df (pd.DataFrame): 输入数据
            is_training (bool): 是否为训练阶段
            fill_values (dict): 预计算的填充值

        Returns:
            pd.DataFrame: 处理后的数据
        """
        self.logger.info(f"处理NaN值 - 阶段: {'训练' if is_training else '预测'}")

        df_processed = df.copy()

        # 统计NaN值
        nan_stats = df_processed.isna().sum()
        total_nan = nan_stats.sum()
        total_cells = df_processed.size

        if total_nan > 0:
            nan_percentage = (total_nan / total_cells) * 100
            self.logger.info(f"发现NaN值: {total_nan}/{total_cells} ({nan_percentage:.2f}%)")

            # 按列统计NaN
            nan_columns = nan_stats[nan_stats > 0]
            for col, nan_count in nan_columns.items():
                nan_pct = (nan_count / len(df_processed)) * 100
                self.logger.info(f"  列 '{col}': {nan_count} NaN ({nan_pct:.2f}%)")

        # 处理不同列类型的NaN
        for col in df_processed.columns:
            if df_processed[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # 数值列处理
                nan_count = df_processed[col].isna().sum()

                if nan_count > 0:
                    if self.nan_strategy == 'drop' and is_training:
                        # 删除包含NaN的行（仅训练阶段）
                        self.logger.warning(f"删除包含列 '{col}' NaN的 {nan_count} 行")
                        df_processed = df_processed.dropna(subset=[col])
                    else:
                        # 计算或使用填充值
                        if is_training:
                            if self.nan_strategy == 'mean':
                                fill_value = df_processed[col].mean()
                                self.nan_fill_values[col] = fill_value
                                self.logger.info(f"列 '{col}' 使用均值填充: {fill_value:.4f}")
                            elif self.nan_strategy == 'median':
                                fill_value = df_processed[col].median()
                                self.nan_fill_values[col] = fill_value
                                self.logger.info(f"列 '{col}' 使用中位数填充: {fill_value:.4f}")
                            elif self.nan_strategy == 'zero':
                                fill_value = 0
                                self.nan_fill_values[col] = fill_value
                                self.logger.info(f"列 '{col}' 使用0填充")
                            else:  # 自定义值
                                fill_value = self.nan_fill_value
                                self.nan_fill_values[col] = fill_value
                                self.logger.info(f"列 '{col}' 使用自定义值填充: {fill_value}")

                            # 保存统计信息
                            self.nan_fill_stats[col] = {
                                'strategy': self.nan_strategy,
                                'fill_value': fill_value,
                                'original_nan_count': nan_count,
                                'original_mean': df_processed[col].mean(),
                                'original_median': df_processed[col].median(),
                                'original_std': df_processed[col].std()
                            }
                        else:
                            # 预测阶段使用训练阶段计算的填充值
                            fill_value = self.nan_fill_values.get(col, self.nan_fill_value)
                            self.logger.debug(f"列 '{col}' 使用训练阶段计算的填充值: {fill_value}")

                        # 填充NaN值
                        df_processed[col] = df_processed[col].fillna(fill_value)

            elif df_processed[col].dtype == 'object':
                # 对象类型列处理（字符串）
                nan_count = df_processed[col].isna().sum()

                if nan_count > 0:
                    if is_training:
                        # 对于类别列，使用众数填充或创建新类别
                        if len(df_processed[col].unique()) < 50:  # 假设是类别列
                            mode_value = df_processed[col].mode()
                            if not mode_value.empty:
                                fill_value = mode_value.iloc[0]
                                self.nan_fill_values[col] = fill_value
                                self.logger.info(f"类别列 '{col}' 使用众数填充: {fill_value}")
                            else:
                                fill_value = 'MISSING'
                                self.nan_fill_values[col] = fill_value
                                self.logger.info(f"类别列 '{col}' 使用'MISSING'填充")
                        else:
                            fill_value = 'MISSING'
                            self.nan_fill_values[col] = fill_value
                            self.logger.info(f"文本列 '{col}' 使用'MISSING'填充")
                    else:
                        fill_value = self.nan_fill_values.get(col, 'MISSING')

                    df_processed[col] = df_processed[col].fillna(fill_value)

        # 验证处理后是否还有NaN
        remaining_nan = df_processed.isna().sum().sum()
        if remaining_nan > 0:
            self.logger.warning(f"处理后仍有 {remaining_nan} 个NaN值")
        else:
            self.logger.info("✅ NaN值处理完成，无剩余NaN值")

        return df_processed

    def preprocess_data(self, df, for_gtnnwr=False, is_training=True):
        """数据预处理（这是原始的方法，需要恢复）"""
        self.logger.info("开始数据预处理...")

        # 创建数据副本
        df_clean = df.copy()

        # 验证必要列
        required_columns = ['station_id', 'date', self.target_column]
        missing_columns = [col for col in required_columns if col not in df_clean.columns]
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")

        # 处理NaN值
        df_clean = self._handle_nan_values(df_clean, is_training=is_training)

        # 确保GTNNWR需要的列都存在
        if self.use_gtnnwr:
            # 创建ID列（GTNNWR需要）
            df_clean['id'] = np.arange(len(df_clean))

            # 检查并确保所有需要的列都存在
            gtnnwr_required = (self.gtnnwr_x_columns + self.gtnnwr_spatial_columns +
                               self.gtnnwr_temp_columns + [self.gtnnwr_id_column])
            missing_gtnnwr = [col for col in gtnnwr_required if col not in df_clean.columns]
            if missing_gtnnwr:
                self.logger.warning(f"GTNNWR缺少以下列: {missing_gtnnwr}")
                # 尝试填充缺失列为0或合适的默认值
                for col in missing_gtnnwr:
                    if col == 'id':
                        df_clean[col] = np.arange(len(df_clean))
                    elif col in ['year', 'month', 'doy']:
                        # 如果是时间列，尝试从date列提取
                        if 'date' in df_clean.columns and pd.api.types.is_datetime64_any_dtype(df_clean['date']):
                            df_clean['date'] = pd.to_datetime(df_clean['date'])
                            if col == 'year':
                                df_clean[col] = df_clean['date'].dt.year
                            elif col == 'month':
                                df_clean[col] = df_clean['date'].dt.month
                            elif col == 'doy':
                                df_clean[col] = df_clean['date'].dt.dayofyear
                        else:
                            df_clean[col] = 0.0
                    else:
                        df_clean[col] = 0.0

        # 处理CSWE无效值（如果存在）
        if 'cswe' in df_clean.columns:
            cswe_invalid_mask = df_clean['cswe'] > 200
            if cswe_invalid_mask.sum() > 0:
                df_clean.loc[cswe_invalid_mask, 'cswe'] = np.nan
                # 重新处理NaN值
                df_clean = self._handle_nan_values(df_clean, is_training=is_training)

        # 确定特征列
        exclude_columns = ['station_id', 'date', self.target_column, 'hydrological_doy', 'id']
        exclude_columns.extend([col for col in df_clean.columns if col.startswith('landuse_hash_')])

        # 保留GTNNWR特征列用于加权
        if self.use_gtnnwr:
            # 确保GTNNWR特征列在特征列中
            for col in self.gtnnwr_x_columns:
                if col not in exclude_columns and col not in df_clean.columns:
                    df_clean[col] = 0.0

        self.feature_columns = [col for col in df_clean.columns if col not in exclude_columns]

        if not self.feature_columns:
            raise ValueError("没有找到可用的特征列")

        # 再次检查特征列中的NaN值
        feature_nan_counts = df_clean[self.feature_columns].isna().sum()
        if feature_nan_counts.sum() > 0:
            self.logger.warning(f"特征列中仍有 {feature_nan_counts.sum()} 个NaN值")
            for col, count in feature_nan_counts[feature_nan_counts > 0].items():
                self.logger.warning(f"  特征列 '{col}': {count} 个NaN")
            # 使用最后一次填充
            df_clean[self.feature_columns] = df_clean[self.feature_columns].fillna(
                df_clean[self.feature_columns].median()
            )

        # 准备数据
        X = df_clean[self.feature_columns].values
        y = df_clean[self.target_column].values

        # 检查目标变量中的NaN
        y_nan_count = np.isnan(y).sum()
        if y_nan_count > 0:
            self.logger.warning(f"目标变量 '{self.target_column}' 中有 {y_nan_count} 个NaN值")
            if self.nan_strategy == 'drop' and is_training:
                # 删除目标变量为NaN的行
                valid_mask = ~np.isnan(y)
                X = X[valid_mask]
                y = y[valid_mask]
                df_clean = df_clean.iloc[valid_mask]
                self.logger.info(f"删除了 {y_nan_count} 个目标变量为NaN的样本")
            else:
                # 填充目标变量的NaN
                y_fill_value = np.nanmedian(y)
                y = np.nan_to_num(y, nan=y_fill_value)
                self.logger.info(f"目标变量使用中位数填充: {y_fill_value:.4f}")

        # 分组信息
        df_clean['year'] = pd.to_datetime(df_clean['date']).dt.year
        station_groups = df_clean['station_id'].values
        year_groups = df_clean['year'].values

        # 为GTNNWR准备数据
        gtnnwr_data = None
        if self.use_gtnnwr:
            gtnnwr_data = df_clean.copy()
            # 确保所有GTNNWR需要的列都存在
            for col in self.gtnnwr_x_columns + self.gtnnwr_spatial_columns + self.gtnnwr_temp_columns + [
                self.gtnnwr_id_column]:
                if col not in gtnnwr_data.columns:
                    if col == 'id':
                        gtnnwr_data[col] = np.arange(len(gtnnwr_data))
                    else:
                        gtnnwr_data[col] = 0.0

        # 最终检查
        x_nan_count = np.isnan(X).sum()
        y_nan_count = np.isnan(y).sum()

        self.logger.info(f"✅ 数据预处理完成")
        self.logger.info(f"  样本数: {len(X)}, 特征数: {len(self.feature_columns)}")
        self.logger.info(f"  X中NaN数量: {x_nan_count}, y中NaN数量: {y_nan_count}")

        # 打印特征统计信息
        self.logger.info(f"  特征统计:")
        for i, col in enumerate(self.feature_columns[:5]):  # 只显示前5个特征
            col_values = X[:, i]
            self.logger.info(
                f"    {col}: 均值={col_values.mean():.4f}, 标准差={col_values.std():.4f}, 范围=[{col_values.min():.4f}, {col_values.max():.4f}]")
        if len(self.feature_columns) > 5:
            self.logger.info(f"    ... 和其他 {len(self.feature_columns) - 5} 个特征")

        return X, y, station_groups, year_groups, gtnnwr_data

    def _train_gtnnwr_for_fold(self, train_data, val_data):
        """为单个折叠训练GTNNWR模型并提取权重"""
        """为单个折叠训练GTNNWR模型并提取权重"""
        self.logger.debug("为当前折叠训练GTNNWR模型...")

        print("\n" + "=" * 80)
        print("🧠 GTNNWR模型训练 (当前折叠)")
        print("=" * 80)

        try:
            # 确保所有需要的列都存在
            print("🔍 检查数据完整性...")
            required_columns = (self.gtnnwr_x_columns + self.gtnnwr_spatial_columns +
                                self.gtnnwr_temp_columns + [self.gtnnwr_id_column] + self.gtnnwr_y_column)

            # 检查数据量是否足够
            if len(train_data) < 10 or len(val_data) < 1:
                print(f"⚠️  数据量不足: 训练集{len(train_data)}样本, 验证集{len(val_data)}样本")
                print("⚠️  跳过GTNNWR训练，返回None权重")
                return None, None

            for col in required_columns:
                if col not in train_data.columns:
                    if col == 'id':
                        train_data[col] = np.arange(len(train_data))
                    else:
                        train_data[col] = 0.0
                    print(f"  ⚠️  训练数据缺失列 '{col}'，已填充")
                if col not in val_data.columns:
                    if col == 'id':
                        val_data[col] = np.arange(len(val_data))
                    else:
                        val_data[col] = 0.0
                    print(f"  ⚠️  验证数据缺失列 '{col}'，已填充")

            # 检查数据形状
            print(f"📊 数据形状:")
            print(f"  训练数据: {train_data.shape}")
            print(f"  验证数据: {val_data.shape}")

            # 检查NaN值
            train_nan = train_data[self.gtnnwr_x_columns].isna().sum().sum()
            val_nan = val_data[self.gtnnwr_x_columns].isna().sum().sum()
            if train_nan > 0 or val_nan > 0:
                print(f"  ⚠️  警告: 训练数据有{train_nan}个NaN，验证数据有{val_nan}个NaN")
                # 使用中位数填充
                for col in self.gtnnwr_x_columns:
                    if col in train_data.columns:
                        median_val = train_data[col].median()
                        train_data[col] = train_data[col].fillna(median_val)
                        val_data[col] = val_data[col].fillna(median_val)

            # 初始化GTNNWR数据集
            print("📦 初始化GTNNWR数据集...")

            # 确定用于马氏距离计算的特征列
            if self.use_feature_mahalanobis and self.feature_columns_for_distance is None:
                # 默认使用所有特征列（排除空间和时间列）
                feature_columns_for_distance = self.gtnnwr_x_columns.copy()
                # 排除空间列
                if self.gtnnwr_spatial_columns:
                    feature_columns_for_distance = [col for col in feature_columns_for_distance
                                                    if col not in self.gtnnwr_spatial_columns]
                # 排除时间列
                if self.gtnnwr_temp_columns:
                    feature_columns_for_distance = [col for col in feature_columns_for_distance
                                                    if col not in self.gtnnwr_temp_columns]
                print(f"  📊 特征马氏距离: 使用 {len(feature_columns_for_distance)} 个特征")
            else:
                feature_columns_for_distance = self.feature_columns_for_distance

            try:
                # 使用init_dataset_split，传入特征马氏距离参数
                # 这里我们传入参数，但不检查返回对象的属性
                train_set, val_set, test_set = datasets.init_dataset_split(
                    train_data=train_data,
                    val_data=val_data,
                    test_data=val_data.head(max(1, min(5, len(val_data) // 2))),
                    x_column=self.gtnnwr_x_columns,
                    y_column=self.gtnnwr_y_column,
                    spatial_column=self.gtnnwr_spatial_columns,
                    temp_column=self.gtnnwr_temp_columns,
                    batch_size=min(1024, len(train_data)),
                    shuffle=False,
                    use_model="gtnnwr",
                    # 新增参数 - 传递给baseDataset的构造函数
                    use_feature_mahalanobis=self.use_feature_mahalanobis,
                    feature_columns_for_distance=feature_columns_for_distance
                )
                print(f"✅ 数据集初始化成功")

                # 🚨 添加详细的debug信息
                print(f"\n🔍 详细维度检查...")
                print(f"  train_set.distances形状: {train_set.distances.shape}")
                print(f"  train_set.distances维度数: {train_set.distances.ndim}")

                # 检查距离矩阵内容
                if hasattr(train_set, 'distances') and train_set.distances is not None:
                    print(f"  距离矩阵第一个样本的形状: {train_set.distances[0].shape}")
                    print(f"  距离矩阵第一个样本的前3个参考点:")
                    for i in range(min(3, len(train_set.distances[0]))):
                        print(f"    参考点{i}: {train_set.distances[0][i]}")

                # 检查是否有temporal维度
                if hasattr(train_set, 'temporal') and train_set.temporal is not None:
                    print(f"  train_set.temporal形状: {train_set.temporal.shape}")
                    if hasattr(train_set, 'distances') and train_set.distances is not None:
                        print(f"  时空结合后总维度: {train_set.distances.shape[-1] + train_set.temporal.shape[-1]}")

                # 检查is_need_STNN
                print(f"  train_set.is_need_STNN: {train_set.is_need_STNN}")
                print(f"  val_set.is_need_STNN: {val_set.is_need_STNN}")

                # 检查simple_distance
                print(f"  train_set.simple_distance: {train_set.simple_distance}")

                print(f"\n🔍 检查数据集初始化参数...")
                print(f"  使用的use_model: {'gtnnwr'}")
                print(f"  使用特征马氏距离: {self.use_feature_mahalanobis}")
                if self.use_feature_mahalanobis:
                    print(f"  马氏距离特征数: {len(feature_columns_for_distance)}")

                # ✅ 这里需要检查正确的属性名：is_need_feature_distance
                # 从你的代码可以看出，baseDataset类使用的是is_need_feature_distance，而不是use_feature_mahalanobis
                if hasattr(train_set, 'is_need_feature_distance'):
                    print(f"  数据集是否使用特征距离: {train_set.is_need_feature_distance}")
                else:
                    print(f"  数据集没有is_need_feature_distance属性")

                # 只打印我们自己设定的参数
                print(f"  配置使用特征马氏距离: {self.use_feature_mahalanobis}")
                if self.use_feature_mahalanobis:
                    print(f"  马氏距离特征数: {len(feature_columns_for_distance)}")
            except Exception as error:
                print(f"❌ 数据集初始化失败: {error}")
                import traceback
                print(f"详细错误:\n{traceback.format_exc()}")
                print("⚠️  跳过GTNNWR训练，返回None权重")
                return None, None

            print(f"✅ 数据集初始化完成:")
            print(f"  训练集样本数: {len(train_set) if hasattr(train_set, '__len__') else 'N/A'}")
            print(f"  验证集样本数: {len(val_set) if hasattr(val_set, '__len__') else 'N/A'}")

            # 检查数据集是否为空
            if (not hasattr(train_set, '__len__') or len(train_set) == 0 or
                    not hasattr(val_set, '__len__') or len(val_set) == 0):
                print(f"❌ 数据集为空或无效")
                print("⚠️  跳过GTNNWR训练，返回None权重")
                return None, None

            # 🚨 手动计算正确维度
            print(f"\n🔧 手动计算正确维度...")

            # 方法1：检查数据加载后的实际维度
            try:
                sample = train_set[0]
                if isinstance(sample, tuple) and len(sample) > 0:
                    actual_input_dim = sample[0].shape[-1]
                    print(f"  从train_set[0]获取的实际输入维度: {actual_input_dim}")

                    # 显示更多信息
                    print(f"  sample[0]形状: {sample[0].shape}")
                    print(f"  参考点数量: {sample[0].shape[0]}")

                    # 检查具体内容
                    if sample[0].shape[0] > 0:
                        print(f"  第一个参考点的值: {sample[0][0]}")
                        if len(sample[0][0]) == 3:
                            print(f"    → 可能是[距离_x, 距离_y, 马氏距离]")
                else:
                    actual_input_dim = train_set.distances.shape[-1]
                    print(f"  从distances获取的输入维度: {actual_input_dim}")
            except Exception as e:
                print(f"  检查维度时出错: {e}")
                actual_input_dim = 3  # 默认假设是3维
                print(f"  使用默认维度: {actual_input_dim}")

            # 确保simple_distance=False
            train_set.simple_distance = False
            val_set.simple_distance = False

            # 检查GTNNWR参数的dense_layers
            print(f"\n🔍 检查并调整dense_layers...")
            dense_layers_param = self.gtnnwr_params.get('dense_layers', [[3], [512, 256, 64]])
            print(f"  原始dense_layers: {dense_layers_param}")

            # 调整dense_layers以匹配实际维度
            if isinstance(dense_layers_param, list) and len(dense_layers_param) >= 2:
                # 确保第一个列表的第一个元素是actual_input_dim
                if not isinstance(dense_layers_param[0], list):
                    # 如果不是列表，转换为列表
                    dense_layers_param[0] = [dense_layers_param[0]]

                if dense_layers_param[0][0] != actual_input_dim:
                    print(f"  ⚠️  维度不匹配: dense_layers[0][0]={dense_layers_param[0][0]}, actual={actual_input_dim}")
                    # 修复：创建一个新的dense_layers
                    stpnn_layers = [actual_input_dim]
                    if len(dense_layers_param[0]) > 1:
                        stpnn_layers.extend(dense_layers_param[0][1:])  # 保持其他层

                    adjusted_dense_layers = [stpnn_layers, dense_layers_param[1]]
                    dense_layers_param = adjusted_dense_layers
                    print(f"  调整后的dense_layers: {dense_layers_param}")
            else:
                # 如果格式不对，创建正确的格式
                print(f"  ⚠️  dense_layers格式不正确，创建新格式")
                dense_layers_param = [[actual_input_dim, 64, 32], [128, 64, 32]]
                print(f"  创建的dense_layers: {dense_layers_param}")

            # 训练GTNNWR模型
            print("\n🏋️ 训练GTNNWR模型...")

            try:
                print(f"\n🚀 创建GTNNWR模型...")
                print(f"  使用dense_layers: {dense_layers_param}")
                print(f"  输入维度: {actual_input_dim}")
                print(f"  是否使用特征马氏距离: {self.use_feature_mahalanobis}")
                print(f"原始 simple_distance: {train_set.simple_distance}")
                # ✅ 修复：检查正确的属性名
                if hasattr(train_set, 'is_need_feature_distance'):
                    print(f"数据集是否使用特征距离: {train_set.is_need_feature_distance}")
                else:
                    print(f"数据集没有is_need_feature_distance属性")
                print(f"配置使用特征马氏距离: {self.use_feature_mahalanobis}")

                # 方法1: 设置 simple_distance=False
                train_set.simple_distance = False
                val_set.simple_distance = False

                # 创建GTNNWR
                gtnnwr = models.GTNNWR(
                    train_dataset=train_set,
                    valid_dataset=val_set,
                    test_dataset=train_set,
                    dense_layers=dense_layers_param,
                    drop_out=self.gtnnwr_params.get('drop_out', 0.4),
                    optimizer=self.gtnnwr_params.get('optimizer', "Adadelta"),
                    optimizer_params=self.gtnnwr_params.get('optimizer_params', {}),
                    model_name=f"GTNNWR_Fold",
                    model_save_path="result/gtnnwr_models_temp",
                    log_path="result/gtnnwr_logs_temp",
                    write_path="result/gtnnwr_runs_temp"
                )

                # 🚨 关键：跳过add_graph()，直接训练
                print("🕸️ 跳过图结构添加（避免维度错误）...")
                # gtnnwr.add_graph()  # 注释掉这行

                print(f"⚙️ 开始训练...")
                gtnnwr.run(
                    max_epoch=min(500, self.gtnnwr_params.get('max_epoch', 3000)),  # 先训练500轮
                    early_stop=min(100, self.gtnnwr_params.get('early_stop', 1000)),  # 早停100轮
                    print_frequency=self.gtnnwr_params.get('print_frequency', 100)
                )
            except Exception as model_error:
                print(f"❌ GTNNWR模型创建或训练失败: {model_error}")
                import traceback
                print(f"详细错误:\n{traceback.format_exc()}")
                print("⚠️  跳过GTNNWR训练，返回None权重")
                return None, None

            # 提取权重矩阵
            def extract_weights(gtnnwr_instance, dataset, dataset_name="数据集"):
                """提取GTNNWR模型输出的权重矩阵"""
                if dataset is None or not hasattr(dataset, 'dataloader'):
                    print(f"  ❌ {dataset_name}无效或没有dataloader")
                    return None

                model = gtnnwr_instance._model
                model.eval()
                device = gtnnwr_instance._device

                all_weights = []
                sample_count = 0

                print(f"\n📥 从{dataset_name}提取权重...")

                with torch.no_grad():
                    try:
                        total_batches = 0
                        for batch_idx, batch in enumerate(dataset.dataloader):
                            if batch is None or len(batch) < 2:
                                continue

                            distances, features = batch[:2]
                            distances = distances.to(device)

                            # 获取模型输出
                            weights = model(distances)

                            # 检查权重中的NaN
                            if torch.isnan(weights).any():
                                print(f"  ⚠️  批次{batch_idx}权重中包含NaN值，使用1填充")
                                weights = torch.nan_to_num(weights, nan=1.0)

                            all_weights.append(weights.cpu().numpy())
                            sample_count += weights.shape[0]
                            total_batches += 1

                        print(f"  ✅ 完成: 总共处理{total_batches}个批次，{sample_count}个样本")

                    except Exception as e:
                        print(f"  ❌ 提取权重时出错: {e}")
                        import traceback
                        print(traceback.format_exc())
                        return None

                if all_weights:
                    weights_combined = np.concatenate(all_weights, axis=0)

                    # 检查并处理NaN值
                    nan_count = np.isnan(weights_combined).sum()
                    if nan_count > 0:
                        print(f"  ⚠️  权重矩阵中有{nan_count}个NaN值，使用1填充")
                        weights_combined = np.nan_to_num(weights_combined, nan=1.0)

                    print(f"  ✅ 提取完成: {weights_combined.shape} (样本数×特征数)")
                    return weights_combined
                else:
                    print(f"  ❌ 提取失败: 没有获取到权重")
                    return None

            # 提取训练集和验证集权重
            train_weights = extract_weights(gtnnwr, train_set, "训练集")
            val_weights = extract_weights(gtnnwr, val_set, "验证集")

            if train_weights is not None and val_weights is not None:
                # 检查并调整维度
                expected_cols = len(self.gtnnwr_x_columns)

                print(f"\n🔧 维度检查与调整:")
                print(f"  期望特征数: {expected_cols}")

                # 检查训练集权重维度
                if train_weights.shape[1] != expected_cols:
                    print(f"  ⚠️  训练权重维度不匹配: {train_weights.shape[1]} != {expected_cols}")
                    if train_weights.shape[1] == expected_cols + 1:
                        train_weights = train_weights[:, :expected_cols]
                        print(f"  ✅ 修复：去掉最后一列，新形状: {train_weights.shape}")
                    elif train_weights.shape[1] > expected_cols:
                        train_weights = train_weights[:, :expected_cols]
                        print(f"  ✅ 修复：截断到期望长度，新形状: {train_weights.shape}")
                    else:
                        padding = np.ones((train_weights.shape[0], expected_cols - train_weights.shape[1]))
                        train_weights = np.hstack([train_weights, padding])
                        print(f"  ✅ 修复：填充到期望长度，新形状: {train_weights.shape}")

                # 检查验证集权重维度
                if val_weights.shape[1] != expected_cols:
                    print(f"  ⚠️  验证权重维度不匹配: {val_weights.shape[1]} != {expected_cols}")
                    if val_weights.shape[1] == expected_cols + 1:
                        val_weights = val_weights[:, :expected_cols]
                        print(f"  ✅ 修复：去掉最后一列，新形状: {val_weights.shape}")
                    elif val_weights.shape[1] > expected_cols:
                        val_weights = val_weights[:, :expected_cols]
                        print(f"  ✅ 修复：截断到期望长度，新形状: {val_weights.shape}")
                    else:
                        padding = np.ones((val_weights.shape[0], expected_cols - val_weights.shape[1]))
                        val_weights = np.hstack([val_weights, padding])
                        print(f"  ✅ 修复：填充到期望长度，新形状: {val_weights.shape}")

                self.logger.debug(f"  提取到权重矩阵: 训练集{train_weights.shape}, 验证集{val_weights.shape}")
                return train_weights, val_weights
            else:
                print(f"\n❌ GTNNWR权重提取失败")
                self.logger.warning("  未能提取到权重矩阵")
                return None, None

        except Exception as e:
            print(f"\n❌ GTNNWR训练失败: {str(e)}")
            import traceback
            print(f"详细错误:\n{traceback.format_exc()}")
            self.logger.warning(f"  GTNNWR训练失败: {str(e)}")
            return None, None

    def _apply_gtnnwr_weights(self, X, weights, feature_columns, gtnnwr_x_columns):
        """应用GTNNWR权重到特征矩阵

        Args:
            X (np.array): 原始特征矩阵
            weights (np.array): 权重矩阵
            feature_columns (list): 特征列名
            gtnnwr_x_columns (list): GTNNWR特征列名

        Returns:
            np.array: 加权后的特征矩阵
        """
        if weights is None:
            self.logger.warning("权重矩阵为None，返回原始特征")
            return X

        # ✅ 修复1: 首先检查样本数是否匹配
        if X.shape[0] != weights.shape[0]:
            self.logger.error(f"❌ 样本数不匹配: X有{X.shape[0]}个样本, 权重有{weights.shape[0]}个样本")

            # 尝试修复样本数不匹配的问题
            if weights.shape[0] < X.shape[0]:
                # 如果权重样本数较少，重复权重以匹配X的样本数
                repeat_times = int(np.ceil(X.shape[0] / weights.shape[0]))
                weights_repeated = np.tile(weights, (repeat_times, 1))
                weights = weights_repeated[:X.shape[0], :]
                self.logger.warning(f"✅ 权重样本数不足，重复权重到{weights.shape[0]}个样本")
            else:
                # 如果权重样本数较多，截断到X的样本数
                weights = weights[:X.shape[0], :]
                self.logger.warning(f"✅ 权重样本数过多，截断到{weights.shape[0]}个样本")

        # ✅ 修复2: 处理维度不匹配问题
        if weights.shape[1] != len(gtnnwr_x_columns):
            self.logger.warning(f"⚠️ 权重矩阵特征数({weights.shape[1]})与GTNNWR特征数({len(gtnnwr_x_columns)})不匹配")

            # 自动调整权重维度
            if weights.shape[1] > len(gtnnwr_x_columns):
                # 如果权重是35列，GTNNWR特征是34列，去掉最后一列
                weights = weights[:, :len(gtnnwr_x_columns)]
                self.logger.info(f"✅ 自动调整：截断权重矩阵到 {weights.shape[1]} 列")
            elif weights.shape[1] < len(gtnnwr_x_columns):
                # 如果权重列数少，填充1.0
                padding = np.ones((weights.shape[0], len(gtnnwr_x_columns) - weights.shape[1]))
                weights = np.hstack([weights, padding])
                self.logger.info(f"✅ 自动调整：填充权重矩阵到 {weights.shape[1]} 列")

        # 检查输入中的NaN
        x_nan_count = np.isnan(X).sum()
        if x_nan_count > 0:
            self.logger.warning(f"⚠️ 输入特征矩阵中有 {x_nan_count} 个NaN值，使用列均值填充")
            col_means = np.nanmean(X, axis=0)
            for i in range(X.shape[1]):
                X[:, i] = np.where(np.isnan(X[:, i]), col_means[i], X[:, i])

        # 检查权重中的NaN
        weights_nan_count = np.isnan(weights).sum()
        if weights_nan_count > 0:
            self.logger.warning(f"⚠️ 权重矩阵中有 {weights_nan_count} 个NaN值，使用1填充")
            weights = np.nan_to_num(weights, nan=1.0)

        # 创建特征映射：特征列到GTNNWR特征列的索引
        feature_to_gtnnwr = {}
        for i, feat in enumerate(feature_columns):
            if feat in gtnnwr_x_columns:
                feature_to_gtnnwr[i] = gtnnwr_x_columns.index(feat)

        # 添加调试信息
        matched_count = len(feature_to_gtnnwr)
        self.logger.info(f"🔍 特征匹配: 匹配了 {matched_count}/{len(feature_columns)} 个特征")

        if matched_count == 0:
            self.logger.warning("⚠️ 没有找到匹配的特征，无法应用权重")
            return X

        # ✅ 关键修复：应用权重，即使有NaN
        X_weighted = X.copy()
        changed_count = 0

        for feat_idx, gtnnwr_idx in feature_to_gtnnwr.items():
            # 获取原始特征值和权重
            original_values = X[:, feat_idx]
            weight_values = weights[:, gtnnwr_idx]

            # 检查并处理NaN
            original_nan = np.isnan(original_values).sum()
            weight_nan = np.isnan(weight_values).sum()

            if original_nan > 0:
                original_values = np.nan_to_num(original_values, nan=0.0)

            if weight_nan > 0:
                weight_values = np.nan_to_num(weight_values, nan=1.0)

            # 应用权重：X × weight
            weighted_values = original_values * weight_values

            # 检查是否真的改变了（忽略NaN）
            mask = ~np.isnan(original_values) & ~np.isnan(weighted_values)
            if mask.any():
                if not np.allclose(original_values[mask], weighted_values[mask], rtol=1e-10):
                    changed_count += 1

            X_weighted[:, feat_idx] = weighted_values

        # 添加验证输出
        change_ratio = changed_count / matched_count if matched_count > 0 else 0
        self.logger.info(f"✅ 权重应用结果: 修改了 {changed_count}/{matched_count} 个特征 ({change_ratio:.1%})")

        # 检查几个关键特征的变化
        if changed_count > 0:
            key_features = ['elevation', 'X', 'Y', 'Z', 'slope', 'doy']
            for feat in key_features:
                if feat in feature_columns and feat in gtnnwr_x_columns:
                    feat_idx = feature_columns.index(feat)
                    gtnnwr_idx = gtnnwr_x_columns.index(feat)

                    # 检查第一个样本
                    if len(X) > 0 and len(weights) > 0:
                        if (not np.isnan(X[0, feat_idx]) and
                                not np.isnan(X_weighted[0, feat_idx]) and
                                feat_idx < weights.shape[1]):
                            original = X[0, feat_idx]
                            weighted = X_weighted[0, feat_idx]
                            weight_val = weights[0, gtnnwr_idx]

                            if abs(weighted - original) > 1e-10:
                                self.logger.info(f"   {feat}: {original:.4f} × {weight_val:.4f} = {weighted:.4f} "
                                                 f"(Δ={weighted - original:+.4f})")

        # 检查输出中的NaN
        output_nan_count = np.isnan(X_weighted).sum()
        if output_nan_count > 0:
            self.logger.warning(f"⚠️ 加权后的特征矩阵中有 {output_nan_count} 个NaN值，使用原始值")
            X_weighted = np.where(np.isnan(X_weighted), X, X_weighted)

        # 验证最终形状
        if X_weighted.shape != X.shape:
            self.logger.error(f"❌ 形状不匹配: 加权后{X_weighted.shape} != 原始{X.shape}")
            return X  # 返回原始特征避免进一步错误

        return X_weighted


    def cross_validate(self, X, y, groups, cv_type='station', gtnnwr_data=None):
        """执行带GTNNWR权重的交叉验证

        Args:
            X (np.array): 特征数据
            y (np.array): 目标变量
            groups (np.array): 分组信息
            cv_type (str): 交叉验证类型 ('station' 或 'yearly')
            gtnnwr_data (pd.DataFrame): GTNNWR需要的完整数据

        Returns:
            dict: 交叉验证结果
        """
        from sklearn.model_selection import LeaveOneGroupOut

        logo = LeaveOneGroupOut()
        all_predictions = []
        all_true_values = []
        fold_results = {}

        fold_maes = []
        fold_rmses = []
        fold_rs = []
        fold_samples = []

        unique_groups = np.unique(groups)
        total_folds = len(unique_groups)

        print("\n" + "=" * 100)
        print(f"🚀 开始{cv_type}交叉验证，共{total_folds}个折叠")
        print(f"使用GTNNWR权重增强: {self.use_gtnnwr}")
        print(f"NaN处理策略: {self.nan_strategy}")
        print("=" * 100)

        self.logger.info(f"开始{cv_type}交叉验证，共{total_folds}个折叠...")
        self.logger.info(f"使用GTNNWR权重增强: {self.use_gtnnwr}")

        for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups)):
            group_id = groups[val_idx[0]]
            train_size = len(train_idx)
            val_size = len(val_idx)

            print("\n" + "=" * 80)
            print(f"🎯 {cv_type} Fold {fold + 1}/{total_folds}: 分组 {group_id}")
            print(f"   训练集: {train_size}样本, 验证集: {val_size}样本")
            print("=" * 80)

            self.logger.info(
                f"{cv_type} Fold {fold + 1}/{total_folds}: {group_id} (训练集{train_size}, 验证集{val_size})")

            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # GTNNWR权重增强
            if self.use_gtnnwr and gtnnwr_data is not None:
                print(f"\n📊 GTNNWR权重增强阶段")

                # 获取当前折叠的训练和验证数据
                train_data_fold = gtnnwr_data.iloc[train_idx].copy()
                val_data_fold = gtnnwr_data.iloc[val_idx].copy()

                print(f"  训练数据形状: {train_data_fold.shape}")
                print(f"  验证数据形状: {val_data_fold.shape}")

                # 训练GTNNWR并提取权重
                print(f"\n🧠 训练GTNNWR模型...")
                train_weights, val_weights = self._train_gtnnwr_for_fold(
                    train_data_fold,
                    val_data_fold
                )

                if train_weights is not None and val_weights is not None:
                    print(f"\n✅ GTNNWR训练完成，准备应用权重")

                    # 应用权重
                    print(f"\n🔄 应用权重到特征矩阵...")
                    X_train = self._apply_gtnnwr_weights(
                        X_train, train_weights,
                        self.feature_columns, self.gtnnwr_x_columns
                    )
                    X_val = self._apply_gtnnwr_weights(
                        X_val, val_weights,
                        self.feature_columns, self.gtnnwr_x_columns
                    )
                else:
                    print(f"\n❌ GTNNWR权重提取失败，使用原始特征")
            else:
                print(f"\n📝 未使用GTNNWR权重增强")

            # 训练XGBoost模型
            print(f"\n🌲 训练XGBoost模型...")
            import xgboost as xgb
            model = xgb.XGBRegressor(**self.params)
            model.fit(X_train, y_train)

            # 预测
            print(f"  进行预测...")
            y_pred = model.predict(X_val)

            # 存储结果
            all_predictions.extend(y_pred)
            all_true_values.extend(y_val)

            # 计算性能指标
            fold_metrics = self._evaluate_predictions(y_val, y_pred)
            fold_results[group_id] = fold_metrics

            fold_maes.append(fold_metrics['MAE'])
            fold_rmses.append(fold_metrics['RMSE'])
            fold_rs.append(fold_metrics['R'])
            fold_samples.append(fold_metrics['样本数'])

            r_display = fold_metrics['R']
            r_str = f"{r_display:.3f}" if not np.isnan(r_display) else "NaN"

            # 打印Fold结果
            print(f"\n📊 Fold {fold + 1} 性能指标:")
            print(f"  MAE:  {fold_metrics['MAE']:.3f} mm")
            print(f"  RMSE: {fold_metrics['RMSE']:.3f} mm")
            print(f"  R:    {r_str}")
            print(f"  样本数: {fold_metrics['样本数']}")

            self.logger.info(
                f"  Fold {fold + 1} 性能: MAE={fold_metrics['MAE']:.3f}, R={r_str}"
            )

        # 计算总体性能
        overall_metrics = self._evaluate_predictions(
            np.array(all_true_values),
            np.array(all_predictions)
        )

        # 计算统计量
        def safe_statistic(values, func):
            valid_values = [v for v in values if not np.isnan(v)]
            if len(valid_values) == 0:
                return np.nan
            return func(valid_values)

        mean_metrics = {
            'MAE': safe_statistic(fold_maes, np.mean),
            'RMSE': safe_statistic(fold_rmses, np.mean),
            'R': safe_statistic(fold_rs, np.mean),
            '样本数': np.sum(fold_samples)
        }

        median_metrics = {
            'MAE': safe_statistic(fold_maes, np.median),
            'RMSE': safe_statistic(fold_rmses, np.median),
            'R': safe_statistic(fold_rs, np.median),
            '样本数': np.sum(fold_samples)
        }

        std_metrics = {
            'MAE': safe_statistic(fold_maes, np.std),
            'RMSE': safe_statistic(fold_rmses, np.std),
            'R': safe_statistic(fold_rs, np.std)
        }

        print("\n" + "=" * 100)
        print(f"🎉 {cv_type}交叉验证完成!")
        print("=" * 100)

        print(f"\n📈 聚合性能指标:")
        print(f"  MAE:  {overall_metrics['MAE']:.3f} mm")
        print(f"  RMSE: {overall_metrics['RMSE']:.3f} mm")
        print(f"  R:    {overall_metrics['R']:.3f}")
        print(f"  总样本数: {overall_metrics['样本数']}")

        self.logger.info(f"✅ {cv_type}交叉验证完成")
        self.logger.info(f"  聚合性能: MAE={overall_metrics['MAE']:.3f}mm, R={overall_metrics['R']:.3f}")

        return {
            'overall': overall_metrics,
            'mean': mean_metrics,
            'median': median_metrics,
            'std': std_metrics,
            'by_fold': fold_results,
            'predictions': np.array(all_predictions),
            'true_values': np.array(all_true_values),
            'folds': total_folds,
            'fold_metrics': {
                'MAE': fold_maes,
                'RMSE': fold_rmses,
                'R': fold_rs,
                'samples': fold_samples
            }
        }

    def _evaluate_predictions(self, y_true, y_pred):
        """评估预测结果"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from scipy.stats import pearsonr

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            return {
                'MAE': np.nan,
                'RMSE': np.nan,
                'R': np.nan,
                'R_pvalue': np.nan,
                '样本数': 0,
                '总样本数': len(y_true),
                '有效样本比例': 0.0
            }

        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

        def safe_pearsonr(x, y):
            if len(x) <= 1 or np.all(x == x[0]) or np.all(y == y[0]):
                return np.nan, np.nan
            if np.std(x) == 0 or np.std(y) == 0:
                return np.nan, np.nan
            try:
                return pearsonr(x, y)
            except:
                return np.nan, np.nan

        r, p_value = safe_pearsonr(y_true_clean, y_pred_clean)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R': r,
            'R_pvalue': p_value,
            '样本数': len(y_true_clean),
            '总样本数': len(y_true),
            '有效样本比例': len(y_true_clean) / len(y_true) if len(y_true) > 0 else 0
        }


    def run_complete_analysis(self, df, output_dir=None):
        """运行完整分析流程"""
        self.logger.info("=" * 70)
        self.logger.info("🚀 开始GTNNW-XGBoost完整分析流程")
        self.logger.info(f"使用特征马氏距离: {self.use_feature_mahalanobis}")
        self.logger.info("=" * 70)

        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"./gtnnw_xgboost_results_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"输出目录: {output_dir}")

        try:
            # 1. 数据预处理
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 1: 数据预处理")
            self.logger.info("=" * 50)

            X, y, station_groups, year_groups, gtnnwr_data = self.preprocess_data(df, is_training=True)

            results = {
                'preprocessing': {
                    'samples': len(X),
                    'features': len(self.feature_columns),
                    'stations': len(np.unique(station_groups)),
                    'years': len(np.unique(year_groups)),
                    'use_gtnnwr': self.use_gtnnwr,
                    'use_feature_mahalanobis': self.use_feature_mahalanobis,
                    'nan_strategy': self.nan_strategy,
                    'nan_fill_stats': self.nan_fill_stats
                }
            }

            # 2. 年度交叉验证
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 2: 年度交叉验证")
            self.logger.info("=" * 50)

            results['yearly_cv'] = self.cross_validate(
                X, y, year_groups, 'yearly', gtnnwr_data
            )

            # 3. 站点交叉验证
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 3: 站点交叉验证")
            self.logger.info("=" * 50)

            results['station_cv'] = self.cross_validate(
                X, y, station_groups, 'station', gtnnwr_data
            )

            # 4. 训练最终模型
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 4: 训练最终模型")
            self.logger.info("=" * 50)

            results['final_model'] = self.train_final_model(X, y, gtnnwr_data)

            # 5. 特征重要性分析
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 5: 特征重要性分析")
            self.logger.info("=" * 50)

            results['feature_importance'] = self.get_feature_importance()

            # 6. 保存结果
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 6: 保存结果")
            self.logger.info("=" * 50)

            self._save_results(results, output_dir)

            # 7. 生成报告
            report = self._generate_report(results)
            print(report)

            self.logger.info("🎯 完整分析完成！")
            return results

        except Exception as e:
            self.logger.error(f"❌ 分析流程失败: {str(e)}")
            raise

    def _save_results(self, results, output_dir):
        """保存结果"""
        try:
            # 保存最终模型
            if 'final_model' in results:
                model_path = f'{output_dir}/final_model.pkl'
                joblib.dump(results['final_model'], model_path)
                self.logger.info(f"✅ 模型保存: {model_path}")

            # 保存NaN处理信息
            nan_info_path = f'{output_dir}/nan_handling_info.json'
            nan_info = {
                'strategy': self.nan_strategy,
                'fill_values': self.nan_fill_values,
                'fill_stats': self.nan_fill_stats
            }
            with open(nan_info_path, 'w', encoding='utf-8') as f:
                json.dump(nan_info, f, indent=2, ensure_ascii=False,
                          default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
            self.logger.info(f"✅ NaN处理信息保存: {nan_info_path}")

            # 保存详细结果
            eval_results = {
                'training_info': {
                    'timestamp': datetime.now().isoformat(),
                    'feature_columns': self.feature_columns,
                    'gtnnwr_x_columns': self.gtnnwr_x_columns,
                    'use_gtnnwr': self.use_gtnnwr,
                    'use_feature_mahalanobis': self.use_feature_mahalanobis,
                    'nan_strategy': self.nan_strategy,
                    'total_samples': results.get('preprocessing', {}).get('samples', 0)
                },
                'model_parameters': self.params,
                'gtnnwr_parameters': self.gtnnwr_params,
                'station_cross_validation': results.get('station_cv', {}),
                'yearly_cross_validation': results.get('yearly_cv', {})
            }

            eval_path = f'{output_dir}/evaluation_results.json'
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False, default=float)
            self.logger.info(f"✅ 详细评估结果保存: {eval_path}")

            # 生成可视化
            self._create_scatter_plots(results, output_dir)

        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")

    def _generate_report(self, results):
        """生成分析报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("📊 GTNNW-XGBoost模型分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"使用GTNNWR权重增强: {self.use_gtnnwr}")
        report_lines.append(f"使用特征马氏距离: {self.use_feature_mahalanobis}")
        report_lines.append(f"NaN处理策略: {self.nan_strategy}")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # 站点CV结果
        if 'station_cv' in results:
            station = results['station_cv']
            report_lines.append("📍 站点交叉验证 (空间评估):")
            report_lines.append(f"  聚合MAE: {station['overall']['MAE']:.3f} mm")
            report_lines.append(f"  聚合RMSE: {station['overall']['RMSE']:.3f} mm")
            report_lines.append(f"  聚合R: {station['overall']['R']:.3f}")
            report_lines.append(f"  折叠数: {station['folds']}")
            report_lines.append("")

        # 年度CV结果
        if 'yearly_cv' in results:
            yearly = results['yearly_cv']
            report_lines.append("📅 年度交叉验证 (时间评估):")
            report_lines.append(f"  聚合MAE: {yearly['overall']['MAE']:.3f} mm")
            report_lines.append(f"  聚合RMSE: {yearly['overall']['RMSE']:.3f} mm")
            report_lines.append(f"  聚合R: {yearly['overall']['R']:.3f}")
            report_lines.append(f"  折叠数: {yearly['folds']}")
            report_lines.append("")

        report_lines.append("\n" + "=" * 80)
        return "\n".join(report_lines)


# 便捷使用函数 - 新增支持特征马氏距离
def train_gtnnw_xgboost_model(data_df, output_dir=None, use_gtnnwr=True,
                              nan_strategy='median', nan_fill_value=0.0,
                              use_feature_mahalanobis=False,
                              feature_columns_for_distance=None):
    """便捷函数：训练GTNNW-XGBoost模型

    Args:
        data_df (pd.DataFrame): 包含特征和SWE的数据
        output_dir (str, optional): 输出目录路径
        use_gtnnwr (bool): 是否使用GTNNWR权重
        nan_strategy (str): NaN处理策略
        nan_fill_value (float): 填充NaN的值
        use_feature_mahalanobis (bool): 是否使用特征马氏距离
        feature_columns_for_distance (list): 用于马氏距离计算的特征列

    Returns:
        dict: 包含所有训练结果的字典
    """
    trainer = GTNNW_XGBoostTrainer(
        use_gtnnwr=use_gtnnwr,
        nan_strategy=nan_strategy,
        nan_fill_value=nan_fill_value,
        use_feature_mahalanobis=use_feature_mahalanobis,
        feature_columns_for_distance=feature_columns_for_distance
    )
    return trainer.run_complete_analysis(data_df, output_dir)


# 对比实验函数 - 新增支持特征马氏距离
def compare_models(data_df, output_dir=None):
    """对比不同配置的性能"""

    print("=" * 80)
    print("🔬 开始模型对比实验")
    print("=" * 80)

    # 1. 纯XGBoost
    print("\n1. 训练纯XGBoost模型...")
    xgb_trainer = GTNNW_XGBoostTrainer(use_gtnnwr=False, nan_strategy='median')
    xgb_results = xgb_trainer.run_complete_analysis(
        data_df,
        output_dir=os.path.join(output_dir, "xgboost_only") if output_dir else None
    )

    # 2. GTNNW-XGBoost (无特征马氏距离)
    print("\n2. 训练GTNNW-XGBoost模型 (无特征马氏距离)...")
    gtnnw_trainer1 = GTNNW_XGBoostTrainer(use_gtnnwr=True, nan_strategy='median',
                                          use_feature_mahalanobis=False)
    gtnnw_results1 = gtnnw_trainer1.run_complete_analysis(
        data_df,
        output_dir=os.path.join(output_dir, "gtnnw_xgboost_no_mahalanobis") if output_dir else None
    )

    # 3. GTNNW-XGBoost (有特征马氏距离)
    print("\n3. 训练GTNNW-XGBoost模型 (有特征马氏距离)...")
    gtnnw_trainer2 = GTNNW_XGBoostTrainer(use_gtnnwr=True, nan_strategy='median',
                                          use_feature_mahalanobis=True)
    gtnnw_results2 = gtnnw_trainer2.run_complete_analysis(
        data_df,
        output_dir=os.path.join(output_dir, "gtnnw_xgboost_with_mahalanobis") if output_dir else None
    )

    # 4. 对比分析
    print("\n" + "=" * 80)
    print("📊 模型对比结果")
    print("=" * 80)

    results_to_compare = [
        ("纯XGBoost", xgb_results),
        ("GTNNW-XGBoost (无马氏距离)", gtnnw_results1),
        ("GTNNW-XGBoost (有马氏距离)", gtnnw_results2)
    ]

    for name, res in results_to_compare:
        if 'station_cv' in res and 'overall' in res['station_cv']:
            r = res['station_cv']['overall']['R']
            if not np.isnan(r):
                print(f"{name}:")
                print(f"  站点CV R = {r:.3f}")
                print(f"  MAE = {res['station_cv']['overall']['MAE']:.3f} mm")
                print(f"  RMSE = {res['station_cv']['overall']['RMSE']:.3f} mm")
                print()

    return {
        'xgboost': xgb_results,
        'gtnnw_xgboost_no_mahalanobis': gtnnw_results1,
        'gtnnw_xgboost_with_mahalanobis': gtnnw_results2
    }