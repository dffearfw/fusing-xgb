import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneGroupOut
import os
import joblib
import json
from datetime import datetime
import seaborn as sns
from scipy import stats

# 新增导入
import torch
import torch.nn as nn
from gnnwr import models, datasets
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger("GNNW_XGBoostTrainer")


class GNNW_XGBoostTrainer:
    """GNNW-XGBoost训练器 - 集成GNNWR权重矩阵与XGBoost"""

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

    # GNNWR参数
    DEFAULT_GNNWR_PARAMS = {
        'dense_layers': [1024, 512, 256],
        'activate_func': nn.PReLU(init=0.4),
        'start_lr': 0.1,
        'optimizer': "Adadelta",
        'max_epoch': 3000,  # 交叉验证中减少训练轮数
        'early_stop': 1000,
        'print_frequency': 100
    }

    def __init__(self, params=None, gnnwr_params=None, use_gnnwr=True):
        """初始化训练器

        Args:
            params (dict, optional): XGBoost参数
            gnnwr_params (dict, optional): GNNWR参数
            use_gnnwr (bool): 是否使用GNNWR权重增强
        """
        self.logger = logger
        self.model = None
        self.feature_columns = None
        self.target_column = 'swe'
        self.use_gnnwr = use_gnnwr

        # 定义GNNWR特征列（与原始GNNWR训练保持一致）
        self.gnnwr_x_columns = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation',
                                'std_slope',
                                'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high', 'std_aspect',
                                'glsnow', 'cswe', 'snow_depth_snow_depth', 'ERA5温度_ERA5温度', 'era5_swe', 'doy',
                                'gldas',
                                'year', 'month', 'scp_start', 'scp_end', 'd1', 'd2', 'X', 'Y', 'Z','da', 'db', 'dc',
                                'dd']
        self.gnnwr_y_column = ['swe']
        self.gnnwr_spatial_columns = ['X', 'Y', 'Z']

        # 更新参数
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        self.gnnwr_params = self.DEFAULT_GNNWR_PARAMS.copy()
        if gnnwr_params:
            self.gnnwr_params.update(gnnwr_params)

        self.logger.info(f"初始化GNNW-XGBoost训练器")
        self.logger.info(f"XGBoost参数: {self.params}")
        self.logger.info(f"使用GNNWR权重增强: {self.use_gnnwr}")

    def preprocess_data(self, df, for_gnnwr=False):
        """数据预处理

        Args:
            df (pd.DataFrame): 原始数据
            for_gnnwr (bool): 是否为GNNWR处理数据

        Returns:
            tuple: 处理后的特征矩阵、目标向量、分组信息
        """
        self.logger.info("开始数据预处理...")

        # 创建数据副本
        df_clean = df.copy()

        # 验证必要列
        required_columns = ['station_id', 'date', self.target_column]
        missing_columns = [col for col in required_columns if col not in df_clean.columns]
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")

        # 确保GNNWR需要的列都存在
        if self.use_gnnwr:
            gnnwr_required = self.gnnwr_x_columns + self.gnnwr_spatial_columns
            missing_gnnwr = [col for col in gnnwr_required if col not in df_clean.columns]
            if missing_gnnwr:
                self.logger.warning(f"GNNWR缺少以下列: {missing_gnnwr}")
                # 尝试填充缺失列为0
                for col in missing_gnnwr:
                    df_clean[col] = 0.0

        # 处理CSWE无效值
        if 'cswe' in df_clean.columns:
            cswe_invalid_mask = df_clean['cswe'] > 200
            if cswe_invalid_mask.sum() > 0:
                df_clean.loc[cswe_invalid_mask, 'cswe'] = np.nan

        # 确定特征列
        exclude_columns = ['station_id', 'date', self.target_column, 'hydrological_doy']
        exclude_columns.extend([col for col in df_clean.columns if col.startswith('landuse_hash_')])

        # 保留GNNWR特征列用于加权
        if self.use_gnnwr:
            # 确保GNNWR特征列在特征列中
            for col in self.gnnwr_x_columns:
                if col not in exclude_columns and col not in df_clean.columns:
                    df_clean[col] = 0.0

        self.feature_columns = [col for col in df_clean.columns if col not in exclude_columns]

        if not self.feature_columns:
            raise ValueError("没有找到可用的特征列")

        # 准备数据
        X = df_clean[self.feature_columns].values
        y = df_clean[self.target_column].values

        # 分组信息
        df_clean['year'] = pd.to_datetime(df_clean['date']).dt.year
        station_groups = df_clean['station_id'].values
        year_groups = df_clean['year'].values

        # 为GNNWR准备数据
        gnnwr_data = None
        if self.use_gnnwr:
            gnnwr_data = df_clean.copy()
            # 确保所有GNNWR需要的列都存在
            for col in self.gnnwr_x_columns + self.gnnwr_spatial_columns:
                if col not in gnnwr_data.columns:
                    gnnwr_data[col] = 0.0

        self.logger.info(f"✅ 数据预处理完成")
        self.logger.info(f"  样本数: {len(X)}, 特征数: {len(self.feature_columns)}")

        return X, y, station_groups, year_groups, gnnwr_data

    def _train_gnnwr_for_fold(self, train_data, val_data):
        """为单个折叠训练GNNWR模型并提取权重

        Args:
            train_data (pd.DataFrame): 训练数据
            val_data (pd.DataFrame): 验证数据

        Returns:
            tuple: (训练集权重矩阵, 验证集权重矩阵)
        """
        self.logger.debug("为当前折叠训练GNNWR模型...")

        try:
            # 确保所有需要的列都存在
            for col in self.gnnwr_x_columns + self.gnnwr_spatial_columns + self.gnnwr_y_column:
                if col not in train_data.columns:
                    train_data[col] = 0.0
                if col not in val_data.columns:
                    val_data[col] = 0.0

            # 初始化GNNWR数据集
            train_set, val_set, _ = datasets.init_dataset_split(
                train_data=train_data,
                val_data=val_data,
                test_data=val_data.head(1),  # 测试集用验证集头1行占位
                x_column=self.gnnwr_x_columns,
                y_column=self.gnnwr_y_column,
                spatial_column=self.gnnwr_spatial_columns,
                batch_size=128,
                shuffle=False,
                use_model="gnnwr"
            )

            # 训练GNNWR模型
            gnnwr = models.GNNWR(
                train_dataset=train_set,
                valid_dataset=val_set,
                test_dataset=train_set,  # 使用训练集作为测试集占位
                dense_layers=self.gnnwr_params['dense_layers'],
                activate_func=self.gnnwr_params['activate_func'],
                start_lr=self.gnnwr_params['start_lr'],
                optimizer=self.gnnwr_params['optimizer'],
                model_name=f"GNNWR_Fold",
                model_save_path="result/gnnwr_models_temp",
                log_path="result/gnnwr_logs_temp",
                write_path="result/gnnwr_runs_temp"
            )

            # 简短训练
            gnnwr.run(
                max_epoch=self.gnnwr_params['max_epoch'],
                early_stop=self.gnnwr_params['early_stop'],
                print_frequency=self.gnnwr_params['print_frequency']
            )

            # 提取权重矩阵
            def extract_weights(gnnwr_instance, dataset):
                model = gnnwr_instance._model
                model.eval()
                device = gnnwr_instance._device

                all_weights = []
                with torch.no_grad():
                    for batch in dataset.dataloader:
                        if len(batch) >= 2:
                            distances, features = batch[:2]
                            distances = distances.to(device)
                            weights = model(distances)
                            all_weights.append(weights.cpu().numpy())

                if all_weights:
                    return np.concatenate(all_weights, axis=0)
                return None

            train_weights = extract_weights(gnnwr, train_set)
            val_weights = extract_weights(gnnwr, val_set)

            if train_weights is not None and val_weights is not None:
                self.logger.debug(f"  提取到权重矩阵: 训练集{train_weights.shape}, 验证集{val_weights.shape}")
                return train_weights, val_weights
            else:
                self.logger.warning("  未能提取到权重矩阵")
                return None, None

        except Exception as e:
            self.logger.warning(f"  GNNWR训练失败: {str(e)}")
            return None, None

    def _apply_gnnwr_weights(self, X, weights, feature_columns, gnnwr_x_columns):
        """应用GNNWR权重到特征矩阵"""
        if weights is None:
            return X

        # 确保权重矩阵形状匹配
        if weights.shape[1] != len(gnnwr_x_columns):
            self.logger.warning(f"权重矩阵特征数({weights.shape[1]})与GNNWR特征数({len(gnnwr_x_columns)})不匹配")
            return X

        # 创建特征映射：特征列到GNNWR特征列的索引
        feature_to_gnnwr = {}
        for i, feat in enumerate(feature_columns):
            if feat in gnnwr_x_columns:
                feature_to_gnnwr[i] = gnnwr_x_columns.index(feat)

        # 🔍 验证：打印应用权重前的信息
        print("\n" + "=" * 80)
        print("🔍 GNNWR权重应用验证")
        print("=" * 80)

        # 1. 打印权重统计
        print(f"权重矩阵形状: {weights.shape}")
        print(f"权重统计:")
        print(f"  均值: {weights.mean():.6f}")
        print(f"  标准差: {weights.std():.6f}")
        print(f"  最小值: {weights.min():.6f}")
        print(f"  最大值: {weights.max():.6f}")
        print(f"  中位数: {np.median(weights):.6f}")

        # 2. 检查权重与1的距离
        distance_from_one = np.abs(weights - 1).mean()
        print(f"权重与1的平均距离: {distance_from_one:.6f}")

        # 3. 统计接近1的权重比例
        close_to_one = np.sum(np.abs(weights - 1) < 0.01) / weights.size
        print(f"与1差异小于0.01的权重比例: {close_to_one:.2%}")

        # 4. 打印匹配的特征信息
        print(f"\n特征匹配情况:")
        print(f"  总特征数: {len(feature_columns)}")
        print(f"  GNNWR特征数: {len(gnnwr_x_columns)}")
        print(f"  匹配的特征数: {len(feature_to_gnnwr)}")

        if len(feature_to_gnnwr) > 0:
            matched_features = [feature_columns[idx] for idx in list(feature_to_gnnwr.keys())[:5]]
            print(f"  前5个匹配特征: {matched_features}")

        # 5. 检查几个关键特征的变化
        key_features = ['elevation', 'X', 'Y', 'Z', 'slope', 'doy']
        print(f"\n关键特征验证 (前3个样本):")

        for feat in key_features:
            if feat in feature_columns and feat in gnnwr_x_columns:
                feat_idx = feature_columns.index(feat)
                gnnwr_idx = gnnwr_x_columns.index(feat)

                # 获取前3个样本
                print(f"\n{feat}:")
                for i in range(min(3, X.shape[0])):
                    original = X[i, feat_idx]
                    weight = weights[i, gnnwr_idx]
                    weighted = original * weight
                    change = weighted - original
                    rel_change = change / (abs(original) + 1e-10) * 100

                    print(f"  样本{i}: {original:.4f} × {weight:.4f} = {weighted:.4f} "
                          f"(变化: {change:+.4f}, 相对: {rel_change:+.2f}%)")

        # 保存原始X用于比较
        X_original = X.copy()

        # 应用权重（只对匹配的特征进行加权）
        X_weighted = X.copy()
        for feat_idx, gnnwr_idx in feature_to_gnnwr.items():
            X_weighted[:, feat_idx] = X[:, feat_idx] * weights[:, gnnwr_idx]

        # 🔍 验证：打印应用权重后的信息
        print(f"\n" + "=" * 60)
        print("权重应用结果统计")
        print("=" * 60)

        # 6. 计算总体变化
        changes = X_weighted - X_original
        abs_changes = np.abs(changes)

        print(f"总体变化统计:")
        print(f"  最大绝对变化: {abs_changes.max():.6f}")
        print(f"  平均绝对变化: {abs_changes.mean():.6f}")
        print(f"  变化 > 0.001 的比例: {(abs_changes > 0.001).sum() / abs_changes.size:.2%}")

        # 7. 按特征统计变化
        print(f"\n按特征变化统计 (前5个匹配特征):")

        if len(feature_to_gnnwr) > 0:
            # 获取前5个匹配特征
            feat_indices = list(feature_to_gnnwr.keys())[:5]

            for feat_idx in feat_indices:
                feat_name = feature_columns[feat_idx]
                feat_changes = changes[:, feat_idx]
                feat_abs_changes = abs_changes[:, feat_idx]

                print(f"\n{feat_name}:")
                print(f"  平均变化: {feat_changes.mean():.6f}")
                print(f"  平均绝对变化: {feat_abs_changes.mean():.6f}")
                print(f"  最大变化: {feat_changes.max():.6f}")
                print(f"  最小变化: {feat_changes.min():.6f}")

                # 检查是否所有变化都接近0
                if feat_abs_changes.mean() < 0.0001:
                    print(f"  ⚠️ 警告: 该特征几乎没有变化！")

        # 8. 验证是否真的改变了
        if np.allclose(X_weighted, X_original, atol=1e-10):
            print(f"\n❌ 严重警告: 加权后特征与原始特征几乎完全相同！")
            print(f"  最大差异: {np.abs(X_weighted - X_original).max():.10f}")
        else:
            print(f"\n✅ 加权成功: 特征已被修改")
            print(f"  差异范围: [{np.min(changes):.6f}, {np.max(changes):.6f}]")

        self.logger.debug(f"  应用权重: 匹配了{len(feature_to_gnnwr)}/{len(feature_columns)}个特征")
        return X_weighted

    def cross_validate(self, X, y, groups, cv_type='station', gnnwr_data=None):
        """执行带GNNWR权重的交叉验证

        Args:
            X (np.array): 特征数据
            y (np.array): 目标变量
            groups (np.array): 分组信息
            cv_type (str): 交叉验证类型 ('station' 或 'yearly')
            gnnwr_data (pd.DataFrame): GNNWR需要的完整数据

        Returns:
            dict: 交叉验证结果
        """
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

        self.logger.info(f"开始{cv_type}交叉验证，共{total_folds}个折叠...")
        self.logger.info(f"使用GNNWR权重增强: {self.use_gnnwr}")

        for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups)):
            group_id = groups[val_idx[0]]
            train_size = len(train_idx)
            val_size = len(val_idx)

            print("\n" + "=" * 100)
            print(f"🎯 {cv_type} Fold {fold + 1}/{total_folds}: {group_id}")
            print(f"   训练集大小: {train_size}, 验证集大小: {val_size}")
            print("=" * 100)

            self.logger.info(
                f"{cv_type} Fold {fold + 1}/{total_folds}: {group_id} (训练集{train_size}, 验证集{val_size})")

            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # GNNWR权重增强
            if self.use_gnnwr and gnnwr_data is not None:
                print(f"\n📊 GNNWR权重增强阶段")

                # 获取当前折叠的训练和验证数据
                train_data_fold = gnnwr_data.iloc[train_idx].copy()
                val_data_fold = gnnwr_data.iloc[val_idx].copy()

                print(f"  训练数据形状: {train_data_fold.shape}")
                print(f"  验证数据形状: {val_data_fold.shape}")

                # 训练GNNWR并提取权重
                print(f"  训练GNNWR模型...")
                train_weights, val_weights = self._train_gnnwr_for_fold(
                    train_data_fold,
                    val_data_fold
                )

                if train_weights is not None and val_weights is not None:
                    print(f"\n✅ GNNWR训练完成，提取到权重矩阵")
                    print(f"  训练集权重形状: {train_weights.shape}")
                    print(f"  验证集权重形状: {val_weights.shape}")

                    # 打印权重统计
                    print(f"  训练集权重统计:")
                    print(f"    均值: {train_weights.mean():.6f}")
                    print(f"    标准差: {train_weights.std():.6f}")
                    print(f"    范围: [{train_weights.min():.6f}, {train_weights.max():.6f}]")

                    # 🔍 详细验证：应用权重前后的特征变化
                    print(f"\n" + "=" * 80)
                    print(f"🧪 详细验证：权重应用效果")
                    print(f"=" * 80)

                    # 1. 保存原始特征
                    X_train_original = X_train.copy()
                    X_val_original = X_val.copy()

                    # 2. 应用权重前的特征统计
                    print(f"\n📈 应用权重前的特征统计:")
                    print(f"  训练集特征范围: [{X_train.min():.4f}, {X_train.max():.4f}]")
                    print(f"  训练集特征均值: {X_train.mean():.4f}")
                    print(f"  验证集特征范围: [{X_val.min():.4f}, {X_val.max():.4f}]")
                    print(f"  验证集特征均值: {X_val.mean():.4f}")

                    # 3. 应用权重
                    print(f"\n🔄 应用权重到特征矩阵...")
                    X_train = self._apply_gnnwr_weights(
                        X_train, train_weights,
                        self.feature_columns, self.gnnwr_x_columns
                    )
                    X_val = self._apply_gnnwr_weights(
                        X_val, val_weights,
                        self.feature_columns, self.gnnwr_x_columns
                    )

                    # 4. 应用权重后的特征统计
                    print(f"\n📊 应用权重后的特征统计:")
                    print(f"  训练集特征范围: [{X_train.min():.4f}, {X_train.max():.4f}]")
                    print(f"  训练集特征均值: {X_train.mean():.4f}")
                    print(f"  验证集特征范围: [{X_val.min():.4f}, {X_val.max():.4f}]")
                    print(f"  验证集特征均值: {X_val.mean():.4f}")

                    # 5. 计算变化量
                    train_changes = X_train - X_train_original
                    val_changes = X_val - X_val_original

                    print(f"\n📉 特征变化分析:")
                    print(f"  训练集变化:")
                    print(f"    最大变化: {train_changes.max():.6f}")
                    print(f"    最小变化: {train_changes.min():.6f}")
                    print(f"    平均绝对变化: {np.abs(train_changes).mean():.6f}")
                    print(f"    显著变化比例(>0.001): {(np.abs(train_changes) > 0.001).sum() / train_changes.size:.2%}")

                    print(f"  验证集变化:")
                    print(f"    最大变化: {val_changes.max():.6f}")
                    print(f"    最小变化: {val_changes.min():.6f}")
                    print(f"    平均绝对变化: {np.abs(val_changes).mean():.6f}")
                    print(f"    显著变化比例(>0.001): {(np.abs(val_changes) > 0.001).sum() / val_changes.size:.2%}")

                    # 6. 检查几个关键特征的变化
                    key_features = ['elevation', 'X', 'Y', 'Z', 'slope', 'doy']
                    print(f"\n🔑 关键特征详细变化 (第一个样本):")

                    for feat in key_features:
                        if feat in self.feature_columns and feat in self.gnnwr_x_columns:
                            feat_idx = self.feature_columns.index(feat)
                            gnnwr_idx = self.gnnwr_x_columns.index(feat)

                            # 训练集第一个样本
                            train_original = X_train_original[0, feat_idx]
                            train_weight = train_weights[0, gnnwr_idx]
                            train_weighted = X_train[0, feat_idx]

                            # 验证集第一个样本
                            val_original = X_val_original[0, feat_idx]
                            val_weight = val_weights[0, gnnwr_idx]
                            val_weighted = X_val[0, feat_idx]

                            print(f"\n  {feat}:")
                            print(f"    训练集: {train_original:.4f} × {train_weight:.4f} = {train_weighted:.4f} "
                                  f"(变化: {train_weighted - train_original:+.4f})")
                            print(f"    验证集: {val_original:.4f} × {val_weight:.4f} = {val_weighted:.4f} "
                                  f"(变化: {val_weighted - val_original:+.4f})")

                    # 7. 检查是否真的改变了
                    train_same = np.allclose(X_train, X_train_original, atol=1e-10)
                    val_same = np.allclose(X_val, X_val_original, atol=1e-10)

                    if train_same and val_same:
                        print(f"\n⚠️ 警告: 加权后特征与原始特征几乎完全相同！")
                        print(f"  训练集最大差异: {np.abs(X_train - X_train_original).max():.10f}")
                        print(f"  验证集最大差异: {np.abs(X_val - X_val_original).max():.10f}")
                    else:
                        print(f"\n✅ 验证通过: 权重成功应用到特征上")

                    self.logger.info(f"  ✅ GNNWR权重应用成功")
                else:
                    print(f"\n❌ GNNWR权重提取失败，使用原始特征")
                    self.logger.info(f"  ⚠️ GNNWR权重提取失败，使用原始特征")
            else:
                print(f"\n📝 未使用GNNWR权重增强")

            # 训练XGBoost模型
            print(f"\n🌲 训练XGBoost模型...")
            model = xgb.XGBRegressor(**self.params)

            # 添加训练进度显示
            print(f"  开始拟合模型 (样本数: {len(X_train)}, 特征数: {X_train.shape[1]})...")

            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            print(f"  模型训练完成，耗时: {training_time:.2f}秒")

            # 预测
            print(f"  进行预测...")
            y_pred = model.predict(X_val)

            # 存储结果
            all_predictions.extend(y_pred)
            all_true_values.extend(y_val)

            # 计算性能指标
            fold_metrics = self.evaluate_predictions(y_val, y_pred)
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
        overall_metrics = self.evaluate_predictions(
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
        print(f"📈 聚合性能指标:")
        print(f"  MAE:  {overall_metrics['MAE']:.3f} mm")
        print(f"  RMSE: {overall_metrics['RMSE']:.3f} mm")
        print(f"  R:    {overall_metrics['R']:.3f}")
        print(f"  总样本数: {overall_metrics['样本数']}")
        print(f"\n📊 折叠统计:")
        print(f"  折叠数: {total_folds}")
        print(f"  MAE均值: {mean_metrics['MAE']:.3f} ± {std_metrics['MAE']:.3f} mm")
        print(f"  R均值:   {mean_metrics['R']:.3f} ± {std_metrics['R']:.3f}")

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

    def evaluate_predictions(self, y_true, y_pred):
        """评估预测结果"""
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

    def train_final_model(self, X, y, gnnwr_data=None):
        """训练最终模型（使用全部数据）"""
        self.logger.info("训练最终XGBoost模型...")

        # GNNWR权重增强
        if self.use_gnnwr and gnnwr_data is not None:
            self.logger.info("为最终模型训练GNNWR...")

            # 使用全部数据训练GNNWR
            train_weights, _ = self._train_gnnwr_for_fold(gnnwr_data, gnnwr_data.head(1))

            if train_weights is not None:
                X = self._apply_gnnwr_weights(
                    X, train_weights,
                    self.feature_columns, self.gnnwr_x_columns
                )
                self.logger.info("✅ 最终模型GNNWR权重应用成功")

        # 训练XGBoost
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)

        self.logger.info("✅ 最终模型训练完成")
        return self.model

    def run_complete_analysis(self, df, output_dir=None):
        """运行完整分析流程 - 先进行年度交叉验证"""
        self.logger.info("=" * 70)
        self.logger.info("🚀 开始GNNW-XGBoost完整分析流程")
        self.logger.info("=" * 70)

        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"./gnnw_xgboost_results_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"输出目录: {output_dir}")

        try:
            # 1. 数据预处理
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 1: 数据预处理")
            self.logger.info("=" * 50)

            X, y, station_groups, year_groups, gnnwr_data = self.preprocess_data(df)

            results = {
                'preprocessing': {
                    'samples': len(X),
                    'features': len(self.feature_columns),
                    'stations': len(np.unique(station_groups)),
                    'years': len(np.unique(year_groups)),
                    'use_gnnwr': self.use_gnnwr
                }
            }

            # 2. 先进行年度交叉验证（数据量较小）
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 2: 年度交叉验证 (数据量较小，先开始)")
            self.logger.info("=" * 50)

            results['yearly_cv'] = self.cross_validate(
                X, y, year_groups, 'yearly', gnnwr_data
            )

            # 3. 再进行站点交叉验证（数据量较大）
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 3: 站点交叉验证 (数据量较大)")
            self.logger.info("=" * 50)

            # 对于站点交叉验证，我们可以使用简化的GNNWR训练（减少轮数）
            if self.use_gnnwr:
                self.logger.info("站点交叉验证使用简化的GNNWR训练（减少到3个epoch）")
                original_epochs = self.gnnwr_params.get('max_epoch', 5)
                self.gnnwr_params['max_epoch'] = 3  # 减少训练轮数

                results['station_cv'] = self.cross_validate(
                    X, y, station_groups, 'station', gnnwr_data
                )

                # 恢复原始设置
                self.gnnwr_params['max_epoch'] = original_epochs
            else:
                results['station_cv'] = self.cross_validate(
                    X, y, station_groups, 'station', gnnwr_data
                )

            # 4. 训练最终模型（使用全部数据）
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 4: 训练最终模型")
            self.logger.info("=" * 50)

            results['final_model'] = self.train_final_model(X, y, gnnwr_data)

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

            # 保存详细结果
            eval_results = {
                'training_info': {
                    'timestamp': datetime.now().isoformat(),
                    'feature_columns': self.feature_columns,
                    'gnnwr_x_columns': self.gnnwr_x_columns,
                    'use_gnnwr': self.use_gnnwr,
                    'total_samples': results.get('preprocessing', {}).get('samples', 0)
                },
                'model_parameters': self.params,
                'gnnwr_parameters': self.gnnwr_params,
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

    def _create_scatter_plots(self, results, output_dir):
        """创建散点图"""
        try:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # 站点CV散点图
            if 'station_cv' in results:
                overall = results['station_cv']['overall']
                self._plot_single_scatter(
                    ax1,
                    results['station_cv']['true_values'],
                    results['station_cv']['predictions'],
                    overall,
                    'Station Cross-Validation'
                )

            # 年度CV散点图
            if 'yearly_cv' in results:
                overall = results['yearly_cv']['overall']
                self._plot_single_scatter(
                    ax2,
                    results['yearly_cv']['true_values'],
                    results['yearly_cv']['predictions'],
                    overall,
                    'Yearly Cross-Validation'
                )

            plt.tight_layout()
            scatter_path = f'{output_dir}/scatter_plots.png'
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"✅ 散点图保存: {scatter_path}")

        except Exception as e:
            self.logger.warning(f"生成散点图失败: {str(e)}")

    def _plot_single_scatter(self, ax, y_true, y_pred, metrics, title):
        """绘制单个散点图"""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            return

        max_range = 175
        ax.plot([0, max_range], [0, max_range], 'k-', alpha=0.8, linewidth=2)
        ax.scatter(y_true_clean, y_pred_clean, alpha=0.6, s=15, c='blue', edgecolors='none')

        ax.set_xlabel('Observed SWE (mm)', fontsize=14)
        ax.set_ylabel('Predicted SWE (mm)', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlim([0, max_range])
        ax.set_ylim([0, max_range])
        ax.grid(True, alpha=0.3)

        stats_text = f"MAE = {metrics['MAE']:.2f} mm\nRMSE = {metrics['RMSE']:.2f} mm\nR = {metrics['R']:.3f}\nN = {len(y_true_clean)}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=13, fontfamily='monospace', weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        importance_scores = self.model.feature_importances_

        if len(importance_scores) != len(self.feature_columns):
            min_length = min(len(importance_scores), len(self.feature_columns))
            importance_scores = importance_scores[:min_length]
            feature_names = self.feature_columns[:min_length]
        else:
            feature_names = self.feature_columns

        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        return feature_importance_df

    def _generate_report(self, results):
        """生成分析报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("📊 GNNW-XGBoost模型分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"使用GNNWR权重增强: {self.use_gnnwr}")
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

        # 性能比较
        if 'station_cv' in results and 'yearly_cv' in results:
            station_r = results['station_cv']['overall']['R']
            yearly_r = results['yearly_cv']['overall']['R']

            report_lines.append("💡 性能分析:")
            if not np.isnan(station_r) and not np.isnan(yearly_r):
                if station_r > yearly_r:
                    report_lines.append(f"  站点CV优于年度CV (R: {station_r:.3f} > {yearly_r:.3f})")
                else:
                    report_lines.append(f"  年度CV优于站点CV (R: {yearly_r:.3f} > {station_r:.3f})")

        report_lines.append("\n" + "=" * 80)
        return "\n".join(report_lines)


# 便捷使用函数
def train_gnnw_xgboost_model(data_df, output_dir=None, use_gnnwr=True):
    """便捷函数：训练GNNW-XGBoost模型

    Args:
        data_df (pd.DataFrame): 包含特征和SWE的数据
        output_dir (str, optional): 输出目录路径
        use_gnnwr (bool): 是否使用GNNWR权重

    Returns:
        dict: 包含所有训练结果的字典
    """
    trainer = GNNW_XGBoostTrainer(use_gnnwr=use_gnnwr)
    return trainer.run_complete_analysis(data_df, output_dir)


# 对比实验函数
def compare_models(data_df, output_dir=None):
    """对比纯XGBoost和GNNW-XGBoost的性能"""

    print("=" * 80)
    print("🔬 开始模型对比实验")
    print("=" * 80)

    # 1. 纯XGBoost
    print("\n1. 训练纯XGBoost模型...")
    xgb_trainer = GNNW_XGBoostTrainer(use_gnnwr=False)
    xgb_results = xgb_trainer.run_complete_analysis(
        data_df,
        output_dir=os.path.join(output_dir, "xgboost_only") if output_dir else None
    )

    # 2. GNNW-XGBoost
    print("\n2. 训练GNNW-XGBoost模型...")
    gnnw_trainer = GNNW_XGBoostTrainer(use_gnnwr=True)
    gnnw_results = gnnw_trainer.run_complete_analysis(
        data_df,
        output_dir=os.path.join(output_dir, "gnnw_xgboost") if output_dir else None
    )

    # 3. 对比分析
    print("\n" + "=" * 80)
    print("📊 模型对比结果")
    print("=" * 80)

    if 'station_cv' in xgb_results and 'station_cv' in gnnw_results:
        xgb_station_r = xgb_results['station_cv']['overall']['R']
        gnnw_station_r = gnnw_results['station_cv']['overall']['R']

        print("站点交叉验证 (空间评估):")
        print(f"  纯XGBoost: R = {xgb_station_r:.3f}")
        print(f"  GNNW-XGBoost: R = {gnnw_station_r:.3f}")

        if not np.isnan(xgb_station_r) and not np.isnan(gnnw_station_r):
            improvement = (gnnw_station_r - xgb_station_r) / abs(xgb_station_r) * 100
            print(f"  GNNW-XGBoost提升: {improvement:+.1f}%")

    if 'yearly_cv' in xgb_results and 'yearly_cv' in gnnw_results:
        xgb_yearly_r = xgb_results['yearly_cv']['overall']['R']
        gnnw_yearly_r = gnnw_results['yearly_cv']['overall']['R']

        print("\n年度交叉验证 (时间评估):")
        print(f"  纯XGBoost: R = {xgb_yearly_r:.3f}")
        print(f"  GNNW-XGBoost: R = {gnnw_yearly_r:.3f}")

        if not np.isnan(xgb_yearly_r) and not np.isnan(gnnw_yearly_r):
            improvement = (gnnw_yearly_r - xgb_yearly_r) / abs(xgb_yearly_r) * 100
            print(f"  GNNW-XGBoost提升: {improvement:+.1f}%")

    return {
        'xgboost': xgb_results,
        'gnnw_xgboost': gnnw_results
    }