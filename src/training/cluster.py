import logging
import warnings

import torch.nn as nn
# import logger
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneGroupOut
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 禁用TF32相关警告（CPU上不需要）
warnings.filterwarnings("ignore", message=".*TF32.*")

# CPU性能优化设置
torch.set_num_threads(24)  # i9-14900KF有24个物理核心
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['OPENMP'] = '1'

# 禁用CUDA相关设置（避免不必要的GPU检查）
torch.backends.cudnn.enabled = False

# 设置矩阵乘法精度（CPU上使用高精度）
torch.set_float32_matmul_precision('high')

# 先创建logger
logger = logging.getLogger("SWEClusterEnsemble")

# 然后导入增强版GNNWR
try:
    from GNNWR import EnhancedSpatialDataset, EnhancedGNNWRTrainer, SpatialWeightCalculator

    HAS_ENHANCED_GNNWR = True
    logger.info("成功导入增强版GNNWR")
except ImportError as e:
    logger.warning(f"无法导入增强版GNNWR: {e}")
    try:
        # 尝试导入基础版
        from GNNWR import SpatialDataset, GNNWRTrainer

        HAS_ENHANCED_GNNWR = False
        logger.info("使用基础版GNNWR")
    except ImportError:
        logger.error("无法导入任何GNNWR版本")
        HAS_ENHANCED_GNNWR = False


        # 创建虚拟类以避免后续错误
        class EnhancedSpatialDataset:
            def __init__(self, features, targets, coords=None):
                self.features = features
                self.targets = targets
                self.coords = coords

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                if self.coords is not None:
                    return self.features[idx], self.targets[idx], self.coords[idx]
                else:
                    return self.features[idx], self.targets[idx]


        class EnhancedGNNWRTrainer:
            def __init__(self, *args, **kwargs):
                logger.warning("使用虚拟EnhancedGNNWRTrainer")

            def train(self, *args, **kwargs):
                logger.warning("虚拟训练方法")

            def predict(self, features, coords=None):
                logger.warning("虚拟预测方法")
                return np.random.normal(50, 20, len(features))


        class SpatialDataset:
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return self.features[idx], self.targets[idx]


        class GNNWRTrainer:
            def __init__(self, *args, **kwargs):
                logger.warning("使用虚拟GNNWRTrainer")

            def train(self, *args, **kwargs):
                logger.warning("虚拟训练方法")

            def predict(self, features):
                logger.warning("虚拟预测方法")
                return np.random.normal(50, 20, len(features))


class SWEClusterEnsemble:
    """SWE聚类集成回归器 - 使用增强版GNNWR进行集成"""



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

    def __init__(self, n_clusters=4, params=None, gnnwr_params=None, use_enhanced_gnnwr=True, use_rf=False, device='auto'):
        """初始化聚类集成回归器

        Args:
            n_clusters (int): 聚类数量
            params (dict): XGBoost参数
            gnnwr_params (dict): GNNWR参数
            use_enhanced_gnnwr (bool): 是否使用增强版GNNWR
        """
        self.logger = logging.getLogger("SWEClusterEnsemble")

        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_assignments = None
        self.cluster_models = {}
        self.gnnwr_trainer = None
        self.feature_columns = None
        self.target_column = 'swe'
        self.use_enhanced_gnnwr = use_enhanced_gnnwr and HAS_ENHANCED_GNNWR
        self.device = device
        self.use_rf = use_rf

        # 关键修复：确保params不为None
        if params is None:
            params = {}  # 确保params至少是空字典

        if use_rf:
            # RF参数
            self.rf_params = {
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', None),
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1
            }
            self.params = params if params else self.DEFAULT_PARAMS.copy()
        else:
            # 原有的XGB参数
            self.params = self.DEFAULT_PARAMS.copy()
            if params:
                self.params.update(params)


        # GNNWR参数
        self.gnnwr_params = {
            'hidden_dims': [128, 64, 32, 16],
            'learning_rate': 0.001,
            'epochs': 200,
            'batch_size': 64,
            'patience': 20,
            'bandwidth':5.0,
            'use_spatial_weights': True,
            'device': device,  # 传递设备参数
            'dropout_rate': 0.3,  # 添加dropout
            'weight_decay': 1e-4,  # 权重衰减
            'num_workers': min(6, os.cpu_count() // 2)
        }
        if gnnwr_params:
            self.gnnwr_params.update(gnnwr_params)

        self.logger.info(f"初始化SWE聚类集成回归器，聚类数: {n_clusters}")
        self.logger.info(f"使用{'增强版' if self.use_enhanced_gnnwr else '基础版'}GNNWR")
        self.logger.info(f"GNNWR参数: {self.gnnwr_params}")

    def preprocess_data(self, df):
        """数据预处理 - 完整调试版本"""
        self.logger.info("开始数据预处理...")

        # 确定特征列和目标列
        if self.feature_columns is None:
            # 自动选择特征列（排除目标列和其他非特征列）
            exclude_cols = [self.target_column, 'station_id', 'year', 'date', 'station', 'group',
                            'longitude', 'latitude', 'lon', 'lat']  # 排除坐标列
            self.feature_columns = [col for col in df.columns if
                                    col not in exclude_cols and df[col].dtype in [np.int64, np.float64]]

        self.logger.info(f"使用特征: {self.feature_columns}")

        # 提取特征和目标
        X = df[self.feature_columns].values
        y = df[self.target_column].values

        # 处理缺失值
        if np.isnan(X).any():
            self.logger.info("处理特征中的缺失值")
            self.feature_imputer = SimpleImputer(strategy='median')
            X = self.feature_imputer.fit_transform(X)
        else:
            self.feature_imputer = None

        # 创建分组信息
        if 'station_id' in df.columns:
            station_groups = df['station_id'].values
        elif 'station' in df.columns:
            station_groups = df['station'].values
        else:
            # 如果没有站点信息，使用索引作为分组
            station_groups = np.arange(len(df))
            self.logger.warning("未找到站点信息，使用索引作为分组")

        if 'year' in df.columns:
            year_groups = df['year'].values
        else:
            # 如果没有年份信息，创建虚拟年份
            year_groups = np.ones(len(df), dtype=int)
            self.logger.warning("未找到年份信息，使用统一年份分组")

        # === 坐标调试部分 ===
        self.logger.info("=== 坐标调试信息 ===")

        # 检查坐标列的存在和内容
        coord_columns = ['longitude', 'latitude', 'lon', 'lat']
        available_coords = [col for col in coord_columns if col in df.columns]
        self.logger.info(f"找到的坐标列: {available_coords}")

        for col in available_coords:
            if col in df.columns:
                non_na_count = df[col].notna().sum()
                dtype = df[col].dtype
                min_val = df[col].min() if non_na_count > 0 else "N/A"
                max_val = df[col].max() if non_na_count > 0 else "N/A"
                self.logger.info(f"  {col}: 非空值={non_na_count}, 类型={dtype}, 范围=[{min_val}, {max_val}]")

        # 提取坐标信息
        coords = None
        if all(col in df.columns for col in ['longitude', 'latitude']):
            coords = df[['longitude', 'latitude']].values
            self.logger.info(f"✅ 使用经纬度坐标: {len(coords)} 个点")
            self.logger.info(
                f"   坐标范围: lon[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], lat[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")

            # 检查是否有NaN坐标
            nan_coords = np.isnan(coords).any(axis=1).sum()
            if nan_coords > 0:
                self.logger.warning(f"⚠️  发现 {nan_coords} 个坐标包含NaN值")
                # 使用均值填充NaN坐标
                for i in range(coords.shape[1]):
                    col_mean = np.nanmean(coords[:, i])
                    nan_mask = np.isnan(coords[:, i])
                    coords[nan_mask, i] = col_mean
                    self.logger.info(f"   列 {i} 的NaN值已用均值 {col_mean:.4f} 填充")

        elif all(col in df.columns for col in ['lon', 'lat']):
            coords = df[['lon', 'lat']].values
            self.logger.info(f"✅ 使用经纬度坐标: {len(coords)} 个点")
        else:
            self.logger.warning("❌ 未找到坐标信息，将使用虚拟坐标")
            unique_stations = np.unique(station_groups)
            station_to_coord = {station: [i, i] for i, station in enumerate(unique_stations)}
            coords = np.array([station_to_coord[station] for station in station_groups])
            self.logger.info(f"   生成虚拟坐标: {len(coords)} 个点")

        self.logger.info(f"数据预处理完成: {len(X)}个样本, {X.shape[1]}个特征")
        self.logger.info(f"站点数: {len(np.unique(station_groups))}, 年份数: {len(np.unique(year_groups))}")
        self.logger.info(f"坐标最终状态: {'可用' if coords is not None else '不可用'}")

        return X, y, station_groups, year_groups, coords

    def perform_clustering(self, X, groups):
        """执行聚类分析

        Args:
            X (np.array): 特征数据
            groups (np.array): 分组信息

        Returns:
            np.array: 聚类标签
        """
        self.logger.info(f"执行K-means聚类，聚类数: {self.n_clusters}")

        # 按站点聚合特征
        unique_groups = np.unique(groups)
        group_features = []

        for group in unique_groups:
            group_mask = groups == group
            group_data = X[group_mask]
            # 使用每个站点的特征均值作为聚类特征
            group_mean = np.nanmean(group_data, axis=0)
            group_features.append(group_mean)

        group_features = np.array(group_features)

        # 处理可能的NaN值
        if np.isnan(group_features).any():
            self.logger.info("处理聚类特征中的缺失值")
            cluster_imputer = SimpleImputer(strategy='median')
            group_features = cluster_imputer.fit_transform(group_features)

        # 执行K-means聚类
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        group_clusters = self.kmeans.fit_predict(group_features)

        # 将聚类标签映射回原始样本
        cluster_assignments = np.zeros(len(X), dtype=int)
        for i, group in enumerate(unique_groups):
            group_mask = groups == group
            cluster_assignments[group_mask] = group_clusters[i]

        # 统计每个聚类的样本数
        cluster_counts = np.bincount(cluster_assignments)
        self.logger.info(f"聚类分布: {dict(enumerate(cluster_counts))}")

        return cluster_assignments

    def train_cluster_models(self, X, y, cluster_labels):
        """为每个聚类训练XGBoost模型

        Args:
            X (np.array): 特征数据
            y (np.array): 目标变量
            cluster_labels (np.array): 聚类标签
        """
        self.logger.info("训练各聚类XGBoost模型...")
        self.cluster_models = {}

        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)

            if cluster_size < 5:
                self.logger.warning(f"聚类 {cluster_id} 样本数过少 ({cluster_size})，跳过训练")
                continue

            X_cluster = X[cluster_mask]
            y_cluster = y[cluster_mask]

            if self.use_rf:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(**self.rf_params)
            else:
                import xgboost as xgb
                model = xgb.XGBRegressor(**self.params)

            model.fit(X_cluster, y_cluster)

            self.cluster_models[cluster_id] = model

            # 评估聚类模型性能
            y_pred_cluster = model.predict(X_cluster)
            cluster_mae = mean_absolute_error(y_cluster, y_pred_cluster)
            cluster_rmse = np.sqrt(mean_squared_error(y_cluster, y_pred_cluster))

            self.logger.info(f"  聚类 {cluster_id}: {cluster_size}样本, MAE={cluster_mae:.3f}, RMSE={cluster_rmse:.3f}")

    def _get_cluster_predictions(self, X, cluster_labels):
        """获取各聚类模型的预测结果

        Args:
            X (np.array): 特征数据
            cluster_labels (np.array): 聚类标签

        Returns:
            np.array: 各聚类模型的预测结果矩阵
        """
        cluster_predictions = np.zeros((len(X), self.n_clusters))

        for cluster_id, model in self.cluster_models.items():
            cluster_mask = cluster_labels == cluster_id
            if np.any(cluster_mask):
                predictions = model.predict(X[cluster_mask])
                cluster_predictions[cluster_mask, cluster_id] = predictions

        return cluster_predictions

    def train_gnnwr_model(self, X, y, cluster_predictions, coords=None):
        """训练GNNWR集成模型 - 内存优化版本"""
        self.logger.info("=== train_gnnwr_method详细调试 ===")
        self.logger.info(f"输入参数ID检查:")
        self.logger.info(f"  coords id: {id(coords)}")
        self.logger.info(f"  coords is None: {coords is None}")

        # 立即检查坐标数据
        if coords is None:
            self.logger.error("❌ 坐标数据在方法入口处就为None!")
            raise ValueError("坐标数据在方法入口处就为None")

        self.logger.info(f"  coords类型: {type(coords)}")
        self.logger.info(f"  coords形状: {coords.shape if hasattr(coords, 'shape') else 'No shape'}")
        self.logger.info(f"  coords长度: {len(coords) if hasattr(coords, '__len__') else 'No length'}")

        self.logger.info("训练GNNWR集成模型...")

        # 使用特征：原始特征 + 各聚类预测
        gnnwr_features = np.hstack([X, cluster_predictions])

        # 添加维度调试
        self.logger.info(f"输入特征维度调试:")
        self.logger.info(f"  X形状: {X.shape}")
        self.logger.info(f"  cluster_predictions形状: {cluster_predictions.shape}")
        self.logger.info(f"  合并后gnnwr_features形状: {gnnwr_features.shape}")

        # 关键修复：创建坐标数据的副本，避免被其他方法修改
        if coords is not None:
            coords_copy = coords.copy()  # 创建副本
            self.logger.info(f"创建坐标副本，原id: {id(coords)}, 副本id: {id(coords_copy)}")
        else:
            coords_copy = None
            self.logger.error("坐标数据为None")
            raise ValueError("坐标数据为None")

        # 处理缺失值
        if np.isnan(gnnwr_features).any():
            self.logger.info("处理GNNWR特征中的缺失值")
            self.gnnwr_imputer = SimpleImputer(strategy='median')
            gnnwr_features_imputed = self.gnnwr_imputer.fit_transform(gnnwr_features)
        else:
            gnnwr_features_imputed = gnnwr_features
            self.gnnwr_imputer = None

        # 添加处理后的维度调试
        self.logger.info(f"处理后特征维度: {gnnwr_features_imputed.shape}")

        # 根据数据大小自动调整参数（统一设置）
        n_samples = len(gnnwr_features_imputed)
        batch_size = min(128, max(32, n_samples // 100))  # 自适应批次大小
        num_workers = min(6, os.cpu_count() // 2)  # 使用一半CPU核心

        self.logger.info(f"数据加载器配置: batch_size={batch_size}, workers={num_workers}")

        if self.use_enhanced_gnnwr:
            # 使用增强版GNNWR
            self.logger.info("使用增强版GNNWR训练器")

            # 检查样本数量，如果太多则使用简化模式
            # 关键修复：使用 coords_copy 而不是 coords
            use_spatial = self.gnnwr_params['use_spatial_weights'] and coords_copy is not None

            if not use_spatial:
                self.logger.warning(f"样本数量较大 ({n_samples}) 或坐标不可用，禁用空间权重计算")
                # 即使禁用空间权重，也要传递坐标数据
                dataset = EnhancedSpatialDataset(
                    features=gnnwr_features_imputed,
                    targets=y,
                    coords=coords_copy  # 仍然传递坐标，只是训练器不使用
                )
            else:
                # 正常模式
                dataset = EnhancedSpatialDataset(
                    features=gnnwr_features_imputed,
                    targets=y,
                    coords=coords_copy
                )

            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,  # 修复：使用自适应批次大小
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,  # 如果使用GPU则启用
                persistent_workers=num_workers > 0
            )

            # 初始化增强版GNNWR训练器
            input_dim = gnnwr_features_imputed.shape[1]
            self.logger.info(f"初始化GNNWR训练器，输入维度: {input_dim}")

            self.gnnwr_trainer = EnhancedGNNWRTrainer(
                input_dim=input_dim,
                coords=coords_copy if use_spatial else None,  # 关键修复：使用副本
                hidden_dims=self.gnnwr_params['hidden_dims'],
                learning_rate=self.gnnwr_params['learning_rate'],
                bandwidth=self.gnnwr_params['bandwidth'],
                use_spatial_weights=use_spatial
            )

            # 训练模型
            self.logger.info(f"开始增强版GNNWR训练，输入维度: {input_dim}")
            try:
                self.gnnwr_trainer.train(
                    train_loader,
                    epochs=self.gnnwr_params['epochs'],
                    patience=self.gnnwr_params['patience']
                )
            except MemoryError as e:
                self.logger.error(f"内存不足: {e}，回退到基础版GNNWR")
                self.use_enhanced_gnnwr = False
                self.train_gnnwr_model(X, y, cluster_predictions, coords)
                return
        else:
            # 使用基础版GNNWR（无空间权重，内存友好）
            self.logger.info("使用基础版GNNWR训练器")

            # 创建数据集
            dataset = SpatialDataset(gnnwr_features_imputed, y)

            # 修复：基础版也使用优化配置
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,  # 使用自适应批次大小
                shuffle=True,
                num_workers=num_workers,  # 添加多线程支持
                pin_memory=False,
                persistent_workers=num_workers > 0
            )

            # 初始化基础版GNNWR训练器
            input_dim = gnnwr_features_imputed.shape[1]
            self.gnnwr_trainer = GNNWRTrainer(
                input_dim=input_dim,
                hidden_dims=self.gnnwr_params['hidden_dims'],
                learning_rate=self.gnnwr_params['learning_rate']
            )

            # 训练模型
            self.logger.info(f"开始基础版GNNWR训练，输入维度: {input_dim}")
            self.gnnwr_trainer.train(
                train_loader,
                epochs=self.gnnwr_params['epochs'],
                patience=self.gnnwr_params['patience']
            )

        # 计算训练集性能
        # 关键修复：使用 coords_copy 而不是 coords
        y_pred = self.predict_with_gnnwr(gnnwr_features_imputed, None, coords_copy)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r_value, _ = pearsonr(y, y_pred)

        self.logger.info(f"GNNWR模型训练完成: MAE={mae:.3f}, RMSE={rmse:.3f}, R={r_value:.3f}")

    def predict_with_gnnwr(self, X, cluster_predictions=None, coords=None):
        """使用GNNWR进行预测 - 修复版本"""
        if self.gnnwr_trainer is None:
            raise ValueError("GNNWR模型尚未训练")

        self.logger.info(f"预测时特征维度调试:")
        self.logger.info(f"  X形状: {X.shape}")

        # 关键修复：如果传入了cluster_predictions，说明X已经是原始特征
        # 需要重新合并特征，但要确保维度一致
        if cluster_predictions is not None:
            self.logger.info(f"  需要合并cluster_predictions: {cluster_predictions.shape}")

            # 检查X的维度是否已经包含了聚类预测
            expected_original_dim = X.shape[1] - self.n_clusters
            if X.shape[1] == expected_original_dim + self.n_clusters:
                # X已经包含了聚类预测，直接使用
                gnnwr_features = X
                self.logger.info(f"  X已经包含聚类预测，直接使用")
            else:
                # 需要合并
                gnnwr_features = np.hstack([X, cluster_predictions])
                self.logger.info(f"  合并后特征维度: {gnnwr_features.shape}")
        else:
            # 如果cluster_predictions为None，说明X已经是合并后的特征
            self.logger.info(f"  X已经是合并后的特征")
            gnnwr_features = X

        # 处理缺失值
        if self.gnnwr_imputer is not None:
            gnnwr_features_imputed = self.gnnwr_imputer.transform(gnnwr_features)
        else:
            gnnwr_features_imputed = gnnwr_features

        # 维度验证
        expected_dim = self.gnnwr_trainer.model.feature_network[0].in_features
        actual_dim = gnnwr_features_imputed.shape[1]

        if actual_dim != expected_dim:
            self.logger.error(f"维度不匹配: 输入特征{actual_dim}维, 模型期望{expected_dim}维")
            raise ValueError(f"特征维度不匹配: 输入{actual_dim} vs 模型{expected_dim}")

        # 预测
        if self.use_enhanced_gnnwr:
            return self.gnnwr_trainer.predict(gnnwr_features_imputed, coords)
        else:
            return self.gnnwr_trainer.predict(gnnwr_features_imputed)

    def validate_feature_dimensions(self, features, stage="training"):
        """验证特征维度一致性"""
        if self.gnnwr_trainer is None:
            return True

        # 获取模型期望的输入维度
        if hasattr(self.gnnwr_trainer.model, 'feature_network'):
            expected_dim = self.gnnwr_trainer.model.feature_network[0].in_features
            actual_dim = features.shape[1]

            if actual_dim != expected_dim:
                self.logger.error(f"{stage}阶段维度不匹配: 实际{actual_dim}维, 期望{expected_dim}维")
                return False

        return True

    def evaluate_predictions(self, y_true, y_pred):
        """评估预测性能

        Args:
            y_true (np.array): 真实值
            y_pred (np.array): 预测值

        Returns:
            dict: 评估指标
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r_value, p_value = pearsonr(y_true, y_pred)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R': r_value,
            'R_squared': r_value ** 2,
            'samples': len(y_true)
        }

    def cross_validate(self, X, y, groups, coords=None, cv_type='station'):
        """执行交叉验证 - 详细坐标调试"""
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()

        all_predictions = []
        all_true_values = []
        fold_results = {}

        unique_groups = np.unique(groups)
        total_folds = len(unique_groups)

        self.logger.info(f"开始{cv_type}交叉验证，共{total_folds}个折叠...")
        self.logger.info(f"初始坐标状态: {'可用' if coords is not None else '不可用'}")

        # 在整个数据集上按站点进行一次聚类
        self.logger.info("在整个数据集上按站点进行聚类分配...")
        self.cluster_assignments = self.perform_clustering(X, groups)

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            group_id = groups[test_idx[0]]
            test_size = len(test_idx)
            train_size = len(train_idx)

            self.logger.info(f"=== Fold {fold + 1} 详细调试 ===")
            self.logger.info(f"训练集大小: {train_size}, 测试集大小: {test_size}")

            # 分割数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]

            # 分割坐标 - 详细检查
            if coords is not None:
                coords_train = coords[train_idx]
                coords_test = coords[test_idx]

                self.logger.info(f"坐标分割结果:")
                self.logger.info(f"  coords_train类型: {type(coords_train)}")
                self.logger.info(f"  coords_train形状: {coords_train.shape}")
                self.logger.info(f"  coords_test类型: {type(coords_test)}")
                self.logger.info(f"  coords_test形状: {coords_test.shape}")

                # 检查是否有空数组
                if len(coords_train) == 0:
                    self.logger.error(f"⚠️  Fold {fold + 1}: coords_train为空数组!")
                if len(coords_test) == 0:
                    self.logger.error(f"⚠️  Fold {fold + 1}: coords_test为空数组!")

            else:
                self.logger.error(f"❌ Fold {fold + 1}: 初始coords为None!")
                coords_train = None
                coords_test = None

            # 使用固定的聚类分配
            train_cluster_labels = self.cluster_assignments[train_idx]
            test_cluster_labels = self.cluster_assignments[test_idx]

            # 训练聚类集成模型
            try:
                # 第一步：为每个聚类训练模型
                self.train_cluster_models(X_train, y_train, train_cluster_labels)

                # 第二步：获取训练集上的聚类预测
                cluster_predictions_train = self._get_cluster_predictions(X_train, train_cluster_labels)

                # 第三步：训练GNNWR集成模型 - 添加前置检查
                if coords_train is None:
                    raise ValueError(f"Fold {fold + 1}: coords_train为None，无法训练GNNWR")
                if len(coords_train) == 0:
                    raise ValueError(f"Fold {fold + 1}: coords_train为空数组，无法训练GNNWR")

                self.train_gnnwr_model(X_train, y_train, cluster_predictions_train, coords_train)

                # 第四步：预测测试集 - 关键修复
                cluster_predictions_test = self._get_cluster_predictions(X_test, test_cluster_labels)

                # 关键修复：测试集特征也需要与聚类预测合并
                test_features_combined = np.hstack([X_test, cluster_predictions_test])
                self.logger.info(f"测试集合并特征形状: {test_features_combined.shape}")

                y_pred = self.predict_with_gnnwr(test_features_combined, None, coords_test)  # 第二个参数传None

                # 存储结果
                all_predictions.extend(y_pred)
                all_true_values.extend(y_test)

                # 计算当前折叠性能
                fold_metrics = self.evaluate_predictions(y_test, y_pred)
                fold_results[group_id] = fold_metrics

                self.logger.info(
                    f"  {cv_type} Fold {fold + 1}/{total_folds}: {group_id} "
                    f"(聚类{test_cluster_labels[0]}, {test_size}样本) - "
                    f"MAE={fold_metrics['MAE']:.3f}, R={fold_metrics['R']:.3f}"
                )

            except Exception as e:
                self.logger.error(f"折叠 {fold + 1} 训练失败: {e}")
                import traceback
                self.logger.error(f"详细错误信息: {traceback.format_exc()}")
                continue

        # 计算总体性能
        overall_metrics = self.evaluate_predictions(
            np.array(all_true_values),
            np.array(all_predictions)
        )

        self.logger.info(f"✅ {cv_type}交叉验证完成")
        self.logger.info(f"  聚合性能: MAE={overall_metrics['MAE']:.3f}mm, R={overall_metrics['R']:.3f}")

        return {
            'overall': overall_metrics,
            'by_fold': fold_results,
            'predictions': np.array(all_predictions),
            'true_values': np.array(all_true_values),
            'folds': total_folds,
            'cluster_assignments': self.cluster_assignments
        }

    def run_complete_analysis(self, df, output_dir=None):
        """运行完整分析流程

        Args:
            df (pd.DataFrame): 输入数据
            output_dir (str, optional): 输出目录路径

        Returns:
            dict: 分析结果
        """
        self.logger.info("=" * 70)
        self.logger.info("🚀 开始SWE聚类集成回归完整分析流程")
        self.logger.info("=" * 70)

        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"./swe_cluster_ensemble_results_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"输出目录: {output_dir}")

        try:
            # 1. 数据预处理
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 1: 数据预处理")
            self.logger.info("=" * 50)

            X, y, station_groups, year_groups, coords = self.preprocess_data(df)

            results = {
                'preprocessing': {
                    'samples': len(X),
                    'features': len(self.feature_columns),
                    'stations': len(np.unique(station_groups)),
                    'years': len(np.unique(year_groups)),
                    'n_clusters': self.n_clusters,
                    'has_coords': coords is not None
                }
            }

            # 2. 在整个数据集上按站点进行聚类
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 2: 站点级聚类分析")
            self.logger.info("=" * 50)

            self.cluster_assignments = self.perform_clustering(X, station_groups)
            results['cluster_assignments'] = self.cluster_assignments

            # # 3. 站点交叉验证（使用固定聚类）
            # self.logger.info("\n" + "=" * 50)
            # self.logger.info("步骤 3: 站点交叉验证")
            # self.logger.info("=" * 50)
            #
            # results['station_cv'] = self.cross_validate(X, y, station_groups, coords, 'station')

            # 4. 年度交叉验证（使用固定聚类）
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 4: 年度交叉验证")
            self.logger.info("=" * 50)

            results['yearly_cv'] = self.cross_validate(X, y, year_groups, coords, 'yearly')

            # 5. 训练最终模型
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 5: 训练最终模型")
            self.logger.info("=" * 50)

            self.fit(X, y, station_groups, coords)

            results['final_model'] = {
                'kmeans': self.kmeans,
                'cluster_models': self.cluster_models,
                'gnnwr_trainer': self.gnnwr_trainer,
                'cluster_assignments': self.cluster_assignments,
                'feature_columns': self.feature_columns
            }

            # 6. 保存结果
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 6: 保存结果")
            self.logger.info("=" * 50)

            self._save_results(results, output_dir)

            # 7. 生成报告
            report = self._generate_report(results)
            print(report)
            self.logger.info("🎯 聚类集成分析完成！")
            return results

        except Exception as e:
            self.logger.error(f"❌ 分析流程失败: {str(e)}")
            raise

    def fit(self, X, y, station_groups, coords=None):
        """在整个数据集上训练模型

        Args:
            X (np.array): 特征数据
            y (np.array): 目标变量
            station_groups (np.array): 站点分组信息
            coords (np.array): 坐标数据
        """
        self.logger.info("在整个数据集上训练聚类集成模型...")

        # 第一步：在整个数据集上按站点进行聚类
        self.logger.info(f"在整个数据集上按站点进行K-means聚类，聚类数: {self.n_clusters}")
        self.cluster_assignments = self.perform_clustering(X, station_groups)

        # 第二步：为每个聚类训练模型
        self.train_cluster_models(X, y, self.cluster_assignments)

        # 第三步：训练GNNWR集成模型
        cluster_predictions = self._get_cluster_predictions(X, self.cluster_assignments)
        self.train_gnnwr_model(X, y, cluster_predictions, coords)

        self.logger.info("✅ 聚类集成模型训练完成")

    def predict(self, X, coords=None):
        """预测新样本

        Args:
            X (np.array): 特征数据
            coords (np.array): 坐标数据

        Returns:
            np.array: 预测结果
        """
        if self.kmeans is None or not self.cluster_models or self.gnnwr_trainer is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        # 第一步：聚类
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
        else:
            X_imputed = X

        cluster_labels = self.kmeans.predict(X_imputed)

        # 第二步：各聚类模型预测
        cluster_predictions = np.zeros((len(X), self.n_clusters))

        for cluster_id, model in self.cluster_models.items():
            cluster_mask = cluster_labels == cluster_id
            if np.any(cluster_mask):
                cluster_predictions[cluster_mask, cluster_id] = model.predict(X[cluster_mask])

        # 第三步：GNNWR集成预测
        return self.predict_with_gnnwr(X, cluster_predictions, coords)

    def _save_results(self, results, output_dir):
        """保存结果到文件

        Args:
            results (dict): 分析结果
            output_dir (str): 输出目录
        """
        self.logger.info("保存分析结果...")

        # 保存模型
        model_path = os.path.join(output_dir, 'swe_cluster_ensemble_model.pkl')
        joblib.dump({
            'kmeans': self.kmeans,
            'cluster_models': self.cluster_models,
            'gnnwr_trainer': self.gnnwr_trainer,
            'feature_columns': self.feature_columns,
            'params': self.params,
            'gnnwr_params': self.gnnwr_params,
            'n_clusters': self.n_clusters,
            'use_enhanced_gnnwr': self.use_enhanced_gnnwr
        }, model_path)

        # 保存结果数据
        results_path = os.path.join(output_dir, 'analysis_results.pkl')
        joblib.dump(results, results_path)

        # 保存文本报告
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_report(results))

        # 生成可视化图表
        self._create_visualizations(results, output_dir)

        self.logger.info(f"结果已保存到: {output_dir}")

    def _generate_report(self, results):
        """生成分析报告 - 修复版本：处理缺失的station_cv"""
        report = []
        report.append("=" * 70)
        report.append("❄️ SWE聚类集成回归分析报告")
        report.append("=" * 70)
        report.append("")

        # 数据概况
        preprocessing = results['preprocessing']
        report.append("📊 数据概况:")
        report.append(f"  样本数量: {preprocessing['samples']}")
        report.append(f"  特征数量: {preprocessing['features']}")
        report.append(f"  站点数量: {preprocessing['stations']}")
        report.append(f"  年份数量: {preprocessing['years']}")
        report.append(f"  聚类数量: {preprocessing['n_clusters']}")
        report.append(f"  使用坐标: {'是' if preprocessing['has_coords'] else '否'}")
        report.append(f"  GNNWR版本: {'增强版' if self.use_enhanced_gnnwr else '基础版'}")
        report.append("")

        # 站点交叉验证结果（如果存在）
        if 'station_cv' in results:
            station_cv = results['station_cv']
            station_overall = station_cv['overall']
            report.append("🏔️ 站点交叉验证结果:")
            report.append(f"  折叠数量: {station_cv['folds']}")
            report.append(f"  MAE: {station_overall['MAE']:.3f} mm")
            report.append(f"  RMSE: {station_overall['RMSE']:.3f} mm")
            report.append(f"  R: {station_overall['R']:.3f}")
            report.append(f"  R²: {station_overall['R_squared']:.3f}")
            report.append("")
        else:
            report.append("🏔️ 站点交叉验证: 已跳过")
            report.append("")

        # 年度交叉验证结果
        yearly_cv = results['yearly_cv']
        yearly_overall = yearly_cv['overall']
        report.append("📅 年度交叉验证结果:")
        report.append(f"  折叠数量: {yearly_cv['folds']}")
        report.append(f"  MAE: {yearly_overall['MAE']:.3f} mm")
        report.append(f"  RMSE: {yearly_overall['RMSE']:.3f} mm")
        report.append(f"  R: {yearly_overall['R']:.3f}")
        report.append(f"  R²: {yearly_overall['R_squared']:.3f}")
        report.append("")

        # 聚类分布
        cluster_counts = np.bincount(results['cluster_assignments'])
        report.append("🔍 聚类分布:")
        for cluster_id, count in enumerate(cluster_counts):
            report.append(
                f"  聚类 {cluster_id}: {count} 个样本 ({count / len(results['cluster_assignments']) * 100:.1f}%)")
        report.append("")

        report.append("🎯 模型配置:")
        report.append(f"  基础模型: {'随机森林' if self.use_rf else 'XGBoost'}")
        if hasattr(self, 'params') and self.params:
            report.append(f"  模型参数: {self.params}")
        report.append(f"  GNNWR参数: {self.gnnwr_params}")

        return "\n".join(report)

    def _create_visualizations(self, results, output_dir):
        """创建可视化图表 - 使用英文标签"""
        self.logger.info("Generating visualizations...")

        try:
            # 检查必要的键是否存在
            if 'yearly_cv' not in results:
                self.logger.warning("Missing yearly CV results, skipping visualization")
                return

            plt.figure(figsize=(12, 10))

            # 1. 年度交叉验证散点图
            plt.subplot(2, 2, 1)
            yearly_cv = results['yearly_cv']
            y_true_yearly = yearly_cv['true_values']
            y_pred_yearly = yearly_cv['predictions']

            plt.scatter(y_true_yearly, y_pred_yearly, alpha=0.6, s=20, color='orange')
            plt.plot([y_true_yearly.min(), y_true_yearly.max()],
                     [y_true_yearly.min(), y_true_yearly.max()], 'r--', alpha=0.8)
            plt.xlabel('True SWE (mm)')
            plt.ylabel('Predicted SWE (mm)')
            plt.title(
                f'Yearly Cross-Validation\nMAE={yearly_cv["overall"]["MAE"]:.2f}, R={yearly_cv["overall"]["R"]:.3f}')
            plt.grid(True, alpha=0.3)

            # 2. 残差分布图
            plt.subplot(2, 2, 2)
            residuals = y_true_yearly - y_pred_yearly
            plt.hist(residuals, bins=30, alpha=0.7, color='skyblue')
            plt.xlabel('Residuals (mm)')
            plt.ylabel('Frequency')
            plt.title('Residual Distribution')
            plt.grid(True, alpha=0.3)

            # 3. 聚类分布图
            plt.subplot(2, 2, 3)
            if 'cluster_assignments' in results:
                cluster_assignments = results['cluster_assignments']
                cluster_counts = np.bincount(cluster_assignments)
                colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))

                bars = plt.bar(range(len(cluster_counts)), cluster_counts, color=colors)
                plt.xlabel('Cluster ID')
                plt.ylabel('Sample Count')
                plt.title('Cluster Distribution')
                plt.xticks(range(len(cluster_counts)))

                # 在柱状图上添加数值标签
                for bar, count in zip(bars, cluster_counts):
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                             f'{count}', ha='center', va='bottom')
            else:
                plt.text(0.5, 0.5, 'No Cluster Data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Cluster Distribution')

            # 4. 性能图
            plt.subplot(2, 2, 4)
            yearly = results['yearly_cv']['overall']
            metrics = ['MAE', 'RMSE', 'R']
            values = [yearly['MAE'], yearly['RMSE'], yearly['R']]
            colors = ['skyblue', 'lightgreen', 'lightcoral']

            bars = plt.bar(metrics, values, color=colors, alpha=0.7)
            plt.ylabel('Value')
            plt.title('Yearly CV Performance')

            # 在柱状图上添加数值标签
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_visualization.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info("✅ Visualization completed")

        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")


def get_feature_importance(self):
    """获取特征重要性（基于各聚类模型的平均重要性）"""
    if not self.cluster_models:
        raise ValueError("聚类模型尚未训练")

    # 收集所有特征的重要性
    all_importances = []
    for cluster_id, model in self.cluster_models.items():
        importance_scores = model.feature_importances_
        all_importances.append(importance_scores)

    # 计算平均重要性
    avg_importance = np.mean(all_importances, axis=0)

    # 创建DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': self.feature_columns,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)

    self.logger.info(f"特征重要性计算完成，最高重要性特征: {feature_importance_df['feature'].iloc[0]}")
    return feature_importance_df


def analyze_cluster_characteristics(self, df):
    """分析各聚类的特征

    Args:
        df (pd.DataFrame): 原始数据

    Returns:
        dict: 聚类分析结果
    """
    if self.cluster_assignments is None:
        raise ValueError("聚类尚未执行")

    self.logger.info("分析各聚类特征...")

    cluster_stats = {}
    feature_cols = [col for col in self.feature_columns if col in df.columns]

    for cluster_id in range(self.n_clusters):
        cluster_mask = self.cluster_assignments == cluster_id
        cluster_data = df[cluster_mask]
        cluster_size = len(cluster_data)

        if cluster_size == 0:
            continue

        stats = {
            'size': cluster_size,
            'swe_mean': cluster_data[self.target_column].mean(),
            'swe_std': cluster_data[self.target_column].std(),
            'features': {}
        }

        # 计算各特征的统计量
        for feature in feature_cols:
            if feature in cluster_data.columns:
                stats['features'][feature] = {
                    'mean': cluster_data[feature].mean(),
                    'std': cluster_data[feature].std(),
                    'median': cluster_data[feature].median()
                }

        cluster_stats[cluster_id] = stats

        self.logger.info(f"  聚类 {cluster_id}: {cluster_size}样本, SWE均值={stats['swe_mean']:.2f}mm")

    return cluster_stats


def create_cluster_analysis_report(self, df, output_dir):
    """创建聚类分析报告

    Args:
        df (pd.DataFrame): 原始数据
        output_dir (str): 输出目录
    """
    try:
        self.logger.info("创建聚类分析报告...")

        # 获取聚类统计
        cluster_stats = self.analyze_cluster_characteristics(df)

        # 创建聚类特征对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 聚类大小分布
        cluster_sizes = [stats['size'] for stats in cluster_stats.values()]
        cluster_ids = list(cluster_stats.keys())

        axes[0, 0].bar(cluster_ids, cluster_sizes, color=plt.cm.Set3(np.linspace(0, 1, len(cluster_ids))))
        axes[0, 0].set_title('各聚类样本数量')
        axes[0, 0].set_xlabel('聚类ID')
        axes[0, 0].set_ylabel('样本数量')
        for i, v in enumerate(cluster_sizes):
            axes[0, 0].text(i, v, str(v), ha='center', va='bottom')

        # 2. 各聚类SWE均值
        swe_means = [stats['swe_mean'] for stats in cluster_stats.values()]
        axes[0, 1].bar(cluster_ids, swe_means, color=plt.cm.Set3(np.linspace(0, 1, len(cluster_ids))))
        axes[0, 1].set_title('各聚类SWE均值')
        axes[0, 1].set_xlabel('聚类ID')
        axes[0, 1].set_ylabel('SWE均值 (mm)')
        for i, v in enumerate(swe_means):
            axes[0, 1].text(i, v, f'{v:.1f}', ha='center', va='bottom')

        # 3. 重要特征在各聚类的分布
        feature_importance = self.get_feature_importance()
        top_features = feature_importance.head(3)['feature'].tolist()

        for i, feature in enumerate(top_features):
            if i >= 2:  # 只显示前两个特征
                break
            feature_means = []
            for cluster_id, stats in cluster_stats.items():
                if feature in stats['features']:
                    feature_means.append(stats['features'][feature]['mean'])
                else:
                    feature_means.append(0)

            axes[1, i].bar(cluster_ids, feature_means,
                           color=plt.cm.Set3(np.linspace(0, 1, len(cluster_ids))))
            axes[1, i].set_title(f'{feature}在各聚类的均值')
            axes[1, i].set_xlabel('聚类ID')
            axes[1, i].set_ylabel(f'{feature}均值')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 保存聚类统计到CSV
        cluster_report = []
        for cluster_id, stats in cluster_stats.items():
            row = {
                'cluster_id': cluster_id,
                'size': stats['size'],
                'swe_mean': stats['swe_mean'],
                'swe_std': stats['swe_std']
            }

            # 添加重要特征信息
            for feature in top_features:
                if feature in stats['features']:
                    row[f'{feature}_mean'] = stats['features'][feature]['mean']
                    row[f'{feature}_std'] = stats['features'][feature]['std']
                else:
                    row[f'{feature}_mean'] = np.nan
                    row[f'{feature}_std'] = np.nan

            cluster_report.append(row)

        cluster_df = pd.DataFrame(cluster_report)
        cluster_df.to_csv(os.path.join(output_dir, 'cluster_statistics.csv'), index=False)

        self.logger.info(f"✅ 聚类分析报告保存完成")

    except Exception as e:
        self.logger.warning(f"创建聚类分析报告失败: {e}")


def compare_with_baseline(self, df, output_dir):
    """与基线模型比较

    Args:
        df (pd.DataFrame): 原始数据
        output_dir (str): 输出目录
    """
    try:
        self.logger.info("与基线模型比较...")

        # 预处理数据
        X, y, station_groups, year_groups, coords = self.preprocess_data(df)

        # 训练普通XGBoost模型作为基线
        from swe_trainer import SWEXGBoostTrainer
        baseline_trainer = SWEXGBoostTrainer(params=self.params)

        # 站点交叉验证
        baseline_station_results = baseline_trainer.cross_validate(X, y, station_groups, 'station')
        baseline_yearly_results = baseline_trainer.cross_validate(X, y, year_groups, 'yearly')

        # 比较结果
        comparison = {
            'station_cv': {
                'baseline_mae': baseline_station_results['overall']['MAE'],
                'ensemble_mae': self.cross_validate(X, y, station_groups, coords, 'station')['overall']['MAE'],
                'baseline_r': baseline_station_results['overall']['R'],
                'ensemble_r': self.cross_validate(X, y, station_groups, coords, 'station')['overall']['R'],
                'improvement_mae': (baseline_station_results['overall']['MAE'] -
                                    self.cross_validate(X, y, station_groups, coords, 'station')['overall']['MAE']),
                'improvement_r': (self.cross_validate(X, y, station_groups, coords, 'station')['overall']['R'] -
                                  baseline_station_results['overall']['R'])
            },
            'yearly_cv': {
                'baseline_mae': baseline_yearly_results['overall']['MAE'],
                'ensemble_mae': self.cross_validate(X, y, year_groups, coords, 'yearly')['overall']['MAE'],
                'baseline_r': baseline_yearly_results['overall']['R'],
                'ensemble_r': self.cross_validate(X, y, year_groups, coords, 'yearly')['overall']['R'],
                'improvement_mae': (baseline_yearly_results['overall']['MAE'] -
                                    self.cross_validate(X, y, year_groups, coords, 'yearly')['overall']['MAE']),
                'improvement_r': (self.cross_validate(X, y, year_groups, coords, 'yearly')['overall']['R'] -
                                  baseline_yearly_results['overall']['R'])
            }
        }

        # 创建比较图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # MAE比较
        methods = ['基线', '聚类集成']
        station_mae = [comparison['station_cv']['baseline_mae'], comparison['station_cv']['ensemble_mae']]
        yearly_mae = [comparison['yearly_cv']['baseline_mae'], comparison['yearly_cv']['ensemble_mae']]

        x = np.arange(len(methods))
        width = 0.35

        ax1.bar(x - width / 2, station_mae, width, label='站点CV', alpha=0.7)
        ax1.bar(x + width / 2, yearly_mae, width, label='年度CV', alpha=0.7)
        ax1.set_xlabel('模型类型')
        ax1.set_ylabel('MAE (mm)')
        ax1.set_title('MAE比较')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # R值比较
        station_r = [comparison['station_cv']['baseline_r'], comparison['station_cv']['ensemble_r']]
        yearly_r = [comparison['yearly_cv']['baseline_r'], comparison['yearly_cv']['ensemble_r']]

        ax2.bar(x - width / 2, station_r, width, label='站点CV', alpha=0.7)
        ax2.bar(x + width / 2, yearly_r, width, label='年度CV', alpha=0.7)
        ax2.set_xlabel('模型类型')
        ax2.set_ylabel('R')
        ax2.set_title('相关系数比较')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'baseline_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 保存比较结果
        comparison_df = pd.DataFrame([
            {
                'method': 'station_cv',
                'baseline_mae': comparison['station_cv']['baseline_mae'],
                'ensemble_mae': comparison['station_cv']['ensemble_mae'],
                'improvement_mae': comparison['station_cv']['improvement_mae'],
                'baseline_r': comparison['station_cv']['baseline_r'],
                'ensemble_r': comparison['station_cv']['ensemble_r'],
                'improvement_r': comparison['station_cv']['improvement_r']
            },
            {
                'method': 'yearly_cv',
                'baseline_mae': comparison['yearly_cv']['baseline_mae'],
                'ensemble_mae': comparison['yearly_cv']['ensemble_mae'],
                'improvement_mae': comparison['yearly_cv']['improvement_mae'],
                'baseline_r': comparison['yearly_cv']['baseline_r'],
                'ensemble_r': comparison['yearly_cv']['ensemble_r'],
                'improvement_r': comparison['yearly_cv']['improvement_r']
            }
        ])

        comparison_df.to_csv(os.path.join(output_dir, 'baseline_comparison.csv'), index=False)

        self.logger.info("✅ 基线比较完成")
        return comparison

    except Exception as e:
        self.logger.warning(f"基线比较失败: {e}")
        return None


# 便捷使用函数
def train_swe_cluster_ensemble(data_df, output_dir=None, n_clusters=4, params=None, use_rf=False,
                               use_enhanced_gnnwr=True, gnnwr_params=None, device='auto'):
    """便捷函数：训练SWE聚类集成模型

    Args:
        data_df (pd.DataFrame): 包含特征和SWE的数据
        output_dir (str, optional): 输出目录路径
        n_clusters (int, optional): 聚类数量
        params (dict, optional): XGBoost参数
        use_enhanced_gnnwr (bool): 是否使用增强版GNNWR
        gnnwr_params (dict): GNNWR参数

    Returns:
        dict: 包含所有训练结果的字典
    """
    trainer = SWEClusterEnsemble(
        n_clusters=n_clusters,
        params=params,
        gnnwr_params=gnnwr_params,
        use_enhanced_gnnwr=use_enhanced_gnnwr,
        use_rf = use_rf, # 传递这个参数
        device = device  # 添加device参数
    )
    return trainer.run_complete_analysis(data_df, output_dir)


def load_swe_cluster_ensemble(model_path):
    """加载已训练的SWE聚类集成模型

    Args:
        model_path (str): 模型文件路径

    Returns:
        SWEClusterEnsemble: 加载的模型实例
    """
    model_data = joblib.load(model_path)

    trainer = SWEClusterEnsemble(
        n_clusters=model_data['n_clusters'],
        params=model_data['params'],
        gnnwr_params=model_data['gnnwr_params'],
        use_enhanced_gnnwr=model_data.get('use_enhanced_gnnwr', True)
    )

    trainer.kmeans = model_data['kmeans']
    trainer.cluster_models = model_data['cluster_models']
    trainer.gnnwr_trainer = model_data['gnnwr_trainer']
    trainer.feature_columns = model_data['feature_columns']
    trainer.cluster_assignments = model_data.get('cluster_assignments')

    # 恢复RF参数（如果存在）
    if 'rf_params' in model_data:
        trainer.rf_params = model_data['rf_params']

    return trainer


# 测试函数
def test_cluster_ensemble():
    """测试聚类集成模型"""
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # 生成空间坐标
    coords = np.random.uniform(0, 100, (n_samples, 2))

    # 生成特征
    features = np.random.randn(n_samples, n_features)

    # 创建具有空间相关性的目标变量
    spatial_effect = np.exp(-0.01 * coords[:, 0]) + np.sin(0.1 * coords[:, 1])
    targets = (features[:, 0] + 2 * features[:, 1] + 0.5 * spatial_effect +
               np.random.normal(0, 0.1, n_samples))

    # 创建模拟数据框
    df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(n_features)])
    df['swe'] = targets
    df['station_id'] = [f'station_{i % 20}' for i in range(n_samples)]
    df['year'] = np.random.randint(2018, 2023, n_samples)
    df['longitude'] = coords[:, 0]
    df['latitude'] = coords[:, 1]

    # 训练模型
    results = train_swe_cluster_ensemble(
        data_df=df,
        n_clusters=3,
        use_enhanced_gnnwr=True
    )

    return results


class PureGNNWRModel(nn.Module):
    """纯净版GNNWR模型 - 直接特征输入，专注深度学习优化"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], output_dim=1,
                 dropout_rate=0.3, use_batch_norm=True):
        super(PureGNNWRModel, self).__init__()

        # 深度特征提取网络
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.feature_network = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(prev_dim // 2, output_dim)
        )

    def forward(self, x, spatial_weights=None, coords=None):
        # 特征提取
        features = self.feature_network(x)

        # 空间平滑（如果提供了空间权重）
        if spatial_weights is not None:
            # 空间平滑：每个位置的特征是其邻近位置的加权平均
            row_sums = torch.sum(spatial_weights, dim=1, keepdim=True)
            normalized_weights = spatial_weights / torch.where(row_sums > 0, row_sums, torch.tensor(1.0))
            smoothed_features = torch.matmul(normalized_weights, features)
            output = self.output_layer(smoothed_features)
        else:
            output = self.output_layer(features)

        return output.squeeze()


class PureGNNWRTrainer:
    """纯净版GNNWR训练器 - 全套深度学习优化"""

    def __init__(self, input_dim, coords, hidden_dims=[128, 64, 32, 16],
                 learning_rate=0.001, bandwidth=10.0, dropout_rate=0.3,
                 weight_decay=1e-4, device='auto', output_std_penalty=0.01):

        # 设备设置
        if device == 'auto':
            self.device = torch.device('cpu')
            torch.set_num_threads(16)
        else:
            self.device = torch.device(device)

        self.output_std_penalty = output_std_penalty
        self.logger = logging.getLogger("PureGNNWR")
        self.logger.info(f"纯净版GNNWR - 使用设备: {self.device}")

        # 模型初始化
        self.model = PureGNNWRModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(self.device)

        # 优化器 - 使用AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        self.criterion = nn.HuberLoss()

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        self.criterion = nn.HuberLoss()  # 使用HuberLoss更稳定

        # 空间权重计算
        self.coords = coords.copy() if coords is not None else None
        self.bandwidth = bandwidth

    def _compute_spatial_weights(self, batch_coords):
        """计算空间权重矩阵"""
        n_batch = batch_coords.shape[0]
        if n_batch <= 1:
            return torch.ones((n_batch, n_batch), device=self.device)

        # 计算欧氏距离
        diff = batch_coords.unsqueeze(1) - batch_coords.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-8)

        # 高斯核函数
        weights = torch.exp(-0.5 * (distances / self.bandwidth) ** 2)

        return weights

    def train(self, train_loader, val_loader=None, epochs=200, early_stopping_patience=20):
        """完整深度学习训练流程 - 修复版本"""

        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        pbar = tqdm(range(epochs), desc="训练进度")

        for epoch in pbar:
            # 训练阶段
            self.model.train()
            epoch_train_loss = 0.0
            batch_count = 0

            for batch in train_loader:
                if len(batch) == 3:
                    batch_features, batch_targets, batch_coords = batch
                else:
                    batch_features, batch_targets = batch
                    batch_coords = None

                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                # 重要：每次迭代前清零梯度
                self.optimizer.zero_grad()

                # 计算空间权重（如果有坐标）
                spatial_weights = None
                if batch_coords is not None:
                    batch_coords = batch_coords.to(self.device)
                    spatial_weights = self._compute_spatial_weights(batch_coords)

                # 前向传播
                outputs = self.model(batch_features, spatial_weights, batch_coords)

                # 计算主损失
                main_loss = self.criterion(outputs, batch_targets)

                # 添加输出多样性惩罚（防止输出恒定）
                output_std = torch.std(outputs)
                diversity_loss = -self.output_std_penalty * output_std  # 鼓励输出有方差

                # 总损失
                total_loss = main_loss + diversity_loss

                # 重要：只调用一次 backward() 和 step()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 更新参数
                self.optimizer.step()

                # 只记录主损失用于显示
                epoch_train_loss += main_loss.item()
                batch_count += 1

            # 计算平均训练损失
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)

            # 验证阶段
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)

                # 学习率调度
                self.scheduler.step(val_loss)

                # 更新进度条
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'train_loss': f'{epoch_train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'patience': f'{patience_counter}/{early_stopping_patience}'
                })

                # 早停逻辑
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_pure_gnnwr_model.pth')
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    pbar.set_description("训练完成 (早停)")
                    self.logger.info(f"早停在epoch {epoch}, 最佳验证loss: {best_val_loss:.6f}")
                    # 加载最佳模型
                    self.model.load_state_dict(torch.load('best_pure_gnnwr_model.pth'))
                    break
            else:
                # 如果没有验证集，使用训练loss
                self.scheduler.step(epoch_train_loss)

                # 更新进度条（无验证集版本）
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'train_loss': f'{epoch_train_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'patience': f'{patience_counter}/{early_stopping_patience}'
                })

            # 日志输出
            if epoch % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                if val_loader is not None:
                    self.logger.info(f"Epoch {epoch:3d} | Train Loss: {epoch_train_loss:.6f} | "
                                     f"Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
                else:
                    self.logger.info(f"Epoch {epoch:3d} | Train Loss: {epoch_train_loss:.6f} | "
                                     f"LR: {current_lr:.2e}")

        pbar.close()

        return train_losses, val_losses

    def validate(self, val_loader):
        """验证集评估"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    batch_features, batch_targets, batch_coords = batch
                else:
                    batch_features, batch_targets = batch
                    batch_coords = None

                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                spatial_weights = None
                if batch_coords is not None:
                    batch_coords = batch_coords.to(self.device)
                    spatial_weights = self._compute_spatial_weights(batch_coords)

                outputs = self.model(batch_features, spatial_weights, batch_coords)
                loss = self.criterion(outputs, batch_targets)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def predict(self, features, coords=None):
        """预测"""
        self.model.eval()

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)

            # 分批预测避免内存溢出
            batch_size = 1024
            predictions = []

            for i in range(0, len(features), batch_size):
                batch_features = features_tensor[i:i + batch_size]

                spatial_weights = None
                if coords is not None:
                    batch_coords = torch.FloatTensor(coords[i:i + batch_size]).to(self.device)
                    spatial_weights = self._compute_spatial_weights(batch_coords)

                batch_pred = self.model(batch_features, spatial_weights, batch_coords)
                predictions.append(batch_pred.cpu().numpy())

            return np.concatenate(predictions)


def train_pure_gnnwr_analysis(df, output_dir=None, test_size=0.2, random_state=42):
    """
    运行纯净版GNNWR分析 - 包含完整交叉验证
    """
    from sklearn.model_selection import train_test_split, LeaveOneGroupOut
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr
    import numpy as np

    logger = logging.getLogger("PureGNNWRAnalysis")
    logger.info("=" * 60)
    logger.info("🚀 开始纯净版GNNWR完整分析流程")
    logger.info("=" * 60)

    try:
        # 使用SWEClusterEnsemble的数据预处理
        ensemble = SWEClusterEnsemble(n_clusters=1)  # 临时实例用于数据预处理
        X, y, station_groups, year_groups, coords = ensemble.preprocess_data(df)

        logger.info(f"数据加载: {len(X)}样本, {X.shape[1]}特征")
        logger.info(f"站点数: {len(np.unique(station_groups))}, 年份数: {len(np.unique(year_groups))}")

        # 1. 站点交叉验证
        logger.info("\n" + "=" * 50)
        logger.info("步骤 1: 站点交叉验证")
        logger.info("=" * 50)

        station_cv_results = pure_gnnwr_cross_validate(
            X, y, station_groups, coords, 'station', logger
        )

        # 2. 年度交叉验证
        logger.info("\n" + "=" * 50)
        logger.info("步骤 2: 年度交叉验证")
        logger.info("=" * 50)

        yearly_cv_results = pure_gnnwr_cross_validate(
            X, y, year_groups, coords, 'yearly', logger
        )

        # 3. 标准训练测试集分割
        logger.info("\n" + "=" * 50)
        logger.info("步骤 3: 标准训练测试集验证")
        logger.info("=" * 50)

        X_train, X_test, y_train, y_test, coords_train, coords_test, station_train, station_test = train_test_split(
            X, y, coords, station_groups, test_size=test_size, random_state=random_state
        )

        logger.info(f"数据划分: 训练集 {len(X_train)}, 测试集 {len(X_test)}")

        # 创建数据集
        train_dataset = EnhancedSpatialDataset(X_train, y_train, coords_train)
        test_dataset = EnhancedSpatialDataset(X_test, y_test, coords_test)

        # 数据加载器
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

        # 训练纯净版GNNWR
        trainer = PureGNNWRTrainer(
            input_dim=X.shape[1],
            coords=coords_train,
            hidden_dims=[128, 64, 32, 16],
            learning_rate=0.001,
            dropout_rate=0.3,
            weight_decay=1e-4,
            device='cpu'  # 使用CPU
        )

        logger.info("开始纯净版GNNWR训练...")

        # 训练
        train_losses, val_losses = trainer.train(train_loader, test_loader, epochs=200)

        # 最终评估
        y_pred = trainer.predict(X_test, coords_test)

        # 计算评估指标
        test_metrics = evaluate_predictions(y_test, y_pred)

        # 整合所有结果
        results = {
            'station_cv': station_cv_results,
            'yearly_cv': yearly_cv_results,
            'standard_test': test_metrics,
            'trainer': trainer,
            'data_info': {
                'total_samples': len(X),
                'n_features': X.shape[1],
                'n_stations': len(np.unique(station_groups)),
                'n_years': len(np.unique(year_groups)),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        }

        # === 新增：保存结果和生成图表 ===
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"./pure_gnnwr_results_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"保存结果到: {output_dir}")

        # 保存模型
        model_path = os.path.join(output_dir, 'pure_gnnwr_model.pth')
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'config': {
                'input_dim': X.shape[1],
                'hidden_dims': [128, 64, 32, 16],
                'learning_rate': 0.001
            }
        }, model_path)

        # 保存结果数据
        results_path = os.path.join(output_dir, 'pure_gnnwr_results.pkl')
        joblib.dump(results, results_path)

        # 生成可视化图表
        create_pure_gnnwr_visualizations(results, output_dir)

        # 生成详细报告
        report_path = os.path.join(output_dir, 'pure_gnnwr_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(generate_detailed_report(results))

        # 输出综合报告
        print_comprehensive_report(results)

        logger.info("🎯 纯净版GNNWR完整分析完成!")
        return results, trainer

    except Exception as e:
        logger.error(f"纯净版GNNWR分析失败: {e}")
        raise


def create_pure_gnnwr_visualizations(results, output_dir):
    """生成纯净版GNNWR的可视化图表 - 包含测试集大小信息"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(18, 12))

    # 1. 站点交叉验证散点图
    plt.subplot(2, 4, 1)
    station_cv = results['station_cv']
    y_true_station = station_cv['true_values']
    y_pred_station = station_cv['predictions']

    plt.scatter(y_true_station, y_pred_station, alpha=0.6, s=20, color='blue')
    plt.plot([y_true_station.min(), y_true_station.max()],
             [y_true_station.min(), y_true_station.max()], 'r--', alpha=0.8)
    plt.xlabel('True SWE (mm)')
    plt.ylabel('Predicted SWE (mm)')
    plt.title(f'Station CV\nMAE={station_cv["overall"]["MAE"]:.2f}, R={station_cv["overall"]["R"]:.3f}')
    plt.grid(True, alpha=0.3)

    # 2. 年度交叉验证散点图
    plt.subplot(2, 4, 2)
    yearly_cv = results['yearly_cv']
    y_true_yearly = yearly_cv['true_values']
    y_pred_yearly = yearly_cv['predictions']

    plt.scatter(y_true_yearly, y_pred_yearly, alpha=0.6, s=20, color='green')
    plt.plot([y_true_yearly.min(), y_true_yearly.max()],
             [y_true_yearly.min(), y_true_yearly.max()], 'r--', alpha=0.8)
    plt.xlabel('True SWE (mm)')
    plt.ylabel('Predicted SWE (mm)')
    plt.title(f'Yearly CV\nMAE={yearly_cv["overall"]["MAE"]:.2f}, R={yearly_cv["overall"]["R"]:.3f}')
    plt.grid(True, alpha=0.3)

    # 3. 测试集大小分布 - 站点
    plt.subplot(2, 4, 3)
    station_test_sizes = [info['test_size'] for info in station_cv['by_fold'].values()]
    plt.hist(station_test_sizes, bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('测试集大小 (样本数)')
    plt.ylabel('折叠数量')
    plt.title(f'站点CV测试集大小分布\n平均={np.mean(station_test_sizes):.1f}')

    # 4. 测试集大小分布 - 年度
    plt.subplot(2, 4, 4)
    yearly_test_sizes = [info['test_size'] for info in yearly_cv['by_fold'].values()]
    plt.hist(yearly_test_sizes, bins=20, alpha=0.7, color='lightgreen')
    plt.xlabel('测试集大小 (样本数)')
    plt.ylabel('折叠数量')
    plt.title(f'年度CV测试集大小分布\n平均={np.mean(yearly_test_sizes):.1f}')

    # 5. 性能对比柱状图
    plt.subplot(2, 4, 5)
    methods = ['Station CV', 'Yearly CV', 'Standard Test']
    mae_values = [
        station_cv['overall']['MAE'],
        yearly_cv['overall']['MAE'],
        results['standard_test']['MAE']
    ]

    bars = plt.bar(methods, mae_values, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('MAE (mm)')
    plt.title('Performance Comparison (MAE)')
    for bar, value in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{value:.2f}', ha='center', va='bottom')

    # 6. R值对比柱状图
    plt.subplot(2, 4, 6)
    r_values = [
        station_cv['overall']['R'],
        yearly_cv['overall']['R'],
        results['standard_test']['R']
    ]

    bars = plt.bar(methods, r_values, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('R')
    plt.title('Performance Comparison (R)')
    for bar, value in zip(bars, r_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    # 7. 折叠统计
    plt.subplot(2, 4, 7)
    fold_stats = {
        '总站点折叠': station_cv['total_folds'],
        '成功站点折叠': station_cv['folds'],
        '总年度折叠': yearly_cv['total_folds'],
        '成功年度折叠': yearly_cv['folds']
    }

    plt.bar(fold_stats.keys(), fold_stats.values(), color='lightgray')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('数量')
    plt.title('交叉验证折叠统计')

    # 8. 残差分布
    plt.subplot(2, 4, 8)
    residuals = y_true_station - y_pred_station
    plt.hist(residuals, bins=30, alpha=0.7, color='orange')
    plt.xlabel('残差 (mm)')
    plt.ylabel('频率')
    plt.title('站点CV残差分布')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pure_gnnwr_comprehensive_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 综合可视化图表已保存")


def generate_detailed_report(results):
    """生成详细的分析报告"""
    report = []
    report.append("=" * 80)
    report.append("🎯 纯净版GNNWR详细分析报告")
    report.append("=" * 80)
    report.append("")

    # 数据概况
    data_info = results['data_info']
    report.append("📊 数据概况:")
    report.append(f"  总样本数: {data_info['total_samples']}")
    report.append(f"  特征数量: {data_info['n_features']}")
    report.append(f"  站点数量: {data_info['n_stations']}")
    report.append(f"  年份数量: {data_info['n_years']}")
    report.append(f"  训练集大小: {data_info['train_size']}")
    report.append(f"  测试集大小: {data_info['test_size']}")
    report.append("")

    # 站点交叉验证详细结果
    station_cv = results['station_cv']
    station_overall = station_cv['overall']
    report.append("🏔️ 站点交叉验证详细结果:")
    report.append(f"  总折叠数: {station_cv['total_folds']}")
    report.append(f"  成功折叠: {station_cv['folds']}")
    report.append(f"  跳过折叠: {station_cv.get('skipped_folds', 0)}")
    report.append(f"  MAE: {station_overall['MAE']:.3f} mm")
    report.append(f"  RMSE: {station_overall['RMSE']:.3f} mm")
    report.append(f"  R: {station_overall['R']:.3f}")
    report.append(f"  R²: {station_overall['R_squared']:.3f}")
    report.append(f"  样本数: {station_overall['samples']}")
    report.append("")

    # 年度交叉验证详细结果
    yearly_cv = results['yearly_cv']
    yearly_overall = yearly_cv['overall']
    report.append("📅 年度交叉验证详细结果:")
    report.append(f"  总折叠数: {yearly_cv['total_folds']}")
    report.append(f"  成功折叠: {yearly_cv['folds']}")
    report.append(f"  跳过折叠: {yearly_cv.get('skipped_folds', 0)}")
    report.append(f"  MAE: {yearly_overall['MAE']:.3f} mm")
    report.append(f"  RMSE: {yearly_overall['RMSE']:.3f} mm")
    report.append(f"  R: {yearly_overall['R']:.3f}")
    report.append(f"  R²: {yearly_overall['R_squared']:.3f}")
    report.append(f"  样本数: {yearly_overall['samples']}")
    report.append("")

    # 标准测试集结果
    standard_test = results['standard_test']
    report.append("🧪 标准测试集结果:")
    report.append(f"  MAE: {standard_test['MAE']:.3f} mm")
    report.append(f"  RMSE: {standard_test['RMSE']:.3f} mm")
    report.append(f"  R: {standard_test['R']:.3f}")
    report.append(f"  R²: {standard_test['R_squared']:.3f}")
    report.append(f"  样本数: {standard_test['samples']}")
    report.append("")

    # 性能总结
    report.append("📈 性能总结:")
    best_mae = min(station_overall['MAE'], yearly_overall['MAE'], standard_test['MAE'])
    best_r = max(station_overall['R'], yearly_overall['R'], standard_test['R'])
    report.append(f"  最佳MAE: {best_mae:.3f} mm")
    report.append(f"  最佳R值: {best_r:.3f}")
    report.append("")

    report.append("=" * 80)
    report.append("报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    report.append("=" * 80)

    return "\n".join(report)


def pure_gnnwr_cross_validate_fixed(X, y, groups, coords, cv_type, logger):
    """修复的交叉验证 - 正确理解LOGO逻辑"""
    from sklearn.model_selection import LeaveOneGroupOut

    logo = LeaveOneGroupOut()
    all_predictions = []
    all_true_values = []
    fold_results = {}
    skipped_folds = 0

    unique_groups = np.unique(groups)
    total_folds = len(unique_groups)

    logger.info(f"开始{cv_type}交叉验证，共{total_folds}个折叠...")

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        group_id = groups[test_idx[0]]
        test_size = len(test_idx)
        train_size = len(train_idx)

        # 训练集应该是很大的（所有其他站点），测试集可能很小
        logger.info(f"Fold {fold + 1}/{total_folds}: {cv_type} {group_id}, 训练集={train_size}, 测试集={test_size}")

        # 分割数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 分割坐标
        coords_train = coords[train_idx] if coords is not None else None
        coords_test = coords[test_idx] if coords is not None else None

        # 验证数据分割正确性
        unique_train_groups = len(np.unique(groups[train_idx]))
        unique_test_groups = len(np.unique(groups[test_idx]))

        logger.debug(f"  训练集包含 {unique_train_groups} 个{'' if cv_type == 'station' else '年份'}")
        logger.debug(f"  测试集包含 {unique_test_groups} 个{'' if cv_type == 'station' else '年份'}")

        try:
            # 使用完整模型（训练集很大，可以用复杂模型）
            trainer = PureGNNWRTrainer(
                input_dim=X.shape[1],
                coords=coords_train,
                hidden_dims=[128, 64, 32, 16],  # 使用完整模型
                learning_rate=0.001,
                dropout_rate=0.3,
                device='cpu',
                output_std_penalty=0.05  # 防止输出恒定
            )

            # 创建数据集 - 使用较大的batch_size（训练集大）
            train_dataset = EnhancedSpatialDataset(X_train, y_train, coords_train)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

            # 训练 - 可以使用更多epoch（训练集大，不容易过拟合）
            trainer.train(train_loader, epochs=100, early_stopping_patience=15)

            # 预测
            y_pred = trainer.predict(X_test, coords_test)

            # 检查预测结果质量
            if len(test_idx) > 1 and np.std(y_pred) < 1e-6:  # 只有测试集>1时才检查
                logger.warning(f"折叠 {fold + 1}: 预测结果恒定，可能模型有问题")
                # 但仍然记录结果

            # 存储结果
            all_predictions.extend(y_pred)
            all_true_values.extend(y_test)

            # 计算当前折叠性能
            fold_metrics = evaluate_predictions(y_test, y_pred)
            fold_results[group_id] = {
                **fold_metrics,
                'train_size': train_size,
                'test_size': test_size
            }

            logger.info(
                f"  {cv_type} Fold {fold + 1}: {group_id} - "
                f"Train={train_size}, Test={test_size}, "
                f"MAE={fold_metrics['MAE']:.3f}, R={fold_metrics['R']:.3f}"
            )

        except Exception as e:
            logger.error(f"折叠 {fold + 1} 训练失败: {e}")
            skipped_folds += 1
            continue

    # 计算总体性能
    if len(all_true_values) == 0:
        logger.error(f"{cv_type}交叉验证没有有效结果")
        return {
            'overall': {'MAE': 0, 'RMSE': 0, 'R': 0, 'R_squared': 0, 'samples': 0},
            'by_fold': {},
            'predictions': np.array([]),
            'true_values': np.array([]),
            'folds': 0,
            'total_folds': total_folds,
            'skipped_folds': skipped_folds
        }

    overall_metrics = evaluate_predictions(
        np.array(all_true_values),
        np.array(all_predictions)
    )

    # 分析折叠结果
    successful_folds = len(fold_results)
    avg_test_size = np.mean([info['test_size'] for info in fold_results.values()])

    logger.info(f"✅ {cv_type}交叉验证完成")
    logger.info(f"  总折叠数: {total_folds}, 成功: {successful_folds}, 失败: {skipped_folds}")
    logger.info(f"  平均测试集大小: {avg_test_size:.1f} 样本/折叠")
    logger.info(f"  聚合性能: MAE={overall_metrics['MAE']:.3f}, R={overall_metrics['R']:.3f}")

    return {
        'overall': overall_metrics,
        'by_fold': fold_results,
        'predictions': np.array(all_predictions),
        'true_values': np.array(all_true_values),
        'folds': successful_folds,
        'total_folds': total_folds,
        'skipped_folds': skipped_folds,
        'avg_test_size': avg_test_size
    }


def _is_constant_data(data, axis=None):
    """检查数据是否恒定（所有值相同）"""
    if data is None or len(data) == 0:
        return True

    if axis is not None:
        # 对于多维数据，检查每个特征是否恒定
        return np.all(np.std(data, axis=axis) == 0)
    else:
        # 对于一维数据
        return np.std(data) == 0


def evaluate_predictions(y_true, y_pred):
    """评估预测性能 - 修复常数输入问题"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr
    import numpy as np
    import warnings

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 检查数据是否恒定
    y_true_std = np.std(y_true)
    y_pred_std = np.std(y_pred)

    # 如果任一数组是常数，相关系数设为0
    if y_true_std == 0 or y_pred_std == 0:
        r_value = 0.0
        warnings.warn(f"检测到常数输入: y_true_std={y_true_std:.6f}, y_pred_std={y_pred_std:.6f}，相关系数设为0")
    else:
        try:
            r_value, _ = pearsonr(y_true, y_pred)
        except:
            r_value = 0.0

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R': r_value,
        'R_squared': r_value ** 2,
        'samples': len(y_true),
        'y_true_std': y_true_std,
        'y_pred_std': y_pred_std
    }


def print_comprehensive_report(results):
    """打印综合报告"""
    print("\n" + "=" * 70)
    print("🎯 纯净版GNNWR完整分析报告")
    print("=" * 70)

    # 数据概况
    data_info = results['data_info']
    print(f"\n📊 数据概况:")
    print(f"  总样本数: {data_info['total_samples']}")
    print(f"  特征数量: {data_info['n_features']}")
    print(f"  站点数量: {data_info['n_stations']}")
    print(f"  年份数量: {data_info['n_years']}")

    # 站点交叉验证结果
    station_cv = results['station_cv']['overall']
    print(f"\n🏔️ 站点交叉验证:")
    print(f"  折叠数量: {results['station_cv']['folds']}")
    print(f"  MAE: {station_cv['MAE']:.3f} mm")
    print(f"  RMSE: {station_cv['RMSE']:.3f} mm")
    print(f"  R: {station_cv['R']:.3f}")
    print(f"  R²: {station_cv['R_squared']:.3f}")

    # 年度交叉验证结果
    yearly_cv = results['yearly_cv']['overall']
    print(f"\n📅 年度交叉验证:")
    print(f"  折叠数量: {results['yearly_cv']['folds']}")
    print(f"  MAE: {yearly_cv['MAE']:.3f} mm")
    print(f"  RMSE: {yearly_cv['RMSE']:.3f} mm")
    print(f"  R: {yearly_cv['R']:.3f}")
    print(f"  R²: {yearly_cv['R_squared']:.3f}")

    # 标准测试集结果
    standard_test = results['standard_test']
    print(f"\n🧪 标准测试集:")
    print(f"  样本数量: {standard_test['samples']}")
    print(f"  MAE: {standard_test['MAE']:.3f} mm")
    print(f"  RMSE: {standard_test['RMSE']:.3f} mm")
    print(f"  R: {standard_test['R']:.3f}")
    print(f"  R²: {standard_test['R_squared']:.3f}")

    print("=" * 70)


# 在SWEClusterEnsemble类中添加一个便捷方法
def SWEClusterEnsemble_run_pure_comparison(self, df):
    """
    在SWEClusterEnsemble类中添加的方法
    用于快速运行纯净版对比实验
    """
    return train_pure_gnnwr_analysis(df)





if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("测试聚类集成模型...")

    # 直接运行纯净版对比（需要先有数据文件）
    try:
        import pandas as pd
        df = pd.read_excel("lu_onehot.xlsx")  # 替换为您的数据文件
        results, trainer = train_pure_gnnwr_analysis(df)
    except Exception as e:
        print(f"示例运行失败: {e}")
        print("请确保有数据文件并修改文件路径")
    print("测试完成！")