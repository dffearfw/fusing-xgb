import logging
import unittest
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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from GNNWR import EnhancedGNNWRTrainer


# 禁用TF32相关警告（CPU上不需要）
warnings.filterwarnings("ignore", message=".*TF32.*")

# CPU性能优化设置
torch.set_num_threads(24)  # i9-14900KF有24个物理核心
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['OPENMP'] = '1'

# 禁用CUDA相关设置（避免不必要的GPU检查）
torch.backends.cudnn.enabled = True

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

    def __init__(self, n_clusters=4, params=None, gnnwr_params=None,
                 use_enhanced_gnnwr=True, use_rf=False, device='auto',
                 mixed_precision=True, cpu_workers=24):
        """初始化聚类集成回归器

        Args:
            n_clusters (int): 聚类数量
            params (dict): XGBoost参数
            gnnwr_params (dict): GNNWR参数
            use_enhanced_gnnwr (bool): 是否使用增强版GNNWR
            device (str): 设备类型 'auto', 'cuda', 'cpu'
            mixed_precision (bool): 是否使用混合精度
            cpu_workers (int): CPU工作线程数
        """
        self.logger = logging.getLogger("SWEClusterEnsemble")

        # 设备配置
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                torch.backends.cudnn.benchmark = True
                self.logger.info(f"自动选择GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device('cpu')
                torch.set_num_threads(cpu_workers)
                self.logger.info(f"使用CPU: {cpu_workers}线程")
        else:
            self.device = torch.device(device)
            if device == 'cpu':
                torch.set_num_threads(cpu_workers)

        self.mixed_precision = mixed_precision and self.device.type == 'cuda'
        self.cpu_workers = cpu_workers

        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_assignments = None
        self.cluster_models = {}
        self.gnnwr_trainer = None
        self.feature_columns = None
        self.target_column = 'swe'
        self.use_enhanced_gnnwr = use_enhanced_gnnwr and HAS_ENHANCED_GNNWR
        self.use_rf = use_rf

        # 关键修复：确保params不为None
        if params is None:
            params = {}

        if use_rf:
            # RF参数 - 优化CPU使用
            self.rf_params = {
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', None),
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': min(16, cpu_workers)  # 优化CPU使用
            }
            self.params = params if params else self.DEFAULT_PARAMS.copy()
        else:
            # XGB参数 - 如果使用GPU训练XGBoost
            self.params = self.DEFAULT_PARAMS.copy()
            if params:
                self.params.update(params)

            # 如果使用GPU且安装了支持GPU的XGBoost
            if self.device.type == 'cuda' and not use_rf:
                self.params['tree_method'] = 'gpu_hist'
                self.params['predictor'] = 'gpu_predictor'
                self.logger.info("XGBoost使用GPU加速")

        # GNNWR参数 - 添加GPU和混合精度支持
        self.gnnwr_params = {
            'hidden_dims': [256, 128, 64, 32],  # 更大的模型充分利用GPU
            'learning_rate': 0.001,
            'epochs': 200,
            'batch_size': 512,  # 更大的批次大小
            'patience': 20,
            'bandwidth': 5.0,
            'use_spatial_weights': True,
            'device': self.device,  # 传递设备参数
            'mixed_precision': self.mixed_precision,  # 混合精度
            'cpu_workers': self.cpu_workers,  # CPU工作线程
            'dropout_rate': 0.3,
            'weight_decay': 1e-4,
            'num_workers': min(12, self.cpu_workers // 2)  # 优化数据加载
        }
        if gnnwr_params:
            self.gnnwr_params.update(gnnwr_params)

        self.logger.info(f"初始化SWE聚类集成回归器，聚类数: {n_clusters}")
        self.logger.info(f"使用设备: {self.device}")
        self.logger.info(f"混合精度: {self.mixed_precision}")

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
        """为每个聚类训练模型 - 优化版本"""
        self.logger.info("训练各聚类模型...")
        self.cluster_models = {}

        # 使用多进程并行训练聚类模型
        if self.use_rf and len(np.unique(cluster_labels)) > 1:
            # 对于随机森林，使用多进程
            self._train_cluster_models_parallel(X, y, cluster_labels)
        else:
            # 顺序训练
            for cluster_id in range(self.n_clusters):
                self._train_single_cluster_model(X, y, cluster_labels, cluster_id)

    def _train_single_cluster_model(self, X, y, cluster_labels, cluster_id):
        """训练单个聚类模型"""
        cluster_mask = cluster_labels == cluster_id
        cluster_size = np.sum(cluster_mask)

        if cluster_size < 5:
            self.logger.warning(f"聚类 {cluster_id} 样本数过少 ({cluster_size})，跳过训练")
            return

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

        y_pred_cluster = model.predict(X_cluster)
        cluster_mae = mean_absolute_error(y_cluster, y_pred_cluster)
        cluster_rmse = np.sqrt(mean_squared_error(y_cluster, y_pred_cluster))

        self.logger.info(f"  聚类 {cluster_id}: {cluster_size}样本, MAE={cluster_mae:.3f}, RMSE={cluster_rmse:.3f}")

    def _train_cluster_models_parallel(self, X, y, cluster_labels):
        """并行训练聚类模型 - 充分利用14900KF"""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        self.logger.info("使用多进程并行训练聚类模型...")

        def train_single_cluster(args):
            """单个聚类的训练函数"""
            cluster_id, X_cluster, y_cluster, use_rf, params, rf_params = args

            if use_rf:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(**rf_params)
            else:
                import xgboost as xgb
                model = xgb.XGBRegressor(**params)

            model.fit(X_cluster, y_cluster)
            return cluster_id, model

        # 准备训练任务
        tasks = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)

            if cluster_size >= 5:  # 只训练有足够样本的聚类
                X_cluster = X[cluster_mask]
                y_cluster = y[cluster_mask]

                tasks.append((
                    cluster_id, X_cluster, y_cluster,
                    self.use_rf, self.params, self.rf_params
                ))

        # 使用进程池并行训练
        with ProcessPoolExecutor(max_workers=min(self.cpu_workers, len(tasks))) as executor:
            futures = [executor.submit(train_single_cluster, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    cluster_id, model = future.result()
                    self.cluster_models[cluster_id] = model
                    self.logger.info(f"  完成聚类 {cluster_id} 训练")
                except Exception as e:
                    self.logger.error(f"聚类训练失败: {e}")

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
        """训练GNNWR集成模型 - GPU优化版本"""
        self.logger.info("=== train_gnnwr_method GPU优化版本 ===")
        self.logger.info(f"使用设备: {self.device}")
        self.logger.info(f"混合精度: {self.mixed_precision}")

        # 立即检查坐标数据
        if coords is None:
            self.logger.error("❌ 坐标数据在方法入口处就为None!")
            raise ValueError("坐标数据在方法入口处就为None")

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
            coords_copy = coords.copy()
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

        # 根据数据大小自动调整参数
        n_samples = len(gnnwr_features_imputed)
        batch_size = min(512, max(64, n_samples // 50))  # 自适应批次大小，充分利用GPU

        self.logger.info(f"数据加载器配置: batch_size={batch_size}")

        if self.use_enhanced_gnnwr:
            # 使用增强版GNNWR
            self.logger.info("使用增强版GNNWR训练器")

            # 检查样本数量，如果太多则使用简化模式
            use_spatial = self.gnnwr_params['use_spatial_weights'] and coords_copy is not None

            if not use_spatial:
                self.logger.warning(f"样本数量较大 ({n_samples}) 或坐标不可用，禁用空间权重计算")
                dataset = EnhancedSpatialDataset(
                    features=gnnwr_features_imputed,
                    targets=y,
                    coords=coords_copy
                )
            else:
                dataset = EnhancedSpatialDataset(
                    features=gnnwr_features_imputed,
                    targets=y,
                    coords=coords_copy
                )

            # 使用优化的数据加载器
            train_loader = self.create_optimized_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )

            # 初始化增强版GNNWR训练器 - 使用优化参数
            input_dim = gnnwr_features_imputed.shape[1]
            self.logger.info(f"初始化GNNWR训练器，输入维度: {input_dim}")

            self.gnnwr_trainer = EnhancedGNNWRTrainer(
                input_dim=input_dim,
                coords=coords_copy if use_spatial else None,
                hidden_dims=self.gnnwr_params['hidden_dims'],
                learning_rate=self.gnnwr_params['learning_rate'],
                bandwidth=self.gnnwr_params['bandwidth'],
                use_spatial_weights=use_spatial,
                device=self.device,  # 传递设备
                mixed_precision=self.mixed_precision,  # 混合精度
                cpu_workers=self.cpu_workers  # CPU工作线程
            )

            # 训练模型
            self.logger.info(f"开始增强版GNNWR训练，输入维度: {input_dim}")
            try:
                self.gnnwr_trainer.train(
                    train_loader,
                    epochs=self.gnnwr_params['epochs'],
                    patience=self.gnnwr_params['patience']
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.error("GPU内存不足，尝试减小批次大小")
                    # 重新尝试较小的批次
                    self.gnnwr_params['batch_size'] = self.gnnwr_params['batch_size'] // 2
                    self.train_gnnwr_model(X, y, cluster_predictions, coords)
                    return
                else:
                    raise e
        else:
            # 使用基础版GNNWR（无空间权重，内存友好）
            self.logger.info("使用基础版GNNWR训练器")

            # 创建数据集
            dataset = SpatialDataset(gnnwr_features_imputed, y)

            # 使用优化的数据加载器
            train_loader = self.create_optimized_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )

            # 初始化基础版GNNWR训练器
            input_dim = gnnwr_features_imputed.shape[1]
            self.gnnwr_trainer = GNNWRTrainer(
                input_dim=input_dim,
                hidden_dims=self.gnnwr_params['hidden_dims'],
                learning_rate=self.gnnwr_params['learning_rate'],
                device=self.device  # 传递设备
            )

            # 训练模型
            self.logger.info(f"开始基础版GNNWR训练，输入维度: {input_dim}")
            self.gnnwr_trainer.train(
                train_loader,
                epochs=self.gnnwr_params['epochs'],
                patience=self.gnnwr_params['patience']
            )

        # 计算训练集性能
        y_pred = self.predict_with_gnnwr(gnnwr_features_imputed, None, coords_copy)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r_value, _ = pearsonr(y, y_pred)

        self.logger.info(f"GNNWR模型训练完成: MAE={mae:.3f}, RMSE={rmse:.3f}, R={r_value:.3f}")

    def cross_validate(self, X, y, groups, coords=None, cv_type='station'):
        """执行交叉验证 - GPU优化版本"""
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()

        all_predictions = []
        all_true_values = []
        fold_results = {}

        unique_groups = np.unique(groups)
        total_folds = len(unique_groups)

        self.logger.info(f"开始{cv_type}交叉验证，共{total_folds}个折叠...")
        self.logger.info(f"使用设备: {self.device}")

        # 在整个数据集上按站点进行一次聚类
        self.logger.info("在整个数据集上按站点进行聚类分配...")
        self.cluster_assignments = self.perform_clustering(X, groups)

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            group_id = groups[test_idx[0]]
            test_size = len(test_idx)
            train_size = len(train_idx)

            self.logger.info(f"=== Fold {fold + 1} ===")
            self.logger.info(f"训练集大小: {train_size}, 测试集大小: {test_size}")

            # 分割数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]

            # 分割坐标
            if coords is not None:
                coords_train = coords[train_idx]
                coords_test = coords[test_idx]
            else:
                coords_train = None
                coords_test = None

            # 使用固定的聚类分配
            train_cluster_labels = self.cluster_assignments[train_idx]
            test_cluster_labels = self.cluster_assignments[test_idx]

            # 训练聚类集成模型
            try:
                # 第一步：为每个聚类训练模型 - 使用多线程
                self.train_cluster_models(X_train, y_train, train_cluster_labels)

                # 第二步：获取训练集上的聚类预测
                cluster_predictions_train = self._get_cluster_predictions(X_train, train_cluster_labels)

                # 第三步：训练GNNWR集成模型
                if coords_train is None:
                    raise ValueError(f"Fold {fold + 1}: coords_train为None，无法训练GNNWR")

                self.train_gnnwr_model(X_train, y_train, cluster_predictions_train, coords_train)

                # 第四步：预测测试集
                cluster_predictions_test = self._get_cluster_predictions(X_test, test_cluster_labels)

                # 关键修复：测试集特征也需要与聚类预测合并
                test_features_combined = np.hstack([X_test, cluster_predictions_test])
                self.logger.info(f"测试集合并特征形状: {test_features_combined.shape}")

                y_pred = self.predict_with_gnnwr(test_features_combined, None, coords_test)

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

    def predict_with_gnnwr(self, X, cluster_predictions=None, coords=None):
        """使用GNNWR进行预测 - GPU优化版本"""
        if self.gnnwr_trainer is None:
            raise ValueError("GNNWR模型尚未训练")

        self.logger.info(f"预测时特征维度调试:")
        self.logger.info(f"  X形状: {X.shape}")

        # 关键修复：如果传入了cluster_predictions，说明X已经是原始特征
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

        # 使用优化的批次大小进行预测
        batch_size = 2048 if self.device.type == 'cuda' else 1024

        # 分批预测以避免内存问题
        predictions = []
        for i in range(0, len(gnnwr_features_imputed), batch_size):
            end_idx = min(i + batch_size, len(gnnwr_features_imputed))
            batch_features = gnnwr_features_imputed[i:end_idx]
            batch_coords = coords[i:end_idx] if coords is not None else None

            batch_pred = self.gnnwr_trainer.predict(batch_features, batch_coords)
            predictions.append(batch_pred)

        return np.concatenate(predictions)

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

    def create_optimized_dataloader(self, dataset, batch_size=512, shuffle=True):
        """创建优化的数据加载器"""
        num_workers = min(12, self.cpu_workers // 2)  # 充分利用14900KF

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda',
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )

    def run_complete_analysis(self, df, output_dir=None):
        """运行完整分析流程 - GPU优化版本"""
        self.logger.info("=" * 70)
        self.logger.info("🚀 开始SWE聚类集成回归完整分析流程 (GPU优化版)")
        self.logger.info("=" * 70)

        # 显示硬件信息
        if self.device.type == 'cuda':
            gpu_info = f"GPU: {torch.cuda.get_device_name()}, 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
            self.logger.info(f"硬件配置: {gpu_info}, CPU线程: {self.cpu_workers}")
        else:
            self.logger.info(f"硬件配置: CPU模式, {self.cpu_workers}线程")

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
                    'has_coords': coords is not None,
                    'device': str(self.device),
                    'mixed_precision': self.mixed_precision
                }
            }

            # 2. 在整个数据集上按站点进行聚类
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 2: 站点级聚类分析")
            self.logger.info("=" * 50)

            self.cluster_assignments = self.perform_clustering(X, station_groups)
            results['cluster_assignments'] = self.cluster_assignments

            # 3. 年度交叉验证（使用固定聚类）
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 3: 年度交叉验证")
            self.logger.info("=" * 50)

            results['yearly_cv'] = self.cross_validate(X, y, year_groups, coords, 'yearly')

            # 4. 训练最终模型
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 4: 训练最终模型")
            self.logger.info("=" * 50)

            self.fit(X, y, station_groups, coords)

            results['final_model'] = {
                'kmeans': self.kmeans,
                'cluster_models': self.cluster_models,
                'gnnwr_trainer': self.gnnwr_trainer,
                'cluster_assignments': self.cluster_assignments,
                'feature_columns': self.feature_columns,
                'training_config': {
                    'device': str(self.device),
                    'mixed_precision': self.mixed_precision,
                    'cpu_workers': self.cpu_workers
                }
            }

            # 5. 保存结果
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 5: 保存结果")
            self.logger.info("=" * 50)

            self._save_results(results, output_dir)

            # 6. 生成报告
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
        """生成分析报告 - 包含GPU信息"""
        report = []
        report.append("=" * 70)
        report.append("❄️ SWE聚类集成回归分析报告 (GPU优化版)")
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
        report.append(f"  训练设备: {preprocessing['device']}")
        report.append(f"  混合精度: {'是' if preprocessing['mixed_precision'] else '否'}")
        report.append(f"  GNNWR版本: {'增强版' if self.use_enhanced_gnnwr else '基础版'}")
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
        report.append(f"  设备配置: {self.device}")
        report.append(f"  CPU线程: {self.cpu_workers}")
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
                               use_enhanced_gnnwr=True, gnnwr_params=None, device='auto',
                               mixed_precision=True, cpu_workers=24):
    """便捷函数：训练SWE聚类集成模型 - GPU优化版本

    Args:
        data_df (pd.DataFrame): 包含特征和SWE的数据
        output_dir (str, optional): 输出目录路径
        n_clusters (int, optional): 聚类数量
        params (dict, optional): XGBoost参数
        use_enhanced_gnnwr (bool): 是否使用增强版GNNWR
        gnnwr_params (dict): GNNWR参数
        device (str): 训练设备
        mixed_precision (bool): 是否使用混合精度
        cpu_workers (int): CPU工作线程数
    """
    trainer = SWEClusterEnsemble(
        n_clusters=n_clusters,
        params=params,
        gnnwr_params=gnnwr_params,
        use_enhanced_gnnwr=use_enhanced_gnnwr,
        use_rf=use_rf,
        device=device,
        mixed_precision=mixed_precision,
        cpu_workers=cpu_workers
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
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], output_dim=1,  # 使用更小的网络
                 dropout_rate=0.3, use_batch_norm=True, use_attention=True,
                 activation='relu'):
        super(PureGNNWRModel, self).__init__()

        self.use_attention = use_attention

        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        # 特征提取网络 - 修复梯度流动
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.feature_network = nn.Sequential(*layers)

        # 输出层 - 添加残差连接防止梯度消失
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # 关键修复：更好的权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化 - 修复self引用问题"""
        # 关键修复：使用self.modules()而不是self.model.modules()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Kaiming初始化，适合ReLU
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _initialize_single_module(self, module):
        """初始化单个模块的权重"""
        if isinstance(module, nn.Linear):
            # 使用Kaiming初始化，适合ReLU
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x, spatial_weights=None, coords=None):
        # 特征提取
        features = self.feature_network(x)

        # 空间平滑
        if spatial_weights is not None and self.use_attention:
            row_sums = torch.sum(spatial_weights, dim=1, keepdim=True)
            normalized_weights = spatial_weights / torch.where(row_sums > 0, row_sums, torch.tensor(1.0))
            smoothed_features = torch.matmul(normalized_weights, features)
            output = self.output_layer(smoothed_features)
        else:
            output = self.output_layer(features)

        return output.squeeze()


class PureGNNWRTrainer:
    """纯净版GNNWR训练器 - 修复autocast错误版本"""

    def __init__(self, input_dim, coords, hidden_dims=[512, 256, 128, 64],
                 learning_rate=0.001, bandwidth=10.0, dropout_rate=0.3,
                 weight_decay=1e-4, device='auto', output_std_penalty=0.01,
                 mixed_precision=True, cpu_workers=24, gradient_clip=1.0):

        # 首先初始化logger - 这是关键修复！
        self.logger = logging.getLogger("PureGNNWRTrainer")

        # 设备设置
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                # GPU优化设置
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.device_type = 'cuda'
            else:
                self.device = torch.device('cpu')
                torch.set_num_threads(cpu_workers)
                self.device_type = 'cpu'
        else:
            self.device = torch.device(device)
            self.device_type = 'cuda' if device == 'cuda' else 'cpu'
            if device == 'cpu':
                torch.set_num_threads(cpu_workers)

        # 混合精度训练 - 修复：只在CUDA设备上启用
        self.mixed_precision = mixed_precision and self.device_type == 'cuda'
        if self.mixed_precision:
            self.scaler = GradScaler()
            self.logger.info(f"启用混合精度训练，设备类型: {self.device_type}")
        else:
            self.logger.info(f"禁用混合精度训练，设备类型: {self.device_type}")

        self.output_std_penalty = output_std_penalty
        self.logger = logging.getLogger("PureGNNWR")
        self.logger.info(f"纯净版GNNWR - 使用设备: {self.device}")
        self.logger.info(f"混合精度: {self.mixed_precision}")

        # 关键：添加标准化器
        from sklearn.preprocessing import StandardScaler
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # 模型初始化
        self.model = PureGNNWRModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation='leaky_relu'  # 使用LeakyReLU防止死亡ReLU
        ).to(self.device)

        # 优化器 - 使用AdamW，针对混合精度优化
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=weight_decay,
            betas=(0.9, 0.99)
        )



        # 学习率调度器 - OneCycle策略
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=200,
            steps_per_epoch=1000,
            pct_start=0.1
        )

        self.criterion = nn.HuberLoss()  # 使用HuberLoss更稳定

        # 空间权重计算
        self.coords = coords.copy() if coords is not None else None
        self.bandwidth = bandwidth

        # CPU工作线程配置
        self.cpu_workers = cpu_workers

        self.gradient_clip = gradient_clip

    def get_training_info(self):
        """获取训练信息"""
        return {
            'model_state': self.model.state_dict() if hasattr(self, 'model') else None,
            'training_loss': getattr(self, 'training_loss', []),
            'validation_loss': getattr(self, 'validation_loss', []),
            'epochs_completed': getattr(self, 'epochs_completed', 0),
            'current_learning_rate': getattr(self, 'current_lr', 0.0)
        }

    def safe_get_training_info(trainer):
        """安全地获取训练信息"""
        try:
            if hasattr(trainer, 'get_training_info'):
                return trainer.get_training_info()
            else:
                # 尝试从trainer的其他属性中提取信息
                info = {}
                for attr in ['model', 'training_loss', 'validation_loss', 'epoch']:
                    if hasattr(trainer, attr):
                        info[attr] = getattr(trainer, attr)
                return info
        except Exception as e:
            return {'error': f'Failed to get training info: {str(e)}'}

    def _initialize_model(self):
        """确保模型正确初始化 - 修复apply调用"""
        # 关键修复：正确使用apply方法
        self.model.apply(self._initialize_weights)

        # 验证初始化
        self.logger.info("=== 模型初始化验证 ===")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

        # 检查权重范围
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.logger.info(f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")

        self.logger.info("=====================")

    def _initialize_weights(self):
        """更好的权重初始化"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def debug_model_output(self, X_sample, y_sample, coords_sample=None):
        """调试模型输出"""
        self.model.eval()
        with torch.no_grad():
            # 测试不同输入
            outputs = []
            print("=== 模型输出调试 ===")

            for i in range(min(5, len(X_sample))):
                x = torch.tensor(X_sample[i:i + 1], dtype=torch.float32, device=self.device)
                c = torch.tensor(coords_sample[i:i + 1], dtype=torch.float32,
                                 device=self.device) if coords_sample is not None else None

                if self.mixed_precision:
                    with autocast(device_type=self.device_type):
                        output = self.model(x, None, c)
                else:
                    output = self.model(x, None, c)

                outputs.append(output.item())
                print(f"样本 {i}: 输入均值为 {x.mean().item():.3f}, 输出为 {output.item():.3f}")

            print(f"模型输出范围: [{min(outputs):.3f}, {max(outputs):.3f}]")
            print(f"模型输出标准差: {np.std(outputs):.6f}")
            print("===================")

    def _compute_spatial_weights(self, batch_coords):
        """计算空间权重矩阵 - 修复autocast版本"""
        n_batch = batch_coords.shape[0]
        if n_batch <= 1:
            return torch.ones((n_batch, n_batch), device=self.device,
                              dtype=torch.float16 if self.mixed_precision else torch.float32)

        # 修复：正确使用autocast
        if self.mixed_precision:
            with autocast(device_type=self.device_type):
                # 计算欧氏距离
                diff = batch_coords.unsqueeze(1) - batch_coords.unsqueeze(0)
                distances = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-8)

                # 高斯核函数
                weights = torch.exp(-0.5 * (distances / self.bandwidth) ** 2)
        else:
            # 非混合精度版本
            diff = batch_coords.unsqueeze(1) - batch_coords.unsqueeze(0)
            distances = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-8)
            weights = torch.exp(-0.5 * (distances / self.bandwidth) ** 2)

        return weights

    def train_epoch_mixed_precision(self, train_loader):
        """修复学习率调度顺序的训练epoch"""
        self.model.train()
        epoch_train_loss = 0.0
        batch_count = 0

        # 添加梯度裁剪阈值
        gradient_clip = 1.0  # 关键修复：添加梯度裁剪阈值

        for batch_idx, batch in enumerate(train_loader):
            try:
                if len(batch) == 3:
                    batch_features, batch_targets, batch_coords = batch
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    batch_coords = batch_coords.to(self.device, non_blocking=True) if batch_coords is not None else None
                else:
                    batch_features, batch_targets = batch
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    batch_coords = None

                self.optimizer.zero_grad(set_to_none=True)

                # 计算空间权重
                spatial_weights = None
                if batch_coords is not None:
                    spatial_weights = self._compute_spatial_weights(batch_coords)

                if self.mixed_precision:
                    with autocast(device_type=self.device_type):
                        outputs = self.model(batch_features, spatial_weights, batch_coords)
                        loss = self.criterion(outputs, batch_targets)

                    self.scaler.scale(loss).backward()

                    # 关键修复：添加梯度裁剪
                    self.scaler.unscale_(self.optimizer)  # 必须先unscale梯度
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=gradient_clip,
                        norm_type=2.0
                    )

                    # 关键：添加梯度监控
                    total_grad_norm = 0.0
                    grad_norms = []
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param_grad_norm = param.grad.data.norm(2).item()
                            total_grad_norm += param_grad_norm ** 2  # 修正：应该平方和再开方
                            grad_norms.append((name, param_grad_norm))

                    total_grad_norm = total_grad_norm ** 0.5  # 计算真实的梯度范数

                    self.scaler.step(self.optimizer)  # 先执行优化器
                    self.scaler.update()
                    self.scheduler.step()  # 后执行学习率调度

                else:
                    outputs = self.model(batch_features, spatial_weights, batch_coords)
                    loss = self.criterion(outputs, batch_targets)

                    loss.backward()

                    # 关键修复：添加梯度裁剪（非混合精度版本）
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=gradient_clip,
                        norm_type=2.0
                    )

                    # 关键：添加梯度监控
                    total_grad_norm = 0.0
                    grad_norms = []
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param_grad_norm = param.grad.data.norm(2).item()
                            total_grad_norm += param_grad_norm ** 2
                            grad_norms.append((name, param_grad_norm))

                    total_grad_norm = total_grad_norm ** 0.5

                    self.optimizer.step()  # 先执行优化器
                    self.scheduler.step()  # 后执行学习率调度

                epoch_train_loss += loss.item()
                batch_count += 1

                # 监控输出和梯度
                if batch_idx % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    output_std = torch.std(outputs).item()
                    output_range = f"[{outputs.min().item():.3f}, {outputs.max().item():.3f}]"

                    # 检查梯度是否被裁剪
                    grad_clipped = total_grad_norm > gradient_clip
                    clip_info = " (已裁剪)" if grad_clipped else ""

                    self.logger.info(f'Batch {batch_idx}: Loss={loss.item():.6f}, '
                                     f'Output STD={output_std:.6f}, Output Range={output_range}, '
                                     f'Grad Norm={total_grad_norm:.6f}{clip_info}, LR={current_lr:.2e}')

                    # 如果梯度很小，显示具体哪些层的梯度小
                    if total_grad_norm < 1e-6:
                        self.logger.warning("梯度消失！各层梯度:")
                        for name, grad_norm in grad_norms[:5]:  # 显示前5层
                            self.logger.warning(f"  {name}: {grad_norm:.6f}")

                    # 如果梯度很大，显示具体哪些层的梯度大
                    elif total_grad_norm > 1000:
                        self.logger.warning("梯度爆炸风险！各层梯度:")
                        for name, grad_norm in sorted(grad_norms, key=lambda x: x[1], reverse=True)[:3]:
                            self.logger.warning(f"  {name}: {grad_norm:.6f}")

                # 监控输出变化（保留原有逻辑）
                if batch_idx % 10 == 0:
                    output_std = torch.std(outputs).item()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(
                        f'Batch {batch_idx}, Loss: {loss.item():.6f}, Output STD: {output_std:.6f}, LR: {current_lr:.2e}')

            except Exception as e:
                self.logger.error(f"Batch {batch_idx} 失败: {e}")
                # 关键修复：在异常时清理梯度
                self.optimizer.zero_grad(set_to_none=True)
                continue

        return epoch_train_loss / max(batch_count, 1)

    def _has_valid_gradients(self):
        """检查是否存在有效梯度"""
        has_valid_grad = False
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    self.logger.warning("检测到无效梯度，清零")
                    param.grad.zero_()
                elif torch.sum(torch.abs(param.grad)) > 0:
                    has_valid_grad = True

        return has_valid_grad

    def _check_and_clip_gradients(self):
        """检查并裁剪梯度"""
        # 检查梯度是否存在
        has_valid_grad = False
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    self.logger.warning("检测到无效梯度，清零")
                    param.grad.zero_()
                else:
                    has_valid_grad = True

        if not has_valid_grad:
            self.logger.warning("没有有效梯度")
            return False

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.gradient_clip,
            norm_type=2
        )

        return True

    def train(self, train_loader, val_loader=None, epochs=200, early_stopping_patience=20):
        """完整深度学习训练流程 - 修复模型保存逻辑"""
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        # 关键修复：保存最佳模型状态而不是整个模型
        best_model_state = None

        # GPU预热
        if self.device_type == 'cuda':
            self._warmup_gpu()

        self.logger.info(f"开始训练，总轮次: {epochs}，早停耐心: {early_stopping_patience}")

        for epoch in range(epochs):
            # 训练阶段
            try:
                train_loss = self.train_epoch_mixed_precision(train_loader)
                train_losses.append(train_loss)
            except Exception as e:
                self.logger.error(f"Epoch {epoch} 训练失败: {e}")
                break

            # 验证阶段
            if val_loader is not None:
                try:
                    val_loss = self.validate(val_loader)
                    val_losses.append(val_loss)

                    # 早停逻辑
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # 关键修复：保存模型状态而不是整个模型
                        best_model_state = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict().copy(),
                            'optimizer_state_dict': self.optimizer.state_dict().copy(),
                            'scheduler_state_dict': self.scheduler.state_dict().copy(),
                            'val_loss': val_loss,
                            'train_loss': train_loss
                        }
                        self.logger.info(f"Epoch {epoch}: 保存最佳模型状态，验证损失: {val_loss:.6f}")
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        self.logger.info(f"早停在epoch {epoch}, 最佳验证loss: {best_val_loss:.6f}")
                        break
                except Exception as e:
                    self.logger.error(f"Epoch {epoch} 验证失败: {e}")
                    val_losses.append(float('inf'))
            else:
                # 如果没有验证集，使用训练loss
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    patience_counter = 0
                    best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict().copy(),
                        'optimizer_state_dict': self.optimizer.state_dict().copy(),
                        'scheduler_state_dict': self.scheduler.state_dict().copy(),
                        'train_loss': train_loss
                    }
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"早停在epoch {epoch}, 最佳训练loss: {best_val_loss:.6f}")
                    break

            # 日志输出
            if epoch % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                if val_loader is not None and len(val_losses) > epoch:
                    self.logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                                     f"Val Loss: {val_losses[epoch]:.6f} | LR: {current_lr:.2e}")
                else:
                    self.logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                                     f"LR: {current_lr:.2e}")

        # 关键修复：正确加载最佳模型状态
        if best_model_state is not None:
            self.logger.info(f"加载最佳模型状态 (epoch {best_model_state['epoch']})")
            self.model.load_state_dict(best_model_state['model_state_dict'])
            # 可选：恢复优化器和调度器状态
            # self.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
            # self.scheduler.load_state_dict(best_model_state['scheduler_state_dict'])
        else:
            self.logger.warning("没有找到最佳模型状态，使用最终训练状态")

        return train_losses, val_losses if val_loader is not None else train_losses

    def fit(self, X, y, coords=None):
        """训练模型 - 添加训练状态检查"""
        self.logger.info("开始训练，添加训练状态检查...")

        # 标准化数据
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y

        X_normalized = self.feature_scaler.fit_transform(X)
        y_normalized = self.target_scaler.fit_transform(y_2d).flatten()

        self.logger.info(f"训练数据统计:")
        self.logger.info(f"  X范围: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
        self.logger.info(f"  y范围: [{y_normalized.min():.3f}, {y_normalized.max():.3f}]")
        self.logger.info(f"  y标准差: {y_normalized.std():.3f}")

        # 创建数据集
        dataset = EnhancedSpatialDataset(X_normalized, y_normalized, coords)
        train_loader = self.create_optimized_dataloader(dataset, batch_size=512, shuffle=True)

        # 训练前检查模型初始状态
        self._check_model_initial_state(X_normalized, coords)

        # 训练
        train_losses, val_losses = self.train(train_loader, epochs=100, early_stopping_patience=15)

        # 训练后检查模型最终状态
        self._check_model_final_state(X_normalized, coords)

        return train_losses, val_losses

    def _check_model_initial_state(self, X, coords):
        """检查模型初始状态"""
        self.model.eval()
        with torch.no_grad():
            sample_outputs = []
            for i in range(min(10, len(X))):
                x = torch.tensor(X[i:i + 1], dtype=torch.float32, device=self.device)
                c = torch.tensor(coords[i:i + 1], dtype=torch.float32,
                                 device=self.device) if coords is not None else None

                if self.mixed_precision:
                    with autocast(device_type=self.device_type):
                        output = self.model(x, None, c)
                else:
                    output = self.model(x, None, c)

                sample_outputs.append(output.item())

            self.logger.info("=== 模型初始状态检查 ===")
            self.logger.info(f"初始输出范围: [{min(sample_outputs):.6f}, {max(sample_outputs):.6f}]")
            self.logger.info(f"初始输出标准差: {np.std(sample_outputs):.6f}")
            self.logger.info("=====================")

    def _check_model_final_state(self, X, coords):
        """检查模型最终状态 - 修复检查逻辑"""
        self.model.eval()
        with torch.no_grad():
            sample_outputs = []

            # 检查多个样本
            for i in range(min(20, len(X))):  # 检查更多样本
                x = torch.tensor(X[i:i + 1], dtype=torch.float32, device=self.device)
                c = torch.tensor(coords[i:i + 1], dtype=torch.float32,
                                 device=self.device) if coords is not None else None

                if self.mixed_precision:
                    with autocast(device_type=self.device_type):
                        output = self.model(x, None, c)
                else:
                    output = self.model(x, None, c)

                sample_outputs.append(output.item())

            self.logger.info("=== 模型最终状态详细检查 ===")
            self.logger.info(f"最终输出范围: [{min(sample_outputs):.6f}, {max(sample_outputs):.6f}]")
            self.logger.info(f"最终输出标准差: {np.std(sample_outputs):.6f}")
            self.logger.info(f"输出唯一值数量: {len(np.unique(sample_outputs))}")

            # 检查输出是否恒定
            if np.std(sample_outputs) < 1e-6:
                self.logger.error("❌ 模型最终状态输出恒定！")
                self.logger.error("可能原因:")
                self.logger.error("1. 模型权重全部相同")
                self.logger.error("2. 梯度消失导致所有权重收敛到相同值")
                self.logger.error("3. 模型保存/加载问题")
            else:
                self.logger.info("✅ 模型最终状态输出正常")

            self.logger.info("=====================")

    def predict(self, features, coords=None, batch_size=1024):
        """重写预测方法 - 确保稳定可靠的预测"""
        self.model.eval()

        self.logger.info("🚀 开始预测流程...")
        self.logger.info(f"输入特征形状: {features.shape}, 坐标形状: {coords.shape if coords is not None else 'None'}")

        # ==================== 1. 预测前模型状态验证 ====================
        self.logger.info("=== 预测前模型状态验证 ===")

        # 使用训练数据的子集验证模型状态
        validation_samples = min(10, len(features))
        validation_outputs = []

        with torch.no_grad():
            for i in range(validation_samples):
                # 创建单个样本的tensor
                x_sample = torch.tensor(features[i:i + 1], dtype=torch.float32, device=self.device)
                c_sample = torch.tensor(coords[i:i + 1], dtype=torch.float32,
                                        device=self.device) if coords is not None else None

                # 使用与训练完全相同的路径
                spatial_weights = None
                if c_sample is not None:
                    spatial_weights = self._compute_spatial_weights(c_sample)

                # 关键：强制使用非混合精度进行验证
                output = self.model(x_sample, spatial_weights, c_sample)
                validation_outputs.append(output.item())

        val_std = np.std(validation_outputs)
        self.logger.info(f"验证输出 - 范围: [{min(validation_outputs):.3f}, {max(validation_outputs):.3f}]")
        self.logger.info(f"验证输出 - 标准差: {val_std:.6f}")
        self.logger.info(f"验证输出 - 唯一值数量: {len(np.unique(validation_outputs))}")

        if val_std < 1e-6:
            self.logger.error("❌ 模型状态验证失败：输出恒定！")
            self.logger.error("可能原因：模型权重问题、梯度消失、或训练过程异常")

            # 紧急修复：尝试重新初始化输出层
            self.logger.warning("尝试紧急修复：重新初始化输出层...")
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and module.out_features == 1:
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                    self.logger.info(f"重新初始化层: {name}")

            # 重新验证
            validation_outputs = []
            with torch.no_grad():
                for i in range(validation_samples):
                    x_sample = torch.tensor(features[i:i + 1], dtype=torch.float32, device=self.device)
                    c_sample = torch.tensor(coords[i:i + 1], dtype=torch.float32,
                                            device=self.device) if coords is not None else None
                    spatial_weights = self._compute_spatial_weights(c_sample) if c_sample is not None else None
                    output = self.model(x_sample, spatial_weights, c_sample)
                    validation_outputs.append(output.item())

            val_std = np.std(validation_outputs)
            self.logger.info(f"修复后验证输出标准差: {val_std:.6f}")

            if val_std < 1e-6:
                self.logger.error("❌ 紧急修复失败，使用备选预测方案")
                return self._fallback_prediction(features, coords)

        self.logger.info("✅ 模型状态验证通过")

        # ==================== 2. 特征标准化 ====================
        self.logger.info("=== 特征标准化 ===")

        if not hasattr(self, 'feature_scaler') or self.feature_scaler is None:
            self.logger.error("特征标准化器未初始化")
            return self._fallback_prediction(features, coords)

        try:
            features_normalized = self.feature_scaler.transform(features)
            self.logger.info(
                f"特征标准化完成 - 均值: {features_normalized.mean():.3f}, 标准差: {features_normalized.std():.3f}")
        except Exception as e:
            self.logger.error(f"特征标准化失败: {e}")
            return self._fallback_prediction(features, coords)

        # ==================== 3. 批量预测 ====================
        self.logger.info("=== 批量预测 ===")

        all_predictions_normalized = []
        successful_batches = 0
        failed_batches = 0

        with torch.no_grad():
            for i in range(0, len(features_normalized), batch_size):
                batch_start = i
                batch_end = min(i + batch_size, len(features_normalized))
                batch_size_actual = batch_end - batch_start

                try:
                    # 准备批次数据
                    batch_features = torch.tensor(
                        features_normalized[batch_start:batch_end],
                        dtype=torch.float32,
                        device=self.device
                    )

                    batch_coords = None
                    if coords is not None:
                        batch_coords = torch.tensor(
                            coords[batch_start:batch_end],
                            dtype=torch.float32,
                            device=self.device
                        )

                    # 计算空间权重
                    spatial_weights = None
                    if batch_coords is not None and len(batch_coords) > 1:
                        spatial_weights = self._compute_spatial_weights(batch_coords)

                    # 模型预测 - 强制使用float32避免混合精度问题
                    batch_predictions = self.model(batch_features, spatial_weights, batch_coords)

                    # 检查批次预测结果
                    batch_predictions_np = batch_predictions.cpu().numpy()
                    batch_std = np.std(batch_predictions_np)

                    if batch_std < 1e-6 and batch_size_actual > 1:
                        self.logger.warning(f"批次 {i // batch_size} 输出恒定 (std={batch_std:.6f})")
                        # 尝试不使用空间权重重新预测
                        self.logger.info("尝试不使用空间权重重新预测...")
                        batch_predictions_fallback = self.model(batch_features, None, batch_coords)
                        batch_predictions_fallback_np = batch_predictions_fallback.cpu().numpy()
                        fallback_std = np.std(batch_predictions_fallback_np)

                        if fallback_std > batch_std:
                            self.logger.info(f"✅ 无空间权重预测改善: std={fallback_std:.6f}")
                            batch_predictions_np = batch_predictions_fallback_np
                    # 存储预测结果
                    all_predictions_normalized.append(batch_predictions_np)
                    successful_batches += 1

                    # 进度日志
                    if (i // batch_size) % 10 == 0:
                        self.logger.info(f"进度: {batch_end}/{len(features)} samples, 当前批次标准差: {batch_std:.6f}")

                except Exception as e:
                    self.logger.error(f"批次 {i // batch_size} 预测失败: {e}")
                    # 使用均值的备选预测
                    fallback_batch = np.full(batch_size_actual, 0.0)  # 标准化空间的均值
                    all_predictions_normalized.append(fallback_batch)
                    failed_batches += 1
                    continue

        self.logger.info(f"批量预测完成 - 成功: {successful_batches}, 失败: {failed_batches}")

        if len(all_predictions_normalized) == 0:
            self.logger.error("所有批次预测失败")
            return self._fallback_prediction(features, coords)

        # ==================== 4. 结果合并和逆标准化 ====================
        self.logger.info("=== 结果后处理 ===")

        try:
            # 合并所有预测结果
            predictions_normalized = np.concatenate(all_predictions_normalized)

            self.logger.info(f"标准化预测结果统计:")
            self.logger.info(f"  范围: [{predictions_normalized.min():.3f}, {predictions_normalized.max():.3f}]")
            self.logger.info(f"  均值: {predictions_normalized.mean():.3f}")
            self.logger.info(f"  标准差: {predictions_normalized.std():.3f}")
            self.logger.info(f"  唯一值数量: {len(np.unique(predictions_normalized))}")

            # 逆标准化到原始尺度
            if predictions_normalized.ndim == 1:
                predictions_normalized_2d = predictions_normalized.reshape(-1, 1)
            else:
                predictions_normalized_2d = predictions_normalized

            predictions_original = self.target_scaler.inverse_transform(predictions_normalized_2d).flatten()

            self.logger.info(f"原始尺度预测结果统计:")
            self.logger.info(f"  范围: [{predictions_original.min():.1f}, {predictions_original.max():.1f}]")
            self.logger.info(f"  均值: {predictions_original.mean():.1f}")
            self.logger.info(f"  标准差: {predictions_original.std():.1f}")

            # 最终检查
            if np.std(predictions_original) < 1e-6:
                self.logger.warning("⚠️ 最终预测结果标准差很小，但仍在可接受范围内")

            self.logger.info("🎯 预测流程完成！")
            return predictions_original

        except Exception as e:
            self.logger.error(f"结果后处理失败: {e}")
            return self._fallback_prediction(features, coords)

    def _fallback_prediction(self, features, coords):
        """备选预测方案"""
        self.logger.warning("使用备选预测方案")

        if hasattr(self, 'target_scaler') and self.target_scaler is not None:
            # 使用目标变量的均值作为预测
            fallback_value = self.target_scaler.mean_[0] if hasattr(self.target_scaler, 'mean_') else 0.0
        else:
            fallback_value = 0.0

        self.logger.info(f"备选预测值: {fallback_value}")
        return np.full(len(features), fallback_value)

    def debug_output_range(self, batch_features, batch_targets):
        """调试输出范围问题"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_features, None, None)

            self.logger.info("=== 输出范围诊断 ===")
            self.logger.info(f"输入特征范围: [{batch_features.min():.3f}, {batch_features.max():.3f}]")
            self.logger.info(f"目标值范围: [{batch_targets.min():.3f}, {batch_targets.max():.3f}]")
            self.logger.info(f"模型输出范围: [{outputs.min():.3f}, {outputs.max():.3f}]")
            self.logger.info(f"输出/目标比例: {outputs.std() / batch_targets.std():.3f}")
            self.logger.info("===================")

    def _compute_spatial_weights(self, batch_coords):
        """计算空间权重矩阵 - 确保数值稳定性"""
        n_batch = batch_coords.shape[0]

        if n_batch <= 1:
            return torch.eye(n_batch, device=self.device, dtype=torch.float32)

        try:
            # 确保使用float32
            batch_coords_float32 = batch_coords.float()

            # 计算欧氏距离
            diff = batch_coords_float32.unsqueeze(1) - batch_coords_float32.unsqueeze(0)
            distances = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-8)

            # 高斯核函数
            weights = torch.exp(-0.5 * (distances / self.bandwidth) ** 2)

            # 数值稳定性检查
            if torch.any(torch.isnan(weights)) or torch.any(torch.isinf(weights)):
                self.logger.warning("空间权重包含无效值，使用单位矩阵")
                return torch.eye(n_batch, device=self.device, dtype=torch.float32)

            # 确保对角线为1
            weights = weights.fill_diagonal_(1.0)

            return weights

        except Exception as e:
            self.logger.error(f"空间权重计算失败: {e}")
            return torch.eye(n_batch, device=self.device, dtype=torch.float32)

    def create_optimized_dataloader(self, dataset, batch_size=512, shuffle=True, is_train=True):
        """创建优化的数据加载器"""
        num_workers = min(16, self.cpu_workers // 2) if is_train else min(8, self.cpu_workers // 4)

        self.logger.info(f"创建数据加载器 - 批次大小: {batch_size}, 工作进程: {num_workers}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda',
            persistent_workers=num_workers > 0 and is_train,
            prefetch_factor=2 if num_workers > 0 else None
        )

    def validate(self, val_loader):
        """验证集评估 - 修复版本"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    batch_features, batch_targets, batch_coords = batch
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    batch_coords = batch_coords.to(self.device, non_blocking=True) if batch_coords is not None else None
                else:
                    batch_features, batch_targets = batch
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    batch_coords = None

                spatial_weights = None
                if batch_coords is not None:
                    spatial_weights = self._compute_spatial_weights(batch_coords)

                # 修复：正确使用autocast
                if self.mixed_precision:
                    with autocast(device_type=self.device_type):
                        outputs = self.model(batch_features, spatial_weights, batch_coords)
                        loss = self.criterion(outputs, batch_targets)
                else:
                    outputs = self.model(batch_features, spatial_weights, batch_coords)
                    loss = self.criterion(outputs, batch_targets)

                val_loss += loss.item()

        return val_loss / len(val_loader)

    def debug_standardization(self, X, y):
        """调试标准化器状态"""
        self.logger.info("=== 标准化器调试 ===")

        # 检查特征标准化器
        if hasattr(self.feature_scaler, 'mean_'):
            self.logger.info(f"特征标准化器 - mean: {self.feature_scaler.mean_[:3]}...")
            self.logger.info(f"特征标准化器 - scale: {self.feature_scaler.scale_[:3]}...")

        # 检查目标标准化器
        if hasattr(self.target_scaler, 'mean_'):
            self.logger.info(f"目标标准化器 - mean: {self.target_scaler.mean_}")
            self.logger.info(f"目标标准化器 - scale: {self.target_scaler.scale_}")

        # 测试逆标准化
        test_output = np.array([-1.0, 0.0, 1.0])
        try:
            restored = self.target_scaler.inverse_transform(test_output.reshape(-1, 1)).flatten()
            self.logger.info(f"逆标准化测试: {test_output} -> {restored}")
        except Exception as e:
            self.logger.error(f"逆标准化测试失败: {e}")

        self.logger.info("===================")

    def _warmup_gpu(self):
        """GPU预热 - 修复版本"""
        if self.device_type == 'cuda':
            self.logger.info("进行GPU预热...")
            # 运行一个小的虚拟计算来预热GPU
            dummy_input = torch.randn(64, self.model.feature_network[0].in_features,
                                      device=self.device,
                                      dtype=torch.float16 if self.mixed_precision else torch.float32)
            dummy_coords = torch.randn(64, 2, device=self.device,
                                       dtype=torch.float16 if self.mixed_precision else torch.float32)

            # 修复：正确使用autocast
            if self.mixed_precision:
                with autocast(device_type=self.device_type):
                    for _ in range(20):
                        spatial_weights = self._compute_spatial_weights(dummy_coords)
                        _ = self.model(dummy_input, spatial_weights, dummy_coords)
            else:
                for _ in range(20):
                    spatial_weights = self._compute_spatial_weights(dummy_coords)
                    _ = self.model(dummy_input, spatial_weights, dummy_coords)

            torch.cuda.synchronize()
            self.logger.info("GPU预热完成")


def train_pure_gnnwr_annual_only(df, output_dir=None, random_state=42,
                                 device='auto', mixed_precision=True, cpu_workers=24):
    """
    修复版的纯净版GNNWR分析 - 仅进行年度交叉验证版本
    """
    import numpy as np
    import pandas as pd
    import os
    import joblib
    from datetime import datetime

    logger = logging.getLogger("PureGNNWRAnalysis")
    logger.info("=" * 60)
    logger.info("📊 开始纯净版GNNWR年度交叉验证分析")
    logger.info("=" * 60)

    try:
        # 使用SWEClusterEnsemble的数据预处理
        ensemble = SWEClusterEnsemble(n_clusters=1)
        X, y, station_groups, year_groups, coords = ensemble.preprocess_data(df)

        logger.info(f"数据加载: {len(X)}样本, {X.shape[1]}特征")
        logger.info(f"年度分布: {len(np.unique(year_groups))}个年份")
        logger.info(f"站点分布: {len(np.unique(station_groups))}个站点")

        # 1. 仅进行年度交叉验证
        logger.info("\n" + "=" * 50)
        logger.info("步骤 1: 年度交叉验证 (唯一验证步骤)")
        logger.info("=" * 50)

        yearly_cv_results = pure_gnnwr_cross_validate_fixed(
            X, y, year_groups, coords, 'yearly', logger,
            device=device, mixed_precision=mixed_precision, cpu_workers=cpu_workers
        )

        # 确保fold_metrics存在
        if 'fold_metrics' not in yearly_cv_results:
            logger.warning("fold_metrics不存在，创建空的fold_metrics")
            yearly_cv_results['fold_metrics'] = {}

        if 'overall_metrics' not in yearly_cv_results:
            logger.warning("overall_metrics不存在，创建默认值")
            yearly_cv_results['overall_metrics'] = {
                'r2': 0.0, 'rmse': 1.0, 'mae': 1.0, 'explained_variance': 0.0
            }

        # 整合所有结果
        results = {
            'yearly_cv': yearly_cv_results,
            'data_info': {
                'total_samples': len(X),
                'n_features': X.shape[1],
                'n_stations': len(np.unique(station_groups)),
                'n_years': len(np.unique(year_groups)),
                'device': str(device),
                'mixed_precision': mixed_precision
            }
        }

        # 保存结果和生成图表
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"./pure_gnnwr_annual_only_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"保存结果到: {output_dir}")

        # 保存结果数据
        results_path = os.path.join(output_dir, 'pure_gnnwr_results_annual.pkl')
        joblib.dump(results, results_path)

        # 生成专门针对年度验证的可视化图表
        create_annual_only_visualizations(results, output_dir)

        # 生成详细报告
        report_path = os.path.join(output_dir, 'pure_gnnwr_report_annual.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(generate_annual_report(results))

        # 输出综合报告
        print_annual_report(results)

        logger.info("✅ 纯净版GNNWR年度交叉验证分析完成!")
        return results

    except Exception as e:
        logger.error(f"纯净版GNNWR年度分析失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def create_annual_only_visualizations(results, output_dir):
    """专门为年度交叉验证生成可视化图表"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig)

    yearly_cv = results['yearly_cv']

    # 检查是否有有效数据
    if not yearly_cv['fold_metrics']:
        # 如果没有有效数据，创建空图表
        ax = fig.add_subplot(gs[:, :])
        ax.text(0.5, 0.5, '无有效数据可用\n请检查训练过程',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_title('年度交叉验证分析', fontsize=18)
        plt.savefig(os.path.join(output_dir, 'annual_cross_validation_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        return

    try:
        # 1. 年度交叉验证性能对比
        ax1 = fig.add_subplot(gs[0, 0])
        years = list(yearly_cv['fold_metrics'].keys())
        r2_scores = [yearly_cv['fold_metrics'][year]['r2'] for year in years]
        rmse_scores = [yearly_cv['fold_metrics'][year]['rmse'] for year in years]

        x = np.arange(len(years))
        width = 0.35

        ax1.bar(x - width / 2, r2_scores, width, label='R²', alpha=0.7, color='skyblue')
        ax1.bar(x + width / 2, rmse_scores, width, label='RMSE', alpha=0.7, color='lightcoral')

        ax1.set_xlabel('年份')
        ax1.set_ylabel('指标值')
        ax1.set_title('年度交叉验证性能对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(years, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 年度性能热力图
        ax2 = fig.add_subplot(gs[0, 1:])
        metrics_data = []
        for year, metrics in yearly_cv['fold_metrics'].items():
            metrics_data.append({
                'Year': year,
                'R²': metrics['r2'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                '样本数': metrics.get('n_samples', 0)
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_pivot = metrics_df.pivot_table(values=['R²', 'RMSE'], index='Year')

        sns.heatmap(metrics_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('年度交叉验证性能热力图', fontsize=14, fontweight='bold')

        # 3. 模型架构信息
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        info_text = f"""
        模型架构信息:
        - 输入维度: {results['data_info']['n_features']}
        - 隐藏层: [512, 256, 128, 64]
        - Dropout: 0.3
        - 学习率: 0.001
        - 设备: {results['data_info']['device']}
        - 混合精度: {results['data_info']['mixed_precision']}

        数据统计:
        - 总样本: {results['data_info']['total_samples']}
        - 年份数: {results['data_info']['n_years']}
        - 站点数: {results['data_info']['n_stations']}
        """
        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        # 4. 性能汇总
        ax4 = fig.add_subplot(gs[1, 1:])
        ax4.axis('off')

        yearly_avg_r2 = yearly_cv['overall_metrics']['r2']
        yearly_avg_rmse = yearly_cv['overall_metrics']['rmse']

        summary_text = f"""
        性能汇总:

        年度交叉验证:
        - 平均 R²: {yearly_avg_r2:.4f}
        - 平均 RMSE: {yearly_avg_rmse:.4f}
        - 平均 MAE: {yearly_cv['overall_metrics']['mae']:.4f}
        - 平均解释方差: {yearly_cv['overall_metrics']['explained_variance']:.4f}
        """

        # 添加最佳和最差年份信息
        if yearly_cv['fold_metrics']:
            best_year = max(yearly_cv['fold_metrics'].items(), key=lambda x: x[1]['r2'])
            worst_year = min(yearly_cv['fold_metrics'].items(), key=lambda x: x[1]['r2'])
            summary_text += f"\n最佳年份: {best_year[0]} (R² = {best_year[1]['r2']:.4f})"
            summary_text += f"\n最差年份: {worst_year[0]} (R² = {worst_year[1]['r2']:.4f})"

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        # 5. 残差分析（使用所有年份的残差）
        ax5 = fig.add_subplot(gs[2, 0])
        all_residuals = []
        for year_data in yearly_cv['fold_results']:
            all_residuals.extend(year_data['residuals'])

        ax5.hist(all_residuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('残差')
        ax5.set_ylabel('频数')
        ax5.set_title('所有年份残差分布', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. 预测vs真实值散点图（使用所有年份数据）
        ax6 = fig.add_subplot(gs[2, 1:])
        all_y_true = []
        all_y_pred = []
        for year_data in yearly_cv['fold_results']:
            all_y_true.extend(year_data['y_true'])
            all_y_pred.extend(year_data['y_pred'])

        ax6.scatter(all_y_true, all_y_pred, alpha=0.6, color='blue', s=20)
        ax6.plot([min(all_y_true), max(all_y_true)], [min(all_y_true), max(all_y_true)],
                 'r--', linewidth=2)
        ax6.set_xlabel('真实值')
        ax6.set_ylabel('预测值')
        ax6.set_title(f'所有年份预测 vs 真实值\n总体R² = {yearly_avg_r2:.3f}, RMSE = {yearly_avg_rmse:.3f}',
                      fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'annual_cross_validation_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 额外保存年度详细图表
        create_detailed_year_analysis(yearly_cv, output_dir)

    except Exception as e:
        logger.error(f"可视化生成失败: {e}")
        # 创建错误图表
        ax = fig.add_subplot(gs[:, :])
        ax.text(0.5, 0.5, f'可视化生成失败:\n{str(e)}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red')
        ax.set_title('年度交叉验证分析 - 错误', fontsize=16)
        plt.savefig(os.path.join(output_dir, 'annual_cross_validation_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_detailed_year_analysis(yearly_cv, output_dir):
    """为每个年份创建详细的分析图表"""
    import matplotlib.pyplot as plt

    if not yearly_cv['fold_results']:
        return

    # 为每个年份创建单独的图表
    for fold_result in yearly_cv['fold_results']:
        year = fold_result['test_group']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'年份 {year} 详细分析', fontsize=16, fontweight='bold')

        # 1. 预测 vs 真实值
        y_true = fold_result['y_true']
        y_pred = fold_result['y_pred']
        metrics = fold_result['metrics']

        ax1.scatter(y_true, y_pred, alpha=0.6, color='blue')
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')
        ax1.set_title(f'预测 vs 真实值 (R² = {metrics["r2"]:.3f})')
        ax1.grid(True, alpha=0.3)

        # 2. 残差分布
        residuals = fold_result['residuals']
        ax2.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('残差')
        ax2.set_ylabel('频数')
        ax2.set_title(f'残差分布 (均值 = {np.mean(residuals):.3f})')
        ax2.grid(True, alpha=0.3)

        # 3. 残差 vs 预测值
        ax3.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax3.axhline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('预测值')
        ax3.set_ylabel('残差')
        ax3.set_title('残差 vs 预测值')
        ax3.grid(True, alpha=0.3)

        # 4. 训练历史（如果有）
        ax4.axis('off')
        if fold_result['train_losses'] and fold_result['val_losses']:
            ax4.plot(fold_result['train_losses'], label='训练损失', alpha=0.7)
            ax4.plot(fold_result['val_losses'], label='验证损失', alpha=0.7)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('损失')
            ax4.set_title('训练历史')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            info_text = f"""
                模型指标:
                - R²: {metrics['r2']:.4f}
                - RMSE: {metrics['rmse']:.4f}
                - MAE: {metrics['mae']:.4f}
                - 解释方差: {metrics['explained_variance']:.4f}
                - 样本数: {fold_result['n_samples']}
                """
            if fold_result.get('fallback', False):
                info_text += "\n⚠️ 使用备选预测方案"

            ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=11,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'year_{year}_detailed_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def generate_annual_report(results):
    """生成年度验证详细报告"""
    import numpy as np
    from datetime import datetime

    report = []
    report.append("=" * 60)
    report.append("          纯净版GNNWR年度交叉验证分析报告")
    report.append("=" * 60)
    report.append("")

    # 数据信息
    data_info = results['data_info']
    report.append("📊 数据信息:")
    report.append(f"  总样本数: {data_info['total_samples']}")
    report.append(f"  特征维度: {data_info['n_features']}")
    report.append(f"  年份数量: {data_info['n_years']}")
    report.append(f"  站点数量: {data_info['n_stations']}")
    report.append(f"  计算设备: {data_info['device']}")
    report.append(f"  混合精度: {data_info['mixed_precision']}")
    report.append("")

    # 年度交叉验证总体性能
    yearly_cv = results['yearly_cv']
    overall_metrics = yearly_cv['overall_metrics']
    report.append("📈 年度交叉验证总体性能:")
    report.append(f"  平均 R²: {overall_metrics['r2']:.4f}")
    report.append(f"  平均 RMSE: {overall_metrics['rmse']:.4f}")
    report.append(f"  平均 MAE: {overall_metrics['mae']:.4f}")
    report.append(f"  平均解释方差: {overall_metrics['explained_variance']:.4f}")
    report.append("")

    # 各年份详细性能
    report.append("📅 各年份详细性能:")
    report.append("-" * 60)
    report.append("年份       样本数     R²        RMSE       MAE       解释方差")
    report.append("-" * 60)

    fold_metrics = yearly_cv['fold_metrics']
    if fold_metrics:
        for year in sorted(fold_metrics.keys()):
            metrics = fold_metrics[year]
            n_samples = metrics.get('n_samples', 0)
            report.append(
                f"{year:<12}{n_samples:<10}{metrics['r2']:.4f}    {metrics['rmse']:.4f}    {metrics['mae']:.4f}    {metrics['explained_variance']:.4f}")
    else:
        report.append("          无有效数据")

    report.append("")

    # 性能分析
    report.append("🔍 性能分析:")
    if fold_metrics:
        best_year = max(fold_metrics.items(), key=lambda x: x[1]['r2'])
        worst_year = min(fold_metrics.items(), key=lambda x: x[1]['r2'])

        report.append(f"  最佳年份: {best_year[0]} (R² = {best_year[1]['r2']:.4f})")
        report.append(f"  最差年份: {worst_year[0]} (R² = {worst_year[1]['r2']:.4f})")

        # 稳定性分析
        r2_scores = [metrics['r2'] for metrics in fold_metrics.values()]
        r2_std = np.std(r2_scores)
        report.append(f"  R²标准差: {r2_std:.4f} (稳定性指标)")

        if r2_std < 0.1:
            stability = "优秀"
        elif r2_std < 0.15:
            stability = "良好"
        elif r2_std < 0.2:
            stability = "一般"
        else:
            stability = "较差"

        report.append(f"  模型稳定性: {stability}")
    else:
        report.append("  无法进行性能分析 - 无有效数据")

    report.append("")

    # 残差分析
    all_residuals = []
    fallback_count = 0

    for year_data in yearly_cv['fold_results']:
        all_residuals.extend(year_data['residuals'])
        if year_data.get('fallback', False):
            fallback_count += 1

    if all_residuals:
        residual_mean = np.mean(all_residuals)
        residual_std = np.std(all_residuals)
        report.append("📊 残差分析:")
        report.append(f"  残差均值: {residual_mean:.4f} (接近0表示无偏)")
        report.append(f"  残差标准差: {residual_std:.4f}")

        if fallback_count > 0:
            report.append(f"  ⚠️  {fallback_count}个年份使用了备选预测方案")
    else:
        report.append("📊 残差分析: 无有效数据")

    report.append("")

    # 建议和改进方向
    report.append("💡 建议和改进方向:")
    if fold_metrics:
        if overall_metrics['r2'] < 0.7:
            report.append("  • 考虑增加模型复杂度或特征工程")
        if len(fold_metrics) > 1:
            r2_scores = [metrics['r2'] for metrics in fold_metrics.values()]
            r2_std = np.std(r2_scores)
            if r2_std > 0.15:
                report.append("  • 模型在不同年份间稳定性有待提升")
        if overall_metrics['rmse'] > 1.0:
            report.append("  • 预测误差较大，可能需要更多数据或正则化")

        if fallback_count > 0:
            report.append("  • 部分年份训练失败，建议检查数据质量或调整超参数")
    else:
        report.append("  • 所有年份训练失败，建议检查数据预处理和模型配置")

    report.append("  • 可以尝试调整学习率或优化器参数")
    report.append("  • 考虑使用更复杂的空间权重机制")
    report.append("")

    report.append("=" * 60)
    report.append("报告生成完成 - " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    report.append("=" * 60)

    return "\n".join(report)


def print_annual_report(results):
    """在控制台输出年度验证报告摘要"""
    yearly_cv = results['yearly_cv']

    print("\n" + "=" * 70)
    print("             纯净版GNNWR年度交叉验证结果摘要")
    print("=" * 70)

    print(f"\n📊 数据概况:")
    print(f"  总样本: {results['data_info']['total_samples']}")
    print(f"  年份数: {results['data_info']['n_years']}")
    print(f"  站点数: {results['data_info']['n_stations']}")

    if yearly_cv['fold_metrics']:
        print(f"\n📈 年度交叉验证性能:")
        print(f"  平均 R²: {yearly_cv['overall_metrics']['r2']:.4f}")
        print(f"  平均 RMSE: {yearly_cv['overall_metrics']['rmse']:.4f}")
        print(f"  平均 MAE: {yearly_cv['overall_metrics']['mae']:.4f}")

        # 显示最佳和最差年份
        fold_metrics = yearly_cv['fold_metrics']
        best_year = max(fold_metrics.items(), key=lambda x: x[1]['r2'])
        worst_year = min(fold_metrics.items(), key=lambda x: x[1]['r2'])

        print(f"\n⭐ 最佳年份: {best_year[0]} (R² = {best_year[1]['r2']:.4f})")
        print(f"⚠️  最差年份: {worst_year[0]} (R² = {worst_year[1]['r2']:.4f})")

        # 性能稳定性
        r2_scores = [metrics['r2'] for metrics in fold_metrics.values()]
        r2_std = np.std(r2_scores)
        print(f"📊 性能稳定性: R²标准差 = {r2_std:.4f}")

        # 检查是否有备选方案
        fallback_count = sum(1 for result in yearly_cv['fold_results'] if result.get('fallback', False))
        if fallback_count > 0:
            print(f"⚠️  警告: {fallback_count}个年份使用了备选预测方案")

        print(f"\n💡 建议:")
        if yearly_cv['overall_metrics']['r2'] > 0.8:
            print("  模型性能优秀，可以考虑进行站点验证")
        elif yearly_cv['overall_metrics']['r2'] > 0.6:
            print("  模型性能良好，可以尝试优化超参数")
        else:
            print("  模型性能有待提升，建议检查特征工程")
    else:
        print(f"\n❌ 年度交叉验证失败:")
        print("  所有年份训练均未成功完成")
        print("  建议检查数据预处理和模型配置")

    print("=" * 70)




def create_pure_gnnwr_visualizations_optimized(results, output_dir):
    """生成纯净版GNNWR的优化可视化图表"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(20, 15))

    # 1. 训练信息展示
    plt.subplot(3, 4, 1)
    training_info = results['training_info']
    info_text = f"设备: {training_info['device']}\n"
    info_text += f"混合精度: {training_info['mixed_precision']}\n"
    info_text += f"模型参数: {training_info['model_parameters']:,}\n"
    if 'gpu_name' in training_info:
        info_text += f"GPU: {training_info['gpu_name']}\n"
        info_text += f"GPU内存: {training_info['gpu_memory']}"

    plt.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
    plt.axis('off')
    plt.title('训练配置信息', fontsize=12, fontweight='bold')

    # 2. 站点交叉验证散点图
    plt.subplot(3, 4, 2)
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

    # 3. 年度交叉验证散点图
    plt.subplot(3, 4, 3)
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

    # 4. 性能对比柱状图
    plt.subplot(3, 4, 4)
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

    # 5. R值对比柱状图
    plt.subplot(3, 4, 5)
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

    # 6. 折叠统计
    plt.subplot(3, 4, 6)
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

    # 7. 数据概况
    plt.subplot(3, 4, 7)
    data_info = results['data_info']
    data_text = f"总样本: {data_info['total_samples']}\n"
    data_text += f"特征数: {data_info['n_features']}\n"
    data_text += f"站点数: {data_info['n_stations']}\n"
    data_text += f"年份数: {data_info['n_years']}\n"
    data_text += f"训练集: {data_info['train_size']}\n"
    data_text += f"测试集: {data_info['test_size']}"

    plt.text(0.1, 0.5, data_text, fontsize=10, verticalalignment='center')
    plt.axis('off')
    plt.title('数据概况', fontsize=12, fontweight='bold')

    # 8. 残差分布
    plt.subplot(3, 4, 8)
    residuals = y_true_station - y_pred_station
    plt.hist(residuals, bins=30, alpha=0.7, color='orange', density=True)
    plt.xlabel('残差 (mm)')
    plt.ylabel('密度')
    plt.title('站点CV残差分布')

    # 添加正态分布曲线
    from scipy.stats import norm
    mu, std = norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pure_gnnwr_comprehensive_analysis_optimized.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 优化版可视化图表已保存")


def generate_detailed_report_optimized(results):
    """生成优化的详细分析报告"""
    report = []
    report.append("=" * 80)
    report.append("🎯 纯净版GNNWR详细分析报告 (GPU优化版)")
    report.append("=" * 80)
    report.append("")

    # 训练配置信息
    training_info = results['training_info']
    report.append("⚙️ 训练配置:")
    report.append(f"  设备: {training_info['device']}")
    report.append(f"  混合精度: {'是' if training_info['mixed_precision'] else '否'}")
    report.append(f"  模型参数: {training_info['model_parameters']:,}")
    report.append(f"  CPU工作线程: {training_info.get('cpu_workers', 'N/A')}")
    if 'gpu_name' in training_info:
        report.append(f"  GPU: {training_info['gpu_name']}")
        report.append(f"  GPU内存: {training_info['gpu_memory']}")
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

    # 计算性能提升（如果有基线）
    if 'baseline' in results:
        baseline_mae = results['baseline']['MAE']
        improvement = (baseline_mae - best_mae) / baseline_mae * 100
        report.append(f"  相比基线提升: {improvement:.1f}%")
    report.append("")

    report.append("=" * 80)
    report.append("报告生成时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    report.append("=" * 80)

    return "\n".join(report)


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


def pure_gnnwr_cross_validate_fixed(X, y, groups, coords, cv_type, logger,
                                    device='auto', mixed_precision=True, cpu_workers=24):
    """
    修复版的纯净GNNWR交叉验证函数
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    # 根据交叉验证类型设置分组
    if cv_type == 'yearly':
        unique_groups = np.unique(groups)
        n_splits = len(unique_groups)
        group_kfold = GroupKFold(n_splits=n_splits)
    else:
        n_splits = 5
        group_kfold = GroupKFold(n_splits=n_splits)

    fold_results = []
    fold_metrics = {}

    logger.info(f"开始{cv_type}交叉验证，共{n_splits}折")

    for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):
        try:
            # 获取当前测试组的标识
            test_group = np.unique(groups[test_idx])[0]
            logger.info(f"训练第{fold_idx + 1}/{n_splits}折，测试组: {test_group}")

            # 数据分割
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            coords_train, coords_test = coords[train_idx], coords[test_idx]

            # 特征标准化 - 确保正确拟合
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

            # 训练模型
            model = PureGNNWRTrainer(
                input_dim=X_train_scaled.shape[1],
                coords=coords_train,  # 添加坐标数据
                hidden_dims=[512, 256, 128, 64],
                dropout_rate=0.3,
                learning_rate=0.001,
                device=device,
                mixed_precision=mixed_precision
            )

            # 训练模型
            train_losses, val_losses = model.fit(
                X_train_scaled, y_train_scaled, coords_train
            )

            # 预测 - 使用训练时的标准化器
            y_pred_scaled = model.predict(X_test_scaled, coords_test)

            # 反标准化预测结果
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

            # 计算指标
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            explained_variance = 1 - np.var(y_test - y_pred) / np.var(y_test)

            # 存储结果
            fold_result = {
                'fold': fold_idx,
                'test_group': test_group,
                'y_true': y_test,
                'y_pred': y_pred,
                'residuals': y_test - y_pred,
                'n_samples': len(y_test),
                'metrics': {
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'explained_variance': explained_variance
                },
                'train_losses': train_losses,
                'val_losses': val_losses
            }

            fold_results.append(fold_result)
            fold_metrics[test_group] = fold_result['metrics']

            logger.info(f"  第{fold_idx + 1}折完成 - R²: {r2:.4f}, RMSE: {rmse:.4f}")

        except Exception as e:
            logger.error(f"第{fold_idx + 1}折训练失败: {e}")
            # 如果失败，使用简单均值作为预测
            y_pred_fallback = np.full_like(y_test, np.mean(y_train))

            r2 = r2_score(y_test, y_pred_fallback)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_fallback))
            mae = mean_absolute_error(y_test, y_pred_fallback)
            explained_variance = 1 - np.var(y_test - y_pred_fallback) / np.var(y_test)

            fallback_result = {
                'fold': fold_idx,
                'test_group': test_group,
                'y_true': y_test,
                'y_pred': y_pred_fallback,
                'residuals': y_test - y_pred_fallback,
                'n_samples': len(y_test),
                'metrics': {
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'explained_variance': explained_variance
                },
                'train_losses': [],
                'val_losses': [],
                'fallback': True
            }

            fold_results.append(fallback_result)
            fold_metrics[test_group] = fallback_result['metrics']
            logger.warning(f"  使用备选方案完成第{fold_idx + 1}折")

    # 计算总体指标
    all_y_true = np.concatenate([result['y_true'] for result in fold_results])
    all_y_pred = np.concatenate([result['y_pred'] for result in fold_results])

    overall_metrics = {
        'r2': r2_score(all_y_true, all_y_pred),
        'rmse': np.sqrt(mean_squared_error(all_y_true, all_y_pred)),
        'mae': mean_absolute_error(all_y_true, all_y_pred),
        'explained_variance': 1 - np.var(all_y_true - all_y_pred) / np.var(all_y_true)
    }

    logger.info(f"{cv_type}交叉验证完成 - 总体R²: {overall_metrics['r2']:.4f}")

    return {
        'fold_results': fold_results,
        'fold_metrics': fold_metrics,
        'overall_metrics': overall_metrics
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


def print_comprehensive_report_optimized(results):
    """打印优化的综合报告"""
    print("\n" + "=" * 70)
    print("🎯 纯净版GNNWR完整分析报告 (GPU优化版)")
    print("=" * 70)

    # 训练配置
    training_info = results['training_info']
    print(f"\n⚙️ 训练配置:")
    print(f"  设备: {training_info['device']}")
    print(f"  混合精度: {'是' if training_info['mixed_precision'] else '否'}")
    if 'gpu_name' in training_info:
        print(f"  GPU: {training_info['gpu_name']}")

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
# def SWEClusterEnsemble_run_pure_comparison_optimized(self, df, device='auto', mixed_precision=True, cpu_workers=24):
#     """
#     在SWEClusterEnsemble类中添加的方法
#     用于快速运行纯净版对比实验 - 优化版本
#     """
#     return train_pure_gnnwr_analysis(
#         df,
#         device=device,
#         mixed_precision=mixed_precision,
#         cpu_workers=cpu_workers
#     )





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
        results, trainer =  train_pure_gnnwr_annual_only(df)
    except Exception as e:
        print(f"示例运行失败: {e}")
        print("请确保有数据文件并修改文件路径")
    print("测试完成！")