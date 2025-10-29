import logging

import logger
import numpy as np
import pandas as pd
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
from torch.utils.data import DataLoader

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

    def __init__(self, n_clusters=4, params=None, gnnwr_params=None, use_enhanced_gnnwr=True):
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

        # XGBoost参数
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        # GNNWR参数
        self.gnnwr_params = {
            'hidden_dims': [64, 32, 16],
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'patience': 10,
            'bandwidth': None,
            'use_spatial_weights': True
        }
        if gnnwr_params:
            self.gnnwr_params.update(gnnwr_params)

        self.logger.info(f"初始化SWE聚类集成回归器，聚类数: {n_clusters}")
        self.logger.info(f"使用{'增强版' if self.use_enhanced_gnnwr else '基础版'}GNNWR")
        self.logger.info(f"GNNWR参数: {self.gnnwr_params}")

    def preprocess_data(self, df):
        """数据预处理

        Args:
            df (pd.DataFrame): 输入数据

        Returns:
            tuple: (X, y, station_groups, year_groups, coords)
        """
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

        # 提取坐标信息（如果可用）
        coords = None
        if all(col in df.columns for col in ['longitude', 'latitude']):
            coords = df[['longitude', 'latitude']].values
            self.logger.info(f"使用经纬度坐标: {len(coords)} 个点")
        elif all(col in df.columns for col in ['lon', 'lat']):
            coords = df[['lon', 'lat']].values
            self.logger.info(f"使用经纬度坐标: {len(coords)} 个点")
        else:
            self.logger.warning("未找到坐标信息，将使用虚拟坐标")
            # 创建基于站点ID的虚拟坐标
            unique_stations = np.unique(station_groups)
            station_to_coord = {station: [i, i] for i, station in enumerate(unique_stations)}
            coords = np.array([station_to_coord[station] for station in station_groups])

        self.logger.info(f"数据预处理完成: {len(X)}个样本, {X.shape[1]}个特征")
        self.logger.info(f"站点数: {len(np.unique(station_groups))}, 年份数: {len(np.unique(year_groups))}")

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

            # 训练XGBoost模型
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
        """训练GNNWR集成模型

        Args:
            X (np.array): 特征数据
            y (np.array): 目标变量
            cluster_predictions (np.array): 各聚类模型的预测结果
            coords (np.array): 坐标数据
        """
        self.logger.info("训练GNNWR集成模型...")

        # 使用特征：原始特征 + 各聚类预测
        gnnwr_features = np.hstack([X, cluster_predictions])

        # 处理缺失值
        if np.isnan(gnnwr_features).any():
            self.logger.info("处理GNNWR特征中的缺失值")
            self.gnnwr_imputer = SimpleImputer(strategy='median')
            gnnwr_features_imputed = self.gnnwr_imputer.fit_transform(gnnwr_features)
        else:
            gnnwr_features_imputed = gnnwr_features
            self.gnnwr_imputer = None

        if self.use_enhanced_gnnwr:
            # 使用增强版GNNWR
            self.logger.info("使用增强版GNNWR训练器")

            # 创建数据集
            dataset = EnhancedSpatialDataset(
                features=gnnwr_features_imputed,
                targets=y,
                coords=coords
            )

            train_loader = DataLoader(
                dataset,
                batch_size=self.gnnwr_params['batch_size'],
                shuffle=True
            )

            # 初始化增强版GNNWR训练器
            input_dim = gnnwr_features_imputed.shape[1]
            self.gnnwr_trainer = EnhancedGNNWRTrainer(
                input_dim=input_dim,
                coords=coords,
                hidden_dims=self.gnnwr_params['hidden_dims'],
                learning_rate=self.gnnwr_params['learning_rate'],
                bandwidth=self.gnnwr_params['bandwidth'],
                use_spatial_weights=self.gnnwr_params['use_spatial_weights']
            )

            # 训练模型
            self.logger.info(f"开始增强版GNNWR训练，输入维度: {input_dim}")
            self.gnnwr_trainer.train(
                train_loader,
                epochs=self.gnnwr_params['epochs'],
                patience=self.gnnwr_params['patience']
            )
        else:
            # 使用基础版GNNWR
            self.logger.info("使用基础版GNNWR训练器")

            # 创建数据集
            dataset = SpatialDataset(gnnwr_features_imputed, y)
            train_loader = DataLoader(
                dataset,
                batch_size=self.gnnwr_params['batch_size'],
                shuffle=True
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
        y_pred = self.predict_with_gnnwr(gnnwr_features_imputed, cluster_predictions, coords)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r_value, _ = pearsonr(y, y_pred)

        self.logger.info(f"GNNWR模型训练完成: MAE={mae:.3f}, RMSE={rmse:.3f}, R={r_value:.3f}")

    def predict_with_gnnwr(self, X, cluster_predictions, coords=None):
        """使用GNNWR进行预测

        Args:
            X (np.array): 特征数据
            cluster_predictions (np.array): 各聚类模型的预测结果
            coords (np.array): 坐标数据

        Returns:
            np.array: 预测结果
        """
        if self.gnnwr_trainer is None:
            raise ValueError("GNNWR模型尚未训练")

        # 使用特征：原始特征 + 各聚类预测
        gnnwr_features = np.hstack([X, cluster_predictions])

        # 处理缺失值
        if self.gnnwr_imputer is not None:
            gnnwr_features_imputed = self.gnnwr_imputer.transform(gnnwr_features)
        else:
            gnnwr_features_imputed = gnnwr_features

        # 预测
        if self.use_enhanced_gnnwr:
            return self.gnnwr_trainer.predict(gnnwr_features_imputed, coords)
        else:
            return self.gnnwr_trainer.predict(gnnwr_features_imputed)

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
        """执行交叉验证 - 使用GNNWR集成"""
        logo = LeaveOneGroupOut()

        all_predictions = []
        all_true_values = []
        fold_results = {}

        unique_groups = np.unique(groups)
        total_folds = len(unique_groups)

        self.logger.info(f"开始{cv_type}交叉验证，共{total_folds}个折叠...")

        # 在整个数据集上按站点进行一次聚类
        self.logger.info("在整个数据集上按站点进行聚类分配...")
        self.cluster_assignments = self.perform_clustering(X, groups)

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            group_id = groups[test_idx[0]]
            test_size = len(test_idx)
            train_size = len(train_idx)

            self.logger.debug(f"{cv_type} Fold {fold + 1}: 训练集{train_size}样本, 测试集{test_size}样本")

            # 分割数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train, groups_test = groups[train_idx], groups[test_idx]
            # 分割坐标（如果可用）
            coords_train = coords[train_idx] if coords is not None else None
            coords_test = coords[test_idx] if coords is not None else None

            # 使用固定的聚类分配
            train_cluster_labels = self.cluster_assignments[train_idx]
            test_cluster_labels = self.cluster_assignments[test_idx]

            # 训练聚类集成模型
            try:
                # 第一步：为每个聚类训练模型
                self.train_cluster_models(X_train, y_train, train_cluster_labels)

                # 第二步：获取训练集上的聚类预测
                cluster_predictions_train = self._get_cluster_predictions(X_train, train_cluster_labels)

                # 第三步：训练GNNWR集成模型
                self.train_gnnwr_model(X_train, y_train, cluster_predictions_train, coords_train)

                # 第四步：预测测试集
                cluster_predictions_test = self._get_cluster_predictions(X_test, test_cluster_labels)
                y_pred = self.predict_with_gnnwr(X_test, cluster_predictions_test, coords_test)

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

            # 3. 站点交叉验证（使用固定聚类）
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 3: 站点交叉验证")
            self.logger.info("=" * 50)

            results['station_cv'] = self.cross_validate(X, y, station_groups, coords, 'station')

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
        """生成分析报告

        Args:
            results (dict): 分析结果

        Returns:
            str: 报告文本
        """
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

        # 站点交叉验证结果
        station_cv = results['station_cv']
        station_overall = station_cv['overall']
        report.append("🏔️ 站点交叉验证结果:")
        report.append(f"  折叠数量: {station_cv['folds']}")
        report.append(f"  MAE: {station_overall['MAE']:.3f} mm")
        report.append(f"  RMSE: {station_overall['RMSE']:.3f} mm")
        report.append(f"  R: {station_overall['R']:.3f}")
        report.append(f"  R²: {station_overall['R_squared']:.3f}")
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
        report.append(f"  XGBoost参数: {self.params}")
        report.append(f"  GNNWR参数: {self.gnnwr_params}")

        return "\n".join(report)

    def _create_visualizations(self, results, output_dir):
        """创建可视化图表

        Args:
            results (dict): 分析结果
            output_dir (str): 输出目录
        """
        self.logger.info("生成可视化图表...")

        try:
            # 1. 预测值与真实值散点图
            plt.figure(figsize=(12, 10))

            # 站点交叉验证散点图
            plt.subplot(2, 2, 1)
            station_cv = results['station_cv']
            y_true_station = station_cv['true_values']
            y_pred_station = station_cv['predictions']

            plt.scatter(y_true_station, y_pred_station, alpha=0.6, s=20)
            plt.plot([y_true_station.min(), y_true_station.max()],
                     [y_true_station.min(), y_true_station.max()], 'r--', alpha=0.8)
            plt.xlabel('真实SWE (mm)')
            plt.ylabel('预测SWE (mm)')
            plt.title(f'站点交叉验证\nMAE={station_cv["overall"]["MAE"]:.2f}, R={station_cv["overall"]["R"]:.3f}')
            plt.grid(True, alpha=0.3)

            # 年度交叉验证散点图
            plt.subplot(2, 2, 2)
            yearly_cv = results['yearly_cv']
            y_true_yearly = yearly_cv['true_values']
            y_pred_yearly = yearly_cv['predictions']

            plt.scatter(y_true_yearly, y_pred_yearly, alpha=0.6, s=20, color='orange')
            plt.plot([y_true_yearly.min(), y_true_yearly.max()],
                     [y_true_yearly.min(), y_true_yearly.max()], 'r--', alpha=0.8)
            plt.xlabel('真实SWE (mm)')
            plt.ylabel('预测SWE (mm)')
            plt.title(f'年度交叉验证\nMAE={yearly_cv["overall"]["MAE"]:.2f}, R={yearly_cv["overall"]["R"]:.3f}')
            plt.grid(True, alpha=0.3)

            # 3. 聚类分布图
            plt.subplot(2, 2, 3)
            cluster_assignments = results['cluster_assignments']
            cluster_counts = np.bincount(cluster_assignments)
            colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))

            bars = plt.bar(range(len(cluster_counts)), cluster_counts, color=colors)
            plt.xlabel('聚类ID')
            plt.ylabel('样本数量')
            plt.title('聚类分布')
            plt.xticks(range(len(cluster_counts)))

            # 在柱状图上添加数值标签
            for bar, count in zip(bars, cluster_counts):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{count}', ha='center', va='bottom')

            # 4. 性能对比图
            plt.subplot(2, 2, 4)
            methods = ['站点CV', '年度CV']
            maes = [station_cv['overall']['MAE'], yearly_cv['overall']['MAE']]
            rs = [station_cv['overall']['R'], yearly_cv['overall']['R']]

            x = np.arange(len(methods))
            width = 0.35

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()

            bars1 = ax1.bar(x - width / 2, maes, width, label='MAE', alpha=0.7, color='skyblue')
            bars2 = ax2.bar(x + width / 2, rs, width, label='R', alpha=0.7, color='lightcoral')

            ax1.set_xlabel('验证方法')
            ax1.set_ylabel('MAE (mm)', color='skyblue')
            ax2.set_ylabel('R', color='lightcoral')
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            plt.title('交叉验证性能对比')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_visualization.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"可视化生成失败: {e}")


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
def train_swe_cluster_ensemble(data_df, output_dir=None, n_clusters=4, params=None,
                               use_enhanced_gnnwr=True, gnnwr_params=None):
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
        use_enhanced_gnnwr=use_enhanced_gnnwr
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


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("测试聚类集成模型...")
    results = test_cluster_ensemble()
    print("测试完成！")