import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from cluster import SWEClusterEnsemble
import logging

logging.getLogger('optuna').setLevel(logging.WARNING)


class SWEOptunaOptimizer:
    """SWE模型超参数优化器"""

    def __init__(self, n_trials=50, timeout=3600):
        self.n_trials = n_trials
        self.timeout = timeout
        self.logger = logging.getLogger("SWEOptunaOptimizer")

    def optimize_rf_params(self, X, y, cv_folds=5):
        """优化随机森林参数"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42
            }

            model = RandomForestRegressor(**params, n_jobs=-1)
            score = cross_val_score(model, X, y, cv=cv_folds,
                                    scoring='neg_mean_absolute_error').mean()
            return -score  # 最小化MAE

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        self.logger.info(f"RF最佳参数: {study.best_params}")
        self.logger.info(f"RF最佳MAE: {study.best_value:.4f}")

        return study.best_params

    def optimize_xgb_params(self, X, y, cv_folds=5):
        """优化XGBoost参数"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42
            }

            model = xgb.XGBRegressor(**params)
            score = cross_val_score(model, X, y, cv=cv_folds,
                                    scoring='neg_mean_absolute_error').mean()
            return -score

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        self.logger.info(f"XGBoost最佳参数: {study.best_params}")
        self.logger.info(f"XGBoost最佳MAE: {study.best_value:.4f}")

        return study.best_params

    def optimize_gnnwr_params(self, X, y, coords, cv_folds=3):
        """优化GNNWR参数（简化版，因为训练较慢）"""

        def objective(trial):
            gnnwr_params = {
                'hidden_dims': [
                    trial.suggest_int('hidden_dim1', 32, 128),
                    trial.suggest_int('hidden_dim2', 16, 64),
                    trial.suggest_int('hidden_dim3', 8, 32)
                ],
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'epochs': 50,  # 减少训练轮数以加速
                'patience': 8,
                'use_spatial_weights': True,
                'bandwidth': trial.suggest_float('bandwidth', 1.0, 20.0)
            }

            # 使用简化评估（因为完整训练太慢）
            try:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                mae_scores = []

                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    coords_train = coords[train_idx] if coords is not None else None

                    # 创建简化模型进行快速评估
                    trainer = SWEClusterEnsemble(
                        n_clusters=4,
                        gnnwr_params=gnnwr_params,
                        use_enhanced_gnnwr=True
                    )

                    # 仅训练GNNWR部分进行快速评估
                    # 这里需要根据实际情况调整评估方法
                    score = self._quick_gnnwr_eval(trainer, X_train, y_train, X_val, y_val, coords_train)
                    mae_scores.append(score)

                return np.mean(mae_scores)

            except Exception as e:
                return float('inf')  # 如果出错返回最差值

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=min(20, self.n_trials),
                       timeout=self.timeout // 2)  # 减少时间

        self.logger.info(f"GNNWR最佳参数: {study.best_params}")

        return study.best_params

    def _quick_gnnwr_eval(self, trainer, X_train, y_train, X_val, y_val, coords):
        """快速GNNWR评估"""
        # 简化评估逻辑，实际使用时需要根据您的代码调整
        try:
            # 这里放置简化的GNNWR训练和评估逻辑
            # 返回MAE值
            return 0.1  # 示例值
        except:
            return float('inf')


def optimize_swe_model(data_df, model_type='rf', n_trials=50):
    """主优化函数"""
    optimizer = SWEOptunaOptimizer(n_trials=n_trials)

    # 数据预处理（排除经纬度特征）
    from cluster import SWEClusterEnsemble
    trainer = SWEClusterEnsemble()
    X, y, _, _, coords = trainer.preprocess_data(data_df)

    if model_type == 'rf':
        best_params = optimizer.optimize_rf_params(X, y)
    elif model_type == 'xgb':
        best_params = optimizer.optimize_xgb_params(X, y)
    elif model_type == 'gnnwr':
        best_params = optimizer.optimize_gnnwr_params(X, y, coords)
    else:
        raise ValueError("model_type必须是 'rf', 'xgb' 或 'gnnwr'")

    return best_params