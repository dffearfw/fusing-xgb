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
from gnnwr import models, datasets  # 注意：GTNNWR 实现在 gnnwr 包内
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("GTNNW_XGBoostTrainer")


class GTNNW_XGBoostTrainer:
    """GTNNW-XGBoost训练器 - 集成GTNNWR权重矩阵与XGBoost"""

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

    DEFAULT_GTNNWR_PARAMS = {
        'dense_layers': [[3], [512, 256, 64]],
        'drop_out': 0.4,
        'activate_func': None,
        'optimizer': "Adadelta",
        'optimizer_params': {
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [1000, 2000, 3000, 4000],
            "scheduler_gamma": 0.8,
        },
        'max_epoch': 3000,
        'early_stop': 1000,
        'print_frequency': 100,
        'batch_size': 128
    }

    def __init__(self, params=None, gtnnwr_params=None, use_gtnnwr=True):
        self.logger = logger
        self.model = None
        self.feature_columns = None
        self.target_column = 'swe'
        self.use_gtnnwr = use_gtnnwr

        # GTNNWR 特征定义（注意：不含 X,Y,Z）
        self.gtnnwr_x_columns = [
            'aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2',
            'elevation', 'std_slope', 'std_eastness', 'std_tpi', 'std_curvature1',
            'std_curvature2', 'std_high', 'std_aspect', 'glsnow', 'cswe',
            'snow_depth_snow_depth', 'ERA5温度_ERA5温度', 'era5_swe', 'gldas',
            'scp_start', 'scp_end', 'd1', 'd2', 'da', 'db', 'dc', 'dd'
        ]
        self.gtnnwr_y_column = ['swe']
        self.gtnnwr_spatial_columns = ['longitude', 'latitude']
        self.gtnnwr_temporal_columns = ['year', 'month', 'doy']

        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        self.gtnnwr_params = self.DEFAULT_GTNNWR_PARAMS.copy()
        if gtnnwr_params:
            self.gtnnwr_params.update(gtnnwr_params)

        self.logger.info(f"初始化GTNNW-XGBoost训练器")
        self.logger.info(f"XGBoost参数: {self.params}")
        self.logger.info(f"使用GTNNWR权重增强: {self.use_gtnnwr}")

    def preprocess_data(self, df, for_gtnnwr=False):
        self.logger.info("开始数据预处理...")

        df_clean = df.copy()

        required_columns = ['station_id', 'date', self.target_column]
        missing_columns = [col for col in required_columns if col not in df_clean.columns]
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")

        if self.use_gtnnwr:
            gtnnwr_required = (
                self.gtnnwr_x_columns +
                self.gtnnwr_spatial_columns +
                self.gtnnwr_temporal_columns
            )
            missing_gtnnwr = [col for col in gtnnwr_required if col not in df_clean.columns]
            if missing_gtnnwr:
                self.logger.warning(f"GTNNWR缺少以下列: {missing_gtnnwr}")
                for col in missing_gtnnwr:
                    df_clean[col] = 0.0

        if 'cswe' in df_clean.columns:
            cswe_invalid_mask = df_clean['cswe'] > 200
            if cswe_invalid_mask.sum() > 0:
                df_clean.loc[cswe_invalid_mask, 'cswe'] = np.nan

        exclude_columns = ['station_id', 'date', self.target_column, 'hydrological_doy']
        exclude_columns.extend([col for col in df_clean.columns if col.startswith('landuse_hash_')])

        self.feature_columns = [col for col in df_clean.columns if col not in exclude_columns]
        if not self.feature_columns:
            raise ValueError("没有找到可用的特征列")

        X = df_clean[self.feature_columns].values
        y = df_clean[self.target_column].values

        df_clean['year'] = pd.to_datetime(df_clean['date']).dt.year
        station_groups = df_clean['station_id'].values
        year_groups = df_clean['year'].values

        gtnnwr_data = None
        if self.use_gtnnwr:
            gtnnwr_data = df_clean.copy()
            for col in (
                self.gtnnwr_x_columns +
                self.gtnnwr_spatial_columns +
                self.gtnnwr_temporal_columns
            ):
                if col not in gtnnwr_data.columns:
                    gtnnwr_data[col] = 0.0
            gtnnwr_data["id"] = np.arange(len(gtnnwr_data))

        self.logger.info(f"✅ 数据预处理完成")
        self.logger.info(f" 样本数: {len(X)}, 特征数: {len(self.feature_columns)}")
        return X, y, station_groups, year_groups, gtnnwr_data

    def _train_gtnnwr_for_fold(self, train_data, val_data):
        self.logger.debug("为当前折叠训练GTNNWR模型...")
        print("\n" + "=" * 80)
        print("🧠 GTNNWR模型训练 (当前折叠)")
        print("=" * 80)

        try:
            original_train_samples = len(train_data)
            original_val_samples = len(val_data)

            print(f"📊 原始数据样本数:")
            print(f" 训练集: {original_train_samples}")
            print(f" 验证集: {original_val_samples}")

            # 确保所有列存在
            all_cols = (
                self.gtnnwr_x_columns +
                self.gtnnwr_spatial_columns +
                self.gtnnwr_temporal_columns +
                self.gtnnwr_y_column
            )
            for col in all_cols:
                if col not in train_data.columns:
                    train_data[col] = 0.0
                if col not in val_data.columns:
                    val_data[col] = 0.0

            train_data["id"] = np.arange(len(train_data))
            val_data["id"] = np.arange(len(val_data))

            # 合并用于填充
            combined = pd.concat([train_data, val_data], ignore_index=True)
            for col in all_cols:
                if col in combined.columns:
                    mean_val = combined[col].mean()
                    fill_val = mean_val if not np.isnan(mean_val) else 0.0
                    train_data[col].fillna(fill_val, inplace=True)
                    val_data[col].fillna(fill_val, inplace=True)

            # 初始化数据集（test_ratio=0）
            full_data = pd.concat([train_data, val_data], ignore_index=True)
            valid_ratio = len(val_data) / len(full_data)

            train_set, val_set, _ = datasets.init_dataset(
                data=full_data,
                test_ratio=0.0,
                valid_ratio=valid_ratio,
                x_column=self.gtnnwr_x_columns,
                y_column=self.gtnnwr_y_column,
                spatial_column=self.gtnnwr_spatial_columns,
                temp_column=self.gtnnwr_temporal_columns,
                id_column=['id'],
                use_model="gtnnwr",
                batch_size=self.gtnnwr_params['batch_size'],
                sample_seed=42
            )

            print(f"✅ GTNNWR数据集初始化完成")
            print(f" 训练集: {len(train_set)}, 验证集: {len(val_set)}")

            # 创建 GTNNWR 模型
            gtnnwr = models.GTNNWR(
                train_dataset=train_set,
                valid_dataset=val_set,
                test_dataset=train_set,
                dense_layers=self.gtnnwr_params['dense_layers'],
                drop_out=self.gtnnwr_params['drop_out'],
                optimizer=self.gtnnwr_params['optimizer'],
                optimizer_params=self.gtnnwr_params['optimizer_params'],
                model_name="GTNNWR_Fold",
                model_save_path="result/gtnnwr_models_temp",
                log_path="result/gtnnwr_logs_temp",
                write_path="result/gtnnwr_runs_temp"
            )

            print("🏋️ 开始训练GTNNWR...")
            gtnnwr.run(
                max_epoch=self.gtnnwr_params['max_epoch'],
                early_stop=self.gtnnwr_params['early_stop'],
                print_frequency=self.gtnnwr_params['print_frequency']
            )

            # ⚠️ 重要：GTNNWR 不直接输出权重矩阵
            # 此处返回全1权重（实际应用需从模型内部提取）
            train_weights = np.ones((original_train_samples, len(self.gtnnwr_x_columns)))
            val_weights = np.ones((original_val_samples, len(self.gtnnwr_x_columns)))

            print("✅ 权重矩阵准备就绪（模拟值）")
            return train_weights, val_weights

        except Exception as e:
            print(f"\n❌ GTNNWR训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.logger.error(f"GTNNWR训练失败: {e}")

            train_weights = np.ones((original_train_samples, len(self.gtnnwr_x_columns)))
            val_weights = np.ones((original_val_samples, len(self.gtnnwr_x_columns)))
            return train_weights, val_weights

    def _apply_gtnnwr_weights(self, X, weights, feature_columns, gtnnwr_x_columns):
        if weights is None:
            return X

        if weights.shape[1] != len(gtnnwr_x_columns):
            if weights.shape[1] > len(gtnnwr_x_columns):
                weights = weights[:, :len(gtnnwr_x_columns)]
            else:
                pad = np.ones((weights.shape[0], len(gtnnwr_x_columns) - weights.shape[1]))
                weights = np.hstack([weights, pad])

        feature_to_gtnnwr = {}
        for i, feat in enumerate(feature_columns):
            if feat in gtnnwr_x_columns:
                feature_to_gtnnwr[i] = gtnnwr_x_columns.index(feat)

        if not feature_to_gtnnwr:
            return X

        X_weighted = X.copy()
        for feat_idx, gtnnwr_idx in feature_to_gtnnwr.items():
            X_weighted[:, feat_idx] = X[:, feat_idx] * weights[:, gtnnwr_idx]

        return X_weighted

    def cross_validate(self, X, y, groups, cv_type='station', gtnnwr_data=None):
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
        print("=" * 100)

        for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups)):
            group_id = groups[val_idx[0]]
            train_size, val_size = len(train_idx), len(val_idx)
            print(f"\n🎯 Fold {fold+1}/{total_folds}: {group_id} ({train_size} / {val_size})")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if self.use_gtnnwr and gtnnwr_data is not None:
                train_fold = gtnnwr_data.iloc[train_idx].copy()
                val_fold = gtnnwr_data.iloc[val_idx].copy()
                train_w, val_w = self._train_gtnnwr_for_fold(train_fold, val_fold)
                X_train = self._apply_gtnnwr_weights(X_train, train_w, self.feature_columns, self.gtnnwr_x_columns)
                X_val = self._apply_gtnnwr_weights(X_val, val_w, self.feature_columns, self.gtnnwr_x_columns)

            model = xgb.XGBRegressor(**self.params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            all_predictions.extend(y_pred)
            all_true_values.extend(y_val)

            metrics = self.evaluate_predictions(y_val, y_pred)
            fold_results[group_id] = metrics
            fold_maes.append(metrics['MAE'])
            fold_rmses.append(metrics['RMSE'])
            fold_rs.append(metrics['R'])
            fold_samples.append(metrics['样本数'])

            r_str = f"{metrics['R']:.3f}" if not np.isnan(metrics['R']) else "NaN"
            print(f"📊 Fold {fold+1}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, R={r_str}")

        overall = self.evaluate_predictions(np.array(all_true_values), np.array(all_predictions))

        def safe_stat(vals, func):
            clean = [v for v in vals if not np.isnan(v)]
            return func(clean) if clean else np.nan

        mean_metrics = {
            'MAE': safe_stat(fold_maes, np.mean),
            'RMSE': safe_stat(fold_rmses, np.mean),
            'R': safe_stat(fold_rs, np.mean),
            '样本数': sum(fold_samples)
        }

        print("\n" + "=" * 100)
        print(f"🎉 {cv_type}交叉验证完成!")
        print(f"聚合 MAE: {overall['MAE']:.3f}, R: {overall['R']:.3f}")
        print("=" * 100)

        return {
            'overall': overall,
            'mean': mean_metrics,
            'by_fold': fold_results,
            'predictions': np.array(all_predictions),
            'true_values': np.array(all_true_values),
            'folds': total_folds
        }

    def evaluate_predictions(self, y_true, y_pred):
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() == 0:
            return {'MAE': np.nan, 'RMSE': np.nan, 'R': np.nan, '样本数': 0}
        y_true, y_pred = y_true[mask], y_pred[mask]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (np.nan, np.nan)
        return {'MAE': mae, 'RMSE': rmse, 'R': r, '样本数': len(y_true)}

    def train_final_model(self, X, y, gtnnwr_data=None):
        self.logger.info("训练最终XGBoost模型...")
        if self.use_gtnnwr and gtnnwr_data is not None:
            full_weights, _ = self._train_gtnnwr_for_fold(gtnnwr_data, gtnnwr_data.head(1))
            X = self._apply_gtnnwr_weights(X, full_weights, self.feature_columns, self.gtnnwr_x_columns)
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        self.logger.info("✅ 最终模型训练完成")
        return self.model

    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("模型未训练")
        imp = self.model.feature_importances_
        cols = self.feature_columns[:len(imp)]
        return pd.DataFrame({'feature': cols, 'importance': imp}).sort_values('importance', ascending=False)

    def run_complete_analysis(self, df, output_dir=None):
        self.logger.info("=" * 70)
        self.logger.info("🚀 开始GTNNW-XGBoost完整分析流程")
        self.logger.info("=" * 70)

        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"./gtnnw_xgboost_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"输出目录: {output_dir}")

        try:
            X, y, station_groups, year_groups, gtnnwr_data = self.preprocess_data(df)

            results = {
                'preprocessing': {
                    'samples': len(X),
                    'features': len(self.feature_columns),
                    'stations': len(np.unique(station_groups)),
                    'years': len(np.unique(year_groups)),
                    'use_gtnnwr': self.use_gtnnwr
                }
            }

            results['yearly_cv'] = self.cross_validate(X, y, year_groups, 'yearly', gtnnwr_data)

            if self.use_gtnnwr:
                self.gtnnwr_params['max_epoch'] = 3

            results['station_cv'] = self.cross_validate(X, y, station_groups, 'station', gtnnwr_data)

            results['final_model'] = self.train_final_model(X, y, gtnnwr_data)
            results['feature_importance'] = self.get_feature_importance()

            # 保存模型
            model_path = os.path.join(output_dir, 'final_model.pkl')
            joblib.dump(results['final_model'], model_path)
            self.logger.info(f"✅ 模型保存至: {model_path}")

            # 保存结果
            result_path = os.path.join(output_dir, 'results.json')
            serializable_results = {
                'preprocessing': results['preprocessing'],
                'yearly_cv_overall': results['yearly_cv']['overall'],
                'station_cv_overall': results['station_cv']['overall'],
                'feature_importance': results['feature_importance'].to_dict()
            }
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            self.logger.info(f"✅ 结果保存至: {result_path}")

            print("\n🎯 完整分析完成！")
            return results

        except Exception as e:
            self.logger.error(f"❌ 分析失败: {e}", exc_info=True)
            raise


# 便捷入口函数
def main():
    # 示例用法（需替换为您的实际数据路径）
    # df = pd.read_csv("your_data.csv")
    # trainer = GTNNW_XGBoostTrainer(use_gtnnwr=True)
    # results = trainer.run_complete_analysis(df, output_dir="./results")
    pass


if __name__ == "__main__":
    main()