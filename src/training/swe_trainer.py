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

logger = logging.getLogger("SWEXGBoostTrainer")


class SWEXGBoostTrainer:
    """SWE XGBoost训练器 - 集成数据预处理、交叉验证和模型训练"""

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

    def __init__(self, params=None):
        """初始化训练器

        Args:
            params (dict, optional): XGBoost参数，如果为None则使用默认参数
        """
        self.logger = logger
        self.model = None
        self.feature_columns = None
        self.target_column = 'swe'

        # 更新参数
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        self.logger.info(f"初始化SWE XGBoost训练器")
        self.logger.info(f"模型参数: {self.params}")

    def validate_data(self, df):
        """验证输入数据

        Args:
            df (pd.DataFrame): 输入数据

        Raises:
            ValueError: 数据验证失败时抛出
        """
        self.logger.info("验证输入数据...")

        # 检查DataFrame类型
        if not isinstance(df, pd.DataFrame):
            raise ValueError("输入数据必须是pandas DataFrame")

        # 检查必要列
        required_columns = ['station_id', 'date', self.target_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")

        # 检查目标变量
        swe_na_count = df[self.target_column].isna().sum()
        swe_total_count = len(df)
        if swe_na_count == swe_total_count:
            raise ValueError(f"{self.target_column}列全部为空值")

        self.logger.info(f"SWE数据完整性: {swe_total_count - swe_na_count}/{swe_total_count} 有效")

        # 检查样本数量
        if len(df) < 10:
            raise ValueError(f"样本数量太少 ({len(df)})，至少需要10个样本")

        # 检查站点数量
        station_count = df['station_id'].nunique()
        if station_count < 2:
            raise ValueError(f"站点数量太少 ({station_count})，至少需要2个站点")

        self.logger.info(f"✅ 数据验证通过: {len(df)} 行, {len(df.columns)} 列, {station_count} 个站点")

    def preprocess_data(self, df):
        """数据预处理

        Args:
            df (pd.DataFrame): 原始数据

        Returns:
            tuple: (X, y, station_groups, year_groups, feature_columns)
                  X: 特征矩阵 (numpy array)
                  y: 目标变量 (numpy array)
                  station_groups: 站点分组信息
                  year_groups: 年份分组信息
                  feature_columns: 特征列名列表
        """
        self.logger.info("开始数据预处理...")

        # 验证数据
        self.validate_data(df)

        # 创建数据副本
        df_clean = df.copy()

        # 删除目标变量为空的样本
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=[self.target_column]).copy()
        removed_count = initial_count - len(df_clean)

        if removed_count > 0:
            self.logger.info(f"删除 {removed_count} 个SWE为空值的样本")
        self.logger.info(f"剩余有效样本: {len(df_clean)} 行")

        # 确定特征列（排除station_id, date, swe）
        exclude_columns = ['station_id', 'date', self.target_column]
        self.feature_columns = [col for col in df_clean.columns if col not in exclude_columns]

        if not self.feature_columns:
            raise ValueError("没有找到可用的特征列")

        self.logger.info(f"使用 {len(self.feature_columns)} 个特征")
        self.logger.debug(f"特征列表: {self.feature_columns}")

        # 准备特征和目标变量
        X = df_clean[self.feature_columns].copy()
        y = df_clean[self.target_column].copy()

        # 处理特征缺失值
        X_processed = self._handle_missing_values(X)

        # 准备分组信息
        df_clean['year'] = pd.to_datetime(df_clean['date']).dt.year
        station_groups = df_clean['station_id'].values
        year_groups = df_clean['year'].values

        # 统计信息
        station_count = len(np.unique(station_groups))
        year_count = len(np.unique(year_groups))
        swe_mean = y.mean()
        swe_std = y.std()

        self.logger.info("✅ 数据预处理完成")
        self.logger.info(f"  📊 样本数: {len(X_processed)}")
        self.logger.info(f"  🔧 特征数: {len(self.feature_columns)}")
        self.logger.info(f"  📍 站点数: {station_count}")
        self.logger.info(f"  📅 年份数: {year_count}")
        self.logger.info(f"  ❄️  SWE统计: 均值={swe_mean:.2f}mm, 标准差={swe_std:.2f}mm")

        return X_processed.values, y.values, station_groups, year_groups

    def _handle_missing_values(self, X):
        """处理特征缺失值

        Args:
            X (pd.DataFrame): 特征数据

        Returns:
            pd.DataFrame: 处理后的特征数据
        """
        self.logger.info("处理特征缺失值...")

        initial_missing = X.isna().sum().sum()
        if initial_missing == 0:
            self.logger.info("没有发现缺失值")
            return X

        X_processed = X.copy()

        # 数值特征用中位数填充
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_missing = X_processed[numeric_cols].isna().sum().sum()
            if numeric_missing > 0:
                X_processed[numeric_cols] = X_processed[numeric_cols].fillna(
                    X_processed[numeric_cols].median()
                )
                self.logger.info(f"填充 {numeric_missing} 个数值特征缺失值")

        # 分类特征用众数填充并编码
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            col_missing = X_processed[col].isna().sum()
            if col_missing > 0:
                # 用众数填充，如果众数不存在则用'missing'
                if len(X_processed[col].mode()) > 0:
                    fill_value = X_processed[col].mode()[0]
                else:
                    fill_value = 'missing'
                X_processed[col] = X_processed[col].fillna(fill_value)
                self.logger.info(f"填充分类特征 '{col}' 的 {col_missing} 个缺失值")

            # 转换为数值编码
            X_processed[col] = X_processed[col].astype('category').cat.codes

        remaining_missing = X_processed.isna().sum().sum()
        if remaining_missing > 0:
            self.logger.warning(f"仍有 {remaining_missing} 个缺失值未处理")

        return X_processed

    def evaluate_predictions(self, y_true, y_pred):
        """评估预测结果

        Args:
            y_true (array-like): 真实值
            y_pred (array-like): 预测值

        Returns:
            dict: 包含评估指标的字典
        """
        # 移除NaN值
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

        # 计算MAE和RMSE
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

        # 安全计算皮尔逊相关系数
        def safe_pearsonr(x, y):
            """安全计算皮尔逊相关系数，处理常数数组情况"""
            # 检查输入数组是否为常数
            if len(x) <= 1:
                return np.nan, np.nan

            if np.all(x == x[0]) or np.all(y == y[0]):
                # 如果任一数组是常数，相关系数未定义
                return np.nan, np.nan

            # 检查方差是否为0
            if np.std(x) == 0 or np.std(y) == 0:
                return np.nan, np.nan

            try:
                return pearsonr(x, y)
            except:
                return np.nan, np.nan

        # 计算皮尔逊相关系数
        r, p_value = safe_pearsonr(y_true_clean, y_pred_clean)

        total_samples = len(y_true)
        valid_samples = len(y_true_clean)
        valid_ratio = valid_samples / total_samples if total_samples > 0 else 0

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R': r,
            'R_pvalue': p_value,
            '样本数': valid_samples,
            '总样本数': total_samples,
            '有效样本比例': valid_ratio
        }

    def cross_validate(self, X, y, groups, cv_type='station'):
        """执行交叉验证

        Args:
            X (np.array): 特征数据
            y (np.array): 目标变量
            groups (np.array): 分组信息
            cv_type (str): 交叉验证类型 ('station' 或 'yearly')

        Returns:
            dict: 交叉验证结果，包含聚合值、平均值和中位数
        """
        logo = LeaveOneGroupOut()

        all_predictions = []
        all_true_values = []
        fold_results = {}

        # 用于存储各折叠的指标
        fold_maes = []
        fold_rmses = []
        fold_rs = []
        fold_samples = []

        unique_groups = np.unique(groups)
        total_folds = len(unique_groups)

        self.logger.info(f"开始{cv_type}交叉验证，共{total_folds}个折叠...")

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            group_id = groups[test_idx[0]]
            test_size = len(test_idx)
            train_size = len(train_idx)

            self.logger.debug(f"{cv_type} Fold {fold + 1}: 训练集{train_size}样本, 测试集{test_size}样本")

            # 分割数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 训练模型
            model = xgb.XGBRegressor(**self.params)
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)

            # 存储结果
            all_predictions.extend(y_pred)
            all_true_values.extend(y_test)

            # 计算当前折叠性能
            fold_metrics = self.evaluate_predictions(y_test, y_pred)
            fold_results[group_id] = fold_metrics

            # 安全显示相关系数
            r_display = fold_metrics['R']
            if np.isnan(r_display):
                r_display_str = "NaN"
            else:
                r_display_str = f"{r_display:.3f}"

            self.logger.info(
                f"  {cv_type} Fold {fold + 1}/{total_folds}: {group_id} "
                f"({test_size}样本) - "
                f"MAE={fold_metrics['MAE']:.3f}, R={r_display_str}"
            )

            # 存储各折叠指标用于统计
            fold_maes.append(fold_metrics['MAE'])
            fold_rmses.append(fold_metrics['RMSE'])
            fold_rs.append(fold_metrics['R'])
            fold_samples.append(fold_metrics['样本数'])

            self.logger.info(
                f"  {cv_type} Fold {fold + 1}/{total_folds}: {group_id} "
                f"({test_size}样本) - "
                f"MAE={fold_metrics['MAE']:.3f}, R={fold_metrics['R']:.3f}"
            )

        # 计算聚合性能（所有测试样本一起计算）
        overall_metrics = self.evaluate_predictions(
            np.array(all_true_values),
            np.array(all_predictions)
        )

        # 计算各折叠指标的平均值和中位数
        def safe_statistic(values, func):
            """安全计算统计量，处理NaN值"""
            valid_values = [v for v in values if not np.isnan(v)]
            if len(valid_values) == 0:
                return np.nan
            return func(valid_values)

        mean_metrics = {
            'MAE': safe_statistic(fold_maes, np.mean),
            'RMSE': safe_statistic(fold_rmses, np.mean),
            'R': safe_statistic(fold_rs, np.mean),
            '样本数': np.sum(fold_samples)  # 总样本数
        }

        median_metrics = {
            'MAE': safe_statistic(fold_maes, np.median),
            'RMSE': safe_statistic(fold_rmses, np.median),
            'R': safe_statistic(fold_rs, np.median),
            '样本数': np.sum(fold_samples)
        }

        # 计算各折叠指标的标准差
        std_metrics = {
            'MAE': safe_statistic(fold_maes, np.std),
            'RMSE': safe_statistic(fold_rmses, np.std),
            'R': safe_statistic(fold_rs, np.std)
        }

        # 安全显示总体相关系数
        overall_r_display = overall_metrics['R']
        mean_r_display = mean_metrics['R']
        median_r_display = median_metrics['R']

        if np.isnan(overall_r_display):
            overall_r_str = "NaN"
        else:
            overall_r_str = f"{overall_r_display:.3f}"

        if np.isnan(mean_r_display):
            mean_r_str = "NaN"
        else:
            mean_r_str = f"{mean_r_display:.3f}"

        if np.isnan(median_r_display):
            median_r_str = "NaN"
        else:
            median_r_str = f"{median_r_display:.3f}"

        self.logger.info(f"✅ {cv_type}交叉验证完成")
        self.logger.info(f"  聚合性能: MAE={overall_metrics['MAE']:.3f}mm, R={overall_r_str}")
        self.logger.info(f"  平均性能: MAE={mean_metrics['MAE']:.3f}mm, R={mean_r_str}")
        self.logger.info(f"  中位数性能: MAE={median_metrics['MAE']:.3f}mm, R={median_r_str}")

        return {
            'overall': overall_metrics,  # 聚合计算：所有测试样本一起计算
            'mean': mean_metrics,  # 平均值：各折叠指标的平均
            'median': median_metrics,  # 中位数：各折叠指标的中位数
            'std': std_metrics,  # 标准差：各折叠指标的变异程度
            'by_fold': fold_results,  # 各折叠详细结果
            'predictions': np.array(all_predictions),
            'true_values': np.array(all_true_values),
            'folds': total_folds,
            'fold_metrics': {  # 各折叠指标列表
                'MAE': fold_maes,
                'RMSE': fold_rmses,
                'R': fold_rs,
                'samples': fold_samples
            }
        }

    def safe_statistic(values, func):
        """安全计算统计量，处理NaN值和空数组"""
        valid_values = [v for v in values if not np.isnan(v)]
        if len(valid_values) == 0:
            return np.nan
        return func(valid_values)

    def train_final_model(self, X, y):
        """训练最终模型

        Args:
            X (np.array): 特征数据
            y (np.array): 目标变量

        Returns:
            xgb.XGBRegressor: 训练好的XGBoost模型
        """
        self.logger.info("训练最终XGBoost模型...")

        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)

        self.logger.info("✅ 最终模型训练完成")
        return self.model

    def get_feature_importance(self):
        """获取特征重要性

        Returns:
            pd.DataFrame: 特征重要性排序，包含'feature'和'importance'列
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 train_final_model 方法")

        importance_scores = self.model.feature_importances_

        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        self.logger.info(f"特征重要性计算完成，最高重要性: {feature_importance_df['importance'].iloc[0]:.4f}")

        return feature_importance_df

    def run_complete_analysis(self, df, output_dir=None):
        """运行完整分析流程

        Args:
            df (pd.DataFrame): 输入数据
            output_dir (str, optional): 输出目录路径

        Returns:
            dict: 包含所有分析结果的字典
        """
        self.logger.info("=" * 70)
        self.logger.info("🚀 开始SWE XGBoost完整分析流程")
        self.logger.info("=" * 70)

        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"./swe_model_results_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"输出目录: {output_dir}")

        try:
            # 1. 数据预处理
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 1: 数据预处理")
            self.logger.info("=" * 50)

            X, y, station_groups, year_groups = self.preprocess_data(df)

            results = {
                'preprocessing': {
                    'samples': len(X),
                    'features': len(self.feature_columns),
                    'stations': len(np.unique(station_groups)),
                    'years': len(np.unique(year_groups))
                }
            }

            # 2. 站点交叉验证
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 2: 站点交叉验证")
            self.logger.info("=" * 50)

            results['station_cv'] = self.cross_validate(X, y, station_groups, 'station')

            # 3. 年度交叉验证
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 3: 年度交叉验证")
            self.logger.info("=" * 50)

            results['yearly_cv'] = self.cross_validate(X, y, year_groups, 'yearly')

            # 4. 训练最终模型
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 4: 训练最终模型")
            self.logger.info("=" * 50)

            results['final_model'] = self.train_final_model(X, y)

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

    def _create_scatter_plots(self, results, output_dir):
        """创建两种交叉验证方法的预测值与实际值散点图

        Args:
            results (dict): 分析结果
            output_dir (str): 输出目录路径
        """
        try:
            self.logger.info("📊 生成散点图...")

            # 设置图形样式
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            sns.set_style("whitegrid")

            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # 站点交叉验证散点图
            if 'station_cv' in results:
                self._plot_single_scatter(
                    ax1,
                    results['station_cv']['true_values'],
                    results['station_cv']['predictions'],
                    results['station_cv']['overall'],
                    '站点交叉验证',
                    'XGBoost SWE based on site cross-validation (mm)'
                )

            # 年度交叉验证散点图
            if 'yearly_cv' in results:
                self._plot_single_scatter(
                    ax2,
                    results['yearly_cv']['true_values'],
                    results['yearly_cv']['predictions'],
                    results['yearly_cv']['overall'],
                    '年度交叉验证',
                    'XGBoost SWE based on annual cross-validation (mm)'
                )

            # 调整布局
            plt.tight_layout()

            # 保存图片
            scatter_path = f'{output_dir}/scatter_plots.png'
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✅ 散点图保存: {scatter_path}")

        except Exception as e:
            self.logger.warning(f"生成散点图失败: {str(e)}")

    def _plot_single_scatter(self, ax, y_true, y_pred, metrics, title, ylabel):
        """绘制单个散点图

        Args:
            ax: matplotlib轴对象
            y_true: 真实值数组
            y_pred: 预测值数组
            metrics: 评估指标字典
            title: 图标题
            ylabel: 纵坐标标签
        """
        # 移除NaN值
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        # 设置坐标轴范围到175mm
        max_range = 175
        min_val = 0
        max_val = max_range

        # 1:1 参考线 - 黑色实线（不显示在图例中）
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.8, linewidth=2)

        # 使用清晰的散点图，根据密度着色
        try:
            # 使用2D核密度估计来着色
            from scipy.stats import gaussian_kde

            # 计算点的密度
            xy = np.vstack([y_true_clean, y_pred_clean])
            z = gaussian_kde(xy)(xy)

            # 按密度排序，确保高密度点在上层
            idx = z.argsort()
            y_true_sorted, y_pred_sorted, z_sorted = y_true_clean[idx], y_pred_clean[idx], z[idx]

            # 绘制散点图，使用清晰的点
            scatter = ax.scatter(y_true_sorted, y_pred_sorted,
                                 c=z_sorted,
                                 cmap='viridis',  # 清晰的蓝-黄颜色映射
                                 s=15,  # 适当大小的点
                                 alpha=0.7,  # 适当的透明度
                                 edgecolors='none',  # 无边框
                                 marker='o')  # 圆形点

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, label='点密度')

        except Exception as e:
            self.logger.warning(f"密度着色失败，使用普通散点图: {e}")
            # 备用方案：普通散点图
            scatter = ax.scatter(y_true_clean, y_pred_clean,
                                 alpha=0.6, s=15, c='blue', edgecolors='none')

        # 添加回归趋势线（虚线）
        if len(y_true_clean) > 1:
            try:
                # 计算线性回归
                slope, intercept, r_value, p_value, std_err = stats.linregress(y_true_clean, y_pred_clean)

                # 生成回归线数据点
                x_reg = np.linspace(min_val, max_val, 100)
                y_reg = slope * x_reg + intercept

                # 绘制回归线 - 红色虚线（不显示在图例中）
                ax.plot(x_reg, y_reg, 'r--', alpha=0.8, linewidth=2)

            except Exception as e:
                self.logger.warning(f"计算回归线失败: {e}")

        # 设置坐标轴
        ax.set_xlabel('实际 SWE (mm)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # 设置坐标轴范围
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_aspect('equal')

        # 添加网格
        ax.grid(True, alpha=0.3)

        # 添加统计信息文本框 - 移到右上角，无边框，放大字体
        mae = metrics['MAE']
        rmse = metrics['RMSE']
        r = metrics['R']
        n = metrics['样本数']

        # 安全处理相关系数显示
        if np.isnan(r):
            r_str = "NaN"
        else:
            r_str = f"{r:.3f}"

        stats_text = f'MAE = {mae:.2f} mm\nRMSE = {rmse:.2f} mm\nR = {r_str}\nN = {n}'

        # 右上角，无边框，放大字体
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=12,  # 放大字体
                fontfamily='monospace',
                weight='bold')  # 加粗

    def _create_combined_scatter_plot(self, results, output_dir):
        """创建合并的散点图（两种方法在同一图中）

        Args:
            results (dict): 分析结果
            output_dir (str): 输出目录路径
        """
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False

            fig, ax = plt.subplots(figsize=(10, 8))

            # 设置坐标轴范围到175mm
            max_range = 175
            min_val = 0
            max_val = max_range

            # 1:1 参考线 - 黑色实线
            ax.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.8, linewidth=2)

            colors = ['#1f77b4', '#ff7f0e']  # 清晰的蓝色和橙色
            labels = ['站点CV', '年度CV']
            methods = ['station_cv', 'yearly_cv']

            # 收集所有点用于密度计算
            all_true = []
            all_pred = []
            all_colors = []

            for i, method in enumerate(methods):
                if method in results:
                    y_true = results[method]['true_values']
                    y_pred = results[method]['predictions']

                    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                    y_true_clean = y_true[mask]
                    y_pred_clean = y_pred[mask]

                    if len(y_true_clean) > 0:
                        # 为每种方法添加点
                        scatter = ax.scatter(y_true_clean, y_pred_clean,
                                             c=colors[i],
                                             s=15,
                                             alpha=0.6,
                                             edgecolors='none',
                                             marker='o',
                                             label=labels[i])

                        all_true.extend(y_true_clean)
                        all_pred.extend(y_pred_clean)
                        all_colors.extend([colors[i]] * len(y_true_clean))

            if len(all_true) == 0:
                ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
                plt.savefig(f'{output_dir}/combined_scatter_plot.png', dpi=300, bbox_inches='tight')
                plt.close()
                return

            # 分别计算两种方法的回归线
            for i, method in enumerate(methods):
                if method in results:
                    y_true = results[method]['true_values']
                    y_pred = results[method]['predictions']

                    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                    y_true_clean = y_true[mask]
                    y_pred_clean = y_pred[mask]

                    if len(y_true_clean) > 1:
                        try:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(y_true_clean, y_pred_clean)

                            x_reg = np.linspace(min_val, max_val, 100)
                            y_reg = slope * x_reg + intercept

                            # 使用不同线型
                            linestyle = '--' if i == 0 else '-.'
                            ax.plot(x_reg, y_reg, color=colors[i], linestyle=linestyle,
                                    alpha=0.8, linewidth=2)
                        except Exception as e:
                            self.logger.warning(f"计算{method}回归线失败: {e}")

            # 设置坐标轴
            ax.set_xlim([min_val, max_val])
            ax.set_ylim([min_val, max_val])
            ax.set_aspect('equal')
            ax.set_xlabel('实际 SWE (mm)')
            ax.set_ylabel('预测 SWE (mm)')
            ax.set_title('SWE预测值与实际值散点图比较', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 添加统计信息 - 右上角，无边框，放大字体
            stats_text = ""
            if 'station_cv' in results:
                station_metrics = results['station_cv']['overall']
                station_r = station_metrics['R']
                station_r_str = f"{station_r:.3f}" if not np.isnan(station_r) else "NaN"
                stats_text += f"站点CV:\nMAE={station_metrics['MAE']:.2f}\nRMSE={station_metrics['RMSE']:.2f}\nR={station_r_str}\nN={station_metrics['样本数']}\n\n"

            if 'yearly_cv' in results:
                yearly_metrics = results['yearly_cv']['overall']
                yearly_r = yearly_metrics['R']
                yearly_r_str = f"{yearly_r:.3f}" if not np.isnan(yearly_r) else "NaN"
                stats_text += f"年度CV:\nMAE={yearly_metrics['MAE']:.2f}\nRMSE={yearly_metrics['RMSE']:.2f}\nR={yearly_r_str}\nN={yearly_metrics['样本数']}"

            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=11,
                    fontfamily='monospace',
                    weight='bold')

            # 添加简单的图例（只显示点，不显示线）
            ax.legend(loc='lower left', framealpha=0.9)

            plt.tight_layout()
            combined_path = f'{output_dir}/combined_scatter_plot.png'
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✅ 合并散点图保存: {combined_path}")

        except Exception as e:
            self.logger.warning(f"生成合并散点图失败: {str(e)}")

    def _save_results(self, results, output_dir):
        """保存结果到文件

        Args:
            results (dict): 分析结果
            output_dir (str): 输出目录路径
        """
        try:
            self.logger.info("💾 保存分析结果...")

            # 1. 保存最终模型
            if 'final_model' in results:
                model_path = f'{output_dir}/final_model.pkl'
                joblib.dump(results['final_model'], model_path)
                self.logger.info(f"✅ 模型保存: {model_path}")

            # 2. 保存站点交叉验证预测结果
            if 'station_cv' in results:
                station_pred_df = pd.DataFrame({
                    'true_swe': results['station_cv']['true_values'],
                    'predicted_swe': results['station_cv']['predictions']
                })
                station_pred_path = f'{output_dir}/station_cv_predictions.csv'
                station_pred_df.to_csv(station_pred_path, index=False)
                self.logger.info(f"✅ 站点CV预测结果保存: {station_pred_path}")

                # 保存站点CV各折叠详细结果
                station_fold_results = []
                for station_id, metrics in results['station_cv']['by_fold'].items():
                    station_fold_results.append({
                        'station_id': station_id,
                        'mae': metrics['MAE'],
                        'rmse': metrics['RMSE'],
                        'r': metrics['R'],
                        'samples': metrics['样本数']
                    })
                station_fold_df = pd.DataFrame(station_fold_results)
                station_fold_path = f'{output_dir}/station_cv_fold_results.csv'
                station_fold_df.to_csv(station_fold_path, index=False)
                self.logger.info(f"✅ 站点CV各折叠结果保存: {station_fold_path}")

            # 3. 保存年度交叉验证预测结果
            if 'yearly_cv' in results:
                yearly_pred_df = pd.DataFrame({
                    'true_swe': results['yearly_cv']['true_values'],
                    'predicted_swe': results['yearly_cv']['predictions']
                })
                yearly_pred_path = f'{output_dir}/yearly_cv_predictions.csv'
                yearly_pred_df.to_csv(yearly_pred_path, index=False)
                self.logger.info(f"✅ 年度CV预测结果保存: {yearly_pred_path}")

                # 保存年度CV各折叠详细结果
                yearly_fold_results = []
                for year, metrics in results['yearly_cv']['by_fold'].items():
                    yearly_fold_results.append({
                        'year': year,
                        'mae': metrics['MAE'],
                        'rmse': metrics['RMSE'],
                        'r': metrics['R'],
                        'samples': metrics['样本数']
                    })
                yearly_fold_df = pd.DataFrame(yearly_fold_results)
                yearly_fold_path = f'{output_dir}/yearly_cv_fold_results.csv'
                yearly_fold_df.to_csv(yearly_fold_path, index=False)
                self.logger.info(f"✅ 年度CV各折叠结果保存: {yearly_fold_path}")

            # 4. 保存特征重要性
            if 'feature_importance' in results:
                feature_path = f'{output_dir}/feature_importance.csv'
                results['feature_importance'].to_csv(feature_path, index=False)
                self.logger.info(f"✅ 特征重要性保存: {feature_path}")

            # 5. 保存详细的评估结果（JSON格式）
            eval_results = {
                'training_info': {
                    'timestamp': datetime.now().isoformat(),
                    'feature_columns': self.feature_columns,
                    'total_samples': results.get('preprocessing', {}).get('samples', 0),
                    'total_stations': results.get('preprocessing', {}).get('stations', 0),
                    'total_years': results.get('preprocessing', {}).get('years', 0)
                },
                'model_parameters': self.params,
                'station_cross_validation': {
                    'overall': results['station_cv']['overall'],
                    'mean': results['station_cv']['mean'],
                    'median': results['station_cv']['median'],
                    'std': results['station_cv']['std'],
                    'folds': results['station_cv']['folds'],
                    'fold_metrics': results['station_cv']['fold_metrics'],
                    'by_fold': {str(k): v for k, v in results['station_cv']['by_fold'].items()}
                },
                'yearly_cross_validation': {
                    'overall': results['yearly_cv']['overall'],
                    'mean': results['yearly_cv']['mean'],
                    'median': results['yearly_cv']['median'],
                    'std': results['yearly_cv']['std'],
                    'folds': results['yearly_cv']['folds'],
                    'fold_metrics': results['yearly_cv']['fold_metrics'],
                    'by_fold': {str(k): v for k, v in results['yearly_cv']['by_fold'].items()}
                },
                'performance_comparison': {
                    'station_better_mae': results['station_cv']['overall']['MAE'] < results['yearly_cv']['overall'][
                        'MAE'],
                    'station_better_rmse': results['station_cv']['overall']['RMSE'] < results['yearly_cv']['overall'][
                        'RMSE'],
                    'station_better_r': results['station_cv']['overall']['R'] > results['yearly_cv']['overall']['R'],
                    'recommended_method': 'station' if results['station_cv']['overall']['R'] >
                                                       results['yearly_cv']['overall']['R'] else 'yearly'
                }
            }

            eval_path = f'{output_dir}/evaluation_results.json'
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False, default=float)
            self.logger.info(f"✅ 详细评估结果保存: {eval_path}")

            # 6. 保存汇总报告（文本格式）
            summary_report = self._generate_summary_report(results)
            summary_path = f'{output_dir}/model_summary_report.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            self.logger.info(f"✅ 汇总报告保存: {summary_path}")

            # 7. 保存训练配置
            config_info = {
                'training_config': {
                    'model_type': 'XGBoost',
                    'target_variable': 'swe',
                    'cross_validation_methods': ['station_loocv', 'yearly_loocv'],
                    'evaluation_metrics': ['MAE', 'RMSE', 'R'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'feature_info': {
                    'total_features': len(self.feature_columns),
                    'feature_list': self.feature_columns
                }
            }

            config_path = f'{output_dir}/training_config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)
            self.logger.info(f"✅ 训练配置保存: {config_path}")

            # 8. 创建结果摘要文件
            self._create_results_summary(results, output_dir)

            # 9. 生成散点图
            self.logger.info("🎨 生成可视化图表...")
            self._create_scatter_plots(results, output_dir)
            self._create_combined_scatter_plot(results, output_dir)

            self.logger.info(f"📁 所有结果已保存到: {output_dir}")

        except Exception as e:
            self.logger.error(f"❌ 保存结果失败: {str(e)}")
            raise

    def _generate_summary_report(self, results):
        """生成简化的汇总报告用于保存

        Args:
            results (dict): 分析结果

        Returns:
            str: 汇总报告文本
        """
        report_lines = []
        report_lines.append("SWE模型训练汇总报告")
        report_lines.append("=" * 50)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # 基本统计
        if 'preprocessing' in results:
            preproc = results['preprocessing']
            report_lines.append("基本统计:")
            report_lines.append(f"  总样本数: {preproc['samples']}")
            report_lines.append(f"  特征数量: {preproc['features']}")
            report_lines.append(f"  站点数量: {preproc['stations']}")
            report_lines.append(f"  年份数量: {preproc['years']}")
            report_lines.append("")

        # 站点交叉验证结果
        if 'station_cv' in results:
            station = results['station_cv']
            report_lines.append("站点交叉验证:")
            report_lines.append(f"  折叠数: {station['folds']}")
            report_lines.append(f"  聚合MAE: {station['overall']['MAE']:.3f} mm")
            report_lines.append(f"  平均MAE: {station['mean']['MAE']:.3f} mm")
            report_lines.append(f"  中位数MAE: {station['median']['MAE']:.3f} mm")
            report_lines.append(f"  聚合R: {station['overall']['R']:.3f}")
            report_lines.append(f"  平均R: {station['mean']['R']:.3f}")
            report_lines.append(f"  中位数R: {station['median']['R']:.3f}")
            report_lines.append("")

        # 年度交叉验证结果
        if 'yearly_cv' in results:
            yearly = results['yearly_cv']
            report_lines.append("年度交叉验证:")
            report_lines.append(f"  折叠数: {yearly['folds']}")
            report_lines.append(f"  聚合MAE: {yearly['overall']['MAE']:.3f} mm")
            report_lines.append(f"  平均MAE: {yearly['mean']['MAE']:.3f} mm")
            report_lines.append(f"  中位数MAE: {yearly['median']['MAE']:.3f} mm")
            report_lines.append(f"  聚合R: {yearly['overall']['R']:.3f}")
            report_lines.append(f"  平均R: {yearly['mean']['R']:.3f}")
            report_lines.append(f"  中位数R: {yearly['median']['R']:.3f}")
            report_lines.append("")

        # 特征重要性
        if 'feature_importance' in results:
            top_features = results['feature_importance'].head(10)
            report_lines.append("前10个重要特征:")
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                report_lines.append(f"  {i:2d}. {row['feature']:<20} {row['importance']:.4f}")

        return "\n".join(report_lines)

    def _create_results_summary(self, results, output_dir):
        """创建结果摘要文件（CSV格式，便于分析）

        Args:
            results (dict): 分析结果
            output_dir (str): 输出目录路径
        """
        try:
            # 创建性能比较摘要
            summary_data = []

            # 站点CV摘要
            if 'station_cv' in results:
                station = results['station_cv']
                summary_data.append({
                    'method': 'station_cv',
                    'folds': station['folds'],
                    'overall_mae': station['overall']['MAE'],
                    'overall_rmse': station['overall']['RMSE'],
                    'overall_r': station['overall']['R'],
                    'mean_mae': station['mean']['MAE'],
                    'mean_rmse': station['mean']['RMSE'],
                    'mean_r': station['mean']['R'],
                    'median_mae': station['median']['MAE'],
                    'median_rmse': station['median']['RMSE'],
                    'median_r': station['median']['R'],
                    'std_mae': station['std']['MAE'],
                    'std_rmse': station['std']['RMSE'],
                    'std_r': station['std']['R'],
                    'samples': station['overall']['样本数']
                })

            # 年度CV摘要
            if 'yearly_cv' in results:
                yearly = results['yearly_cv']
                summary_data.append({
                    'method': 'yearly_cv',
                    'folds': yearly['folds'],
                    'overall_mae': yearly['overall']['MAE'],
                    'overall_rmse': yearly['overall']['RMSE'],
                    'overall_r': yearly['overall']['R'],
                    'mean_mae': yearly['mean']['MAE'],
                    'mean_rmse': yearly['mean']['RMSE'],
                    'mean_r': yearly['mean']['R'],
                    'median_mae': yearly['median']['MAE'],
                    'median_rmse': yearly['median']['RMSE'],
                    'median_r': yearly['median']['R'],
                    'std_mae': yearly['std']['MAE'],
                    'std_rmse': yearly['std']['RMSE'],
                    'std_r': yearly['std']['R'],
                    'samples': yearly['overall']['样本数']
                })

            # 保存摘要
            summary_df = pd.DataFrame(summary_data)
            summary_path = f'{output_dir}/performance_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            self.logger.info(f"✅ 性能摘要保存: {summary_path}")

        except Exception as e:
            self.logger.warning(f"创建结果摘要失败: {str(e)}")

    def _generate_report(self, results):
        """生成分析报告

        Args:
            results (dict): 分析结果

        Returns:
            str: 格式化的报告文本
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("📊 SWE XGBoost模型分析报告 - 详细统计")
        report_lines.append("=" * 80)

        # 基本信息
        report_lines.append(f"\n📋 基本信息:")
        report_lines.append(f"  训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"  特征数量: {len(self.feature_columns)}")

        if 'preprocessing' in results:
            preproc = results['preprocessing']
            report_lines.append(f"  样本数量: {preproc['samples']}")
            report_lines.append(f"  站点数量: {preproc['stations']}")
            report_lines.append(f"  年份数量: {preproc['years']}")

        # 站点交叉验证结果
        if 'station_cv' in results:
            station_overall = results['station_cv']['overall']
            station_mean = results['station_cv']['mean']
            station_median = results['station_cv']['median']
            station_std = results['station_cv']['std']

            # 安全处理相关系数显示
            station_overall_r = station_overall['R']
            station_mean_r = station_mean['R']
            station_median_r = station_median['R']

            station_overall_r_str = f"{station_overall_r:.3f}" if not np.isnan(station_overall_r) else "NaN"
            station_mean_r_str = f"{station_mean_r:.3f}" if not np.isnan(station_mean_r) else "NaN"
            station_median_r_str = f"{station_median_r:.3f}" if not np.isnan(station_median_r) else "NaN"

            report_lines.append(f"\n📍 站点交叉验证 (空间评估):")
            report_lines.append(f"  ┌{'─' * 20}┬{'─' * 10}┬{'─' * 10}┬{'─' * 10}┐")
            report_lines.append(f"  │ {'指标':18} │ {'聚合':8} │ {'平均':8} │ {'中位数':8} │")
            report_lines.append(f"  ├{'─' * 20}┼{'─' * 10}┼{'─' * 10}┼{'─' * 10}┤")
            report_lines.append(
                f"  │ {'MAE (mm)':18} │ {station_overall['MAE']:8.3f} │ {station_mean['MAE']:8.3f} │ {station_median['MAE']:8.3f} │")
            report_lines.append(
                f"  │ {'RMSE (mm)':18} │ {station_overall['RMSE']:8.3f} │ {station_mean['RMSE']:8.3f} │ {station_median['RMSE']:8.3f} │")
            report_lines.append(
                f"  │ {'R':18} │ {station_overall_r_str:>8} │ {station_mean_r_str:>8} │ {station_median_r_str:>8} │")
            report_lines.append(f"  └{'─' * 20}┴{'─' * 10}┴{'─' * 10}┴{'─' * 10}┘")

            report_lines.append(f"  折叠数: {results['station_cv']['folds']}")
            report_lines.append(f"  样本数: {station_overall['样本数']}")

            # 添加标准差信息
            report_lines.append(
                f"  各折叠标准差: MAE±{station_std['MAE']:.3f}, RMSE±{station_std['RMSE']:.3f}, R±{station_std['R']:.3f}")

        # 年度交叉验证结果
        if 'yearly_cv' in results:
            yearly_overall = results['yearly_cv']['overall']
            yearly_mean = results['yearly_cv']['mean']
            yearly_median = results['yearly_cv']['median']
            yearly_std = results['yearly_cv']['std']

            # 安全处理相关系数显示
            yearly_overall_r = yearly_overall['R']
            yearly_mean_r = yearly_mean['R']
            yearly_median_r = yearly_median['R']

            yearly_overall_r_str = f"{yearly_overall_r:.3f}" if not np.isnan(yearly_overall_r) else "NaN"
            yearly_mean_r_str = f"{yearly_mean_r:.3f}" if not np.isnan(yearly_mean_r) else "NaN"
            yearly_median_r_str = f"{yearly_median_r:.3f}" if not np.isnan(yearly_median_r) else "NaN"

            report_lines.append(f"\n📅 年度交叉验证 (时间评估):")
            report_lines.append(f"  ┌{'─' * 20}┬{'─' * 10}┬{'─' * 10}┬{'─' * 10}┐")
            report_lines.append(f"  │ {'指标':18} │ {'聚合':8} │ {'平均':8} │ {'中位数':8} │")
            report_lines.append(f"  ├{'─' * 20}┼{'─' * 10}┼{'─' * 10}┼{'─' * 10}┤")
            report_lines.append(
                f"  │ {'MAE (mm)':18} │ {yearly_overall['MAE']:8.3f} │ {yearly_mean['MAE']:8.3f} │ {yearly_median['MAE']:8.3f} │")
            report_lines.append(
                f"  │ {'RMSE (mm)':18} │ {yearly_overall['RMSE']:8.3f} │ {yearly_mean['RMSE']:8.3f} │ {yearly_median['RMSE']:8.3f} │")
            report_lines.append(
                f"  │ {'R':18} │ {yearly_overall_r_str:>8} │ {yearly_mean_r_str:>8} │ {yearly_median_r_str:>8} │")
            report_lines.append(f"  └{'─' * 20}┴{'─' * 10}┴{'─' * 10}┴{'─' * 10}┘")

            report_lines.append(f"  折叠数: {results['yearly_cv']['folds']}")
            report_lines.append(f"  样本数: {yearly_overall['样本数']}")
            report_lines.append(
                f"  各折叠标准差: MAE±{yearly_std['MAE']:.3f}, RMSE±{yearly_std['RMSE']:.3f}, R±{yearly_std['R']:.3f}")

        # 模型参数
        report_lines.append(f"\n⚙️ 模型参数:")
        for key, value in self.params.items():
            if key in ['n_estimators', 'max_depth', 'min_child_weight']:
                report_lines.append(f"  {key}: {value}")
            elif key in ['learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'gamma']:
                report_lines.append(f"  {key}: {value}")

        # 特征重要性
        if 'feature_importance' in results:
            top_features = results['feature_importance'].head(5)
            report_lines.append(f"\n🔍 前5个重要特征:")
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                report_lines.append(f"  {i}. {row['feature']}: {row['importance']:.4f}")

        # 性能比较和建议
        if 'station_cv' in results and 'yearly_cv' in results:
            # 使用聚合值进行比较
            station_r = results['station_cv']['overall']['R']
            yearly_r = results['yearly_cv']['overall']['R']

            # 只有当两个R值都不是NaN时才进行比较
            if not np.isnan(station_r) and not np.isnan(yearly_r):
                report_lines.append(f"\n💡 性能分析和建议:")
                report_lines.append(f"  • 聚合性能比较:")
                if station_r > yearly_r:
                    report_lines.append(f"    站点CV优于年度CV (R: {station_r:.3f} > {yearly_r:.3f})")
                else:
                    report_lines.append(f"    年度CV优于站点CV (R: {yearly_r:.3f} > {station_r:.3f})")
            else:
                report_lines.append(f"\n💡 性能分析:")
                if np.isnan(station_r) and np.isnan(yearly_r):
                    report_lines.append(f"  • 两种方法的相关系数均为NaN，无法比较R值")
                elif np.isnan(station_r):
                    report_lines.append(f"  • 站点CV相关系数为NaN，年度CV R={yearly_r:.3f}")
                else:
                    report_lines.append(f"  • 年度CV相关系数为NaN，站点CV R={station_r:.3f}")

            # 比较稳定性（标准差越小越稳定）
            station_mae_std = results['station_cv']['std']['MAE']
            yearly_mae_std = results['yearly_cv']['std']['MAE']

            if station_mae_std < yearly_mae_std:
                report_lines.append(f"  • 稳定性比较: 站点CV更稳定 (MAE标准差: {station_mae_std:.3f} < {yearly_mae_std:.3f})")
            else:
                report_lines.append(f"  • 稳定性比较: 年度CV更稳定 (MAE标准差: {yearly_mae_std:.3f} < {station_mae_std:.3f})")

            # 最终建议
            if station_r > yearly_r and station_mae_std < yearly_mae_std:
                report_lines.append(f"  ✅ 强烈推荐使用站点交叉验证进行空间评估")
            elif yearly_r > station_r and yearly_mae_std < station_mae_std:
                report_lines.append(f"  ✅ 强烈推荐使用年度交叉验证进行时间评估")
            else:
                report_lines.append(f"  ⚠️  两种方法各有优势，请根据具体应用场景选择")

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)

    # 便捷使用函数


def train_swe_model(data_df, output_dir=None, params=None):
    """便捷函数：训练SWE模型

        Args:
            data_df (pd.DataFrame): 包含特征和SWE的数据
            output_dir (str, optional): 输出目录路径
            params (dict, optional): XGBoost参数

        Returns:
            dict: 包含所有训练结果的字典

        Example:
            >>> results = train_swe_model(df, output_dir='./results')
            >>> print(f"站点CV R: {results['station_cv']['overall']['R']:.3f}")
        """
    trainer = SWEXGBoostTrainer(params=params)
    return trainer.run_complete_analysis(data_df, output_dir)
