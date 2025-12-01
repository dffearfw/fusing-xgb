"""
GTNNWR局部线性回归合理性验证模块
提供三种验证方法：
1. 残差空间自相关分析 (Moran's I)
2. 局部R²空间分布分析
3. 随机森林基准测试
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import KNN
from esda.moran import Moran
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')


class GTNNWRValidator:
    """
    GTNNWR模型验证器

    Parameters
    ----------
    gtnnwr : 训练好的GTNNWR模型
    train_data : 训练集DataFrame
    val_data : 验证集DataFrame
    test_data : 测试集DataFrame
    x_columns : 特征列名列表
    y_column : 目标列名 (默认'swe')
    cluster_col : 聚类列名 (默认'cluster')
    output_dir : 输出目录 (默认'../demo_result')
    """

    def __init__(self, gtnnwr, train_data, val_data, test_data,
                 x_columns, y_column='swe', cluster_col='cluster',
                 output_dir='../demo_result'):
        self.gtnnwr = gtnnwr
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.x_columns = x_columns
        self.y_column = y_column
        self.cluster_col = cluster_col
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

    def analyze_residual_spatial_autocorrelation(self):
        """
        分析GTNNWR残差的空间自相关性
        如果Moran's I显著，说明线性假设不成立
        """
        print("\n=== 1. 残差空间自相关分析 ===")

        # 获取测试集预测结果
        results_df = self.gtnnwr.reg_result(only_return=True)
        test_results = results_df[results_df['dataset_belong'] == 'test'].copy()

        # 计算残差 (原始尺度)
        residuals = test_results[self.y_column] - test_results['denormalized_pred_result']

        # 创建GeoDataFrame
        geometry = [Point(xy) for xy in zip(test_results['X'], test_results['Y'])]
        gdf = gpd.GeoDataFrame(test_results, geometry=geometry)

        # 构建空间权重矩阵 (使用K最近邻)
        coords = np.array([[geom.x, geom.y] for geom in gdf.geometry])
        w = KNN(coords, k=8)  # 使用8个最近邻

        # 计算Moran's I
        moran = Moran(residuals, w)

        print(f"Moran's I: {moran.I:.4f}")
        print(f"P-value: {moran.p_norm:.4f}")
        print(f"期望值: {moran.EI:.4f}")

        # 可视化残差空间分布
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # 左图：残差空间分布
        gdf.plot(column=residuals, ax=ax[0], legend=True, cmap='coolwarm',
                 legend_kwds={'label': "Residuals", 'orientation': "horizontal"})
        ax[0].set_title("残差空间分布")

        # 右图：Moran散点图
        moran_scatter = moran.scramble_permutations()
        ax[1].scatter(moran_scatter[:, 0], moran_scatter[:, 1], alpha=0.6)
        ax[1].plot(moran_scatter[:, 0], moran_scatter[:, 1], 'r')
        ax[1].set_xlabel("标准化残差")
        ax[1].set_ylabel("空间滞后残差")
        ax[1].set_title(f"Moran散点图 (I={moran.I:.3f})")
        ax[1].axhline(0, color='red', linestyle='--')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'residual_spatial_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

        return moran.I, moran.p_norm

    def analyze_local_r2_distribution(self):
        """
        分析GTNNWR局部R²的空间分布
        如果局部R²变异大，说明空间异质性显著
        """
        print("\n=== 2. 局部R²空间分布分析 ===")

        # 获取测试集结果
        results_df = self.gtnnwr.reg_result(only_return=True)
        test_results = results_df[results_df['dataset_belong'] == 'test'].copy()

        # 提取模型学习到的空间权重
        coef_cols = [col for col in test_results.columns if col.startswith('coef_')]
        spatial_weights = test_results[coef_cols].values

        # 获取全局OLS系数
        global_coefs = self.gtnnwr._coefficient

        # 获取特征数据
        x_data = test_results[self.gtnnwr._train_dataset.x].values
        x_data = np.hstack([x_data, np.ones((x_data.shape[0], 1))])  # 添加截距项

        # 计算每个样本的局部预测和R²
        local_r2_list = []
        for i in range(len(spatial_weights)):
            # 局部预测 = 空间权重 @ 全局系数
            local_pred = np.dot(spatial_weights[i], global_coefs)

            # 计算局部R²
            y_true = test_results[self.y_column].iloc[i]
            sse = np.sum((y_true - local_pred) ** 2)
            sst = np.sum((test_results[self.y_column] - np.mean(test_results[self.y_column])) ** 2)
            local_r2 = 1 - (sse / sst) if sst > 0 else 0
            local_r2_list.append(local_r2)

        # 添加到结果DataFrame
        test_results['local_r2'] = local_r2_list

        # 统计分析
        mean_r2 = np.mean(local_r2_list)
        std_r2 = np.std(local_r2_list)
        cv = std_r2 / mean_r2

        print(f"局部R²统计:")
        print(f"  均值: {mean_r2:.4f}")
        print(f"  标准差: {std_r2:.4f}")
        print(f"  最小值: {np.min(local_r2_list):.4f}")
        print(f"  最大值: {np.max(local_r2_list):.4f}")
        print(f"  变异系数(CV): {cv:.4f}")

        # 空间可视化
        gdf = gpd.GeoDataFrame(test_results, geometry=[Point(xy) for xy in zip(test_results['X'], test_results['Y'])])

        fig, ax = plt.subplots(figsize=(12, 8))
        gdf.plot(column='local_r2', ax=ax, legend=True, cmap='viridis',
                 legend_kwds={'label': "Local R²", 'orientation': "horizontal"})
        ax.set_title("局部R²空间分布")
        plt.savefig(os.path.join(self.output_dir, 'local_r2_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()

        return mean_r2, std_r2, cv

    def benchmark_with_random_forest(self):
        """
        使用随机森林作为非线性基准，对比GTNNWR性能
        """
        print("\n=== 3. 随机森林基准测试 ===")

        # 按聚类训练RF模型
        rf_models = {}
        rf_metrics = {}

        for cluster_id in self.train_data[self.cluster_col].unique():
            print(f"\n训练簇 {cluster_id} 的随机森林模型...")

            # 筛选当前簇的数据
            train_cluster = self.train_data[self.train_data[self.cluster_col] == cluster_id]
            val_cluster = self.val_data[self.val_data[self.cluster_col] == cluster_id]

            # 训练随机森林
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )

            rf.fit(train_cluster[self.x_columns], train_cluster[self.y_column])
            rf_models[cluster_id] = rf

            # 在验证集上评估
            val_pred = rf.predict(val_cluster[self.x_columns])
            val_r2 = r2_score(val_cluster[self.y_column], val_pred)
            rf_metrics[cluster_id] = {'val_r2': val_r2}
            print(f"  验证R²: {val_r2:.4f}")

        # 在测试集上评估所有RF模型
        rf_predictions = []
        for _, test_row in self.test_data.iterrows():
            cluster_id = test_row[self.cluster_col]
            rf = rf_models[cluster_id]
            pred = rf.predict(test_row[self.x_columns].values.reshape(1, -1))[0]
            rf_predictions.append(pred)

        # 计算整体性能指标
        rf_r2 = r2_score(self.test_data[self.y_column], rf_predictions)
        rf_rmse = np.sqrt(mean_squared_error(self.test_data[self.y_column], rf_predictions))

        print(f"\n随机森林整体性能:")
        print(f"  测试R²: {rf_r2:.4f}")
        print(f"  测试RMSE: {rf_rmse:.4f}")

        return {
            'rf_r2': rf_r2,
            'rf_rmse': rf_rmse,
            'cluster_metrics': rf_metrics,
            'models': rf_models
        }

    def run_comprehensive_validation(self):
        """
        运行完整的验证流程
        """
        print("=" * 60)
        print("GTNNWR局部线性回归合理性验证")
        print("=" * 60)

        # 1. 残差空间自相关分析
        moran_i, moran_p = self.analyze_residual_spatial_autocorrelation()

        # 2. 局部R²分布分析
        mean_r2, std_r2, cv = self.analyze_local_r2_distribution()

        # 3. 随机森林基准测试
        rf_results = self.benchmark_with_random_forest()

        # 4. 综合结论
        print("\n" + "=" * 60)
        print("综合验证结论")
        print("=" * 60)

        print("\n1. 线性假设检验:")
        if moran_p < 0.05:
            print(f"❌ 残差存在显著空间自相关 (Moran's I={moran_i:.3f}, p={moran_p:.4f})")
            print("   → 线性模型未充分捕捉空间结构，建议考虑非线性方法")
        else:
            print(f"✅ 残差空间随机 (Moran's I={moran_i:.3f}, p={moran_p:.4f})")
            print("   → 线性假设在空间结构上可接受")

        print("\n2. 空间异质性检验:")
        if cv > 0.3:  # 经验阈值
            print(f"❌ 局部R²变异过大 (CV={cv:.3f})")
            print("   → 空间关系存在显著异质性，全局线性假设不成立")
        else:
            print(f"✅ 局部R²变异适中 (CV={cv:.3f})")
            print("   → 空间关系相对均质，线性假设可接受")

        print("\n3. 非线性基准对比:")
        # 获取GTNNWR测试性能
        results_df = self.gtnnwr.reg_result(only_return=True)
        gtnnwr_test = results_df[results_df['dataset_belong'] == 'test'].copy()
        gtnnwr_r2 = r2_score(gtnnwr_test[self.y_column], gtnnwr_test['denormalized_pred_result'])
        gtnnwr_rmse = np.sqrt(mean_squared_error(gtnnwr_test[self.y_column], gtnnwr_test['denormalized_pred_result']))

        print(f"GTNNWR测试性能: R²={gtnnwr_r2:.4f}, RMSE={gtnnwr_rmse:.4f}")
        print(f"随机森林测试性能: R²={rf_results['rf_r2']:.4f}, RMSE={rf_results['rf_rmse']:.4f}")

        if rf_results['rf_r2'] > gtnnwr_r2 + 0.05:  # 5%提升阈值
            print(f"❌ 随机森林显著优于GTNNWR (ΔR²={rf_results['rf_r2'] - gtnnwr_r2:+.4f})")
            print("   → 强烈建议使用非线性方法替代线性回归")
        elif rf_results['rf_r2'] < gtnnwr_r2 - 0.05:
            print(f"✅ GTNNWR显著优于随机森林 (ΔR²={rf_results['rf_r2'] - gtnnwr_r2:+.4f})")
            print("   → 当前线性回归假设合理")
        else:
            print(f"⚠️  随机森林与GTNNWR性能相当 (ΔR²={rf_results['rf_r2'] - gtnnwr_r2:+.4f})")
            print("   → 可考虑混合模型")

        return {
            'moran_p': moran_p,
            'r2_cv': cv,
            'gtnnwr_r2': gtnnwr_r2,
            'rf_r2': rf_results['rf_r2'],
            'recommendation': 'nonlinear' if (
                        moran_p < 0.05 or cv > 0.3 or rf_results['rf_r2'] > gtnnwr_r2 + 0.05) else 'linear'
        }
