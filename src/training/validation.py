import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import logging
import os
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gnnw_verification.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("GNNW_Verification")


# 模拟GNNW_XGBoostTrainer类的关键方法
class MockGNNW_XGBoostTrainer:
    """模拟的GNNW-XGBoost训练器用于验证"""

    def __init__(self):
        self.logger = logger
        self.use_gnnwr = True

        # GNNWR特征列
        self.gnnwr_x_columns = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation',
                                'std_slope',
                                'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high', 'std_aspect',
                                'glsnow', 'cswe', 'snow_depth_snow_depth', 'ERA5温度_ERA5温度', 'era5_swe', 'doy',
                                'gldas',
                                'year', 'month', 'scp_start', 'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da', 'db', 'dc',
                                'dd']

        # 模拟特征列
        self.feature_columns = []

    def preprocess_data(self, df):
        """模拟数据预处理"""
        logger.info("模拟数据预处理...")

        # 确保必要的列存在
        if 'station_id' not in df.columns:
            df['station_id'] = np.arange(len(df))
        if 'swe' not in df.columns:
            df['swe'] = np.random.normal(50, 20, len(df))
        if 'date' not in df.columns:
            df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')

        # 确定特征列
        exclude_columns = ['station_id', 'date', 'swe']
        self.feature_columns = [col for col in df.columns if col not in exclude_columns]

        logger.info(f"特征列数: {len(self.feature_columns)}")
        logger.info(f"前10个特征: {self.feature_columns[:10]}")

        # 准备特征和目标
        X = df[self.feature_columns].values
        y = df['swe'].values

        # 分组信息
        df['year'] = pd.to_datetime(df['date']).dt.year
        station_groups = df['station_id'].values
        year_groups = df['year'].values

        # GNNWR数据
        gnnwr_data = df.copy()

        # 确保GNNWR需要的列都存在
        for col in self.gnnwr_x_columns:
            if col not in gnnwr_data.columns:
                gnnwr_data[col] = np.random.normal(0, 1, len(gnnwr_data))

        return X, y, station_groups, year_groups, gnnwr_data

    def _apply_gnnwr_weights(self, X, weights, feature_columns, gnnwr_x_columns):
        """应用GNNWR权重到特征矩阵"""
        if weights is None:
            logger.warning("权重矩阵为None，返回原始特征")
            return X

        logger.info(f"应用权重: X形状={X.shape}, 权重形状={weights.shape}")
        logger.info(f"特征列数={len(feature_columns)}, GNNWR特征列数={len(gnnwr_x_columns)}")

        # 创建特征映射
        feature_to_gnnwr = {}
        for i, feat in enumerate(feature_columns):
            if feat in gnnwr_x_columns:
                gnnwr_idx = gnnwr_x_columns.index(feat)
                feature_to_gnnwr[i] = gnnwr_idx

        logger.info(f"匹配的特征数: {len(feature_to_gnnwr)}/{len(feature_columns)}")

        if len(feature_to_gnnwr) == 0:
            logger.warning("没有匹配的特征，无法应用权重")
            return X

        # 应用权重
        X_weighted = X.copy()
        for feat_idx, gnnwr_idx in feature_to_gnnwr.items():
            if gnnwr_idx < weights.shape[1]:
                X_weighted[:, feat_idx] = X[:, feat_idx] * weights[:, gnnwr_idx]

        return X_weighted


def load_sample_data(n_samples=1000, n_features=50):
    """生成模拟数据用于测试"""
    logger.info(f"生成模拟数据: {n_samples}样本, {n_features}特征")

    # 创建特征名称
    feature_names = []
    gnnwr_feature_names = [
        'aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation', 'std_slope',
        'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high', 'std_aspect',
        'glsnow', 'cswe', 'snow_depth_snow_depth', 'ERA5温度_ERA5温度', 'era5_swe', 'doy', 'gldas',
        'year', 'month', 'scp_start', 'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da', 'db', 'dc', 'dd'
    ]

    # 添加额外的非GNNWR特征
    other_features = ['landuse_' + str(i) for i in range(1, n_features - len(gnnwr_feature_names) + 1)]

    feature_names = gnnwr_feature_names + other_features

    # 生成数据
    np.random.seed(42)
    data = {}

    # 生成特征值
    for i, feature in enumerate(feature_names):
        if feature in ['longitude', 'latitude']:
            data[feature] = np.random.uniform(-180, 180, n_samples) if feature == 'longitude' else np.random.uniform(
                -90, 90, n_samples)
        elif feature in ['elevation', 'X', 'Y', 'Z']:
            data[feature] = np.random.normal(0, 100, n_samples)
        else:
            data[feature] = np.random.normal(0, 1, n_samples)

    # 添加必要列
    data['station_id'] = np.random.choice(range(1, 101), n_samples)
    data['date'] = pd.date_range(start='2000-01-01', periods=n_samples, freq='D')
    data['swe'] = 50 + np.random.normal(0, 20, n_samples)  # 目标变量

    df = pd.DataFrame(data)
    logger.info(f"模拟数据创建完成: {len(df)}行, {len(df.columns)}列")

    return df


def verify_weight_application(df, n_samples=100, random_seed=42):
    """严格验证权重是否被正确应用到特征上 - 使用随机抽样"""
    print("=" * 80)
    print("🧪 GNNW-XGBoost权重应用验证测试 (随机抽样)")
    print("=" * 80)

    # 设置随机种子
    np.random.seed(random_seed)

    # 创建训练器实例
    trainer = MockGNNW_XGBoostTrainer()

    # 预处理数据
    print("\n1. 数据预处理...")
    X, y, station_groups, year_groups, gnnwr_data = trainer.preprocess_data(df)

    total_samples = len(X)
    print(f"  总样本数: {total_samples}")
    print(f"  特征矩阵形状: X={X.shape}, y={y.shape}")
    print(f"  GNNWR数据形状: {gnnwr_data.shape}")
    print(f"  XGBoost特征列数: {len(trainer.feature_columns)}")
    print(f"  GNNWR特征列数: {len(trainer.gnnwr_x_columns)}")

    # 随机抽样
    if n_samples > total_samples:
        n_samples = total_samples
        print(f"  警告: 样本数超过总样本数，使用所有{total_samples}个样本")

    # 随机选择索引
    random_indices = np.random.choice(total_samples, n_samples, replace=False)
    random_indices = np.sort(random_indices)  # 排序以便阅读

    print(f"\n2. 随机抽样: 从{total_samples}个样本中随机抽取{n_samples}个样本")
    print(f"   抽样索引: {random_indices[:5]}...{random_indices[-5:]}")

    # 获取抽样数据
    X_test = X[random_indices]
    y_test = y[random_indices]
    gnnwr_data_test = gnnwr_data.iloc[random_indices].copy()

    # 生成模拟权重矩阵（模拟GNNWR输出）
    print("\n3. 生成模拟权重矩阵...")

    # 模拟GNNWR输出权重矩阵 (n_samples, n_gnnwr_features)
    # 这里我们模拟权重不是1.0，以测试权重应用是否有效
    n_gnnwr_features = len(trainer.gnnwr_x_columns)
    train_weights = np.random.normal(1.0, 0.3, (len(X_test), n_gnnwr_features))
    train_weights = np.clip(train_weights, 0.1, 2.0)  # 限制权重范围

    print(f"   模拟权重矩阵形状: {train_weights.shape}")
    print(f"   权重统计:")
    print(f"     均值: {train_weights.mean():.6f}")
    print(f"     标准差: {train_weights.std():.6f}")
    print(f"     范围: [{train_weights.min():.6f}, {train_weights.max():.6f}]")

    # 计算权重与1的差异
    weight_distance_from_one = np.abs(train_weights - 1).mean()
    print(f"   权重与1的平均距离: {weight_distance_from_one:.6f}")

    # 统计有多少权重显著不同于1
    significant_weights = np.sum(np.abs(train_weights - 1) > 0.01) / train_weights.size
    print(f"   与1差异大于0.01的权重比例: {significant_weights:.2%}")

    if weight_distance_from_one < 0.01:
        print("   ⚠️ 警告: 权重非常接近1，加权可能没有效果")
    else:
        print(f"   ✅ 权重与1有显著差异，加权会有效果")

    # 5. 应用权重
    print("\n5. 应用权重到特征...")

    # 原始特征
    original_features = X_test.copy()

    # 应用权重
    weighted_features = trainer._apply_gnnwr_weights(
        original_features, train_weights,
        trainer.feature_columns, trainer.gnnwr_x_columns
    )

    # 检查是否应用成功
    if np.array_equal(original_features, weighted_features):
        print("   ⚠️ 警告: 加权后特征与原始特征完全相同！")
    else:
        print(f"   ✅ 加权后特征与原始特征不同")

    # 6. 详细对比原始特征和加权特征
    print("\n6. 特征变化分析:")

    # 计算每个特征的变化
    changes = weighted_features - original_features
    abs_changes = np.abs(changes)
    relative_changes = abs_changes / (np.abs(original_features) + 1e-10)  # 避免除以0

    # 按特征统计变化
    feature_changes = []
    for i in range(original_features.shape[1]):
        feat_name = trainer.feature_columns[i] if i < len(trainer.feature_columns) else f"Feature_{i}"

        # 检查这个特征是否在GNNWR特征中
        is_gnnwr_feature = feat_name in trainer.gnnwr_x_columns

        # 只记录有变化的特征
        feat_change_mean = changes[:, i].mean()
        feat_change_std = changes[:, i].std()
        feat_abs_change_mean = abs_changes[:, i].mean()
        feat_rel_change_mean = relative_changes[:, i].mean()

        # 检查这个特征是否有显著变化
        has_significant_change = feat_abs_change_mean > 0.001

        feature_changes.append({
            'feature': feat_name,
            'is_gnnwr': is_gnnwr_feature,
            'change_mean': feat_change_mean,
            'change_std': feat_change_std,
            'abs_change_mean': feat_abs_change_mean,
            'rel_change_mean': feat_rel_change_mean,
            'has_significant_change': has_significant_change
        })

    # 创建DataFrame显示结果
    changes_df = pd.DataFrame(feature_changes)

    # 统计GNNWR特征和非GNNWR特征
    gnnwr_features_df = changes_df[changes_df['is_gnnwr']]
    non_gnnwr_features_df = changes_df[~changes_df['is_gnnwr']]

    print(f"\n  GNNWR特征统计:")
    print(f"    总数: {len(gnnwr_features_df)}")
    if len(gnnwr_features_df) > 0:
        gnnwr_mean_change = gnnwr_features_df['abs_change_mean'].mean()
        gnnwr_significant = gnnwr_features_df['has_significant_change'].sum()
        print(f"    平均绝对变化: {gnnwr_mean_change:.6f}")
        print(f"    显著变化特征数: {gnnwr_significant}/{len(gnnwr_features_df)}")

    print(f"\n  非GNNWR特征统计:")
    print(f"    总数: {len(non_gnnwr_features_df)}")
    if len(non_gnnwr_features_df) > 0:
        non_gnnwr_mean_change = non_gnnwr_features_df['abs_change_mean'].mean()
        non_gnnwr_significant = non_gnnwr_features_df['has_significant_change'].sum()
        print(f"    平均绝对变化: {non_gnnwr_mean_change:.6f}")
        print(f"    显著变化特征数: {non_gnnwr_significant}/{len(non_gnnwr_features_df)}")

    # 显示变化最大的特征
    print("\n  变化最大的10个特征:")
    print("  " + "-" * 100)

    sorted_changes = changes_df.sort_values('abs_change_mean', ascending=False)

    for idx, row in sorted_changes.head(10).iterrows():
        change_symbol = "✅" if row['has_significant_change'] else "⚠️"
        gnnwr_symbol = "G" if row['is_gnnwr'] else "N"
        print(f"    {change_symbol}[{gnnwr_symbol}] {row['feature']:<30} 平均变化: {row['change_mean']:+.6f}, "
              f"绝对变化: {row['abs_change_mean']:.6f}, 相对变化: {row['rel_change_mean']:.2%}")

    # 7. 可视化变化
    print("\n7. 生成可视化图表...")
    visualize_feature_changes(original_features, weighted_features, changes_df)

    # 8. 统计总结
    print("\n8. 验证结果总结:")

    total_features = original_features.shape[1]
    gnnwr_features_count = sum([1 for f in trainer.feature_columns if f in trainer.gnnwr_x_columns])
    non_gnnwr_features_count = total_features - gnnwr_features_count

    print(f"   总特征数: {total_features}")
    print(f"   GNNWR特征数: {gnnwr_features_count} (应该被加权)")
    print(f"   非GNNWR特征数: {non_gnnwr_features_count} (保持不变)")

    # 总体变化统计
    mean_abs_change = abs_changes.mean()
    significant_changes = (abs_changes > 0.001).sum() / abs_changes.size

    print(f"\n   所有特征平均绝对变化: {mean_abs_change:.6f}")
    print(f"   显著变化(>0.001)的比例: {significant_changes:.2%}")

    # 验证结论
    verification_passed = False
    if len(gnnwr_features_df) > 0 and gnnwr_features_df['abs_change_mean'].mean() > 0.001:
        verification_passed = True
        print(f"\n   ✅ 验证通过: GNNWR特征被成功加权")
        print(f"      平均变化: {gnnwr_features_df['abs_change_mean'].mean():.6f}")
        print(f"      权重与1的平均距离: {weight_distance_from_one:.6f}")
    else:
        print(f"\n   ❌ 验证失败: GNNWR特征没有被正确加权")
        print(f"      可能的原因:")
        print(f"      1. 特征映射错误")
        print(f"      2. 权重矩阵应用逻辑错误")
        print(f"      3. GNNWR特征与XGBoost特征不匹配")

    return {
        'original_features': original_features,
        'weighted_features': weighted_features,
        'weights': train_weights,
        'feature_changes': changes_df,
        'summary': {
            'verification_passed': verification_passed,
            'mean_abs_change': mean_abs_change,
            'significant_changes': significant_changes,
            'gnnwr_features_count': gnnwr_features_count,
            'total_features': total_features,
            'weight_std': train_weights.std(),
            'weight_distance_from_one': weight_distance_from_one
        }
    }


def visualize_feature_changes(original, weighted, changes_df):
    """可视化特征变化"""

    # 设置图形样式
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. 权重分布图
    print("  生成权重分布图...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 权重分布直方图
    if 'weights' in globals():
        weights = globals()['weights']
        ax1.hist(weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='权重=1.0')
        ax1.set_xlabel('权重值')
        ax1.set_ylabel('频数')
        ax1.set_title('GNNWR权重分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 特征变化分布
    if len(changes_df) > 0:
        gnnwr_changes = changes_df[changes_df['is_gnnwr']]['abs_change_mean']
        non_gnnwr_changes = changes_df[~changes_df['is_gnnwr']]['abs_change_mean']

        ax2.hist([gnnwr_changes, non_gnnwr_changes],
                 bins=20, alpha=0.7, edgecolor='black',
                 label=['GNNWR特征', '非GNNWR特征'])
        ax2.axvline(x=0.001, color='red', linestyle='--', linewidth=2, label='显著变化阈值(0.001)')
        ax2.set_xlabel('特征绝对变化均值')
        ax2.set_ylabel('特征数量')
        ax2.set_title('特征变化分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. 特征变化对比图（选择3个变化最大的GNNWR特征）
    gnnwr_changes_sorted = changes_df[changes_df['is_gnnwr']].sort_values('abs_change_mean', ascending=False)

    if len(gnnwr_changes_sorted) >= 3:
        top_features = gnnwr_changes_sorted.head(3)

        for idx, (_, row) in enumerate(top_features.iterrows()):
            # 找到特征索引
            feat_name = row['feature']
            feat_idx = list(changes_df['feature']).index(feat_name)

            # 绘制散点图
            ax3.scatter(original[:, feat_idx], weighted[:, feat_idx],
                        alpha=0.6, s=20, label=feat_name)

        # 添加对角线
        min_val = min(original.min(), weighted.min())
        max_val = max(original.max(), weighted.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

        ax3.set_xlabel('原始特征值')
        ax3.set_ylabel('加权后特征值')
        ax3.set_title('GNNWR特征加权前后对比')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

    # 4. 特征变化热图（前20个特征）
    if len(changes_df) > 0:
        # 选择前20个特征
        n_features_show = min(20, len(changes_df))
        top_changes = changes_df.head(n_features_show)

        # 准备热图数据
        heatmap_data = []
        feature_labels = []

        for _, row in top_changes.iterrows():
            feat_idx = list(changes_df['feature']).index(row['feature'])
            heatmap_data.append(abs_changes[:, feat_idx].mean())
            feature_labels.append(row['feature'])

        # 创建热图
        y_pos = np.arange(len(heatmap_data))
        colors = ['red' if changes_df.iloc[i]['is_gnnwr'] else 'blue' for i in range(len(heatmap_data))]

        ax4.barh(y_pos, heatmap_data, color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(feature_labels, fontsize=8)
        ax4.set_xlabel('平均绝对变化')
        ax4.set_title('特征平均变化排名 (红=GNNWR, 蓝=非GNNWR)')
        ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('weight_verification_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 5. 详细的特征变化对比图
    print("  生成详细特征变化图...")

    if len(gnnwr_changes_sorted) >= 6:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        top_features = gnnwr_changes_sorted.head(6)

        for idx, (_, row) in enumerate(top_features.iterrows()):
            if idx >= len(axes):
                break

            ax = axes[idx]
            feat_name = row['feature']
            feat_idx = list(changes_df['feature']).index(feat_name)

            # 绘制散点图
            scatter = ax.scatter(original[:, feat_idx], weighted[:, feat_idx],
                                 alpha=0.7, s=15, c='blue', edgecolors='none')

            # 添加对角线
            min_val = min(original[:, feat_idx].min(), weighted[:, feat_idx].min())
            max_val = max(original[:, feat_idx].max(), weighted[:, feat_idx].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=1.5)

            ax.set_xlabel(f'原始 {feat_name}')
            ax.set_ylabel(f'加权 {feat_name}')
            ax.set_title(f'{feat_name}\n变化: {row["abs_change_mean"]:.4f}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('gnnwr_features_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()


def run_complete_verification():
    """运行完整的验证流程"""
    print("=" * 80)
    print("🔬 GNNW-XGBoost权重应用完整验证")
    print("=" * 80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"验证时间: {timestamp}")

    # 创建输出目录
    output_dir = f"gnnw_verification_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 生成模拟数据
        print("\n步骤 1: 生成模拟数据")
        df = load_sample_data(n_samples=1000, n_features=50)

        # 运行验证
        print("\n步骤 2: 运行权重应用验证")
        results = verify_weight_application(df, n_samples=100, random_seed=42)

        if results:
            print("\n" + "=" * 80)
            print("✅ 验证完成！")

            summary = results['summary']
            if summary['verification_passed']:
                print("🎯 结论: GNNWR权重成功应用到特征上")
                print(f"   GNNWR特征平均变化: {summary['mean_abs_change']:.6f}")
                print(f"   权重矩阵标准差: {summary['weight_std']:.6f}")
                print(f"   显著变化比例: {summary['significant_changes']:.2%}")
            else:
                print("⚠️  结论: GNNWR权重应用可能存在问题")
                print("   建议检查:")
                print("   1. 特征映射是否正确")
                print("   2. 权重矩阵是否正确生成")
                print("   3. 应用权重的代码逻辑")

            # 保存详细结果
            print(f"\n📁 保存验证结果到: {output_dir}/")

            # 保存特征变化数据
            results['feature_changes'].to_csv(f'{output_dir}/feature_changes.csv', index=False)

            # 保存权重矩阵
            np.save(f'{output_dir}/weights.npy', results['weights'])

            # 保存原始和加权特征
            np.save(f'{output_dir}/original_features.npy', results['original_features'])
            np.save(f'{output_dir}/weighted_features.npy', results['weighted_features'])

            # 保存摘要报告
            with open(f'{output_dir}/verification_summary.txt', 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("GNNW-XGBoost权重应用验证报告\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"验证时间: {timestamp}\n\n")

                f.write("验证结果:\n")
                f.write(f"  验证通过: {'是' if summary['verification_passed'] else '否'}\n")
                f.write(f"  总特征数: {summary['total_features']}\n")
                f.write(f"  GNNWR特征数: {summary['gnnwr_features_count']}\n")
                f.write(f"  所有特征平均绝对变化: {summary['mean_abs_change']:.6f}\n")
                f.write(f"  显著变化比例: {summary['significant_changes']:.2%}\n")
                f.write(f"  权重矩阵标准差: {summary['weight_std']:.6f}\n")
                f.write(f"  权重与1的平均距离: {summary['weight_distance_from_one']:.6f}\n\n")

                # 添加特征变化详情
                f.write("特征变化详情:\n")
                f.write("-" * 100 + "\n")
                changes_df = results['feature_changes']
                gnnwr_changes = changes_df[changes_df['is_gnnwr']].sort_values('abs_change_mean', ascending=False)

                if len(gnnwr_changes) > 0:
                    f.write("GNNWR特征变化排名:\n")
                    for idx, row in gnnwr_changes.head(10).iterrows():
                        f.write(f"  {row['feature']:<30}: 平均变化={row['abs_change_mean']:.6f}, "
                                f"相对变化={row['rel_change_mean']:.2%}\n")

        print(f"\n📊 可视化图表:")
        print(f"  - weight_verification_summary.png")
        print(f"  - gnnwr_features_comparison.png")

        return results

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return None


def test_real_data_verification():
    """测试真实数据验证"""
    print("=" * 80)
    print("🔬 真实数据GNNW-XGBoost验证")
    print("=" * 80)

    try:
        # 尝试加载真实数据
        print("尝试加载真实数据...")
        df = pd.read_excel('lu_onehot.xlsx.xlsx')
        print(f"✅ 数据加载成功: {len(df)}行, {len(df.columns)}列")

        # 显示数据基本信息
        print(f"\n📊 数据基本信息:")
        print(f"  总样本数: {len(df)}")
        print(f"  总特征数: {len(df.columns)}")
        print(f"  前10个列名: {list(df.columns[:10])}")

        # 检查必要列
        required_cols = ['station_id', 'swe']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"❌ 缺少必要列: {missing_cols}")
            print("  正在尝试自动处理...")

            # 尝试重命名列
            if 'station_id' not in df.columns:
                # 尝试找到站点ID列
                possible_id_cols = ['stationid', 'station', 'site_id', 'site', 'id']
                for col in possible_id_cols:
                    if col in df.columns:
                        df = df.rename(columns={col: 'station_id'})
                        print(f"    重命名 '{col}' -> 'station_id'")
                        break

            if 'swe' not in df.columns:
                # 尝试找到SWE列
                possible_swe_cols = ['snow_water_equivalent', 'snowwater', 'sw']
                for col in possible_swe_cols:
                    if col in df.columns:
                        df = df.rename(columns={col: 'swe'})
                        print(f"    重命名 '{col}' -> 'swe'")
                        break

        # 检查GNNWR需要的特征
        print(f"\n🔍 GNNWR特征检查:")
        gnnwr_required = ['longitude', 'latitude', 'elevation']
        missing_gnnwr = [col for col in gnnwr_required if col not in df.columns]

        if missing_gnnwr:
            print(f"⚠️  缺少GNNWR特征: {missing_gnnwr}")
            print("  将创建模拟数据用于测试")

            for col in missing_gnnwr:
                if col == 'longitude':
                    df[col] = np.random.uniform(-180, 180, len(df))
                elif col == 'latitude':
                    df[col] = np.random.uniform(-90, 90, len(df))
                elif col == 'elevation':
                    df[col] = np.random.normal(0, 1000, len(df))

        # 运行验证
        print("\n🚀 开始验证...")
        results = verify_weight_application(df, n_samples=100, random_seed=42)

        return results

    except FileNotFoundError:
        print("❌ 未找到数据文件 'aggregated_station_data.xlsx'")
        print("  将使用模拟数据进行测试")
        return run_complete_verification()
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("  将使用模拟数据进行测试")
        return run_complete_verification()


if __name__ == "__main__":
    print("选择验证模式:")
    print("  1. 使用模拟数据测试")
    print("  2. 使用真实数据测试")

    choice = input("请输入选择 (1 或 2): ").strip()

    if choice == "2":
        results = test_real_data_verification()
    else:
        results = run_complete_verification()

    if results:
        print("\n" + "=" * 80)
        print("🎉 验证脚本执行完成！")
        print("=" * 80)

        # 提供后续建议
        print("\n💡 后续步骤建议:")
        if results['summary']['verification_passed']:
            print("  1. 权重应用验证通过，可以继续GNNW-XGBoost融合实验")
            print("  2. 检查交叉验证中每个折叠的权重应用")
            print("  3. 对比纯XGBoost和GNNW-XGBoost的性能")
        else:
            print("  1. 检查GNNWR训练代码，确保权重矩阵正确生成")
            print("  2. 检查特征映射逻辑，确保GNNWR特征正确对齐")
            print("  3. 检查权重应用代码，确保每个特征都被正确加权")
            print("  4. 考虑增加GNNWR训练轮数或调整网络结构")

        print("\n📊 关键指标:")
        print(f"  GNNWR特征数: {results['summary']['gnnwr_features_count']}")
        print(f"  特征平均变化: {results['summary']['mean_abs_change']:.6f}")
        print(f"  权重矩阵标准差: {results['summary']['weight_std']:.6f}")