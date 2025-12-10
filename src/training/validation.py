import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gnnw_xgboost_trainer import GNNW_XGBoostTrainer


def verify_weight_application(df, n_samples=100):
    """严格验证权重是否被正确应用到特征上"""
    print("=" * 80)
    print("🧪 GNNW-XGBoost权重应用验证测试")
    print("=" * 80)

    # 创建训练器实例
    trainer = GNNW_XGBoostTrainer(use_gnnwr=True)

    # 预处理数据
    print("\n1. 数据预处理...")
    X, y, station_groups, year_groups, gnnwr_data = trainer.preprocess_data(df)

    print(f"  原始数据形状: X={X.shape}, y={y.shape}")
    print(f"  GNNWR数据形状: {gnnwr_data.shape}")
    print(f"  特征列数: {len(trainer.feature_columns)}")
    print(f"  GNNWR特征列: {len(trainer.gnnwr_x_columns)}")

    # 只取前n_samples个样本进行测试
    X_test = X[:n_samples]
    y_test = y[:n_samples]
    gnnwr_data_test = gnnwr_data.iloc[:n_samples].copy()

    print(f"\n2. 测试样本: 使用前{n_samples}个样本")

    # 获取权重矩阵（用全部样本作为训练集，自己作为验证集来获得权重）
    print("\n3. 获取权重矩阵...")
    train_weights, val_weights = trainer._train_gnnwr_for_fold_debug(gnnwr_data_test, gnnwr_data_test.head(1))

    if train_weights is None:
        print("❌ 无法获取权重矩阵")
        return

    print(f"\n4. 权重矩阵分析:")
    print(f"   形状: {train_weights.shape}")
    print(f"   均值: {train_weights.mean():.6f}")
    print(f"   标准差: {train_weights.std():.6f}")
    print(f"   范围: [{train_weights.min():.6f}, {train_weights.max():.6f}]")
    print(f"   负权重比例: {(train_weights < 0).mean():.2%}")

    # 检查权重是否显著不同于1（如果是1，乘以权重就没变化）
    weight_distance_from_one = np.abs(train_weights - 1).mean()
    print(f"   权重与1的平均距离: {weight_distance_from_one:.6f}")

    if weight_distance_from_one < 0.01:
        print("   ⚠️ 警告: 权重非常接近1，加权可能没有效果")
    else:
        print(f"   ✅ 权重与1有显著差异，加权会有效果")

    # 5. 应用权重
    print("\n5. 应用权重到特征...")

    # 原始特征
    original_features = X_test.copy()

    # 应用权重
    weighted_features = trainer._apply_gnnwr_weights_with_debug(
        original_features, train_weights,
        trainer.feature_columns, trainer.gnnwr_x_columns
    )

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

        # 统计变化
        if is_gnnwr_feature:
            feat_change_mean = changes[:, i].mean()
            feat_change_std = changes[:, i].std()
            feat_abs_change_mean = abs_changes[:, i].mean()
            feat_rel_change_mean = relative_changes[:, i].mean()

            feature_changes.append({
                'feature': feat_name,
                'is_gnnwr': is_gnnwr_feature,
                'change_mean': feat_change_mean,
                'change_std': feat_change_std,
                'abs_change_mean': feat_abs_change_mean,
                'rel_change_mean': feat_rel_change_mean
            })

    # 创建DataFrame显示结果
    changes_df = pd.DataFrame(feature_changes)

    print("\n  GNNWR特征的变化统计:")
    print("  " + "-" * 100)

    # 按绝对变化排序
    sorted_changes = changes_df.sort_values('abs_change_mean', ascending=False)

    for idx, row in sorted_changes.head(10).iterrows():
        print(f"    {row['feature']:<30} 平均变化: {row['change_mean']:+.6f} (±{row['change_std']:.6f}), "
              f"绝对变化: {row['abs_change_mean']:.6f}, 相对变化: {row['rel_change_mean']:.2%}")

    # 7. 可视化变化
    print("\n7. 可视化特征变化...")
    visualize_feature_changes(original_features, weighted_features, changes_df, n_top=10)

    # 8. 统计总结
    print("\n8. 验证结果总结:")

    total_features = original_features.shape[1]
    gnnwr_features_count = sum([1 for f in trainer.feature_columns if f in trainer.gnnwr_x_columns])
    non_gnnwr_features_count = total_features - gnnwr_features_count

    print(f"   总特征数: {total_features}")
    print(f"   GNNWR特征数: {gnnwr_features_count} (会加权)")
    print(f"   非GNNWR特征数: {non_gnnwr_features_count} (保持不变)")

    # 检查是否有特征确实被加权了
    mean_abs_change = abs_changes.mean()
    significant_changes = (abs_changes > 1e-6).sum() / abs_changes.size

    print(f"\n   特征平均绝对变化: {mean_abs_change:.6f}")
    print(f"   有显著变化(>1e-6)的比例: {significant_changes:.2%}")

    if mean_abs_change > 0.001 and significant_changes > 0.1:
        print(f"\n   ✅ 验证通过: 权重已成功应用到特征上")
        print(f"      平均变化: {mean_abs_change:.6f}")
        print(f"      显著变化比例: {significant_changes:.2%}")
    else:
        print(f"\n   ❌ 验证失败: 特征变化太小或没有变化")
        print(f"      请检查权重矩阵和特征对齐")

    return {
        'original_features': original_features,
        'weighted_features': weighted_features,
        'weights': train_weights,
        'feature_changes': changes_df,
        'summary': {
            'mean_abs_change': mean_abs_change,
            'significant_changes': significant_changes,
            'gnnwr_features_count': gnnwr_features_count,
            'total_features': total_features
        }
    }


def visualize_feature_changes(original, weighted, changes_df, n_top=10):
    """可视化特征变化"""

    # 选择变化最大的特征
    top_features = changes_df.nlargest(min(n_top, len(changes_df)), 'abs_change_mean')

    fig, axes = plt.subplots(2, min(3, len(top_features)), figsize=(15, 8))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top_features.iterrows()):
        if idx >= len(axes):
            break

        # 找到特征索引
        feat_name = row['feature']
        feat_idx = list(changes_df['feature']).index(feat_name)

        ax = axes[idx]

        # 原始特征和加权特征
        ax.scatter(original[:, feat_idx], weighted[:, feat_idx], alpha=0.6, s=20)

        # 对角线（y=x线）
        min_val = min(original[:, feat_idx].min(), weighted[:, feat_idx].min())
        max_val = max(original[:, feat_idx].max(), weighted[:, feat_idx].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

        ax.set_xlabel(f'原始 {feat_name}')
        ax.set_ylabel(f'加权 {feat_name}')
        ax.set_title(f'{feat_name}\n变化: {row["abs_change_mean"]:.4f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('weight_application_verification.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 创建权重分布图
    plt.figure(figsize=(10, 6))

    # 提取所有权重值
    all_weights = []
    for idx, row in changes_df.iterrows():
        feat_idx = list(changes_df['feature']).index(row['feature'])
        all_weights.extend(row['abs_change_mean'])

    # 绘制权重分布
    plt.hist(all_weights, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('特征变化绝对值')
    plt.ylabel('频数')
    plt.title('GNNWR权重应用后特征变化分布')
    plt.grid(True, alpha=0.3)
    plt.savefig('feature_changes_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


# 在GNNW_XGBoostTrainer类中添加调试方法
def _apply_gnnwr_weights_with_debug(self, X, weights, feature_columns, gnnwr_x_columns):
    """带调试信息的权重应用"""
    if weights is None:
        print("❌ 权重矩阵为None")
        return X

    print(f"\n🔧 权重应用详细信息:")
    print(f"  输入X形状: {X.shape}")
    print(f"  权重矩阵形状: {weights.shape}")
    print(f"  XGBoost特征数: {len(feature_columns)}")
    print(f"  GNNWR特征数: {len(gnnwr_x_columns)}")

    # 创建特征映射
    feature_to_gnnwr = {}
    feature_map_info = []

    for i, feat in enumerate(feature_columns):
        if feat in gnnwr_x_columns:
            gnnwr_idx = gnnwr_x_columns.index(feat)
            feature_to_gnnwr[i] = gnnwr_idx
            feature_map_info.append(f"{feat} (XGB idx:{i} -> GNNWR idx:{gnnwr_idx})")

    print(f"\n  特征映射 ({len(feature_to_gnnwr)}个匹配特征):")
    for info in feature_map_info[:10]:  # 只显示前10个
        print(f"    {info}")

    if len(feature_map_info) > 10:
        print(f"    ... 还有{len(feature_map_info) - 10}个特征")

    # 应用权重
    X_weighted = X.copy()
    total_change = 0

    for feat_idx, gnnwr_idx in feature_to_gnnwr.items():
        if gnnwr_idx < weights.shape[1]:
            original_values = X[:, feat_idx]
            weight_values = weights[:, gnnwr_idx]
            weighted_values = original_values * weight_values

            # 计算变化
            change = np.abs(weighted_values - original_values).mean()
            total_change += change

            # 如果变化显著，打印详细信息
            if change > 0.01:  # 变化大于0.01
                print(f"\n   显著变化特征: {feature_columns[feat_idx]}")
                print(f"     原始值范围: [{original_values.min():.3f}, {original_values.max():.3f}]")
                print(f"     权重值范围: [{weight_values.min():.3f}, {weight_values.max():.3f}]")
                print(f"     加权值范围: [{weighted_values.min():.3f}, {weighted_values.max():.3f}]")
                print(f"     平均绝对变化: {change:.6f}")

            X_weighted[:, feat_idx] = weighted_values

    print(f"\n  总平均变化: {total_change / len(feature_to_gnnwr):.6f}")
    return X_weighted


def _train_gnnwr_for_fold_debug(self, train_data, val_data):
    """带调试信息的GNNWR训练"""
    print("🔬 GNNWR训练调试模式...")
    print(f"  训练数据形状: {train_data.shape}")
    print(f"  验证数据形状: {val_data.shape}")

    # 调用原始方法
    train_weights, val_weights = self._train_gnnwr_for_fold(train_data, val_data)

    if train_weights is not None:
        print(f"  训练权重矩阵统计:")
        print(f"    形状: {train_weights.shape}")
        print(f"    均值: {train_weights.mean():.6f}")
        print(f"    标准差: {train_weights.std():.6f}")
        print(f"    最小权重: {train_weights.min():.6f}")
        print(f"    最大权重: {train_weights.max():.6f}")

        # 检查权重分布
        unique_weights = np.unique(train_weights)
        print(f"    唯一权重值数量: {len(unique_weights)}")

        # 如果权重太单一，可能有错误
        if len(unique_weights) < 10:
            print(f"    权重值: {unique_weights[:10]}")

    return train_weights, val_weights


# 将调试方法添加到类中
GNNW_XGBoostTrainer._apply_gnnwr_weights_with_debug = _apply_gnnwr_weights_with_debug
GNNW_XGBoostTrainer._train_gnnwr_for_fold_debug = _train_gnnwr_for_fold_debug


# 主验证函数
def main_verification():
    """主验证函数"""
    print("开始GNNW-XGBoost权重应用验证...")

    # 加载数据
    df = pd.read_excel('aggregated_station_data.xlsx')

    # 运行验证
    results = verify_weight_application(df, n_samples=100)

    if results:
        print("\n" + "=" * 80)
        print("✅ 验证完成！")

        summary = results['summary']
        if summary['mean_abs_change'] > 0.001 and summary['significant_changes'] > 0.1:
            print("🎯 结论: 权重已成功应用到特征上")
            print(f"   平均特征变化: {summary['mean_abs_change']:.6f}")
            print(f"   显著变化比例: {summary['significant_changes']:.2%}")
            print(f"   {summary['gnnwr_features_count']}/{summary['total_features']}个特征被加权")
        else:
            print("⚠️  警告: 特征变化太小")
            print("   可能的原因:")
            print("   1. GNNWR没有学到有效的权重")
            print("   2. 权重值接近1.0")
            print("   3. 特征和权重没有正确对齐")
            print("   4. GNNWR训练轮数太少")

        # 保存验证结果
        pd.DataFrame(results['feature_changes']).to_csv('feature_changes_analysis.csv', index=False)

        print("\n📁 验证结果已保存:")
        print("   - weight_application_verification.png (可视化)")
        print("   - feature_changes_distribution.png (分布图)")
        print("   - feature_changes_analysis.csv (详细数据)")

    return results


if __name__ == "__main__":
    main_verification()