import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def find_combined_excel(combined_dir):
    """在combined目录中查找Excel文件"""
    import os
    import glob

    # 查找所有Excel文件
    excel_files = glob.glob(os.path.join(combined_dir, "*.xlsx"))
    excel_files.extend(glob.glob(os.path.join(combined_dir, "*.xls")))

    if not excel_files:
        raise FileNotFoundError(f"在目录 {combined_dir} 中未找到Excel文件")

    # 返回第一个找到的Excel文件
    return excel_files[0]


def load_data(config_path):
    """
    加载数据并预处理
    """
    try:
        # 加载配置文件
        config = load_config(config_path)

        # 获取combined目录路径
        combined_dir = config['outputs']['combined']
        if not combined_dir.startswith('E:/'):
            combined_dir = config['input_root'] + '/' + combined_dir

        print(f"查找数据文件在: {combined_dir}")

        # 查找Excel文件
        excel_file_path = find_combined_excel(combined_dir)
        print(f"找到数据文件: {excel_file_path}")

        df = pd.read_excel(excel_file_path)
        print(f"数据加载成功，形状: {df.shape}")

        # 检查必要的列是否存在
        required_cols = ['日期', '经度', '纬度', 'snowdepth', 'DOY', 'swe']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")

        # 处理日期
        df['日期'] = pd.to_datetime(df['日期'])
        df['年份'] = df['日期'].dt.year
        df['月份'] = df['日期'].dt.month

        # 数据质量检查
        print(f"数据年份范围: {df['年份'].min()} - {df['年份'].max()}")
        print(f"SWE统计 - 均值: {df['swe'].mean():.2f}, 标准差: {df['swe'].std():.2f}")
        print(f"雪深统计 - 均值: {df['snowdepth'].mean():.2f}, 标准差: {df['snowdepth'].std():.2f}")

        # 检查缺失值
        missing_data = df[required_cols].isnull().sum()
        if missing_data.any():
            print("缺失值统计:")
            print(missing_data[missing_data > 0])
            # 删除有缺失值的行
            df = df.dropna(subset=required_cols)
            print(f"删除缺失值后数据形状: {df.shape}")

        return df

    except Exception as e:
        print(f"数据加载失败: {e}")
        return None


def annual_cross_validation_xgboost(df, target_col='swe', year_col='年份'):
    """
    执行年度交叉验证的XGBoost建模
    """
    # 使用指定的特征列
    feature_cols = ['经度', '纬度', 'snowdepth', 'DOY']

    print(f"使用的特征: {feature_cols}")

    # 获取唯一的年份并排序
    years = sorted(df[year_col].unique())
    print(f"可用年份: {years}")

    # 存储每年的结果
    results = {
        'years': [],
        'train_r2': [], 'val_r2': [],
        'train_rmse': [], 'val_rmse': [],
        'train_mae': [], 'val_mae': [],
        'models': [],
        'feature_importances': [],
        'y_true': [], 'y_pred': []
    }

    # 年度交叉验证
    for test_year in years:
        print(f"\n=== 训练年份: {[y for y in years if y != test_year]}, 验证年份: {test_year} ===")

        # 分割数据
        train_mask = df[year_col] != test_year
        test_mask = df[year_col] == test_year

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, target_col]

        print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_test)}")

        if len(X_test) == 0:
            print(f"警告: {test_year}年没有验证数据，跳过")
            continue

        # 使用您指定的参数
        params = {
            'n_estimators': 60,
            'learning_rate': 0.17,
            'max_depth': 5,
            'min_child_weight': 5,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.5,
            'reg_alpha': 0.05,
            'random_state': 42
        }

        # 创建并训练模型
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 计算评估指标
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # 存储结果
        results['years'].append(test_year)
        results['train_r2'].append(train_r2)
        results['val_r2'].append(test_r2)
        results['train_rmse'].append(train_rmse)
        results['val_rmse'].append(test_rmse)
        results['train_mae'].append(train_mae)
        results['val_mae'].append(test_mae)
        results['models'].append(model)
        results['feature_importances'].append(
            dict(zip(feature_cols, model.feature_importances_))
        )
        results['y_true'].extend(y_test.values)
        results['y_pred'].extend(y_test_pred)

        print(f"训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        print(f"验证集 R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    return results, feature_cols


def plot_comprehensive_results(results, feature_cols):
    """
    绘制综合结果分析图
    """
    fig = plt.figure(figsize=(20, 16))

    # 1. 性能指标趋势
    ax1 = plt.subplot(3, 3, 1)
    years = results['years']
    ax1.plot(years, results['train_r2'], 'o-', label='训练集 R²', linewidth=2, markersize=8)
    ax1.plot(years, results['val_r2'], 's-', label='验证集 R²', linewidth=2, markersize=8)
    ax1.set_xlabel('验证年份')
    ax1.set_ylabel('R² 分数')
    ax1.set_title('年度交叉验证 - R² 分数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. RMSE趋势
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(years, results['train_rmse'], 'o-', label='训练集 RMSE', linewidth=2, markersize=8)
    ax2.plot(years, results['val_rmse'], 's-', label='验证集 RMSE', linewidth=2, markersize=8)
    ax2.set_xlabel('验证年份')
    ax2.set_ylabel('RMSE')
    ax2.set_title('年度交叉验证 - RMSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 特征重要性（平均）
    ax3 = plt.subplot(3, 3, 3)
    feature_importance_avg = pd.DataFrame(results['feature_importances']).mean().sort_values()
    feature_importance_avg.plot(kind='barh', ax=ax3)
    ax3.set_xlabel('平均特征重要性')
    ax3.set_title('XGBoost 特征重要性（跨年份平均）')

    # 4. 预测值与真实值散点图（所有年份）
    ax4 = plt.subplot(3, 3, 4)
    y_true = results['y_true']
    y_pred = results['y_pred']
    ax4.scatter(y_true, y_pred, alpha=0.6, s=30)
    ax4.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
    ax4.set_xlabel('真实值')
    ax4.set_ylabel('预测值')
    overall_r2 = r2_score(y_true, y_pred)
    ax4.set_title(f'预测vs真实 (总体 R²: {overall_r2:.3f})')

    # 5. 残差分析
    ax5 = plt.subplot(3, 3, 5)
    residuals = np.array(y_pred) - np.array(y_true)
    ax5.scatter(y_pred, residuals, alpha=0.6, s=30)
    ax5.axhline(y=0, color='r', linestyle='--')
    ax5.set_xlabel('预测值')
    ax5.set_ylabel('残差')
    ax5.set_title('残差分析图')

    # 6. 年度性能热力图
    ax6 = plt.subplot(3, 3, 6)
    performance_data = {
        'R²': results['val_r2'],
        'RMSE': results['val_rmse'],
        'MAE': results['val_mae']
    }
    performance_df = pd.DataFrame(performance_data, index=years)
    sns.heatmap(performance_df.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6)
    ax6.set_title('年度性能热力图')

    # 7. 特征重要性年度变化
    ax7 = plt.subplot(3, 3, 7)
    importance_df = pd.DataFrame(results['feature_importances'], index=years)
    importance_df.plot(kind='line', marker='o', ax=ax7)
    ax7.set_xlabel('年份')
    ax7.set_ylabel('特征重要性')
    ax7.set_title('特征重要性年度变化')
    ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 8. 误差分布
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax8.axvline(x=0, color='r', linestyle='--')
    ax8.set_xlabel('残差')
    ax8.set_ylabel('频数')
    ax8.set_title('误差分布')

    plt.tight_layout()
    plt.show()

    return overall_r2


def analyze_model_performance(results):
    """
    深入分析模型性能
    """
    print("\n" + "=" * 60)
    print("模型性能深度分析")
    print("=" * 60)

    # 基本统计
    overall_results = pd.DataFrame({
        '验证年份': results['years'],
        '训练集_R2': results['train_r2'],
        '验证集_R2': results['val_r2'],
        '训练集_RMSE': results['train_rmse'],
        '验证集_RMSE': results['val_rmse'],
        '训练集_MAE': results['train_mae'],
        '验证集_MAE': results['val_mae']
    })

    print(overall_results.round(4))

    # 性能统计
    val_r2_scores = results['val_r2']
    val_rmse_scores = results['val_rmse']
    val_mae_scores = results['val_mae']

    print(f"\n验证集性能统计:")
    print(f"R² - 平均值: {np.mean(val_r2_scores):.4f} ± {np.std(val_r2_scores):.4f}")
    print(f"R² - 范围: [{np.min(val_r2_scores):.4f}, {np.max(val_r2_scores):.4f}]")
    print(f"RMSE - 平均值: {np.mean(val_rmse_scores):.4f} ± {np.std(val_rmse_scores):.4f}")
    print(f"MAE - 平均值: {np.mean(val_mae_scores):.4f} ± {np.std(val_mae_scores):.4f}")

    # 最佳和最差年份
    best_year_idx = np.argmax(val_r2_scores)
    worst_year_idx = np.argmin(val_r2_scores)

    print(f"\n最佳性能年份: {results['years'][best_year_idx]} (R²: {val_r2_scores[best_year_idx]:.4f})")
    print(f"最差性能年份: {results['years'][worst_year_idx]} (R²: {val_r2_scores[worst_year_idx]:.4f})")

    # 特征重要性分析
    print(f"\n特征重要性分析:")
    feature_importance_avg = pd.DataFrame(results['feature_importances']).mean().sort_values(ascending=False)
    for feature, importance in feature_importance_avg.items():
        print(f"  {feature}: {importance:.4f}")


def main():
    """
    主函数
    """
    # 配置文件路径
    config_path = "path.yaml"  # 替换为您的配置文件路径

    # 加载数据
    df = load_data(config_path)

    if df is None:
        print("数据加载失败，请检查文件路径和数据格式")
        return

    print("\n数据基本信息:")
    print(f"数据形状: {df.shape}")
    print(f"年份分布:\n{df['年份'].value_counts().sort_index()}")

    # 执行年度交叉验证
    print("\n开始年度交叉验证...")
    results, feature_cols = annual_cross_validation_xgboost(df)

    # 分析结果
    analyze_model_performance(results)

    # 绘制结果
    print("\n生成可视化结果...")
    overall_r2 = plot_comprehensive_results(results, feature_cols)

    # 保存最佳模型
    best_idx = np.argmax(results['val_r2'])
    best_model = results['models'][best_idx]
    best_year = results['years'][best_idx]

    print(f"\n最佳模型信息:")
    print(f"验证年份: {best_year}")
    print(f"验证集 R²: {results['val_r2'][best_idx]:.4f}")
    print(f"验证集 RMSE: {results['val_rmse'][best_idx]:.4f}")
    print(f"总体 R²: {overall_r2:.4f}")

    # 保存模型（可选）
    # import joblib
    # joblib.dump(best_model, f'xgboost_best_model_{best_year}.pkl')
    # print(f"最佳模型已保存为: xgboost_best_model_{best_year}.pkl")


if __name__ == "__main__":
    main()