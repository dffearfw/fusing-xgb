import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# 读取数据
# 假设您的Excel文件包含以下列：站点ID, 日期, 经度, 纬度, snowdepth, DOY, 其他特征, swe
def load_data(file_path):
    """
    加载数据并提取年份用于年度交叉验证
    """
    df = pd.read_excel(file_path)

    # 确保日期列是datetime格式
    df['日期'] = pd.to_datetime(df['日期'])

    # 提取年份作为分组依据
    df['年份'] = df['日期'].dt.year

    return df


# 年度交叉验证函数
def annual_cross_validation_xgboost(df, target_col='swe', year_col='年份'):
    """
    执行年度交叉验证的XGBoost建模
    """
    # 特征列（根据您的描述）
    feature_cols = ['经度', '纬度', 'snowdepth', 'DOY']

    # 获取唯一的年份并排序
    years = sorted(df[year_col].unique())
    print(f"可用年份: {years}")

    # 存储每年的结果
    results = {
        'years': [],
        'train_r2': [],
        'val_r2': [],
        'train_rmse': [],
        'val_rmse': [],
        'train_mae': [],
        'val_mae': [],
        'models': []
    }

    # 年度交叉验证：每次留出一年的数据作为验证集
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

        # 定义XGBoost参数
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

        print(f"训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        print(f"验证集 R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    return results, feature_cols


# 可视化结果
def plot_results(results):
    """
    绘制年度交叉验证结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # R² 分数
    axes[0, 0].plot(results['years'], results['train_r2'], 'o-', label='训练集 R²', linewidth=2)
    axes[0, 0].plot(results['years'], results['val_r2'], 's-', label='验证集 R²', linewidth=2)
    axes[0, 0].set_xlabel('验证年份')
    axes[0, 0].set_ylabel('R² 分数')
    axes[0, 0].set_title('年度交叉验证 - R² 分数')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # RMSE
    axes[0, 1].plot(results['years'], results['train_rmse'], 'o-', label='训练集 RMSE', linewidth=2)
    axes[0, 1].plot(results['years'], results['val_rmse'], 's-', label='验证集 RMSE', linewidth=2)
    axes[0, 1].set_xlabel('验证年份')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('年度交叉验证 - RMSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # MAE
    axes[1, 0].plot(results['years'], results['train_mae'], 'o-', label='训练集 MAE', linewidth=2)
    axes[1, 0].plot(results['years'], results['val_mae'], 's-', label='验证集 MAE', linewidth=2)
    axes[1, 0].set_xlabel('验证年份')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('年度交叉验证 - MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 特征重要性（使用最后一个模型）
    if results['models']:
        model = results['models'][-1]
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'])
        axes[1, 1].set_xlabel('特征重要性')
        axes[1, 1].set_title('XGBoost 特征重要性')

    plt.tight_layout()
    plt.show()


# 主函数
def main():
    # 加载数据（请替换为您的实际文件路径）
    file_path = "your_data_file.xlsx"  # 替换为您的Excel文件路径
    df = load_data(file_path)

    print("数据基本信息:")
    print(f"数据形状: {df.shape}")
    print(f"年份分布: {df['年份'].value_counts().sort_index()}")
    print(f"特征列: {['经度', '纬度', 'snowdepth', 'DOY']}")
    print(f"目标列: swe")

    # 执行年度交叉验证
    results, feature_cols = annual_cross_validation_xgboost(df)

    # 打印总体结果
    print("\n" + "=" * 50)
    print("年度交叉验证总体结果:")
    print("=" * 50)

    overall_results = pd.DataFrame({
        '验证年份': results['years'],
        '训练集_R2': results['train_r2'],
        '验证集_R2': results['val_r2'],
        '训练集_RMSE': results['train_rmse'],
        '验证集_RMSE': results['val_rmse'],
        '训练集_MAE': results['train_mae'],
        '验证集_MAE': results['val_mae']
    })

    print(overall_results)

    # 计算平均性能
    avg_val_r2 = np.mean(results['val_r2'])
    avg_val_rmse = np.mean(results['val_rmse'])
    avg_val_mae = np.mean(results['val_mae'])

    print(f"\n平均验证性能:")
    print(f"平均 R²: {avg_val_r2:.4f}")
    print(f"平均 RMSE: {avg_val_rmse:.4f}")
    print(f"平均 MAE: {avg_val_mae:.4f}")

    # 绘制结果
    plot_results(results)

    # 保存最佳模型（可以选择验证集性能最好的模型）
    best_idx = np.argmax(results['val_r2'])
    best_model = results['models'][best_idx]
    best_year = results['years'][best_idx]

    print(f"\n最佳模型对应验证年份: {best_year}, R²: {results['val_r2'][best_idx]:.4f}")

    # 可以保存模型用于后续预测
    # import joblib
    # joblib.dump(best_model, f'xgboost_model_{best_year}.pkl')


if __name__ == "__main__":
    main()