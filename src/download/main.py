
import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


# 1. 数据加载与预处理
def load_data():
    # 假设有多个SWE数据源（CSV格式）和地理空间数据（Shapefile）
    swe_satellite = pd.read_csv('satellite_swe.csv')  # 卫星SWE数据
    swe_model = pd.read_csv('model_swe.csv')  # 模型模拟SWE
    ground_truth = pd.read_csv('ground_measurements.csv')  # 地面观测真值

    # 地理空间数据 (需包含坐标或网格ID)
    geo_data = gpd.read_file('terrain_features.shp')

    # 根据坐标/ID合并所有数据
    merged_data = swe_satellite.merge(swe_model, on=['latitude', 'longitude', 'date'])
    merged_data = merged_data.merge(ground_truth, on=['latitude', 'longitude', 'date'])
    merged_data = merged_data.merge(geo_data, on=['latitude', 'longitude'])

    return merged_data


# 2. 特征工程
def create_features(df):
    # 地理空间特征（示例）
    df['northness'] = np.cos(np.radians(df['aspect']))  # 坡向转换
    df['eastness'] = np.sin(np.radians(df['aspect']))

    # 地形交互特征
    df['elevation_slope'] = df['elevation'] * df['slope']

    # 时间特征（如果数据包含时间维度）
    df['doy'] = pd.to_datetime(df['date']).dt.dayofyear
    df['month'] = pd.to_datetime(df['date']).dt.month

    # 多源SWE特征直接使用列名
    return df


# 3. 构建训练集
def prepare_training_data(df):
    # 定义特征和目标变量
    swe_sources = ['swe_sat', 'swe_model']  # 多源SWE数据列名
    geo_features = ['elevation', 'slope', 'northness', 'eastness', 'landcover']
    time_features = ['doy', 'month'] if 'date' in df.columns else []

    features = swe_sources + geo_features + time_features
    target = 'swe_ground_truth'  # 地面观测作为训练目标

    # 移除缺失值
    df_clean = df.dropna(subset=[target] + features)

    X = df_clean[features]
    y = df_clean[target]

    return X, y


# 4. 训练XGBoost模型
def train_xgboost(X, y):
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 构建预处理和模型管道
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            random_state=42
        ))
    ])

    # 训练（使用早停）
    pipeline.fit(
        X_train, y_train,
        xgb__eval_set=[(X_test, y_test)],
        xgb__verbose=False
    )

    # 评估
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return pipeline, X_test, y_test


# 5. 生成新SWE预测
def predict_new_swe(model, new_data_path):
    # 新数据加载（需包含所有特征）
    new_data = pd.read_csv(new_data_path)
    new_data = create_features(new_data)  # 应用相同的特征工程

    # 确保特征与训练时一致
    required_features = model.named_steps['xgb'].get_booster().feature_names
    new_data = new_data[required_features]

    # 预测
    predictions = model.predict(new_data)
    return predictions


# 6. 结果可视化
def plot_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True SWE (mm)')
    plt.ylabel('Predicted SWE (mm)')
    plt.title('XGBoost SWE Fusion Performance')
    plt.grid(True)
    plt.savefig('swe_fusion_scatter.png', dpi=300)


# 主流程
if __name__ == "__main__":
    # 加载和处理数据
    df = load_data()
    df = create_features(df)

    # 准备训练数据
    X, y = prepare_training_data(df)

    # 训练模型
    model, X_test, y_test = train_xgboost(X, y)

    # 可视化验证结果
    y_pred = model.predict(X_test)
    plot_results(y_test, y_pred)

    # 对新地理区域预测
    new_predictions = predict_new_swe(model, 'new_region_data.csv')
    print(f"Generated {len(new_predictions)} new SWE predictions")

    # 可选：保存预测结果
    output_df = pd.DataFrame({
        'latitude': X_test['latitude'],
        'longitude': X_test['longitude'],
        'predicted_swe': y_pred
    })
    output_df.to_csv('fused_swe_predictions.csv', index=False)

