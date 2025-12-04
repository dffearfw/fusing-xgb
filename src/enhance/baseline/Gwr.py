import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

# ===================== 1. 加载数据 =====================
excel_path = 'D:/pyworkspace/fusing xgb/src/pre-process/aggregated_station_data.xlsx'
df = pd.read_excel(excel_path)
geometry = [Point(xy) for xy in zip(df['X'], df['Y'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
gdf = gdf.to_crs("EPSG:32648")

# 划分训练/测试（比如 80% train）
np.random.seed(42)
train_mask = np.random.rand(len(gdf)) < 0.8
train_gdf = gdf[train_mask].copy()
test_gdf = gdf[~train_mask].copy()

print(f"✅ 训练集: {len(train_gdf)}，测试集: {len(test_gdf)}")

# ===================== 2. 变量准备 =====================
x_cols = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation',
          'std_slope', 'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2',
          'std_high', 'std_aspect', 'glsnow', 'cswe', 'snow_depth_snow_depth',
          'ERA5温度_ERA5温度', 'era5_swe', 'doy', 'gldas', 'year', 'month', 'scp_start',
          'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da', 'db', 'dc', 'dd'] + \
         [f'landuse_{i}' for i in [11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 46, 51, 52, 53, 62, 63, 64]]

y_col = 'swe'

# 过滤缺失列
x_cols = [col for col in x_cols if col in gdf.columns]


# 提取坐标和特征
def get_coords_and_X(gdf):
    coords = np.array([[p.x, p.y] for p in gdf.geometry])
    X = gdf[x_cols].values
    y = gdf[y_col].values
    return coords, X, y


train_coords, X_train, y_train = get_coords_and_X(train_gdf)
test_coords, X_test, y_test = get_coords_and_X(test_gdf)


# ===================== 3. 手动实现 GWR 预测函数（关键！） =====================
def gwr_predict(train_coords, X_train, y_train, test_coords, bw, kernel='gaussian'):
    """
    手动实现 GWR 对测试集的预测
    参数:
        bw: 带宽（经纬度建议 0.001 ~ 0.01）
    """
    preds = []
    n_test = len(test_coords)

    # 预计算所有距离矩阵（节省重复计算）
    dist_matrix = cdist(test_coords, train_coords, metric='euclidean')  # shape: (n_test, n_train)

    for i in range(n_test):
        d = dist_matrix[i]  # 当前测试点到所有训练点的距离

        # 高斯核权重
        if kernel == 'gaussian':
            weights = np.exp(-0.5 * (d / bw) ** 2)
        else:
            raise NotImplementedError("只支持 gaussian 核")

        # 如果所有权重接近0，跳过（避免奇异矩阵）
        if weights.max() < 1e-10:
            preds.append(y_train.mean())
            continue

        # 加权线性回归
        try:
            # 构造加权设计矩阵
            W = np.diag(weights)
            XtWX = X_train.T @ W @ X_train
            XtWy = X_train.T @ W @ y_train

            # 求解 (X^T W X)^{-1} X^T W y
            beta = np.linalg.solve(XtWX, XtWy)
            pred = X_test[i] @ beta
            preds.append(pred)
        except np.linalg.LinAlgError:
            # 矩阵奇异时回退到全局均值
            preds.append(y_train.mean())

    return np.array(preds)


# ===================== 4. 设置带宽（关键！） =====================
# 经纬度坐标：带宽单位是“度”，典型值 0.001 ~ 0.01
# 投影坐标（米）：典型值 100 ~ 10000
BANDWIDTH = 0.01  # 你可以调整这个值（越小越局部，越大越平滑）

print(f"🔧 使用带宽: {BANDWIDTH}")

# ===================== 5. 执行预测（单进程，省内存） =====================
print("⏳ 正在对测试集进行 GWR 预测（Hold-Out）...")
y_pred = gwr_predict(
    train_coords, X_train, y_train,
    test_coords, bw=BANDWIDTH
)

# ===================== 6. 评估结果 =====================
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 GWR Hold-Out 预测结果:")
print(f"   测试集 R² = {r2:.4f}")
print(f"   测试集 RMSE = {rmse:.4f}")

# 同时对比 OLS
ols = LinearRegression().fit(X_train, y_train)
ols_pred = ols.predict(X_test)
ols_r2 = r2_score(y_test, ols_pred)
ols_rmse = np.sqrt(mean_squared_error(y_test, ols_pred))

print(f"\n📈 OLS Hold-Out 结果:")
print(f"   测试集 R² = {ols_r2:.4f}")
print(f"   测试集 RMSE = {ols_rmse:.4f}")

# ===================== 7. 保存结果 =====================
test_gdf['gwr_pred'] = y_pred
test_gdf[['swe', 'gwr_pred']].to_csv('gwr_holdout_predictions.csv', index=False)
print("\n💾 Hold-Out 预测结果已保存到 gwr_holdout_predictions.csv")