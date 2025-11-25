import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
data = pd.read_excel('processed_data.xlsx')
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data['swe'], kde=True)
plt.title('Original SWE Distribution')

# 如果严重偏斜，尝试对数变换
# 加1是为了处理swe=0的情况
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(data['swe']), kde=True)
plt.title('Log(SWE + 1) Distribution')
plt.show()


x_columns = [
    'aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation', 'std_slope',
    'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high', 'std_aspect', 'glsnow',
    'cswe', 'snow_depth_snow_depth', 'ERA5温度_ERA5温度', 'era5_swe', 'gldas',
    'scp_start', 'scp_end', 'd1', 'd2','da', 'db', 'dc', 'dd',
    'landuse_11', 'landuse_12', 'landuse_21', 'landuse_22', 'landuse_23', 'landuse_24',
    'landuse_31', 'landuse_32', 'landuse_33', 'landuse_41',  'landuse_43',
    'landuse_46', 'landuse_51', 'landuse_52', 'landuse_53', 'landuse_62', 'landuse_64'
]
sample_data = data.sample(n=min(5000, len(data)))  # 抽样防止计算过慢
correlation_matrix = sample_data[x_columns + ['swe']].corr()
print(correlation_matrix['swe'].sort_values(ascending=False))