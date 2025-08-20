import os
import ee
import requests
import pandas as pd
from datetime import datetime
import time

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

# 初始化GEE
ee.Initialize(project='agile-bonbon-466206-f2')

# 定义中国边界
china = ee.FeatureCollection("FAO/GAUL/2015/level0") \
    .filter(ee.Filter.eq('ADM0_NAME', 'China'))

# 加载CSWE数据集 (确认ID正确性)
dataset = ee.ImageCollection('CAS/CSWE/SWE') \
    .filterDate('1980-01-01', '2020-12-31') \
    .filterBounds(china)


# 定义下载函数
def download_swe(year):
    # 按年过滤数据
    annual_collection = dataset.filterDate(f'{year}-01-01', f'{year}-12-31')

    # 检查数据是否存在
    if annual_collection.size().getInfo() == 0:
        print(f"No data found for {year}")
        return None

    # 导出任务配置
    export_task = ee.batch.Export.image.toDrive(
        image=annual_collection.toBands(),  # 合并为多波段图像
        description=f'CSWE_China_{year}',
        folder='GEE_CSWE',  # Google Drive文件夹名
        fileNamePrefix=f'china_swe_{year}',
        region=china.geometry().bounds(),  # 中国边界
        scale=25000,  # 25km分辨率 (CSWE原始分辨率)
        crs='EPSG:4326',  # WGS84坐标系
        maxPixels=1e10
    )
    export_task.start()
    return export_task.id


# 批量下载1980-2020年数据
tasks = {}
for year in range(1980, 2021):
    task_id = download_swe(year)
    if task_id:
        tasks[year] = task_id
    time.sleep(1)  # 避免请求超限

# 保存任务ID (用于后续跟踪)
pd.Series(tasks).to_csv("GEE_task_ids.csv")
print("导出任务已提交至Google Drive，请等待处理完成")