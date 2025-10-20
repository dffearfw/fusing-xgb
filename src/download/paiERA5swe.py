import ee
import time
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import urllib3

# 代理设置（根据实际情况调整）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GEE初始化
try:
    ee.Initialize(project='agile-bonbon-466206-f2')
    logging.info("GEE初始化成功")
except Exception as e:
    logging.error(f"GEE初始化失败: {e}")
    exit(1)


def get_china_boundary():
    """获取中国边界"""
    region = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
        .filter(ee.Filter.eq('country_na', 'China'))
    return region.geometry().simplify(maxError=5000).bounds()


china_region = get_china_boundary()


def download_era5_swe(start_date, end_date):
    """下载ERA5-Land雪水当量数据"""
    try:
        # 生成任务描述
        date_str = start_date.replace('-', '')[:6]
        task_desc = f"ERA5_SWE_{date_str}"

        # 加载ERA5-Land数据
        dataset = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterDate(start_date, end_date) \
            .select('snow_depth_water_equivalent')  # 雪水当量波段

        # 检查数据是否存在
        size = dataset.size().getInfo()
        if size == 0:
            logging.warning(f"{start_date} 到 {end_date} 无可用数据，跳过")
            return None

        # 转换为多波段影像
        daily_img = dataset.toBands().clip(china_region)

        # 设置导出参数
        export_params = {
            'image': daily_img,
            'description': task_desc,
            'folder': 'ERA5_SWE_China',
            'scale': 11132,  # ERA5-Land分辨率约0.1度≈11km
            'region': china_region,
            'maxPixels': 1e10,
            'fileFormat': 'GeoTIFF',
            'formatOptions': {
                'cloudOptimized': True
            }
        }

        # 启动导出任务
        task = ee.batch.Export.image.toDrive(**export_params)
        task.start()
        logging.info(f"启动任务: {task_desc} (ID: {task.id})")
        return task.id

    except Exception as e:
        logging.error(f"任务启动失败 {start_date}-{end_date}: {e}")
        return None


def batch_download_swe(start_year=1981, end_year=2020):
    """批量下载雪水当量数据"""
    task_records = []

    # 生成日期范围（按月）
    date_ranges = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            if month == 12:
                end = f"{year + 1}-01-01"
            else:
                end = f"{year}-{month + 1:02d}-01"
            date_ranges.append((start, end))

    logging.info(f"准备处理 {len(date_ranges)} 个月的数据")

    # 批量处理参数
    batch_size = 3  # 减小批量大小避免超额
    delay_between_batches = 60  # 批次间延迟
    delay_between_tasks = 15  # 任务间延迟

    for i in range(0, len(date_ranges), batch_size):
        batch = date_ranges[i:i + batch_size]
        logging.info(f"处理批次 {i // batch_size + 1}/{(len(date_ranges) - 1) // batch_size + 1}")

        for start, end in batch:
            task_id = download_era5_swe(start, end)
            if task_id:
                task_records.append({
                    'start_date': start,
                    'end_date': end,
                    'task_id': task_id,
                    'status': 'SUBMITTED',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                time.sleep(delay_between_tasks)  # 任务间延迟

        # 保存进度
        save_progress(task_records)

        # 批次间延迟（最后一个批次不需要等待）
        if i + batch_size < len(date_ranges):
            logging.info(f"等待 {delay_between_batches} 秒后继续...")
            time.sleep(delay_between_batches)

    return task_records


def save_progress(records, filename='era5_swe_download_progress.csv'):
    """保存下载进度"""
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    logging.info(f"进度已保存至 {filename}")


def monitor_tasks(progress_file='era5_swe_download_progress.csv'):
    """监控任务状态"""
    if not os.path.exists(progress_file):
        logging.error("进度文件不存在")
        return None

    df = pd.read_csv(progress_file)

    try:
        tasks = ee.batch.Task.list()
        task_dict = {task.id: task.status() for task in tasks}

        updated = False
        for i, row in df.iterrows():
            task_id = row['task_id']
            if task_id in task_dict:
                status = task_dict[task_id]['state']
                if row['status'] != status:
                    df.at[i, 'status'] = status
                    updated = True
                    logging.info(f"任务 {task_id} 状态更新: {status}")

        if updated:
            df.to_csv(progress_file, index=False)
            logging.info("进度文件已更新")

        # 统计状态
        status_counts = df['status'].value_counts()
        logging.info("任务状态统计:\n" + str(status_counts))

        return df

    except Exception as e:
        logging.error(f"监控任务时出错: {e}")
        return df


def check_data_availability():
    """检查数据可用性"""
    # ERA5-Land数据从1981年开始
    test_start = "1981-01-01"
    test_end = "1981-01-02"

    dataset = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterDate(test_start, test_end) \
        .select('snow_depth_water_equivalent')

    size = dataset.size().getInfo()
    bands = dataset.first().bandNames().getInfo()

    logging.info(f"数据可用性检查: 影像数量={size}, 波段={bands}")
    return size > 0


if __name__ == "__main__":
    # SSL设置
    urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL'

    # 检查数据可用性
    if check_data_availability():
        logging.info("ERA5-Land雪水当量数据可用，开始下载...")

        # 批量下载（可根据需要调整年份范围）
        task_records = batch_download_swe(2000, 2020)  # 示例：下载2000-2020年数据

        logging.info("所有任务已提交，使用 monitor_tasks() 函数监控进度")
    else:
        logging.error("ERA5-Land雪水当量数据不可用，请检查数据集名称")