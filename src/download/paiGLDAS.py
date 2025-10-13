import ee
import time
import os
import sys
import logging
import pandas as pd
from datetime import datetime,timedelta
import retrying

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GLDAS_SWE_Downloader')


# 1. 初始化GEE API
def initialize_gee():
    """初始化GEE API，带重试机制"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ee.Initialize(project='agile-bonbon-466206-f2')
            logger.info("GEE初始化成功")
            return True
        except ee.EEException as e:
            logger.error(f"GEE初始化失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if "Please authorize access" in str(e):
                logger.info("请先在终端运行: earthengine authenticate")
            time.sleep(5)
    logger.error("GEE初始化失败，超出最大重试次数")
    return False


# 2. 定义中国区域边界
def get_china_boundary():
    """获取中国精确边界"""
    try:
        # 使用FAO全球行政区划数据
        countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
        china = countries.filter(ee.Filter.eq('ADM0_NAME', 'China'))
        return china.geometry().simplify(maxError=5000).bounds()
    except Exception as e:
        logger.error(f"获取中国边界失败: {e}")
        # 备选方案：使用矩形近似
        return ee.Geometry.Rectangle([73.66, 18.16, 135.05, 53.56])


# 3. 保存进度函数
def save_progress(records, filename='GLDAS_daily_download_progress.csv'):
    """保存任务进度到CSV"""
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    logger.info(f"进度已保存至 {filename}")


# 4. 下载函数 - 针对GLDAS数据优化
@retrying.retry(stop_max_attempt_number=3, wait_fixed=10000)
def download_gldas_daily_swe(year):
    """下载单年度的GLDAS日SWE数据"""
    try:
        # GLDAS数据集信息 - CLSM日分辨率数据集
        dataset_id = "NASA/GLDAS/V022/CLSM/G025/DA1D"
        swe_band = 'SWE_tavg'  # 雪水当量波段

        # 时间范围
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"  # 改为当年12月31日

        # 加载GLDAS数据 - 已经是日分辨率数据
        daily_col = ee.ImageCollection(dataset_id) \
            .filterDate(start_date, end_date) \
            .select(swe_band)

        # 检查是否有数据
        size = daily_col.size().getInfo()
        if size == 0:
            logger.warning(f"{year}年无可用数据，跳过")
            return None

        # 对于日分辨率数据，直接使用即可，无需时间聚合
        # 按日期排序确保数据顺序正确
        daily_col_sorted = daily_col.sort('system:time_start')

        # 转换为多波段影像（每日一个波段）
        daily_img = daily_col_sorted.toBands().clip(china_region)

        # 设置导出参数
        export_params = {
            'image': daily_img,
            'description': f"GLDAS_SWE_China_{year}",
            'folder': 'GLDAS_SWE_China_Daily',
            'scale': 25000,  # GLDAS原生分辨率0.25度≈25km
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
        logger.info(f"启动{year}年任务: {task.id}")
        return task.id

    except Exception as e:
        logger.error(f"任务启动失败: {year}年 - {e}")
        raise


# 5. 批量下载函数
def batch_download_gldas_swe(start_year=2013, end_year=2018):
    """分年下载GLDAS SWE数据"""
    # 检查进度文件
    progress_file = 'logs/GLDAS_daily_download_progress.csv'
    if os.path.exists(progress_file):
        df_existing = pd.read_csv(progress_file)
        existing_years = set(df_existing['year'])
        logger.info(f"发现现有进度文件，包含 {len(df_existing)} 条记录")
    else:
        existing_years = set()
        df_existing = pd.DataFrame(columns=['year', 'task_id', 'status', 'timestamp'])

    # 生成所有年份
    all_years = list(range(start_year, end_year + 1))
    pending_years = [y for y in all_years if y not in existing_years]

    logger.info(f"总年份: {len(all_years)}, 待处理: {len(pending_years)}, 已完成: {len(existing_years)}")

    if not pending_years:
        logger.info("所有年份已处理完成")
        return df_existing.to_dict('records')

    # 任务记录
    task_records = df_existing.to_dict('records')

    # 分批处理 - 每年一个任务
    batch_size = 1  # GLDAS数据量大，每年一个任务
    delay = 300  # 5分钟延迟

    for i, year in enumerate(pending_years):
        task_id = download_gldas_daily_swe(year)
        if task_id:
            record = {
                'year': year,
                'task_id': task_id,
                'status': 'SUBMITTED',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            task_records.append(record)
            save_progress(task_records, progress_file)

            # 计算剩余时间
            remaining = len(pending_years) - (i + 1)
            logger.info(f"已提交{year}年任务 ({i + 1}/{len(pending_years)}), 剩余 {remaining} 年, 等待 {delay} 秒")
            time.sleep(delay)

    return task_records


# 6. 监控任务函数
def monitor_tasks(progress_file='GLDAS_daily_download_progress.csv'):
    """监控任务状态并更新进度"""
    if not os.path.exists(progress_file):
        logger.error("进度文件不存在")
        return None

    try:
        df = pd.read_csv(progress_file)
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
                    logger.info(f"任务 {task_id} 状态更新: {status}")

        if updated:
            df.to_csv(progress_file, index=False)
            logger.info("进度文件已更新")

        # 统计状态
        status_counts = df['status'].value_counts()
        logger.info("任务状态统计:\n" + str(status_counts))

        # 发送通知
        send_status_notification(status_counts)

        return df
    except Exception as e:
        logger.error(f"监控任务失败: {e}")
        return None


def send_status_notification(status_counts):
    """发送状态通知"""
    completed = status_counts.get('COMPLETED', 0)
    total = status_counts.sum()
    progress = f"{completed}/{total} ({completed / total * 100:.1f}%)"
    logger.info(f"当前进度: {progress}")

    # 这里可以添加邮件通知功能
    # send_email(f"GLDAS下载进度: {progress}")


# 7. 自动监控函数
def auto_monitor(interval_hours=2, max_checks=100):
    """自动定期监控任务进度"""
    check_count = 0
    while check_count < max_checks:
        logger.info(f"开始第 {check_count + 1}/{max_checks} 次监控检查")
        monitor_tasks()

        # 等待指定时间
        wait_seconds = interval_hours * 3600
        logger.info(f"等待 {interval_hours} 小时后再次检查...")
        time.sleep(wait_seconds)
        check_count += 1


# 8. 主程序入口
if __name__ == "__main__":
    # 初始化GEE
    if not initialize_gee():
        sys.exit(1)

    # 获取中国区域
    global china_region
    china_region = get_china_boundary()
    logger.info("中国区域边界计算完成")

    # 用户选择模式
    print("\n==== GLDAS中国区域SWE数据下载器 ====")
    print("1. 启动下载任务 (2003-2018)")
    print("2. 监控任务进度")
    print("3. 自动监控模式 (每2小时检查)")
    print("4. 退出程序")

    mode = input("请选择模式: ").strip()

    if mode == "1":
        logger.info("开始下载任务...")
        task_records = batch_download_gldas_swe(2016, 2016)
        logger.info(f"所有任务已提交，共 {len(task_records)} 个任务")

    elif mode == "2":
        logger.info("开始监控任务进度...")
        monitor_tasks()

    elif mode == "3":
        logger.info("启动自动监控模式，每2小时检查一次...")
        auto_monitor()

    else:
        logger.info("程序退出")