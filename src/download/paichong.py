import ee
import time
import requests
import os
from datetime import datetime,timedelta
import logging
import pandas as pd
# import requests.packages.urllib3.util.ssl_
import urllib3
from urllib.request import urlopen

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'


logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')


try:
    ee.Initialize(project='agile-bonbon-466206-f2')
    logging.info("gee initalize successfully")
except Exception as e:
    logging.error(f"fail:{e}")
    logging.info("终端运行")
    exit(1)


def get_boundary():
    region = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
        .filter(ee.Filter.eq('country_na', 'China'))
    return region.geometry().simplify(maxError=5000).bounds()


china_region = get_boundary()

def download_era(stat_dat,end_dat):
    data_str=stat_dat.replace('-','')[0:6]
    task_desc=f"era5_temperature_{data_str}"

    dataset=ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")\
        .filterDate(stat_dat,end_dat)\
        .select('temperature_2m')

    daily_img=dataset.toBands().clip(china_region)


    export_params={
        'image':daily_img,
        'description':task_desc,
        'folder':'GLDAS daily swe',
        'scale':11132,
        'region':china_region,
        'maxPixels':1e10,
        'fileFormat':'GeoTIFF',
        'formatOptions':{
            'cloudOptimized':True,
            'noData':-32767
        }
    }
    try:
        task=ee.batch.Export.image.toDrive(**export_params)
        task.start()
        logging.info(f"begin:{task_desc}(id:{task.id})")
        return task.id
    except Exception as e:
        logging.error(f"fail:{e}")
        return None

def batch_download(startyear=1981,endyear=2020):
    task_record=[]

    date_ranges=[]
    for year in range(startyear,endyear+1):
        for month in range(1,13):
            start=f"{year}-{month:02d}-01"
            if month==12:
                end=f"{year+1}-01-01"
            else:
                end=f"{year}-{month+1:02d}-01"
            date_ranges.append((start,end))

    logging.info(f"处理{len(date_ranges)}个月数据")

    batch_size=5
    delay=60

    for i in range(0,len(date_ranges),batch_size):
        batch=date_ranges[i:i+batch_size]
        batch_tasks=[]

        for start,end in batch:
            task_id=download_era(start,end)
            if task_id:
                task_record.append({
                    'start-date':start,
                    'end_date':end,
                    'task-id':task_id,
                    'status':'SUBMITTED'
                })
                time.sleep(10)

        save_progress(task_record)
        logging.info(f"已提交批次{i//batch_size+1}/{(len(date_ranges)//batch_size)+1}.等待{delay}秒")
        time.sleep(delay)

    return task_record

def save_progress(records):
    df=pd.DataFrame(records)
    df.to_csv('era5 daily temperature downloadprogress.csv',index=False)

def monitor_tasks(progress_file='era5 daily temperature downloadprogress.csv'):
    if not os.path.exists(progress_file):
        logging.error("进度文件不存在")
        return
    df=pd.read_csv(progress_file)
    tasks=ee.batch.Task.list()
    task_dict={task.id:task.status() for task in tasks}

    updated=False
    for i,row in df.iterows():
        task_id=row['task_id']
        if task_id in task_dict:
            status=task_dict[task_id]['state']
            if row['status']!=status:
                df.at[i,'status']=status
                updated=True
                logging.info(f"任务{task_id}状态更新:{status}")

    if updated:
        df.to_csv(progress_file,index=False)
        logging.info("进度文件已更新")

    status_counts=df['status'].value_counts()
    logging.info("任务状态统计\n"+str(status_counts))

    return df

if __name__=="__main__":

    urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL'

    task_records=batch_download(1981,2020)

    # urlopen('https://www.howsmyssl.com/a/check').read()































































