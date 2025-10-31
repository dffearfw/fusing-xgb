import ee
import os
import zipfile
import shutil
import time
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

ee.Authenticate()
# 初始化 GEE
ee.Initialize(project='agile-bonbon-466206-f2')

# 设置输出目录
OUTPUT_DIR = "E:\dem"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Google Drive API 认证
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def get_gdrive_service():
    """获取Google Drive服务"""
    creds = None
    token_file = os.path.join(OUTPUT_DIR, 'token.json')
    creds_file = os.path.join(OUTPUT_DIR, 'credentials.json')

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_file):
                raise FileNotFoundError(
                    "请将Google Drive API凭证保存为 'credentials.json' 并放置在输出目录中")

            flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def monitor_tasks(tasks, timeout=86400):
    """监控任务状态"""
    start_time = time.time()
    pending_tasks = tasks.copy()

    while pending_tasks and (time.time() - start_time) < timeout:
        for i, task in enumerate(pending_tasks[:]):
            status = task.status()
            state = status['state']

            if state == 'COMPLETED':
                print(f'任务 {task.id} 完成')
                pending_tasks.remove(task)
            elif state == 'FAILED':
                print(f'任务 {task.id} 失败: {status.get("error_message", "未知错误")}')
                pending_tasks.remove(task)
            elif state == 'CANCELED':
                print(f'任务 {task.id} 被取消')
                pending_tasks.remove(task)

        if pending_tasks:
            print(f'剩余任务: {len(pending_tasks)}，等待30秒...')
            time.sleep(30)

    return pending_tasks


def download_from_drive(service, task, output_dir):
    """从Google Drive下载文件"""
    # 获取导出文件信息
    export_info = task.status()['destination_uris'][0]
    file_id = export_info.split('=')[1]

    # 查询文件元数据
    file_metadata = service.files().get(fileId=file_id, fields='name').execute()
    file_name = file_metadata['name']
    output_zip = os.path.join(output_dir, file_name)

    # 下载文件
    request = service.files().get_media(fileId=file_id)
    with open(output_zip, 'wb') as f:
        f.write(request.execute())

    print(f'已下载: {file_name}')
    return output_zip


def download_srtm_global():
    """下载全球90m SRTM DEM数据"""
    # 1. 加载全球SRTM数据
    dem = ee.Image("CGIAR/SRTM90_V4")

    # 2. 创建全球网格（5度×5度的瓦片）
    grid = []

    # 全球范围：-180到180经度，-60到60纬度（SRTM覆盖范围）
    for lon in range(-180, 180, 5):
        for lat in range(-60, 60, 5):
            lon_min = lon
            lon_max = lon + 5
            lat_min = lat
            lat_max = lat + 5

            # 创建矩形区域
            grid.append(ee.Geometry.Rectangle(
                [lon_min, lat_min, lon_max, lat_max]))

    tasks = []
    print(f"创建 {len(grid)} 个下载区块")

    # 创建Google Drive服务
    drive_service = get_gdrive_service()

    for i, roi in enumerate(grid):
        try:
            # 裁剪到当前瓦片
            tile = dem.clip(roi)

            # 创建下载任务
            task = ee.batch.Export.image.toDrive(
                image=tile,
                description=f'SRTM_Global_{i}',
                folder='GEE_DEM_Global',
                fileNamePrefix=f'global_dem_{i}',
                scale=90,
                crs='EPSG:4326',
                region=roi.getInfo()['coordinates'],
                maxPixels=1e10,
                fileFormat='GeoTIFF'
            )
            task.start()
            tasks.append(task)
            print(
                f'启动下载任务 {i + 1}/{len(grid)}: 经度{roi.getInfo()["coordinates"][0][0][0]:.1f}-{roi.getInfo()["coordinates"][0][2][0]:.1f}, 纬度{roi.getInfo()["coordinates"][0][0][1]:.1f}-{roi.getInfo()["coordinates"][0][2][1]:.1f}')
            time.sleep(15)  # 增加等待时间避免请求限制

        except Exception as e:
            print(f'区块 {i} 创建失败: {str(e)}')
            # 继续处理下一个区块

    # 3. 监控任务状态
    print("等待任务完成... (可能需要数小时到数天)")
    pending = monitor_tasks(tasks, timeout=172800)  # 48小时超时

    if pending:
        print(f"警告: {len(pending)} 个任务未完成")

    # 4. 下载到本地
    print("开始下载到本地...")
    successful_downloads = 0

    for i, task in enumerate(tasks):
        try:
            # 检查任务状态
            status = task.status()
            if status['state'] != 'COMPLETED':
                print(f'跳过未完成任务 {task.id}, 状态: {status["state"]}')
                continue

            # 下载文件
            output_zip = download_from_drive(drive_service, task, OUTPUT_DIR)

            # 解压
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(OUTPUT_DIR)
            os.remove(output_zip)
            successful_downloads += 1
            print(f'处理完成区块 {i + 1}/{len(tasks)}')

            # 检查空间
            free_gb = shutil.disk_usage(OUTPUT_DIR).free / (1024 ** 3)
            if free_gb < 5:  # 增加最小空间要求
                print(f"空间不足! 剩余空间: {free_gb:.1f}GB")
                break

        except Exception as e:
            print(f"区块 {i} 下载失败: {str(e)}")

    print(f"全球DEM数据已保存到: {os.path.abspath(OUTPUT_DIR)}")
    print(f"成功下载 {successful_downloads}/{len(tasks)} 个区块")


def main():
    # 检查空间 - 全球数据需要更多空间
    free_space = shutil.disk_usage(OUTPUT_DIR).free / (1024 ** 3)
    print(f"可用空间: {free_space:.1f}GB")

    # 全球SRTM数据大约需要10-15GB
    if free_space < 20:
        raise RuntimeError(f"存储空间不足! 全球DEM需要至少20GB, 当前: {free_space:.1f}GB")

    # 确认用户意图
    response = input("这将下载全球SRTM DEM数据（约10-15GB），是否继续？(y/n): ")
    if response.lower() != 'y':
        print("操作已取消")
        return

    download_srtm_global()

    # 计算总大小
    total_size = 0
    tif_count = 0
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.tif'):
            fp = os.path.join(OUTPUT_DIR, f)
            total_size += os.path.getsize(fp)
            tif_count += 1

    print(f"总下载大小: {total_size / (1024 ** 3):.2f} GB")
    print(f"文件数量: {tif_count} 个TIFF文件")


if __name__ == "__main__":
    main()