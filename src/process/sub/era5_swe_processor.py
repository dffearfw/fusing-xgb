import logging
import sqlite3

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

from src.process.config import config
from src.process.sub.baseor import BaseProcessor  # 假设有基础处理器

logger = logging.getLogger("ERA5SWEProcessor")


class ERA5SWEProcessor(BaseProcessor):
    """ERA5 SWE 数据处理器"""

    def __init__(self, secure_processor=None, station_filter=None):
        super().__init__()
        self.logger = logging.getLogger("ERA5SWEProcessor")
        self.conf = config.era5_swe
        self.output_dir = Path(config.get_output_dir('era5_swe'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.secure_processor = secure_processor
        self.station_filter = station_filter

        # 初始化数据库连接
        self.db_conn = self.connect_database()

        # 初始化投影转换器（如果需要）
        self.proj_transformer = None

    def connect_database(self):
        """连接站点数据库"""
        # 复用GLSnowProcessor的数据库连接逻辑
        db_path = config.get_station_db_path()
        try:
            conn = sqlite3.connect(db_path)
            conn.enable_load_extension(True)
            conn.load_extension("mod_spatialite")  # 加载空间扩展
            return conn
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")
            raise

    def get_stations_for_date(self, target_date):
        """获取特定日期有效的站点（处理浮点日期格式）"""
        try:
            # 转换为数据库格式 (YYYYMMDD.0)
            db_date_format = float(target_date.strftime('%Y%m%d'))

            # 使用参数化查询
            query = "SELECT station_ID, Longitude, Latitude FROM stations WHERE time = ?"
            df = pd.read_sql_query(query, self.db_conn, params=(db_date_format,))

            if not df.empty:
                # 确保数值类型
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                df = df.dropna(subset=['Latitude', 'Longitude'])

                return df.to_dict('records')
            else:
                self.logger.warning(f"未找到时间 {db_date_format} 的站点记录")
                return []

        except Exception as e:
            self.logger.exception(f"获取站点失败: {str(e)}")
            return []

    def get_file_path(self, date):
        """根据日期生成文件路径"""
        try:
            # ERA5 SWE 文件按月存储
            path_template = self.conf['path_template']

            # 替换年月占位符
            replacements = {
                '{year}': date.strftime('%Y'),
                '{month}': date.strftime('%m')
            }

            file_path = path_template
            for placeholder, value in replacements.items():
                file_path = file_path.replace(placeholder, value)

            return Path(file_path)

        except Exception as e:
            self.logger.error(f"文件路径生成失败: {str(e)}")
            return None

    def extract_values(self, date, file_path):
        """从GeoTIFF文件中提取站点值"""
        try:
            # 获取该日期有效的站点
            stations = self.get_stations_for_date(date)
            if not stations:
                self.logger.warning(f"未找到 {date} 的有效站点")
                return pd.DataFrame()

            # 安全提取坐标
            lats, lons, station_ids = [], [], []
            for station in stations:
                if not isinstance(station, dict):
                    continue
                try:
                    lats.append(float(station.get('Latitude', station.get('latitude'))))
                    lons.append(float(station.get('Longitude', station.get('longitude'))))
                    station_ids.append(station.get('station_ID', station.get('station_id')))
                except (TypeError, ValueError):
                    pass

            if not station_ids:
                self.logger.error("没有有效的站点坐标")
                return pd.DataFrame()

            # 计算波段索引（每个波段对应一天）
            day_of_month = date.day
            band_index = day_of_month + self.conf.get('band_offset', 0)

            # 打开GeoTIFF文件
            with rasterio.open(file_path) as src:
                # 验证波段索引
                if band_index > src.count or band_index < 1:
                    self.logger.error(f"波段索引 {band_index} 超出范围 (1-{src.count})")
                    return pd.DataFrame()

                # 读取波段数据
                band_data = src.read(band_index)

                # 获取地理变换信息
                transform = src.transform

                # 提取站点值
                results = []
                for i, (lon, lat) in enumerate(zip(lons, lats)):
                    try:
                        # 将经纬度转换为行列号
                        row, col = rowcol(transform, lon, lat)

                        # 检查行列号是否在有效范围内
                        if 0 <= row < src.height and 0 <= col < src.width:
                            value = float(band_data[row, col])

                            # 检查填充值和有效范围
                            fill_value = self.conf.get('fill_value', -32768.0)
                            valid_min = self.conf.get('valid_min', 0.0)

                            if value != fill_value and value >= valid_min:
                                # 应用缩放因子
                                scale_factor = self.conf.get('scale_factor', 1.0)
                                value *= scale_factor

                                results.append({
                                    'station_id': station_ids[i],
                                    'date': date.strftime('%Y-%m-%d'),
                                    'value': value,
                                    'file_used': file_path.name
                                })
                        else:
                            self.logger.debug(f"站点 {station_ids[i]} 坐标超出图像范围")

                    except Exception as e:
                        self.logger.warning(f"提取站点 {station_ids[i]} 值失败: {str(e)}")

                return pd.DataFrame(results)

        except Exception as e:
            self.logger.exception(f"提取错误: {str(e)}")
            return pd.DataFrame()

    def process_range(self, start_date, end_date):
        """处理日期范围"""
        self.logger.info(f"开始处理 {start_date} 至 {end_date} 的ERA5 SWE数据")

        # 转换为日期范围
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        date_range = pd.date_range(start_date, end_date, freq='D')

        all_results = []

        for current_date in date_range:
            try:
                # 获取文件路径
                file_path = self.get_file_path(current_date)
                if not file_path or not file_path.exists():
                    self.logger.warning(f"文件不存在: {file_path}")
                    continue

                # 提取数据
                daily_df = self.extract_values(current_date, file_path)

                # 记录数据量
                self.logger.info(f"日期 {current_date}: 提取 {len(daily_df)} 条记录")

                if not daily_df.empty:
                    # 数据后处理
                    processed_df = self.postprocess_data(daily_df, current_date)
                    all_results.append(processed_df)

            except Exception as e:
                self.logger.error(f"{current_date} 处理失败: {str(e)}")
                continue

        # 合并结果
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            output_path = self.save_results(final_df, start_date, end_date)

            # 生成数据摘要
            self.generate_data_summary(final_df, Path(output_path))

            self.logger.info(f"处理完成! 结果保存至: {output_path}")
            return output_path
        else:
            self.logger.warning("未生成有效结果")
            return None

    def postprocess_data(self, df, date):
        """数据后处理"""
        if df.empty:
            return df

        # 创建副本
        result_df = df.copy()

        # 应用任何必要的后处理
        # 例如: 单位转换、质量控制等

        # 添加处理时间
        processing_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_df.loc[:, 'processing_time'] = processing_time

        return result_df

    def save_results(self, df, start_date, end_date):
        """保存结果"""
        try:
            # 确定输出格式
            output_format = self.conf.get('output_format', 'parquet').lower()
            base_filename = f"era5_swe_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

            # 根据格式选择保存方法
            if output_format == 'parquet':
                try:
                    output_path = self.output_dir / f"{base_filename}.parquet"
                    df.to_parquet(output_path, index=False)
                    self.logger.info(f"结果保存为 Parquet: {output_path}")
                    return str(output_path)
                except ImportError:
                    self.logger.warning("Parquet 支持不可用，回退到 CSV 格式")
                    output_format = 'csv'

            if output_format == 'csv':
                output_path = self.output_dir / f"{base_filename}.csv"
                df.to_csv(output_path, index=False)
                self.logger.info(f"结果保存为 CSV: {output_path}")
                return str(output_path)

            else:
                self.logger.warning(f"未知的输出格式: {output_format}，使用 CSV")
                output_path = self.output_dir / f"{base_filename}.csv"
                df.to_csv(output_path, index=False)
                self.logger.info(f"结果保存为 CSV: {output_path}")
                return str(output_path)

        except Exception as e:
            self.logger.exception(f"保存结果失败: {str(e)}")
            return None

    def generate_data_summary(self, df, output_path):
        """生成数据摘要"""
        try:
            summary_path = output_path.with_suffix('.summary.txt')

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=== ERA5 SWE 数据摘要 ===\n")
                f.write(f"记录数: {len(df)}\n")
                f.write(f"时间段: {df['date'].min()} 至 {df['date'].max()}\n")
                f.write(f"站点数: {df['station_id'].nunique()}\n")
                f.write(f"值范围: {df['value'].min():.6f} 至 {df['value'].max():.6f}\n")
                f.write(f"平均值: {df['value'].mean():.6f}\n")
                f.write(f"中位数: {df['value'].median():.6f}\n")

                f.write("\n=== 前10条记录 ===\n")
                f.write(df.head(10).to_string())

                f.write("\n\n=== 站点统计 ===\n")
                station_stats = df.groupby('station_id')['value'].agg(['count', 'min', 'max', 'mean'])
                f.write(station_stats.to_string())

            self.logger.info(f"数据摘要已生成: {summary_path}")

        except Exception as e:
            self.logger.warning(f"生成数据摘要失败: {str(e)}")