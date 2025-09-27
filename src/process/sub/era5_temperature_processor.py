"""
ERA5温度数据处理器 - 精确匹配版
只提取数据库中实际存在的站点-日期组合的温度值
"""
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import rasterio
from rasterio.transform import rowcol
import re
import gc

logger = logging.getLogger("ERA5TemperatureProcessor")


class ERA5TemperatureProcessor:
    """ERA5温度数据处理器（精确匹配版）"""

    def __init__(self, secure_processor=None, station_filter=None):
        from src.process.config import config

        self.logger = logging.getLogger("ERA5TemperatureProcessor")
        self.conf = config.era5_temperature
        self.secure_processor = secure_processor
        self.station_filter = station_filter

        # 数据库连接
        self.db_conn = None
        try:
            self.db_conn = self.connect_database()
            self.logger.info("数据库连接成功")
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")

        # 输出目录
        self.output_dir = Path(config.get_output_dir('era5_temperature'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 基础数据目录
        self.base_data_dir = Path(config.paths.get('input_root', 'data')) / self.conf['data_dir']
        self.logger.info(f"数据目录: {self.base_data_dir}")

    def connect_database(self):
        from src.process.config import config
        db_path = config.get_station_db_path()
        try:
            conn = sqlite3.connect(db_path)
            return conn
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")
            raise

    def get_station_date_records(self, start_date, end_date):
        """
        获取指定日期范围内有记录的站点信息
        只返回数据库中实际存在的站点-日期组合
        """
        try:
            # 转换日期格式为YYYYMMDD
            start_ymd = int(start_date.replace('-', ''))
            end_ymd = int(end_date.replace('-', ''))

            query = """
            SELECT DISTINCT station_ID, Longitude, Latitude, time
            FROM stations 
            WHERE time BETWEEN ? AND ?
            """
            params = [start_ymd, end_ymd]

            if self.station_filter:
                if self.station_filter['type'] == 'bbox':
                    minx, miny, maxx, maxy = self.station_filter['value']
                    query += " AND Longitude BETWEEN ? AND ? AND Latitude BETWEEN ? AND ?"
                    params.extend([minx, maxx, miny, maxy])
                elif self.station_filter['type'] == 'ids':
                    placeholders = ','.join(['?'] * len(self.station_filter['value']))
                    query += f" AND station_ID IN ({placeholders})"
                    params.extend(self.station_filter['value'])

            df = pd.read_sql_query(query, self.db_conn, params=params)

            if not df.empty:
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce').astype(np.float32)
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce').astype(np.float32)
                df = df.dropna(subset=['Latitude', 'Longitude'])

                # 转换time为日期字符串
                df['date_str'] = df['time'].astype(str).apply(
                    lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if len(x) == 8 else x
                )

                self.logger.info(f"找到 {len(df)} 条站点-日期记录")
                return df.to_dict('records')
            else:
                self.logger.warning("未找到站点记录")
                return []

        except Exception as e:
            self.logger.exception(f"获取站点记录失败: {str(e)}")
            return []

    def _get_monthly_data(self, year, month):
        """一次性读取整个月份的温度数据到内存"""
        try:
            filename = f"era5_temperature_{year:04d}{month:02d}.tif"
            file_path = self.base_data_dir / filename

            if not file_path.exists():
                self.logger.warning(f"数据文件不存在: {file_path}")
                return None

            with rasterio.open(file_path) as src:
                # 一次性读取整个数据数组
                data = src.read(1)
                transform = src.transform
                fill_value = self.conf.get('fill_value', -32768.0)

                return {
                    'data': data,
                    'transform': transform,
                    'fill_value': fill_value,
                    'crs': src.crs
                }

        except Exception as e:
            self.logger.error(f"读取月份数据失败 {year}-{month}: {str(e)}")
            return None

    def extract_temperature_values(self, monthly_data, stations):
        """从内存中的数据数组中提取温度值"""
        results = []
        success_count = 0

        data = monthly_data['data']
        transform = monthly_data['transform']
        fill_value = monthly_data['fill_value']

        for station in stations:
            try:
                lon = float(station['Longitude'])
                lat = float(station['Latitude'])
                station_id = station['station_ID']

                # 将经纬度转换为行列号
                row, col = rowcol(transform, lon, lat)
                row, col = int(row), int(col)

                # 检查行列号是否有效
                if (0 <= row < data.shape[0] and
                        0 <= col < data.shape[1]):

                    temperature = data[row, col]

                    # 检查是否为有效值
                    if temperature != fill_value and not np.isnan(temperature):
                        # 应用比例因子和偏移量
                        scale_factor = self.conf.get('scale_factor', 1.0)
                        add_offset = self.conf.get('add_offset', 0.0)
                        temperature = temperature * scale_factor + add_offset

                        # 转换为摄氏度（如果单位是开尔文）
                        if self.conf.get('unit') == 'K':
                            temperature = temperature - 273.15

                        date_obj = datetime.strptime(station['date_str'], '%Y-%m-%d')

                        results.append({
                            'station_id': station_id,
                            'temperature': np.float32(temperature),
                            'longitude': np.float32(lon),
                            'latitude': np.float32(lat),
                            'date': station['date_str'],
                            'year': np.int16(date_obj.year),
                            'month': np.int8(date_obj.month),
                            'day': np.int8(date_obj.day),
                            'original_time': station['time']
                        })
                        success_count += 1

            except Exception as e:
                continue

        return results, success_count

    def process_range(self, start_date, end_date):
        """处理指定日期范围内的温度数据"""
        try:
            # 获取数据库中实际存在的站点-日期记录
            station_records = self.get_station_date_records(start_date, end_date)
            if not station_records:
                self.logger.warning("没有找到在指定日期范围内的站点记录")
                return None

            self.logger.info(f"开始处理 {len(station_records)} 条站点-日期记录")

            # 按月份分组
            monthly_groups = {}
            for record in station_records:
                try:
                    time_str = str(record['time'])
                    if len(time_str) == 8:
                        year = int(time_str[:4])
                        month = int(time_str[4:6])
                        month_key = f"{year:04d}-{month:02d}"

                        if month_key not in monthly_groups:
                            monthly_groups[month_key] = []
                        monthly_groups[month_key].append(record)
                except:
                    continue

            self.logger.info(f"按月份分组: {len(monthly_groups)} 个月份")

            all_results = []
            total_processed = 0

            # 处理每个月份
            for month_key, records in monthly_groups.items():
                try:
                    year = int(month_key[:4])
                    month = int(month_key[5:7])

                    self.logger.info(f"处理 {month_key}: {len(records)} 条记录")

                    # 一次性读取整个月份的数据
                    monthly_data = self._get_monthly_data(year, month)
                    if monthly_data is None:
                        continue

                    # 提取温度值
                    results, success_count = self.extract_temperature_values(monthly_data, records)

                    if results:
                        all_results.extend(results)
                        total_processed += success_count
                        self.logger.info(f"{month_key}: 成功提取 {success_count}/{len(records)} 条记录")
                    else:
                        self.logger.warning(f"{month_key}: 未提取到有效数据")

                    # 释放内存
                    del monthly_data
                    gc.collect()

                except Exception as e:
                    self.logger.error(f"处理月份 {month_key} 失败: {str(e)}")
                    continue

            # 保存结果
            if all_results:
                df = pd.DataFrame(all_results)
                output_path = self.save_results(df)

                self.logger.info(f"处理完成! 总共处理 {total_processed} 条记录")
                self.logger.info(f"输出文件: {output_path}")

                return output_path
            else:
                self.logger.warning("未生成有效结果")
                return None

        except Exception as e:
            self.logger.exception(f"处理日期范围失败: {str(e)}")
            return None

    def save_results(self, df):
        """保存结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"era5_temperature_{timestamp}"

            output_format = self.conf.get('output_format', 'parquet').lower()

            if output_format == 'parquet':
                parquet_path = self.output_dir / f"{base_filename}.parquet"
                df.to_parquet(parquet_path, index=False)
                self.logger.info(f"结果保存为 Parquet: {parquet_path}")
                return str(parquet_path)
            else:
                csv_path = self.output_dir / f"{base_filename}.csv"
                df.to_csv(csv_path, index=False, float_format='%.3f')
                self.logger.info(f"结果保存为 CSV: {csv_path}")
                return str(csv_path)

        except Exception as e:
            self.logger.exception(f"保存结果失败: {str(e)}")
            return None

    def close(self):
        try:
            if self.db_conn:
                self.db_conn.close()
        except Exception as e:
            self.logger.warning(f"关闭数据库连接时出错: {str(e)}")

    def __del__(self):
        self.close()