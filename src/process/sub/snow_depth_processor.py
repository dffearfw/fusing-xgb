"""
积雪深度数据处理器 - 修正版
根据stations表中的time列提取数据
"""
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import re
import gc

logger = logging.getLogger("SnowDepthProcessor")


class SnowDepthProcessor:
    """积雪深度ASCII文本数据处理器（完整修复版）"""

    def __init__(self, secure_processor=None, station_filter=None):
        """
        初始化积雪深度处理器
        """
        from src.process.config import config

        self.logger = logging.getLogger("SnowDepthProcessor")
        self.conf = config.snow_depth
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
        self.output_dir = Path(config.get_output_dir('snow_depth'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 基础数据目录
        self.base_data_dir = Path(config.paths.get('input_root', 'data')) / self.conf['data_dir']
        self.logger.info(f"数据目录: {self.base_data_dir}")

    def connect_database(self):
        """连接站点数据库"""
        from src.process.config import config

        db_path = config.get_station_db_path()
        try:
            conn = sqlite3.connect(db_path)
            conn.enable_load_extension(True)

            try:
                conn.load_extension("mod_spatialite")
                self.logger.debug("成功加载空间扩展")
            except Exception as e:
                self.logger.warning(f"加载空间扩展失败: {str(e)}")

            return conn
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")
            raise

    def get_stations_with_dates(self, start_date, end_date):
        """获取指定日期范围内有记录的站点信息"""
        try:
            # 将日期转换为YYYYMMDD格式
            start_ymd = int(start_date.replace('-', ''))
            end_ymd = int(end_date.replace('-', ''))

            query = """
            SELECT station_ID, Longitude, Latitude, time
            FROM stations 
            WHERE time BETWEEN ? AND ?
            """
            params = [start_ymd, end_ymd]

            # 添加站点过滤器
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

                # 将time转换为日期字符串格式
                df['date_str'] = df['time'].astype(str).apply(
                    lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if len(x) == 8 else x
                )

                self.logger.info(f"找到 {len(df)} 条站点-日期记录 ({start_date} 到 {end_date})")
                return df.to_dict('records')
            else:
                self.logger.warning(f"在 {start_date} 到 {end_date} 范围内未找到站点记录")
                return []

        except Exception as e:
            self.logger.exception(f"获取站点记录失败: {str(e)}")
            return []

    def process_range(self, start_date, end_date):
        """处理日期范围数据"""
        self.logger.info(f"处理积雪深度数据范围: {start_date} 到 {end_date}")

        try:
            # 获取站点-日期记录
            station_date_records = self.get_stations_with_dates(start_date, end_date)

            if not station_date_records:
                self.logger.warning(f"在 {start_date} 到 {end_date} 范围内没有找到站点记录")
                return None

            self.logger.info(f"找到 {len(station_date_records)} 条站点-日期记录")

            # 处理记录
            result_df = self.process_station_date_records(station_date_records)

            # 检查结果
            if result_df is None or result_df.empty:
                self.logger.warning("未生成有效结果")
                return None

            # 保存结果
            output_path = self.save_results(result_df, start_date, end_date)

            self.logger.info(f"处理完成! 生成 {len(result_df)} 条记录")
            return output_path

        except Exception as e:
            self.logger.exception(f"处理失败: {str(e)}")
            return None

    def process_station_date_records(self, station_date_records):
        """处理站点-日期记录"""
        try:
            if not station_date_records:
                self.logger.warning("没有站点-日期记录需要处理")
                return pd.DataFrame()

            results = []
            processed_count = 0
            success_count = 0

            # 按日期分组
            date_groups = {}
            for record in station_date_records:
                date_str = record['date_str']
                if date_str not in date_groups:
                    date_groups[date_str] = []
                date_groups[date_str].append(record)

            self.logger.info(f"按日期分组: {len(date_groups)} 个唯一日期")

            # 处理每个日期的数据
            for date_str, records in date_groups.items():
                try:
                    # 构建文件名
                    filename = self._get_filename_from_date_str(date_str)
                    file_path = self.base_data_dir / filename

                    if not file_path.exists():
                        self.logger.debug(f"数据文件不存在: {file_path}")
                        continue

                    # 解析文件头信息
                    header = self._parse_ascii_grid_header(file_path)

                    # 处理该日期的所有站点
                    for record in records:
                        try:
                            lon = float(record['Longitude'])
                            lat = float(record['Latitude'])
                            station_id = record['station_ID']

                            # 坐标转换为行列号
                            row, col = self._coords_to_rowcol(lon, lat, header)

                            if 0 <= row < header['nrows'] and 0 <= col < header['ncols']:
                                # 读取像素值
                                pixel_value = self._get_pixel_value(file_path, row, col)

                                if pixel_value != header['nodata_value'] and not np.isnan(pixel_value):
                                    results.append({
                                        'station_id': station_id,
                                        'date': date_str,
                                        'snow_depth': pixel_value,
                                        'longitude': lon,
                                        'latitude': lat,
                                        'file_used': filename
                                    })
                                    success_count += 1
                            else:
                                self.logger.debug(f"站点 {station_id} 坐标超出范围")

                            processed_count += 1

                        except Exception as e:
                            self.logger.debug(f"处理站点 {record.get('station_ID')} 失败: {str(e)}")
                            continue

                    # 进度记录
                    if len(results) > 0 and len(results) % 1000 == 0:
                        self.logger.info(f"处理进度: {processed_count} 记录, 成功: {success_count}")

                except Exception as e:
                    self.logger.error(f"处理日期 {date_str} 失败: {str(e)}")
                    continue

            # 创建DataFrame
            if results:
                result_df = pd.DataFrame(results)
                self.logger.info(f"处理完成: 总计 {len(result_df)} 条有效记录")
                return result_df
            else:
                self.logger.warning("未提取到任何有效数据")
                return pd.DataFrame()

        except Exception as e:
            self.logger.exception(f"处理站点日期记录失败: {str(e)}")
            return pd.DataFrame()

    def save_results(self, df, start_date, end_date):
        """保存结果数据"""
        try:
            # 生成文件名
            start_str = start_date.replace('-', '')
            end_str = end_date.replace('-', '')
            base_filename = f"snow_depth_{start_str}_{end_str}"

            # 总是保存 CSV 格式
            csv_path = self.output_dir / f"{base_filename}.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"结果保存为 CSV: {csv_path}")

            # 保存为 Parquet
            parquet_path = self.output_dir / f"{base_filename}.parquet"
            df.to_parquet(parquet_path, index=False)
            self.logger.info(f"结果保存为 Parquet: {parquet_path}")

            return str(parquet_path)

        except Exception as e:
            self.logger.exception(f"保存结果失败: {str(e)}")
            return None

    def _get_filename_from_date_str(self, date_str):
        """根据日期字符串生成文件名"""
        return f"{date_str.replace('-', '')}.txt"

    def _parse_ascii_grid_header(self, file_path):
        """解析ASCII Grid文件的头信息"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                header = {}
                for i in range(6):
                    line = f.readline().strip()
                    if not line:
                        raise ValueError("文件头不完整")

                    parts = re.split(r'\s+', line, maxsplit=1)
                    if len(parts) == 2:
                        key = parts[0].lower()
                        value = parts[1].strip()

                        if key in ['ncols', 'nrows']:
                            header[key] = int(value)
                        elif key in ['xllcenter', 'yllcenter', 'cellsize', 'nodata_value']:
                            header[key] = float(value)
                        else:
                            header[key] = value

                # 验证必要字段
                required_keys = ['ncols', 'nrows', 'xllcenter', 'yllcenter', 'cellsize', 'nodata_value']
                missing_keys = [key for key in required_keys if key not in header]
                if missing_keys:
                    raise ValueError(f"缺少必要的头文件信息: {missing_keys}")

                return header

        except Exception as e:
            self.logger.error(f"解析ASCII文件头失败 {file_path}: {str(e)}")
            raise

    def _coords_to_rowcol(self, lon, lat, header):
        """将经纬度坐标转换为行列号"""
        try:
            xll = header['xllcenter']
            yll = header['yllcenter']
            cell_size = header['cellsize']
            nrows = header['nrows']

            # 计算列号（基于经度）
            col = int((lon - xll) / cell_size)

            # 计算行号（基于纬度，注意ASCII Grid是从上到下存储）
            row = int((yll + (nrows - 1) * cell_size - lat) / cell_size)

            return row, col

        except Exception as e:
            self.logger.error(f"坐标转换失败: lon={lon}, lat={lat}, error={str(e)}")
            return -1, -1

    def _get_pixel_value(self, file_path, row, col):
        """读取指定行列的像素值"""
        try:
            header = self._parse_ascii_grid_header(file_path)

            # 验证行列号
            if row < 0 or row >= header['nrows'] or col < 0 or col >= header['ncols']:
                return header['nodata_value']

            with open(file_path, 'r', encoding='utf-8') as f:
                # 跳过头文件（6行）
                for _ in range(6):
                    f.readline()

                # 跳转到指定行
                for i in range(row):
                    f.readline()

                # 读取指定行
                line = f.readline().strip()
                if line:
                    values = line.split()
                    if col < len(values):
                        try:
                            return np.float32(values[col])
                        except ValueError:
                            return header['nodata_value']

            return header['nodata_value']

        except Exception as e:
            self.logger.error(f"读取像素值失败 {file_path}: {str(e)}")
            return None

    def close(self):
        """关闭资源"""
        try:
            if self.db_conn:
                self.db_conn.close()
                self.logger.debug("数据库连接已关闭")
        except Exception as e:
            self.logger.warning(f"关闭数据库连接时出错: {str(e)}")

    def __del__(self):
        """析构函数"""
        self.close()