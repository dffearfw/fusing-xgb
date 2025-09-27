"""
Create on 2025/8/21

@auther:Thinkpad
"""
# FIXME: 无法解析文件名,无法从文件名解析卫星传感器信息

import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import h5py
import re

logger = logging.getLogger("CSWEProcessor")


class CSWEProcessor:
    """CSWE 数据处理器"""

    def __init__(self, secure_processor=None, station_filter=None):
        """
        初始化 CSWE 处理器

        参数:
            secure_processor: 安全处理器实例
            station_filter: 站点过滤器
        """
        from src.process.config import config

        self.logger = logging.getLogger("CSWEProcessor")
        self.conf = config.cswe
        self.secure_processor = secure_processor
        self.station_filter = station_filter

        # 输出目录
        self.output_dir = Path(config.get_output_dir('cswe'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据库连接
        self.db_conn = self.connect_database()

        self.logger.info("初始化 CSWE 处理器完成")

    def connect_database(self):
        """连接站点数据库"""
        from src.process.config import config

        db_path = config.get_station_db_path()
        try:
            conn = sqlite3.connect(db_path)
            conn.enable_load_extension(True)

            # 尝试加载空间扩展
            try:
                conn.load_extension("mod_spatialite")
                self.logger.debug("成功加载空间扩展")
            except Exception as e:
                self.logger.warning(f"加载空间扩展失败: {str(e)}")

            return conn
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")
            raise

    def get_stations_for_date(self, target_date):
        """获取特定日期有效的站点（处理浮点日期格式）"""
        try:
            # 转换为数据库格式 (YYYYMMDD.0)
            db_date_format = float(target_date.strftime('%Y%m%d'))

            # 基础查询
            query = "SELECT station_ID, Longitude, Latitude FROM stations WHERE time = ?"
            params = [db_date_format]

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

            # 执行查询
            df = pd.read_sql_query(query, self.db_conn, params=params)

            if not df.empty:
                # 确保数值类型
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

                # 移除无效坐标
                df = df.dropna(subset=['Latitude', 'Longitude'])

                self.logger.debug(f"找到 {len(df)} 个有效站点")
                return df.to_dict('records')
            else:
                self.logger.warning(f"未找到时间 {db_date_format} 的站点记录")
                return []

        except Exception as e:
            self.logger.exception(f"获取站点失败: {str(e)}")
            return []

    def parse_filename(self, filename):
        """简化版文件名解析"""
        try:
            # 直接使用配置中的文件模式
            pattern = r"(.+)_SWE_(\d{8})_DAILY_025KM_V1\.2\.h5"
            match = re.match(pattern, filename)

            if match:
                satellite_sensor = match.group(1)  # 如 "F17_SSMIS"
                date_str = match.group(2)  # 如 "20130102"

                # 验证是否为支持的组合
                valid_combinations = self.conf.get('satellite_sensor_combinations', [])
                if valid_combinations and satellite_sensor not in valid_combinations:
                    self.logger.warning(f"不支持的卫星传感器组合: {satellite_sensor}")
                    return None, None

                # 解析日期
                date = datetime.strptime(date_str, '%Y%m%d')
                return satellite_sensor, date

            self.logger.error(f"无法解析文件名: {filename}")
            return None, None

        except Exception as e:
            self.logger.exception(f"文件名解析失败: {str(e)}")
            return None, None

    def find_file_for_date(self, date):
        """根据日期查找匹配的文件"""
        try:
            data_dir = Path(self.conf['data_dir'])

            if not data_dir.exists():
                self.logger.error(f"数据目录不存在: {data_dir}")
                return None

            # 构建日期字符串
            date_str = date.strftime('%Y%m%d')

            # 获取所有卫星传感器组合
            combinations = self.conf.get('satellite_sensor_combinations', [])

            # 查找匹配的文件
            for combination in combinations:
                # 构建文件名
                filename = self.conf['file_pattern'].format(
                    satellite_sensor=combination,
                    date=date_str
                )
                file_path = data_dir / filename

                if file_path.exists():
                    self.logger.debug(f"找到匹配文件: {file_path}")
                    return file_path

            self.logger.warning(f"未找到日期 {date_str} 的CSWE文件")
            return None

        except Exception as e:
            self.logger.exception(f"查找文件失败: {str(e)}")
            return None

    def extract_values(self, date, file_path):
        """从HDF5文件中提取站点值 - 针对2D网格坐标优化版"""
        try:
            # 从文件名解析卫星传感器信息
            result = self.parse_filename(file_path.name)
            if not result or None in result:
                self.logger.error(f"无法从文件名解析信息: {file_path.name}")
                return pd.DataFrame()

            satellite_sensor, file_date = result

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

            # 打开HDF5文件
            with h5py.File(file_path, 'r') as h5_file:
                # 调试：打印文件结构
                self.logger.debug("=== HDF5文件结构 ===")
                for key in h5_file.keys():
                    dataset = h5_file[key]
                    shape = dataset.shape if hasattr(dataset, 'shape') else 'N/A'
                    self.logger.debug(f"数据集: {key}, 形状: {shape}")
                    if hasattr(dataset, 'attrs') and dataset.attrs:
                        self.logger.debug(f"  属性: {dict(dataset.attrs)}")

                # 查找SWE数据集
                dataset_name = None
                swe_candidates = ['SWE[mm]', 'SWE', 'swe', 'Snow_Water_Equivalent', 'SD[cm]']
                for candidate in swe_candidates:
                    if candidate in h5_file.keys():
                        dataset_name = candidate
                        break

                if dataset_name is None:
                    self.logger.error("找不到SWE数据集，可用数据集: %s", list(h5_file.keys()))
                    return pd.DataFrame()

                dataset = h5_file[dataset_name]
                self.logger.info(f"使用数据集: {dataset_name}, 形状: {dataset.shape}")

                # 获取SWE数据
                data = dataset[:]
                data_min, data_max = np.nanmin(data), np.nanmax(data)
                self.logger.info(f"SWE数据范围: {data_min:.2f} 到 {data_max:.2f}")

                # 获取经纬度网格
                if 'Latitude' in h5_file and 'Longitude' in h5_file:
                    lat_grid = np.array(h5_file['Latitude'][:])
                    lon_grid = np.array(h5_file['Longitude'][:])
                    self.logger.debug(f"纬度网格范围: {np.nanmin(lat_grid):.2f} 到 {np.nanmax(lat_grid):.2f}")
                    self.logger.debug(f"经度网格范围: {np.nanmin(lon_grid):.2f} 到 {np.nanmax(lon_grid):.2f}")
                else:
                    self.logger.error("找不到经纬度网格数据集")
                    return pd.DataFrame()

                # 验证网格形状匹配
                if lat_grid.shape != data.shape or lon_grid.shape != data.shape:
                    self.logger.error(f"网格形状不匹配: data{data.shape}, lat{lat_grid.shape}, lon{lon_grid.shape}")
                    return pd.DataFrame()

                # 提取站点值
                results = []
                valid_count = 0
                invalid_count = 0

                for i, (target_lon, target_lat) in enumerate(zip(lons, lats)):
                    try:
                        # 在2D网格中找到最近的点
                        distance = (lon_grid - target_lon) ** 2 + (lat_grid - target_lat) ** 2
                        min_idx = np.unravel_index(np.argmin(distance), distance.shape)
                        row_idx, col_idx = min_idx

                        value = float(data[row_idx, col_idx])
                        matched_lat = lat_grid[row_idx, col_idx]
                        matched_lon = lon_grid[row_idx, col_idx]
                        distance_km = self._calculate_distance(target_lat, target_lon, matched_lat, matched_lon)

                        # 检查填充值和有效范围
                        fill_value = self.conf.get('fill_value', 255.0)  # 根据样例数据，填充值可能是255
                        valid_min = self.conf.get('valid_min', 0.0)
                        valid_max = self.conf.get('valid_max', 1000.0)  # 添加最大值检查

                        # 调试信息
                        if i < 5:  # 只记录前5个站点的详细匹配信息
                            self.logger.debug(
                                f"站点 {station_ids[i]}: 原始({target_lat:.3f}, {target_lon:.3f}) -> "
                                f"网格({matched_lat:.3f}, {matched_lon:.3f}), 距离: {distance_km:.2f}km, "
                                f"值: {value}"
                            )

                        if (value != fill_value and
                                valid_min <= value <= valid_max and
                                distance_km < 50.0):  # 最大允许匹配距离50km

                            # 应用缩放因子
                            scale_factor = self.conf.get('scale_factor', 1.0)
                            value *= scale_factor

                            results.append({
                                'station_id': station_ids[i],
                                'date': date.strftime('%Y-%m-%d'),
                                'value': value,
                                'file_used': file_path.name,
                                'dataset': dataset_name,
                                'satellite_sensor': satellite_sensor,
                                'original_lat': target_lat,
                                'original_lon': target_lon,
                                'matched_lat': matched_lat,
                                'matched_lon': matched_lon,
                                'distance_km': distance_km,
                                'grid_row': row_idx,
                                'grid_col': col_idx
                            })
                            valid_count += 1
                        else:
                            invalid_count += 1
                            if value == fill_value:
                                self.logger.debug(f"站点 {station_ids[i]} 值为填充值: {value}")
                            elif not (valid_min <= value <= valid_max):
                                self.logger.debug(f"站点 {station_ids[i]} 值超出有效范围: {value}")
                            else:
                                self.logger.debug(f"站点 {station_ids[i]} 匹配距离过远: {distance_km:.2f}km")

                    except Exception as e:
                        self.logger.warning(f"提取站点 {station_ids[i]} 值失败: {str(e)}")
                        invalid_count += 1

                # 记录统计信息
                self.logger.info(
                    f"提取完成: 有效{valid_count}个, 无效{invalid_count}个, "
                    f"成功率: {valid_count / (valid_count + invalid_count) * 100:.1f}%"
                )

                if valid_count == 0:
                    self.logger.warning("未提取到任何有效数据，可能的原因:")
                    self.logger.warning("1. 站点坐标不在数据覆盖范围内")
                    self.logger.warning("2. 所有值都是填充值(255)")
                    self.logger.warning("3. 匹配距离过远")
                    self.logger.warning(f"数据覆盖范围: 纬度({np.nanmin(lat_grid):.2f} to {np.nanmax(lat_grid):.2f}), "
                                        f"经度({np.nanmin(lon_grid):.2f} to {np.nanmax(lon_grid):.2f})")

                return pd.DataFrame(results)

        except Exception as e:
            self.logger.exception(f"提取错误: {str(e)}")
            return pd.DataFrame()

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """计算两点之间的距离(km) - 使用Haversine公式"""
        from math import radians, sin, cos, sqrt, atan2

        # 将十进制度数转化为弧度
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = 6371 * c  # 地球平均半径(km)

        return distance

    def postprocess_data(self, df, date):
        """数据后处理"""
        if df.empty:
            return df

        # 创建副本
        result_df = df.copy()

        # 过滤无效值
        valid_min = self.conf.get('valid_min', 0)
        fill_value = self.conf.get('fill_value', -9999)

        # 使用 .loc 进行条件过滤
        mask = (result_df['value'] >= valid_min) & (result_df['value'] != fill_value)
        result_df = result_df.loc[mask].copy()

        # 应用缩放因子
        scale_factor = self.conf.get('scale_factor', 1.0)
        if scale_factor != 1.0:
            result_df.loc[:, 'value'] = result_df['value'] * scale_factor

        # 添加处理时间
        processing_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_df.loc[:, 'processing_time'] = processing_time

        self.logger.debug(f"后处理完成: {len(result_df)} 条记录")
        return result_df

    def save_results(self, df, start_date, end_date):
        """保存结果"""
        try:
            base_filename = f"cswe_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

            # 总是保存 CSV 格式（用于查看）
            csv_path = self.output_dir / f"{base_filename}.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"结果保存为 CSV (用于查看): {csv_path}")

            # 根据配置保存其他格式
            output_format = self.conf.get('output_format', 'parquet').lower()

            if output_format == 'parquet':
                try:
                    parquet_path = self.output_dir / f"{base_filename}.parquet"
                    df.to_parquet(parquet_path, index=False)
                    self.logger.info(f"结果保存为 Parquet (用于处理): {parquet_path}")
                    return str(parquet_path)
                except ImportError:
                    self.logger.warning("Parquet 支持不可用，仅保存 CSV 格式")
                    return str(csv_path)

            elif output_format == 'feather':
                try:
                    feather_path = self.output_dir / f"{base_filename}.feather"
                    df.to_feather(feather_path)
                    self.logger.info(f"结果保存为 Feather: {feather_path}")
                    return str(feather_path)
                except ImportError:
                    self.logger.warning("Feather 支持不可用，仅保存 CSV 格式")
                    return str(csv_path)

            else:
                self.logger.warning(f"未知的输出格式: {output_format}，仅保存 CSV 格式")
                return str(csv_path)

        except Exception as e:
            self.logger.exception(f"保存结果失败: {str(e)}")
            return None

    def generate_data_summary(self, df, output_path):
        """生成数据摘要"""
        try:
            summary_path = output_path.with_suffix('.summary.txt')

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=== CSWE 数据摘要 ===\n")
                f.write(f"记录数: {len(df)}\n")
                f.write(f"时间段: {df['date'].min()} 至 {df['date'].max()}\n")
                f.write(f"站点数: {df['station_id'].nunique()}\n")
                f.write(f"值范围: {df['value'].min():.6f} 至 {df['value'].max():.6f}\n")
                f.write(f"平均值: {df['value'].mean():.6f}\n")
                f.write(f"中位数: {df['value'].median():.6f}\n")

                # 添加数据集统计
                if 'dataset' in df.columns:
                    f.write(f"\n数据集分布:\n")
                    dataset_stats = df['dataset'].value_counts()
                    f.write(dataset_stats.to_string())

                f.write("\n=== 前10条记录 ===\n")
                f.write(df.head(10).to_string())

                f.write("\n\n=== 站点统计 ===\n")
                station_stats = df.groupby('station_id')['value'].agg(['count', 'min', 'max', 'mean'])
                f.write(station_stats.to_string())

            self.logger.info(f"数据摘要已生成: {summary_path}")

        except Exception as e:
            self.logger.warning(f"生成数据摘要失败: {str(e)}")

    def process_range(self, start_date, end_date):
        """处理日期范围"""
        self.logger.info(f"开始处理 {start_date} 至 {end_date} 的CSWE数据")

        # 转换为日期范围
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        date_range = pd.date_range(start_date, end_date, freq='D')

        all_results = []
        processed_dates = 0
        success_dates = 0

        for current_date in date_range:
            processed_dates += 1
            try:
                # 查找文件
                file_path = self.find_file_for_date(current_date)
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
                    if not processed_df.empty:
                        all_results.append(processed_df)
                        success_dates += 1

            except Exception as e:
                self.logger.error(f"{current_date} 处理失败: {str(e)}")
                continue

        # 合并结果
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            output_path = self.save_results(final_df, start_date, end_date)

            # 生成数据摘要
            self.generate_data_summary(final_df, Path(output_path))

            self.logger.info(
                f"处理完成! 成功处理 {success_dates}/{processed_dates} 天, "
                f"总计 {len(final_df)} 条记录, 结果保存至: {output_path}"
            )
            return output_path
        else:
            self.logger.warning(f"未生成有效结果, 处理了 {processed_dates} 天")
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
        """析构函数，确保资源被释放"""
        self.close()

    def _find_coordinates(self, h5_file, dataset):
        """查找HDF5文件中的经纬度坐标信息（修复类型错误）"""
        try:
            # 方法1: 检查独立的lat/lon数据集
            if 'lat' in h5_file and 'lon' in h5_file:
                try:
                    latitudes = np.array(h5_file['lat'][:])  # 确保转换为numpy数组
                    longitudes = np.array(h5_file['lon'][:])
                    self.logger.debug("从独立数据集获取坐标")
                    return latitudes, longitudes
                except Exception as e:
                    self.logger.warning(f"读取独立坐标数据集失败: {str(e)}")

            # 方法2: 检查数据集属性
            if hasattr(dataset, 'attrs'):
                attrs = dict(dataset.attrs)

                # 检查常见的坐标属性名
                coord_keys = {
                    'lat': ['lat', 'latitude', 'lats', 'Latitude', 'LAT', 'Y', 'y'],
                    'lon': ['lon', 'longitude', 'lons', 'Longitude', 'LON', 'X', 'x']
                }

                lat_data = None
                lon_data = None

                # 查找纬度数据
                for key in coord_keys['lat']:
                    if key in attrs:
                        lat_data = np.array(attrs[key])  # 转换为numpy数组
                        break

                # 查找经度数据
                for key in coord_keys['lon']:
                    if key in attrs:
                        lon_data = np.array(attrs[key])  # 转换为numpy数组
                        break

                if lat_data is not None and lon_data is not None:
                    self.logger.debug("从数据集属性获取坐标")
                    return lat_data, lon_data

            # 方法3: 检查坐标变量（常见于NetCDF4格式的HDF5）
            coord_vars = ['lat', 'lon', 'latitude', 'longitude', 'y', 'x']
            for var_name in coord_vars:
                if var_name in h5_file:
                    try:
                        var_data = np.array(h5_file[var_name][:])
                        if var_name in ['lat', 'latitude', 'y']:
                            latitudes = var_data
                        elif var_name in ['lon', 'longitude', 'x']:
                            longitudes = var_data
                    except Exception as e:
                        self.logger.debug(f"读取坐标变量 {var_name} 失败: {str(e)}")

            if 'latitudes' in locals() and 'longitudes' in locals():
                self.logger.debug("从坐标变量获取坐标")
                return latitudes, longitudes

            # 方法4: 尝试从数据维度推断
            try:
                # 获取数据形状
                data_shape = dataset.shape

                if len(data_shape) == 2:
                    # 假设是 (lat, lon) 格式
                    lat_size, lon_size = data_shape

                    # 创建默认的WGS84网格
                    latitudes = np.linspace(-90.0, 90.0, lat_size)
                    longitudes = np.linspace(-180.0, 180.0, lon_size)

                    self.logger.info(f"从数据维度推断坐标: 纬度{lat_size}点, 经度{lon_size}点")
                    return latitudes, longitudes

            except Exception as e:
                self.logger.debug(f"从维度推断坐标失败: {str(e)}")

            # 方法5: 使用默认的WGS84网格
            self.logger.warning("无法从文件获取坐标信息，使用默认WGS84网格")

            # 假设0.25度分辨率，根据实际情况调整
            lat_res = 0.25
            lon_res = 0.25

            latitudes = np.arange(-90.0, 90.0 + lat_res, lat_res)
            longitudes = np.arange(-180.0, 180.0 + lon_res, lon_res)

            # 确保坐标数量与数据形状匹配
            try:
                if len(dataset.shape) == 2:
                    lat_dim, lon_dim = dataset.shape
                    if len(latitudes) != lat_dim:
                        latitudes = np.linspace(-90.0, 90.0, lat_dim)
                    if len(longitudes) != lon_dim:
                        longitudes = np.linspace(-180.0, 180.0, lon_dim)
            except:
                pass

            self.logger.info(f"使用默认网格: 纬度{len(latitudes)}点, 经度{len(longitudes)}点")
            return latitudes, longitudes

        except Exception as e:
            self.logger.error(f"查找坐标信息失败: {str(e)}")
            return None, None

