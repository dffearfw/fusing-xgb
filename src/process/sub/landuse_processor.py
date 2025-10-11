import logging
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from src.process.config import config
from datetime import datetime

logger = logging.getLogger("LandUseProcessor")


class LandUseProcessor:
    """静态土地利用分类数据处理器"""

    def __init__(self, secure_processor=None, station_filter=None):
        self.logger = logging.getLogger("LandUseProcessor")
        self.conf = config.landuse
        self.secure_processor = secure_processor
        self.station_filter = station_filter

        # 输出目录
        self.output_dir = Path(config.get_output_dir('landuse'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据库连接
        self.db_conn = self.connect_database()

        # 分类映射
        self.class_mapping = self.conf.get('class_mapping', {})

        # 标记为静态数据处理器
        self.is_static = self.conf.get('is_static', True)
        self.static_year = self.conf.get('static_year', 2015)

        self.logger.info(f"初始化土地利用处理器完成 - 静态数据: {self.is_static}, 年份: {self.static_year}")

    def get_date_range(self):
        """获取日期范围 - 静态数据专用"""
        if self.is_static:
            # 静态数据返回固定日期范围

            start_date = datetime(self.static_year, 1, 1)
            end_date = datetime(self.static_year, 12, 31)
            self.logger.info(f"静态数据日期范围: {start_date.date()} 至 {end_date.date()}")
            return start_date, end_date
        else:
            # 动态数据从配置读取
            date_range = self.conf.get('date_range', {})
            if not date_range:
                self.logger.error("动态数据缺少日期范围配置")
                return None, None

            try:
                start_date = datetime.strptime(date_range['start'], '%Y-%m-%d')
                end_date = datetime.strptime(date_range['end'], '%Y-%m-%d')
                return start_date, end_date
            except Exception as e:
                self.logger.error(f"解析日期范围失败: {e}")
                return None, None

    def process_range(self, start_date, end_date):
        """处理日期范围 - 为所有时间点的站点提供土地利用数据"""
        self.logger.info(f"土地利用数据处理: 为所有时间点的站点提供土地利用分类")

        # 对于土地利用数据，我们忽略传入的日期范围
        # 因为土地利用是静态的，应该为所有时间点的站点都提供数据
        return self.process()

    def process(self):
        """处理土地利用数据 - 为所有站点提供土地利用分类"""
        self.logger.info("开始处理土地利用数据（为所有站点）")

        try:
            # 获取所有站点（不限制时间）
            stations = self.get_all_stations()
            if not stations:
                self.logger.error("没有有效站点，无法处理")
                return None

            self.logger.info(f"找到 {len(stations)} 个站点，开始提取土地利用信息")

            # 提取土地利用值
            result_df = self.extract_landuse_values(stations)

            if result_df.empty:
                self.logger.warning("未提取到有效的土地利用数据")
                return None

            # 数据后处理
            processed_df = self.postprocess_data(result_df)

            # 保存结果
            output_path = self.save_results(processed_df)

            if output_path:
                # 生成数据摘要
                self._generate_landuse_summary(processed_df, Path(output_path))

                self.logger.info(f"处理完成! 总计 {len(processed_df)} 条记录, 结果保存至: {output_path}")
                return output_path
            else:
                self.logger.error("保存结果失败")
                return None

        except Exception as e:
            self.logger.error(f"土地利用处理失败: {str(e)}")
            self.logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
            return None

    def connect_database(self):
        """连接站点数据库"""
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

    def get_all_stations(self):
        """获取所有站点（不限制时间）"""
        try:
            # 获取数据库中所有不重复的站点及其坐标
            query = """
            SELECT DISTINCT station_ID, Longitude, Latitude 
            FROM stations 
            WHERE Longitude IS NOT NULL AND Latitude IS NOT NULL
            """

            df = pd.read_sql_query(query, self.db_conn)

            if not df.empty:
                # 确保数值类型
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

                # 移除无效坐标
                df = df.dropna(subset=['Latitude', 'Longitude'])

                self.logger.info(f"找到 {len(df)} 个有效站点（所有时间点）")

                # 记录坐标范围用于调试
                lon_range = (df['Longitude'].min(), df['Longitude'].max())
                lat_range = (df['Latitude'].min(), df['Latitude'].max())
                self.logger.info(
                    f"站点坐标范围: 经度({lon_range[0]:.2f} to {lon_range[1]:.2f}), 纬度({lat_range[0]:.2f} to {lat_range[1]:.2f})")

                # 检查是否在土地利用图像范围内
                self._check_station_coverage(df)

                return df.to_dict('records')
            else:
                self.logger.warning("未找到有效站点记录")
                return []

        except Exception as e:
            self.logger.exception(f"获取站点失败: {str(e)}")
            return []

    def extract_landuse_values(self, stations):
        """提取土地利用分类值 - 修复版本：只返回站点ID和土地利用信息，不包含日期"""
        try:
            data_file = Path(self.conf['path_template'])

            if not data_file.exists():
                self.logger.error(f"土地利用数据文件不存在: {data_file}")
                return pd.DataFrame()

            self.logger.info(f"处理土地利用数据: {data_file}")

            # 禁用GDAL磁盘空间检查
            import os
            original_check = os.environ.get('CHECK_DISK_FREE_SPACE', 'YES')
            os.environ['CHECK_DISK_FREE_SPACE'] = 'FALSE'

            results = []
            processed_count = 0
            success_count = 0

            try:
                with rasterio.open(data_file) as src:
                    self.logger.info(f"数据文件信息: CRS={src.crs}, 范围={src.bounds}")

                    # 创建投影转换器
                    transformer = Transformer.from_crs(
                        "EPSG:4326",  # WGS84
                        src.crs,  # Krasovsky_1940_Albers
                        always_xy=True
                    )

                    # 批量处理站点
                    for station in stations:
                        try:
                            lon = float(station.get('Longitude'))
                            lat = float(station.get('Latitude'))
                            station_id = station.get('station_ID')

                            # 将WGS84坐标转换为数据坐标系
                            x, y = transformer.transform(lon, lat)

                            # 将平面坐标转换为行列号
                            row, col = rowcol(src.transform, x, y)
                            row, col = int(row), int(col)

                            # 检查行列号是否在有效范围内
                            if 0 <= row < src.height and 0 <= col < src.width:
                                # 使用窗口读取单个像素值
                                window = rasterio.windows.Window(col, row, 1, 1)
                                value_array = src.read(1, window=window, boundless=True,
                                                       fill_value=self.conf.get('fill_value', 255))
                                value = int(value_array[0, 0])

                                fill_value = self.conf.get('fill_value', 255)
                                valid_min = self.conf.get('valid_min', 1)
                                valid_max = self.conf.get('valid_max', 62)

                                # 检查值是否有效
                                if value != fill_value and valid_min <= value <= valid_max:
                                    # 获取分类名称
                                    class_name = self.class_mapping.get(value, f"未知类型_{value}")

                                    # 关键修复：只包含站点ID和土地利用信息，不包含日期！
                                    results.append({
                                        'station_id': station_id,
                                        'landuse_code': value,
                                        'landuse_class': class_name,
                                        'longitude': lon,  # 保留坐标用于调试
                                        'latitude': lat,
                                        # 不包含任何日期字段！
                                        'source_file': data_file.name
                                    })
                                    success_count += 1
                            else:
                                self.logger.debug(f"站点 {station_id} 坐标超出图像范围: 行{row}, 列{col}")

                            processed_count += 1

                            # 每处理200个站点记录一次进度
                            if processed_count % 200 == 0:
                                self.logger.info(f"处理进度: {processed_count}/{len(stations)} 站点, 成功: {success_count}")

                        except Exception as e:
                            self.logger.debug(f"处理站点 {station.get('station_ID')} 失败: {str(e)}")
                            continue

            finally:
                os.environ['CHECK_DISK_FREE_SPACE'] = original_check

            self.logger.info(f"土地利用提取完成: 处理 {processed_count} 站点, 成功提取 {success_count} 条记录")

            if results:
                df = pd.DataFrame(results)

                # 确保没有日期列
                date_columns = ['date', 'year', 'month', 'data_year', 'processing_year']
                for col in date_columns:
                    if col in df.columns:
                        df = df.drop(col, axis=1)
                        self.logger.info(f"移除意外的日期列: {col}")

                # 统计分类分布
                self._log_class_distribution(df)

                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.exception(f"提取土地利用值失败: {str(e)}")
            return pd.DataFrame()

    def _check_station_coverage(self, stations_df):
        """检查站点在土地利用图像中的覆盖情况"""
        try:
            data_file = Path(self.conf['path_template'])
            if not data_file.exists():
                return

            with rasterio.open(data_file) as src:
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

                in_bounds_count = 0
                out_of_bounds_count = 0

                for _, station in stations_df.iterrows():
                    lon = station['Longitude']
                    lat = station['Latitude']

                    try:
                        x, y = transformer.transform(lon, lat)
                        image_bounds = src.bounds

                        if (image_bounds.left <= x <= image_bounds.right and
                                image_bounds.bottom <= y <= image_bounds.top):
                            in_bounds_count += 1
                        else:
                            out_of_bounds_count += 1
                    except:
                        out_of_bounds_count += 1

                self.logger.info(f"站点覆盖检查: {in_bounds_count} 个在图像范围内, {out_of_bounds_count} 个在图像范围外")

        except Exception as e:
            self.logger.warning(f"站点覆盖检查失败: {str(e)}")

    def _log_class_distribution(self, df):
        """记录土地利用分类分布"""
        if 'landuse_code' in df.columns:
            class_dist = df['landuse_code'].value_counts().sort_index()
            self.logger.info("土地利用分类分布:")
            for code, count in class_dist.items():
                class_name = self.class_mapping.get(code, f"未知类型_{code}")
                percentage = (count / len(df)) * 100
                self.logger.info(f"  {code:2d}({class_name:8s}): {count:4d} 个站点 ({percentage:5.1f}%)")

    def postprocess_data(self, df):
        """数据后处理 - 修复版本：不添加处理时间"""
        if df.empty:
            return df

        # 创建副本
        result_df = df.copy()

        # 关键修复：不添加处理时间，保持数据纯净
        # 土地利用数据应该是纯粹的静态数据，不包含任何时间信息

        # 确保没有时间相关列
        time_columns = ['processing_time', 'data_version']
        for col in time_columns:
            if col in result_df.columns:
                result_df = result_df.drop(col, axis=1)

        self.logger.info(f"后处理完成: {len(result_df)} 条记录（纯净的静态数据）")
        return result_df

    def save_results(self, df):
        """保存结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"landuse_2015_{timestamp}"

            # 总是保存 CSV 格式
            csv_path = self.output_dir / f"{base_filename}.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"结果保存为 CSV: {csv_path}")

            # 根据配置保存其他格式
            output_conf = self.conf.get('output', {})
            output_format = output_conf.get('primary_format', 'parquet').lower()

            if output_format == 'parquet':
                try:
                    parquet_path = self.output_dir / f"{base_filename}.parquet"
                    df.to_parquet(parquet_path, index=False)
                    self.logger.info(f"结果保存为 Parquet: {parquet_path}")
                    return str(parquet_path)
                except ImportError:
                    self.logger.warning("Parquet 支持不可用，仅保存 CSV 格式")
                    return str(csv_path)
            else:
                return str(csv_path)

        except Exception as e:
            self.logger.exception(f"保存结果失败: {str(e)}")
            return None



    def _generate_landuse_summary(self, df, output_path):
        """生成土地利用数据摘要"""
        try:
            summary_path = output_path.with_suffix('.summary.txt')

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=== 2015年土地利用数据摘要 ===\n")
                f.write(f"记录数: {len(df)}\n")
                f.write(f"站点数: {df['station_id'].nunique()}\n")
                f.write(f"数据年份: 2015\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # 土地利用分类统计
                f.write("\n=== 土地利用分类分布 ===\n")
                class_stats = df.groupby(['landuse_code', 'landuse_class']).size().reset_index(name='count')
                class_stats = class_stats.sort_values('count', ascending=False)
                class_stats['percentage'] = (class_stats['count'] / len(df) * 100).round(1)
                f.write(class_stats.to_string(index=False))

                # 空间分布统计
                f.write("\n\n=== 空间分布 ===\n")
                f.write(f"经度范围: {df['longitude'].min():.2f} 至 {df['longitude'].max():.2f}\n")
                f.write(f"纬度范围: {df['latitude'].min():.2f} 至 {df['latitude'].max():.2f}\n")

                # 前10条记录
                f.write("\n\n=== 前10条记录 ===\n")
                f.write(df.head(10).to_string())

            self.logger.info(f"数据摘要已生成: {summary_path}")

        except Exception as e:
            self.logger.warning(f"生成数据摘要失败: {str(e)}")

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