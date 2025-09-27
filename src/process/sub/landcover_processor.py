import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer, CRS
import tempfile

logger = logging.getLogger("LandcoverProcessor")


class LandcoverProcessor:
    """地物分类数据处理器"""

    def __init__(self, secure_processor=None, station_filter=None):
        """
        初始化地物分类处理器

        参数:
            secure_processor: 安全处理器实例
            station_filter: 站点过滤器
        """
        from src.process.config import config

        self.logger = logging.getLogger("LandcoverProcessor")
        self.conf = config.landcover
        self.secure_processor = secure_processor
        self.station_filter = station_filter

        # 输出目录
        self.output_dir = Path(config.get_output_dir('landcover'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据库连接
        self.db_conn = self.connect_database()

        # 投影转换器
        self.proj_transformer = None

        self.logger.info("初始化地物分类处理器完成")

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
        """获取特定日期有效的站点"""
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

    def get_file_path(self, date):
        """根据日期生成文件路径"""
        try:
            # 地物分类文件按年存储
            path_template = self.conf['path_template']

            # 替换年占位符
            replacements = {
                '{year}': date.strftime('%Y')
            }

            file_path = path_template
            for placeholder, value in replacements.items():
                file_path = file_path.replace(placeholder, value)

            return Path(file_path)

        except Exception as e:
            self.logger.error(f"文件路径生成失败: {str(e)}")
            return None

    def _reproject_to_wgs84(self, src_path, dst_path):
        """将AIBERS投影重投影到WGS84"""
        try:
            with rasterio.open(src_path) as src:
                # 计算目标变换和形状
                transform, width, height = calculate_default_transform(
                    src.crs, 'EPSG:4326', src.width, src.height, *src.bounds
                )

                # 目标元数据
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': 'EPSG:4326',
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                # 重投影
                with rasterio.open(dst_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs='EPSG:4326',
                            resampling=Resampling.nearest
                        )

            return True
        except Exception as e:
            self.logger.error(f"重投影失败: {str(e)}")
            return False

    def extract_values(self, date, file_path):
        """从TIFF文件中提取站点值 - 修复坐标转换问题"""
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

            # 禁用GDAL磁盘空间检查
            import os
            original_check = os.environ.get('CHECK_DISK_FREE_SPACE', 'YES')
            os.environ['CHECK_DISK_FREE_SPACE'] = 'FALSE'

            results = []
            try:
                with rasterio.open(file_path) as src:
                    self.logger.info(f"文件CRS: {src.crs}")
                    self.logger.info(f"文件范围: {src.bounds}")

                    # 创建坐标转换器
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(
                        "EPSG:4326",  # WGS84
                        src.crs,  # Albers投影
                        always_xy=True
                    )

                    # 转换所有站点坐标到Albers投影
                    alber_xs, alber_ys = transformer.transform(lons, lats)

                    # 详细的范围分析
                    image_bounds = src.bounds
                    self.logger.info("=== 坐标范围分析 ===")
                    self.logger.info(f"图像X范围: {image_bounds.left:.2f} 到 {image_bounds.right:.2f}")
                    self.logger.info(f"图像Y范围: {image_bounds.bottom:.2f} 到 {image_bounds.top:.2f}")

                    in_bounds_count = 0
                    out_of_bounds_examples = []

                    for i, (alber_x, alber_y) in enumerate(zip(alber_xs, alber_ys)):
                        in_x = image_bounds.left <= alber_x <= image_bounds.right
                        in_y = image_bounds.bottom <= alber_y <= image_bounds.top

                        if in_x and in_y:
                            in_bounds_count += 1
                            # 记录前几个在范围内的站点用于调试
                            if len(out_of_bounds_examples) < 3:
                                out_of_bounds_examples.append(
                                    f"站点{station_ids[i]}: WGS84({lons[i]:.2f}, {lats[i]:.2f}) -> "
                                    f"Albers({alber_x:.2f}, {alber_y:.2f}) - 在范围内"
                                )
                        else:
                            # 记录前几个超出范围的站点
                            if len(out_of_bounds_examples) < 6:  # 增加记录数量
                                reason = []
                                if not in_x:
                                    reason.append(f"X超出[{image_bounds.left:.0f}, {image_bounds.right:.0f}]")
                                if not in_y:
                                    reason.append(f"Y超出[{image_bounds.bottom:.0f}, {image_bounds.top:.0f}]")

                                out_of_bounds_examples.append(
                                    f"站点{station_ids[i]}: WGS84({lons[i]:.2f}, {lats[i]:.2f}) -> "
                                    f"Albers({alber_x:.2f}, {alber_y:.2f}) - {', '.join(reason)}"
                                )

                    self.logger.info(f"站点位置分析: 总共{len(alber_xs)}个, 在图像范围内{in_bounds_count}个")

                    if out_of_bounds_examples:
                        self.logger.info("站点坐标转换示例:")
                        for example in out_of_bounds_examples:
                            self.logger.info(f"  {example}")

                    if in_bounds_count == 0:
                        self.logger.error("没有任何站点在图像范围内！")
                        self.logger.error("可能的原因:")
                        self.logger.error("1. 站点坐标错误或格式问题")
                        self.logger.error("2. 图像覆盖区域与站点区域不匹配")
                        self.logger.error("3. 坐标系统定义不一致")

                        # 添加详细的坐标分析
                        self._analyze_coordinate_problems(lons, lats, alber_xs, alber_ys, image_bounds, station_ids)
                        return pd.DataFrame()

                    # 提取在范围内的站点值
                    for i, (alber_x, alber_y) in enumerate(zip(alber_xs, alber_ys)):
                        # 检查是否在范围内
                        if not (image_bounds.left <= alber_x <= image_bounds.right and
                                image_bounds.bottom <= alber_y <= image_bounds.top):
                            continue

                        try:
                            # 将Albers坐标转换为行列号
                            row, col = rowcol(src.transform, alber_x, alber_y)

                            if 0 <= row < src.height and 0 <= col < src.width:
                                # 使用窗口读取单个像素
                                window = ((row, row + 1), (col, col + 1))
                                value_data = src.read(1, window=window)
                                value = int(value_data[0, 0]) if value_data.size > 0 else None

                                if value is not None:
                                    # 检查填充值和有效范围
                                    fill_value = self.conf.get('fill_value', 255)
                                    valid_min = self.conf.get('valid_min', 1)

                                    if value != fill_value and value >= valid_min:
                                        class_name = self.conf.get('landcover_classes', {}).get(value, f"Class_{value}")

                                        results.append({
                                            'station_id': station_ids[i],
                                            'date': date.strftime('%Y-%m-%d'),
                                            'value': value,
                                            'class_name': class_name,
                                            'file_used': file_path.name,
                                            'original_lon': lons[i],
                                            'original_lat': lats[i],
                                            'albers_x': alber_x,
                                            'albers_y': alber_y,
                                            'pixel_row': row,
                                            'pixel_col': col
                                        })
                                        self.logger.debug(f"成功提取站点 {station_ids[i]}: 值={value}")
                            else:
                                self.logger.debug(f"站点 {station_ids[i]} 行列号超出范围: ({row}, {col})")

                        except Exception as e:
                            self.logger.warning(f"提取站点 {station_ids[i]} 值失败: {str(e)}")

            finally:
                os.environ['CHECK_DISK_FREE_SPACE'] = original_check

            self.logger.info(f"提取完成: 成功{len(results)}条记录")
            return pd.DataFrame(results)

        except Exception as e:
            self.logger.exception(f"提取错误: {str(e)}")
            return pd.DataFrame()

    def _analyze_coordinate_problems(self, lons, lats, alber_xs, alber_ys, image_bounds, station_ids):
        """详细分析坐标问题"""
        self.logger.info("=== 详细坐标分析 ===")

        # 分析经纬度范围
        lon_min, lon_max = min(lons), max(lons)
        lat_min, lat_max = min(lats), max(lats)
        self.logger.info(f"站点经纬度范围: 经度({lon_min:.2f} to {lon_max:.2f}), 纬度({lat_min:.2f} to {lat_max:.2f})")

        # 分析Albers坐标范围
        x_min, x_max = min(alber_xs), max(alber_xs)
        y_min, y_max = min(alber_ys), max(alber_ys)
        self.logger.info(f"转换后Albers范围: X({x_min:.2f} to {x_max:.2f}), Y({y_min:.2f} to {y_max:.2f})")

        # 检查坐标符号问题
        negative_lons = sum(1 for lon in lons if lon < 0)
        negative_lats = sum(1 for lat in lats if lat < 0)
        self.logger.info(f"负经度站点: {negative_lons}个, 负纬度站点: {negative_lats}个")

        # 检查可能的坐标顺序问题
        if abs(lon_min) > 180 or abs(lon_max) > 180:
            self.logger.warning("经度值超出正常范围(-180 to 180)，可能存在坐标顺序问题")
        if abs(lat_min) > 90 or abs(lat_max) > 90:
            self.logger.warning("纬度值超出正常范围(-90 to 90)，可能存在坐标顺序问题")

        # 显示前10个站点的详细坐标
        self.logger.info("前10个站点详细坐标:")
        for i in range(min(10, len(station_ids))):
            self.logger.info(
                f"  站点{station_ids[i]}: 经度={lons[i]:.6f}, 纬度={lats[i]:.6f} -> X={alber_xs[i]:.2f}, Y={alber_ys[i]:.2f}")

    def postprocess_data(self, df, date):
        """数据后处理"""
        if df.empty:
            return df

        # 创建副本
        result_df = df.copy()

        # 过滤无效值
        fill_value = self.conf.get('fill_value', 255)
        valid_min = self.conf.get('valid_min', 1)

        # 使用 .loc 进行条件过滤
        mask = (result_df['value'] >= valid_min) & (result_df['value'] != fill_value)
        result_df = result_df.loc[mask].copy()

        # 添加处理时间
        processing_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_df.loc[:, 'processing_time'] = processing_time

        self.logger.debug(f"后处理完成: {len(result_df)} 条记录")
        return result_df

    def save_results(self, df, start_date, end_date):
        """保存结果 - 调整文件名"""
        try:
            # 使用年份范围而不是具体日期
            base_filename = f"landcover_{start_date.year}_{end_date.year}"

            # 总是保存 CSV 格式
            csv_path = self.output_dir / f"{base_filename}.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"结果保存为 CSV: {csv_path}")

            # 根据配置保存其他格式
            output_format = self.conf.get('output_format', 'parquet').lower()

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

    def process_range(self, start_date, end_date):
        """处理日期范围 - 完整版本"""
        self.logger.info(f"开始处理 {start_date} 至 {end_date} 的地物分类数据")

        # 转换为日期范围
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # 获取时间范围内的所有唯一站点
        all_stations = self._get_all_stations_in_range(start_date, end_date)

        if not all_stations:
            self.logger.error("在时间范围内没有找到任何站点数据")
            return None

        self.logger.info(f"找到 {len(all_stations)} 个唯一站点在处理时间范围内")

        # 按年处理地物分类数据
        years = range(start_date.year, end_date.year + 1)
        processed_years = 0
        success_years = 0
        all_results = []

        for year in years:
            processed_years += 1
            try:
                # 获取该年的地物分类文件路径
                file_path = self.get_file_path(datetime(year, 1, 1))

                # 检查文件是否存在
                if not file_path:
                    self.logger.warning(f"{year}年: 无法生成文件路径")
                    continue

                if not file_path.exists():
                    self.logger.warning(f"{year}年文件不存在: {file_path}")
                    continue

                self.logger.info(f"处理 {year} 年地物分类数据: {file_path.name}")

                # 提取数据 - 使用所有站点和当前年份
                yearly_df = self.extract_values_for_stations(all_stations, year, file_path)

                if not yearly_df.empty:
                    # 添加处理元数据
                    yearly_df['processing_year'] = year
                    yearly_df['source_file'] = file_path.name

                    all_results.append(yearly_df)
                    success_years += 1

                    # 统计该年的数据情况
                    unique_stations = yearly_df['station_id'].nunique()
                    self.logger.info(
                        f"{year}年: 成功提取 {len(yearly_df)} 条记录, "
                        f"涉及 {unique_stations} 个站点"
                    )

                    # 记录地物分类分布
                    if 'value' in yearly_df.columns:
                        value_counts = yearly_df['value'].value_counts().head(5)
                        self.logger.debug(f"{year}年主要地物类型: {value_counts.to_dict()}")
                else:
                    self.logger.warning(f"{year}年: 未提取到有效数据")

            except Exception as e:
                self.logger.error(f"{year} 年处理失败: {str(e)}")
                continue

        # 合并所有年份的结果
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)

            # 添加处理元数据
            processing_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            final_df['processing_time'] = processing_time
            final_df['processor_version'] = '1.0'

            # 数据统计
            total_records = len(final_df)
            unique_stations = final_df['station_id'].nunique()
            years_covered = final_df['processing_year'].nunique()

            self.logger.info(
                f"数据合并完成: 总计 {total_records} 条记录, "
                f"{unique_stations} 个站点, 覆盖 {years_covered} 年"
            )

            # 保存结果
            output_path = self.save_results(final_df, start_date, end_date)

            if output_path:
                # 生成数据摘要
                self._generate_landcover_summary(final_df, Path(output_path))

                self.logger.info(
                    f"处理完成! 成功处理 {success_years}/{processed_years} 年, "
                    f"结果保存至: {output_path}"
                )

                # 最终统计报告
                self._log_final_statistics(final_df, success_years, processed_years)

                return output_path
            else:
                self.logger.error("保存结果失败")
                return None
        else:
            self.logger.warning(
                f"未生成有效结果, 处理了 {processed_years} 年, "
                f"成功 {success_years} 年"
            )
            return None

    def _generate_landcover_summary(self, df, output_path):
        """生成地物分类数据摘要"""
        try:
            summary_path = output_path.with_suffix('.summary.txt')

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=== 地物分类数据摘要 ===\n")
                f.write(f"记录数: {len(df)}\n")
                f.write(f"时间段: {df['date'].min()} 至 {df['date'].max()}\n")
                f.write(f"站点数: {df['station_id'].nunique()}\n")

                # 地物分类统计
                f.write("\n=== 地物分类分布 ===\n")
                class_stats = df.groupby(['value', 'class_name']).size().reset_index(name='count')
                class_stats = class_stats.sort_values('count', ascending=False)
                f.write(class_stats.to_string(index=False))

                f.write("\n\n=== 前10条记录 ===\n")
                f.write(df.head(10).to_string())

                f.write("\n\n=== 站点统计 ===\n")
                station_stats = df.groupby('station_id')['value'].agg(['count', 'nunique'])
                station_stats.columns = ['总记录数', '地物类型数']
                f.write(station_stats.to_string())

            self.logger.info(f"数据摘要已生成: {summary_path}")

        except Exception as e:
            self.logger.warning(f"生成数据摘要失败: {str(e)}")

    def extract_values_for_stations(self, stations, year, file_path):
        """为指定站点列表提取值 - 保留原始日期"""
        try:
            # 直接从传入的站点列表提取坐标
            results = []

            for station in stations:
                try:
                    lat = float(station['Latitude'])
                    lon = float(station['Longitude'])
                    station_id = station['station_ID']

                    # 获取站点的实际日期（从数据库记录中）
                    station_date = self._get_station_date(station_id, year)
                    if not station_date:
                        continue

                    # 禁用GDAL磁盘空间检查
                    import os
                    os.environ['CHECK_DISK_FREE_SPACE'] = 'FALSE'

                    with rasterio.open(file_path) as src:
                        # 创建坐标转换器
                        from pyproj import Transformer
                        transformer = Transformer.from_crs(
                            "EPSG:4326", src.crs, always_xy=True
                        )

                        # 转换坐标
                        alber_x, alber_y = transformer.transform(lon, lat)

                        image_bounds = src.bounds

                        if (image_bounds.left <= alber_x <= image_bounds.right and
                                image_bounds.bottom <= alber_y <= image_bounds.top):

                            try:
                                row, col = rowcol(src.transform, alber_x, alber_y)
                                if 0 <= row < src.height and 0 <= col < src.width:
                                    window = ((row, row + 1), (col, col + 1))
                                    value_data = src.read(1, window=window)
                                    value = int(value_data[0, 0]) if value_data.size > 0 else None

                                    if value is not None:
                                        fill_value = self.conf.get('fill_value', 255)
                                        valid_min = self.conf.get('valid_min', 1)

                                        if value != fill_value and value >= valid_min:
                                            class_name = self.conf.get('landcover_classes', {}).get(value,
                                                                                                    f"Class_{value}")

                                            results.append({
                                                'station_id': station_id,
                                                'date': station_date.strftime('%Y-%m-%d'),  # 真实日期
                                                'value': value,
                                                'class_name': class_name,
                                                'file_used': file_path.name,
                                                'original_lon': lon,
                                                'original_lat': lat,
                                                'albers_x': alber_x,
                                                'albers_y': alber_y,
                                                'data_year': year  # 额外保留年份信息
                                            })
                            except Exception as e:
                                self.logger.debug(f"提取站点 {station_id} 值失败: {str(e)}")
                                continue

                except Exception as e:
                    self.logger.warning(f"处理站点 {station.get('station_ID', 'unknown')} 失败: {str(e)}")
                    continue

            return pd.DataFrame(results)

        except Exception as e:
            self.logger.error(f"提取站点值失败: {str(e)}")
            return pd.DataFrame()

    def _get_all_stations_in_range(self, start_date, end_date):
        """获取时间范围内的所有唯一站点及其最新位置"""
        try:
            # 转换为数据库格式
            start_db = float(start_date.strftime('%Y%m%d'))
            end_db = float(end_date.strftime('%Y%m%d'))

            # 获取每个站点的最新位置
            query = """
            SELECT s1.station_ID, s1.Longitude, s1.Latitude, s1.time
            FROM stations s1
            INNER JOIN (
                SELECT station_ID, MAX(time) as max_time
                FROM stations 
                WHERE time BETWEEN ? AND ?
                GROUP BY station_ID
            ) s2 ON s1.station_ID = s2.station_ID AND s1.time = s2.max_time
            """

            df = pd.read_sql_query(query, self.db_conn, params=[start_db, end_db])

            if not df.empty:
                # 确保数值类型
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                df = df.dropna(subset=['Latitude', 'Longitude'])

                self.logger.info(f"找到 {len(df)} 个唯一站点的最新位置")

                # 记录坐标范围
                lon_range = (df['Longitude'].min(), df['Longitude'].max())
                lat_range = (df['Latitude'].min(), df['Latitude'].max())
                self.logger.info(
                    f"站点坐标范围: 经度({lon_range[0]:.2f} to {lon_range[1]:.2f}), "
                    f"纬度({lat_range[0]:.2f} to {lat_range[1]:.2f})"
                )

                return df.to_dict('records')
            else:
                self.logger.warning("在时间范围内没有找到站点记录")
                return []

        except Exception as e:
            self.logger.exception(f"获取站点失败: {str(e)}")
            return []

    def _get_station_date(self, station_id, year):
        """获取站点在指定年份的实际日期"""
        try:
            # 查找该站点在指定年份的任意一个日期
            query = """
            SELECT time FROM stations 
            WHERE station_ID = ? AND time BETWEEN ? AND ?
            LIMIT 1
            """

            start_db = float(f"{year}0101")  # 年份开始
            end_db = float(f"{year}1231")  # 年份结束

            df = pd.read_sql_query(query, self.db_conn, params=[station_id, start_db, end_db])

            if not df.empty:
                # 转换数据库格式的日期 (YYYYMMDD.0) 为 datetime
                db_time = df.iloc[0]['time']
                date_str = f"{int(db_time):08d}"  # 转换为8位数字字符串
                return datetime.strptime(date_str, '%Y%m%d')
            else:
                # 如果该年份没有数据，使用年份中的某一天（但不是固定的1月1日）
                return datetime(year, 6, 15)  # 使用年中日期作为默认

        except Exception as e:
            self.logger.debug(f"获取站点 {station_id} 日期失败: {str(e)}")
            return datetime(year, 6, 15)  # 默认日期

    def _log_final_statistics(self, final_df, success_years, processed_years):
        """记录最终统计信息"""
        try:
            # 基本统计
            total_records = len(final_df)
            unique_stations = final_df['station_id'].nunique()
            years_with_data = final_df['processing_year'].nunique()

            # 地物分类统计
            if 'value' in final_df.columns:
                class_stats = final_df['value'].value_counts()
                top_classes = class_stats.head(5).to_dict()

                self.logger.info("=== 最终处理统计 ===")
                self.logger.info(f"总记录数: {total_records}")
                self.logger.info(f"唯一站点数: {unique_stations}")
                self.logger.info(f"有数据年份数: {years_with_data}")
                self.logger.info(f"处理成功率: {success_years}/{processed_years} 年")
                self.logger.info(f"主要地物类型分布: {top_classes}")

                # 各年数据量统计
                yearly_counts = final_df['processing_year'].value_counts().sort_index()
                self.logger.info("各年数据量统计:")
                for year, count in yearly_counts.items():
                    self.logger.info(f"  {year}年: {count} 条记录")

        except Exception as e:
            self.logger.warning(f"生成统计信息失败: {str(e)}")

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