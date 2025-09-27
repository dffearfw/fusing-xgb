"""
Create on 2025/9/7

@auther:Thinkpad
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

logger = logging.getLogger("TerrainFeaturesProcessor")


class TerrainFeaturesProcessor:
    """地理特征数据处理器"""

    def __init__(self, secure_processor=None, station_filter=None):
        """
        初始化地理特征处理器

        参数:
            secure_processor: 安全处理器实例
            station_filter: 站点过滤器
        """
        from src.process.config import config

        self.logger = logging.getLogger("TerrainFeaturesProcessor")
        self.conf = config.terrain_features
        self.secure_processor = secure_processor
        self.station_filter = station_filter

        # 初始化db_conn为None，避免后续错误
        self.db_conn = None

        # 输出目录
        self.output_dir = Path(config.get_output_dir('terrain_features'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据库连接（增加错误处理）
        try:
            self.db_conn = self.connect_database()
            self.logger.info("数据库连接成功")
        except Exception as e:
            self.logger.error(f"数据库连接初始化失败: {str(e)}")
            # 不raise异常，允许继续运行但记录错误

        # 投影转换器（如果需要转换到WGS84）
        self.proj_transformer = None

        self.logger.info("初始化地理特征处理器完成")

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

    def get_stations(self):
        """获取所有站点（地理特征数据不依赖时间）"""
        try:
            # 基础查询
            query = "SELECT station_ID, Longitude, Latitude FROM stations"
            params = []

            # 添加站点过滤器
            if self.station_filter:
                if self.station_filter['type'] == 'bbox':
                    minx, miny, maxx, maxy = self.station_filter['value']
                    query += " WHERE Longitude BETWEEN ? AND ? AND Latitude BETWEEN ? AND ?"
                    params.extend([minx, maxx, miny, maxy])
                elif self.station_filter['type'] == 'ids':
                    placeholders = ','.join(['?'] * len(self.station_filter['value']))
                    query += f" WHERE station_ID IN ({placeholders})"
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
                self.logger.warning("未找到有效站点记录")
                return []

        except Exception as e:
            self.logger.exception(f"获取站点失败: {str(e)}")
            return []

    def _find_arcgrid_directory(self, feature_type):
        """查找ArcGIS Grid格式的数据目录"""
        try:
            base_dir = Path(self.conf['base_data_dir'])
            feature_name = self.conf['feature_types'].get(feature_type)

            if not feature_name:
                self.logger.error(f"未知的特征类型: {feature_type}")
                return None

            # 查找特征目录
            feature_dir = base_dir / feature_name

            if not feature_dir.exists():
                self.logger.warning(f"特征目录不存在: {feature_dir}")
                return None

            # 检查ArcGrid格式文件
            arcgrid_files = list(feature_dir.glob("*"))
            if not arcgrid_files:
                self.logger.warning(f"特征目录为空: {feature_dir}")
                return None

            # 检查是否包含ArcGrid格式文件
            has_arcgrid = any(file.suffix in ['.adf', '.dblbnd', '.hdr', '.prj', '.sta'] for file in arcgrid_files)

            if has_arcgrid:
                return feature_dir
            else:
                self.logger.warning(f"目录不包含ArcGrid格式文件: {feature_dir}")
                return None

        except Exception as e:
            self.logger.error(f"查找特征目录失败: {str(e)}")
            return None

    def _find_arcgrid_data_file(self, feature_dir):
        """在ArcGrid目录中找到数据文件"""
        try:
            # 查找w001001.adf文件（主要数据文件）
            data_file = feature_dir / "w001001.adf"
            if data_file.exists():
                return data_file

            # 查找其他可能的数据文件
            for file_pattern in ['w001001', 'w001001x', '*.adf']:
                matching_files = list(feature_dir.glob(file_pattern))
                if matching_files:
                    return matching_files[0]

            self.logger.warning(f"未找到数据文件 in: {feature_dir}")
            return None

        except Exception as e:
            self.logger.error(f"查找数据文件失败: {str(e)}")
            return None

    def extract_feature_values(self, feature_type, stations):
        """提取特定地理特征的值（内存优化版）"""
        try:
            # 查找特征目录
            feature_dir = self._find_arcgrid_directory(feature_type)
            if not feature_dir:
                return pd.DataFrame()

            # 查找数据文件
            data_file = self._find_arcgrid_data_file(feature_dir)
            if not data_file:
                return pd.DataFrame()

            self.logger.info(f"处理特征 {feature_type}: {data_file}")

            # 打开ArcGrid文件
            with rasterio.open(data_file) as src:
                # 获取数据投影信息
                src_crs = src.crs
                self.logger.info(f"数据坐标系: {src_crs}")
                self.logger.info(f"数据尺寸: {src.width} x {src.height}, 分辨率: {src.res}")

                # 创建投影转换器
                try:
                    from pyproj import Transformer
                    wgs84_crs = "EPSG:4326"
                    transformer = Transformer.from_crs(wgs84_crs, src_crs, always_xy=True)
                except Exception as e:
                    self.logger.error(f"创建投影转换器失败: {str(e)}")
                    return pd.DataFrame()

                # 提取站点值（使用逐点读取，避免加载整个数组）
                results = []
                processed_count = 0
                success_count = 0

                for station in stations:
                    try:
                        lon = float(station.get('Longitude', station.get('longitude')))
                        lat = float(station.get('Latitude', station.get('latitude')))
                        station_id = station.get('station_ID', station.get('station_id'))

                        # 将WGS84坐标转换为数据坐标系
                        try:
                            x, y = transformer.transform(lon, lat)
                        except Exception as e:
                            self.logger.debug(f"坐标转换失败 站点 {station_id}: {str(e)}")
                            continue

                        # 将平面坐标转换为行列号
                        row, col = rowcol(src.transform, x, y)
                        row, col = int(row), int(col)

                        # 检查行列号是否在有效范围内
                        if 0 <= row < src.height and 0 <= col < src.width:
                            # 使用窗口读取单个像素值（内存高效）
                            try:
                                # 创建1x1的窗口
                                window = rasterio.windows.Window(col, row, 1, 1)

                                # 只读取单个像素
                                value_array = src.read(1, window=window, boundless=True,
                                                       fill_value=self.conf.get('fill_value', -9999.0))
                                value = float(value_array[0, 0])

                                # 检查填充值
                                fill_value = self.conf.get('fill_value', -9999.0)

                                if value != fill_value and not np.isnan(value):
                                    results.append({
                                        'station_id': station_id,
                                        'feature_type': feature_type,
                                        'value': value,
                                        'longitude': lon,
                                        'latitude': lat,
                                        'projected_x': x,
                                        'projected_y': y
                                    })
                                    success_count += 1

                            except Exception as e:
                                self.logger.debug(f"读取像素失败 站点 {station_id}: {str(e)}")
                        else:
                            self.logger.debug(f"站点 {station_id} 坐标超出图像范围: 行{row}, 列{col}")

                        processed_count += 1

                        # 每处理1000个站点记录一次进度
                        if processed_count % 1000 == 0:
                            self.logger.info(f"处理进度: {processed_count}/{len(stations)} 站点, 成功: {success_count}")

                    except Exception as e:
                        self.logger.debug(f"处理站点 {station.get('station_ID')} 失败: {str(e)}")
                        continue

                self.logger.info(f"特征 {feature_type}: 处理 {processed_count} 站点, 成功提取 {success_count} 条记录")
                return pd.DataFrame(results)

        except Exception as e:
            self.logger.exception(f"提取特征 {feature_type} 错误: {str(e)}")
            return pd.DataFrame()

    def extract_all_features(self, stations):
        """提取所有地理特征的值（修复版本）"""
        all_results = []

        # 获取要处理的特征类型
        feature_types = list(self.conf.get('feature_types', {}).keys())
        process_all = self.conf.get('extract_all_features', True)

        if not process_all:
            # 只处理主要特征
            main_features = ['elevation', 'slope', 'aspect']
            feature_types = [ft for ft in feature_types if ft in main_features]

        self.logger.info(f"开始提取 {len(feature_types)} 个地理特征: {feature_types}")

        for i, feature_type in enumerate(feature_types):
            try:
                self.logger.info(f"正在处理特征 ({i + 1}/{len(feature_types)}): {feature_type}")

                # 提取单个特征
                feature_df = self.extract_feature_values_batch(feature_type, stations, batch_size=500)

                if not feature_df.empty:
                    # 立即检查数据
                    self.logger.info(
                        f"特征 {feature_type}: 提取 {len(feature_df)} 条记录, 唯一站点: {feature_df['station_id'].nunique()}")

                    # 检查特征类型是否正确
                    unique_features = feature_df['feature_type'].unique()
                    self.logger.debug(f"DataFrame中的特征类型: {unique_features}")

                    # 确保feature_type列的值正确
                    if len(unique_features) == 1 and unique_features[0] == feature_type:
                        all_results.append(feature_df)
                        self.logger.info(f"✓ 特征 {feature_type} 添加成功")
                    else:
                        self.logger.warning(f"特征类型不匹配! 期望: {feature_type}, 实际: {unique_features}")
                        # 强制修正特征类型
                        feature_df['feature_type'] = feature_type
                        all_results.append(feature_df)
                else:
                    self.logger.warning(f"特征 {feature_type}: 未提取到数据")

            except Exception as e:
                self.logger.error(f"提取特征 {feature_type} 失败: {str(e)}")
                continue

        # 合并所有结果
        if all_results:
            try:
                # 检查每个DataFrame的特征类型
                for i, df in enumerate(all_results):
                    self.logger.debug(f"结果 {i}: 特征类型 = {df['feature_type'].unique()}, 记录数 = {len(df)}")

                final_df = pd.concat(all_results, ignore_index=True)

                # 验证最终结果
                unique_features_final = final_df['feature_type'].unique()
                self.logger.info(f"最终合并结果: 总记录数 = {len(final_df)}, 特征类型 = {unique_features_final}, 每种特征记录数:")

                feature_counts = final_df['feature_type'].value_counts()
                for feature, count in feature_counts.items():
                    self.logger.info(f"  {feature}: {count} 条记录")

                return final_df

            except Exception as e:
                self.logger.error(f"合并结果失败: {str(e)}")
                return pd.DataFrame()
        else:
            self.logger.warning("未生成有效结果")
            return pd.DataFrame()

    def extract_feature_values_batch(self, feature_type, stations, batch_size=1000):
        """批量提取特征值（进一步优化内存）"""
        try:
            # 查找特征目录和数据文件
            feature_dir = self._find_arcgrid_directory(feature_type)
            if not feature_dir:
                return pd.DataFrame()

            data_file = self._find_arcgrid_data_file(feature_dir)
            if not data_file:
                return pd.DataFrame()

            self.logger.info(f"批量处理特征 {feature_type}: {data_file}")

            results = []
            total_batches = (len(stations) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(stations))
                batch_stations = stations[start_idx:end_idx]

                batch_results = self._process_station_batch(feature_type, data_file, batch_stations)
                results.extend(batch_results)

                self.logger.info(f"批次 {batch_idx + 1}/{total_batches} 完成: {len(batch_results)} 条记录")

                # 手动垃圾回收
                import gc
                gc.collect()

            return pd.DataFrame(results)

        except Exception as e:
            self.logger.exception(f"批量提取特征 {feature_type} 错误: {str(e)}")
            return pd.DataFrame()

    def _process_station_batch(self, feature_type, data_file, stations):
        """处理单个站点批次"""
        results = []

        with rasterio.open(data_file) as src:
            # 创建投影转换器
            try:
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            except Exception as e:
                self.logger.error(f"创建投影转换器失败: {str(e)}")
                return results

            for station in stations:
                try:
                    lon = float(station.get('Longitude'))
                    lat = float(station.get('Latitude'))
                    station_id = station.get('station_ID')

                    # 坐标转换
                    x, y = transformer.transform(lon, lat)
                    row, col = rowcol(src.transform, x, y)
                    row, col = int(row), int(col)

                    if 0 <= row < src.height and 0 <= col < src.width:
                        # 读取单个像素
                        window = rasterio.windows.Window(col, row, 1, 1)
                        value_array = src.read(1, window=window, boundless=True,
                                               fill_value=self.conf.get('fill_value', -9999.0))
                        value = float(value_array[0, 0])

                        fill_value = self.conf.get('fill_value', -9999.0)
                        if value != fill_value and not np.isnan(value):
                            results.append({
                                'station_id': station_id,
                                'feature_type': feature_type,
                                'value': value,
                                'longitude': lon,
                                'latitude': lat
                            })

                except Exception as e:
                    continue

        return results

    def postprocess_data(self, df):
        """数据后处理"""
        if df.empty:
            return df

        # 创建副本
        result_df = df.copy()

        # 过滤填充值
        fill_value = self.conf.get('fill_value', -9999.0)
        mask = result_df['value'] != fill_value
        result_df = result_df.loc[mask].copy()

        # 添加处理时间
        processing_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_df.loc[:, 'processing_time'] = processing_time

        self.logger.debug(f"后处理完成: {len(result_df)} 条记录")
        return result_df

    def save_results(self, df):
        """保存结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"terrain_features_{timestamp}"

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

            else:
                self.logger.warning(f"未知的输出格式: {output_format}，仅保存 CSV 格式")
                return str(csv_path)

        except Exception as e:
            self.logger.exception(f"保存结果失败: {str(e)}")
            return None

    def _save_temp_results(self, df, feature_type):
        """保存临时结果以释放内存"""
        try:
            temp_dir = self.output_dir / "temp"
            temp_dir.mkdir(exist_ok=True)

            temp_file = temp_dir / f"temp_{feature_type}_{datetime.now().strftime('%H%M%S')}.parquet"
            df.to_parquet(temp_file, index=False)

            self.logger.debug(f"保存临时结果: {temp_file}")

        except Exception as e:
            self.logger.warning(f"保存临时结果失败: {str(e)}")

    def process(self):
        """处理地理特征数据"""
        self.logger.info("开始处理地理特征数据")

        # 获取所有站点
        stations = self.get_stations()
        if not stations:
            self.logger.error("没有有效站点，无法处理")
            return None

        self.logger.info(f"找到 {len(stations)} 个站点，开始提取地理特征")

        # 提取所有特征
        result_df = self.extract_all_features(stations)

        if not result_df.empty:
            # 数据后处理
            processed_df = self.postprocess_data(result_df)

            # 保存结果
            output_path = self.save_results(processed_df)

            # 生成数据摘要
            self._generate_terrain_summary(processed_df, Path(output_path))

            self.logger.info(
                f"处理完成! 总计 {len(processed_df)} 条记录, 结果保存至: {output_path}"
            )
            return output_path
        else:
            self.logger.warning("未生成有效结果")
            return None

    def _generate_terrain_summary(self, df, output_path):
        """生成地理特征数据摘要"""
        try:
            summary_path = output_path.with_suffix('.summary.txt')

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=== 地理特征数据摘要 ===\n")
                f.write(f"记录数: {len(df)}\n")
                f.write(f"站点数: {df['station_id'].nunique()}\n")
                f.write(f"特征类型数: {df['feature_type'].nunique()}\n")

                # 特征类型统计
                f.write("\n=== 特征类型分布 ===\n")
                feature_stats = df['feature_type'].value_counts()
                f.write(feature_stats.to_string())

                # 特征值统计
                f.write("\n\n=== 特征值统计 ===\n")
                value_stats = df.groupby('feature_type')['value'].agg(['count', 'min', 'max', 'mean', 'std']).round(3)
                f.write(value_stats.to_string())

                # 前20条记录
                f.write("\n\n=== 前20条记录 ===\n")
                f.write(df.head(20).to_string())

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
        """析构函数，确保资源被释放"""
        self.close()