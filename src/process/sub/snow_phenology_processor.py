import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import rasterio
from rasterio.transform import rowcol

logger = logging.getLogger("SnowPhenologyProcessor")


class SnowPhenologyProcessor:
    """积雪物候数据处理器"""

    def __init__(self, secure_processor=None, station_filter=None):
        """
        初始化积雪物候处理器

        参数:
            secure_processor: 安全处理器实例
            station_filter: 站点过滤器
        """
        from src.process.config import config

        self.logger = logging.getLogger("SnowPhenologyProcessor")
        self.conf = config.snow_phenology
        self.secure_processor = secure_processor
        self.station_filter = station_filter

        # 输出目录
        self.output_dir = Path(config.get_output_dir('snow_phenology'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据库连接
        self.db_conn = self.connect_database()

        self.logger.info("初始化积雪物候处理器完成")

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

    def get_file_paths(self, hydrological_year):
        """根据水文年获取初日和终日文件路径"""
        try:
            base_dir = Path(self.conf['base_data_dir'])

            # 水文年对应的文件年份范围
            file_start_year = hydrological_year
            file_end_year = hydrological_year + 1

            # 获取子目录配置
            sub_dirs = self.conf.get('sub_directories', {})
            start_dir = sub_dirs.get('start', 'SCS')
            end_dir = sub_dirs.get('end', 'SCM')

            # 构建完整目录路径
            start_data_dir = base_dir / start_dir
            end_data_dir = base_dir / end_dir

            # 检查目录是否存在
            if not start_data_dir.exists():
                self.logger.error(f"初日数据目录不存在: {start_data_dir}")
                return {'start': None, 'end': None}

            if not end_data_dir.exists():
                self.logger.error(f"终日数据目录不存在: {end_data_dir}")
                return {'start': None, 'end': None}

            # 构建文件名模式
            file_pattern = self.conf.get('file_pattern', 'NIEER_AVHRR_{dataset_type}_500m_{year}-{next_year}.tif')

            # 构建初日文件名 - 使用 format 方法替换占位符
            start_filename = file_pattern.format(
                dataset_type='SCS',
                year=file_start_year,
                next_year=file_end_year
            )
            start_path = start_data_dir / start_filename

            # 构建终日文件名
            end_filename = file_pattern.format(
                dataset_type='SCM',
                year=file_start_year,
                next_year=file_end_year
            )
            end_path = end_data_dir / end_filename

            self.logger.debug(f"水文年 {hydrological_year} 对应文件: {file_start_year}-{file_end_year}")
            self.logger.debug(f"初日文件路径: {start_path}")
            self.logger.debug(f"终日文件路径: {end_path}")

            # 检查文件是否存在
            start_exists = start_path.exists() if start_path else False
            end_exists = end_path.exists() if end_path else False

            if not start_exists:
                self.logger.warning(f"初日文件不存在: {start_path}")
            if not end_exists:
                self.logger.warning(f"终日文件不存在: {end_path}")

            return {
                'start': start_path if start_exists else None,
                'end': end_path if end_exists else None
            }

        except Exception as e:
            self.logger.exception(f"文件路径生成失败: {str(e)}")
            return {'start': None, 'end': None}

    def _extract_from_file(self, file_path, dataset_type, stations, hydrological_year, target_date):
        """从单个文件中提取数据（支持水文年和水文年DOY）"""
        if not file_path or not file_path.exists():
            self.logger.warning(f"{dataset_type} 文件不存在: {file_path}")
            return pd.DataFrame()

        try:
            with rasterio.open(file_path) as src:
                # 读取数据（16位有符号整型）
                data = src.read(1).astype(np.int32)  # 转换为32位避免溢出

                # 获取地理变换信息
                transform = src.transform

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

                # 提取站点值
                results = []
                for i, (lon, lat) in enumerate(zip(lons, lats)):
                    try:
                        # 将经纬度转换为行列号
                        row, col = rowcol(transform, lon, lat)

                        # 检查行列号是否在有效范围内
                        if 0 <= row < src.height and 0 <= col < src.width:
                            day_of_year = int(data[row, col])

                            # 检查填充值和有效范围
                            fill_value = self.conf.get('fill_value', -32768)
                            valid_min = self.conf.get('valid_min', 1)
                            valid_max = self.conf.get('valid_max', 366)

                            if day_of_year != fill_value and valid_min <= day_of_year <= valid_max:
                                # 计算水文年DOY
                                hydrological_doy = self._calculate_hydrological_doy(day_of_year, dataset_type)

                                # 转换为实际日期（考虑水文年）
                                try:
                                    # 水文年开始日期（通常是9月1日）
                                    hydro_config = self.conf.get('hydrological_year', {})
                                    start_month = hydro_config.get('start_month', 9)
                                    start_day = hydro_config.get('start_day', 1)

                                    hydro_start_date = datetime(hydrological_year, start_month, start_day)
                                    snow_date = hydro_start_date + timedelta(days=day_of_year - 1)
                                    date_str = snow_date.strftime('%Y-%m-%d')

                                    # 检查日期是否在目标日期附近（用于验证）
                                    date_diff = abs((snow_date - target_date).days)
                                    if date_diff > 30:  # 如果差异超过30天，记录警告
                                        self.logger.warning(
                                            f"站点 {station_ids[i]} 提取日期 {date_str} "
                                            f"与目标日期 {target_date.strftime('%Y-%m-%d')} 差异较大: {date_diff} 天"
                                        )

                                except Exception as e:
                                    self.logger.warning(f"日期转换失败: {str(e)}")
                                    date_str = f"{hydrological_year}-{day_of_year:03d}"

                                results.append({
                                    'station_id': station_ids[i],
                                    'hydrological_year': hydrological_year,
                                    'day_of_year': day_of_year,
                                    'hydrological_doy': hydrological_doy,  # 新增水文年DOY
                                    'date': date_str,
                                    'target_date': target_date.strftime('%Y-%m-%d'),
                                    'dataset_type': dataset_type,
                                    'file_used': file_path.name,
                                    'file_hydrological_year': f"{hydrological_year}-{hydrological_year + 1}"
                                })
                        else:
                            self.logger.debug(f"站点 {station_ids[i]} 坐标超出图像范围")

                    except Exception as e:
                        self.logger.warning(f"提取站点 {station_ids[i]} 值失败: {str(e)}")

                return pd.DataFrame(results)

        except Exception as e:
            self.logger.exception(f"提取 {dataset_type} 数据错误: {str(e)}")
            return pd.DataFrame()

    def extract_values(self, date):
        """提取指定日期的积雪物候数据"""
        try:
            # 获取对应的水文年
            hydrological_year = self._get_hydrological_year(date)
            self.logger.debug(f"日期 {date} 属于水文年: {hydrological_year}")

            # 获取该日期有效的站点
            stations = self.get_stations_for_date(date)
            if not stations:
                self.logger.warning(f"未找到 {date} 的有效站点")
                return pd.DataFrame()

            # 获取文件路径
            file_paths = self.get_file_paths(hydrological_year)

            all_results = []

            # 处理初日数据
            start_df = self._extract_from_file(file_paths['start'], 'start', stations, hydrological_year, date)
            if not start_df.empty:
                all_results.append(start_df)
                self.logger.info(f"水文年 {hydrological_year} 初日: 提取 {len(start_df)} 条记录")

            # 处理终日数据
            end_df = self._extract_from_file(file_paths['end'], 'end', stations, hydrological_year, date)
            if not end_df.empty:
                all_results.append(end_df)
                self.logger.info(f"水文年 {hydrological_year} 终日: 提取 {len(end_df)} 条记录")

            if all_results:
                return pd.concat(all_results, ignore_index=True)
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.exception(f"提取 {date} 数据错误: {str(e)}")
            return pd.DataFrame()

    def save_results(self, df, start_date, end_date):
        """保存结果"""
        try:
            # 只使用日期部分，不包含时间
            start_str = start_date.strftime('%Y-%m-%d')  # 例如: 2012-09-01
            end_str = end_date.strftime('%Y-%m-%d')  # 例如: 2018-08-31

            base_filename = f"snow_phenology_{start_str}_{end_str}"

            # 总是保存 CSV 格式（用于查看）
            csv_path = self.output_dir / f"{base_filename}.csv"

            df.to_csv(csv_path, index=False)
            self.logger.info(f"结果保存为 CSV (用于查看): {csv_path}")

            # 根据配置保存其他格式
            output_format = self.conf.get('output', {}).get('primary_format', 'parquet').lower()

            if output_format == 'parquet':
                try:
                    parquet_path = self.output_dir / f"{base_filename}.parquet"
                    df.to_parquet(parquet_path, index=False)
                    self.logger.info(f"结果保存为 Parquet (用于处理): {parquet_path}")
                    return str(parquet_path)
                except ImportError:
                    self.logger.warning("Parquet 支持不可用，仅保存 CSV 格式")
                    return str(csv_path)
                except Exception as e:
                    self.logger.error(f"保存Parquet文件失败: {str(e)}")
                    return str(csv_path)

            else:
                self.logger.warning(f"未知的输出格式: {output_format}，仅保存 CSV 格式")
                return str(csv_path)

        except Exception as e:
            self.logger.exception(f"保存结果失败: {str(e)}")
            return None

    def process_range(self, start_date, end_date):
        """处理日期范围（按水文年处理）"""
        self.logger.info(f"开始处理 {start_date} 至 {end_date} 的积雪物候数据（水文年）")

        # 验证数据目录结构
        if not self.validate_data_directories():
            self.logger.error("数据目录验证失败，无法继续处理")
            return None

        # 转换为日期范围
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # 生成日期序列（按天处理）
        date_range = pd.date_range(start_date, end_date, freq='D')

        all_results = []
        processed_dates = 0
        success_dates = 0

        for current_date in date_range:
            processed_dates += 1
            try:
                # 提取数据
                daily_df = self.extract_values(current_date)

                if not daily_df.empty:
                    # 数据后处理
                    processed_df = self.postprocess_data(daily_df, current_date)
                    if not processed_df.empty:
                        all_results.append(processed_df)
                        success_dates += 1

                        if success_dates % 30 == 0:  # 每30天记录一次进度
                            self.logger.info(f"处理进度: {success_dates}/{processed_dates} 天")
                else:
                    if processed_dates % 100 == 0:  # 每100天记录一次无数据情况
                        self.logger.debug(f"日期 {current_date}: 未提取到数据")

            except Exception as e:
                self.logger.error(f"{current_date} 处理失败: {str(e)}")
                continue

        # 合并结果
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            output_path = self.save_results(final_df, start_date, end_date)

            # 生成数据摘要
            self._generate_phenology_summary(final_df, Path(output_path))

            self.logger.info(
                f"处理完成! 成功处理 {success_dates}/{processed_dates} 天, "
                f"总计 {len(final_df)} 条记录, 结果保存至: {output_path}"
            )
            return output_path
        else:
            self.logger.warning(f"未生成有效结果, 处理了 {processed_dates} 天")
            return None

    def validate_data_directories(self):
        """验证数据目录结构"""
        try:
            base_dir = Path(self.conf['base_data_dir'])

            if not base_dir.exists():
                self.logger.error(f"基础数据目录不存在: {base_dir}")
                return False

            # 检查子目录
            sub_dirs = self.conf.get('sub_directories', {})
            missing_dirs = []

            for dir_type, dir_name in sub_dirs.items():
                dir_path = base_dir / dir_name
                if not dir_path.exists():
                    missing_dirs.append(f"{dir_type}({dir_name})")
                    self.logger.error(f"{dir_type} 目录不存在: {dir_path}")

            if missing_dirs:
                self.logger.error(f"缺少以下数据目录: {', '.join(missing_dirs)}")
                return False

            # 检查目录是否为空
            empty_dirs = []
            for dir_type, dir_name in sub_dirs.items():
                dir_path = base_dir / dir_name
                if not any(dir_path.iterdir()):
                    empty_dirs.append(f"{dir_type}({dir_name})")
                    self.logger.warning(f"{dir_type} 目录为空: {dir_path}")

            if empty_dirs:
                self.logger.warning(f"以下目录为空: {', '.join(empty_dirs)}")

            return True

        except Exception as e:
            self.logger.error(f"验证数据目录失败: {str(e)}")
            return False

    def validate_hydrological_doy_calculation(self):
        """验证水文年DOY计算是否正确"""
        try:
            # 测试用例：自然年DOY和水文年DOY的对应关系
            test_cases = [
                # (自然年DOY, 数据类型, 期望水文年DOY)
                (244, 'start', 1),  # 9月1日初日 -> 水文年DOY 1
                (244, 'end', 1),  # 9月1日终日 -> 水文年DOY 1
                (365, 'start', 122),  # 12月31日初日 -> 水文年DOY 122
                (365, 'end', 122),  # 12月31日终日 -> 水文年DOY 122
                (1, 'start', 123),  # 1月1日初日 -> 水文年DOY 123
                (1, 'end', 123),  # 1月1日终日 -> 水文年DOY 123
                (243, 'start', 366),  # 8月31日初日 -> 水文年DOY 366
                (243, 'end', 366),  # 8月31日终日 -> 水文年DOY 366
            ]

            self.logger.info("水文年DOY计算验证:")
            for natural_doy, dataset_type, expected_hydro_doy in test_cases:
                actual_hydro_doy = self._calculate_hydrological_doy(natural_doy, dataset_type)
                status = "✅" if actual_hydro_doy == expected_hydro_doy else "❌"
                self.logger.info(f"  {status} 自然年DOY {natural_doy} ({dataset_type}) -> "
                                 f"期望: {expected_hydro_doy}, 实际: {actual_hydro_doy}")

            return True

        except Exception as e:
            self.logger.error(f"水文年DOY验证失败: {str(e)}")
            return False

    def postprocess_data(self, df, year):
        """数据后处理 - 包含水文年DOY验证"""
        if df.empty:
            return df

        # 创建副本
        result_df = df.copy()

        # 过滤无效值
        fill_value = self.conf.get('fill_value', -32768)
        valid_min = self.conf.get('valid_min', 1)
        valid_max = self.conf.get('valid_max', 366)

        # 使用 .loc 进行条件过滤
        mask = (result_df['day_of_year'] >= valid_min) & \
               (result_df['day_of_year'] <= valid_max) & \
               (result_df['day_of_year'] != fill_value)
        result_df = result_df.loc[mask].copy()

        # 验证水文年DOY的范围
        if 'hydrological_doy' in result_df.columns:
            hydro_doy_valid = (result_df['hydrological_doy'] >= 1) & (result_df['hydrological_doy'] <= 366)
            invalid_hydro_doy = (~hydro_doy_valid).sum()
            if invalid_hydro_doy > 0:
                self.logger.warning(f"发现 {invalid_hydro_doy} 条记录的水文年DOY超出有效范围")
                result_df = result_df[hydro_doy_valid].copy()

        # 添加处理时间
        processing_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_df.loc[:, 'processing_time'] = processing_time

        # 统计水文年DOY信息
        if 'hydrological_doy' in result_df.columns:
            hydro_doy_stats = result_df['hydrological_doy'].describe()
            self.logger.info(f"水文年DOY统计: 均值={hydro_doy_stats['mean']:.1f}, "
                             f"范围={hydro_doy_stats['min']:.0f}-{hydro_doy_stats['max']:.0f}")

        self.logger.debug(f"后处理完成: {len(result_df)} 条记录")
        return result_df

    def _get_hydrological_year(self, date):
        """根据日期获取对应的水文年"""
        try:
            # 获取水文年配置
            hydro_config = self.conf.get('hydrological_year', {})
            start_month = hydro_config.get('start_month', 9)
            start_day = hydro_config.get('start_day', 1)

            # 如果日期在水文年开始之后，属于当前水文年
            # 否则属于上一个水文年
            if date.month > start_month or (date.month == start_month and date.day >= start_day):
                hydro_year = date.year
            else:
                hydro_year = date.year - 1

            return hydro_year

        except Exception as e:
            self.logger.error(f"计算水文年失败: {str(e)}")
            return date.year  # 默认返回日历年

    def _calculate_hydrological_doy(self, natural_doy, dataset_type):
        """计算水文年DOY

        水文年从9月1日开始：
        - 初日数据：直接使用自然年DOY（因为初日通常在水文年内）
        - 终日数据：需要调整（因为终日可能跨越水文年）

        Args:
            natural_doy: 自然年DOY
            dataset_type: 数据类型（'start' 或 'end'）

        Returns:
            int: 水文年DOY
        """
        try:
            # 获取水文年配置
            hydro_config = self.conf.get('hydrological_year', {})
            start_month = hydro_config.get('start_month', 9)
            start_day = hydro_config.get('start_day', 1)

            # 计算水文年开始的DOY（自然年）
            hydro_start_doy = datetime(2000, start_month, start_day).timetuple().tm_yday

            if dataset_type == 'start':
                # 初日数据：通常在水文年内，直接使用
                # 9月1日之后：DOY - hydro_start_doy + 1
                # 9月1日之前：属于下一个水文年，但这种情况较少
                if natural_doy >= hydro_start_doy:
                    hydrological_doy = natural_doy - hydro_start_doy + 1
                else:
                    # 如果初日在9月1日之前，可能是下一个水文年的开始
                    hydrological_doy = natural_doy + (365 - hydro_start_doy) + 1
            else:
                # 终日数据：需要特殊处理
                # 终日可能在当前水文年或下一个水文年
                if natural_doy >= hydro_start_doy:
                    # 当前水文年的终日
                    hydrological_doy = natural_doy - hydro_start_doy + 1
                else:
                    # 下一个水文年的终日（年初的终日）
                    hydrological_doy = natural_doy + (365 - hydro_start_doy) + 1

            # 确保DOY在有效范围内
            if hydrological_doy < 1:
                return 1
            elif hydrological_doy > 366:
                return 366
            else:
                return int(hydrological_doy)

        except Exception as e:
            self.logger.warning(f"计算水文年DOY失败: {str(e)}")
            return natural_doy  # 返回原始DOY作为备用

    def _generate_phenology_summary(self, df, output_path):
        """生成积雪物候数据摘要"""
        try:
            summary_path = output_path.with_suffix('.summary.txt')

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=== 积雪物候数据摘要 ===\n")
                f.write(f"记录数: {len(df)}\n")
                f.write(f"时间范围: {df['year'].min()} 至 {df['year'].max()}\n")
                f.write(f"站点数: {df['station_id'].nunique()}\n")

                # 数据集类型统计
                f.write("\n=== 数据集分布 ===\n")
                type_stats = df['dataset_type'].value_counts()
                f.write(type_stats.to_string())

                # 年度统计
                f.write("\n\n=== 年度统计 ===\n")
                year_stats = df.groupby(['year', 'dataset_type']).size().unstack(fill_value=0)
                f.write(year_stats.to_string())

                # 站点统计
                f.write("\n\n=== 站点统计 ===\n")
                station_stats = df.groupby('station_id').agg({
                    'day_of_year': ['count', 'min', 'max', 'mean']
                }).round(1)
                f.write(station_stats.to_string())

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