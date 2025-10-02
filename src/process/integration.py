import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import traceback

logger = logging.getLogger("DataIntegrator")


class DataIntegrator:
    def __init__(self, output_dir, secure_processor=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("DataIntegrator")
        self.source_data = {}
        self.db_conn = None
        self.db_fields_config = {
            'swe（mm）': 'swe',
            'Altitude(m)': 'altitude',

            # 'snowDepth(mm)': 'snow_depth',
            # 'snowDensity(g/cm3)': 'snow_density'
        }

    def connect_database(self):
        """连接站点数据库"""
        try:
            from src.process.config import config
            db_path = config.get_station_db_path()
            self.db_conn = sqlite3.connect(db_path)
            self.logger.info("数据库连接成功")
            return True
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")
            return False

    def add_source(self, name, file_path):
        """添加数据源 - 完整版本"""
        try:
            path = Path(file_path)

            if not path.exists():
                self.logger.error(f"文件不存在: {path}")
                return False

            # 读取文件
            if path.suffix.lower() == '.parquet':
                df = pd.read_parquet(path)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                try:
                    df = pd.read_excel(path, engine='openpyxl')
                except Exception as e:
                    self.logger.warning(f"openpyxl引擎失败: {str(e)}，尝试xlrd")
                    try:
                        df = pd.read_excel(path, engine='xlrd')
                    except Exception as e:
                        self.logger.error(f"所有Excel引擎都失败: {str(e)}")
                        return False
            else:
                try:
                    df = pd.read_csv(path)
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(path, encoding='latin1')
                    except:
                        try:
                            df = pd.read_csv(path, encoding='gbk')
                        except:
                            self.logger.error("CSV文件读取失败")
                            return False

            # 标准化日期格式
            df = self._standardize_date_format(df, name)

            # 标准化列名
            df = self._standardize_column_names(df, name)

            # 数据验证
            validation_result = self._validate_dataframe(df, name)
            if not validation_result['valid']:
                self.logger.warning(f"数据源 {name} 验证警告: {validation_result['message']}")

            # 记录坐标信息
            if self._has_coordinate_info(df):
                coord_columns = self._get_coordinate_columns(df)
                stations_with_coords = df['station_id'].nunique()
                self.logger.info(f"数据源 {name} 包含坐标信息: {coord_columns}, 涉及 {stations_with_coords} 个站点")

            self.source_data[name] = df
            self.logger.info(f"添加 {name}: {len(df)} 行, 列: {list(df.columns)}")
            return True

        except Exception as e:
            self.logger.error(f"添加 {name} 失败: {e}")
            return False

    def _standardize_date_format(self, df, source_name):
        """统一日期格式为字符串 YYYY-MM-DD"""
        try:
            if 'date' not in df.columns:
                return df

            df_copy = df.copy()

            # 转换为统一的字符串格式
            if 'datetime' in str(df_copy['date'].dtype):
                df_copy['date'] = df_copy['date'].dt.strftime('%Y-%m-%d')
            else:
                try:
                    df_copy['date'] = pd.to_datetime(df_copy['date']).dt.strftime('%Y-%m-%d')
                except Exception as e:
                    try:
                        df_copy['date'] = df_copy['date'].str.replace('/', '-')
                        df_copy['date'] = pd.to_datetime(df_copy['date']).dt.strftime('%Y-%m-%d')
                    except:
                        self.logger.warning(f"{source_name}: 无法标准化日期格式")

            return df_copy

        except Exception as e:
            self.logger.error(f"标准化 {source_name} 日期格式失败: {str(e)}")
            return df

    def _standardize_column_names(self, df, source_name):
        """标准化列名"""
        try:
            df_copy = df.copy()
            column_mapping = {}

            # 站点ID标准化
            if 'station_ID' in df_copy.columns and 'station_id' not in df_copy.columns:
                column_mapping['station_ID'] = 'station_id'
            if 'Station_ID' in df_copy.columns and 'station_id' not in df_copy.columns:
                column_mapping['Station_ID'] = 'station_id'

            # 坐标标准化
            coord_mapping = {
                'Longitude': 'longitude', 'Lon': 'longitude',
                'Latitude': 'latitude', 'Lat': 'latitude'
            }

            for old_name, new_name in coord_mapping.items():
                if old_name in df_copy.columns and new_name not in df_copy.columns:
                    column_mapping[old_name] = new_name

            if column_mapping:
                df_copy = df_copy.rename(columns=column_mapping)
                self.logger.debug(f"{source_name}: 重命名列 {list(column_mapping.keys())}")

            return df_copy

        except Exception as e:
            self.logger.error(f"标准化 {source_name} 列名失败: {str(e)}")
            return df

    def _validate_dataframe(self, df, source_name):
        """验证DataFrame结构"""
        validation = {'valid': True, 'message': ''}

        if 'station_id' not in df.columns:
            validation['valid'] = False
            validation['message'] = "缺少station_id列"
            return validation

        numeric_cols = df.select_dtypes(include=['number']).columns
        value_cols = [col for col in numeric_cols if col not in ['station_id', 'date', 'year', 'month', 'day']]

        if not value_cols:
            validation['valid'] = False
            validation['message'] = "没有找到数值列"
            return validation

        station_count = df['station_id'].nunique()
        if station_count == 0:
            validation['valid'] = False
            validation['message'] = "没有有效的站点数据"
            return validation

        if 'date' in df.columns:
            date_count = df['date'].nunique()
            validation['message'] = f"动态数据源，{station_count}个站点，{date_count}个时间点"
        else:
            validation['message'] = f"静态数据源，{station_count}个站点"

        return validation

    def _has_coordinate_info(self, df):
        """检查数据框是否包含坐标信息"""
        coordinate_columns = ['longitude', 'latitude', 'Longitude', 'Latitude', 'lon', 'lat', 'Lon', 'Lat']
        return any(col in df.columns for col in coordinate_columns)

    def _get_coordinate_columns(self, df):
        """获取数据框中的坐标列名"""
        coordinate_mapping = {
            'longitude': ['longitude', 'Longitude', 'lon', 'Lon'],
            'latitude': ['latitude', 'Latitude', 'lat', 'Lat']
        }

        found_columns = []
        for standard_name, possible_names in coordinate_mapping.items():
            for name in possible_names:
                if name in df.columns:
                    found_columns.append(name)
                    break

        return found_columns

    def get_database_data(self, station_ids, start_date, end_date):
        """统一获取数据库中配置的所有字段"""
        try:
            if not self.db_conn:
                if not self.connect_database():
                    return pd.DataFrame()

            if not self.db_fields_config:
                self.logger.warning("没有配置数据库字段")
                return pd.DataFrame()

            # 构建查询
            field_selects = []
            for db_field in self.db_fields_config.keys():
                field_selects.append(f'"{db_field}"')

            query = f"""
            SELECT station_ID, time, {', '.join(field_selects)}
            FROM stations 
            WHERE time BETWEEN ? AND ?
            """

            # 转换日期格式
            start_db = int(start_date.strftime('%Y%m%d'))
            end_db = int(end_date.strftime('%Y%m%d'))

            df = pd.read_sql_query(query, self.db_conn, params=[start_db, end_db])

            if not df.empty:
                # 处理数据
                df['date'] = df['time'].apply(
                    lambda x: datetime.strptime(str(int(x)), '%Y%m%d').strftime('%Y-%m-%d')
                )
                df['station_id'] = df['station_ID'].astype(str)

                # 重命名列
                result_df = df[['station_id', 'date'] + list(self.db_fields_config.keys())]
                result_df = result_df.rename(columns=self.db_fields_config)

                # 过滤站点
                if station_ids:
                    result_df = result_df[result_df['station_id'].isin(station_ids)]

                self.logger.info(f"✅ 从数据库获取 {len(self.db_fields_config)} 个字段: {list(self.db_fields_config.values())}")

                # 统计每个字段的有效数据量
                for field in self.db_fields_config.values():
                    if field in result_df.columns:
                        valid_count = result_df[field].notna().sum()
                        self.logger.info(f"  {field}: {valid_count} 条有效记录")

                return result_df
            else:
                self.logger.warning("数据库中未找到指定时间范围内的数据")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取数据库数据失败: {str(e)}")
            return pd.DataFrame()

    def _create_correct_wide_table(self):
        """创建正确的宽表 - 详细调试版本"""
        if not self.source_data:
            return pd.DataFrame()

        self.logger.info("创建正确宽表...")

        try:
            # 1. 分离不同类型的数据源
            static_dfs = []
            yearly_dfs = []
            dynamic_dfs = []
            phenology_dfs = []
            gldas_dfs = []
            coordinate_dfs = []

            # 收集所有站点和日期范围
            all_station_ids = set()
            all_dates = set()

            for source_name, source_df in self.source_data.items():
                source_df = self._standardize_station_id_format(source_df, source_name)

            for source_name, source_df in self.source_data.items():
                # 收集站点ID
                if 'station_id' in source_df.columns:
                    all_station_ids.update(source_df['station_id'].unique())

                # 收集日期
                if 'date' in source_df.columns:
                    all_dates.update(source_df['date'].unique())

                # 分类数据源
                if self._is_gldas_data(source_name, source_df):
                    gldas_dfs.append((source_name, source_df))
                elif self._is_snow_phenology_data(source_name, source_df):
                    phenology_dfs.append((source_name, source_df))
                elif self._is_terrain_features(source_name, source_df):
                    static_dfs.append((source_name, source_df))
                elif self._is_yearly_data(source_name, source_df):
                    yearly_dfs.append((source_name, source_df))
                elif 'date' in source_df.columns and 'station_id' in source_df.columns:
                    # 动态数据源
                    numeric_cols = source_df.select_dtypes(include=['number']).columns
                    value_cols = [col for col in numeric_cols if
                                  col not in ['station_id', 'date', 'year', 'month', 'day', 'dataset_type']]
                    if value_cols:
                        value_col = value_cols[0]
                        source_wide = source_df[['station_id', 'date', value_col]].copy()
                        source_wide = source_wide.rename(columns={value_col: source_name})
                        source_wide = source_wide.drop_duplicates(['station_id', 'date'])
                        dynamic_dfs.append((source_name, source_wide))
                elif 'station_id' in source_df.columns:
                    static_dfs.append((source_name, source_df))

                # 记录坐标信息
                if self._has_coordinate_info(source_df):
                    coordinate_dfs.append((source_name, source_df))

            self.logger.info(f"数据源分类结果:")
            self.logger.info(f"  动态数据源: {len(dynamic_dfs)} 个")
            self.logger.info(f"  静态数据源: {len(static_dfs)} 个")
            self.logger.info(f"  年度数据源: {len(yearly_dfs)} 个")
            self.logger.info(f"  总站点数: {len(all_station_ids)}")
            self.logger.info(f"  总日期数: {len(all_dates)}")

            # 2. 从数据库获取SWE数据和海拔数据
            swe_df = pd.DataFrame()
            altitude_df = pd.DataFrame()
            if all_station_ids and all_dates:
                start_date = min(all_dates) if all_dates else datetime(2013, 1, 1)
                end_date = max(all_dates) if all_dates else datetime(2018, 12, 31)

                self.logger.info(f"获取数据库数据: {len(all_station_ids)} 个站点, {len(all_dates)} 个日期")
                swe_df = self.get_swe_from_database(list(all_station_ids),
                                                    pd.to_datetime(start_date),
                                                    pd.to_datetime(end_date))

                # 获取海拔数据
                altitude_df = self.get_altitude_from_database(list(all_station_ids))

                self.logger.info(f"数据库数据获取结果:")
                self.logger.info(f"  SWE数据: {len(swe_df)} 行")
                self.logger.info(f"  海拔数据: {len(altitude_df)} 行")

            # 3. 创建基础框架
            final_wide = pd.DataFrame()

            if dynamic_dfs:
                first_dynamic_name, final_wide = dynamic_dfs[0]
                self.logger.info(f"以动态数据源 {first_dynamic_name} 为基础框架: {len(final_wide)} 行")
                self.logger.info(f"基础框架列: {list(final_wide.columns)}")

                # 合并其他动态数据源
                for i in range(1, len(dynamic_dfs)):
                    name, next_df = dynamic_dfs[i]
                    before_count = len(final_wide)
                    next_df = self._standardize_station_id_format(next_df, name)
                    final_wide = final_wide.merge(next_df, on=['station_id', 'date'], how='left')
                    after_count = len(final_wide)
                    self.logger.info(f"合并动态数据源 {name}: {before_count} -> {after_count} 行")
            else:
                self.logger.warning("没有动态数据源，使用静态数据处理")
                return self._handle_static_only_case(static_dfs, yearly_dfs, phenology_dfs, gldas_dfs)

            # 4. 合并数据库SWE数据
            if not swe_df.empty:
                self.logger.info(f"合并SWE数据前 - final_wide形状: {final_wide.shape}")
                self.logger.info(f"SWE数据形状: {swe_df.shape}")

                before_count = len(final_wide)
                final_wide = final_wide.merge(swe_df, on=['station_id', 'date'], how='left')
                after_count = len(final_wide)
                swe_valid_count = final_wide['swe'].notna().sum()
                self.logger.info(f"合并数据库SWE数据: {before_count} -> {after_count} 行, 有效值: {swe_valid_count}")

            # 5. 合并数据库海拔数据 - 使用suffixes参数
            if not altitude_df.empty:
                self.logger.info(f"合并海拔数据...")
                self.logger.info(f"  合并前final_wide列: {list(final_wide.columns)}")
                self.logger.info(f"  海拔数据列: {list(altitude_df.columns)}")

                before_count = len(final_wide)

                # 使用suffixes参数明确指定后缀
                final_wide = final_wide.merge(
                    altitude_df,
                    on=['station_id'],
                    how='left',
                    suffixes=('', '_to_drop')  # 原表不加后缀，新表加_to_drop后缀
                )

                after_count = len(final_wide)

                # 检查合并后的列
                self.logger.info(f"  合并后final_wide列: {list(final_wide.columns)}")

                # 处理带后缀的列 - 保留新合并的数据，删除旧数据
                columns_to_drop = []
                columns_to_rename = {}

                for col in final_wide.columns:
                    if col.endswith('_to_drop'):
                        base_col = col[:-8]  # 去掉 '_to_drop' 后缀
                        # 如果原表中有同名列，删除原表的列，重命名新列
                        if base_col in final_wide.columns:
                            columns_to_drop.append(base_col)  # 删除原表的列
                            columns_to_rename[col] = base_col  # 重命名新列为原名
                        else:
                            columns_to_rename[col] = base_col  # 直接重命名

                # 执行删除和重命名
                if columns_to_drop:
                    final_wide = final_wide.drop(columns=columns_to_drop)
                    self.logger.info(f"  删除原列: {columns_to_drop}")

                if columns_to_rename:
                    final_wide = final_wide.rename(columns=columns_to_rename)
                    self.logger.info(f"  重命名列: {columns_to_rename}")

                altitude_valid_count = final_wide['altitude'].notna().sum() if 'altitude' in final_wide.columns else 0
                self.logger.info(f"合并数据库海拔数据: {before_count} -> {after_count} 行, 有效值: {altitude_valid_count}")
                self.logger.info(f"  最终列: {list(final_wide.columns)}")

            # 6. 处理GLDAS数据
            if gldas_dfs:
                for gldas_name, gldas_df in gldas_dfs:
                    self.logger.info(f"处理GLDAS数据 {gldas_name}: {len(gldas_df)} 行")

                    feature_columns = ['station_id', 'date']
                    target_columns = ['doy', 'seasonal_doy_Da', 'seasonal_doy_Db',
                                      'seasonal_doy_Dc', 'seasonal_doy_Dd', 'gldas']

                    available_columns = []
                    for col in target_columns:
                        if col in gldas_df.columns:
                            available_columns.append(col)
                            feature_columns.append(col)

                    if available_columns:
                        gldas_wide = gldas_df[feature_columns].copy()
                        gldas_wide = gldas_wide.drop_duplicates(['station_id', 'date'])

                        before_count = len(final_wide)
                        final_wide = final_wide.merge(gldas_wide, on=['station_id', 'date'], how='left')
                        after_count = len(final_wide)

                        for col in available_columns:
                            valid_count = final_wide[col].notna().sum()
                            self.logger.info(f"GLDAS特征 {col}: {valid_count} 有效记录")

                        self.logger.info(f"合并GLDAS数据: {before_count} -> {after_count} 行")

              # 7. 处理积雪物候数据
            if phenology_dfs:
                for phenology_name, phenology_df in phenology_dfs:
                    self.logger.info(f"处理积雪物候数据 {phenology_name}: {len(phenology_df)} 行")

                    if 'dataset_type' in phenology_df.columns and 'hydrological_year' in phenology_df.columns:
                        value_col = 'day_of_year' if 'day_of_year' in phenology_df.columns else 'value'

                        if value_col in phenology_df.columns:
                            # 分离初日和终日数据
                            start_data = phenology_df[phenology_df['dataset_type'] == 'start'][
                                ['station_id', 'hydrological_year', value_col]].copy()
                            start_data = start_data.rename(columns={value_col: f'{phenology_name}_start'})
                            start_data = start_data.drop_duplicates(['station_id', 'hydrological_year'])

                            end_data = phenology_df[phenology_df['dataset_type'] == 'end'][
                                ['station_id', 'hydrological_year', value_col]].copy()
                            end_data = end_data.rename(columns={value_col: f'{phenology_name}_end'})
                            end_data = end_data.drop_duplicates(['station_id', 'hydrological_year'])

                            # 添加水文年列
                            final_wide = final_wide.copy()
                            final_wide['hydrological_year'] = final_wide['date'].apply(
                                lambda x: self._get_hydrological_year_from_str(x)
                            )

                            # 合并初日数据
                            if not start_data.empty:
                                before_count = final_wide[
                                    f'{phenology_name}_start'].notna().sum() if f'{phenology_name}_start' in final_wide.columns else 0
                                final_wide = final_wide.merge(start_data, on=['station_id', 'hydrological_year'],
                                                              how='left')
                                after_count = final_wide[f'{phenology_name}_start'].notna().sum()
                                self.logger.info(f"合并初日数据: {before_count} -> {after_count} 有效记录")

                            # 合并终日数据
                            if not end_data.empty:
                                before_count = final_wide[
                                    f'{phenology_name}_end'].notna().sum() if f'{phenology_name}_end' in final_wide.columns else 0
                                final_wide = final_wide.merge(end_data, on=['station_id', 'hydrological_year'],
                                                              how='left')
                                after_count = final_wide[f'{phenology_name}_end'].notna().sum()
                                self.logger.info(f"合并终日数据: {before_count} -> {after_count} 有效记录")

                            # 移除临时列
                            final_wide = final_wide.drop('hydrological_year', axis=1)

            # 8. 处理年度数据
            if yearly_dfs:
                for yearly_name, yearly_df in yearly_dfs:
                    self.logger.info(f"处理年度数据 {yearly_name}")

                    if 'year' not in yearly_df.columns:
                        if 'date' in yearly_df.columns:
                            yearly_df = yearly_df.copy()
                            yearly_df['year'] = pd.to_datetime(yearly_df['date']).dt.year
                        else:
                            self.logger.warning(f"年度数据 {yearly_name} 缺少年份信息，跳过")
                            continue

                    numeric_cols = yearly_df.select_dtypes(include=['number']).columns
                    value_cols = [col for col in numeric_cols if col not in ['station_id', 'date', 'year']]

                    if not value_cols:
                        self.logger.warning(f"年度数据 {yearly_name} 没有数值列，跳过")
                        continue

                    value_col = value_cols[0]
                    yearly_mapping = yearly_df.set_index(['station_id', 'year'])[value_col].to_dict()

                    final_wide = final_wide.copy()
                    if 'year' not in final_wide.columns:
                        final_wide['year'] = pd.to_datetime(final_wide['date']).dt.year

                    def get_yearly_value(row):
                        key = (row['station_id'], row['year'])
                        return yearly_mapping.get(key)

                    final_wide[yearly_name] = final_wide.apply(get_yearly_value, axis=1)

                    filled_count = final_wide[yearly_name].notna().sum()
                    self.logger.info(f"年度数据 {yearly_name}: 填充了 {filled_count} 条记录")

                    final_wide = final_wide.drop('year', axis=1)

            # 9. 合并所有静态数据源
            if static_dfs:
                static_combined = static_dfs[0][1]

                for i in range(1, len(static_dfs)):
                    name, static_df = static_dfs[i]
                    before_cols = len(static_combined.columns)
                    static_combined = static_combined.merge(static_df, on='station_id', how='outer')
                    after_cols = len(static_combined.columns)
                    added_cols = after_cols - before_cols
                    self.logger.info(f"合并静态数据源 {name}: 添加 {added_cols} 列")

                before_count = len(final_wide)
                final_wide = final_wide.merge(static_combined, on='station_id', how='left')
                after_count = len(final_wide)
                self.logger.info(f"合并所有静态数据: {before_count} -> {after_count} 行")

            # 10. 确保坐标信息完整
            final_wide = self._ensure_complete_coordinates(final_wide)

            # 11. 排序和整理
            final_wide = final_wide.sort_values(['station_id', 'date']).reset_index(drop=True)

            # 最终数据验证
            self._validate_final_wide_table(final_wide)

            self.logger.info(f"✅ 宽表创建完成: {final_wide.shape}")
            return final_wide

        except Exception as e:
            self.logger.error(f"创建宽表失败: {e}")
            import traceback
            self.logger.debug(f"详细错误: {traceback.format_exc()}")
            return pd.DataFrame()

    def _standardize_station_id_format(self, df, source_name):
        """统一station_id为字符串格式"""
        try:
            if 'station_id' not in df.columns:
                return df

            df_copy = df.copy()

            # 检查当前station_id的数据类型
            station_id_dtype = str(df_copy['station_id'].dtype)
            self.logger.debug(f"{source_name} station_id类型: {station_id_dtype}")

            # 转换为统一的字符串格式
            if station_id_dtype != 'object':
                df_copy['station_id'] = df_copy['station_id'].astype(str)
                self.logger.debug(f"{source_name}: 转换station_id到字符串格式")

            return df_copy

        except Exception as e:
            self.logger.error(f"标准化 {source_name} station_id格式失败: {str(e)}")
            return df

    def _add_coordinate_info(self, df, coordinate_dfs):
        """添加坐标信息到数据框"""
        try:
            if not coordinate_dfs:
                return df

            # 使用第一个包含坐标信息的数据源
            coord_source_name, coord_df = coordinate_dfs[0]
            self.logger.info(f"从 {coord_source_name} 提取坐标信息")

            coord_columns = self._get_coordinate_columns(coord_df)
            if coord_columns:
                # 标准化坐标列名
                coord_mapping = {}
                for col in coord_columns:
                    if col.lower() in ['longitude', 'lon']:
                        coord_mapping[col] = 'longitude'
                    elif col.lower() in ['latitude', 'lat']:
                        coord_mapping[col] = 'latitude'

                coord_df_standardized = coord_df.rename(columns=coord_mapping)
                coord_mapping_df = coord_df_standardized[['station_id', 'longitude', 'latitude']].drop_duplicates(
                    'station_id')

                # 合并坐标信息
                before_coord_count = df[['longitude', 'latitude']].notna().all(
                    axis=1).sum() if 'longitude' in df.columns and 'latitude' in df.columns else 0
                df = df.merge(coord_mapping_df, on='station_id', how='left', suffixes=('', '_dup'))

                # 处理重复列
                for col in ['longitude', 'latitude']:
                    if f'{col}_dup' in df.columns:
                        # 如果原列为空，使用新列的值填充
                        mask = df[col].isna() & df[f'{col}_dup'].notna()
                        df.loc[mask, col] = df.loc[mask, f'{col}_dup']
                        df = df.drop(f'{col}_dup', axis=1)

                after_coord_count = df[['longitude', 'latitude']].notna().all(axis=1).sum()
                self.logger.info(f"坐标信息添加: {before_coord_count} -> {after_coord_count} 完整坐标记录")

            return df

        except Exception as e:
            self.logger.error(f"添加坐标信息失败: {str(e)}")
            return df

    def emergency_fix(self, issue_type='all'):
        """应急数据修复 - 增强版本，专门处理数据类型问题"""
        try:
            fix_count = 0

            if not self.source_data:
                self.logger.warning("没有数据源可修复")
                return fix_count

            # 强制标准化所有数据源的station_id格式
            self.logger.info("=== 强制标准化所有station_id格式 ===")
            for name, df in self.source_data.items():
                if 'station_id' in df.columns:
                    original_type = str(df['station_id'].dtype)
                    fixed_df = self._standardize_station_id_format(df, f"修复-{name}")
                    new_type = str(fixed_df['station_id'].dtype)

                    if original_type != new_type:
                        self.source_data[name] = fixed_df
                        fix_count += 1
                        self.logger.info(f"✅ 修复 {name}: station_id {original_type} -> {new_type}")

            # 处理重复数据
            if issue_type in ['all', 'duplicates']:
                fix_count += self._fix_duplicates()

            # 处理列名问题
            if issue_type in ['all', 'column_names']:
                fix_count += self._fix_column_names()

            self.logger.info(f"🎯 应急修复完成: 总共执行了 {fix_count} 个修复")
            return fix_count

        except Exception as e:
            self.logger.error(f"应急修复失败: {e}")
            return 0

    def _fix_duplicates(self):
        """修复重复数据"""
        fix_count = 0
        for name, df in self.source_data.items():
            original_count = len(df)
            df_clean = df.drop_duplicates()
            removed_count = original_count - len(df_clean)
            if removed_count > 0:
                self.source_data[name] = df_clean
                fix_count += removed_count
                self.logger.info(f"✅ 去重 {name}: 移除 {removed_count} 个重复行")
        return fix_count

    def _fix_column_names(self):
        """修复列名问题"""
        fix_count = 0
        for name, df in self.source_data.items():
            column_mapping = {}
            if 'station_ID' in df.columns and 'station_id' not in df.columns:
                column_mapping['station_ID'] = 'station_id'
            if 'Station_ID' in df.columns and 'station_id' not in df.columns:
                column_mapping['Station_ID'] = 'station_id'

            if column_mapping:
                df_renamed = df.rename(columns=column_mapping)
                self.source_data[name] = df_renamed
                fix_count += len(column_mapping)
                self.logger.info(f"✅ 列名修复 {name}: {list(column_mapping.keys())}")
        return fix_count

    def _ensure_complete_coordinates(self, df):
        """确保坐标信息完整"""
        try:
            # 检查是否缺少坐标信息
            if 'longitude' not in df.columns or 'latitude' not in df.columns:
                self.logger.info("尝试从所有数据源补充坐标信息")
                df = self._supplement_coordinates_from_all_sources(df)

            # 统计坐标完整性
            if 'longitude' in df.columns and 'latitude' in df.columns:
                complete_coords = df[['longitude', 'latitude']].notna().all(axis=1).sum()
                total_records = len(df)
                completeness = (complete_coords / total_records) * 100
                self.logger.info(f"坐标完整性: {complete_coords}/{total_records} ({completeness:.1f}%)")

            return df

        except Exception as e:
            self.logger.error(f"确保坐标完整失败: {str(e)}")
            return df

    def _supplement_coordinates_from_all_sources(self, df):
        """从所有数据源补充坐标信息"""
        try:
            all_coordinates = {}

            for source_name, source_df in self.source_data.items():
                if self._has_coordinate_info(source_df):
                    coord_columns = self._get_coordinate_columns(source_df)
                    if len(coord_columns) >= 2:
                        coord_info = source_df[['station_id'] + coord_columns].drop_duplicates('station_id')

                        for _, row in coord_info.iterrows():
                            station_id = row['station_id']
                            if station_id not in all_coordinates:
                                # 确定经度和纬度列
                                lon_col = next((col for col in coord_columns if col.lower() in ['longitude', 'lon']),
                                               None)
                                lat_col = next((col for col in coord_columns if col.lower() in ['latitude', 'lat']),
                                               None)

                                if lon_col and lat_col and pd.notna(row[lon_col]) and pd.notna(row[lat_col]):
                                    all_coordinates[station_id] = {
                                        'longitude': float(row[lon_col]),
                                        'latitude': float(row[lat_col])
                                    }

            # 将坐标信息添加到数据框
            if all_coordinates:
                coord_df = pd.DataFrame([
                    {'station_id': sid, 'longitude': info['longitude'], 'latitude': info['latitude']}
                    for sid, info in all_coordinates.items()
                ])

                if 'longitude' in df.columns and 'latitude' in df.columns:
                    # 更新现有的空值坐标
                    for station_id, coords in all_coordinates.items():
                        mask = (df['station_id'] == station_id) & (df['longitude'].isna() | df['latitude'].isna())
                        df.loc[mask, 'longitude'] = coords['longitude']
                        df.loc[mask, 'latitude'] = coords['latitude']
                else:
                    # 添加新的坐标列
                    df = df.merge(coord_df, on='station_id', how='left')

                self.logger.info(f"从 {len(all_coordinates)} 个数据源补充了坐标信息")

            return df

        except Exception as e:
            self.logger.error(f"补充坐标信息失败: {str(e)}")
            return df

    def _get_hydrological_year_from_str(self, date_str):
        """从日期字符串计算水文年"""
        try:
            date = pd.to_datetime(date_str)
            if date.month >= 9 or (date.month == 9 and date.day >= 1):
                return date.year
            else:
                return date.year - 1
        except:
            return None

    def _is_gldas_data(self, source_name, df):
        """判断是否为GLDAS数据"""
        gldas_keywords = ['gldas', 'seasonal_doy']
        if any(keyword in source_name.lower() for keyword in gldas_keywords):
            return True

        gldas_columns = ['seasonal_doy_Da', 'seasonal_doy_Db', 'seasonal_doy_Dc', 'seasonal_doy_Dd']
        return any(col in df.columns for col in gldas_columns)

    def _is_snow_phenology_data(self, source_name, df):
        """判断是否为积雪物候数据"""
        snow_keywords = ['snow_phenology', 'snow_start', 'snow_end', 'phenology', 'scp']
        if any(keyword in source_name.lower() for keyword in snow_keywords):
            return True

        return 'dataset_type' in df.columns and 'hydrological_year' in df.columns

    def _is_terrain_features(self, source_name, df):
        """判断是否为地形特征数据"""
        if 'terrain' in source_name.lower() or 'feature' in source_name.lower():
            return True

        numeric_cols = df.select_dtypes(include=['number']).columns
        feature_cols = [col for col in numeric_cols if col not in ['station_id', 'date', 'longitude', 'latitude']]

        return len(feature_cols) > 1 and 'date' not in df.columns

    def _is_yearly_data(self, source_name, df):
        """判断是否为年度数据"""
        if 'landcover' in source_name.lower():
            return True

        if 'station_id' in df.columns:
            station_counts = df.groupby('station_id').size()
            return station_counts.mean() < 5

        return False

    def _handle_static_only_case(self, static_dfs, yearly_dfs, phenology_dfs, gldas_dfs):
        """处理只有静态数据的情况"""
        try:
            self.logger.info("处理纯静态数据情况")

            all_dfs = []

            for name, df in static_dfs + yearly_dfs + phenology_dfs + gldas_dfs:
                df_copy = df.copy()
                if 'date' not in df_copy.columns:
                    df_copy['date'] = datetime.now().strftime('%Y-%m-%d')
                all_dfs.append(df_copy)

            if not all_dfs:
                return pd.DataFrame()

            final_df = all_dfs[0]
            for i in range(1, len(all_dfs)):
                common_keys = ['station_id', 'date']
                available_keys = [key for key in common_keys if key in final_df.columns and key in all_dfs[i].columns]

                if available_keys:
                    final_df = final_df.merge(all_dfs[i], on=available_keys, how='outer')

            self.logger.info(f"纯静态数据合并完成: {final_df.shape}")
            return final_df

        except Exception as e:
            self.logger.error(f"处理纯静态数据失败: {e}")
            return pd.DataFrame()

    def _validate_final_wide_table(self, df):
        """验证最终宽表数据 - 包含海拔数据"""
        self.logger.info("=== 最终宽表验证 ===")
        self.logger.info(f"总行数: {len(df)}")
        self.logger.info(f"总列数: {len(df.columns)}")

        # 检查坐标信息
        if 'longitude' in df.columns and 'latitude' in df.columns:
            coord_count = df[['longitude', 'latitude']].notna().all(axis=1).sum()
            coord_percentage = (coord_count / len(df)) * 100
            self.logger.info(f"坐标信息完整性: {coord_count}/{len(df)} ({coord_percentage:.1f}%)")

        # 检查SWE数据
        if 'swe' in df.columns:
            swe_count = df['swe'].notna().sum()
            swe_percentage = (swe_count / len(df)) * 100
            self.logger.info(f"SWE数据完整性: {swe_count}/{len(df)} ({swe_percentage:.1f}%)")

        # 检查海拔数据
        if 'altitude' in df.columns:
            altitude_count = df['altitude'].notna().sum()
            altitude_percentage = (altitude_count / len(df)) * 100
            altitude_stats = df['altitude'].describe()
            self.logger.info(f"海拔数据完整性: {altitude_count}/{len(df)} ({altitude_percentage:.1f}%)")
            self.logger.info(
                f"海拔统计 - 均值: {altitude_stats['mean']:.2f}, 范围: {altitude_stats['min']:.2f}-{altitude_stats['max']:.2f}")

        # 显示前5行数据示例
        if len(df) > 0:
            self.logger.info("前5行数据示例:")
            sample_cols = ['station_id', 'date']
            if 'longitude' in df.columns and 'latitude' in df.columns:
                sample_cols.extend(['longitude', 'latitude'])
            if 'swe' in df.columns:
                sample_cols.append('swe')
            if 'altitude' in df.columns:
                sample_cols.append('altitude')

            # 添加其他数据列
            other_cols = [col for col in df.columns if col not in sample_cols]
            sample_cols.extend(other_cols[:5])  # 显示前5个其他列

            sample_df = df[sample_cols].head(5)
            for _, row in sample_df.iterrows():
                values = []
                for col in sample_cols:
                    if col not in ['station_id', 'date'] and pd.notna(row[col]):
                        values.append(f"{col}: {row[col]}")
                value_str = ", ".join(values) if values else "无数据"
                self.logger.info(f"  站点 {row['station_id']} | 日期 {row['date']} | {value_str}")

    def save_master_excel(self, format_type='wide'):
        """保存主Excel文件"""
        if not self.source_data:
            self.logger.error("没有数据源可整合")
            return None

        try:
            # 先执行数据修复
            fix_count = self.emergency_fix()
            self.logger.info(f"数据修复完成: {fix_count} 个问题已修复")

            # 创建宽表
            if format_type == 'wide':
                final_df = self._create_correct_wide_table()
            else:
                final_df = self.get_combined_data()

            if final_df is None or final_df.empty:
                self.logger.error("最终数据为空或为None")
                return None

            # 数据完整性验证
            final_df = self._validate_data_integrity(final_df)

            if final_df.empty:
                self.logger.error("验证后数据为空")
                return None

            # 生成输出文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"integrated_data_{timestamp}.xlsx"

            # 保存Excel文件
            final_df.to_excel(output_path, index=False)
            self.logger.info(f"✅ 整合数据保存成功: {output_path}")

            return str(output_path)

        except Exception as e:
            self.logger.error(f"保存失败: {e}")
            self.logger.debug(f"详细错误: {traceback.format_exc()}")
            return None

    def get_swe_from_database(self, station_ids, start_date, end_date):
        """从数据库获取SWE数据和海拔高度 - 增强版本"""
        try:
            if not self.db_conn:
                if not self.connect_database():
                    return pd.DataFrame()

            # 根据实际数据库列名调整
            swe_column = 'swe（mm）'
            station_id_column = 'station_ID'
            time_column = 'time'
            altitude_column = 'Altitude(m)'  # 新增海拔列

            self.logger.info(f"获取数据库数据 - SWE列: '{swe_column}', 海拔列: '{altitude_column}'")

            # 构建查询 - 同时获取SWE和海拔数据
            query = f"""
            SELECT {station_id_column}, {time_column}, "{swe_column}", "{altitude_column}"
            FROM stations 
            WHERE "{swe_column}" IS NOT NULL 
                AND {time_column} BETWEEN ? AND ?
            """

            # 转换日期格式为数据库中的格式 (YYYYMMDD)
            start_db = int(start_date.strftime('%Y%m%d'))
            end_db = int(end_date.strftime('%Y%m%d'))

            params = [start_db, end_db]
            df = pd.read_sql_query(query, self.db_conn, params=params)

            if not df.empty:
                # 转换数据库格式的日期 (YYYYMMDD -> YYYY-MM-DD)
                df['date'] = df[time_column].apply(
                    lambda x: datetime.strptime(str(int(x)), '%Y%m%d').strftime('%Y-%m-%d')
                )

                # 标准化站点ID列名
                df['station_id'] = df[station_id_column].astype(str)

                # 选择需要的列，包括海拔
                result_df = df[['station_id', 'date', swe_column, altitude_column]].copy()
                result_df = result_df.rename(columns={
                    swe_column: 'swe',
                    altitude_column: 'altitude'
                })

                # 站点匹配调试
                needed_stations = set(station_ids)
                available_stations = set(result_df['station_id'].unique())
                matched_stations = needed_stations & available_stations

                self.logger.info(f"站点匹配详情:")
                self.logger.info(f"  所需站点数量: {len(needed_stations)}")
                self.logger.info(f"  数据库有数据的站点数量: {len(available_stations)}")
                self.logger.info(f"  匹配的站点数量: {len(matched_stations)}")

                if matched_stations:
                    result_df = result_df[result_df['station_id'].isin(matched_stations)]

                    # 数据统计
                    swe_stats = result_df['swe'].describe()
                    altitude_stats = result_df['altitude'].describe()
                    unique_dates = result_df['date'].nunique()

                    self.logger.info(f"✅ 成功获取数据库数据: {len(result_df)} 条记录")
                    self.logger.info(f"  SWE数据覆盖 {unique_dates} 个日期")
                    self.logger.info(
                        f"  SWE统计 - 均值: {swe_stats['mean']:.2f}, 范围: {swe_stats['min']:.2f}-{swe_stats['max']:.2f}")
                    self.logger.info(
                        f"  海拔统计 - 均值: {altitude_stats['mean']:.2f}, 范围: {altitude_stats['min']:.2f}-{altitude_stats['max']:.2f}")

                    return result_df
                else:
                    self.logger.warning("没有匹配的站点ID")
                    return pd.DataFrame()
            else:
                self.logger.warning("数据库中未找到指定时间范围内的数据")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取数据库数据失败: {str(e)}")
            return pd.DataFrame()

    def get_static_altitude_data(self, station_ids):
        """获取站点的静态海拔数据（每个站点一个海拔值）"""
        try:
            if not self.db_conn:
                if not self.connect_database():
                    return pd.DataFrame()

            altitude_column = 'Altitude(m)'
            station_id_column = 'station_ID'

            # 获取每个站点的平均海拔（静态特征）
            query = f"""
            SELECT {station_id_column}, AVG("{altitude_column}") as altitude
            FROM stations 
            WHERE "{altitude_column}" IS NOT NULL
            GROUP BY {station_id_column}
            """

            df = pd.read_sql_query(query, self.db_conn)

            if not df.empty:
                df['station_id'] = df[station_id_column].astype(str)
                result_df = df[['station_id', 'altitude']].drop_duplicates('station_id')

                # 过滤需要的站点
                if station_ids:
                    result_df = result_df[result_df['station_id'].isin(station_ids)]

                self.logger.info(f"获取静态海拔数据: {len(result_df)} 个站点")
                return result_df
            else:
                self.logger.warning("数据库中未找到海拔数据")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取静态海拔数据失败: {str(e)}")
            return pd.DataFrame()

    def get_altitude_from_database(self, station_ids):
        """从数据库获取海拔数据"""
        try:
            if not self.db_conn:
                if not self.connect_database():
                    return pd.DataFrame()

            altitude_column = 'Altitude(m)'
            self.logger.info(f"获取海拔数据，使用列: '{altitude_column}'")

            # 获取每个站点的海拔数据（取平均值）
            query = f"""
            SELECT station_ID, AVG("{altitude_column}") as altitude
            FROM stations 
            WHERE "{altitude_column}" IS NOT NULL
            GROUP BY station_ID
            """

            df = pd.read_sql_query(query, self.db_conn)

            if not df.empty:
                df['station_id'] = df['station_ID'].astype(str)
                df = df[['station_id', 'altitude']]

                # 过滤需要的站点
                if station_ids:
                    df = df[df['station_id'].isin(station_ids)]

                self.logger.info(f"✅ 成功获取海拔数据: {len(df)} 个站点")

                # 显示海拔统计
                altitude_stats = df['altitude'].describe()
                self.logger.info(
                    f"  海拔统计 - 均值: {altitude_stats['mean']:.2f}, 范围: {altitude_stats['min']:.2f}-{altitude_stats['max']:.2f}")

                return df
            else:
                self.logger.warning("数据库中未找到海拔数据")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取海拔数据失败: {str(e)}")
            return pd.DataFrame()

    def _query_swe_batch(self, station_ids, swe_column, start_db, end_db):
        """分批查询SWE数据"""
        try:
            placeholders = ','.join(['?'] * len(station_ids))
            query = f"""
            SELECT station_ID, time, {swe_column} 
            FROM stations 
            WHERE station_ID IN ({placeholders}) AND time BETWEEN ? AND ?
            """
            params = station_ids + [start_db, end_db]

            df = pd.read_sql_query(query, self.db_conn, params=params)
            return df

        except Exception as e:
            self.logger.error(f"分批查询SWE数据失败: {str(e)}")
            return pd.DataFrame()

    def check_database_structure(self):
        """检查数据库表结构 - 详细版本"""
        try:
            if not self.db_conn:
                if not self.connect_database():
                    return None

            # 获取表结构信息
            tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = pd.read_sql_query(tables_query, self.db_conn)
            self.logger.info(f"数据库中的表: {tables['name'].tolist()}")

            # 检查stations表的列
            if 'stations' in tables['name'].values:
                columns_query = "PRAGMA table_info(stations);"
                columns = pd.read_sql_query(columns_query, self.db_conn)
                column_names = columns['name'].tolist()
                self.logger.info(f"stations表的完整列信息:")
                for _, row in columns.iterrows():
                    self.logger.info(f"  - {row['name']} ({row['type']})")
                return column_names
            else:
                self.logger.error("stations表不存在")
                return None

        except Exception as e:
            self.logger.error(f"检查数据库结构失败: {str(e)}")
            return None

    def get_combined_data(self):
        """合并数据"""
        if not self.source_data:
            return pd.DataFrame()

        dfs = []
        for name, df in self.source_data.items():
            df_copy = df.copy()
            df_copy['data_source'] = name
            dfs.append(df_copy)

        return pd.concat(dfs, ignore_index=True)

    def _validate_data_integrity(self, df):
        """验证数据完整性，确保没有虚假的站点-日期组合"""
        if df.empty:
            return df

        # 获取数据列
        data_cols = [col for col in df.columns if col not in ['station_id', 'date', 'data_source']]

        if not data_cols:
            return df

        # 统计每个数据源的有效记录数
        self.logger.info("=== 数据有效性验证 ===")
        for col in data_cols:
            valid_count = df[col].notna().sum()
            total_count = len(df)
            fill_rate = (valid_count / total_count) * 100 if total_count > 0 else 0
            self.logger.info(f"{col}: {valid_count}/{total_count} 有效记录 ({fill_rate:.1f}%)")

        # 识别可能的问题记录
        empty_records = df[data_cols].isna().all(axis=1)
        if empty_records.any():
            empty_count = empty_records.sum()
            self.logger.warning(f"发现 {empty_count} 条全空记录")

            # 显示一些示例
            empty_examples = df[empty_records][['station_id', 'date']].head(5)
            self.logger.info("全空记录示例:")
            for _, row in empty_examples.iterrows():
                self.logger.info(f"  站点 {row['station_id']}, 日期 {row['date']}")

        return df

    def generate_report(self):
        """生成数据整合报告"""
        try:
            report_lines = []
            report_lines.append("=== 数据整合报告 ===")
            report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")

            # 数据源统计
            report_lines.append("=== 数据源统计 ===")
            for name, df in self.source_data.items():
                station_count = df['station_id'].nunique() if 'station_id' in df.columns else 0
                date_count = df['date'].nunique() if 'date' in df.columns else "N/A"
                record_count = len(df)

                report_lines.append(f"{name}:")
                report_lines.append(f"  - 记录数: {record_count}")
                report_lines.append(f"  - 站点数: {station_count}")
                report_lines.append(f"  - 日期数: {date_count}")

                # 检查坐标信息
                if self._has_coordinate_info(df):
                    coord_columns = self._get_coordinate_columns(df)
                    report_lines.append(f"  - 坐标信息: {coord_columns}")

                report_lines.append("")

            # 特征列统计
            report_lines.append("=== 特征列统计 ===")
            all_columns = set()
            for name, df in self.source_data.items():
                all_columns.update(df.columns)

            # 排除ID和日期列
            feature_columns = [col for col in all_columns if col not in ['station_id', 'date', 'data_source']]
            report_lines.append(f"总特征列数: {len(feature_columns)}")
            report_lines.append("特征列列表:")
            for col in sorted(feature_columns):
                report_lines.append(f"  - {col}")

            return "\n".join(report_lines)

        except Exception as e:
            self.logger.error(f"生成报告失败: {str(e)}")
            return f"报告生成失败: {str(e)}"

    def clear_data(self):
        """清空数据"""
        self.source_data.clear()
        self.logger.info("数据已清空")

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