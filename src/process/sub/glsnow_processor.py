import os.path
from datetime import datetime
import re
from src.process.config import config
import pandas as pd
import xarray as xr
import sqlite3
import logging
import numpy as np
from pathlib import Path
import pyproj
from pyproj import Transformer


class GLSnowProcessor:
    def __init__(self):
        self.logger = logging.getLogger("GLSnowProcessor")
        self.conf = config.glsnow
        self.output_dir = Path(config.get_output_dir('glsnow'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_conn = self.connect_database()
        self.proj_transformer=None

    def _init_projection(self, ds):
        """初始化投影转换器（修复DataArray属性访问）"""
        try:
            # 尝试从文件获取投影参数
            proj_params = {}

            # 检查全局属性
            if hasattr(ds, 'attrs') and 'crs' in ds.attrs:
                # 从全局属性获取CRS信息
                crs_info = ds.attrs['crs']
                if hasattr(crs_info, 'attrs'):
                    # 如果crs是DataArray，从其属性获取参数
                    proj_params = crs_info.attrs
                elif isinstance(crs_info, dict):
                    # 如果crs是字典
                    proj_params = crs_info
            elif hasattr(ds, 'crs') and ds.crs is not None:
                # 处理crs DataArray
                if hasattr(ds.crs, 'attrs'):
                    proj_params = ds.crs.attrs
                else:
                    # 尝试直接访问常见属性
                    proj_params = {
                        'grid_mapping_name': getattr(ds.crs, 'grid_mapping_name', None),
                        'latitude_of_projection_origin': getattr(ds.crs, 'latitude_of_projection_origin', None),
                        'longitude_of_projection_origin': getattr(ds.crs, 'longitude_of_projection_origin', None),
                        'false_easting': getattr(ds.crs, 'false_easting', None),
                        'false_northing': getattr(ds.crs, 'false_northing', None),
                        'earth_radius': getattr(ds.crs, 'earth_radius', None),
                    }

            # 如果从文件无法获取，使用配置中的投影参数
            if not proj_params:
                proj_params = self.conf.get('projection', {})
                self.logger.warning("使用配置中的投影参数，文件中的投影信息不可用")

            # 记录投影参数用于调试
            self.logger.debug(f"投影参数: {proj_params}")

            # 提取必要的投影参数
            lat_0 = proj_params.get('latitude_of_projection_origin') or proj_params.get('latitude_of_projection_origin',
                                                                                        90.0)
            lon_0 = proj_params.get('longitude_of_projection_origin') or proj_params.get(
                'longitude_of_projection_origin', 0.0)
            x_0 = proj_params.get('false_easting') or proj_params.get('false_easting', 0.0)
            y_0 = proj_params.get('false_northing') or proj_params.get('false_northing', 0.0)
            R = proj_params.get('earth_radius') or proj_params.get('earth_radius', 6371228.0)

            # 创建 LAEA 投影
            laea_proj = pyproj.Proj(proj='laea',
                                    lat_0=float(lat_0),
                                    lon_0=float(lon_0),
                                    x_0=float(x_0),
                                    y_0=float(y_0),
                                    R=float(R))

            # 创建 WGS84 到 LAEA 的转换器
            wgs84 = pyproj.Proj('epsg:4326')
            self.proj_transformer = Transformer.from_proj(wgs84, laea_proj, always_xy=True)

            self.logger.info(f"成功初始化投影转换: LAEA(lat_0={lat_0}, lon_0={lon_0})")
            return True
        except Exception as e:
            self.logger.exception(f"投影初始化失败: {str(e)}")
            return False

    def connect_database(self):
        """连接站点数据库"""
        db_path = config.get_station_db_path()
        try:
            conn = sqlite3.connect(db_path)
            conn.enable_load_extension(True)
            conn.load_extension("mod_spatialite")  # 加载空间扩展
            return conn
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")
            raise

    def extract_values(self, date, file_path):
        """使用配置的维度名称提取数据"""
        try:
            # 从文件名解析实际日期
            file_date = self.parse_date_from_filename(file_path.name)
            if not file_date:
                self.logger.error(f"无法从文件名解析日期: {file_path.name}")
                return pd.DataFrame()

            # 获取该日期有效的站点
            stations = self.get_stations_for_date(file_date)
            if not stations:
                self.logger.warning(f"未找到 {file_date} 的有效站点")
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

            with xr.open_dataset(file_path) as ds:
                if not self._init_projection(ds):
                    self.logger.error("无法初始化投影转换")
                    return pd.DataFrame()

                station_x,station_y=self.proj_transformer.transform(lons,lats)
                # 获取维度名称
                lon_var = self.conf.get('lon_var', 'x')
                lat_var = self.conf.get('lat_var', 'y')
                data_var_name = self.conf.get('variable', 'swe')

                # 验证变量存在
                if data_var_name not in ds:
                    self.logger.error(f"变量 '{data_var_name}' 不存在于文件中。可用变量: {list(ds.data_vars)}")
                    return pd.DataFrame()

                # 验证维度存在
                if lon_var not in ds.dims or lat_var not in ds.dims:
                    self.logger.error(f"维度不匹配: lon_dim={lon_var}, lat_dim={lat_var}。文件维度: {list(ds.dims)}")
                    return pd.DataFrame()

                data_var = ds[data_var_name]

                # 创建坐标数据集
                points = xr.Dataset({
                    'station': ('station', station_ids),
                    'lat': ('station', lats),
                    'lon': ('station', lons)
                })

                # 提取数据
                extracted = data_var.sel(
                    **{
                        lat_var: xr.DataArray(station_y,dims='station'),
                        lon_var: xr.DataArray(station_x,dims='station')
                    },
                    method='nearest'
                )

                # 构建结果
                results = []
                for i in range(len(station_ids)):
                    value = float(extracted.values[i])
                    if np.isnan(value) or value == self.conf.get('fill_value', -9999):
                        continue

                    results.append({
                        'station_id': station_ids[i],
                        'date': file_date.strftime('%Y-%m-%d'),
                        'value': value,
                        'file_used': file_path.name
                    })

                return pd.DataFrame(results)

        except Exception as e:
            self.logger.exception(f"提取错误: {str(e)}")
            return pd.DataFrame()

    def parse_date_from_filename(self, filename):
        """从文件名解析日期（增强版）"""
        try:
            # 尝试提取8位数字日期
            match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
            if match:
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))

            # 尝试提取其他格式
            for fmt in ['%Y-%m-%d', '%Y%m%d', '%m_%d_%Y']:
                try:
                    match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', filename)
                    if match:
                        return datetime.strptime(match.group(1), fmt)
                except:
                    continue

            self.logger.warning(f"无法从文件名解析日期: {filename}")
            return None
        except Exception as e:
            self.logger.error(f"日期解析错误: {str(e)}")
            return None

    def get_stations_for_date(self, target_date):
        """获取特定日期有效的站点（处理浮点日期格式）"""
        try:
            # 将目标日期转换为数据库存储格式 (YYYYMMDD.0)
            db_date_format = float(target_date.strftime('%Y%m%d'))

            query = f"""
            SELECT station_ID, Longitude, Latitude 
            FROM stations
            WHERE time = {db_date_format}
            """

            self.logger.debug(f"执行查询: {query}")

            df = pd.read_sql_query(query, self.db_conn)

            # 检查结果
            if df.empty:
                self.logger.warning(f"未找到时间 {db_date_format} 的站点记录")
                return []

            # 确保数值类型
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

            # 移除无效坐标
            df = df.dropna(subset=['Latitude', 'Longitude'])

            self.logger.info(f"找到 {len(df)} 个有效站点")
            return df.to_dict('records')

        except Exception as e:
            self.logger.exception(f"获取站点失败: {str(e)}")
            return []

    def get_stations(self):
        query = "SELECT station_ID,Longitude,Latitude from stations"

        # 读取数据时显式指定数据类型
        df = pd.read_sql_query(query, self.db_conn, dtype={
            'station_ID': str,
            'Longitude': float,  # 强制转换为浮点数
            'Latitude': float  # 强制转换为浮点数
        })

        # 额外的类型检查
        if df['Latitude'].dtype != float or df['Longitude'].dtype != float:
            self.logger.warning("经纬度类型错误，尝试强制转换")
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

            # 过滤无效坐标
            df = df.dropna(subset=['Latitude', 'Longitude'])

        return df

    def process_range(self, start_date, end_date):
        """处理指定日期范围内的所有数据"""
        self.logger.info(f"开始处理 {start_date} 至 {end_date} 的数据")

        # 获取所有日期
        date_range = pd.date_range(start_date, end_date, freq='D')
        all_results = []

        for current_date in date_range:
            try:
                # 1. 获取当日文件路径
                file_path = self.get_file_path(current_date)
                if not Path(file_path).exists():
                    self.logger.warning(f"文件不存在: {file_path}")

                    continue
                else:
                    self.logger.info(f"开始处理: {file_path}")

                # 2. 提取数据
                daily_df = self.extract_values(current_date, file_path)
                if daily_df.empty:
                    self.logger.warning(f"提取到空数据：{file_path}")
                else:
                    self.logger.info(f"提取到{len(daily_df)}条记录")

                # 3. 数据后处理
                processed_df = self.postprocess_data(daily_df, current_date)

                if processed_df.empty:
                    self.logger.warning(
                        f"后处理过滤后为空  "
                        f"过滤条件：valid_min={self.conf['valid_min']}"
                    )
                if not processed_df.empty:
                    all_results.append(processed_df)

            except Exception as e:
                self.logger.error(f"{current_date} 处理失败: {str(e)}")
                continue

        # 4. 合并保存最终结果
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            output_path = self.save_results(final_df, start_date, end_date)
            self.logger.info(f"处理完成! 结果保存至: {output_path}")
            return output_path
        else:
            self.logger.warning("未生成有效结果")
            return None

    def get_file_path(self, date):
        """修复版：直接格式化字符串"""
        path_template = self.conf['path_template']
        return Path(path_template.format(
            year=date.strftime('%Y'),
            month=date.strftime('%m'),
            day=date.strftime('%d'),
            date=date.strftime('%Y%m%d')
        ))

    def postprocess_data(self, df, date):
        """数据后处理（修复SettingWithCopyWarning）"""
        if df.empty:
            return df

        # 创建副本以避免修改原始数据
        result_df = df.copy()

        # 过滤无效值
        valid_min = self.conf.get('valid_min', 0)
        fill_value = self.conf.get('fill_value', -9999)

        # 使用 .loc 进行条件过滤
        mask = (result_df['value'] >= valid_min) & (result_df['value'] != fill_value)
        result_df = result_df.loc[mask].copy()

        # 应用缩放因子（使用 .loc 确保修改副本）
        scale_factor = self.conf.get('scale_factor', 1.0)
        if scale_factor != 1.0:
            result_df.loc[:, 'value'] = result_df['value'] * scale_factor

        # 添加处理时间（使用 .loc 确保修改副本）
        processing_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_df.loc[:, 'processing_time'] = processing_time

        return result_df

    def save_results(self, df, start_date, end_date):
        """保存结果为多种格式"""
        try:
            base_filename = f"glsnow_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

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
                    return str(parquet_path)  # 返回主要格式的路径
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