# TODO:修改加载cswe配置方法，添加新方法

import yaml
import logging
from pathlib import Path
import os


class ConfigLoader:
    def __init__(self, config_dir="config"):
        """
        初始化配置加载器

        参数:
        config_dir: 配置目录路径
        """
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent.parent.parent
        self.config_dir = current_dir / config_dir

        # 确保配置目录存在
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            logging.warning(f"创建配置目录: {self.config_dir}")

        # 加载配置
        self.paths = self._load_config_file("paths.yaml")
        self.glsnow = self._preprocess_glsnow_config(
            self._load_config_file("sources/glsnow.yaml")
        )
        # self.gldas = self._load_config_file("sources/gldas.yaml")
        # self.modis = self._load_config_file("sources/modis.yaml")
        self.era5_swe = self._load_config_file("sources/era5_swe.yaml")
        self.cswe = self._load_config_file("sources/cswe.yaml")
        self.landcover = self._load_config_file("sources/landcover.yaml")
        self.snow_phenology = self._load_config_file("sources/scp.yaml")
        self.terrain_features = self._load_config_file("sources/terrain_features.yaml")
        self.snow_depth = self._load_config_file("sources/sd.yaml")
        self.era5_temperature = self._load_config_file("sources/era5_temperature.yaml")

        # 日志配置
        self.log_config = self._get_log_config()

    def _load_config_file(self, relative_path):
        """加载指定的配置文件"""
        config_path = self.config_dir / relative_path

        # 检查文件是否存在
        if not config_path.exists():
            logging.error(f"配置文件不存在: {config_path}")
            return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"加载配置文件 {config_path} 失败: {str(e)}")
            return {}

    def _preprocess_glsnow_config(self, config_data):
        """
        预处理GLSnow配置，生成完整的文件路径模板

        参数:
        config_data: 从YAML加载的原始配置
        """
        if not config_data:
            return {}

        # 1. 确保有数据根目录
        data_root = self.paths.get('input_root', 'data')

        # 2. 转换文件模式为路径模板
        file_pattern = config_data.get('file_pattern', '')

        # 替换括号为花括号并添加格式化占位符
        if '(' in file_pattern and ')' in file_pattern:
            # 替换模式: (YYYYMMDD) -> {date}
            formatted_pattern = file_pattern.replace('(', '{').replace(')', '}')

            # 添加其他可能的占位符
            formatted_pattern = formatted_pattern.replace('{year}', '{year}') \
                .replace('{month}', '{month}') \
                .replace('{day}', '{day}')
        else:
            # 使用默认占位符
            formatted_pattern = file_pattern

        # 3. 构建完整路径模板
        config_data['path_template'] = str(
            Path(data_root) /
            config_data.get('data_dir', 'glsnow') /
            formatted_pattern
        )

        # 4. 设置默认值
        config_data.setdefault('valid_min', -9999)
        config_data.setdefault('scale_factor', 1.0)
        config_data.setdefault('time_var', 'time')
        config_data.setdefault('output_format', 'parquet')

        return config_data

    def _get_log_config(self):
        """获取日志配置文件路径"""
        # 从paths配置中获取日志配置路径
        log_config = self.paths.get('log_config', 'logging.conf')

        # 构建完整路径
        log_config_path = self.config_dir / log_config

        # 检查文件是否存在
        if not log_config_path.exists():
            logging.warning(f"日志配置文件不存在: {log_config_path}")
            return None

        return str(log_config_path)

    def get_station_db_path(self):
        """获取站点数据库路径"""
        # 从paths配置中获取数据库路径
        db_path = self.paths.get('station_db', 'data/stations.db')

        # 构建完整路径
        return str(Path(__file__).parent.parent / db_path)

    def get_output_dir(self, source):
        """获取指定数据源的输出目录"""
        # 从paths配置中获取输出目录
        output_base = self.paths.get('outputs', {})
        output_path = output_base.get(source, f"outputs/{source}")

        # 构建完整路径
        full_path = Path(__file__).parent.parent / output_path

        # 确保目录存在
        full_path.mkdir(parents=True, exist_ok=True)

        return str(full_path)

    def get_encryption_key(self):
        """获取加密密钥路径"""
        # 从paths配置中获取密钥路径
        key_path = self.paths.get('encryption_key', 'secret.key')

        # 构建完整路径
        key_file = Path(__file__).parent.parent / key_path

        # 如果密钥文件不存在则创建
        if not key_file.exists():
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            logging.warning(f"已生成新的加密密钥: {key_file}")

        return str(key_file)


# 全局配置实例
config = ConfigLoader()