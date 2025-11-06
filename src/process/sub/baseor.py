import abc
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import yaml

from src.process.utils.security import SecureProcessor


def _get_logger(name: str) -> logging.Logger:
    """获取日志记录器"""
    return logging.getLogger(name)


class BaseProcessor(metaclass=abc.ABCMeta):
    """
    基础数据处理器抽象基类
    提供统一的数据处理接口和通用功能
    """

    def __init__(self, config_path: str, output_dir: str,
                 secure_processor: Optional[SecureProcessor] = None):
        """
        初始化处理器

        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
            secure_processor: 安全处理器实例
        """
        self.config = self.load_config(config_path)
        self.output_dir = Path(output_dir)
        self.secure_processor = secure_processor or SecureProcessor()
        self.logger = _get_logger(self.__class__.__name__)

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化状态
        self.processed_count = 0
        self.failed_count = 0

        self.logger.info(f"初始化 {self.__class__.__name__}")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {config_path}, 错误: {e}")
            raise

    @abc.abstractmethod
    def process_date(self, date: datetime) -> Optional[str]:
        """
        处理单个日期的数据

        Args:
            date: 处理日期

        Returns:
            处理结果文件路径，如果失败返回None
        """
        pass

    def process_range(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        处理日期范围内的数据

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            处理统计信息
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            date_range = pd.date_range(start_dt, end_dt)
            self.logger.info(f"处理日期范围: {start_date} 到 {end_date}, 共 {len(date_range)} 天")

            results = []
            for date in date_range:
                try:
                    result_path = self.process_date(date)
                    if result_path:
                        results.append(result_path)
                        self.processed_count += 1
                    else:
                        self.failed_count += 1
                except Exception as e:
                    self.logger.error(f"处理日期 {date.strftime('%Y-%m-%d')} 失败: {e}")
                    self.failed_count += 1

            return {
                'total_dates': len(date_range),
                'processed': self.processed_count,
                'failed': self.failed_count,
                'success_rate': self.processed_count / len(date_range) if len(date_range) > 0 else 0,
                'results': results
            }

        except Exception as e:
            self.logger.error(f"处理日期范围失败: {e}")
            raise

    def process_all(self) -> Dict[str, Any]:
        """处理配置文件中定义的所有日期"""
        try:
            date_range = self.config.get('date_range', {})
            start_date = date_range.get('start')
            end_date = date_range.get('end')

            if not start_date or not end_date:
                raise ValueError("配置文件中缺少日期范围设置")

            return self.process_range(start_date, end_date)

        except Exception as e:
            self.logger.error(f"处理所有日期失败: {e}")
            raise

    def get_output_path(self, filename: str, format_type: str = 'parquet') -> Path:
        """
        获取输出文件路径

        Args:
            filename: 文件名
            format_type: 文件格式 (parquet/csv)

        Returns:
            完整输出路径
        """
        if format_type == 'parquet':
            return self.output_dir / f"{filename}.parquet"
        elif format_type == 'csv':
            return self.output_dir / f"{filename}.csv"
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")

    def save_result(self, df: pd.DataFrame, filename: str,
                   format_type: str = 'parquet') -> str:
        """
        保存处理结果

        Args:
            df: 数据框
            filename: 文件名
            format_type: 文件格式

        Returns:
            保存的文件路径
        """
        try:
            output_path = self.get_output_path(filename, format_type)

            if format_type == 'parquet':
                df.to_parquet(output_path, index=False)
            elif format_type == 'csv':
                df.to_csv(output_path, index=False)

            self.logger.info(f"结果已保存: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
            raise

    def close(self):
        """清理资源"""
        self.logger.info(f"处理器关闭: 成功 {self.processed_count}, 失败 {self.failed_count}")





































