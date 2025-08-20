import pandas as pd
import os
import logging
from pathlib import Path
import tempfile


class DataIntegrator:
    def __init__(self, output_dir, secure_processor=None):
        """
        初始化数据整合器

        参数:
        output_dir: 输出目录路径
        secure_processor: 安全处理器实例 (可选)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.secure_processor = secure_processor
        self.logger = logging.getLogger("DataIntegrator")
        self.source_data = {}  # 存储各数据源数据 {source_name: DataFrame}

    def add_source(self, name, file_path):
        """添加数据源（修复Parquet文件读取）"""
        try:
            file_path = Path(file_path)

            # 根据文件扩展名选择正确的读取方式
            if file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.feather':
                df = pd.read_feather(file_path)
            else:
                self.logger.warning(f"不支持的文件格式: {file_path.suffix}")
                return

            # 添加元数据
            df['source'] = name
            df['file_path'] = str(file_path)

            self.source_data[name].append(df)
            self.logger.info(f"成功添加数据源: {name}, 记录数: {len(df)}")

        except Exception as e:
            self.logger.error(f"添加数据源 {name} 失败: {str(e)}")

    def _process_file(self, source_name, file_path):
        """处理单个数据文件"""
        try:
            # 处理加密文件
            if self.secure_processor and file_path.endswith('.enc'):
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                    decrypted_path = tmp.name

                # 解密文件
                self.secure_processor.decrypt_file(file_path, decrypted_path)
                df = pd.read_csv(decrypted_path)
                os.remove(decrypted_path)  # 清理临时文件
            else:
                # 直接读取CSV文件
                df = pd.read_csv(file_path)

            # 添加到数据源
            if source_name not in self.source_data:
                self.source_data[source_name] = df
            else:
                self.source_data[source_name] = pd.concat(
                    [self.source_data[source_name], df], ignore_index=True
                )
        except Exception as e:
            self.logger.error(f"处理文件 {file_path} 失败: {str(e)}")

    def save_master_excel(self):
        """保存主Excel文件（修复文件处理）"""
        if not self.source_data:
            self.logger.warning("没有数据可保存")
            return None

        try:
            # 合并所有数据
            combined_df = pd.concat(self.source_data, ignore_index=True)

            # 保存为Excel
            output_path = self.output_dir / "combined_data.xlsx"
            combined_df.to_excel(output_path, index=False)

            self.logger.info(f"主Excel文件已保存: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"保存Excel文件失败: {str(e)}")
            return None

    def get_combined_data(self):
        """获取整合后的数据DataFrame"""
        if not self.source_data:
            return pd.DataFrame()
        return pd.concat(self.source_data.values(), ignore_index=True)

    def generate_report(self):
        pass




