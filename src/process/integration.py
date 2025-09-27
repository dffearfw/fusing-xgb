
import pandas as pd

import logging
from pathlib import Path
from datetime import datetime


class DataIntegrator:
    def __init__(self, output_dir, secure_processor=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("DataIntegrator")
        self.source_data = {}

    def add_source(self, name, file_path):
        """添加数据源"""
        try:
            path = Path(file_path)
            if path.suffix.lower() == '.parquet':
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)

            self.source_data[name] = df
            self.logger.info(f"添加 {name}: {len(df)} 行")
            return True
        except Exception as e:
            self.logger.error(f"添加 {name} 失败: {e}")
            return False

    def save_master_excel(self, format_type='wide'):
        """保存主Excel文件 - 最终正确版本"""
        if not self.source_data:
            return None

        try:
            if format_type == 'wide':
                final_df = self._create_correct_wide_table()
            else:
                final_df = self.get_combined_data()

            if final_df.empty:
                self.logger.error("最终数据为空")
                return None

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"correct_wide_table_{timestamp}.xlsx"

            final_df.to_excel(output_path, index=False)
            self.logger.info(f"✅ 正确宽表保存成功: {output_path}")

            # 在这里调用全面验证
            self._comprehensive_validation(final_df)

            return str(output_path)

        except Exception as e:
            self.logger.error(f"保存失败: {e}")
            return None

    def _create_correct_wide_table(self):
        """创建正确的宽表 - 基于您展示的数据结构"""
        if not self.source_data:
            return pd.DataFrame()

        self.logger.info("创建正确宽表...")

        try:
            # 1. 为每个数据源准备数据
            wide_dfs = []

            for source_name, source_df in self.source_data.items():
                # 找到数值列
                numeric_cols = source_df.select_dtypes(include=['number']).columns
                value_cols = [col for col in numeric_cols if col not in ['station_id', 'date', 'year', 'month', 'day']]

                if not value_cols:
                    self.logger.warning(f"数据源 {source_name} 没有找到数值列")
                    continue

                # 使用第一个数值列
                value_col = value_cols[0]

                # 准备数据
                if 'station_id' in source_df.columns and 'date' in source_df.columns:
                    source_wide = source_df[['station_id', 'date', value_col]].copy()
                    source_wide = source_wide.rename(columns={value_col: source_name})
                    source_wide = source_wide.drop_duplicates(['station_id', 'date'])
                    wide_dfs.append(source_wide)
                    self.logger.info(f"准备 {source_name}: {len(source_wide)} 行")
                else:
                    self.logger.warning(f"数据源 {source_name} 缺少station_id或date列")

            if not wide_dfs:
                self.logger.error("没有可用的数据源")
                return pd.DataFrame()

            # 2. 逐个合并数据源
            final_wide = wide_dfs[0]  # 从第一个数据源开始

            for i in range(1, len(wide_dfs)):
                next_df = wide_dfs[i]
                final_wide = final_wide.merge(next_df, on=['station_id', 'date'], how='outer')
                self.logger.info(f"合并后: {final_wide.shape}")

            # 3. 按站点和时间排序
            final_wide = final_wide.sort_values(['station_id', 'date']).reset_index(drop=True)

            self.logger.info(f"✅ 宽表创建完成: {final_wide.shape}")
            return final_wide

        except Exception as e:
            self.logger.error(f"创建宽表失败: {e}")
            return pd.DataFrame()

    def _validate_final_result(self, df):
        """验证最终结果"""
        self.logger.info("=== 最终结果验证 ===")
        self.logger.info(f"总行数: {len(df)}")
        self.logger.info(f"列名: {list(df.columns)}")

        # 检查各数据源的填充情况
        data_cols = [col for col in df.columns if col not in ['station_id', 'date']]
        for col in data_cols:
            non_null = df[col].notna().sum()
            self.logger.info(f"{col}: {non_null} 行有值")

        # 显示样例数据
        if len(df) > 0:
            self.logger.info("前5行数据样例:")
            for i in range(min(5, len(df))):
                row = df.iloc[i]
                values = []
                for col in data_cols:
                    if pd.notna(row[col]):
                        values.append(f"{col}: {row[col]}")

                if values:
                    self.logger.info(f"  站点 {row['station_id']} | 时间 {row['date']} | {', '.join(values)}")

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

    def emergency_fix(self, issue_type='duplicates'):
        """应急数据修复"""
        if issue_type == 'duplicates':
            # 处理重复数据
            for name, df in self.source_data.items():
                if 'station_id' in df.columns and 'date' in df.columns:
                    self.source_data[name] = df.drop_duplicates(['station_id', 'date'])

        elif issue_type == 'null_values':
            # 处理空值
            pass

    def generate_report(self):
        return "报告生成完成"

    def _comprehensive_validation(self, df):
        """全面验证宽表数据"""
        self.logger.info("=== 全面数据验证 ===")

        # 1. 基本统计
        self.logger.info(f"表格形状: {df.shape}")
        self.logger.info(f"唯一站点数: {df['station_id'].nunique()}")
        self.logger.info(f"时间点数: {df['date'].nunique()}")
        self.logger.info(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")

        # 2. 数据完整性检查
        data_cols = [col for col in df.columns if col not in ['station_id', 'date']]
        self.logger.info("=== 各数据源填充情况 ===")

        for col in data_cols:
            non_null = df[col].notna().sum()
            fill_rate = (non_null / len(df)) * 100
            if non_null > 0:
                avg_val = df[col].mean()
                min_val = df[col].min()
                max_val = df[col].max()
                self.logger.info(
                    f"{col}: {non_null}行有值({fill_rate:.1f}%), 范围: {min_val:.2f}-{max_val:.2f}, 平均: {avg_val:.2f}")
            else:
                self.logger.info(f"{col}: 0行有值")

        # 3. 检查重复行
        duplicates = df.duplicated(['station_id', 'date']).sum()
        self.logger.info(f"重复行数: {duplicates}")

        # 4. 抽样显示具体数据
        self.logger.info("=== 数据抽样显示 ===")

        # 显示前5行
        self.logger.info("前5行数据:")
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            values = []
            for col in data_cols:
                if pd.notna(row[col]):
                    values.append(f"{col}: {row[col]}")
            self.logger.info(f"  行{i + 1}: 站点{row['station_id']}, 时间{row['date']}")
            if values:
                self.logger.info(f"      数据: {', '.join(values)}")

        # 5. 检查特定已知数据点
        self.logger.info("=== 已知数据点验证 ===")
        known_points = [
            (50136, '2013-01-05', 'snow_depth', 14.1),
            (54764, '2013-01-02', 'ERA5温度', 2.67)
        ]

        for station, date, data_type, expected_value in known_points:
            match = df[(df['station_id'] == station) & (df['date'] == date)]
            if not match.empty:
                actual_value = match.iloc[0].get(data_type)
                if pd.notna(actual_value):
                    diff = abs(actual_value - expected_value)
                    status = "✓" if diff < 0.1 else "✗"
                    self.logger.info(
                        f"{status} 站点{station} {date} {data_type}: 期望{expected_value}, 实际{actual_value:.2f}")
                else:
                    self.logger.info(f"✗ 站点{station} {date} {data_type}: 期望{expected_value}, 实际无数据")
            else:
                self.logger.info(f"✗ 站点{station} {date}: 无记录")

    def clear_data(self):
        self.source_data.clear()


