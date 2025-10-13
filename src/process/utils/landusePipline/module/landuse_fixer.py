import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LandUseFixer")


class LandUseFixer:
    def __init__(self):
        self.logger = logging.getLogger("LandUseFixer")

    def fix_landuse_data(self, input_file, output_file=None):
        """
        修复土地利用数据

        Args:
            input_file: 输入的Excel文件路径
            output_file: 输出的Excel文件路径，如果为None则自动生成

        Returns:
            str: 输出文件路径
        """
        try:
            self.logger.info(f"开始修复土地利用数据: {input_file}")

            # 读取Excel文件
            df = pd.read_excel(input_file)
            original_count = len(df)
            self.logger.info(f"读取数据: {original_count} 行, {len(df.columns)} 列")
            self.logger.info(f"原始列: {list(df.columns)}")

            # 检查必要的列
            required_cols = ['station_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"缺少必要列: {missing_cols}")
                return None

            # 步骤1: 修复landuse数据（先修复值）
            df = self._fix_landuse_values(df)

            # 步骤2: 删除指定列
            df = self._remove_columns(df)

            # 步骤3: 清理错误日期记录（最后删除记录）
            df = self._clean_bad_dates(df)

            # 生成输出文件名
            if output_file is None:
                input_path = Path(input_file)
                output_file = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"

            # 保存修复后的文件
            df.to_excel(output_file, index=False)
            self.logger.info(f"修复完成，结果保存至: {output_file}")
            self.logger.info(f"最终数据: {len(df)} 行 (原始: {original_count} 行), {len(df.columns)} 列")
            self.logger.info(f"最终列: {list(df.columns)}")

            return str(output_file)

        except Exception as e:
            self.logger.error(f"修复失败: {str(e)}")
            return None

    def _fix_landuse_values(self, df):
        """修复landuse数据值 - 第一步执行"""
        self.logger.info("=== 第一步: 修复landuse数据 ===")

        # 查找landuse相关的列
        landuse_cols = [col for col in df.columns if 'landuse' in col.lower()]
        if not landuse_cols:
            self.logger.warning("没有找到landuse相关的列")
            return df

        self.logger.info(f"找到landuse列: {landuse_cols}")

        # 修复前的统计
        self._log_before_fix(df, landuse_cols)

        # 为每个landuse列进行修复
        for landuse_col in landuse_cols:
            df = self._fix_single_landuse_column(df, landuse_col)

        # 修复后的统计
        self._log_after_fix(df, landuse_cols)

        return df

    def _fix_single_landuse_column(self, df, landuse_col):
        """修复单个landuse列"""
        self.logger.info(f"修复列: {landuse_col}")

        # 找出每个station_id的landuse值（取第一个非空值）
        station_landuse_map = {}

        for station_id in df['station_id'].unique():
            station_data = df[df['station_id'] == station_id]
            # 找到该站点的第一个非空landuse值
            valid_values = station_data[landuse_col].dropna()
            if not valid_values.empty:
                landuse_value = valid_values.iloc[0]
                station_landuse_map[station_id] = landuse_value
                self.logger.debug(f"站点 {station_id}: {landuse_col} = {landuse_value}")
            else:
                self.logger.warning(f"站点 {station_id}: 没有找到{landuse_col}值")

        # 应用修复：为每个站点的所有记录设置相同的landuse值
        fixed_count = 0
        for station_id, landuse_value in station_landuse_map.items():
            # 找出该站点landuse值为空的记录
            mask = (df['station_id'] == station_id) & (df[landuse_col].isna())
            empty_count = mask.sum()

            if empty_count > 0:
                df.loc[mask, landuse_col] = landuse_value
                fixed_count += empty_count
                self.logger.debug(f"为站点 {station_id} 的 {empty_count} 条记录设置{landuse_col}值: {landuse_value}")

        self.logger.info(f"列 {landuse_col}: 修复了 {fixed_count} 条记录")
        return df

    def _remove_columns(self, df):
        """删除指定列 - 第二步执行"""
        self.logger.info("=== 第二步: 删除指定列 ===")

        # 要删除的列
        columns_to_remove = [
            'landuse_class',
            'longitude_x',
            'latitude_x'
        ]

        # 检查哪些列存在
        existing_columns = [col for col in columns_to_remove if col in df.columns]
        non_existing_columns = [col for col in columns_to_remove if col not in df.columns]

        if existing_columns:
            self.logger.info(f"删除列: {existing_columns}")
            df = df.drop(columns=existing_columns)

        if non_existing_columns:
            self.logger.info(f"以下列不存在，跳过删除: {non_existing_columns}")

        return df

    def _clean_bad_dates(self, df):
        """清理错误日期记录 - 第三步执行"""
        self.logger.info("=== 第三步: 清理错误日期记录 ===")

        if 'date' not in df.columns:
            self.logger.warning("没有找到date列，跳过日期清理")
            return df

        # 记录清理前的状态
        before_count = len(df)
        self.logger.info(f"清理前记录数: {before_count}")

        # 定义错误日期模式
        bad_date_patterns = [
            '2025-10-11 00:00:00',
            '2025-10-11',
            '2025-10-10 00:00:00',  # 可能的前一天日期
            '2025-10-10'
        ]

        # 找出错误日期记录
        bad_date_mask = df['date'].isin(bad_date_patterns)
        bad_date_count = bad_date_mask.sum()

        if bad_date_count > 0:
            self.logger.info(f"发现 {bad_date_count} 条错误日期记录")

            # 显示错误日期的样本
            bad_samples = df[bad_date_mask][['station_id', 'date']].head(5)
            self.logger.info("错误日期记录样本:")
            for _, row in bad_samples.iterrows():
                self.logger.info(f"  站点 {row['station_id']}: 日期 {row['date']}")

            # 删除错误日期记录
            df = df[~bad_date_mask]
            after_count = len(df)
            removed_count = before_count - after_count

            self.logger.info(f"删除 {removed_count} 条错误日期记录")
            self.logger.info(f"清理后记录数: {after_count}")
        else:
            self.logger.info("没有发现错误日期记录")

        return df

    def _log_before_fix(self, df, landuse_cols):
        """记录修复前的统计信息"""
        self.logger.info("=== 修复前统计 ===")
        self.logger.info(f"总记录数: {len(df)}")
        self.logger.info(f"总站点数: {df['station_id'].nunique()}")

        for landuse_col in landuse_cols:
            filled_count = df[landuse_col].notna().sum()
            fill_rate = (filled_count / len(df)) * 100
            self.logger.info(f"{landuse_col}: {filled_count}/{len(df)} 有值 ({fill_rate:.1f}%)")

            # 统计每个站点的landuse覆盖情况
            station_coverage = df.groupby('station_id').apply(
                lambda x: x[landuse_col].notna().any()
            )
            stations_with_data = station_coverage.sum()
            total_stations = len(station_coverage)
            station_coverage_rate = (stations_with_data / total_stations) * 100
            self.logger.info(f"  {stations_with_data}/{total_stations} 个站点有landuse数据 ({station_coverage_rate:.1f}%)")

    def _log_after_fix(self, df, landuse_cols):
        """记录修复后的统计信息"""
        self.logger.info("=== 修复后统计 ===")

        for landuse_col in landuse_cols:
            filled_count = df[landuse_col].notna().sum()
            fill_rate = (filled_count / len(df)) * 100
            self.logger.info(f"{landuse_col}: {filled_count}/{len(df)} 有值 ({fill_rate:.1f}%)")

            # 检查每个站点的landuse值是否一致
            station_consistency = df.groupby('station_id')[landuse_col].apply(
                lambda x: x.nunique() == 1 if x.notna().any() else True
            )
            consistent_stations = station_consistency.sum()
            total_stations = len(station_consistency)
            consistency_rate = (consistent_stations / total_stations) * 100
            self.logger.info(f"  {consistent_stations}/{total_stations} 个站点的landuse值一致 ({consistency_rate:.1f}%)")

    def analyze_data(self, input_file):
        """分析数据，不进行修复"""
        try:
            df = pd.read_excel(input_file)
            self.logger.info(f"数据分析: {input_file}")
            self.logger.info(f"总记录数: {len(df)}")
            self.logger.info(f"总站点数: {df['station_id'].nunique()}")
            self.logger.info(f"所有列: {list(df.columns)}")

            # 检查要删除的列
            columns_to_check = ['landuse_class', 'longitude_x', 'latitude_x']
            existing_columns = [col for col in columns_to_check if col in df.columns]
            if existing_columns:
                self.logger.info(f"将删除的列: {existing_columns}")

            # 检查错误日期
            if 'date' in df.columns:
                bad_dates = ['2025-10-11 00:00:00', '2025-10-11']
                bad_date_count = df['date'].isin(bad_dates).sum()
                if bad_date_count > 0:
                    self.logger.warning(f"发现 {bad_date_count} 条错误日期记录")
                    bad_samples = df[df['date'].isin(bad_dates)][['station_id', 'date']].head(3)
                    self.logger.info("错误日期样本:")
                    for _, row in bad_samples.iterrows():
                        self.logger.info(f"  站点 {row['station_id']}: {row['date']}")
                else:
                    self.logger.info("没有发现错误日期记录")

            # 分析landuse列
            landuse_cols = [col for col in df.columns if 'landuse' in col.lower()]
            if not landuse_cols:
                self.logger.warning("没有找到landuse相关的列")
                return

            for landuse_col in landuse_cols:
                self._analyze_landuse_column(df, landuse_col)

        except Exception as e:
            self.logger.error(f"分析失败: {str(e)}")

    def _analyze_landuse_column(self, df, landuse_col):
        """分析单个landuse列"""
        self.logger.info(f"\n分析列: {landuse_col}")

        # 基本统计
        filled_count = df[landuse_col].notna().sum()
        fill_rate = (filled_count / len(df)) * 100
        self.logger.info(f"填充率: {filled_count}/{len(df)} ({fill_rate:.1f}%)")

        # 站点覆盖统计
        station_coverage = df.groupby('station_id').apply(
            lambda x: x[landuse_col].notna().any()
        )
        stations_with_data = station_coverage.sum()
        total_stations = len(station_coverage)
        station_coverage_rate = (stations_with_data / total_stations) * 100
        self.logger.info(f"有数据的站点: {stations_with_data}/{total_stations} ({station_coverage_rate:.1f}%)")

        # 检查每个站点的landuse值一致性
        inconsistent_stations = []
        for station_id in df['station_id'].unique():
            station_data = df[df['station_id'] == station_id]
            landuse_values = station_data[landuse_col].dropna().unique()
            if len(landuse_values) > 1:
                inconsistent_stations.append((station_id, landuse_values))

        if inconsistent_stations:
            self.logger.warning(f"发现 {len(inconsistent_stations)} 个站点的landuse值不一致")
            for station_id, values in inconsistent_stations[:3]:  # 只显示前3个
                self.logger.warning(f"  站点 {station_id}: {values}")
        else:
            self.logger.info("所有站点的landuse值一致")

        # 显示landuse值分布
        if filled_count > 0:
            value_counts = df[landuse_col].value_counts().head(10)
            self.logger.info("前10个landuse值分布:")
            for value, count in value_counts.items():
                self.logger.info(f"  {value}: {count} 条记录")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='土地利用数据修复工具')
    parser.add_argument('input_file', help='输入的Excel文件路径')
    parser.add_argument('-o', '--output', help='输出的Excel文件路径')
    parser.add_argument('-a', '--analyze', action='store_true', help='只分析数据，不进行修复')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细日志')

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    fixer = LandUseFixer()

    if args.analyze:
        # 只分析数据
        fixer.analyze_data(args.input_file)
    else:
        # 修复数据
        result = fixer.fix_landuse_data(args.input_file, args.output)
        if result:
            print(f"✅ 修复完成: {result}")
        else:
            print("❌ 修复失败")
            sys.exit(1)


if __name__ == "__main__":
    main()