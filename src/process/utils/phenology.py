import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import os

logger = logging.getLogger("PhenologyUtils")


class AstronomicalSeasonCalculator:
    """天文季节计算器 - 基于实际天文日期，包含Excel输入输出"""

    def __init__(self):
        """初始化天文季节计算器"""
        self.logger = logger

        # 2013-2017年实际的二分二至日日期（近似值）
        self.actual_equinox_solstice_dates = {
            2012: {
                'autumn_equinox': '2012-09-22',  # 秋分
                'winter_solstice': '2012-12-21',  # 冬至
                'spring_equinox': '2013-03-20',  # 春分（属于2014水文年）
                'summer_solstice': '2013-06-21'  # 夏至（属于2014水文年）
            },
            2013: {
                'autumn_equinox': '2013-09-22',  # 秋分
                'winter_solstice': '2013-12-21',  # 冬至
                'spring_equinox': '2014-03-20',  # 春分（属于2014水文年）
                'summer_solstice': '2014-06-21'  # 夏至（属于2014水文年）
            },
            2014: {
                'autumn_equinox': '2014-09-23',
                'winter_solstice': '2014-12-21',
                'spring_equinox': '2015-03-20',
                'summer_solstice': '2015-06-21'
            },
            2015: {
                'autumn_equinox': '2015-09-23',
                'winter_solstice': '2015-12-22',
                'spring_equinox': '2016-03-20',
                'summer_solstice': '2016-06-20'
            },
            2016: {
                'autumn_equinox': '2016-09-22',
                'winter_solstice': '2016-12-21',
                'spring_equinox': '2017-03-20',
                'summer_solstice': '2017-06-21'
            },
            2017: {
                'autumn_equinox': '2017-09-22',
                'winter_solstice': '2017-12-21',
                'spring_equinox': '2018-03-20',
                'summer_solstice': '2018-06-21'
            }
        }

    def _date_to_hydrological_doy(self, date_str: str, hydrological_year: int) -> int:
        """
        将日期转换为水文年DOY

        Args:
            date_str: 日期字符串 (YYYY-MM-DD)
            hydrological_year: 水文年

        Returns:
            int: 水文年DOY
        """
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')

            # 计算水文年开始日期
            hydro_start = datetime(hydrological_year - 1, 9, 1)

            # 计算天数差
            doy = (date - hydro_start).days + 1

            # 确保在有效范围内
            if doy < 1:
                doy += 365  # 属于上一个水文年
            elif doy > 366:
                doy -= 365  # 属于下一个水文年

            return doy

        except Exception as e:
            self.logger.error(f"日期转换失败 {date_str}: {e}")
            return np.nan

    def get_seasonal_doy_for_year(self, hydrological_year: int) -> Dict[str, int]:
        """
        获取指定水文年的二分二至日DOY

        Args:
            hydrological_year: 水文年

        Returns:
            dict: 季节名称到水文年DOY的映射
        """
        try:
            seasonal_doy = {}

            # 获取该水文年对应的天文日期
            year_data = self.actual_equinox_solstice_dates.get(hydrological_year, {})

            for season, date_str in year_data.items():
                doy = self._date_to_hydrological_doy(date_str, hydrological_year)
                if not np.isnan(doy):
                    seasonal_doy[season] = doy

            return seasonal_doy

        except Exception as e:
            self.logger.error(f"获取 {hydrological_year} 年季节DOY失败: {e}")
            return {}

    def calculate_inverse_distance(self, hydrological_doy: int, hydrological_year: int) -> Dict[str, float]:
        """
        计算DOY与指定水文年二分二至日的逆距离

        Args:
            hydrological_doy: 水文年DOY (1-366)
            hydrological_year: 水文年

        Returns:
            dict: 包含到各个季节点的逆距离
        """
        try:
            if pd.isna(hydrological_doy) or pd.isna(hydrological_year):
                return self._get_empty_distances()

            if hydrological_doy < 1 or hydrological_doy > 366:
                return self._get_empty_distances()

            # 获取该水文年的实际季节DOY
            seasonal_doy = self.get_seasonal_doy_for_year(hydrological_year)

            if not seasonal_doy:
                self.logger.warning(f"未找到 {hydrological_year} 水文年的季节数据")
                return self._get_empty_distances()

            distances = {}
            for season, season_doy in seasonal_doy.items():
                if pd.isna(season_doy):
                    distances[f'inverse_dist_{season}'] = np.nan
                    continue

                # 计算距离（考虑水文年循环）
                distance1 = abs(hydrological_doy - season_doy)
                distance2 = abs(hydrological_doy - (season_doy + 365))
                distance3 = abs(hydrological_doy - (season_doy - 365))
                distance = min(distance1, distance2, distance3)

                # 计算逆距离（避免除以0）
                if distance == 0:
                    inverse_distance = 1.0
                else:
                    inverse_distance = 1.0 / distance

                distances[f'inverse_dist_{season}'] = inverse_distance

            return distances

        except Exception as e:
            self.logger.error(f"计算逆距离失败: {e}")
            return self._get_empty_distances()

    def _get_empty_distances(self) -> Dict[str, float]:
        """返回空的距离字典"""
        return {
            'inverse_dist_autumn_equinox': np.nan,
            'inverse_dist_winter_solstice': np.nan,
            'inverse_dist_spring_equinox': np.nan,
            'inverse_dist_summer_solstice': np.nan
        }

    def add_seasonal_features(self, df: pd.DataFrame,
                              hydrological_doy_col: str = 'hydrological_doy',
                              hydrological_year_col: str = 'hydrological_year') -> pd.DataFrame:
        """
        为数据框添加基于实际天文日期的季节逆距离特征

        Args:
            df: 包含水文年DOY和水文年的数据框
            hydrological_doy_col: 水文年DOY列名
            hydrological_year_col: 水文年列名

        Returns:
            DataFrame: 添加了季节特征的数据框
        """
        try:
            if hydrological_doy_col not in df.columns:
                self.logger.warning(f"数据框中没有 {hydrological_doy_col} 列")
                return df

            if hydrological_year_col not in df.columns:
                self.logger.warning(f"数据框中没有 {hydrological_year_col} 列")
                return df

            df_processed = df.copy()

            # 为每条记录计算季节逆距离
            seasonal_features = df_processed.apply(
                lambda row: self.calculate_inverse_distance(
                    row[hydrological_doy_col],
                    row[hydrological_year_col]
                ),
                axis=1
            )

            # 将字典列展开为多个列
            seasonal_df = pd.DataFrame(seasonal_features.tolist(), index=df_processed.index)

            # 合并到原数据框
            df_processed = pd.concat([df_processed, seasonal_df], axis=1)

            # 统计特征添加情况
            valid_count = df_processed[hydrological_doy_col].notna().sum()
            self.logger.info(f"✅ 天文季节特征添加完成: {valid_count} 条记录")

            # 显示特征统计
            for col in seasonal_df.columns:
                if col in df_processed.columns:
                    valid_values = df_processed[col].notna().sum()
                    mean_value = df_processed[col].mean()
                    self.logger.debug(f"  {col}: {valid_values} 有效值, 均值: {mean_value:.4f}")

            return df_processed

        except Exception as e:
            self.logger.error(f"添加天文季节特征失败: {e}")
            return df

    # ==================== Excel 输入输出方法 ====================

    def read_data_from_excel(self, file_path: Union[str, Path],
                             sheet_name: str = 0,
                             required_columns: List[str] = None) -> pd.DataFrame:
        """
        从Excel文件读取数据

        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称
            required_columns: 必需的列名列表

        Returns:
            DataFrame: 读取的数据
        """
        try:
            file_path = r"E:\data\gisws\final_results.csv"

            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            self.logger.info(f"正在读取Excel文件: {file_path}")

            # 读取Excel文件
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            self.logger.info(f"成功读取数据: {len(df)} 行, {len(df.columns)} 列")
            self.logger.info(f"数据列: {list(df.columns)}")

            # 检查必需列
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"缺少必需列: {missing_columns}")

            return df

        except Exception as e:
            self.logger.error(f"读取Excel文件失败: {e}")
            raise

    def save_data_to_excel(self, df: pd.DataFrame,
                           output_path: Union[str, Path],
                           sheet_name: str = 'seasonal_features',
                           include_summary: bool = True):
        """
        保存数据到Excel文件

        Args:
            df: 要保存的数据框
            output_path: 输出文件路径
            sheet_name: 工作表名称
            include_summary: 是否包含统计摘要
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"正在保存数据到Excel: {output_path}")

            # 创建Excel写入器
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 保存主数据
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                # 保存统计摘要
                if include_summary:
                    self._save_summary_sheet(df, writer)

                # 保存季节DOY参考表
                self._save_seasonal_reference_sheet(writer)

            self.logger.info(f"✅ 数据保存成功: {output_path}")

        except Exception as e:
            self.logger.error(f"保存Excel文件失败: {e}")
            raise

    def _save_summary_sheet(self, df: pd.DataFrame, writer: pd.ExcelWriter):
        """保存统计摘要工作表"""
        try:
            summary_data = []

            # 基本统计
            summary_data.append(['基本统计', ''])
            summary_data.append(['总记录数', len(df)])
            summary_data.append(['有效水文年DOY记录', df['hydrological_doy'].notna().sum()])
            summary_data.append(['有效水文年记录', df['hydrological_year'].notna().sum()])
            summary_data.append(['', ''])

            # 季节特征统计
            summary_data.append(['季节特征统计', ''])
            seasonal_cols = [col for col in df.columns if col.startswith('inverse_dist_')]
            for col in seasonal_cols:
                valid_count = df[col].notna().sum()
                mean_val = df[col].mean()
                std_val = df[col].std()
                summary_data.append([col, f"有效值: {valid_count}, 均值: {mean_val:.4f}, 标准差: {std_val:.4f}"])

            # 转换为DataFrame并保存
            summary_df = pd.DataFrame(summary_data, columns=['项目', '数值'])
            summary_df.to_excel(writer, sheet_name='统计摘要', index=False)

        except Exception as e:
            self.logger.warning(f"保存统计摘要失败: {e}")

    def _save_seasonal_reference_sheet(self, writer: pd.ExcelWriter):
        """保存季节DOY参考表"""
        try:
            reference_data = []
            reference_data.append(['水文年', '季节', '实际日期', '水文年DOY'])

            for hydro_year in sorted(self.actual_equinox_solstice_dates.keys()):
                seasonal_doy = self.get_seasonal_doy_for_year(hydro_year)
                for season, doy in seasonal_doy.items():
                    actual_date = self.actual_equinox_solstice_dates[hydro_year][season]
                    reference_data.append([hydro_year, season, actual_date, doy])

            reference_df = pd.DataFrame(reference_data[1:], columns=reference_data[0])
            reference_df.to_excel(writer, sheet_name='季节DOY参考', index=False)

        except Exception as e:
            self.logger.warning(f"保存季节参考表失败: {e}")

    def process_excel_file(self, input_file: Union[str, Path],
                           output_file: Union[str, Path] = None,
                           hydrological_doy_col: str = 'hydrological_doy',
                           hydrological_year_col: str = 'hydrological_year') -> str:
        """
        处理Excel文件的完整流程

        Args:
            input_file: 输入Excel文件路径
            output_file: 输出Excel文件路径（如为None则自动生成）
            hydrological_doy_col: 水文年DOY列名
            hydrological_year_col: 水文年列名

        Returns:
            str: 输出文件路径
        """
        try:
            # 读取数据
            required_cols = [hydrological_doy_col, hydrological_year_col]
            df = self.read_data_from_excel(input_file, required_columns=required_cols)

            # 添加季节特征
            df_with_features = self.add_seasonal_features(
                df, hydrological_doy_col, hydrological_year_col
            )

            # 生成输出文件名
            if output_file is None:
                input_path = Path(input_file)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = input_path.parent / f"{input_path.stem}_with_seasonal_{timestamp}.xlsx"

            # 保存结果
            self.save_data_to_excel(df_with_features, output_file)

            return str(output_file)

        except Exception as e:
            self.logger.error(f"处理Excel文件失败: {e}")
            raise


# 便捷函数
def process_excel_with_seasonal_features(input_file: Union[str, Path],
                                         output_file: Union[str, Path] = None,
                                         hydrological_doy_col: str = 'hydrological_doy',
                                         hydrological_year_col: str = 'hydrological_year') -> str:
    """
    便捷函数：处理Excel文件并添加季节特征

    Args:
        input_file: 输入Excel文件路径
        output_file: 输出Excel文件路径
        hydrological_doy_col: 水文年DOY列名
        hydrological_year_col: 水文年列名

    Returns:
        str: 输出文件路径
    """
    calculator = AstronomicalSeasonCalculator()
    return calculator.process_excel_file(input_file, output_file, hydrological_doy_col, hydrological_year_col)


def batch_process_excel_files(input_dir: Union[str, Path],
                              output_dir: Union[str, Path] = None,
                              file_pattern: str = "*.xlsx") -> List[str]:
    """
    批量处理Excel文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        file_pattern: 文件匹配模式

    Returns:
        List[str]: 处理成功的文件路径列表
    """
    calculator = AstronomicalSeasonCalculator()
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir / "processed"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_files = []
    excel_files = list(input_dir.glob(file_pattern))

    logger.info(f"找到 {len(excel_files)} 个Excel文件")

    for input_file in excel_files:
        try:
            output_file = output_dir / f"{input_file.stem}_with_seasonal.xlsx"
            result_file = calculator.process_excel_file(input_file, output_file)
            processed_files.append(result_file)
            logger.info(f"✅ 处理完成: {input_file.name} -> {output_file.name}")
        except Exception as e:
            logger.error(f"❌ 处理失败 {input_file.name}: {e}")

    return processed_files


def test_excel_processing():
    """测试Excel处理功能"""
    calculator = AstronomicalSeasonCalculator()

    # 创建测试数据
    test_data = pd.DataFrame({
        'station_id': [f'ST{i:03d}' for i in range(1, 11)],
        'hydrological_year': [2014, 2014, 2015, 2015, 2016, 2016, 2017, 2017, 2014, 2015],
        'hydrological_doy': [1, 100, 200, 300, 50, 150, 250, 350, 91, 182],
        'swe': np.random.uniform(10, 100, 10)
    })

    # 创建测试目录
    test_dir = Path('./test_output')
    test_dir.mkdir(exist_ok=True)

    # 保存测试数据
    test_input_file = test_dir / 'test_input_data.xlsx'
    test_data.to_excel(test_input_file, index=False)
    print(f"✅ 创建测试文件: {test_input_file}")

    # 处理测试文件
    try:
        output_file = calculator.process_excel_file(test_input_file)
        print(f"✅ 处理完成: {output_file}")

        # 读取并显示结果
        result_df = pd.read_excel(output_file, sheet_name='seasonal_features')
        print(f"\n处理结果预览:")
        print(result_df[['station_id', 'hydrological_year', 'hydrological_doy',
                         'inverse_dist_autumn_equinox', 'inverse_dist_winter_solstice']].head())

    except Exception as e:
        print(f"❌ 处理失败: {e}")


def main():
    """主函数 - 提供命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description='水文年季节特征计算工具')
    parser.add_argument('--input', '-i', required=True, help='输入Excel文件路径')
    parser.add_argument('--output', '-o', help='输出Excel文件路径')
    parser.add_argument('--doy-col', default='hydrological_doy', help='水文年DOY列名')
    parser.add_argument('--year-col', default='hydrological_year', help='水文年列名')
    parser.add_argument('--batch', action='store_true', help='批量处理目录中的所有Excel文件')
    parser.add_argument('--input-dir', help='输入目录（批量处理时使用）')
    parser.add_argument('--output-dir', help='输出目录（批量处理时使用）')

    args = parser.parse_args()

    if args.batch:
        if not args.input_dir:
            print("❌ 批量处理需要指定 --input-dir")
            return

        print("开始批量处理Excel文件...")
        processed_files = batch_process_excel_files(
            args.input_dir,
            args.output_dir
        )
        print(f"✅ 批量处理完成: {len(processed_files)} 个文件")

    else:
        print(f"处理单个文件: {args.input}")
        try:
            output_file = process_excel_with_seasonal_features(
                args.input,
                args.output,
                args.doy_col,
                args.year_col
            )
            print(f"✅ 处理完成: {output_file}")
        except Exception as e:
            print(f"❌ 处理失败: {e}")


if __name__ == "__main__":
    # 运行测试
    print("测试Excel处理功能...")
    test_excel_processing()

    # 如果要使用命令行，取消下面的注释
    # import sys
    # if len(sys.argv) > 1:
    #     main()
    # else:
    #     test_excel_processing()
