import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger("StableDataPipeline")


class StableDataPipeline:
    """
    稳定数据管道 - 精确计算二分二至日
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        self.reference_df: Optional[pd.DataFrame] = None
        self.integrated_df: Optional[pd.DataFrame] = None
        self.corrected_df: Optional[pd.DataFrame] = None

        self.config = {
            'merge_keys': ['station_id', 'date'],
            'protected_columns': ['scp_start', 'scp_end'],
            'backup_original': True
        }

        # 2012-2018年精确的二分二至日日期
        self.solstice_equinox_dates = {
            2012: {
                'autumn_equinox': '2012-09-22',
                'winter_solstice': '2012-12-21',
                'spring_equinox': '2013-03-20',
                'summer_solstice': '2013-06-21'
            },
            2013: {
                'autumn_equinox': '2013-09-22',
                'winter_solstice': '2013-12-21',
                'spring_equinox': '2014-03-20',
                'summer_solstice': '2014-06-21'
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
            },
            2018: {
                'autumn_equinox': '2018-09-23',
                'winter_solstice': '2018-12-21',
                'spring_equinox': '2019-03-20',
                'summer_solstice': '2019-06-21'
            }
        }

    def calculate_solstice_equinox_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        精确计算到二分二至日的距离

        使用2012-2018年实际的二分二至日日期，计算每个记录到对应水文年关键日的距离
        """
        try:
            df_copy = df.copy()

            # 确保有水文年DOY和水文年
            if 'hydrological_doy' not in df_copy.columns or 'hydrological_year' not in df_copy.columns:
                self.logger.error("❌ 需要先计算水文年DOY和水文年")
                return df_copy

            self.logger.info("🌍 精确计算到二分二至日的距离...")

            # 为每个记录计算到四个关键日的距离
            for event in ['autumn_equinox', 'winter_solstice', 'spring_equinox', 'summer_solstice']:
                distance_col = f'distance_to_{event}'
                df_copy[distance_col] = np.nan

                # 对每个水文年分别处理
                for hydro_year in df_copy['hydrological_year'].unique():
                    if pd.isna(hydro_year):
                        continue

                    hydro_year = int(hydro_year)

                    # 获取该水文年的关键日日期
                    if hydro_year in self.solstice_equinox_dates:
                        event_date = self.solstice_equinox_dates[hydro_year][event]

                        # 计算关键日的水文年DOY
                        event_dt = pd.to_datetime(event_date)
                        event_hydro_doy = self._calculate_hydrological_doy_for_date(event_dt)

                        # 计算该水文年内所有记录到这个关键日的距离
                        mask = df_copy['hydrological_year'] == hydro_year
                        df_copy.loc[mask, distance_col] = abs(df_copy.loc[mask, 'hydrological_doy'] - event_hydro_doy)

            # 计算到最近的关键日距离
            distances = df_copy[[f'distance_to_{event}' for event in
                                 ['autumn_equinox', 'winter_solstice', 'spring_equinox', 'summer_solstice']]]
            df_copy['distance_to_nearest_solstice_equinox'] = distances.min(axis=1)
            df_copy['nearest_solstice_equinox'] = distances.idxmin(axis=1).str.replace('distance_to_', '')

            # 统计信息
            self.logger.info("📊 二分二至日距离统计:")
            for event in ['autumn_equinox', 'winter_solstice', 'spring_equinox', 'summer_solstice']:
                col = f'distance_to_{event}'
                valid_count = df_copy[col].notna().sum()
                if valid_count > 0:
                    min_dist = df_copy[col].min()
                    max_dist = df_copy[col].max()
                    mean_dist = df_copy[col].mean()
                    self.logger.info(
                        f"  {event}: {valid_count} 记录, 距离范围 {min_dist:.0f}-{max_dist:.0f}, 平均 {mean_dist:.1f}")

            # 统计最近关键日分布
            nearest_counts = df_copy['nearest_solstice_equinox'].value_counts()
            self.logger.info("  最近关键日分布:")
            for event, count in nearest_counts.items():
                percentage = (count / len(df_copy)) * 100
                self.logger.info(f"    {event}: {count} 条记录 ({percentage:.1f}%)")

            return df_copy

        except Exception as e:
            self.logger.error(f"❌ 计算二分二至日距离失败: {e}")
            return df

    def _calculate_hydrological_doy_for_date(self, date):
        """
        计算指定日期的水文年DOY
        """
        try:
            # 计算自然年DOY
            natural_doy = date.dayofyear

            # 判断是否为闰年
            is_leap_year = (date.year % 4 == 0 and date.year % 100 != 0) or (date.year % 400 == 0)

            # 计算水文年DOY
            if date.month >= 9:
                if is_leap_year:
                    hydrological_doy = natural_doy - 244
                else:
                    hydrological_doy = natural_doy - 243
            else:
                hydrological_doy = natural_doy + 122

            return hydrological_doy

        except Exception as e:
            self.logger.error(f"❌ 计算日期 {date} 的水文年DOY失败: {e}")
            return np.nan

    def validate_solstice_equinox_calculation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证精确的二分二至日距离计算
        """
        validation_results = {
            'test_cases': [],
            'summary': {}
        }

        try:
            # 测试关键日期（使用实际的二分二至日）
            test_dates = [
                # 2013年关键日
                ('2013-09-22', 2014, 1, 'autumn_equinox', 0),  # 秋分
                ('2013-12-21', 2014, 91, 'winter_solstice', 0),  # 冬至
                ('2014-03-20', 2014, 201, 'spring_equinox', 0),  # 春分
                ('2014-06-21', 2014, 294, 'summer_solstice', 0),  # 夏至

                # 2016年关键日（闰年）
                ('2016-09-22', 2017, 1, 'autumn_equinox', 0),  # 秋分
                ('2016-12-21', 2017, 91, 'winter_solstice', 0),  # 冬至
                ('2017-03-20', 2017, 201, 'spring_equinox', 0),  # 春分
                ('2017-06-21', 2017, 294, 'summer_solstice', 0),  # 夏至

                # 中间日期测试
                ('2013-10-01', 2014, 10, 'autumn_equinox', 9),  # 秋分后9天
                ('2014-01-01', 2014, 123, 'winter_solstice', 32),  # 冬至后32天
                ('2014-04-01', 2014, 213, 'spring_equinox', 12),  # 春分后12天
                ('2014-07-01', 2014, 304, 'summer_solstice', 10),  # 夏至后10天
            ]

            self.logger.info("🔍 精确二分二至日距离计算验证:")

            passed_tests = 0
            total_tests = len(test_dates)

            for test_date, expected_hydro_year, expected_hydro_doy, expected_nearest, expected_distance in test_dates:
                # 在数据中查找测试日期
                test_row = df[df['date'] == test_date]

                test_result = {
                    'date': test_date,
                    'expected_hydro_year': expected_hydro_year,
                    'expected_hydro_doy': expected_hydro_doy,
                    'expected_nearest': expected_nearest,
                    'expected_distance': expected_distance,
                    'found_in_data': not test_row.empty
                }

                if not test_row.empty:
                    actual_hydro_year = test_row['hydrological_year'].iloc[0]
                    actual_hydro_doy = test_row['hydrological_doy'].iloc[0]
                    actual_nearest = test_row['nearest_solstice_equinox'].iloc[0]
                    actual_distance = test_row['distance_to_nearest_solstice_equinox'].iloc[0]

                    test_result.update({
                        'actual_hydro_year': actual_hydro_year,
                        'actual_hydro_doy': actual_hydro_doy,
                        'actual_nearest': actual_nearest,
                        'actual_distance': actual_distance,
                        'hydro_year_correct': actual_hydro_year == expected_hydro_year,
                        'hydro_doy_correct': actual_hydro_doy == expected_hydro_doy,
                        'nearest_correct': actual_nearest == expected_nearest,
                        'distance_correct': actual_distance == expected_distance
                    })

                    # 检查所有字段是否正确
                    all_correct = (actual_hydro_year == expected_hydro_year and
                                   actual_hydro_doy == expected_hydro_doy and
                                   actual_nearest == expected_nearest and
                                   actual_distance == expected_distance)

                    status = "✅" if all_correct else "❌"

                    if all_correct:
                        passed_tests += 1

                    self.logger.info(f"  {status} {test_date}: "
                                     f"水文年={actual_hydro_year}(期望{expected_hydro_year}), "
                                     f"水文年DOY={actual_hydro_doy}(期望{expected_hydro_doy}), "
                                     f"最近={actual_nearest}(期望{expected_nearest}), "
                                     f"距离={actual_distance}(期望{expected_distance})")
                else:
                    test_result.update({
                        'actual_hydro_year': None,
                        'actual_hydro_doy': None,
                        'actual_nearest': None,
                        'actual_distance': None,
                        'hydro_year_correct': False,
                        'hydro_doy_correct': False,
                        'nearest_correct': False,
                        'distance_correct': False
                    })

                    self.logger.warning(f"  ⚠️  测试日期 {test_date} 在数据中不存在")

                validation_results['test_cases'].append(test_result)

            # 统计验证结果
            validation_results['summary'] = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            }

            self.logger.info(f"📊 精确二分二至日验证结果: {passed_tests}/{total_tests} 测试通过 "
                             f"({validation_results['summary']['success_rate']:.1f}%)")

            return validation_results

        except Exception as e:
            self.logger.error(f"❌ 验证二分二至日计算失败: {e}")
            return validation_results

    # 其他方法保持不变...
    def calculate_hydrological_doy(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算水文年DOY"""
        try:
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy['natural_doy'] = df_copy['date'].dt.dayofyear
            df_copy['year'] = df_copy['date'].dt.year
            df_copy['month'] = df_copy['date'].dt.month

            def is_leap_year(year):
                return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

            df_copy['is_leap_year'] = df_copy['year'].apply(is_leap_year)

            def get_hydrological_doy(row):
                try:
                    natural_doy = row['natural_doy']
                    month = row['month']
                    is_leap = row['is_leap_year']

                    if pd.isna(natural_doy) or pd.isna(month):
                        return np.nan

                    if month >= 9:
                        if is_leap:
                            hydrological_doy = natural_doy - 244
                        else:
                            hydrological_doy = natural_doy - 243
                    else:
                        hydrological_doy = natural_doy + 122

                    if hydrological_doy < 1:
                        return 1
                    elif hydrological_doy > 366:
                        return 366
                    else:
                        return int(hydrological_doy)

                except Exception:
                    return np.nan

            df_copy['hydrological_doy'] = df_copy.apply(get_hydrological_doy, axis=1)

            def get_hydrological_year(date):
                try:
                    if pd.isna(date):
                        return np.nan
                    year = date.year
                    month = date.month
                    if month >= 9:
                        return year + 1
                    else:
                        return year
                except Exception:
                    return np.nan

            df_copy['hydrological_year'] = df_copy['date'].apply(get_hydrological_year)

            self.logger.info(f"✅ 水文年DOY计算完成: {df_copy['hydrological_doy'].notna().sum()}/{len(df_copy)} 有效记录")
            return df_copy

        except Exception as e:
            self.logger.error(f"❌ 计算水文年DOY失败: {e}")
            return df

    # 其他方法（find_and_save_duplicates, load_reference_data, load_integrated_data, copy_scp_columns等）
    # 保持不变，这里省略以节省空间...

    def generate_solstice_equinox_report(self) -> str:
        """生成精确的二分二至日报告"""
        if self.corrected_df is None:
            return "无数据可生成报告"

        try:
            report_lines = [
                "=" * 60,
                "精确二分二至日距离计算报告",
                "=" * 60,
                f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ]

            # 显示2012-2018年精确的二分二至日
            report_lines.append("2012-2018年精确二分二至日日期:")
            for year, events in self.solstice_equinox_dates.items():
                report_lines.append(f"  {year}水文年:")
                for event, date in events.items():
                    # 计算该日期的水文年DOY
                    event_dt = pd.to_datetime(date)
                    hydro_doy = self._calculate_hydrological_doy_for_date(event_dt)
                    report_lines.append(f"    {event}: {date} (水文年DOY={hydro_doy})")
            report_lines.append("")

            # 最近关键日分布
            if 'nearest_solstice_equinox' in self.corrected_df.columns:
                nearest_counts = self.corrected_df['nearest_solstice_equinox'].value_counts()
                report_lines.append("最近关键日分布:")
                for event, count in nearest_counts.items():
                    percentage = (count / len(self.corrected_df)) * 100
                    report_lines.append(f"  - {event}: {count} 条记录 ({percentage:.1f}%)")
                report_lines.append("")

            # 距离统计
            if 'distance_to_nearest_solstice_equinox' in self.corrected_df.columns:
                dist_stats = self.corrected_df['distance_to_nearest_solstice_equinox'].describe()
                report_lines.extend([
                    "到最近关键日距离统计:",
                    f"最小值: {dist_stats['min']:.0f} 天",
                    f"最大值: {dist_stats['max']:.0f} 天",
                    f"平均值: {dist_stats['mean']:.1f} 天",
                    f"标准差: {dist_stats['std']:.1f} 天",
                    ""
                ])

            # 各关键日距离统计
            solstice_cols = [col for col in self.corrected_df.columns if
                             col.startswith('distance_to_') and not col.endswith('nearest')]
            if solstice_cols:
                report_lines.append("各关键日距离统计:")
                for col in solstice_cols:
                    event = col.replace('distance_to_', '')
                    event_stats = self.corrected_df[col].describe()
                    valid_count = self.corrected_df[col].notna().sum()
                    report_lines.append(f"  - {event}: {valid_count} 条有效记录")
                    report_lines.append(f"    最小值: {event_stats['min']:.0f} 天")
                    report_lines.append(f"    最大值: {event_stats['max']:.0f} 天")
                    report_lines.append(f"    平均值: {event_stats['mean']:.1f} 天")
                report_lines.append("")

            # 按水文年统计
            if 'hydrological_year' in self.corrected_df.columns:
                hydro_years = sorted(self.corrected_df['hydrological_year'].dropna().unique())
                report_lines.append("各水文年覆盖情况:")
                for hydro_year in hydro_years:
                    year_mask = self.corrected_df['hydrological_year'] == hydro_year
                    year_count = year_mask.sum()
                    if year_count > 0:
                        report_lines.append(f"  {int(hydro_year)}水文年: {year_count} 条记录")
                report_lines.append("")

            # 示例数据
            report_lines.append("数据示例 (前5条):")
            sample_cols = ['station_id', 'date', 'hydrological_year', 'hydrological_doy',
                           'nearest_solstice_equinox', 'distance_to_nearest_solstice_equinox']

            for event in ['autumn_equinox', 'winter_solstice', 'spring_equinox', 'summer_solstice']:
                sample_cols.append(f'distance_to_{event}')

            if 'scp_start' in self.corrected_df.columns:
                sample_cols.extend(['scp_start', 'scp_end'])

            sample_data = self.corrected_df[sample_cols].head()
            for _, row in sample_data.iterrows():
                sample_line = f"  站点 {row['station_id']} | 日期 {row['date'].strftime('%Y-%m-%d')} | "
                sample_line += f"水文年 {int(row['hydrological_year'])} | "
                sample_line += f"水文年DOY {row['hydrological_doy']} | "
                sample_line += f"最近 {row['nearest_solstice_equinox']} | "
                sample_line += f"距离 {row['distance_to_nearest_solstice_equinox']} 天"

                # 显示各关键日距离
                distances = []
                for event in ['autumn_equinox', 'winter_solstice', 'spring_equinox', 'summer_solstice']:
                    dist = row[f'distance_to_{event}']
                    if pd.notna(dist):
                        distances.append(f"{event[:3]}:{int(dist)}")
                if distances:
                    sample_line += f" | 各距离[{', '.join(distances)}]"

                if 'scp_start' in row and pd.notna(row['scp_start']):
                    sample_line += f" | SCP[{int(row['scp_start'])}-{int(row['scp_end'])}]"

                report_lines.append(sample_line)

            return "\n".join(report_lines)

        except Exception as e:
            self.logger.error(f"❌ 生成二分二至日报告失败: {e}")
            return f"生成报告失败: {e}"

    # 其他方法保持不变...
    def find_and_save_duplicates(self):
        """找出并保存所有重复记录"""
        if self.reference_df is None or self.integrated_df is None:
            self.logger.error("❌ 数据未加载完成")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        ref_duplicates = self.reference_df[
            self.reference_df.duplicated(subset=['station_id', 'date'], keep=False)
        ].copy()

        if len(ref_duplicates) > 0:
            ref_duplicates['_duplicate_group'] = ref_duplicates.groupby(['station_id', 'date']).ngroup()
            ref_output_path = self.output_dir / f"reference_duplicates_{timestamp}.xlsx"
            ref_duplicates.to_excel(ref_output_path, index=False)
            self.logger.info(f"📁 参考数据重复记录已保存: {ref_output_path}")

    def load_reference_data(self, file_path: str) -> bool:
        """加载参考数据"""
        try:
            self.reference_df = pd.read_excel(file_path)
            self.logger.info(f"✅ 参考数据加载: {len(self.reference_df)}行, {len(self.reference_df.columns)}列")

            required_cols = self.config['merge_keys'] + self.config['protected_columns']
            missing_cols = [col for col in required_cols if col not in self.reference_df.columns]

            if missing_cols:
                self.logger.error(f"❌ 参考数据缺少必需列: {missing_cols}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"❌ 加载参考数据失败: {e}")
            return False

    def load_integrated_data(self, file_path: str) -> bool:
        """加载整合数据"""
        try:
            self.integrated_df = pd.read_excel(file_path)
            self.logger.info(f"✅ 整合数据加载: {len(self.integrated_df)}行, {len(self.integrated_df.columns)}列")

            missing_cols = [col for col in self.config['merge_keys'] if col not in self.integrated_df.columns]
            if missing_cols:
                self.logger.error(f"❌ 整合数据缺少关键列: {missing_cols}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"❌ 加载整合数据失败: {e}")
            return False

    def copy_scp_columns(self) -> bool:
        """
        复制SCP列并计算精确的二分二至日距离
        """
        if self.reference_df is None or self.integrated_df is None:
            self.logger.error("❌ 数据未加载完成")
            return False

        try:
            # 先找出重复记录
            self.logger.info("🔍 分析重复记录...")
            self.find_and_save_duplicates()

            # 记录原始记录数
            original_integrated_count = len(self.integrated_df)

            # 创建修正数据副本
            self.corrected_df = self.integrated_df.copy()

            # 准备参考数据
            ref_subset = self.reference_df[self.config['merge_keys'] + self.config['protected_columns']].copy()

            self.logger.info("🔄 处理数据格式和空值...")

            # 处理参考数据
            ref_subset = ref_subset.dropna(subset=['station_id'])
            ref_subset['station_id'] = (
                ref_subset['station_id']
                    .astype(float)
                    .apply(lambda x: str(int(x)) if pd.notna(x) else None)
            )
            ref_subset = ref_subset.dropna(subset=['station_id'])
            ref_subset['date'] = pd.to_datetime(ref_subset['date']).dt.strftime('%Y-%m-%d')

            # 去重
            ref_before_dedup = len(ref_subset)
            ref_subset = ref_subset.drop_duplicates(subset=self.config['merge_keys'])
            if ref_before_dedup != len(ref_subset):
                self.logger.info(f"✅ 参考数据去重: {ref_before_dedup} → {len(ref_subset)}")

            # 处理整合数据
            self.corrected_df['station_id'] = self.corrected_df['station_id'].astype(str)
            self.corrected_df['date'] = pd.to_datetime(self.corrected_df['date']).dt.strftime('%Y-%m-%d')

            # 去重
            int_before_dedup = len(self.corrected_df)
            self.corrected_df = self.corrected_df.drop_duplicates(subset=self.config['merge_keys'])
            if int_before_dedup != len(self.corrected_df):
                self.logger.info(f"✅ 整合数据去重: {int_before_dedup} → {len(self.corrected_df)}")

            # 执行合并
            self.logger.info("🔄 执行数据合并...")
            merged = self.corrected_df.merge(
                ref_subset,
                on=self.config['merge_keys'],
                how='left',
                suffixes=('', '_ref')
            )

            # 更新SCP列
            if 'scp_start_ref' in merged.columns:
                mask = merged['scp_start_ref'].notna()
                merged.loc[mask, 'scp_start'] = merged.loc[mask, 'scp_start_ref']
                self.logger.info(f"✅ 更新了 {mask.sum()} 条 scp_start 记录")

            if 'scp_end_ref' in merged.columns:
                mask = merged['scp_end_ref'].notna()
                merged.loc[mask, 'scp_end'] = merged.loc[mask, 'scp_end_ref']
                self.logger.info(f"✅ 更新了 {mask.sum()} 条 scp_end 记录")

            # 移除临时列
            columns_to_keep = [col for col in merged.columns if not col.endswith('_ref')]
            self.corrected_df = merged[columns_to_keep]

            # 计算水文年DOY
            self.logger.info("🌊 计算水文年DOY...")
            self.corrected_df = self.calculate_hydrological_doy(self.corrected_df)

            # 计算精确的二分二至日距离
            self.logger.info("🌍 计算精确的二分二至日距离...")
            self.corrected_df = self.calculate_solstice_equinox_distances(self.corrected_df)

            # 验证计算
            self.logger.info("🔍 验证精确计算...")
            solstice_validation = self.validate_solstice_equinox_calculation(self.corrected_df)

            self.logger.info("📊 处理完成!")
            self.logger.info(f"  原始记录数: {original_integrated_count}")
            self.logger.info(f"  最终记录数: {len(self.corrected_df)}")
            self.logger.info(
                f"  精确二分二至日验证: {solstice_validation['summary']['passed_tests']}/{solstice_validation['summary']['total_tests']} 通过")

            return True

        except Exception as e:
            self.logger.error(f"❌ 处理失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return False

    def validate_data_integrity(self) -> Dict[str, Any]:
        """数据完整性验证"""
        if self.corrected_df is None:
            return {}

        validation = {
            'basic_stats': {
                'total_rows': len(self.corrected_df),
                'total_columns': len(self.corrected_df.columns),
                'unique_stations': self.corrected_df['station_id'].nunique(),
                'date_range': f"{self.corrected_df['date'].min()} 到 {self.corrected_df['date'].max()}"
            },
            'scp_coverage': {},
            'hydrological_info': {},
            'solstice_equinox_info': {},
            'duplicates': self.corrected_df.duplicated(subset=['station_id', 'date']).sum()
        }

        # SCP列覆盖统计
        if 'scp_start' in self.corrected_df.columns:
            scp_start_coverage = (self.corrected_df['scp_start'].notna().sum() / len(self.corrected_df)) * 100
            validation['scp_coverage']['scp_start'] = f"{scp_start_coverage:.1f}%"

        if 'scp_end' in self.corrected_df.columns:
            scp_end_coverage = (self.corrected_df['scp_end'].notna().sum() / len(self.corrected_df)) * 100
            validation['scp_coverage']['scp_end'] = f"{scp_end_coverage:.1f}%"

        # 水文年信息统计
        if 'hydrological_doy' in self.corrected_df.columns:
            hydro_doy_valid = self.corrected_df['hydrological_doy'].notna().sum()
            hydro_year_valid = self.corrected_df['hydrological_year'].notna().sum()
            leap_year_count = self.corrected_df['is_leap_year'].sum()

            validation['hydrological_info'] = {
                'hydro_doy_coverage': f"{(hydro_doy_valid / len(self.corrected_df)) * 100:.1f}%",
                'hydro_year_coverage': f"{(hydro_year_valid / len(self.corrected_df)) * 100:.1f}%",
                'hydro_year_range': f"{self.corrected_df['hydrological_year'].min():.0f} - {self.corrected_df['hydrological_year'].max():.0f}",
                'hydro_doy_range': f"{self.corrected_df['hydrological_doy'].min():.0f} - {self.corrected_df['hydrological_doy'].max():.0f}",
                'leap_year_records': f"{leap_year_count}/{len(self.corrected_df)} ({(leap_year_count / len(self.corrected_df)) * 100:.1f}%)"
            }

        # 二分二至日信息统计
        solstice_cols = [col for col in self.corrected_df.columns if
                         col.startswith('distance_to_') and not col.endswith('nearest')]
        if solstice_cols:
            nearest_counts = self.corrected_df['nearest_solstice_equinox'].value_counts()
            avg_nearest_distance = self.corrected_df['distance_to_nearest_solstice_equinox'].mean()

            validation['solstice_equinox_info'] = {
                'nearest_distribution': nearest_counts.to_dict(),
                'avg_distance_to_nearest': f"{avg_nearest_distance:.1f}",
                'min_distance_to_nearest': f"{self.corrected_df['distance_to_nearest_solstice_equinox'].min():.0f}",
                'max_distance_to_nearest': f"{self.corrected_df['distance_to_nearest_solstice_equinox'].max():.0f}"
            }

        return validation

    def save_results(self, suffix: str = "") -> Optional[str]:
        """保存结果"""
        if self.corrected_df is None:
            self.logger.error("❌ 没有修正数据可保存")
            return None

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            suffix_str = f"_{suffix}" if suffix else ""
            filename = f"corrected_data{suffix_str}_{timestamp}.xlsx"
            output_path = self.output_dir / filename

            self.corrected_df.to_excel(output_path, index=False)
            self.logger.info(f"💾 结果保存成功: {output_path}")

            return str(output_path)

        except Exception as e:
            self.logger.error(f"❌ 保存结果失败: {e}")
            return None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    pipeline = StableDataPipeline("pipeline_output")

    # 加载数据
    if not pipeline.load_reference_data("D:/pyworkspace/fusing xgb/src/training/integrated_data.xlsx"):
        return

    if not pipeline.load_integrated_data("D:/pyworkspace/fusing xgb/src/process/integration/pipeline_output/corrected_data_with_corrected_hydro_doy_20251021_192055.xlsx"):
        return

    # 执行修复
    if pipeline.copy_scp_columns():
        validation = pipeline.validate_data_integrity()

        print("\n📊 最终结果:")
        print(f"  SCP列覆盖: {validation['scp_coverage']}")
        print(f"  重复记录: {validation['duplicates']}")
        if validation['hydrological_info']:
            print(f"  水文年信息: {validation['hydrological_info']}")
        if validation['solstice_equinox_info']:
            print(f"  二分二至日信息: {validation['solstice_equinox_info']}")

        # 生成精确的二分二至日报告
        solstice_report = pipeline.generate_solstice_equinox_report()
        print(f"\n🌍 精确二分二至日详细报告:\n{solstice_report}")

        output_path = pipeline.save_results("with_exact_solstice_distances")

        if output_path:
            print(f"\n✅ 处理完成！输出文件: {output_path}")
    else:
        print("\n❌ 处理失败！")


if __name__ == "__main__":
    main()