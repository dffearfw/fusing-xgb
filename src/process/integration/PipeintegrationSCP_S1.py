import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger("StableDataPipeline")


class StableDataPipeline:
    """
    稳定数据管道 - 找出重复记录
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

    def find_and_save_duplicates(self):
        """找出并保存所有重复记录"""
        if self.reference_df is None or self.integrated_df is None:
            self.logger.error("❌ 数据未加载完成")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. 找出参考数据的重复记录
        ref_duplicates = self.reference_df[
            self.reference_df.duplicated(subset=['station_id', 'date'], keep=False)
        ].copy()

        if len(ref_duplicates) > 0:
            ref_duplicates['_duplicate_group'] = ref_duplicates.groupby(['station_id', 'date']).ngroup()
            ref_output_path = self.output_dir / f"reference_duplicates_{timestamp}.xlsx"
            ref_duplicates.to_excel(ref_output_path, index=False)
            self.logger.info(f"📁 参考数据重复记录已保存: {ref_output_path}")

            # 显示重复统计
            ref_dup_stats = ref_duplicates.groupby(['station_id', 'date']).size().reset_index(name='count')
            self.logger.info("🔍 参考数据重复记录详情:")
            for _, row in ref_dup_stats.iterrows():
                self.logger.info(f"  站点: {row['station_id']}, 日期: {row['date']}, 重复次数: {row['count']}")

        # 2. 找出整合数据的重复记录
        int_duplicates = self.integrated_df[
            self.integrated_df.duplicated(subset=['station_id', 'date'], keep=False)
        ].copy()

        if len(int_duplicates) > 0:
            int_duplicates['_duplicate_group'] = int_duplicates.groupby(['station_id', 'date']).ngroup()
            int_output_path = self.output_dir / f"integrated_duplicates_{timestamp}.xlsx"
            int_duplicates.to_excel(int_output_path, index=False)
            self.logger.info(f"📁 整合数据重复记录已保存: {int_output_path}")

            # 显示重复统计
            int_dup_stats = int_duplicates.groupby(['station_id', 'date']).size().reset_index(name='count')
            self.logger.info("🔍 整合数据重复记录详情:")
            for _, row in int_dup_stats.iterrows():
                self.logger.info(f"  站点: {row['station_id']}, 日期: {row['date']}, 重复次数: {row['count']}")

        # 3. 找出合并后产生的重复记录（如果有的话）
        if self.corrected_df is not None:
            corrected_duplicates = self.corrected_df[
                self.corrected_df.duplicated(subset=['station_id', 'date'], keep=False)
            ].copy()

            if len(corrected_duplicates) > 0:
                corrected_duplicates['_duplicate_group'] = corrected_duplicates.groupby(['station_id', 'date']).ngroup()
                corrected_output_path = self.output_dir / f"corrected_duplicates_{timestamp}.xlsx"
                corrected_duplicates.to_excel(corrected_output_path, index=False)
                self.logger.info(f"📁 修正后数据重复记录已保存: {corrected_output_path}")

        # 4. 生成重复记录分析报告
        self._generate_duplicate_report(ref_duplicates, int_duplicates, timestamp)

    def _generate_duplicate_report(self, ref_duplicates, int_duplicates, timestamp):
        """生成重复记录分析报告"""
        report_lines = [
            "=" * 60,
            "重复记录分析报告",
            "=" * 60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        # 参考数据重复分析
        if len(ref_duplicates) > 0:
            ref_dup_stats = ref_duplicates.groupby(['station_id', 'date']).size().reset_index(name='count')
            report_lines.extend([
                "参考数据重复记录:",
                f"总重复组数: {len(ref_dup_stats)}",
                f"总重复记录数: {len(ref_duplicates)}",
                ""
            ])

            report_lines.append("重复记录详情:")
            for _, row in ref_dup_stats.iterrows():
                report_lines.append(f"  - 站点: {row['station_id']}, 日期: {row['date']}, 重复次数: {row['count']}")
        else:
            report_lines.append("参考数据: 无重复记录")

        report_lines.append("")

        # 整合数据重复分析
        if len(int_duplicates) > 0:
            int_dup_stats = int_duplicates.groupby(['station_id', 'date']).size().reset_index(name='count')
            report_lines.extend([
                "整合数据重复记录:",
                f"总重复组数: {len(int_dup_stats)}",
                f"总重复记录数: {len(int_duplicates)}",
                ""
            ])

            report_lines.append("重复记录详情:")
            for _, row in int_dup_stats.iterrows():
                report_lines.append(f"  - 站点: {row['station_id']}, 日期: {row['date']}, 重复次数: {row['count']}")
        else:
            report_lines.append("整合数据: 无重复记录")

        # 保存报告
        report_path = self.output_dir / f"duplicate_analysis_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"📄 重复记录分析报告已保存: {report_path}")

        # 在控制台输出摘要
        print("\n" + "=" * 50)
        print("重复记录摘要")
        print("=" * 50)
        if len(ref_duplicates) > 0:
            print(f"参考数据: {len(ref_duplicates)} 条重复记录")
        if len(int_duplicates) > 0:
            print(f"整合数据: {len(int_duplicates)} 条重复记录")

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
        复制SCP列并找出重复记录
        """
        if self.reference_df is None or self.integrated_df is None:
            self.logger.error("❌ 数据未加载完成")
            return False

        try:
            # 先找出重复记录
            self.logger.info("🔍 开始分析重复记录...")
            self.find_and_save_duplicates()

            # 记录原始记录数
            original_integrated_count = len(self.integrated_df)

            # 创建修正数据副本
            self.corrected_df = self.integrated_df.copy()

            # 准备参考数据
            ref_subset = self.reference_df[self.config['merge_keys'] + self.config['protected_columns']].copy()

            self.logger.info("🔄 处理数据格式和空值...")

            # 1. 处理参考数据
            ref_subset = ref_subset.dropna(subset=['station_id'])

            # 安全转换
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

            # 2. 处理整合数据
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

            self.logger.info("📊 处理完成!")
            self.logger.info(f"  原始记录数: {original_integrated_count}")
            self.logger.info(f"  最终记录数: {len(self.corrected_df)}")

            return True

        except Exception as e:
            self.logger.error(f"❌ 复制SCP列失败: {e}")
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
            'duplicates': self.corrected_df.duplicated(subset=['station_id', 'date']).sum()
        }

        if 'scp_start' in self.corrected_df.columns:
            scp_start_coverage = (self.corrected_df['scp_start'].notna().sum() / len(self.corrected_df)) * 100
            validation['scp_coverage']['scp_start'] = f"{scp_start_coverage:.1f}%"

        if 'scp_end' in self.corrected_df.columns:
            scp_end_coverage = (self.corrected_df['scp_end'].notna().sum() / len(self.corrected_df)) * 100
            validation['scp_coverage']['scp_end'] = f"{scp_end_coverage:.1f}%"

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
    if not pipeline.load_reference_data("D:/pyworkspace/fusing xgb/src/training/integrated_data.xlsx "):
        return

    if not pipeline.load_integrated_data("D:/pyworkspace/fusing xgb/src/outputs/combined/integrated_data_20251020_200459.xlsx"):
        return

    # 执行修复
    if pipeline.copy_scp_columns():
        validation = pipeline.validate_data_integrity()
        print("\n📊 最终结果:")
        print(f"  SCP列覆盖: {validation['scp_coverage']}")
        print(f"  重复记录: {validation['duplicates']}")

        output_path = pipeline.save_results("fixed_scp")

        if output_path:
            print(f"\n✅ 处理完成！输出文件: {output_path}")
    else:
        print("\n❌ 处理失败！")


if __name__ == "__main__":
    main()