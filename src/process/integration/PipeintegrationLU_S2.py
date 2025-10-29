import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger("LandUseOneHotPipeline")


class LandUseOneHotPipeline:
    """
    土地利用类型独热编码管道
    将cnlucc_landuse_code从标签编码转换为独热编码
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        self.input_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.unique_codes: List[int] = []

    def load_data(self, file_path: str) -> bool:
        """加载数据"""
        try:
            self.input_df = pd.read_excel(file_path)
            self.logger.info(f"✅ 数据加载成功: {len(self.input_df)}行, {len(self.input_df.columns)}列")

            # 检查是否包含landuse_code列
            if 'cnlucc_landuse_code' not in self.input_df.columns:
                self.logger.error("❌ 数据中未找到 cnlucc_landuse_code 列")
                return False

            return True

        except Exception as e:
            self.logger.error(f"❌ 加载数据失败: {e}")
            return False

    def analyze_landuse_distribution(self) -> Dict[str, Any]:
        """分析土地利用类型分布"""
        if self.input_df is None:
            return {}

        try:
            analysis = {}

            # 统计土地利用类型
            landuse_counts = self.input_df['cnlucc_landuse_code'].value_counts().sort_index()
            analysis['landuse_distribution'] = landuse_counts.to_dict()

            # 统计缺失值
            missing_count = self.input_df['cnlucc_landuse_code'].isna().sum()
            analysis['missing_count'] = missing_count
            analysis['missing_percentage'] = (missing_count / len(self.input_df)) * 100

            # 识别所有唯一的土地利用类型
            self.unique_codes = sorted(self.input_df['cnlucc_landuse_code'].dropna().unique().astype(int))
            analysis['unique_landuse_codes'] = self.unique_codes
            analysis['num_unique_codes'] = len(self.unique_codes)

            # 输出分析结果
            self.logger.info("📊 土地利用类型分析:")
            self.logger.info(f"  唯一编码数量: {analysis['num_unique_codes']}")
            self.logger.info(f"  缺失值数量: {analysis['missing_count']} ({analysis['missing_percentage']:.1f}%)")

            self.logger.info("  土地利用类型分布:")
            for code in self.unique_codes:
                count = landuse_counts[code]
                percentage = (count / len(self.input_df)) * 100
                self.logger.info(f"    {code}: {count} 条记录 ({percentage:.1f}%)")

            return analysis

        except Exception as e:
            self.logger.error(f"❌ 分析土地利用类型失败: {e}")
            return {}

    def create_one_hot_encoding(self, analysis: Dict[str, Any]) -> pd.DataFrame:
        """创建独热编码"""
        if self.input_df is None:
            return pd.DataFrame()

        try:
            # 创建数据副本
            self.processed_df = self.input_df.copy()

            self.logger.info(f"🔄 为 {len(self.unique_codes)} 种土地利用类型创建独热编码...")

            # 为每个土地利用类型创建独热编码列
            for code in self.unique_codes:
                col_name = f"landuse_{code}"
                # 创建二进制列：如果landuse_code等于当前code则为1，否则为0
                self.processed_df[col_name] = (self.processed_df['cnlucc_landuse_code'] == code).astype(int)

                # 统计该列的1的数量
                ones_count = self.processed_df[col_name].sum()
                percentage = (ones_count / len(self.processed_df)) * 100

                self.logger.info(f"  ✅ {col_name}: {ones_count} 条记录 ({percentage:.1f}%)")

            # 检查独热编码的正确性
            self._validate_one_hot_encoding()

            self.logger.info(f"✅ 独热编码完成: 新增 {len(self.unique_codes)} 列")

            return self.processed_df

        except Exception as e:
            self.logger.error(f"❌ 创建独热编码失败: {e}")
            return pd.DataFrame()

    def _validate_one_hot_encoding(self):
        """验证独热编码的正确性"""
        try:
            # 检查每行是否只有一个1（互斥性）
            one_hot_cols = [f"landuse_{code}" for code in self.unique_codes]
            row_sums = self.processed_df[one_hot_cols].sum(axis=1)

            # 统计每行1的数量
            sum_counts = row_sums.value_counts().sort_index()

            self.logger.info("🔍 独热编码验证:")
            for sum_val, count in sum_counts.items():
                percentage = (count / len(self.processed_df)) * 100
                self.logger.info(f"  每行{sum_val}个1: {count} 行 ({percentage:.1f}%)")

            # 检查是否有行没有1（缺失值情况）
            zero_rows = (row_sums == 0).sum()
            if zero_rows > 0:
                self.logger.warning(f"  ⚠️  {zero_rows} 行没有土地利用类型编码（缺失值）")

            # 检查是否有行有多个1（编码错误）
            multi_one_rows = (row_sums > 1).sum()
            if multi_one_rows > 0:
                self.logger.error(f"  ❌ {multi_one_rows} 行有多个土地利用类型编码（编码错误）")
            else:
                self.logger.info("  ✅ 所有行都有且只有一个土地利用类型编码")

        except Exception as e:
            self.logger.error(f"❌ 验证独热编码失败: {e}")

    def create_landuse_mapping_report(self, analysis: Dict[str, Any]) -> str:
        """生成土地利用类型映射报告"""
        report_lines = [
            "=" * 60,
            "土地利用类型独热编码报告",
            "=" * 60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        report_lines.extend([
            "基本统计:",
            f"总记录数: {len(self.input_df)}",
            f"唯一土地利用编码数: {analysis['num_unique_codes']}",
            f"缺失值数量: {analysis['missing_count']} ({analysis['missing_percentage']:.1f}%)",
            ""
        ])

        # 土地利用类型分布
        report_lines.append("土地利用类型分布:")
        landuse_counts = analysis['landuse_distribution']
        for code in self.unique_codes:
            count = landuse_counts[code]
            percentage = (count / len(self.input_df)) * 100
            report_lines.append(f"  {code}: {count} 记录 ({percentage:.1f}%)")

        # 独热编码列信息
        if self.processed_df is not None:
            report_lines.extend([
                "",
                "生成的独热编码列:",
            ])
            for code in self.unique_codes:
                col_name = f"landuse_{code}"
                ones_count = self.processed_df[col_name].sum()
                percentage = (ones_count / len(self.processed_df)) * 100
                report_lines.append(f"  {col_name}: {ones_count} 个1 ({percentage:.1f}%)")

        # 数据示例 - 完全重写，避免列访问问题
        if self.processed_df is not None:
            report_lines.extend([
                "",
                "数据示例 (前3条):",
            ])

            # 只显示基本信息和原始编码
            sample_data = self.processed_df[['station_id', 'date', 'cnlucc_landuse_code']].head(3)

            for _, row in sample_data.iterrows():
                sample_line = f"  站点 {row['station_id']} | 日期 {row['date']} | "
                sample_line += f"原编码: {row['cnlucc_landuse_code']}"
                report_lines.append(sample_line)

            # 添加独热编码的单独说明
            report_lines.extend([
                "",
                "独热编码说明:",
                f"- 共生成 {len(self.unique_codes)} 个独热编码列",
                f"- 列名格式: landuse_编码 (如 landuse_10, landuse_20 等)",
                f"- 每个记录在对应的土地利用类型列中值为1，其他为0",
                f"- 所有独热编码列已成功添加到数据中"
            ])

        return "\n".join(report_lines)

    def save_processed_data(self, suffix: str = "") -> Optional[str]:
        """保存处理后的数据"""
        if self.processed_df is None:
            self.logger.error("❌ 没有处理后的数据可保存")
            return None

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            suffix_str = f"_{suffix}" if suffix else ""
            filename = f"landuse_onehot_data{suffix_str}_{timestamp}.xlsx"
            output_path = self.output_dir / filename

            self.processed_df.to_excel(output_path, index=False)
            self.logger.info(f"💾 处理后的数据保存成功: {output_path}")

            return str(output_path)

        except Exception as e:
            self.logger.error(f"❌ 保存数据失败: {e}")
            return None

    def run_pipeline(self, input_file: str) -> bool:
        """运行完整的土地利用类型独热编码管道"""
        self.logger.info("🚀 启动土地利用类型独热编码管道")

        # 1. 加载数据
        if not self.load_data(input_file):
            return False

        # 2. 分析土地利用类型分布
        analysis = self.analyze_landuse_distribution()
        if not analysis:
            return False

        # 3. 创建独热编码
        processed_df = self.create_one_hot_encoding(analysis)
        if processed_df.empty:
            return False

        # 4. 生成报告
        report = self.create_landuse_mapping_report(analysis)
        print(f"\n📄 土地利用类型报告:\n{report}")

        # 5. 保存结果
        output_path = self.save_processed_data("onehot")

        if output_path:
            self.logger.info(f"✅ 管道执行完成! 输出文件: {output_path}")

            # 显示总结信息
            one_hot_cols = [f"landuse_{code}" for code in self.unique_codes]
            print(f"\n🎉 处理完成总结:")
            print(f"  原始数据: {len(self.input_df)} 行, {len(self.input_df.columns)} 列")
            print(f"  处理后数据: {len(self.processed_df)} 行, {len(self.processed_df.columns)} 列")
            print(f"  新增独热编码列: {len(one_hot_cols)} 列")
            print(f"  土地利用类型总数: {analysis['num_unique_codes']} 种")
            print(f"  土地利用编码列表: {self.unique_codes}")
            print(f"  输出文件: {output_path}")

            return True
        else:
            return False


def main():
    """使用示例"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建管道实例
    pipeline = LandUseOneHotPipeline("landuse_output")

    # 运行管道
    success = pipeline.run_pipeline("D:/pyworkspace/fusing xgb/src/training/corrected_data.xlsx")  # 替换为你的输入文件路径

    if success:
        print("\n🎉 土地利用类型独热编码处理完成!")
    else:
        print("\n❌ 处理失败!")


if __name__ == "__main__":
    main()
