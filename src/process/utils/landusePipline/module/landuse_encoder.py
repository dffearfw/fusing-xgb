import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import sys
from sklearn.feature_extraction import FeatureHasher
import hashlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LandUseEncoder")


class LandUseEncoder:
    def __init__(self):
        self.logger = logging.getLogger("LandUseEncoder")
        self.feature_hasher = None
        self.encoding_mapping = {}

    def feature_hashing_encoding(self, df, landuse_col='landuse_code', n_features=10):
        """
        特征哈希编码

        Args:
            df: 输入DataFrame
            landuse_col: landuse列名
            n_features: 哈希特征维度

        Returns:
            DataFrame: 编码后的DataFrame
        """
        self.logger.info(f"=== 特征哈希编码 ===")
        self.logger.info(f"原始列: {landuse_col}")
        self.logger.info(f"目标维度: {n_features}")

        if landuse_col not in df.columns:
            self.logger.error(f"列 {landuse_col} 不存在")
            return df

        # 检查唯一值数量
        unique_values = df[landuse_col].nunique()
        self.logger.info(f"唯一landuse值数量: {unique_values}")

        # 创建特征哈希器
        self.feature_hasher = FeatureHasher(n_features=n_features, input_type='string')

        # 准备数据（转换为字符串）
        landuse_values = df[landuse_col].astype(str).fillna('unknown')

        # 应用特征哈希
        hashed_features = self.feature_hasher.transform(
            [[val] for val in landuse_values]
        ).toarray()

        # 创建哈希特征列名
        hash_cols = [f'landuse_hash_{i}' for i in range(n_features)]

        # 添加到DataFrame
        hash_df = pd.DataFrame(hashed_features, columns=hash_cols, index=df.index)
        result_df = pd.concat([df, hash_df], axis=1)

        # 记录编码信息
        self.encoding_mapping['feature_hashing'] = {
            'original_col': landuse_col,
            'hash_cols': hash_cols,
            'n_features': n_features
        }

        self.logger.info(f"特征哈希完成: {landuse_col} -> {n_features} 维特征")
        self.logger.info(f"新增列: {hash_cols}")

        return result_df

    def frequency_encoding(self, df, landuse_col='landuse_code'):
        """
        频率编码

        Args:
            df: 输入DataFrame
            landuse_col: landuse列名

        Returns:
            DataFrame: 编码后的DataFrame
        """
        self.logger.info(f"=== 频率编码 ===")

        if landuse_col not in df.columns:
            self.logger.error(f"列 {landuse_col} 不存在")
            return df

        # 计算频率
        freq_map = df[landuse_col].value_counts(normalize=True).to_dict()

        # 应用频率编码
        freq_col = f'{landuse_col}_freq'
        df[freq_col] = df[landuse_col].map(freq_map)

        # 记录编码信息
        self.encoding_mapping['frequency'] = {
            'original_col': landuse_col,
            'freq_col': freq_col,
            'freq_map': freq_map
        }

        self.logger.info(f"频率编码完成: {landuse_col} -> {freq_col}")
        self.logger.info(f"频率分布: {len(freq_map)} 个类别")

        return df

    def target_encoding(self, df, landuse_col='landuse_code', target_col='swe', smoothing=10):
        """
        目标编码（如果有目标变量）

        Args:
            df: 输入DataFrame
            landuse_col: landuse列名
            target_col: 目标列名
            smoothing: 平滑参数

        Returns:
            DataFrame: 编码后的DataFrame
        """
        self.logger.info(f"=== 目标编码 ===")

        if landuse_col not in df.columns:
            self.logger.error(f"列 {landuse_col} 不存在")
            return df

        if target_col not in df.columns:
            self.logger.error(f"目标列 {target_col} 不存在")
            return df

        # 计算全局均值
        global_mean = df[target_col].mean()

        # 计算每个类别的统计
        stats = df.groupby(landuse_col)[target_col].agg(['mean', 'count'])
        stats['smooth'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)

        # 创建编码映射
        target_map = stats['smooth'].to_dict()

        # 应用目标编码
        target_col_name = f'{landuse_col}_target'
        df[target_col_name] = df[landuse_col].map(target_map)

        # 记录编码信息
        self.encoding_mapping['target'] = {
            'original_col': landuse_col,
            'target_col': target_col_name,
            'target_map': target_map,
            'global_mean': global_mean
        }

        self.logger.info(f"目标编码完成: {landuse_col} -> {target_col_name}")
        self.logger.info(f"全局均值: {global_mean:.3f}")

        return df

    def one_hot_encoding(self, df, landuse_col='landuse_code', max_categories=20):
        """
        独热编码（适用于类别数较少的情况）

        Args:
            df: 输入DataFrame
            landuse_col: landuse列名
            max_categories: 最大类别数限制

        Returns:
            DataFrame: 编码后的DataFrame
        """
        self.logger.info(f"=== 独热编码 ===")

        if landuse_col not in df.columns:
            self.logger.error(f"列 {landuse_col} 不存在")
            return df

        # 检查类别数量
        unique_count = df[landuse_col].nunique()
        if unique_count > max_categories:
            self.logger.warning(f"类别数 {unique_count} 超过限制 {max_categories}，建议使用特征哈希")
            return df

        # 应用独热编码
        encoded_df = pd.get_dummies(df[landuse_col], prefix='landuse')
        result_df = pd.concat([df, encoded_df], axis=1)

        # 记录编码信息
        self.encoding_mapping['one_hot'] = {
            'original_col': landuse_col,
            'encoded_cols': list(encoded_df.columns),
            'n_categories': unique_count
        }

        self.logger.info(f"独热编码完成: {landuse_col} -> {unique_count} 维特征")
        self.logger.info(f"新增列: {list(encoded_df.columns)}")

        return result_df

    def analyze_landuse_distribution(self, df, landuse_col='landuse_code'):
        """分析landuse分布"""
        self.logger.info(f"=== Landuse分布分析 ===")

        if landuse_col not in df.columns:
            self.logger.error(f"列 {landuse_col} 不存在")
            return

        # 基本统计
        value_counts = df[landuse_col].value_counts()
        total_count = len(df)
        unique_count = len(value_counts)

        self.logger.info(f"总记录数: {total_count}")
        self.logger.info(f"唯一landuse值: {unique_count}")
        self.logger.info(f"缺失值: {df[landuse_col].isna().sum()}")

        # 显示分布
        self.logger.info("Landuse值分布:")
        for value, count in value_counts.head(10).items():
            percentage = (count / total_count) * 100
            self.logger.info(f"  {value}: {count} 条记录 ({percentage:.1f}%)")

        if unique_count > 10:
            others_count = value_counts.iloc[10:].sum()
            others_percentage = (others_count / total_count) * 100
            self.logger.info(f"  其他: {others_count} 条记录 ({others_percentage:.1f}%)")

        return value_counts

    def compare_encoding_schemes(self, df, landuse_col='landuse_code', target_col=None):
        """比较不同编码方案"""
        self.logger.info(f"=== 编码方案比较 ===")

        # 分析原始分布
        value_counts = self.analyze_landuse_distribution(df, landuse_col)
        unique_count = len(value_counts)

        self.logger.info(f"\n推荐编码方案:")

        if unique_count <= 10:
            self.logger.info("✅ 独热编码: 类别数少，适合独热编码")
        elif unique_count <= 50:
            self.logger.info("✅ 特征哈希: 类别数适中，特征哈希效果最好")
            self.logger.info("   推荐维度: 8-16 维")
        else:
            self.logger.info("✅ 特征哈希: 类别数多，必须使用特征哈希")
            self.logger.info("   推荐维度: 16-32 维")

        if target_col and target_col in df.columns:
            self.logger.info("✅ 目标编码: 有目标变量，目标编码可能效果更好")

        self.logger.info("❌ 独热编码: 维度爆炸风险")
        self.logger.info("❌ 标签编码: 会引入错误的顺序关系")

        return {
            'unique_count': unique_count,
            'value_counts': value_counts
        }


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='土地利用特征编码工具')
    parser.add_argument('input_file', help='输入的Excel文件路径')
    parser.add_argument('-o', '--output', help='输出的Excel文件路径')
    parser.add_argument('--landuse-col', default='landuse_code', help='landuse列名')
    parser.add_argument('--target-col', help='目标列名（用于目标编码）')

    # 编码方案
    parser.add_argument('--hashing', action='store_true', help='使用特征哈希编码')
    parser.add_argument('--frequency', action='store_true', help='使用频率编码')
    parser.add_argument('--target', action='store_true', help='使用目标编码')
    parser.add_argument('--one-hot', action='store_true', help='使用独热编码')

    # 参数
    parser.add_argument('--hash-dim', type=int, default=10, help='特征哈希维度')
    parser.add_argument('--analyze', action='store_true', help='只分析数据，不编码')
    parser.add_argument('--compare', action='store_true', help='比较不同编码方案')

    args = parser.parse_args()

    encoder = LandUseEncoder()

    try:
        # 读取数据
        df = pd.read_excel(args.input_file)
        logger.info(f"读取数据: {len(df)} 行, {len(df.columns)} 列")

        if args.analyze:
            # 只分析数据
            encoder.analyze_landuse_distribution(df, args.landuse_col)
            return

        if args.compare:
            # 比较编码方案
            encoder.compare_encoding_schemes(df, args.landuse_col, args.target_col)
            return

        # 应用编码方案
        if args.hashing:
            df = encoder.feature_hashing_encoding(df, args.landuse_col, args.hash_dim)

        if args.frequency:
            df = encoder.frequency_encoding(df, args.landuse_col)

        if args.target and args.target_col:
            df = encoder.target_encoding(df, args.landuse_col, args.target_col)

        if args.one_hot:
            df = encoder.one_hot_encoding(df, args.landuse_col)

        # 如果没有指定编码方案，使用特征哈希作为默认
        if not any([args.hashing, args.frequency, args.target, args.one_hot]):
            logger.info("未指定编码方案，使用默认特征哈希")
            df = encoder.feature_hashing_encoding(df, args.landuse_col, args.hash_dim)

        # 保存结果
        if args.output:
            output_file = args.output
        else:
            input_path = Path(args.input_file)
            output_file = input_path.parent / f"{input_path.stem}_encoded{input_path.suffix}"

        df.to_excel(output_file, index=False)
        logger.info(f"编码完成，结果保存至: {output_file}")
        logger.info(f"最终数据维度: {len(df)} 行, {len(df.columns)} 列")

    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()