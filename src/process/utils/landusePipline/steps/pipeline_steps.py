# steps/pipeline_steps.py
import pandas as pd
from pathlib import Path
import logging
from typing import List

# 修复导入路径
from ..core.pipeline_core import PipelineStep, PipelineContext


# 数据修复步骤
@PipelineStep(name="data_fixer", version="v1")
def fix_landuse_data(ctx: PipelineContext, input_file: str,
                     output_suffix: str = "_fixed") -> str:
    """修复土地利用数据"""
    # 动态导入，避免循环依赖
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # 假设你的原始模块在同一项目根目录下
    try:
        from landuse_fixer import LandUseFixer
    except ImportError:
        # 如果模块不在同一目录，尝试其他路径
        raise ImportError("无法导入 landuse_fixer 模块，请确保它在Python路径中")

    fixer = LandUseFixer()
    output_file = ctx.output_dir / f"{Path(input_file).stem}{output_suffix}.xlsx"

    result = fixer.fix_landuse_data(input_file, str(output_file))
    if not result:
        raise ValueError("数据修复失败")

    return result


@PipelineStep(name="feature_encoder", version="hashing")
def encode_with_hashing(ctx: PipelineContext, input_file: str,
                        landuse_col: str = "landuse_code",
                        n_features: int = 12) -> str:
    """特征哈希编码"""
    # 动态导入
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        from landuse_encoder import LandUseEncoder
    except ImportError:
        raise ImportError("无法导入 landuse_encoder 模块")

    encoder = LandUseEncoder()
    df = pd.read_excel(input_file)

    df_encoded = encoder.feature_hashing_encoding(
        df, landuse_col=landuse_col, n_features=n_features
    )

    output_file = ctx.output_dir / f"{Path(input_file).stem}_hashed.xlsx"
    df_encoded.to_excel(output_file, index=False)

    return str(output_file)


@PipelineStep(name="data_analyzer", version="basic")
def analyze_data(ctx: PipelineContext, input_file: str) -> str:
    """数据分析步骤"""
    # 动态导入
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        from landuse_fixer import LandUseFixer
        from landuse_encoder import LandUseEncoder
    except ImportError as e:
        logging.warning(f"无法导入分析模块: {e}")
        return input_file

    fixer = LandUseFixer()
    encoder = LandUseEncoder()

    # 分析原始数据
    fixer.analyze_data(input_file)

    df = pd.read_excel(input_file)
    if 'landuse_code' in df.columns:
        encoder.compare_encoding_schemes(df)

    # 返回原文件，不修改数据
    return input_file