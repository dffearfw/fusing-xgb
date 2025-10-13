# run_pipeline.py
import argparse
import logging
import sys
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from dataclasses import dataclass
import json

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在可以直接导入你的原始模块
try:
    from module.landuse_fixer import LandUseFixer
    from module.landuse_encoder import LandUseEncoder
except ImportError as e:
    print(f"❌ 导入原始模块失败: {e}")
    print("请确保 landuse_fixer.py 和 landuse_encoder.py 在Python路径中")
    sys.exit(1)


# ==================== 核心Pipeline框架 ====================

@dataclass
class PipelineContext:
    """Pipeline执行上下文"""
    input_file: str
    output_dir: Path
    step_data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.step_data = {}
        self.metadata = {}
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class PipelineStep:
    """Pipeline步骤装饰器"""
    _registry = {}

    def __init__(self, name: str, version: str = "v1"):
        self.name = name
        self.version = version

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(ctx: PipelineContext, *args, **kwargs):
            step_key = f"{self.name}_{self.version}"

            # 检查输入文件
            input_file = ctx.step_data.get('previous_output', ctx.input_file)
            if not Path(input_file).exists():
                raise FileNotFoundError(f"输入文件不存在: {input_file}")

            logging.info(f"🎯 执行步骤: {step_key}")

            # 执行步骤函数
            result = func(ctx, input_file, *args, **kwargs)

            # 保存步骤结果
            ctx.step_data[step_key] = {
                'input': input_file,
                'output': result,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            ctx.step_data['previous_output'] = result

            logging.info(f"✅ 步骤完成: {step_key} -> {result}")
            return result

        # 注册步骤
        self._registry[f"{self.name}_{self.version}"] = wrapper
        return wrapper

    @classmethod
    def get_step(cls, name: str, version: str = "v1"):
        return cls._registry.get(f"{name}_{version}")


class Pipeline:
    """Pipeline执行器"""

    def __init__(self, name: str = "LandUsePipeline"):
        self.name = name
        self.steps: List[Dict] = []
        self.logger = logging.getLogger(name)

    def add_step(self, step_name: str, step_func: Callable, **kwargs):
        """添加处理步骤"""
        self.steps.append({
            'name': step_name,
            'func': step_func,
            'kwargs': kwargs
        })
        return self

    def execute(self, input_file: str, output_dir: str = None) -> str:
        """执行Pipeline"""
        if output_dir is None:
            output_dir = Path(input_file).parent / "pipeline_output"

        ctx = PipelineContext(input_file, output_dir)

        self.logger.info(f"🚀 启动Pipeline: {self.name}")
        self.logger.info(f"输入: {input_file}")
        self.logger.info(f"步骤数: {len(self.steps)}")

        try:
            for i, step in enumerate(self.steps, 1):
                self.logger.info(f"\n📦 步骤 {i}/{len(self.steps)}: {step['name']}")
                step['func'](ctx, **step['kwargs'])

            final_output = ctx.step_data['previous_output']
            self.logger.info(f"\n🎉 Pipeline完成: {final_output}")

            # 保存执行摘要
            self._save_execution_summary(ctx)

            return final_output

        except Exception as e:
            self.logger.error(f"❌ Pipeline失败: {e}")
            raise

    def _save_execution_summary(self, ctx: PipelineContext):
        """保存执行摘要"""
        summary = {
            'pipeline_name': self.name,
            'execution_time': pd.Timestamp.now().isoformat(),
            'input_file': ctx.input_file,
            'output_dir': str(ctx.output_dir),
            'steps': ctx.step_data
        }

        summary_file = ctx.output_dir / "execution_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📊 执行摘要: {summary_file}")


# ==================== Pipeline步骤定义 ====================

@PipelineStep(name="data_fixer", version="v1")
def fix_landuse_data(ctx: PipelineContext, input_file: str,
                     output_suffix: str = "_fixed") -> str:
    """修复土地利用数据"""
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
    encoder = LandUseEncoder()
    df = pd.read_excel(input_file)

    df_encoded = encoder.feature_hashing_encoding(
        df, landuse_col=landuse_col, n_features=n_features
    )

    output_file = ctx.output_dir / f"{Path(input_file).stem}_hashed.xlsx"
    df_encoded.to_excel(output_file, index=False)

    return str(output_file)


@PipelineStep(name="feature_encoder", version="composite")
def encode_composite(ctx: PipelineContext, input_file: str,
                     methods: List[str] = None) -> str:
    """复合编码策略"""
    encoder = LandUseEncoder()
    df = pd.read_excel(input_file)

    methods = methods or ['hashing', 'frequency']

    for method in methods:
        if method == 'hashing':
            df = encoder.feature_hashing_encoding(df, n_features=10)
        elif method == 'frequency':
            df = encoder.frequency_encoding(df)
        elif method == 'target':
            if 'swe' in df.columns:
                df = encoder.target_encoding(df, target_col='swe')

    output_file = ctx.output_dir / f"{Path(input_file).stem}_composite_encoded.xlsx"
    df.to_excel(output_file, index=False)

    return str(output_file)


@PipelineStep(name="data_analyzer", version="basic")
def analyze_data(ctx: PipelineContext, input_file: str) -> str:
    """数据分析步骤"""
    fixer = LandUseFixer()
    encoder = LandUseEncoder()

    # 分析原始数据
    fixer.analyze_data(input_file)

    df = pd.read_excel(input_file)
    if 'landuse_code' in df.columns:
        encoder.compare_encoding_schemes(df)

    # 返回原文件，不修改数据
    return input_file


# ==================== Pipeline构建器 ====================

class LandUsePipelineBuilder:
    """Pipeline构建器 - 流畅接口"""

    def __init__(self):
        self.pipeline = Pipeline()

    def fix_data(self, version: str = "v1", **kwargs) -> 'LandUsePipelineBuilder':
        """添加数据修复步骤"""
        step_func = PipelineStep.get_step("data_fixer", version)
        if not step_func:
            raise ValueError(f"未知的数据修复版本: {version}")

        self.pipeline.add_step(f"data_fixing_{version}", step_func, **kwargs)
        return self

    def encode_features(self, method: str = "hashing", **kwargs) -> 'LandUsePipelineBuilder':
        """添加特征编码步骤"""
        step_func = PipelineStep.get_step("feature_encoder", method)
        if not step_func:
            raise ValueError(f"未知的编码方法: {method}")

        self.pipeline.add_step(f"feature_encoding_{method}", step_func, **kwargs)
        return self

    def analyze(self) -> 'LandUsePipelineBuilder':
        """添加分析步骤"""
        step_func = PipelineStep.get_step("data_analyzer", "basic")
        self.pipeline.add_step("data_analysis", step_func)
        return self

    def build(self) -> Pipeline:
        """构建Pipeline"""
        return self.pipeline


# 预定义Pipeline配方
class PipelineRecipes:
    """预定义的Pipeline配方"""

    @staticmethod
    def basic_clean_encode() -> Pipeline:
        """基础清洗+编码"""
        return (LandUsePipelineBuilder()
                .fix_data("v1")
                .encode_features("hashing", n_features=10)
                .build())

    @staticmethod
    def advanced_analysis() -> Pipeline:
        """高级分析流程"""
        return (LandUsePipelineBuilder()
                .analyze()
                .fix_data("v1")
                .encode_features("composite", methods=['hashing', 'frequency'])
                .build())

    @staticmethod
    def quick_fix() -> Pipeline:
        """快速修复"""
        return (LandUsePipelineBuilder()
                .fix_data("v1")
                .build())


# ==================== 主程序 ====================

def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log', encoding='utf-8')
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='土地利用数据处理Pipeline')
    parser.add_argument('input_file', help='输入Excel文件路径')
    parser.add_argument('-o', '--output-dir', help='输出目录')
    parser.add_argument('--recipe', choices=['basic', 'advanced', 'quick'],
                        default='basic', help='预定义配方')
    parser.add_argument('--custom', action='store_true', help='自定义流程')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细日志')

    args = parser.parse_args()
    setup_logging(args.verbose)

    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        logging.error(f"输入文件不存在: {args.input_file}")
        return 1

    try:
        if args.custom:
            # 自定义流程
            pipeline = (LandUsePipelineBuilder()
                        .fix_data("v1")
                        .encode_features("hashing", n_features=12)
                        .analyze()
                        .build())
        else:
            # 使用预定义配方
            recipes = {
                'basic': PipelineRecipes.basic_clean_encode,
                'advanced': PipelineRecipes.advanced_analysis,
                'quick': PipelineRecipes.quick_fix
            }
            pipeline = recipes[args.recipe]()

        # 执行Pipeline
        result = pipeline.execute(args.input_file, args.output_dir)
        print(f"🎉 Pipeline执行成功!")
        print(f"📁 输出文件: {result}")

    except Exception as e:
        logging.error(f"Pipeline执行失败: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    exit(main())