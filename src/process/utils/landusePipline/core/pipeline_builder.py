# core/pipeline_builder.py
import logging
from typing import List, Dict, Any

# 修复导入路径 - 使用相对导入
from .pipeline_core import Pipeline, PipelineStep, PipelineContext


# 延迟导入步骤，避免循环导入
def import_steps():
    from ..steps import pipeline_steps
    return pipeline_steps


class LandUsePipelineBuilder:
    """Pipeline构建器 - 流畅接口"""

    def __init__(self):
        self.pipeline = Pipeline()
        self.logger = logging.getLogger("PipelineBuilder")

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

    def add_custom_step(self, step_name: str, step_func, **kwargs) -> 'LandUsePipelineBuilder':
        """添加自定义步骤"""
        self.pipeline.add_step(step_name, step_func, **kwargs)
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