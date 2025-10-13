# pipeline_core.py
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
import pandas as pd
import logging
import inspect
from dataclasses import dataclass
from contextlib import contextmanager


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
        self.output_dir.mkdir(exist_ok=True)


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
    """精巧的Pipeline执行器"""

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
        return self  # 支持链式调用

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
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📊 执行摘要: {summary_file}")