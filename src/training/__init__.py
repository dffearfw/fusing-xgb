"""
Create on 2025/10/1

@auther:Thinkpad
"""
"""
SWE (Snow Water Equivalent) 模型训练模块

提供XGBoost模型的训练、交叉验证和评估功能。

主要类:
    SWEXGBoostTrainer: 完整的SWE模型训练器

主要函数:
    train_swe_model: 一键训练模型的便捷函数

示例:
    >>> from src.training import train_swe_model
    >>> results = train_swe_model(data_df, output_dir='./results')
"""

from .swe_trainer import SWEXGBoostTrainer, train_swe_model

__version__ = "1.0.0"
__author__ = "wang"

__all__ = [
    'SWEXGBoostTrainer',
    'train_swe_model'
]