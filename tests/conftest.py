import pytest
import sys
import os
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

@pytest.fixture
def sample_data():
    """提供测试用的样本数据"""
    return {
        'station_id': 'test_station',
        'longitude': 116.0,
        'latitude': 40.0,
        'doy': 150,
        'year': 2023
    }

@pytest.fixture
def temp_output_dir(tmp_path):
    """临时输出目录"""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir