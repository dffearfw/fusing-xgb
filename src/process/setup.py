# TODO: 用setup的方式完成命令行运行

from setuptools import setup, find_packages

setup(
    name="fusing-xgb",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "xarray",
        "netCDF4",
        "pyproj",
        "rasterio",
        "h5py",
        "pyarrow",
        "scikit-learn",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "fusing-xgb=src.process.main:main",
        ],
    },
)