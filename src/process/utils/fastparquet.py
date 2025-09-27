"""
Create on 2025/8/20

@auther:Thinkpad
"""
import pandas as pd
import pyarrow.parquet as pq

df=pq.read_table('../../outputs/snow_depth/snow_depth_20250909_120035.parquet')
pf=df.to_pandas

print(pf)