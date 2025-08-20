"""
Create on 2025/8/20

@auther:Thinkpad
"""
import pandas as pd
import pyarrow.parquet as pq

df=pq.read_table('../../outputs/glsnow/glsnow_2013-01-02_2017-12-31.parquet')
pf=df.to_pandas

print(pf)