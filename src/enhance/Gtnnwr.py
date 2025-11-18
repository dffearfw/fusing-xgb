import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from gnnwr.datasets import init_dataset
from gnnwr.models import GTNNWR

data = pd.read_excel('lu_onehot.xlsx')

data["id"] = np.arange(len(data))
train_dataset, val_dataset, test_dataset = init_dataset(data=data,
                                                        test_ratio=0.15,
                                                        valid_ratio=0.1,
                                                        x_column=['aspect', 'slope', 'eastness', 'tpi', 'curvature1',
                                                                  'curvature2', 'elevation', 'std_slope',
                                                                  'std_eastness', 'std_tpi', 'std_curvature1',
                                                                  'std_curvature2', 'std_high', 'std_aspect', 'glsnow',
                                                                  'cswe', 'snow_depth_snow_depth',
                                                                  'ERA5温度_ERA5温度', 'era5_swe', 'doy', 'gldas',
                                                                   'scp_start', 'scp_end', 'd1', 'd2',
                                                                   'Z', 'da', 'db', 'dc', 'dd',
                                                                  'landuse_11', 'landuse_12', 'landuse_21',
                                                                  'landuse_22',
                                                                  'landuse_23', 'landuse_24', 'landuse_31',
                                                                  'landuse_32', 'landuse_33', 'landuse_41',
                                                                  'landuse_42',
                                                                  'landuse_43', 'landuse_46',
                                                                  'landuse_51', 'landuse_52', 'landuse_53',
                                                                  'landuse_62', 'landuse_63', 'landuse_64'],
                                                        y_column=['swe'],
                                                        spatial_column=['X', 'Y'],
                                                        temp_column=['year', 'month','day'],
                                                        id_column=['id'],
                                                        use_model="gtnnwr",
                                                        sample_seed=48,
                                                        batch_size=1024)
optimizer_params = {
    "scheduler":"MultiStepLR",
    "scheduler_milestones":[1000, 2000, 3000, 4000],
    "scheduler_gamma":0.8,
}
gtnnwr = GTNNWR(train_dataset, val_dataset, test_dataset, [[3], [512,256,64]],drop_out=0.4,optimizer='Adadelta',optimizer_params=optimizer_params,
                write_path = "../demo_result/gtnnwr_runs", # 这里需要修改
                model_name="GTNNWR_DSi")
gtnnwr.add_graph()

gtnnwr.run(15000,1000)

gtnnwr.load_model('../demo_result/gtnnwr_models/GTNNWR_DSi.pkl')

gtnnwr.result()