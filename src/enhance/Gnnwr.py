import os
import sys
from gnnwr import models,datasets,utils
import pandas as pd
import torch.nn as nn

data = pd.read_excel('lu_onehot.xlsx')
data.head(5)

data = data.sample(frac=1,random_state=42)
indices = data.index.tolist()
train_idx = indices[:int(0.7*len(data))]
val_idx = indices[int(0.7*len(data)):int(0.8*len(data))]
test_idx = indices[int(0.8*len(data)):]

train_data = data.loc[train_idx]
val_data = data.loc[val_idx]
test_data = data.loc[test_idx]

x_column=['aspect','slope','eastness','tpi','curvature1','curvature2','elevation','std_slope','std_eastness','std_tpi','std_curvature1',
          'std_curvature2','std_high','std_aspect','glsnow','cswe','snow_depth_snow_depth','ERA5温度_ERA5温度','era5_swe','doy','gldas',
          'year','month','scp_start','scp_end','d1','d2','X','Y','Z','da','db','dc','dd','landuse_11','landuse_12','landuse_21','landuse_22',
          'landuse_23','landuse_24','landuse_31','landuse_32','landuse_33','landuse_41','landuse_42','landuse_43','landuse_46',
          'landuse_51','landuse_52','landuse_53','landuse_62','landuse_63','landuse_64']
y_column=['swe']
spatial_column=['longitude'	,'latitude']

train_set, val_set, test_set = datasets.init_dataset_split(
                                            train_data=train_data,
                                            val_data=val_data,
                                            test_data=test_data,
                                            x_column=x_column,
                                            y_column=y_column,
                                            spatial_column=spatial_column,
                                            batch_size = 1024,
                                            use_model="gnnwr")

optimizer_params = {
    "scheduler":"MultiStepLR",
    "scheduler_milestones":[500, 1000, 1500, 2000],
    "scheduler_gamma":0.75,
}
gnnwr = models.GNNWR(train_dataset = train_set,
                     valid_dataset = val_set,
                     test_dataset = test_set,
                     dense_layers = [1024, 512, 256],
                     activate_func = nn.PReLU(init=0.4),
                     start_lr = 0.1,
                     optimizer = "Adadelta",
                     model_name = "GNNWR_PM25",
                     model_save_path = "result/gnnwr_models",
                     log_path = "result/gnnwr_logs",
                     write_path = "result/gnnwr_runs", # 这里需要修改
                     optimizer_params = optimizer_params
                     )
gnnwr.add_graph()

gnnwr.run(max_epoch = 4000,early_stop=1000,print_frequency = 500)

gnnwr.load_model('result/gnnwr_models/GNNWR_PM25.pkl')

gnnwr.result()