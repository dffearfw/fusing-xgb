# 测试arcpy文件

# -*- coding: utf-8 -*-
import sys
import os
import arcpy

# 添加ArcGIS路径（关键！）
sys.path.append(r"C:\Program Files (x86)\ArcGIS\Desktop10.8\bin")
os.environ['PATH'] = r"C:\Program Files (x86)\ArcGIS\Desktop10.8\bin;" + os.environ['PATH']

# 设置工作环境
arcpy.env.overwriteOutput = True
arcpy.env.workspace = r"E:\data\gisws"

# 初始化许可（关键！）
if arcpy.CheckExtension("Spatial") == "Available":
    arcpy.CheckOutExtension("Spatial")
else:
    raise Exception("Spatial Analyst许可不可用")

# 示例处理：创建缓冲区
print("开始处理...")

arcpy.CheckInExtension("Spatial")
