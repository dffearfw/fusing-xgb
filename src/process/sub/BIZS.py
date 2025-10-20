import shutil
import subprocess
import os
import sys
import pandas as pd
import tempfile
from pathlib import Path

# ===== 配置区域 =====
ARCGIS_PYTHON = r"C:\Python27\ArcGIS10.8\python.exe"
SCRIPT_NAME = "test_arcpy_processor.py"
ARCGIS_BIN_PATH = r"C:\Program Files (x86)\ArcGIS\Desktop10.8\bin"
excel_path = r"D:\pyworkspace\fusing xgb\config\sources\samples.xlsx"
raster_dir = r"E:\data\GLDAS_SWE"
output_dir = r"E:\data\gisws"


# ====================
def prepare_data():
    df = pd.read_excel(excel_path)
    df['date'] = pd.to_datetime(df['time'], format='%Y%m%d')
    df['year'] = df['date'].dt.year
    df['doy'] = df['date'].dt.dayofyear
    return df


# ====================
def generate_py27_code_fixed(year, temp_csv, output_dir, raster_dir):
    """修复波段引用问题的Python 2.7代码"""
    raster_filename = "GLDAS_SWE_China_{}.tif".format(year)

    code_lines = [
        "# -*- coding: utf-8 -*-",
        "import arcpy",
        "import csv",
        "import os",
        "import sys",
        "from collections import defaultdict",
        "",
        "reload(sys)",
        "sys.setdefaultencoding('utf-8')",
        "",
        "gis_workspace = r'E:\\data\\gisws'",
        "if not os.path.exists(gis_workspace):",
        "    os.makedirs(gis_workspace)",
        "arcpy.env.workspace = gis_workspace",
        "arcpy.env.scratchWorkspace = gis_workspace",
        "",
        "temp_csv = r'" + temp_csv + "'",
        "raster_path = os.path.join(r'" + raster_dir + "', r'" + raster_filename + "')",
        "output_csv = r'" + os.path.join(output_dir, "results_{}.csv".format(year)) + "'",
        "",
        "print('处理栅格: %s' % raster_path)",
        "if not os.path.exists(raster_path):",
        "    print('错误: 栅格文件不存在')",
        "    sys.exit(1)",
        "",
        "arcpy.env.overwriteOutput = True",
        "arcpy.CheckOutExtension('Spatial')",
        "",
        "# 创建临时工作空间",
        "temp_gdb = os.path.join(gis_workspace, 'temp_" + str(year) + ".gdb')",
        "if arcpy.Exists(temp_gdb):",
        "    arcpy.Delete_management(temp_gdb)",
        "arcpy.CreateFileGDB_management(os.path.dirname(temp_gdb), os.path.basename(temp_gdb))",
        "",
        "# 创建点要素类",
        "points_fc = os.path.join(temp_gdb, 'points')",
        "arcpy.CreateFeatureclass_management(",
        "    os.path.dirname(points_fc), ",
        "    os.path.basename(points_fc), ",
        "    'POINT', ",
        "    spatial_reference=arcpy.SpatialReference(4490)",
        ")",
        "",
        "# 添加字段",
        "arcpy.AddField_management(points_fc, 'station_ID', 'TEXT', field_length=20)",
        "arcpy.AddField_management(points_fc, 'doy', 'SHORT')",
        "",
        "# 插入点数据",
        "with open(temp_csv, 'rb') as f:",
        "    reader = csv.DictReader(f)",
        "    with arcpy.da.InsertCursor(points_fc, ['SHAPE@XY', 'station_ID', 'doy']) as cursor:",
        "        for row in reader:",
        "            try:",
        "                Longitude = float(row['Longitude'])",
        "                Latitude = float(row['Latitude'])",
        "                station_ID = row['station_ID']",
        "                doy = int(row['doy'])",
        "                cursor.insertRow([(Longitude, Latitude), station_ID, doy])",
        "            except Exception as e:",
        "                print('插入错误: %s' % str(e))",
        "",
        "# 获取波段信息 - 使用正确的方法",
        "print('检查栅格波段...')",
        "raster = arcpy.Raster(raster_path)",
        "band_count = raster.bandCount",
        "print('波段数量: %d' % band_count)",
        "",
        "# 方法1: 直接使用栅格对象创建波段引用",
        "band_list = []",
        "for i in range(1, band_count + 1):",
        "    try:",
        "        # 方法1: 使用Raster对象创建波段",
        "        band_raster = arcpy.Raster(raster_path + '\\Band_' + str(i))",
        "        field_name = 'band_' + str(i)",
        "        band_list.append([band_raster, field_name])",
        "        print('成功添加波段 %d' % i)",
        "    except Exception as e:",
        "        print('方法1失败: %s' % str(e))",
        "        try:",
        "            # 方法2: 使用MakeRasterLayer",
        "            band_layer = 'band_' + str(i)",
        "            arcpy.MakeRasterLayer_management(raster_path, band_layer, band_index=i)",
        "            field_name = 'band_' + str(i)",
        "            band_list.append([band_layer, field_name])",
        "            print('方法2成功添加波段 %d' % i)",
        "        except Exception as e2:",
        "            print('方法2也失败: %s' % str(e2))",
        "",
        "# 如果上述方法都失败，尝试单波段提取",
        "if not band_list:",
        "    print('尝试单波段逐个提取...')",
        "    results = defaultdict(dict)",
        "    ",
        "    for doy in range(1, band_count + 1):",
        "        try:",
        "            # 为每个波段创建临时要素类",
        "            temp_points = os.path.join(temp_gdb, 'points_band_' + str(doy))",
        "            arcpy.CopyFeatures_management(points_fc, temp_points)",
        "            ",
        "            # 提取单个波段",
        "            band_raster = arcpy.Raster(raster_path + '\\Band_' + str(doy))",
        "            arcpy.sa.ExtractValuesToPoints(temp_points, band_raster, temp_points + '_extracted')",
        "            ",
        "            # 读取结果",
        "            with arcpy.da.SearchCursor(temp_points + '_extracted', ['station_ID', 'doy', 'RASTERVALU']) as cursor:",
        "                for row in cursor:",
        "                    station_ID = row[0]",
        "                    target_doy = row[1]",
        "                    if target_doy == doy:  # 只匹配对应的DOY",
        "                        results[station_ID][doy] = row[2]",
        "            ",
        "            print('完成波段 %d 提取' % doy)",
        "            ",
        "        except Exception as e:",
        "            print('波段 %d 提取失败: %s' % (doy, str(e)))",
        "",
        "    # 写入结果",
        "    with open(output_csv, 'wb') as f:",
        "        writer = csv.writer(f)",
        "        writer.writerow(['station_ID', 'doy', 'value'])",
        "        for station_ID, doy_values in results.iteritems():",
        "            for doy, value in doy_values.iteritems():",
        "                writer.writerow([station_ID, doy, value])",
        "",
        "    print('单波段提取完成，写入 %d 条记录' % sum(len(v) for v in results.values()))",
        "",
        "else:",
        "    # 使用多波段提取",
        "    print('使用多波段提取方法...')",
        "    try:",
        "        arcpy.sa.ExtractMultiValuesToPoints(points_fc, band_list)",
        "        print('多波段提取成功')",
        "",
        "        # 读取结果",
        "        results = defaultdict(dict)",
        "        fields = ['station_ID', 'doy'] + ['band_' + str(i) for i in range(1, band_count + 1)]",
        "",
        "        with arcpy.da.SearchCursor(points_fc, fields) as cursor:",
        "            for row in cursor:",
        "                station_ID = row[0]",
        "                doy = row[1]",
        "                ",
        "                # 波段索引从1开始，DOY从1开始，直接对应",
        "                if 1 <= doy <= band_count:",
        "                    band_value = row[1 + doy]  # 前两个字段后是波段值",
        "                    results[station_ID][doy] = band_value",
        "",
        "        # 写入CSV",
        "        with open(output_csv, 'wb') as f:",
        "            writer = csv.writer(f)",
        "            writer.writerow(['station_ID', 'doy', 'value'])",
        "            for station_ID, doy_values in results.iteritems():",
        "                for doy, value in doy_values.iteritems():",
        "                    writer.writerow([station_ID, doy, value])",
        "",
        "        print('多波段提取完成，写入 %d 条记录' % sum(len(v) for v in results.values()))",
        "",
        "    except Exception as e:",
        "        print('多波段提取失败: %s' % str(e))",
        "",
        "# 清理",
        "try:",
        "    arcpy.Delete_management(temp_gdb)",
        "    print('清理临时数据完成')",
        "except:",
        "    print('清理临时数据失败')",
        "",
        "print('完成 " + str(year) + " 年处理')",
        "sys.stdout.flush()"
    ]

    return "\n".join(code_lines)


def process_single_year(year, year_df, output_dir, raster_dir):
    """处理单个年份数据"""
    print(f"\n{year}年数据摘要:")
    print(f"站点数量: {year_df['station_ID'].nunique()}")
    print(f"记录数量: {len(year_df)}")
    print(f"前5个站点ID: {year_df['station_ID'].unique()[:5]}")

    # 创建临时工作目录
    temp_dir = tempfile.mkdtemp()

    try:
        # 创建临时CSV
        temp_csv = Path(temp_dir) / f"stations_{year}.csv"
        year_df[['station_ID', 'Longitude', 'Latitude', 'doy']].to_csv(temp_csv, index=False)

        # 生成并保存Python 2.7脚本
        py27_script = Path(temp_dir) / f"extract_{year}.py"
        py27_code = generate_py27_code_fixed(year, str(temp_csv), output_dir, raster_dir)

        with open(py27_script, "w", encoding="utf-8") as f:
            f.write(py27_code)

        # 执行子进程
        cmd = [ARCGIS_PYTHON, str(py27_script)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        # 实时输出处理
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[PY27 {year}] {output.strip()}")

        # 等待进程结束并检查状态
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read()
            print(f"错误 ({year}): {stderr}")
            return False

        return True
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


def merge_results(output_dir, df):
    """合并所有年度结果并关联原始属性"""
    all_dfs = []
    for year in range(2013, 2018):
        result_file = Path(output_dir) / f"results_{year}.csv"
        if result_file.exists():
            try:
                df_year = pd.read_csv(result_file)
                df_year['year'] = year
                all_dfs.append(df_year)
            except Exception as e:
                print(f"读取 {year} 年结果失败: {str(e)}")

    if not all_dfs:
        print("错误: 未找到任何结果文件")
        return None

    # 合并提取结果
    result_df = pd.concat(all_dfs, ignore_index=True)

    # 创建日期字段
    result_df['date'] = pd.to_datetime(
        result_df['year'].astype(str) + result_df['doy'].astype(str).str.zfill(3),
        format='%Y%j',
        errors='coerce'
    )

    # 删除无效日期
    result_df = result_df.dropna(subset=['date'])

    # 关联原始属性
    final_df = pd.merge(
        df,
        result_df[['station_ID', 'date', 'value']],
        on=['station_ID', 'date'],
        how='left'
    )

    # 验证结果
    print("\n最终结果验证:")
    print(f"总记录数: {len(final_df)}")
    print(f"唯一站点ID数量: {final_df['station_ID'].nunique()}")
    print(f"空值比例: {final_df['value'].isna().mean():.2%}")

    return final_df


def main():
    """主处理函数"""
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 准备数据
    df = prepare_data()

    # 按年份处理 (2013-2018)
    for year in range(2013, 2018):
        year_df = df[df['year'] == year]
        if len(year_df) == 0:
            print(f"跳过 {year} 年（无数据）")
            continue

        print(f"处理 {year} 年数据...")
        success = process_single_year(year, year_df, output_dir, raster_dir)
        status = "成功" if success else "失败"
        print(f"{year} 年处理{status}")

    # 合并结果
    final_df = merge_results(output_dir, df)
    if final_df is not None:
        final_output = Path(output_dir) / "final_results.csv"
        final_df.to_csv(final_output, index=False)

        valid_count = final_df['value'].count()
        total_count = len(final_df)
        print(f"处理完成! 结果保存至: {final_output}")
        print(f"总记录数: {total_count}, 有效值数: {valid_count} ({valid_count / total_count:.1%})")
    else:
        print("错误: 未生成最终结果文件")


if __name__ == "__main__":
    main()