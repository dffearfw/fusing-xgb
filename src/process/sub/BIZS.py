# 旧版arcpy处理栅格文件

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
def generate_py27_code(year, temp_csv, output_dir, raster_dir):
    """生成Python 2.7处理代码，使用直接拼接避免格式化问题"""
    raster_filename = "GLDAS_SWE_China_{}.tif".format(year)

    # 使用直接拼接构建代码字符串
    code_lines = [
        "# -*- coding: utf-8 -*-",
        "import arcpy",
        "import csv",
        "import os",
        "import sys",
        "import tempfile",
        "import datetime",
        "from collections import defaultdict",
        "",
        "# 设置UTF-8编码环境",
        "reload(sys)",
        "sys.setdefaultencoding('utf-8')",
        "",
        "gis_workspace = r'E:\\data\\gisws'",
        "if not os.path.exists(gis_workspace):",
        "    os.makedirs(gis_workspace)",
        "arcpy.env.workspace = gis_workspace",
        "arcpy.env.scratchWorkspace = gis_workspace",
        "# 配置路径",
        "temp_csv = r'" + temp_csv + "'",
        "raster_path = os.path.join(r'" + raster_dir + "', r'" + raster_filename + "')",
        "output_csv = r'" + os.path.join(output_dir, "results_{}.csv".format(year)) + "'",
        "",
        "# 打印路径用于调试",
        "print('加载栅格: %s' % raster_path)",
        "if not os.path.exists(raster_path):",
        "    print('错误: 栅格文件不存在 - %s' % raster_path)",
        "    sys.exit(1)",
        "",
        "# 设置环境",
        "arcpy.env.overwriteOutput = True",
        "arcpy.CheckOutExtension('Spatial')",
        "",
        "# 获取年份的天数",
        "year_val=" + str(year),
        "is_leap_year = (" + str(year) + " % 4 == 0 and " + str(year) + " % 100 != 0) or (" + str(
            year) + " % 400 == 0)",
        "days_in_year = 366 if is_leap_year else 365",
        "print('%s年有%s天，%s' % (" + str(year) + ", days_in_year, '闰年' if is_leap_year else '平年'))",
        "",
        "# 创建临时工作空间",
        "temp_gdb = os.path.join(gis_workspace, 'temp_{}.gdb'.format(year_val))",
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
        "# 添加字段 - 使用新字段名",
        "arcpy.AddField_management(points_fc, 'station_ID', 'TEXT', field_length=20)",
        "arcpy.AddField_management(points_fc, 'doy', 'SHORT')",
        "",
        "# 插入点数据",
        "with open(temp_csv, 'rb') as f:",
        "    reader = csv.DictReader(f)",
        "    # 调试：打印CSV列名",
        "    print('CSV列名: %s' % reader.fieldnames)",
        "    ",
        "    with arcpy.da.InsertCursor(points_fc, ['SHAPE@XY', 'station_ID', 'doy']) as cursor:",
        "        for idx, row in enumerate(reader):",
        "            try:",
        "                # 调试：打印前3行数据",
        "                if idx < 3:",
        "                    print('行 %d: station ID=%s, Longitude=%s, Latitude=%s, doy=%s' % (",
        "                        idx + 1, row.get('station_ID'), row.get('Longitude'),",
        "                        row.get('Latitude'), row.get('doy')))",
        "                ",
        "                Longitude = float(row['Longitude'])",
        "                Latitude = float(row['Latitude'])",
        "                station_ID = row['station_ID']",
        "                doy = int(row['doy'])",
        "                cursor.insertRow([(Longitude, Latitude), station_ID, doy])",
        "            except Exception as e:",
        "                print('行 %d 插入错误: %s' % (idx + 1, str(e)))",
        "                print('问题行: %s' % row)",
        "",

        "# 验证要素类中的站点ID",
        "station_IDs = set()",
        "with arcpy.da.SearchCursor(points_fc, ['station_ID']) as cursor:",
        "    for row in cursor:",
        "        station_IDs.add(row[0])",
        "print('要素类中唯一站点ID数量: %d' % len(station_IDs))",
        "if len(station_IDs) < 10:",
        "    print('前10个站点ID: %s' % list(station_IDs)[:10])",
        "else:",
        "    print('示例站点ID: %s' % list(station_IDs)[:5])",
        "",

        # ... [其他代码] ...
        "# 获取实际波段数量",
        "raster = arcpy.Raster(raster_path)",
        "actual_band_count = raster.bandCount",
        "print('栅格波段数量: %d' % actual_band_count)",
        "",
        "# 准备波段列表 - 根据实际名称和数量",
        "band_list = []",
        "for band_index in range(0, actual_band_count):",
        "    band_name = '%d_SWE_inst' % band_index",
        "    field_name = 'b%d' % band_index",
        "    band_list.append([raster_path + '\\\\' + band_name, field_name])",
        "",
        "# 提取栅格值到点",
        "arcpy.sa.ExtractMultiValuesToPoints(points_fc, band_list)",
        "",
        "# 创建结果字典",
        "results = defaultdict(dict)",
        "",
        "# 读取提取结果",
        "fields = ['station_ID', 'doy'] + ['b%d' % i for i in range(0, actual_band_count)]",
        "with arcpy.da.SearchCursor(points_fc, fields) as cursor:",
        "    for row in cursor:",
        "        station_ID = row[0]  # station ID",
        "        doy = row[1]        # doy",
        "        ",
        "        # 计算波段索引: doy=1 -> 波段0, doy=2 -> 波段1",
        "        band_index = doy - 1",
        "        ",
        "        # 确保索引在有效范围内",
        "        if 0 <= band_index < actual_band_count:",
        "            band_value = row[2 + band_index]  # 前两个字段后开始是波段值",
        "            results[station_ID][doy] = band_value",
        "        else:",
        "            print('警告: 站点%s的DOY%d超出范围(1-%d)' % (station_ID, doy, actual_band_count))",
        "",
        "# 写入结果CSV",
        "record_count = 0",
        "with open(output_csv, 'wb') as f:",
        "    writer = csv.writer(f)",
        "    writer.writerow(['station_ID', 'doy', 'value'])",
        "    for station_ID, doy_values in results.iteritems():",
        "        # 调试：验证站点ID",
        "        if station_ID is None or station_ID == '':",
        "            print('警告: 发现空站点ID')",
        "            continue",
        "            ",
        "        for doy, value in doy_values.iteritems():",
        "            writer.writerow([station_ID, doy, value])",
        "            record_count += 1",
        "            ",
        "            # 调试：打印前3条记录",
        "            if record_count <= 3:",
        "                print('结果记录: station_ID=%s, doy=%d, value=%s' % (station_ID, doy, value))",
        "",
        "print('写入 %d 条结果记录' % record_count)",

        "# 清理临时数据",
        "arcpy.Delete_management(temp_gdb)",
        "print('完成 %d 年处理' % " + str(year) + ")",
        "sys.stdout.flush()"
    ]

    return "\n".join(code_lines)


def process_single_year(year, year_df, output_dir, raster_dir):
    """处理单个年份数据"""
    # 调试：打印年度数据摘要
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
        py27_code = generate_py27_code(year, str(temp_csv), output_dir, raster_dir)

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
    for year in range(2013, 2018):  # 2013到2018年
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

    # 关联原始属性 - 使用新字段名
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

    # 检查站点ID分布
    if final_df['station_ID'].nunique() < 10:
        print("\n所有站点ID:")
        print(final_df['station_ID'].unique())
    else:
        print("\n前10个站点ID:")
        print(final_df['station_ID'].unique()[:10])

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

        # 保存时包含所有原始属性
        final_df.to_csv(final_output, index=False)

        # 结果验证
        valid_count = final_df['value'].count()
        total_count = len(final_df)
        print(f"处理完成! 结果保存至: {final_output}")
        print(f"总记录数: {total_count}, 有效值数: {valid_count} ({valid_count / total_count:.1%})")
    else:
        print("错误: 未生成最终结果文件")


if __name__ == "__main__":
    main()
