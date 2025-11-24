import pandas as pd

# --- 请确保您的CSV文件和这个Python脚本在同一个文件夹下 ---
input_filename = 'samples_copy2.csv'
output_filename = 'samples_for_gee_final.csv'

try:
    # 1. 读取原始文件
    df = pd.read_csv(input_filename)
    print(f"成功读取文件: {input_filename}")
    print("文件信息:")
    print(df.info())
    print("\n前5行数据:")
    print(df.head())

    # 2. 关键步骤：将经纬度列强制转换为字符串，然后再转换为数字
    # *** 修正：使用正确的列名（首字母大写） ***
    df['Longitude'] = pd.to_numeric(df['Longitude'].astype(str), errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'].astype(str), errors='coerce')

    # 3. 删除任何转换失败的行（即经纬度为空的行）
    initial_rows = len(df)
    df.dropna(subset=['Longitude', 'Latitude'], inplace=True)
    final_rows = len(df)

    print(f"\n清理前总行数: {initial_rows}")
    print(f"清理后总行数: {final_rows} (删除了 {initial_rows - final_rows} 行无效数据)")

    # 4. 保存为全新的、干净的CSV文件
    df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"\n✅ 成功！干净的CSV文件 '{output_filename}' 已生成。")
    print("现在，请将这个新文件上传到GEE，它会完美工作。")

except FileNotFoundError:
    print(f"❌ 错误：找不到文件 '{input_filename}'。请检查文件名和路径。")
except KeyError as e:
    print(f"❌ 错误：找不到列名 {e}。")
    print("   您的列名可能是：", df.columns.tolist())
except Exception as e:
    print(f"❌ 处理文件时发生错误: {e}")

