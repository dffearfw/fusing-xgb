import pandas as pd


def aggregate_station_data(input_file, output_file):
    """
    按station_id合并时序数据，每个station_id对应一条记录，各特征取平均值

    Args:
        input_file: 输入xlsx文件路径
        output_file: 输出xlsx文件路径
    """

    # 读取数据
    df = pd.read_excel(input_file)
    print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"station_id数量: {df['station_id'].nunique()}")

    # 识别需要聚合的列（排除station_id和date）
    exclude_cols = ['station_id', 'date']
    numeric_cols = []

    for col in df.columns:
        if col not in exclude_cols:
            # 检查是否为数值列
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                print(f"警告: 列 '{col}' 不是数值类型，将被排除")

    print(f"将聚合 {len(numeric_cols)} 个数值列")

    # 按station_id分组并计算平均值
    aggregated_df = df.groupby('station_id')[numeric_cols].mean().reset_index()

    # 保存结果
    aggregated_df.to_excel(output_file, index=False)
    print(f"\n聚合完成!")
    print(f"结果数据: {len(aggregated_df)} 行, {len(aggregated_df.columns)} 列")
    print(f"已保存到: {output_file}")

    # 显示前几行结果
    print("\n前5行结果预览:")
    print(aggregated_df.head())

    return aggregated_df


# 使用示例
if __name__ == "__main__":
    input_xlsx = "processed_data.xlsx"  # 替换为您的输入文件路径
    output_xlsx = "aggregated_station_data.xlsx"  # 输出文件路径

    # 执行聚合
    result = aggregate_station_data(input_xlsx, output_xlsx)
