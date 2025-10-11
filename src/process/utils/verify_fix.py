def verify_landuse_fix():
    """验证土地利用数据修复"""
    from src.process.sub.landuse_processor import LandUseProcessor
    import pandas as pd

    print("=== 验证土地利用数据修复 ===")

    # 创建土地利用处理器
    processor = LandUseProcessor()

    # 处理土地利用数据
    result_path = processor.process()

    if result_path:
        # 读取土地利用结果
        landuse_df = pd.read_parquet(result_path) if result_path.endswith('.parquet') else pd.read_csv(result_path)
        print(f"土地利用数据记录数: {len(landuse_df)}")

        # 检查列
        print("土地利用数据列:", list(landuse_df.columns))

        # 检查是否有时间列（不应该有）
        time_columns = ['date', 'year', 'month', 'processing_time', 'data_year']
        found_time_columns = [col for col in time_columns if col in landuse_df.columns]

        if found_time_columns:
            print(f"❌ 错误: 土地利用数据仍然包含时间列: {found_time_columns}")
        else:
            print("✅ 正确: 土地利用数据不包含时间列")

        # 显示数据示例
        print("土地利用数据示例:")
        print(landuse_df.head())

        # 检查每个站点是否只有一条记录
        station_counts = landuse_df['station_id'].value_counts()
        duplicate_stations = station_counts[station_counts > 1]
        if len(duplicate_stations) > 0:
            print(f"⚠️ 警告: 有 {len(duplicate_stations)} 个站点有多条记录")
        else:
            print("✅ 正确: 每个站点只有一条记录")

    else:
        print("❌ 土地利用处理失败")


if __name__ == "__main__":
    verify_landuse_fix()