import json
import pandas as pd
import os
from optuna_optimizer import optimize_swe_model


def find_data_file(filename):
    """查找数据文件"""
    search_paths = [
        filename,
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', filename),
        os.path.join('/data/', filename),
    ]

    for path in search_paths:
        if os.path.exists(path):
            print(f"✅ 找到数据文件: {path}")
            return path

    print("❌ 找不到数据文件，尝试的路径:")
    for path in search_paths:
        print(f"  {path}")
    return None


def load_data(file_path):
    """加载数据文件，支持多种格式和编码"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.csv':
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'latin1', 'cp936']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✅ 使用 {encoding} 编码成功加载CSV文件: {len(df)} 行")
                    return df
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"无法用任何编码加载CSV文件: {file_path}")

        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            print(f"✅ 成功加载Excel文件: {len(df)} 行")
            return df

        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path)
            print(f"✅ 成功加载Parquet文件: {len(df)} 行")
            return df

        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        raise


def run_optimized_training(data_path=None):
    """运行优化后的训练"""

    # 如果没有提供路径，尝试自动查找
    if data_path is None:
        possible_files = ['lu_onehot.xlsx', 'data.csv', 'swe_data.csv']
        for filename in possible_files:
            data_path = find_data_file(filename)
            if data_path:
                break
        else:
            print("❌ 请提供数据文件路径")
            return

    # 加载数据
    print(f"📥 加载数据: {data_path}")
    df = load_data(data_path)

    print(f"📊 数据概况: {len(df)} 行, {len(df.columns)} 列")
    print(f"📋 数据列: {list(df.columns)}")

    # 检查必要列
    required_cols = ['swe', 'station_id', 'date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  缺少必要列: {missing_cols}")
        print("💡 可用列:", list(df.columns))
        return

    print("🚀 开始超参数优化...")

    try:
        # 优化RF参数
        print("\n🔧 优化随机森林参数...")
        rf_params = optimize_swe_model(df, 'rf', n_trials=30)  # 减少试验次数加速

        # 优化XGBoost参数
        print("\n🔧 优化XGBoost参数...")
        xgb_params = optimize_swe_model(df, 'xgb', n_trials=30)

        # 优化GNNWR参数（耗时较长，减少试验次数）
        print("\n🔧 优化GNNWR参数...")
        gnnwr_params = optimize_swe_model(df, 'gnnwr', n_trials=10)

        # 保存优化结果
        optimized_params = {
            'rf': rf_params,
            'xgb': xgb_params,
            'gnnwr': gnnwr_params
        }

        with open('optimized_params.json', 'w', encoding='utf-8') as f:
            json.dump(optimized_params, f, indent=2, ensure_ascii=False)

        print("\n✅ 超参数优化完成！")
        print("📊 优化结果已保存到 optimized_params.json")

        # 显示优化结果
        for model_type, params in optimized_params.items():
            print(f"\n{model_type.upper()} 最佳参数:")
            for key, value in params.items():
                print(f"  {key}: {value}")

        return optimized_params

    except Exception as e:
        print(f"❌ 优化过程失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    # 支持命令行参数
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = None

    run_optimized_training(data_path)