import json
from optuna_optimizer import optimize_swe_model
import pandas as pd


def run_optimized_training(data_path):
    """运行优化后的训练"""

    # 加载数据
    df = pd.read_csv(data_path)

    print("🚀 开始超参数优化...")

    # 优化RF参数
    print("🔧 优化随机森林参数...")
    rf_params = optimize_swe_model(df, 'rf', n_trials=50)

    # 优化XGBoost参数
    print("🔧 优化XGBoost参数...")
    xgb_params = optimize_swe_model(df, 'xgb', n_trials=50)

    # 优化GNNWR参数
    print("🔧 优化GNNWR参数...")
    gnnwr_params = optimize_swe_model(df, 'gnnwr', n_trials=20)

    # 保存优化结果
    optimized_params = {
        'rf': rf_params,
        'xgb': xgb_params,
        'gnnwr': gnnwr_params
    }

    with open('optimized_params.json', 'w') as f:
        json.dump(optimized_params, f, indent=2)

    print("✅ 超参数优化完成！")
    print("📊 优化结果:")
    for model_type, params in optimized_params.items():
        print(f"  {model_type.upper()}: {params}")

    return optimized_params


if __name__ == "__main__":
    run_optimized_training('data.csv')