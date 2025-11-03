import logging
import sys
import os
import argparse
import pandas as pd
from cluster import train_swe_cluster_ensemble

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 现在可以直接导入
from swe_trainer import SWEXGBoostTrainer, train_swe_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('swe_training.log', encoding='utf-8')
    ]
)

logger = logging.getLogger("SWETrainingMain")


def build_model_parameters(args):
    """根据命令行参数构建模型参数字典

    Args:
        args: argparse参数对象

    Returns:
        dict: XGBoost参数字典
    """
    params = {
        'n_estimators': args.trees,
        'learning_rate': args.lr,
        'max_depth': args.depth,
        'min_child_weight': getattr(args, 'min_child_weight', 5),
        'gamma': getattr(args, 'gamma', 0),
        'subsample': args.subsample,
        'colsample_bytree': args.colsample,
        'reg_alpha': getattr(args, 'reg_alpha', 0.05),
        'random_state': 42,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    # 可选：添加其他参数，如果用户在命令行中指定了的话
    optional_params = ['reg_lambda', 'max_delta_step', 'scale_pos_weight']
    for param in optional_params:
        if hasattr(args, param) and getattr(args, param) is not None:
            params[param] = getattr(args, param)

    return params


def main():
    """主函数 - 命令行接口 - 修复版本：删除重复训练"""
    parser = argparse.ArgumentParser(
        description='SWE XGBoost模型训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py -d data.csv
  python main.py -d data.csv -o ./results --trees 100 --lr 0.1
        """
    )

    parser.add_argument('--data', '-d', required=True,
                        help='输入数据文件路径 (支持CSV/Excel/Parquet)')
    parser.add_argument('--output', '-o', default=None,
                        help='输出目录路径 (默认:自动生成时间戳目录)')
    parser.add_argument('--trees', '-n', type=int, default=60,
                        help='树的数量 (默认: 60)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.17,
                        help='学习率 (默认: 0.17)')
    parser.add_argument('--depth', type=int, default=5,
                        help='树的最大深度 (默认: 5)')
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='子采样比例 (默认: 0.8)')
    parser.add_argument('--colsample', type=float, default=0.5,
                        help='特征采样比例 (默认: 0.5)')

    parser.add_argument('--cluster-mode', action='store_true',
                       help='使用聚类集成模式')
    parser.add_argument('--n-clusters', type=int, default=4,
                       help='聚类数量 (默认: 4)')
    parser.add_argument('--use-rf', action='store_true',
                        help='在聚类集成中使用随机森林代替XGBoost')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='训练设备: auto(自动选择), cuda(GPU), cpu(CPU)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='数据加载工作进程数 (默认: 自动设置)')
    parser.add_argument('--optimize', choices=['rf', 'xgb', 'gnnwr', 'all'],
                        help='使用Optuna优化指定模型的超参数')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Optuna优化试验次数')

    args = parser.parse_args()

    try:
        logger.info("🚀 启动SWE模型训练程序")
        logger.info(f"输入文件: {args.data}")
        logger.info(f"输出目录: {args.output or '自动生成'}")

        # 1. 加载数据
        logger.info("📥 加载数据...")
        df = load_data(args.data)

        if df.empty:
            logger.error("数据加载失败或数据为空")
            return 1

        logger.info(f"数据加载成功: {len(df)} 行, {len(df.columns)} 列")
        logger.info(f"数据列: {list(df.columns)}")

        # 构建模型参数
        params = build_model_parameters(args)
        logger.info(f"模型参数: n_estimators={params['n_estimators']}, "
                    f"learning_rate={params['learning_rate']}, "
                    f"max_depth={params['max_depth']}")

        if args.cluster_mode:
            # 使用聚类集成模式
            logger.info("🎯 使用聚类集成模式")
            logger.info(f"聚类数量: {args.n_clusters}")
            logger.info(f"使用{'随机森林' if args.use_rf else 'XGBoost'}作为基础模型")

            results = train_swe_cluster_ensemble(
                data_df=df,
                output_dir=args.output,
                n_clusters=args.n_clusters,
                params=build_model_parameters(args),
                use_rf=args.use_rf,
                device=args.device
            )
        else:
            # 使用原有模式（保持XGBoost不变）
            from swe_trainer import train_swe_model
            logger.info("🎯 使用标准XGBoost模式")
            results = train_swe_model(
                data_df=df,
                output_dir=args.output,
                params=params
            )

        # 在主函数中添加优化逻辑
        if args.optimize:
            from optuna_optimizer import optimize_swe_model
            logger.info(f"开始超参数优化: {args.optimize}, 试验次数: {args.n_trials}")

            if args.optimize == 'all':
                for model_type in ['rf', 'gnnwr']:
                    best_params = optimize_swe_model(df, model_type, args.n_trials)
                    logger.info(f"{model_type} 最佳参数: {best_params}")
            else:
                best_params = optimize_swe_model(df, args.optimize, args.n_trials)
                logger.info(f"最佳参数: {best_params}")

        logger.info("✅ 模型训练完成！")

        # 显示关键结果
        print_summary(results)

        return 0

    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return 1


def load_data(file_path):
    """加载数据文件

    Args:
        file_path (str): 文件路径

    Returns:
        pd.DataFrame: 加载的数据

    Raises:
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式不支持时抛出
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
            logger.info(f"CSV文件加载成功: {len(df)} 行")
            return df

        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            logger.info(f"Excel文件加载成功: {len(df)} 行")
            return df

        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path)
            logger.info(f"Parquet文件加载成功: {len(df)} 行")
            return df

        else:
            raise ValueError(f"不支持的文件格式: {file_ext}。支持格式: CSV, Excel, Parquet")

    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        # 尝试其他编码方式加载CSV
        if file_ext == '.csv':
            try:
                logger.info("尝试使用GBK编码加载CSV...")
                df = pd.read_csv(file_path, encoding='gbk')
                logger.info(f"使用GBK编码加载成功: {len(df)} 行")
                return df
            except:
                try:
                    logger.info("尝试使用latin1编码加载CSV...")
                    df = pd.read_csv(file_path, encoding='latin1')
                    logger.info(f"使用latin1编码加载成功: {len(df)} 行")
                    return df
                except:
                    raise ValueError(f"CSV文件无法用任何编码加载: {e}")
        else:
            raise


def print_summary(results):
    """打印结果摘要 - 修复版本：处理缺失的station_cv"""
    print("\n" + "=" * 70)
    print("🎉 SWE模型训练完成摘要")
    print("=" * 70)

    # 站点交叉验证结果（如果存在）
    if 'station_cv' in results:
        station = results['station_cv']['overall']
        print(f"\n📍 站点交叉验证 (空间评估):")
        print(f"   MAE:  {station['MAE']:8.3f} mm")
        print(f"   RMSE: {station['RMSE']:8.3f} mm")
        print(f"   R:    {station['R']:8.3f}")
        print(f"   样本数: {station.get('样本数', station.get('samples', 'N/A')):>6}")
        print(f"   折叠数: {results['station_cv']['folds']:6d}")

    # 年度交叉验证结果
    if 'yearly_cv' in results:
        yearly = results['yearly_cv']['overall']
        print(f"\n📅 年度交叉验证 (时间评估):")
        print(f"   MAE:  {yearly['MAE']:8.3f} mm")
        print(f"   RMSE: {yearly['RMSE']:8.3f} mm")
        print(f"   R:    {yearly['R']:8.3f}")
        print(f"   样本数: {yearly.get('样本数', yearly.get('samples', 'N/A')):>6}")
        print(f"   折叠数: {results['yearly_cv']['folds']:6d}")

    # 特征重要性（如果存在）
    if 'feature_importance' in results:
        top_features = results['feature_importance'].head(3)
        print(f"\n🔍 重要特征 Top 3:")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"   {i}. {row['feature']:20} {row['importance']:.4f}")

    # 性能比较（如果两者都存在）
    if 'station_cv' in results and 'yearly_cv' in results:
        station_r = results['station_cv']['overall']['R']
        yearly_r = results['yearly_cv']['overall']['R']

        print(f"\n💡 建议:")
        if station_r > yearly_r:
            print(f"   站点CV性能更优，推荐用于空间评估")
        else:
            print(f"   年度CV性能更优，推荐用于时间评估")
    else:
        print(f"\n💡 建议:")
        print(f"   使用年度交叉验证结果进行评估")

    print("=" * 70)
    print("📁 详细结果已保存到输出目录")


def interactive_mode():
    """交互式模式"""
    print("\n🔍 SWE模型训练 - 交互模式")
    print("-" * 50)

    try:
        # 获取数据文件路径
        data_file = input("请输入数据文件路径: ").strip()
        if not data_file:
            print("❌ 必须提供数据文件路径")
            return

        if not os.path.exists(data_file):
            print(f"❌ 文件不存在: {data_file}")
            return

        # 加载数据
        print("📥 加载数据...")
        df = load_data(data_file)
        if df.empty:
            print("❌ 数据加载失败")
            return

        print(f"✅ 数据加载成功: {len(df)} 行, {len(df.columns)} 列")

        # 显示数据基本信息
        print(f"\n📊 数据概览:")
        print(f"  站点数量: {df['station_id'].nunique()}")
        print(f"  日期范围: {df['date'].min()} 到 {df['date'].max()}")
        print(f"  SWE统计: 均值={df['swe'].mean():.2f}mm, 标准差={df['swe'].std():.2f}mm")

        # 选择输出目录
        output_dir = input("\n请输入输出目录 (回车使用默认): ").strip()
        if not output_dir:
            output_dir = None
            print("使用默认输出目录")

        # 可选：自定义参数
        print(f"\n⚙️ 模型参数 (使用默认值请直接回车):")

        trees = input(f"树的数量 [默认: 60]: ").strip()
        lr = input(f"学习率 [默认: 0.17]: ").strip()
        depth = input(f"最大深度 [默认: 5]: ").strip()

        # 构建参数字典
        params = {}
        if trees:
            params['n_estimators'] = int(trees)
        if lr:
            params['learning_rate'] = float(lr)
        if depth:
            params['max_depth'] = int(depth)

        # 确认开始训练
        print(f"\n🔍 训练配置:")
        print(f"  数据文件: {data_file}")
        print(f"  输出目录: {output_dir or '自动生成'}")
        print(f"  树的数量: {params.get('n_estimators', 60)}")
        print(f"  学习率: {params.get('learning_rate', 0.17)}")
        print(f"  最大深度: {params.get('max_depth', 5)}")

        confirm = input("\n开始训练模型? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ 取消训练")
            return

        # 训练模型
        print("\n🚀 开始训练...")
        results = train_swe_model(df, output_dir, params)

        # 显示结果
        print_summary(results)

        print(f"\n🎯 训练完成！")

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")


def check_dependencies():
    """检查依赖库是否安装"""
    required_packages = {
        'pandas': 'pd',
        'numpy': 'np',
        'xgboost': 'xgb',
        'scikit-learn': 'sklearn',
        'scipy': 'scipy'
    }

    missing_packages = []

    for package, short_name in required_packages.items():
        try:
            if package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ 缺少必要依赖库:")
        for package in missing_packages:
            print(f"  - {package}")
        print(f"\n请安装: pip install {' '.join(missing_packages)}")
        return False

    return True


def show_help():
    """显示帮助信息"""
    print("""
SWE XGBoost模型训练工具

使用方法:

1. 命令行模式:
   python main.py --data <数据文件> [选项]

2. 交互模式:
   python main.py

支持的数据格式:
   • CSV文件 (.csv)
   • Excel文件 (.xlsx, .xls)  
   • Parquet文件 (.parquet)

必要数据列:
   • station_id: 站点ID
   • date: 日期
   • swe: 雪水当量值

输出结果:
   • 训练好的模型文件 (.pkl)
   • 交叉验证预测结果 (.csv)
   • 特征重要性排序 (.csv)
   • 详细评估报告 (.json, .txt)

示例:
   python main.py -d data.csv -o ./results --trees 100 --lr 0.1
    """)


if __name__ == "__main__":
    # 显示欢迎信息
    print("=" * 60)
    print("❄️  SWE XGBoost模型训练工具")
    print("=" * 60)

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 如果没有命令行参数，进入交互模式
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        # 检查是否是帮助请求
        if any(arg in sys.argv for arg in ['-h', '--help', 'help']):
            show_help()
            sys.exit(0)

        # 运行命令行模式
        exit_code = main()
        sys.exit(exit_code)