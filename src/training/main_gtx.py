import logging
import sys
import os
import argparse

import numpy as np
import pandas as pd




# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入训练器
try:
    from gtnnw_xgboost_trianer import GTNNW_XGBoostTrainer, train_gtnnw_xgboost_model, compare_models
    from swe_trainer import SWEXGBoostTrainer, train_swe_model
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保 gtnnw_xgboost_trainer.py 和 swe_trainer.py 在当前目录")
    sys.exit(1)

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
    """主函数 - 命令行接口 - 支持GTNNW-XGBoost融合"""
    parser = argparse.ArgumentParser(
        description='SWE XGBoost/GTNNW-XGBoost模型训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py -d data.csv
  python main.py -d data.csv -o ./results --trees 100 --lr 0.1
  python main.py -d data.csv --use-gtnnwr              # 使用GTNNWR权重增强
  python main.py -d data.csv --compare-models        # 对比纯XGBoost和GTNNW-XGBoost
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

    # GTNNWR相关参数
    parser.add_argument('--use-gtnnwr', action='store_true',
                        help='使用GTNNWR权重增强XGBoost')
    parser.add_argument('--gtnnwr-epochs', type=int, default=5,
                        help='GTNNWR训练轮数 (默认: 5)')
    parser.add_argument('--compare-models', action='store_true',
                        help='对比纯XGBoost和GTNNW-XGBoost性能')
    parser.add_argument('--no-gtnnwr', action='store_true',
                        help='强制不使用GTNNWR（用于对比实验）')

    # GTNNWR特有参数
    parser.add_argument('--graph-layers', type=str, default="[[3], [512,256,64]]",
                        help='GTNNWR图卷积层结构 (默认: [[3], [512,256,64]])')
    parser.add_argument('--drop-out', type=float, default=0.4,
                        help='GTNNWR dropout比例 (默认: 0.4)')

    # 其他参数
    parser.add_argument('--cluster-mode', action='store_true',
                        help='使用聚类集成模式')
    parser.add_argument('--n-clusters', type=int, default=4,
                        help='聚类数量 (默认: 4)')
    parser.add_argument('--use-rf', action='store_true',
                        help='在聚类集成中使用随机森林代替XGBoost')
    parser.add_argument('--optimize', choices=['rf', 'xgb', 'gtnnwr', 'all'],
                        help='使用Optuna优化指定模型的超参数')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Optuna优化试验次数')
    parser.add_argument('--pure-gtnnwr', action='store_true',
                        help='运行纯净版GTNNWR对比实验')

    args = parser.parse_args()

    try:
        logger.info("🚀 启动SWE模型训练程序")
        logger.info(f"输入文件: {args.data}")
        logger.info(f"输出目录: {args.output or '自动生成'}")
        logger.info(f"使用GTNNWR权重增强: {args.use_gtnnwr}")

        # 检查参数一致性
        if args.no_gtnnwr and args.use_gtnnwr:
            logger.warning("--no-gtnnwr和--use-gtnnwr同时指定，将禁用GTNNWR")
            args.use_gtnnwr = False

        # 1. 加载数据
        logger.info("📥 加载数据...")
        df = load_data(args.data)

        if df.empty:
            logger.error("数据加载失败或数据为空")
            return 1

        logger.info(f"数据加载成功: {len(df)} 行, {len(df.columns)} 列")

        # 检查GTNNWR所需的关键特征列是否存在
        if args.use_gtnnwr or args.compare_models:
            # GTNNWR需要空间列和时间列
            gtnnwr_required_cols = ['X', 'Y', 'year', 'month', 'doy']
            missing_cols = [col for col in gtnnwr_required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"GTNNWR需要但数据中缺少的列: {missing_cols}")
                logger.warning("GTNNWR可能无法正常工作，建议检查数据")
                if args.use_gtnnwr:
                    response = input("继续使用GTNNWR吗？(y/n): ").strip().lower()
                    if response != 'y':
                        logger.info("禁用GTNNWR，使用纯XGBoost")
                        args.use_gtnnwr = False

        # 构建模型参数
        params = build_model_parameters(args)

        if args.cluster_mode:
            # 使用聚类集成模式
            logger.info("🎯 使用聚类集成模式")
            logger.info(f"聚类数量: {args.n_clusters}")
            logger.info(f"使用{'随机森林' if args.use_rf else 'XGBoost'}作为基础模型")

            # 这里需要导入聚类集成函数
            try:
                from cluster import train_swe_cluster_ensemble
                results = train_swe_cluster_ensemble(
                    data_df=df,
                    output_dir=args.output,
                    n_clusters=args.n_clusters,
                    params=build_model_parameters(args),
                    use_rf=args.use_rf
                )
            except ImportError:
                logger.error("聚类集成模块未找到，请确保cluster.py在目录中")
                return 1

        elif args.compare_models:
            # 对比实验模式
            logger.info("🔬 启动模型对比实验：纯XGBoost vs GTNNW-XGBoost")

            if args.output is None:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                args.output = f"./model_comparison_{timestamp}"

            comparison_results = compare_models(df, args.output)

            # 显示对比结果
            print_comparison_summary(comparison_results)

            return 0

        elif args.use_gtnnwr:
            # GTNNW-XGBoost模式
            logger.info("🎯 使用GTNNW-XGBoost融合模式")
            logger.info(f"GTNNWR训练轮数: {args.gtnnwr_epochs}")
            logger.info(f"GTNNWR图卷积层: {args.graph_layers}")
            logger.info(f"GTNNWR dropout比例: {args.drop_out}")

            # 解析图卷积层结构
            try:
                graph_layers = eval(args.graph_layers)
            except:
                logger.warning(f"无法解析图卷积层结构: {args.graph_layers}，使用默认值")
                graph_layers = [[3], [512, 256, 64]]

            # 配置GTNNWR参数
            gtnnwr_params = {
                'max_epoch': args.gtnnwr_epochs,
                'graph_layers': graph_layers,
                'drop_out': args.drop_out,
                'optimizer_params': {
                    "scheduler": "MultiStepLR",
                    "scheduler_milestones": [1000, 2000, 3000, 4000],
                    "scheduler_gamma": 0.8,
                }
            }

            # 创建训练器
            trainer = GTNNW_XGBoostTrainer(
                params=params,
                gtnnwr_params=gtnnwr_params,
                use_gtnnwr=True
            )

            # 运行完整分析
            results = trainer.run_complete_analysis(df, args.output)

        elif args.no_gtnnwr:
            # 强制纯XGBoost模式
            logger.info("🎯 使用纯XGBoost模式（强制禁用GTNNWR）")
            results = train_swe_model(df, args.output, params)

        else:
            # 默认纯XGBoost模式
            logger.info("🎯 使用标准XGBoost模式")
            results = train_swe_model(df, args.output, params)

        # 超参数优化
        if args.optimize:
            try:
                from optuna_optimizer import optimize_swe_model
                logger.info(f"开始超参数优化: {args.optimize}, 试验次数: {args.n_trials}")

                if args.optimize == 'all':
                    for model_type in ['rf', 'gtnnwr']:
                        best_params = optimize_swe_model(df, model_type, args.n_trials)
                        logger.info(f"{model_type} 最佳参数: {best_params}")
                else:
                    best_params = optimize_swe_model(df, args.optimize, args.n_trials)
                    logger.info(f"最佳参数: {best_params}")
            except ImportError:
                logger.warning("Optuna优化模块未找到，跳过优化")

        if args.pure_gtnnwr:
            try:
                from cluster import train_pure_gtnnwr_analysis
                results = train_pure_gtnnwr_analysis(df)
            except ImportError:
                logger.warning("纯净版GTNNWR分析模块未找到")

        logger.info("✅ 模型训练完成！")

        # 显示关键结果
        print_summary(results)

        return 0

    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return 1


def print_comparison_summary(comparison_results):
    """打印模型对比结果摘要"""
    print("\n" + "=" * 80)
    print("📊 模型对比实验结果")
    print("=" * 80)

    xgb_results = comparison_results.get('xgboost', {})
    gtnnw_results = comparison_results.get('gtnnw_xgboost', {})

    # 站点交叉验证对比
    if 'station_cv' in xgb_results and 'station_cv' in gtnnw_results:
        xgb_station = xgb_results['station_cv']['overall']
        gtnnw_station = gtnnw_results['station_cv']['overall']

        print("\n📍 站点交叉验证 (空间评估):")
        print(f"  {'指标':<10} {'纯XGBoost':<12} {'GTNNW-XGBoost':<12} {'提升':<10}")
        print("-" * 50)

        # MAE对比
        xgb_mae = xgb_station.get('MAE', np.nan)
        gtnnw_mae = gtnnw_station.get('MAE', np.nan)
        if not np.isnan(xgb_mae) and not np.isnan(gtnnw_mae):
            mae_improve = (xgb_mae - gtnnw_mae) / xgb_mae * 100
            print(f"  {'MAE (mm)':<10} {xgb_mae:<12.3f} {gtnnw_mae:<12.3f} {mae_improve:>+8.1f}%")

        # R对比
        xgb_r = xgb_station.get('R', np.nan)
        gtnnw_r = gtnnw_station.get('R', np.nan)
        if not np.isnan(xgb_r) and not np.isnan(gtnnw_r):
            r_improve = (gtnnw_r - xgb_r) / abs(xgb_r) * 100
            print(f"  {'R':<10} {xgb_r:<12.3f} {gtnnw_r:<12.3f} {r_improve:>+8.1f}%")

    # 年度交叉验证对比
    if 'yearly_cv' in xgb_results and 'yearly_cv' in gtnnw_results:
        xgb_yearly = xgb_results['yearly_cv']['overall']
        gtnnw_yearly = gtnnw_results['yearly_cv']['overall']

        print("\n📅 年度交叉验证 (时间评估):")
        print(f"  {'指标':<10} {'纯XGBoost':<12} {'GTNNW-XGBoost':<12} {'提升':<10}")
        print("-" * 50)

        # MAE对比
        xgb_mae = xgb_yearly.get('MAE', np.nan)
        gtnnw_mae = gtnnw_yearly.get('MAE', np.nan)
        if not np.isnan(xgb_mae) and not np.isnan(gtnnw_mae):
            mae_improve = (xgb_mae - gtnnw_mae) / xgb_mae * 100
            print(f"  {'MAE (mm)':<10} {xgb_mae:<12.3f} {gtnnw_mae:<12.3f} {mae_improve:>+8.1f}%")

        # R对比
        xgb_r = xgb_yearly.get('R', np.nan)
        gtnnw_r = gtnnw_yearly.get('R', np.nan)
        if not np.isnan(xgb_r) and not np.isnan(gtnnw_r):
            r_improve = (gtnnw_r - xgb_r) / abs(xgb_r) * 100
            print(f"  {'R':<10} {xgb_r:<12.3f} {gtnnw_r:<12.3f} {r_improve:>+8.1f}%")

    print("\n💡 结论:")
    if 'r_improve' in locals() and not np.isnan(r_improve):
        if r_improve > 0:
            print(f"  ✅ GTNNW-XGBoost相比纯XGBoost在R指标上提升了{r_improve:.1f}%")
            print(f"  ✅ 建议使用GTNNW-XGBoost进行SWE预测")
        else:
            print(f"  ⚠️  GTNNW-XGBoost相比纯XGBoost在R指标上下降了{abs(r_improve):.1f}%")
            print(f"  ⚠️  建议继续使用纯XGBoost进行SWE预测")

    print("=" * 80)


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
    """打印结果摘要"""
    print("\n" + "=" * 70)
    print("🎉 SWE模型训练完成摘要")
    print("=" * 70)

    # 显示模型类型
    if hasattr(results, 'use_gtnnwr'):
        model_type = "GTNNW-XGBoost" if results.use_gtnnwr else "纯XGBoost"
        print(f"模型类型: {model_type}")
    elif 'preprocessing' in results and 'use_gtnnwr' in results['preprocessing']:
        model_type = "GTNNW-XGBoost" if results['preprocessing']['use_gtnnwr'] else "纯XGBoost"
        print(f"模型类型: {model_type}")

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

        # 询问是否使用GTNNWR
        use_gtnnwr = input("\n是否使用GTNNWR权重增强？(y/n): ").strip().lower()
        use_gtnnwr = use_gtnnwr == 'y'

        if use_gtnnwr:
            print("🔧 将使用GTNNW-XGBoost融合模型")
            # 检查必要列
            required_cols = ['X', 'Y', 'year', 'month', 'doy']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️  缺少GTNNWR需要的列: {missing_cols}")
                print("GTNNWR可能无法正常工作")
                proceed = input("继续吗？(y/n): ").strip().lower()
                if proceed != 'y':
                    use_gtnnwr = False
                    print("切换为纯XGBoost模式")
        else:
            print("🔧 将使用纯XGBoost模型")

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

        # 如果是GTNNWR模式，询问训练轮数和其他参数
        if use_gtnnwr:
            gtnnwr_epochs = input(f"GTNNWR训练轮数 [默认: 5]: ").strip()
            gtnnwr_epochs = int(gtnnwr_epochs) if gtnnwr_epochs else 5

            graph_layers = input(f"GTNNWR图卷积层 [默认: [[3], [512,256,64]]]: ").strip()
            graph_layers = graph_layers if graph_layers else "[[3], [512,256,64]]"

            drop_out = input(f"GTNNWR dropout比例 [默认: 0.4]: ").strip()
            drop_out = float(drop_out) if drop_out else 0.4

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
        print(f"  模型类型: {'GTNNW-XGBoost' if use_gtnnwr else '纯XGBoost'}")
        if use_gtnnwr:
            print(f"  GTNNWR训练轮数: {gtnnwr_epochs}")
            print(f"  GTNNWR图卷积层: {graph_layers}")
            print(f"  GTNNWR dropout比例: {drop_out}")
        print(f"  树的数量: {params.get('n_estimators', 60)}")
        print(f"  学习率: {params.get('learning_rate', 0.17)}")
        print(f"  最大深度: {params.get('max_depth', 5)}")

        confirm = input("\n开始训练模型? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ 取消训练")
            return

        # 训练模型
        print("\n🚀 开始训练...")

        if use_gtnnwr:
            # GTNNW-XGBoost训练
            try:
                graph_layers_eval = eval(graph_layers)
            except:
                graph_layers_eval = [[3], [512, 256, 64]]

            gtnnwr_params = {
                'max_epoch': gtnnwr_epochs,
                'graph_layers': graph_layers_eval,
                'drop_out': drop_out,
                'optimizer_params': {
                    "scheduler": "MultiStepLR",
                    "scheduler_milestones": [1000, 2000, 3000, 4000],
                    "scheduler_gamma": 0.8,
                }
            }
            trainer = GTNNW_XGBoostTrainer(
                params=params,
                gtnnwr_params=gtnnwr_params,
                use_gtnnwr=True
            )
            results = trainer.run_complete_analysis(df, output_dir)
        else:
            # 纯XGBoost训练
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
        'scipy': 'scipy',
        'torch': 'torch'
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
SWE XGBoost/GTNNW-XGBoost模型训练工具

使用方法:

1. 命令行模式:
   python main.py --data <数据文件> [选项]

2. 交互模式:
   python main.py

主要选项:
   --use-gtnnwr: 使用GTNNWR权重增强XGBoost (GTNNW-XGBoost融合)
   --compare-models: 对比纯XGBoost和GTNNW-XGBoost性能
   --gtnnwr-epochs: GTNNWR训练轮数 (默认: 5)
   --graph-layers: GTNNWR图卷积层结构 (默认: [[3], [512,256,64]])
   --drop-out: GTNNWR dropout比例 (默认: 0.4)
   --no-gtnnwr: 强制禁用GTNNWR，使用纯XGBoost

支持的数据格式:
   • CSV文件 (.csv)
   • Excel文件 (.xlsx, .xls)  
   • Parquet文件 (.parquet)

必要数据列:
   • station_id: 站点ID
   • date: 日期
   • swe: 雪水当量值

GTNNWR额外需要:
   • X, Y: 空间坐标
   • year, month, doy: 时间信息

输出结果:
   • 训练好的模型文件 (.pkl)
   • 交叉验证预测结果 (.csv)
   • 特征重要性排序 (.csv)
   • 详细评估报告 (.json, .txt)

示例:
   python main.py -d data.csv                       # 纯XGBoost
   python main.py -d data.csv --use-gtnnwr           # GTNNW-XGBoost
   python main.py -d data.csv --compare-models      # 对比实验
   python main.py -d data.csv -o ./results --trees 100 --lr 0.1
    """)


if __name__ == "__main__":
    # 显示欢迎信息
    print("=" * 70)
    print("❄️  SWE XGBoost/GTNNW-XGBoost模型训练工具")
    print("=" * 70)

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