import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import argparse
import logging
import traceback
from datetime import datetime
from pathlib import Path
import pandas as pd
from config import config
from integration import DataIntegrator
from utils.security import SecureProcessor
from utils.logging_setup import setup_logging


def parse_arguments():
    """解析命令行参数（增强版）"""
    parser = argparse.ArgumentParser(description='多源栅格数据处理管道',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--sources', nargs='+',
                        choices=['glsnow', 'gldas', 'modis', 'era5_swe', 'cswe',
                                 'landcover', 'snow_phenology', 'terrain_features',
                                 'snow_depth', 'era5_temperature', 'all'],
                        default=['all'],
                        help='要处理的数据源')

    parser.add_argument('--start-date', type=str,
                        help='处理起始日期 (YYYY-MM-DD)')

    parser.add_argument('--end-date', type=str,
                        help='处理结束日期 (YYYY-MM-DD)')

    parser.add_argument('--no-encrypt', action='store_true',
                        help='禁用输出加密')

    parser.add_argument('--stations', type=str,
                        default='all',
                        help='站点筛选 (all/region:id1,id2/bbox:minx,miny,maxx,maxy/ids:id1,id2)')

    parser.add_argument('--output-dir', type=str,
                        help='自定义输出目录')

    parser.add_argument('--max-workers', type=int, default=1,
                        help='最大并行工作进程数')

    parser.add_argument('--skip-integration', action='store_true',
                        help='跳过数据整合步骤')

    parser.add_argument('--dry-run', action='store_true',
                        help='干跑模式，只显示计划不实际执行')

    parser.add_argument('--terrain-features', nargs='+',
                        choices=['elevation', 'slope', 'aspect', 'eastness', 'tpi',
                                 'curvature1', 'curvature2', 'std_slope', 'std_eastness',
                                 'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high', 'std_aspect'],
                        help='选择要处理的地形特征')

    parser.add_argument('--debug-terrain', action='store_true',
                        help='调试模式：只处理elevation, slope, aspect三个主要特征')

    return parser.parse_args()


def validate_date(date_input):
    """验证日期格式（支持字符串和datetime对象）"""
    try:
        if isinstance(date_input, str):
            # 字符串日期
            return datetime.strptime(date_input, '%Y-%m-%d')
        elif hasattr(date_input, 'strftime'):
            # 已经是datetime对象
            return date_input
        else:
            raise ValueError(f"不支持的日期格式: {type(date_input)}")
    except ValueError:
        raise argparse.ArgumentTypeError(f"无效的日期格式: {date_input}，请使用 YYYY-MM-DD")


def get_station_filter(args):
    """解析站点筛选条件（增强错误处理）"""
    if args.stations == 'all':
        return None

    try:
        if args.stations.startswith('region:'):
            region = args.stations.split(':')[1]
            return {'type': 'region', 'value': region}

        if args.stations.startswith('bbox:'):
            coords = list(map(float, args.stations.split(':')[1].split(',')))
            if len(coords) != 4:
                raise ValueError("bbox需要4个坐标值: minx,miny,maxx,maxy")
            return {'type': 'bbox', 'value': coords}

        if args.stations.startswith('ids:'):
            ids = args.stations.split(':')[1].split(',')
            if not ids:
                raise ValueError("站点ID列表不能为空")
            return {'type': 'ids', 'value': ids}

        raise ValueError(f"无效的站点筛选条件: {args.stations}")

    except Exception as e:
        logging.error(f"解析站点筛选条件失败: {str(e)}")
        raise


def create_processor(source, secure_processor, station_filter=None):
    """创建数据处理器实例（修复参数传递问题）"""
    processor_map = {
        'glsnow': ('src.process.sub.glsnow_processor', 'GLSnowProcessor'),
        'era5_swe': ('src.process.sub.era5_swe_processor', 'ERA5SWEProcessor'),
        'cswe': ('src.process.sub.cswe_processor', 'CSWEProcessor'),
        'landcover': ('src.process.sub.landcover_processor', 'LandcoverProcessor'),
        'snow_phenology': ('src.process.sub.snow_phenology_processor', 'SnowPhenologyProcessor'),
        'terrain_features': ('src.process.sub.terrain_features_processor', 'TerrainFeaturesProcessor'),
        'snow_depth': ('src.process.sub.snow_depth_processor', 'SnowDepthProcessor'),
        'era5_temperature': ('src.process.sub.era5_temperature_processor', 'ERA5TemperatureProcessor'),
    }

    if source not in processor_map:
        raise ValueError(f"未知的数据源: {source}")

    try:
        module_path, class_name = processor_map[source]
        module = __import__(module_path, fromlist=[class_name])
        processor_class = getattr(module, class_name)

        # 检查处理器类的初始化参数
        import inspect
        sig = inspect.signature(processor_class.__init__)

        # 根据处理器需要的参数动态创建
        if 'secure_processor' in sig.parameters and 'station_filter' in sig.parameters:
            return processor_class(
                secure_processor=secure_processor,
                station_filter=station_filter
            )
        elif 'station_filter' in sig.parameters:
            return processor_class(station_filter=station_filter)
        else:
            return processor_class()  # 无参数处理器

    except ImportError as e:
        logging.error(f"导入处理器 {source} 失败: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"创建处理器 {source} 实例失败: {str(e)}")
        raise


def get_date_range(source, args):
    """获取处理日期范围（修复日期格式问题）"""
    try:
        # 地理特征数据不依赖时间
        if source == 'terrain_features':
            return None, None

        conf = getattr(config, source, {})
        date_range_conf = conf.get('date_range', {})

        # 获取日期字符串
        start_date_str = args.start_date or date_range_conf.get('start')
        end_date_str = args.end_date or date_range_conf.get('end')

        if not start_date_str or not end_date_str:
            raise ValueError(f"数据源 {source} 缺少日期范围配置")

        # 如果已经是datetime对象，转换为字符串
        if hasattr(start_date_str, 'strftime'):
            start_date_str = start_date_str.strftime('%Y-%m-%d')
        if hasattr(end_date_str, 'strftime'):
            end_date_str = end_date_str.strftime('%Y-%m-%d')

        # 验证日期格式
        validate_date(start_date_str)
        validate_date(end_date_str)

        return start_date_str, end_date_str

    except Exception as e:
        logging.error(f"获取日期范围失败: {str(e)}")
        raise


def process_source(source, integrator, secure_processor, args, station_filter):
    """处理单个数据源（完全重写）"""
    logger = logging.getLogger(f"main.{source}")

    try:
        logger.info(f"🚀 开始处理 {source.upper()} 数据")

        if args.dry_run:
            logger.info(f"✅ 干跑模式: 计划处理 {source}")
            return True

        # 初始化处理器
        processor = create_processor(source, secure_processor, station_filter)

        # 获取日期范围
        start_date, end_date = get_date_range(source, args)

        # 记录处理参数
        logger.info(f"📋 处理参数: 站点筛选={station_filter}, 日期范围={start_date} 到 {end_date}")

        # 执行处理
        if source == 'terrain_features':
            result_path = processor.process()
        else:
            result_path = processor.process_range(start_date, end_date)

        # 验证处理结果
        if not result_path:
            logger.warning(f"❌ {source.upper()} 未生成结果文件")
            return False

        result_path = Path(result_path)
        if not result_path.exists():
            logger.warning(f"❌ {source.upper()} 结果文件不存在: {result_path}")
            return False

        if result_path.stat().st_size == 0:
            logger.warning(f"❌ {source.upper()} 结果文件为空: {result_path}")
            return False

        # 添加到整合器
        display_name = getattr(config, source, {}).get('name', source)
        success = integrator.add_source(display_name, result_path)

        if success:
            logger.info(f"✅ {source.upper()} 处理完成，结果已添加到整合器")
            return True
        else:
            logger.warning(f"⚠️  {source.upper()} 处理完成但添加到整合器失败")
            return False

    except Exception as e:
        logger.error(f"❌ {source.upper()} 处理失败: {str(e)}")
        logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
        return False
    finally:
        # 确保资源清理
        try:
            if 'processor' in locals():
                processor.close()
        except:
            pass


def initialize_environment(args):
    """初始化运行环境"""
    # 设置项目根目录
    project_root = Path(__file__).parent.parent
    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))

    # 初始化日志
    log_config = config._get_log_config()
    setup_logging(log_config)

    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("🌐 启动多源栅格数据处理管道")
    logger.info("=" * 60)

    # 输出运行配置
    logger.info(f"📁 项目根目录: {project_root}")
    logger.info(f"⚙️  运行参数: sources={args.sources}, stations={args.stations}")
    logger.info(f"📅 日期范围: start={args.start_date}, end={args.end_date}")
    logger.info(f"🔐 加密: {'禁用' if args.no_encrypt else '启用'}")
    logger.info(f"🏃 干跑模式: {'是' if args.dry_run else '否'}")

    return logger


def main():
    """主处理流程（完全重写）"""
    try:
        # 解析参数
        args = parse_arguments()

        # 初始化环境
        logger = initialize_environment(args)

        if args.dry_run:
            logger.info("🔍 干跑模式完成，只显示计划不实际执行")
            return 0

        # 初始化安全处理器
        secure_processor = None
        if not args.no_encrypt:
            try:
                secure_processor = SecureProcessor(
                    encryption_key=config.paths.get('encryption_key')
                )
                logger.info("🔐 安全处理器初始化成功")
            except Exception as e:
                logger.error(f"❌ 安全处理器初始化失败: {str(e)}")
                return 1

        # 解析站点筛选条件
        station_filter = None
        try:
            station_filter = get_station_filter(args)
            if station_filter:
                logger.info(f"📍 应用站点筛选: {station_filter}")
        except Exception as e:
            logger.error(f"❌ 站点筛选条件解析失败: {str(e)}")
            return 1

        # 初始化数据整合器
        output_dir = args.output_dir or config.get_output_dir('combined')
        integrator = DataIntegrator(output_dir=output_dir, secure_processor=secure_processor)
        logger.info(f"📂 输出目录: {output_dir}")

        # 确定要处理的数据源
        if 'all' in args.sources:
            sources_to_process = ['cswe','landcover'
                , 'glsnow']  # 默认处理所有源  ,'snow_depth','era5_temperature','era5_swe','snow_phenology',
            logger.info("🌍 处理所有可用数据源")
        else:
            sources_to_process = args.sources
            logger.info(f"🎯 处理指定数据源: {sources_to_process}")

        # 处理各数据源
        success_count = 0
        failed_sources = []

        for i, source in enumerate(sources_to_process, 1):
            logger.info(f"\n🔧 处理进度: {i}/{len(sources_to_process)} - {source}")

            try:
                success = process_source(source, integrator, secure_processor, args, station_filter)
                if success:
                    success_count += 1
                else:
                    failed_sources.append(source)

            except KeyboardInterrupt:
                logger.warning("⏹️  用户中断处理")
                break
            except Exception as e:
                logger.error(f"❌ 处理 {source} 时发生未预期错误: {str(e)}")
                failed_sources.append(source)

        # 整合结果
        if success_count > 0 and not args.skip_integration:
            logger.info(f"\n📊 开始整合 {success_count} 个数据源的结果")

            try:
                # 紧急修复可能的数据结构问题
                fix_count = integrator.emergency_fix()
                if fix_count > 0:
                    logger.warning(f"🔧 执行了 {fix_count} 个紧急修复")

                master_output = integrator.save_master_excel(format_type='wide')
                if master_output:
                    logger.info(f"✅ 主输出文件已生成: {master_output}")

                    # 生成统计报告
                    report = integrator.generate_report()
                    logger.info(f"📈 数据处理统计报告:\n{report}")

                    # 保存报告到文件
                    report_path = Path(master_output).with_suffix('.report.txt')
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                    logger.info(f"📝 详细报告已保存: {report_path}")
                else:
                    logger.error("❌ 主输出文件生成失败")

            except Exception as e:
                logger.error(f"❌ 结果整合失败: {str(e)}")
                logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
        else:
            logger.info("⏭️  跳过数据整合步骤")

        # 输出总结报告
        logger.info("\n" + "=" * 60)
        logger.info("🏁 处理完成总结")
        logger.info("=" * 60)
        logger.info(f"✅ 成功处理: {success_count}/{len(sources_to_process)} 个数据源")

        if failed_sources:
            logger.info(f"❌ 失败的数据源: {failed_sources}")

        if success_count == 0:
            logger.warning("⚠️  没有成功处理任何数据源")
            return 1

        # 清理临时文件
        if secure_processor:
            try:
                secure_processor.clean_secure_tempfiles()
                logger.info("🧹 临时文件清理完成")
            except Exception as e:
                logger.warning(f"⚠️  临时文件清理失败: {str(e)}")

        logger.info("🎉 处理流程完成")
        return 0

    except KeyboardInterrupt:
        logger.warning("⏹️  程序被用户中断")
        return 130
    except Exception as e:
        logger.error(f"💥 程序执行失败: {str(e)}")
        logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
        return 1
    finally:
        # 确保最后的资源清理
        try:
            if 'integrator' in locals():
                integrator.clear_data()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
