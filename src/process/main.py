#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
栅格数据处理主程序
功能：从多源栅格数据（GLDAS/MODIS/ERA5）提取站点值并整合输出
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.process.config import config
from src.process.sub import (
    glsnow_processor,

)
from src.process.integration import DataIntegrator
from src.process.utils.security import SecureProcessor
from src.process.utils.logging_setup import setup_logging
import netCDF4

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='栅格数据处理管道')
    parser.add_argument('--sources', nargs='+',
                        choices=['glsnow', 'gldas', 'modis', 'era5', 'all'],
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
                        help='站点筛选 (all/region:id1,id2/bbox:minx,miny,maxx,maxy)')
    return parser.parse_args()


def get_station_filter(args):
    """解析站点筛选条件"""
    if args.stations == 'all':
        return None

    if args.stations.startswith('region:'):
        region = args.stations.split(':')[1]
        return {'type': 'region', 'value': region}

    if args.stations.startswith('bbox:'):
        coords = list(map(float, args.stations.split(':')[1].split(',')))
        return {'type': 'bbox', 'value': coords}

    if args.stations.startswith('ids:'):
        ids = args.stations.split(':')[1].split(',')
        return {'type': 'ids', 'value': ids}

    raise ValueError(f"无效的站点筛选条件: {args.stations}")


def create_processor(source, secure_processor, station_filter=None):
    """创建数据处理器实例 - 修复导入问题"""
    # 确保正确导入处理器类而不是模块
    if source == 'glsnow':
        from src.process.sub.glsnow_processor import GLSnowProcessor
        return GLSnowProcessor()
    # elif source == 'gldas':
    #     from src.processors.gldas_processor import GLDASProcessor
    #     return GLDASProcessor(secure_processor=secure_processor, station_filter=station_filter)
    # elif source == 'modis':
    #     from src.processors.modis_processor import MODISProcessor
    #     return MODISProcessor(secure_processor=secure_processor, station_filter=station_filter)
    # elif source == 'era5':
    #     from src.processors.era5_processor import ERA5Processor
    #     return ERA5Processor(secure_processor=secure_processor, station_filter=station_filter)
    else:
        raise ValueError(f"未知的数据源: {source}")


def process_source(source, integrator, secure_processor, args, station_filter):
    """处理单个数据源"""
    logger = logging.getLogger(f"main.{source}")
    try:
        logger.info(f"开始处理 {source.upper()} 数据")

        # 初始化处理器
        processor = create_processor(source, secure_processor, station_filter)

        # 获取配置中的日期范围
        conf = getattr(config, source)
        start_date = args.start_date or conf['date_range']['start']
        end_date = args.end_date or conf['date_range']['end']

        # 处理数据
        result_path = processor.process_range(start_date, end_date)

        if result_path:
            integrator.add_source(conf['name'], result_path)
            logger.info(f"{source.upper()} 处理完成，结果已添加到整合器")
        return True
    except Exception as e:
        logger.exception(f"{source.upper()} 处理失败: {str(e)}")
        return False


def main():
    """主处理流程"""
    # 解析参数
    args = parse_arguments()

    # 初始化日志
    setup_logging(config._get_log_config())
    logger = logging.getLogger("main")
    logger.info("启动栅格数据处理管道")

    # 初始化安全处理器
    secure_processor = None if args.no_encrypt else SecureProcessor(
        encryption_key=config.paths.get('encryption_key')
    )

    # 解析站点筛选条件
    station_filter = get_station_filter(args)
    if station_filter:
        logger.info(f"应用站点筛选: {station_filter}")

    # 初始化数据整合器
    integrator = DataIntegrator(
        output_dir=config.get_output_dir('combined'),
        secure_processor=secure_processor
    )

    # 确定要处理的数据源
    sources_to_process = (
        ['glsnow']
        if 'all' in args.sources
        else args.sources
    )

    # 处理各数据源
    success_count = 0
    for source in sources_to_process:
        if process_source(source, integrator, secure_processor, args, station_filter):
            success_count += 1

    # 整合结果
    if success_count > 0:
        logger.info("开始整合结果数据")
        try:
            master_output = integrator.save_master_excel()
            logger.info(f"主输出文件已生成: {master_output}")

            # 生成统计报告
            report = integrator.generate_report()
            logger.info(f"数据处理统计:\n{report}")
        except Exception as e:
            logger.error(f"结果整合失败: {str(e)}")
    else:
        logger.warning("没有成功处理的数据源，跳过整合步骤")

    # 清理临时文件
    if secure_processor:
        secure_processor.clean_secure_tempfiles()

    logger.info(f"处理完成，成功处理 {success_count}/{len(sources_to_process)} 个数据源")


if __name__ == "__main__":
    main()
