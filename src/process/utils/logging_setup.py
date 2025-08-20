import logging
import logging.config
import os
from pathlib import Path
import sys


def setup_logging(log_config_path):
    """安全配置日志系统，解决 KeyError 问题"""
    # 确保日志目录存在
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 创建临时日志记录器
    temp_logger = logging.getLogger("LogSetup")
    temp_logger.setLevel(logging.INFO)
    temp_handler = logging.StreamHandler(sys.stdout)
    temp_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    temp_logger.addHandler(temp_handler)

    # 检查配置文件是否存在
    if not os.path.exists(log_config_path):
        temp_logger.error(f"日志配置文件不存在: {log_config_path}")
        return _fallback_logging(logs_dir)

    # 验证配置文件
    if not _validate_log_config(log_config_path):
        temp_logger.error(f"日志配置文件无效: {log_config_path}")
        return _fallback_logging(logs_dir)

    # 尝试加载配置
    try:
        logging.config.fileConfig(
            log_config_path,
            defaults={'logdir': str(logs_dir)},
            disable_existing_loggers=False
        )
        logger = logging.getLogger("main")
        logger.info(f"日志系统已从 {log_config_path} 初始化")
        return True
    except Exception as e:
        temp_logger.error(f"加载日志配置失败: {str(e)}")
        return _fallback_logging(logs_dir)


def _validate_log_config(config_path):
    """验证日志配置文件结构"""
    required_sections = [
        'loggers', 'handlers', 'formatters',
        'logger_root', 'handler_fileHandler',
        'handler_consoleHandler', 'formatter_simpleFormatter'
    ]

    try:
        with open(config_path, 'r') as f:
            content = f.read()

        # 检查所有必需部分
        missing = [section for section in required_sections if f"[{section}]" not in content]
        if missing:
            print(f"配置文件缺少部分: {', '.join(missing)}")
            return False

        return True
    except Exception:
        return False


def _fallback_logging(logs_dir):
    """回退到基本日志配置"""
    # 创建安全日志格式
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(name)s - %(message)s')

    # 文件处理器
    file_handler = logging.FileHandler(logs_dir / 'fallback.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 记录错误信息
    logger = logging.getLogger("main")
    logger.error("无法加载日志配置，使用回退配置")
    return False
