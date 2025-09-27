import logging
import logging.config
import logging.handlers
import os
from pathlib import Path
import sys


def setup_logging(log_config_path):
    """安全配置日志系统，解决 KeyError 问题，并添加日志轮转功能"""
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
        # 先读取配置文件内容
        with open(log_config_path, 'r') as f:
            config_content = f.read()

        # 替换文件处理器为轮转文件处理器
        if 'class=FileHandler' in config_content:
            config_content = config_content.replace(
                'class=FileHandler',
                'class=logging.handlers.RotatingFileHandler'
            )
            # 添加轮转参数
            config_content = config_content.replace(
                "args=('%(logdir)s/processing.log', 'a')",
                "args=('%(logdir)s/processing.log', 'a', 10485760, 5)"  # 10MB, 保留5个备份
            )

        # 创建临时配置文件
        temp_config_path = logs_dir / "temp_logging.conf"
        with open(temp_config_path, 'w') as f:
            f.write(config_content)

        # 使用修改后的配置
        logging.config.fileConfig(
            temp_config_path,
            defaults={'logdir': str(logs_dir)},
            disable_existing_loggers=False
        )

        # 删除临时配置文件
        os.remove(temp_config_path)

        logger = logging.getLogger("main")
        logger.info(f"日志系统已从 {log_config_path} 初始化，启用日志轮转功能")
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
    """回退到基本日志配置，使用轮转文件处理器"""
    # 创建安全日志格式
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(name)s - %(message)s')

    # 使用轮转文件处理器 (10MB, 保留5个备份)
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / 'processing.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 移除所有现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 添加新处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 记录错误信息
    logger = logging.getLogger("main")
    logger.error("无法加载日志配置，使用回退配置（带轮转功能）")
    return True


def cleanup_old_logs(logs_dir, max_total_size_mb=100, max_backup_count=10):
    """
    清理旧日志文件

    参数:
        logs_dir: 日志目录路径
        max_total_size_mb: 最大总日志大小(MB)
        max_backup_count: 最大备份文件数量
    """
    try:
        logs_path = Path(logs_dir)
        if not logs_path.exists():
            return

        # 获取所有日志文件
        log_files = list(logs_path.glob("*.log*"))
        log_files.sort(key=os.path.getmtime, reverse=True)  # 按修改时间排序，最新的在前

        # 计算总大小
        total_size = sum(f.stat().st_size for f in log_files)
        max_total_size = max_total_size_mb * 1024 * 1024  # 转换为字节

        # 如果总大小超过限制或文件数量超过限制，删除最旧的文件
        files_to_remove = []

        if total_size > max_total_size:
            # 按从旧到新的顺序计算需要删除哪些文件
            log_files_asc = sorted(log_files, key=os.path.getmtime)  # 按修改时间升序排序，最旧的在前
            current_size = total_size

            for file in log_files_asc:
                if current_size <= max_total_size and len(log_files) - len(files_to_remove) <= max_backup_count:
                    break

                files_to_remove.append(file)
                current_size -= file.stat().st_size

        elif len(log_files) > max_backup_count:
            # 文件数量超过限制，删除最旧的文件
            files_to_remove = log_files[max_backup_count:]

        # 删除文件
        for file in files_to_remove:
            try:
                file.unlink()
                logging.getLogger("LogCleanup").info(f"删除旧日志文件: {file.name}")
            except Exception as e:
                logging.getLogger("LogCleanup").error(f"删除日志文件失败 {file.name}: {str(e)}")

    except Exception as e:
        logging.getLogger("LogCleanup").error(f"清理日志文件失败: {str(e)}")


# 添加定期清理任务（可选）
def setup_log_cleanup_scheduler(logs_dir, interval_hours=24):
    """设置定期日志清理任务"""
    try:
        import threading
        import time

        def cleanup_task():
            while True:
                cleanup_old_logs(logs_dir)
                time.sleep(interval_hours * 3600)  # 等待指定小时

        # 启动后台线程
        thread = threading.Thread(target=cleanup_task, daemon=True)
        thread.start()

        logging.getLogger("LogCleanup").info(f"已启动日志清理任务，每 {interval_hours} 小时运行一次")

    except ImportError:
        logging.getLogger("LogCleanup").warning("无法启动自动日志清理，需要 threading 模块")
    except Exception as e:
        logging.getLogger("LogCleanup").error(f"启动日志清理任务失败: {str(e)}")