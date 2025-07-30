"""该模块为应用程序提供了一个单例日志管理器。"""

import logging
import os
from datetime import datetime
from threading import Lock
from typing import Any, Dict

from .config_manager import config_manager


class SingletonMeta(type):
    """
    一个用于创建单例类的线程安全元类。
    """

    _instances: Dict[type, Any] = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class LoggerManager(metaclass=SingletonMeta):
    """
    一个单例类，用于管理应用程序的日志记录配置。
    它确保日志记录只设置一次，并根据应用程序的配置提供一致的日志记录环境。
    """

    _initialized: bool = False

    def __init__(self):
        """
        为整个应用程序初始化和配置日志系统。
        由于单例模式，此逻辑仅运行一次。
        """
        if self._initialized:
            return

        is_dev_mode = config_manager.get("logging.is_dev_mode", default=False)

        if not is_dev_mode:
            logging.disable(logging.CRITICAL)
            print("Logging is disabled.")
            self._initialized = True
            return

        log_level_str = config_manager.get("logging.log_level", default="INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_to_console = config_manager.get("logging.log_to_console", default=True)

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        log_dir = os.path.join(
            project_root, config_manager.get("paths.log_dir", default="log")
        )
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"simulation_{timestamp}.log"
        log_file_path = os.path.join(log_dir, log_file_name)

        handlers = []
        file_handler = logging.FileHandler(log_file_path)
        handlers.append(file_handler)

        if log_to_console:
            stream_handler = logging.StreamHandler()
            handlers.append(stream_handler)

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )

        logger = logging.getLogger(__name__)
        logger.info(
            f"Logging initialized. Log level: {log_level_str}. Output file: {log_file_path}"
        )
        self._initialized = True


# Global instance for easy initialization in the main application entry point.
logger_manager = LoggerManager()

# Example of how to use this:
#
# At the very top of your main script (e.g., gui_main.py), you would add:
# from src.manager.logger_manager import logger_manager
#
# Then, in any other module, you can get a logger instance as usual:
# import logging
# logger = logging.getLogger(__name__)
# logger.info("This is a test message.")
