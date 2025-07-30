"""该模块为应用程序提供了一个单例配置管理器。"""

import json
import logging
import os
from threading import Lock
from typing import Any, Dict

# Configure logger
logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """
    一个用于创建单例类的线程安全元类。
    """

    _instances: Dict[type, Any] = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        """
        在实例化单例类时调用此方法。
        它确保只创建一个类的实例。
        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class ConfigManager(metaclass=SingletonMeta):
    """
    一个单例类，用于管理来自JSON文件的配置设置。
    它确保配置只加载一次，并提供对配置数据的全局访问点。
    """

    _config: Dict[str, Any] = {}
    _initialized: bool = False

    def __init__(self):
        """
        初始化ConfigManager。初始化逻辑将只运行一次。
        """
        if self._initialized:
            return

        # Define project root relative to this file's location (src/manager/config_manager.py)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        config_path = os.path.join(project_root, "config.json")

        try:
            logger.info(f"Loading configuration from: {config_path}")
            with open(config_path, "r") as f:
                self._config = json.load(f)
            logger.info("Configuration loaded successfully.")
        except FileNotFoundError:
            logger.error(
                f"FATAL: Configuration file not found at '{config_path}'. Please ensure it exists."
            )
            raise
        except json.JSONDecodeError as e:
            logger.error(f"FATAL: Error decoding JSON from '{config_path}': {e}")
            raise
        except Exception as e:
            logger.error(
                f"FATAL: An unexpected error occurred while loading the config: {e}"
            )
            raise

        self._initialized = True

    def get(self, key: str, default: Any = None) -> Any:
        """
        通过键从配置中检索值。
        支持使用点表示法（例如，'paths.output_dir'）的嵌套键。

        参数:
            key (str): 要检索的值的键。
            default (Any, optional): 如果找不到键，则返回的默认值。

        返回:
            Any: 配置值或默认值。
        """
        try:
            # Start with the full config dictionary
            value = self._config
            # Traverse the dictionary using the keys separated by dots
            for k in key.split("."):
                value = value[k]
            return value
        except KeyError:
            logger.warning(
                f"Key '{key}' not found in config. Returning default value: {default}"
            )
            return default

    @property
    def all_config(self) -> Dict[str, Any]:
        """
        返回整个配置字典的副本。

        返回:
            Dict[str, Any]: 配置的副本。
        """
        return self._config.copy()


# A global instance for easy, consistent access across the project.
config_manager = ConfigManager()

# Example of how to use this in other files:
#
# from src.manager.config_manager import config_manager
#
# # Get a nested value
# output_dir = config_manager.get("paths.output_dir")
# default_model = config_manager.get("simulation.model_name")
#
# # Get a value with a default if it might be missing
# author = config_manager.get("project.author", default="Unknown")
#
# print(f"Output will be saved to: {output_dir}")
