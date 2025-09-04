import logging
import os
import sys
from typing import Any, Dict

from tricys.utils.file_utils import delete_old_logs

logger = logging.getLogger(__name__)


def setup_logging(config: Dict[str, Any]):
    """Configures the logging module based on the application configuration."""
    log_config = config.get("logging", {})
    log_level_str = log_config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_to_console = log_config.get("log_to_console", True)
    run_timestamp = config.get("run_timestamp")

    log_dir_path = log_config.get("log_dir")
    log_count = log_config.get("log_count", 5)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers to prevent duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_dir_path:
        abs_log_dir = os.path.abspath(log_dir_path)
        os.makedirs(abs_log_dir, exist_ok=True)
        delete_old_logs(abs_log_dir, log_count)
        log_file_path = os.path.join(abs_log_dir, f"simulation_{run_timestamp}.log")

        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file_path}")
