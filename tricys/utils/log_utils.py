import functools
import json
import logging
import os
import sys
import time
from typing import Any, Dict

from pythonjsonlogger import jsonlogger

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

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
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

        # If a main log path is provided (for analysis cases), add it as an additional handler
        main_log_path = log_config.get("main_log_path")
        if main_log_path:
            try:
                # Ensure the directory for the main log exists, just in case
                os.makedirs(os.path.dirname(main_log_path), exist_ok=True)

                main_log_handler = logging.FileHandler(
                    main_log_path, mode="a", encoding="utf-8"
                )
                main_log_handler.setFormatter(formatter)
                root_logger.addHandler(main_log_handler)
                logger.info(f"Also logging to main log file: {main_log_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to attach main log handler for {main_log_path}: {e}"
                )

        logger.info(f"Logging to file: {log_file_path}")
        # Log the full runtime configuration in a compact JSON format
        logger.info(
            f"Runtime Configuration (compact JSON): {json.dumps(config, separators=(',', ':'), ensure_ascii=False)}"
        )


def log_execution_time(func):
    """A decorator to log the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        logger.info(
            "Function executed",
            extra={
                "function_name": func.__name__,
                "function_module": func.__module__,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return result

    return wrapper
