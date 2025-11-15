import functools
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict

from pythonjsonlogger import jsonlogger

logger = logging.getLogger(__name__)


def delete_old_logs(log_path: str, max_files: int) -> None:
    """Deletes the oldest log files in a directory to meet a specified limit.

    Checks the number of `.log` files in the given directory and removes the
    oldest ones based on modification time until the file count matches the
    `max_files` limit.

    Args:
        log_path: The path to the directory containing log files.
        max_files: The maximum number of `.log` files to retain.

    Note:
        Only processes files with .log extension. Sorts by modification time
        (oldest first) before deletion. Does nothing if current count <= max_files.
    """
    log_files = [
        os.path.join(log_path, f) for f in os.listdir(log_path) if f.endswith(".log")
    ]

    if len(log_files) > max_files:
        # Sort by modification time, oldest first
        log_files.sort(key=os.path.getmtime)

        # Calculate how many files to delete
        files_to_delete_count = len(log_files) - max_files

        # Delete the oldest files
        for i in range(files_to_delete_count):
            os.remove(log_files[i])


def setup_logging(
    config: Dict[str, Any], original_config: Dict[str, Any] = None
) -> None:
    """Configures the logging module based on the application configuration.

    Args:
        config: The main configuration dictionary containing logging settings.
        original_config: Optional original configuration for additional logging.

    Note:
        Sets up JSON formatted logging to console and/or file. Manages log file rotation
        via delete_old_logs(). Supports main_log_path for analysis cases. Logs both
        runtime and original configurations in compact JSON format. Clears existing
        handlers to prevent duplicates.
    """
    log_config = config.get("logging", {})
    log_level_str = log_config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_to_console = log_config.get("log_to_console", True)
    run_timestamp = config.get("run_timestamp")

    log_dir_path = config.get("paths", {}).get("log_dir")
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
        if original_config:
            logger.info(
                f"Original Configuration (compact JSON): {json.dumps(original_config, separators=(',', ':'), ensure_ascii=False)}"
            )


def log_execution_time(func: Callable) -> Callable:
    """A decorator to log the execution time of a function.

    Args:
        func: The function to be decorated.

    Returns:
        The wrapped function that logs execution time.

    Note:
        Measures execution time using time.perf_counter(). Logs function name,
        module, and duration in milliseconds. Uses structured logging with extra fields.
    """

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


def restore_configs_from_log(
    timestamp: str,
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    """Finds the log file for a given timestamp and restores configurations.

    Args:
        timestamp: The timestamp directory name to search for log files.

    Returns:
        A tuple of (runtime_config, original_config) or (None, None) if not found.

    Note:
        Searches in timestamp/simulation_{timestamp}.log and timestamp/log/ directory.
        Parses JSON log entries to find "Runtime Configuration" and "Original Configuration"
        messages. Returns parsed configurations as dictionaries.
    """
    log_file_path = None
    # Define potential locations for the log file
    search_paths = [
        os.path.join(timestamp, f"simulation_{timestamp}.log"),  # analysis style
        os.path.join(timestamp, "log"),  # simulation style
    ]

    for path in search_paths:
        if os.path.isfile(path):
            log_file_path = path
            break
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.startswith("simulation_") and f.endswith(".log"):
                    log_file_path = os.path.join(path, f)
                    break
            if log_file_path:
                break

    if not log_file_path:
        print(
            f"ERROR: Main log file not found for timestamp {timestamp}", file=sys.stderr
        )
        return None, None

    runtime_config_str = None
    original_config_str = None
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if "message" in log_entry:
                        if log_entry["message"].startswith(
                            "Runtime Configuration (compact JSON):"
                        ):
                            runtime_config_str = log_entry["message"].replace(
                                "Runtime Configuration (compact JSON): ", ""
                            )
                        elif log_entry["message"].startswith(
                            "Original Configuration (compact JSON):"
                        ):
                            original_config_str = log_entry["message"].replace(
                                "Original Configuration (compact JSON): ", ""
                            )
                    if runtime_config_str and original_config_str:
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"ERROR: Failed to read log file {log_file_path}: {e}", file=sys.stderr)
        return None, None

    if not runtime_config_str or not original_config_str:
        print(
            "ERROR: Could not find runtime and/or original configuration in log file.",
            file=sys.stderr,
        )
        return None, None

    try:
        runtime_config = json.loads(runtime_config_str)
        original_config = json.loads(original_config_str)
        return runtime_config, original_config
    except json.JSONDecodeError as e:
        print(
            f"ERROR: Failed to parse configuration from log file: {e}", file=sys.stderr
        )
        return None, None
