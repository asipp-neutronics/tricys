import json
import logging
import os
import shutil
from pathlib import Path

import pytest

from tricys.utils.log_utils import (
    delete_old_logs,
    restore_configs_from_log,
    setup_logging,
)

TEST_DIR = "temp_log_utils_test"


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Set up and tear down the test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def test_delete_old_logs():
    """Test delete_old_logs."""
    log_dir = Path(TEST_DIR)
    # Create more files than the limit
    for i in range(10):
        (log_dir / f"log_{i}.log").touch()

    delete_old_logs(str(log_dir), 5)

    remaining_files = list(log_dir.glob("*.log"))
    assert len(remaining_files) == 5


def test_setup_logging():
    """Test setup_logging."""
    log_dir = Path(TEST_DIR) / "logs"
    config = {
        "logging": {"log_level": "INFO", "log_to_console": False, "log_count": 2},
        "paths": {"log_dir": str(log_dir)},
        "run_timestamp": "20250101_120000",
    }

    # Create some old logs to test deletion
    log_dir.mkdir()
    (log_dir / "old_log1.log").touch()
    (log_dir / "old_log2.log").touch()

    setup_logging(config)

    log_file = log_dir / "simulation_20250101_120000.log"
    assert log_file.exists()

    remaining_files = list(log_dir.glob("*.log"))
    assert len(remaining_files) <= 3

    logging.shutdown()


def test_restore_configs_from_log():
    """Test restore_configs_from_log."""
    timestamp = "20250101_120000"
    log_dir = Path(TEST_DIR) / timestamp / "log"
    log_dir.mkdir(parents=True)

    log_file = log_dir / f"simulation_{timestamp}.log"

    runtime_config = {"a": 1}
    original_config = {"b": 2}

    # Create a realistic JSON log file
    with open(log_file, "w", encoding="utf-8") as f:
        log_record_rt = {
            "asctime": "2025-01-01 12:00:00,000",
            "name": "tricys.utils.config_utils",
            "levelname": "INFO",
            "message": f"Runtime Configuration (compact JSON): {json.dumps(runtime_config)}",
        }
        log_record_org = {
            "asctime": "2025-01-01 12:00:00,000",
            "name": "tricys.utils.config_utils",
            "levelname": "INFO",
            "message": f"Original Configuration (compact JSON): {json.dumps(original_config)}",
        }
        f.write(json.dumps(log_record_rt) + "\n")
        f.write(json.dumps(log_record_org) + "\n")

    rt_config, org_config = restore_configs_from_log(str(Path(TEST_DIR) / timestamp))

    assert rt_config == runtime_config
    assert org_config == original_config
