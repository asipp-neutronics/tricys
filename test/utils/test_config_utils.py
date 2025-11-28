import os
import shutil
from pathlib import Path

import pytest

from tricys.utils.config_utils import (
    analysis_validate_analysis_cases_config,
    basic_validate_config,
    convert_relative_paths_to_absolute,
)

TEST_DIR = "temp_config_utils_test"


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Set up and tear down the test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def test_convert_relative_paths_to_absolute():
    """Test convert_relative_paths_to_absolute."""
    base_dir = os.path.abspath(TEST_DIR)
    config = {
        "paths": {"package_path": "model.mo"},
        "a_list": [{"log_dir": "logs/"}],
    }

    abs_config = convert_relative_paths_to_absolute(config, base_dir)

    assert os.path.normpath(abs_config["paths"]["package_path"]) == os.path.normpath(
        os.path.join(base_dir, "model.mo")
    )
    assert os.path.normpath(abs_config["a_list"][0]["log_dir"]) == os.path.normpath(
        os.path.join(base_dir, "logs/")
    )


def test_basic_validate_config_success():
    """Test basic_validate_config with a valid config."""
    package_path = Path(TEST_DIR) / "model.mo"
    package_path.touch()

    config = {
        "paths": {"package_path": str(package_path)},
        "simulation": {
            "model_name": "MyModel",
            "stop_time": 10,
            "step_size": 0.1,
            "variableFilter": "time|sub.var",
        },
    }

    try:
        basic_validate_config(config)
    except SystemExit as e:
        pytest.fail(f"Validation failed unexpectedly: {e}")


def test_basic_validate_config_missing_key():
    """Test basic_validate_config with a missing key."""
    config = {"paths": {}}
    with pytest.raises(SystemExit):
        basic_validate_config(config)


def test_analysis_validate_analysis_cases_config_success():
    """Test analysis_validate_analysis_cases_config with a valid config."""
    config = {
        "sensitivity_analysis": {
            "analysis_cases": [
                {
                    "name": "case1",
                    "independent_variable": "p1",
                    "independent_variable_sampling": [1, 2, 3],
                }
            ]
        }
    }
    assert analysis_validate_analysis_cases_config(config)


def test_analysis_validate_analysis_cases_config_fail():
    """Test analysis_validate_analysis_cases_config with an invalid config."""
    config = {"sensitivity_analysis": {}}
    assert not analysis_validate_analysis_cases_config(config)

    config = {"sensitivity_analysis": {"analysis_cases": [{}]}}
    assert not analysis_validate_analysis_cases_config(config)
