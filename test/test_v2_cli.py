import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tricys.main import main as cli_main
from tricys.utils.config_utils import basic_prepare_config

TEST_DIR = Path("temp_v2_cli_test").resolve()


@pytest.fixture(autouse=True)
def setup_teardown():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)


def test_basic_prepare_config_v2_yaml():
    """Verify that basic_prepare_config loads and validates TRICYS 2.0 YAML configuration."""
    model_file = TEST_DIR / "MyModel.mo"
    model_file.touch()

    config_yaml_path = TEST_DIR / "config_v2.yaml"
    v2_data = {
        "model": {
            "fileName": str(model_file),
            "modelName": "MyPackage.MyModel",
        },
        "scenario": {
            "stopTime": 500.0,
            "stepSize": 0.5,
            "method": "dassl",
            "variableFilter": "time|val.*",
        },
        "task": {
            "type": "sweep",
            "parameters": [
                {"name": "CaseA", "TBR": 1.15},
                {"name": "CaseB", "TBR": 1.25},
            ],
            "parallel": True,
            "max_workers": 2,
        },
        "visualizer": {
            "launch_on_finish": False,
        },
    }

    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(v2_data, f)

    runtime_cfg, orig_cfg = basic_prepare_config(str(config_yaml_path))

    assert runtime_cfg["model"]["modelName"] == "MyPackage.MyModel"
    assert runtime_cfg["scenario"]["stopTime"] == 500.0
    assert runtime_cfg["task"]["type"] == "sweep"
    assert len(runtime_cfg["task"]["parameters"]) == 2
    # Check backwards-compatible synthesized paths
    assert "paths" in runtime_cfg
    assert runtime_cfg["paths"]["package_path"] == str(model_file)
    assert os.path.isdir(runtime_cfg["paths"]["results_dir"])


def test_cli_run_v2_yaml_dispatch():
    """Verify that 'tricys run -c config.yaml' dispatches to simulation_main with TRICYS 2.0 config."""
    model_file = TEST_DIR / "TestModel.mo"
    model_file.touch()

    config_yaml_path = TEST_DIR / "config_run.yaml"
    v2_data = {
        "model": {
            "fileName": str(model_file),
            "modelName": "TestModel",
        },
        "scenario": {
            "stopTime": 100.0,
            "stepSize": 1.0,
        },
        "task": {
            "type": "simulate",
        },
    }

    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(v2_data, f)

    test_argv = ["tricys", "run", "-c", str(config_yaml_path), "--csv"]

    with (
        patch("sys.argv", test_argv),
        patch("tricys.main.simulation_main") as mock_sim_main,
    ):
        cli_main()
        assert mock_sim_main.called
        called_config = mock_sim_main.call_args[0][0]
        assert called_config["model"]["modelName"] == "TestModel"
        assert called_config["task"]["type"] == "simulate"
        assert mock_sim_main.call_args[1]["export_csv"] is True


def test_cli_flags_enhanced_and_turbo_v2():
    """Verify that --enhanced and --turbo correctly update TRICYS 2.0 task config."""
    model_file = TEST_DIR / "TestModel.mo"
    model_file.touch()

    config_yaml_path = TEST_DIR / "config_turbo.yaml"
    v2_data = {
        "model": {
            "fileName": str(model_file),
            "modelName": "TestModel",
        },
        "scenario": {
            "stopTime": 100.0,
            "stepSize": 1.0,
        },
        "task": {
            "type": "sweep",
            "parameters": [{"TBR": 1.1}, {"TBR": 1.2}],
            "parallel": False,
        },
    }

    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(v2_data, f)

    test_argv = ["tricys", "run", "-c", str(config_yaml_path), "--enhanced", "--turbo"]

    with (
        patch("sys.argv", test_argv),
        patch("tricys.main.simulation_main") as mock_sim_main,
    ):
        cli_main()
        assert mock_sim_main.called
        called_config = mock_sim_main.call_args[0][0]
        assert called_config["task"]["parallel"] is True
        assert called_config["task"]["max_workers"] >= 1


def test_cli_visualizer_launch_on_finish():
    """Verify that visualizer.launch_on_finish triggers visualizer_main."""
    model_file = TEST_DIR / "TestModel.mo"
    model_file.touch()

    config_yaml_path = TEST_DIR / "config_vis.yaml"
    v2_data = {
        "model": {
            "fileName": str(model_file),
            "modelName": "TestModel",
        },
        "scenario": {
            "stopTime": 10.0,
            "stepSize": 1.0,
        },
        "task": {
            "type": "simulate",
        },
        "visualizer": {
            "port": 8888,
            "launch_on_finish": True,
        },
    }

    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(v2_data, f)

    test_argv = ["tricys", "-c", str(config_yaml_path)]

    with (
        patch("sys.argv", test_argv),
        patch("tricys.main.simulation_main"),
        patch("tricys.visualizer.main.start") as mock_vis_start,
    ):
        cli_main()
        assert mock_vis_start.called


def test_v2_example_04_pulse_width_sweep_config():
    """Verify that 04_sweep_pulse_width_required_tbr.yaml parses cleanly as valid V2 config."""
    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "04_sweep_pulse_width_required_tbr.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "sweep"
    assert len(runtime_cfg["task"]["parameters"]) == 3
    widths = [p["pulseSource.width"] for p in runtime_cfg["task"]["parameters"]]
    assert widths == [100.0, 80.0, 60.0]

    # Verify metrics
    metrics = runtime_cfg["task"]["metrics"]
    assert "startup_inventory" in metrics
    assert "Required_TBR" in metrics
    assert metrics["startup_inventory"]["evaluate"] == "startup_inventory"
    assert metrics["Required_TBR"]["evaluate"] == "bisection_search"
    assert metrics["Required_TBR"]["parameter"] == "blanket.TBR"
