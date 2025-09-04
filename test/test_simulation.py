import gc
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from tricys.simulation import run_simulation

# --- Test Setup ---

# Resolve project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
MODEL_PATH = os.path.join(project_root, "example", "gui", "example_model", "package.mo")

# Use POSIX paths for cross-platform compatibility in configs
MODEL_PATH_POSIX = Path(MODEL_PATH).as_posix()

# Define a base output directory for all tests in this file
BASE_OUTPUT_DIR = os.path.splitext(__file__)[0]


def get_base_config(test_output_dir: Path):
    """Creates a base configuration dictionary for co-simulation tests."""
    return {
        "paths": {
            "package_path": MODEL_PATH_POSIX,
            "results_dir": test_output_dir / "results",
            "temp_dir": test_output_dir / "temp",
        },
        "logging": {
            "log_level": "WARNING",  # Keep logs quiet during tests
            "log_to_console": False,
            "log_dir": test_output_dir / "logs",
        },
        "simulation": {
            "model_name": "example_model.Cycle",
            "variableFilter": r"time|sds\.I\[1\]",
            "stop_time": 1.0,  # Use short stop time for tests
            "step_size": 0.5,
            "max_workers": 2,
            "keep_temp_files": True,
        },
        "co_simulation": [
            {
                "submodel_name": "example_model.DIV",
                "instance_name": "div",
                "handler_module": "tricys.handlers.div_handler",
                "handler_function": "run_div_simulation",
                "params": {"dummy_value": 2.5},
            }
        ],
        "simulation_parameters": {},
    }


@pytest.fixture(scope="function")
def setup_and_teardown(request):
    """
    Fixture to create a unique directory for each test function and handle cleanup.
    """
    # Create a unique directory based on the test function's name
    test_name = request.node.name
    test_output_dir = Path(BASE_OUTPUT_DIR) / test_name

    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir)

    # Yield the unique directory path to the test function
    yield test_output_dir

    # Cleanup logic runs after the test function completes
    gc.collect()  # Ensure all file handles are released
    if not getattr(request.node, "test_passed", False):
        print(
            f"\nTest '{test_name}' failed. Intermediate files kept at: {test_output_dir}"
        )
    else:
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)


@pytest.fixture(scope="module", autouse=True)
def final_cleanup():
    """
    A module-scoped fixture to clean up the base output directory
    after all tests in this file have run.
    """
    # Let all tests run
    yield
    # This cleanup logic runs once after all tests in the file are complete.
    gc.collect()
    if Path(BASE_OUTPUT_DIR).exists():
        print(f"\nAll tests finished. Cleaning up base directory: {BASE_OUTPUT_DIR}")
        try:
            shutil.rmtree(BASE_OUTPUT_DIR)
        except OSError as e:
            print(f"Error during final cleanup of {BASE_OUTPUT_DIR}: {e}")


# --- Test Cases ---


@pytest.mark.parametrize("use_cosim", [True, False])
def test_single_run(setup_and_teardown, request, use_cosim):
    """
    Tests the run_simulation orchestrator for a single job.
    This test is parameterized to run with and without co-simulation.
    """
    test_output_dir = setup_and_teardown
    config = get_base_config(test_output_dir)
    config["simulation"]["concurrent"] = False
    config["run_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not use_cosim:
        # For non-integration test, remove co-simulation config
        del config["co_simulation"]

    run_simulation(config)

    base_results_dir = Path(config["paths"]["results_dir"])
    run_results_dir = base_results_dir / config["run_timestamp"]
    assert run_results_dir.is_dir(), "Timestamped result directory not found"
    result_files = list(run_results_dir.glob("simulation_result*.csv"))
    assert len(result_files) == 1, "Expected one final result file"

    request.node.test_passed = True


@pytest.mark.parametrize("concurrent", [True, False])
@pytest.mark.parametrize("use_cosim", [True, False])
def test_batch_run(setup_and_teardown, request, concurrent, use_cosim):
    """
    Tests that a batch (sweep) simulation runs and merges results correctly.
    This test is parameterized to cover:
    - Concurrent and non-concurrent runs
    - With and without co-simulation
    """
    test_output_dir = setup_and_teardown
    config = get_base_config(test_output_dir)
    config["simulation_parameters"] = {"blanket.TBR": [1.05, 1.10]}
    config["simulation"]["concurrent"] = concurrent
    config["simulation"]["keep_temp_files"] = True  # Keep temp files for this test
    config["run_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not use_cosim:
        # For non-integration test, remove co-simulation config
        del config["co_simulation"]

    # --- Run Test ---
    run_simulation(config)

    # --- Assertions ---
    base_results_dir = Path(config["paths"]["results_dir"])
    run_results_dir = base_results_dir / config["run_timestamp"]

    # 1. Assert that the final timestamped results directory and merged file exist
    assert run_results_dir.is_dir(), "Timestamped result directory not found"
    final_sweep_file = run_results_dir / "sweep_results.csv"
    assert final_sweep_file.is_file(), "Final sweep_results.csv not found"

    # 2. Assert the content of the merged file is correct
    df = pd.read_csv(final_sweep_file)
    assert (
        len(df.columns) == 3
    ), f"Expected 3 columns (time + 2 jobs), but got {len(df.columns)}"
    assert "blanket.TBR=1.05" in df.columns
    assert "blanket.TBR=1.1" in df.columns

    # 3. Assert that the temp directory was correctly created and kept
    temp_dir = Path(config["paths"]["temp_dir"])
    run_temp_dir = temp_dir / config["run_timestamp"]
    assert (
        run_temp_dir.is_dir()
    ), "Timestamped temp directory should exist because keep_temp_files is True"

    request.node.test_passed = True
