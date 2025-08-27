import os
import shutil
from pathlib import Path
import pandas as pd
import gc

from tricys.simulation import run_simulation

# Path to the real modelica package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
MODEL_PATH = os.path.join(project_root, "example", "example_model", "package.mo")
# Make sure the path is in POSIX format for Modelica
MODEL_PATH_POSIX = Path(MODEL_PATH).as_posix()

# Define an output directory based on the test file's name
OUTPUT_DIR = os.path.splitext(__file__)[0]


def get_base_config():
    """Creates a base configuration dictionary for tests."""
    results_dir = os.path.join(OUTPUT_DIR, "results")
    temp_dir = os.path.join(OUTPUT_DIR, "temp")
    return {
        "paths": {
            "package_path": MODEL_PATH_POSIX,
            "db_path": os.path.join(OUTPUT_DIR, "params.db"),
            "results_dir": results_dir,
            "temp_dir": temp_dir,
        },
        "logging": {
            "log_level": "DEBUG",
            "log_to_console": False,
            "log_dir": os.path.join(OUTPUT_DIR, "logs"),
        },
        "simulation": {
            "model_name": "example_model.Cycle",
            "variableFilter": "time|sds\\.I\\[1\\]",
            "stop_time": 1.0,  # Use a very short time for testing
            "step_size": 1.0,
            "max_workers": 2,
            "keep_temp_files": True,
            "concurrent": True,
        },
        "simulation_parameters": {},
    }


def setup_and_teardown(request):
    """Fixture to create the test output directory.

    A finalizer is added to clean up the directory ONLY if the test passes.
    """
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    def cleanup():
        gc.collect()  # Ensure all file handles are released
        if getattr(request.node, "test_passed", False):
            print(f"\nTest passed. Cleaning up test directory: {OUTPUT_DIR}")
            if os.path.exists(OUTPUT_DIR):
                shutil.rmtree(OUTPUT_DIR)
        else:
            print(f"\nTest failed. Intermediate files kept at: {OUTPUT_DIR}")

    request.addfinalizer(cleanup)


def test_simulation_single_run(request):
    """
    Tests a single simulation run using the real example model.
    """
    setup_and_teardown(request)
    config = get_base_config()
    config["simulation_parameters"] = {"blanket.TBR": 1.1}

    results_dir = Path(config["paths"]["results_dir"])

    run_simulation(
        config=config,
        package_path=config["paths"]["package_path"],
        results_dir=str(results_dir),
        temp_dir=config["paths"]["temp_dir"],
    )

    # Assertions
    result_file = results_dir / "simulation_results.csv"
    assert result_file.exists()
    df = pd.read_csv(result_file)
    assert "time" in df.columns
    assert "sds.I[1]" in df.columns
    assert len(df) > 0

    # If all assertions pass, mark test as passed for cleanup
    request.node.test_passed = True


def test_simulation_sweep_run(request):
    """
    Tests a parameter sweep simulation run using the real example model.
    """
    setup_and_teardown(request)
    config = get_base_config()
    config["simulation"]["stop_time"] = 2.0  # A bit longer to see changes
    config["simulation_parameters"] = {
        "blanket.TBR": "1.10:1.20:0.1",  # 2 values: 1.10, 1.20
        "blanket.T": [10, 20],  # 2 values
    }  # This will result in 2*2=4 jobs

    results_dir = Path(config["paths"]["results_dir"])
    temp_dir = Path(config["paths"]["temp_dir"])

    run_simulation(
        config=config,
        package_path=config["paths"]["package_path"],
        results_dir=str(results_dir),
        temp_dir=str(temp_dir),
    )

    # Assertions
    result_file = results_dir / "sweep_results.csv"
    assert result_file.exists()

    import re

    df = pd.read_csv(result_file)
    assert "time" in df.columns

    # Check for the 4 job columns using a regex to be more flexible
    pattern = re.compile(r"blanket\.TBR=[\d.]+_blanket\.T=[\d.]+")
    matching_columns = [col for col in df.columns if pattern.match(col)]

    assert len(matching_columns) == 4
    assert len(df) > 0

    # Check that intermediate files are kept
    assert len(list(temp_dir.glob("*.csv"))) >= 4

    # If all assertions pass, mark test as passed for cleanup
    request.node.test_passed = True
