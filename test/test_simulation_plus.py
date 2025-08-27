import os
import shutil
from pathlib import Path
import pandas as pd
import gc

from tricys.simulation_plus import run_co_simulation_workflow

# Path to the real modelica package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
MODEL_PATH = os.path.join(project_root, "example", "example_model", "package.mo")
# Make sure the path is in POSIX format for Modelica
MODEL_PATH_POSIX = Path(MODEL_PATH).as_posix()

# Define an output directory based on the test file's name
OUTPUT_DIR = os.path.splitext(__file__)[0]


def get_base_config():
    """Creates a base configuration dictionary for co-simulation tests."""
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
            "variableFilter": "time|sds\\.I\[1\]",
            "stop_time": 1.0,
            "step_size": 1.0,
            "max_workers": 2,
            "keep_temp_files": True,
            "concurrent": True,
        },
        "simulation_parameters": {},
    }


def setup_and_teardown(request):
    """
    Fixture to create output directories and cleanup after co-simulation tests.
    Cleanup happens ONLY if the test passes.
    """
    # Cleanup previous runs
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Define paths for generated models
    model_dir = Path(project_root) / "example" / "example_model"
    interceptor_model_path = model_dir / "DIV_Interceptor.mo"
    intercepted_system_path = model_dir / "Cycle_Intercepted.mo"

    def cleanup():
        gc.collect()  # Ensure all file handles are released
        if getattr(request.node, "test_passed", False):
            print(f"\nTest passed. Cleaning up generated files...")
            if os.path.exists(OUTPUT_DIR):
                shutil.rmtree(OUTPUT_DIR)
            if interceptor_model_path.exists():
                os.remove(interceptor_model_path)
            if intercepted_system_path.exists():
                os.remove(intercepted_system_path)
        else:
            print(f"\nTest failed. Intermediate files kept.")
            print(f"Output dir: {OUTPUT_DIR}")
            print(f"Generated models: {model_dir}")

    request.addfinalizer(cleanup)


def test_co_simulation_workflow(request):
    """
    Tests the co-simulation workflow using the real example model and a dummy handler.
    """
    setup_and_teardown(request)
    config = get_base_config()
    config["simulation"]["stop_time"] = 1.0
    config["simulation"]["variableFilter"] = "time|div\\.from_plasma\[1\]"

    config["co_simulation"] = [
        {
            "submodel_name": "example_model.DIV",
            "instance_name": "div",
            "handler_module": "tricys.handlers.div_handler",
            "handler_function": "run_div_simulation",
            "params": {"dummy_value": 2.5},
        }
    ]

    job_params = config.get("simulation_parameters", {})
    job_id = 0

    # --- Run Test ---
    run_co_simulation_workflow(config,job_params,job_id)

    # --- Assertions ---
    results_dir = Path(config["paths"]["results_dir"])
    temp_dir = Path(config["paths"]["temp_dir"])
    model_dir = Path(project_root) / "example" / "example_model"
    interceptor_model_path = model_dir / "DIV_Interceptor.mo"
    intercepted_system_path = model_dir / "Cycle_Intercepted.mo"

    assert any(temp_dir.glob("primary_inputs.csv"))
    assert any(temp_dir.glob("div_outputs.csv"))
    assert interceptor_model_path.exists()
    assert intercepted_system_path.exists()

    final_result_files = list(results_dir.glob("co_simulation_*.csv"))
    assert len(final_result_files) > 0
    df = pd.read_csv(final_result_files[0])
    assert "time" in df.columns
    assert "div.from_plasma[1]" in df.columns

    # If all assertions pass, mark test as passed for cleanup
    request.node.test_passed = True