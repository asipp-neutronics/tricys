import gc
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# --- Test Setup ---

# Resolve project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
MODEL_PATH = os.path.join(project_root, "example", "example_model", "package.mo")

# Use POSIX paths for cross-platform compatibility in configs
MODEL_PATH_POSIX = Path(MODEL_PATH).as_posix()

# Define a base output directory for all tests in this file
BASE_OUTPUT_DIR = os.path.splitext(__file__)[0]


def get_base_analysis_config(test_output_dir: Path):
    """Creates a base configuration dictionary for analysis tests."""
    return {
        "paths": {
            "package_path": MODEL_PATH_POSIX,
            "results_dir": str(test_output_dir / "results"),
            "temp_dir": str(test_output_dir / "temp"),
            "db_path": str(test_output_dir / "data" / "parameters.db"),
        },
        "logging": {
            "log_level": "WARNING",  # Keep logs quiet during tests
            "log_to_console": False,
            "log_dir": str(test_output_dir / "logs"),
        },
        "simulation": {
            "model_name": "example_model.Cycle",
            "variableFilter": r"time|sds\.I\[1\]",
            "stop_time": 10.0,  # Use short stop time for tests
            "step_size": 1.0,
            "max_workers": 2,
            "keep_temp_files": True,
            "concurrent": False,
        },
        "simulation_parameters": {"i_iss.T": 18.0},
        "sensitivity_analysis": {
            "enabled": True,
            "analysis_cases": [
                {
                    "name": "Test_Analysis",
                    "independent_variable": "plasma1.fb",
                    "independent_variable_sampling": [0.09],
                    "dependent_variables": [
                        "Startup_Inventory",
                        "Self_Sufficiency_Time",
                    ],
                    "plot_type": "bar",
                    "combine_plots": True,
                    "sweep_time": False,
                }
            ],
            "metrics_definition": {
                "Startup_Inventory": {
                    "source_column": "sds.I[1]",
                    "method": "calculate_startup_inventory",
                },
                "Self_Sufficiency_Time": {
                    "source_column": "sds.I[1]",
                    "method": "time_of_turning_point",
                },
            },
        },
        "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
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
        request.node.parent.failed = True
        print(
            f"\nTest '{test_name}' failed. Intermediate files kept at: {test_output_dir}"
        )
    else:
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)


@pytest.fixture(scope="module", autouse=True)
def final_cleanup(request):
    """
    A module-scoped fixture to clean up the base output directory
    after all tests in this file have run.
    """
    # Let all tests run
    yield
    # This cleanup logic runs once after all tests in the file are complete.
    if getattr(request.node, "failed", False):
        print(
            f"\nOne or more tests failed. Skipping final cleanup of base directory: {BASE_OUTPUT_DIR}"
        )
        return

    gc.collect()
    if Path(BASE_OUTPUT_DIR).exists():
        print(f"\nAll tests finished. Cleaning up base directory: {BASE_OUTPUT_DIR}")
        try:
            shutil.rmtree(BASE_OUTPUT_DIR)
        except OSError as e:
            print(f"Error during final cleanup of {BASE_OUTPUT_DIR}: {e}")


@pytest.mark.parametrize("analysis_format", ["single_object", "list_format"])
def test_analysis_single_case(setup_and_teardown, request, analysis_format):
    """Tests real analysis simulation with single analysis case."""
    test_output_dir = setup_and_teardown
    config = get_base_analysis_config(test_output_dir)

    # Configure for real simulation (remove mocking)
    config["simulation"]["stop_time"] = 100.0  # Longer simulation for real analysis
    config["simulation"]["step_size"] = 10.0
    config["simulation"]["concurrent"] = False
    config["simulation"]["keep_temp_files"] = True

    # Format analysis_cases based on parameter
    if analysis_format == "single_object":
        config["sensitivity_analysis"]["analysis_cases"] = {
            "name": "Real_Single_Analysis",
            "independent_variable": "plasma1.fb",
            "independent_variable_sampling": [0.09],
            "dependent_variables": ["Startup_Inventory", "Self_Sufficiency_Time"],
            "plot_type": "bar",
            "combine_plots": True,
            "sweep_time": False,
        }
    else:  # list_format
        config["sensitivity_analysis"]["analysis_cases"] = [
            {
                "name": "Real_List_Analysis",
                "independent_variable": "plasma1.fb",
                "independent_variable_sampling": [0.09],
                "dependent_variables": ["Startup_Inventory", "Self_Sufficiency_Time"],
                "plot_type": "bar",
                "combine_plots": True,
                "sweep_time": False,
            }
        ]

    # Change to test directory for analysis_cases execution
    original_cwd = os.getcwd()
    os.chdir(test_output_dir)

    try:
        from tricys.simulation_analysis import run_simulation

        run_simulation(config)

        # Check analysis_cases directory structure
        analysis_cases_dir = (
            test_output_dir / "analysis_cases" / config["run_timestamp"]
        )
        if analysis_format == "single_object":
            case_dir = analysis_cases_dir / "Real_Single_Analysis"
        else:
            case_dir = analysis_cases_dir / "Real_List_Analysis"

        assert case_dir.exists(), f"Analysis case directory not found: {case_dir}"

        # Check results directory
        case_results_dir = case_dir / "results"
        assert case_results_dir.exists(), "Case results directory not found"

        # Check for result files
        result_files = list(case_results_dir.glob("simulation_result*.csv"))
        assert len(result_files) >= 1, "Expected at least one result file"

        # Check for analysis summary if sensitivity analysis is enabled
        summary_files = list(case_results_dir.glob("sensitivity_analysis_summary*.csv"))
        assert len(summary_files) >= 1, "Expected sensitivity analysis summary file"

        # Verify config file was created
        config_file = case_dir / "config.json"
        assert config_file.exists(), "Config file not found in case directory"

        request.node.test_passed = True

    finally:
        os.chdir(original_cwd)


@pytest.mark.parametrize("concurrent", [True, False])
def test_analysis_multiple_cases(setup_and_teardown, request, concurrent):
    """Tests real analysis simulation with multiple analysis cases."""
    test_output_dir = setup_and_teardown
    config = get_base_analysis_config(test_output_dir)

    # Configure for real simulation
    config["simulation"]["stop_time"] = 100.0  # Longer simulation for real analysis
    config["simulation"]["step_size"] = 10.0
    config["simulation"]["concurrent"] = concurrent
    config["simulation"]["max_workers"] = 2
    config["simulation"]["keep_temp_files"] = True

    # Configure multiple analysis cases
    config["sensitivity_analysis"]["analysis_cases"] = [
        {
            "name": "Case_A_Real",
            "independent_variable": "plasma1.fb",
            "independent_variable_sampling": [0.09],
            "dependent_variables": ["Startup_Inventory"],
            "plot_type": "bar",
            "combine_plots": False,
            "sweep_time": False,
        },
        {
            "name": "Case_B_Real",
            "independent_variable": "i_iss.T",
            "independent_variable_sampling": [18.0],
            "dependent_variables": ["Self_Sufficiency_Time"],
            "plot_type": "line",
            "combine_plots": True,
            "sweep_time": False,
        },
    ]

    # Change to test directory for analysis_cases execution
    original_cwd = os.getcwd()
    os.chdir(test_output_dir)

    try:
        from tricys.simulation_analysis import run_simulation

        run_simulation(config)

        # Check analysis_cases directory structure
        analysis_cases_dir = (
            test_output_dir / "analysis_cases" / config["run_timestamp"]
        )
        assert analysis_cases_dir.exists(), "Analysis cases directory not found"

        # Check both case directories
        case_a_dir = analysis_cases_dir / "Case_A_Real"
        case_b_dir = analysis_cases_dir / "Case_B_Real"

        assert case_a_dir.exists(), "Case A directory not found"
        assert case_b_dir.exists(), "Case B directory not found"

        # Check results for both cases
        for case_dir, case_name in [
            (case_a_dir, "Case_A_Real"),
            (case_b_dir, "Case_B_Real"),
        ]:
            case_results_dir = case_dir / "results"
            assert (
                case_results_dir.exists()
            ), f"Results directory not found for {case_name}"

            result_files = list(case_results_dir.glob("simulation_result*.csv"))
            assert len(result_files) >= 1, f"Expected result file for {case_name}"

            summary_files = list(
                case_results_dir.glob("sensitivity_analysis_summary*.csv")
            )
            assert len(summary_files) >= 1, f"Expected summary file for {case_name}"

            config_file = case_dir / "config.json"
            assert config_file.exists(), f"Config file not found for {case_name}"

        # Check for report file
        report_files = list(
            (analysis_cases_dir).glob(f"execution_report_{config['run_timestamp']}.md")
        )
        assert len(report_files) >= 1, "Expected analysis cases report file"

        request.node.test_passed = True

    finally:
        os.chdir(original_cwd)


def test_analysis_with_parameter_sweep(setup_and_teardown, request):
    """Tests real analysis simulation with parameter sweep."""
    test_output_dir = setup_and_teardown
    config = get_base_analysis_config(test_output_dir)

    # Configure for real simulation with parameter sweep
    config["simulation"]["stop_time"] = 100.0
    config["simulation"]["step_size"] = 10.0
    config["simulation"]["concurrent"] = False
    config["simulation"]["keep_temp_files"] = True

    # Configure analysis case with multiple sampling points
    config["sensitivity_analysis"]["analysis_cases"] = {
        "name": "Parameter_Sweep_Analysis",
        "independent_variable": "plasma1.fb",
        "independent_variable_sampling": [0.08, 0.09, 0.10],  # Multiple points
        "dependent_variables": ["Startup_Inventory", "Self_Sufficiency_Time"],
        "plot_type": "line",
        "combine_plots": True,
        "sweep_time": ["sds.I[1]"],  # Enable sweep time plotting
    }

    # Change to test directory for analysis_cases execution
    original_cwd = os.getcwd()
    os.chdir(test_output_dir)

    try:
        from tricys.simulation_analysis import run_simulation

        run_simulation(config)

        # Check analysis_cases directory structure
        analysis_cases_dir = (
            test_output_dir / "analysis_cases" / config["run_timestamp"]
        )
        case_dir = analysis_cases_dir / "Parameter_Sweep_Analysis"

        assert case_dir.exists(), "Analysis case directory not found"

        # Check results directory
        case_results_dir = case_dir / "results"
        assert case_results_dir.exists(), "Case results directory not found"

        # Check for sweep results (should have multiple jobs)
        sweep_files = list(case_results_dir.glob("sweep_results*.csv"))
        assert len(sweep_files) >= 1, "Expected sweep results file"

        # Verify sweep results contain multiple parameter combinations
        sweep_df = pd.read_csv(sweep_files[0])
        assert (
            len(sweep_df.columns) >= 4
        ), "Expected multiple parameter columns in sweep results"  # time + 3 parameters

        # Check for analysis summary
        summary_files = list(case_results_dir.glob("sensitivity_analysis_summary*.csv"))
        assert len(summary_files) >= 1, "Expected sensitivity analysis summary file"

        # Verify summary contains analysis results
        summary_df = pd.read_csv(summary_files[0])
        assert (
            "plasma1.fb" in summary_df.columns
        ), "Independent variable not found in summary"
        assert len(summary_df) == 3, "Expected 3 rows for 3 parameter values"

        # Check for sweep time plot if enabled
        plot_files = list(case_results_dir.glob("*sweep*.png"))
        if config["sensitivity_analysis"]["analysis_cases"]["sweep_time"]:
            assert len(plot_files) >= 1, "Expected sweep time series plot"

        request.node.test_passed = True

    finally:
        os.chdir(original_cwd)


def test_analysis_with_bisection_search(setup_and_teardown, request):
    """Tests real analysis simulation with optimization (Required_TBR)."""
    test_output_dir = setup_and_teardown
    config = get_base_analysis_config(test_output_dir)

    # Configure for real simulation with optimization
    config["simulation"]["stop_time"] = 1000.0  # Longer simulation for optimization
    config["simulation"]["step_size"] = 100.0
    config["simulation"]["concurrent"] = False
    config["simulation"]["keep_temp_files"] = True

    # Configure analysis case with optimization
    config["sensitivity_analysis"]["analysis_cases"] = {
        "name": "Optimization_Analysis",
        "independent_variable": "plasma1.fb",
        "independent_variable_sampling": [0.09],
        "dependent_variables": [
            "Startup_Inventory",
            "Self_Sufficiency_Time",
            "Required_TBR",
        ],
        "plot_type": "bar",
        "combine_plots": True,
        "sweep_time": False,
    }

    # Add Required_TBR optimization configuration
    config["sensitivity_analysis"]["metrics_definition"]["Required_TBR"] = {
        "method": "bisection_search",
        "parameter_to_optimize": "blanket.TBR",
        "search_range": [0.8, 1.5],
        "tolerance": 0.01,
        "max_iterations": 5,  # Reduced for faster testing
    }

    # Change to test directory for analysis_cases execution
    original_cwd = os.getcwd()
    os.chdir(test_output_dir)

    try:
        from tricys.simulation_analysis import run_simulation

        run_simulation(config)

        # Check analysis_cases directory structure
        analysis_cases_dir = (
            test_output_dir / "analysis_cases" / config["run_timestamp"]
        )
        case_dir = analysis_cases_dir / "Optimization_Analysis"

        assert case_dir.exists(), "Analysis case directory not found"

        # Check results directory
        case_results_dir = case_dir / "results"
        assert case_results_dir.exists(), "Case results directory not found"

        # Check for optimization summary
        optimization_files = list(case_results_dir.glob("requierd_tbr_summary*.csv"))
        assert len(optimization_files) >= 1, "Expected optimization summary file"

        # Verify optimization results
        opt_df = pd.read_csv(optimization_files[0])
        assert (
            "Required_TBR" in opt_df.columns
        ), "Required_TBR not found in optimization results"
        assert (
            "Required_Self_Sufficiency_Time" in opt_df.columns
        ), "Required_Self_Sufficiency_Time not found"

        # Check for sensitivity analysis summary with merged results
        summary_files = list(case_results_dir.glob("sensitivity_analysis_summary*.csv"))
        assert len(summary_files) >= 1, "Expected sensitivity analysis summary file"

        # Verify merged summary contains both analysis and optimization results
        summary_df = pd.read_csv(summary_files[0])
        assert (
            "plasma1.fb" in summary_df.columns
        ), "Independent variable not found in summary"
        assert (
            "Required_TBR" in summary_df.columns
        ), "Required_TBR not found in merged summary"

        request.node.test_passed = True

    finally:
        os.chdir(original_cwd)


def test_analysis_file_based_sampling(setup_and_teardown, request):
    """Tests real analysis simulation with file-based sampling from a CSV, where each row is a job."""
    test_output_dir = setup_and_teardown
    config = get_base_analysis_config(test_output_dir)

    # Create test data file with multiple parameters per job (row)
    data_file = test_output_dir / "sampling_data.csv"
    sampling_data = pd.DataFrame(
        {"plasma1.fb": [0.08, 0.09, 0.10], "i_iss.T": [18.0, 19.0, 20.0]}
    )
    sampling_data.to_csv(data_file, index=False)

    # Configure for real simulation with file-based sampling
    config["simulation"]["stop_time"] = 100.0
    config["simulation"]["step_size"] = 10.0
    config["simulation"]["concurrent"] = False
    config["simulation"]["keep_temp_files"] = True

    # No base simulation parameters needed as they come from the file
    config["simulation_parameters"] = {}

    # Configure analysis case for file-based sampling
    config["sensitivity_analysis"]["analysis_cases"] = {
        "name": "File_Based_Analysis",
        "independent_variable": "file",
        "independent_variable_sampling": data_file.as_posix(),
        "dependent_variables": ["Startup_Inventory", "Self_Sufficiency_Time"],
        "plot_type": "line",
        "combine_plots": True,
        "sweep_time": False,
    }

    # Change to test directory for analysis_cases execution
    original_cwd = os.getcwd()
    os.chdir(test_output_dir)

    try:
        from tricys.simulation_analysis import run_simulation

        run_simulation(config)

        # Check analysis_cases directory structure
        analysis_cases_dir = (
            test_output_dir / "analysis_cases" / config["run_timestamp"]
        )
        case_dir = analysis_cases_dir / "File_Based_Analysis"
        assert case_dir.exists(), "Analysis case directory not found"

        # Check results directory
        case_results_dir = case_dir / "results"
        assert case_results_dir.exists(), "Case results directory not found"

        # Check for sweep results (should have multiple jobs from file)
        sweep_files = list(case_results_dir.glob("sweep_results*.csv"))
        assert (
            len(sweep_files) >= 1
        ), "Expected sweep results file for file-based sampling"

        # Verify sweep results contain data from file (3 jobs)
        sweep_df = pd.read_csv(sweep_files[0])
        # 1 'time' column + 1 var ('sds.I[1]') * 3 jobs = 4 columns
        assert (
            len(sweep_df.columns) == 4
        ), f"Expected 4 columns from file sampling (time + 3 jobs), but got {len(sweep_df.columns)}"

        # Check for analysis summary
        summary_files = list(case_results_dir.glob("sensitivity_analysis_summary*.csv"))
        assert len(summary_files) >= 1, "Expected sensitivity analysis summary file"

        # Verify summary contains file-based sampling results
        summary_df = pd.read_csv(summary_files[0])
        assert len(summary_df) == 3, "Expected 3 rows for 3 jobs in the file"
        # When independent_variable is 'file', the columns from the input CSV should be in the summary
        assert (
            "plasma1.fb" in summary_df.columns
        ), "Parameter 'plasma1.fb' not found in summary"
        assert (
            "i_iss.T" in summary_df.columns
        ), "Parameter 'i_iss.T' not found in summary"

        request.node.test_passed = True

    finally:
        os.chdir(original_cwd)
