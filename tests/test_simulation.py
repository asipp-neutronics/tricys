import numpy as np
from src.simulation import run_parameter_sweep
from src.result_processor import combine_simulation_results


def test_simulation_case1():
    """
    Test simulation with blanket.T and blanket.TBR parameters.
    """
    package_path = "./example/package.mo"
    model_name = "example.Cycle"
    param_A_values = {"blanket.T": [1.0, 1.1, 1.2]}
    param_B_sweep = {"blanket.TBR": np.linspace(1.05, 1.15, 5)}
    stop_time = 5000.0
    step_size = 1.0
    temp_dir = "./data"

    output_csv_files = run_parameter_sweep(
        package_path, model_name, param_A_values, param_B_sweep, stop_time, step_size, temp_dir
    )
    result_path = combine_simulation_results(
        param_A_values, param_B_sweep, temp_dir, output_csv_files)

    assert result_path.endswith(
        ".csv"), f"Expected CSV file, got {result_path}"
    print(f"Test case 1 completed: {result_path}")


if __name__ == "__main__":
    test_simulation_case1()
