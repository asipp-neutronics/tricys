import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tricys.analysis.salib import TricysSALibAnalyzer

TEST_DIR = "temp_salib_test"


@pytest.fixture
def base_config():
    """Provides a base configuration for the analyzer."""
    return {
        "paths": {"package_path": "dummy_package.mo", "temp_dir": TEST_DIR},
        "simulation": {"model_name": "dummy_model", "stop_time": 10},
        "simulation_parameters": {"param1": 1.0},
    }


@pytest.fixture
def analyzer(base_config):
    """Provides an instance of the TricysSALibAnalyzer."""
    # Ensure the dummy package file exists
    with open(base_config["paths"]["package_path"], "w") as f:
        f.write("model dummy_model parameter Real param1=1; end dummy_model;")
    return TricysSALibAnalyzer(base_config)


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Set up and tear down the test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    if os.path.exists("dummy_package.mo"):
        os.remove("dummy_package.mo")


def test_define_problem(analyzer):
    """Test the define_problem method."""
    param_bounds = {"x1": [-3.14, 3.14], "x2": [0.0, 1.0]}
    problem = analyzer.define_problem(param_bounds)
    assert problem["num_vars"] == 2
    assert problem["names"] == ["x1", "x2"]
    assert problem["bounds"] == [[-3.14, 3.14], [0.0, 1.0]]


def test_generate_samples(analyzer):
    """Test the generate_samples method."""
    param_bounds = {"x1": [-3.14, 3.14], "x2": [0.0, 1.0]}
    analyzer.define_problem(param_bounds)

    # Test Sobol
    samples = analyzer.generate_samples(method="sobol", N=1)
    assert samples.shape == (6, 2)

    # Test Morris
    samples = analyzer.generate_samples(method="morris", N=10)
    assert samples.shape == (30, 2)

    # Test FAST
    samples = analyzer.generate_samples(method="fast", N=100)
    assert samples.shape == (200, 2)

    # Test LHS
    samples = analyzer.generate_samples(method="latin", N=100)
    assert samples.shape == (100, 2)


def test_run_tricys_simulations(analyzer):
    """Test the run_tricys_simulations method."""
    param_bounds = {"x1": [-3.14, 3.14], "x2": [0.0, 1.0]}
    analyzer.define_problem(param_bounds)
    analyzer.generate_samples(method="latin", N=10)

    csv_path = analyzer.run_tricys_simulations()

    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 10
    assert "x1" in df.columns
    assert "x2" in df.columns
    assert "param1" in df.columns


def test_generate_tricys_config(analyzer):
    """Test the generate_tricys_config method."""
    param_bounds = {"x1": [-3.14, 3.14], "x2": [0.0, 1.0]}
    analyzer.define_problem(param_bounds)
    analyzer.generate_samples(method="latin", N=10)
    csv_path = analyzer.run_tricys_simulations()

    config = analyzer.generate_tricys_config(csv_file_path=csv_path)

    assert config["simulation_parameters"]["file"] == os.path.abspath(csv_path)
    assert (
        config["sensitivity_analysis"]["analysis_case"]["independent_variable"]
        == "file"
    )


def test_load_tricys_results(analyzer):
    """Test the load_tricys_results method."""
    param_bounds = {"x1": [-3.14, 3.14], "x2": [0.0, 1.0]}
    analyzer.define_problem(param_bounds)

    dummy_results = {
        "x1": np.random.rand(10),
        "x2": np.random.rand(10),
        "Startup_Inventory": np.random.rand(10),
        "Self_Sufficiency_Time": np.random.rand(10),
    }
    df = pd.DataFrame(dummy_results)
    csv_path = Path(TEST_DIR) / "summary.csv"
    df.to_csv(csv_path, index=False)

    results = analyzer.load_tricys_results(
        str(csv_path), output_metrics=["Startup_Inventory", "Self_Sufficiency_Time"]
    )

    assert results.shape == (10, 2)


def test_analysis_methods(analyzer):
    """Test the analysis methods with dummy data."""
    param_bounds = {"x1": [-3.14, 3.14], "x2": [0.0, 1.0]}
    analyzer.define_problem(param_bounds)

    # Sobol
    analyzer.generate_samples(method="sobol", N=1)
    analyzer.simulation_results = np.random.rand(6, 1)
    sobol_results = analyzer.analyze_sobol(output_index=0)
    assert "S1" in sobol_results
    assert "ST" in sobol_results

    # Morris
    analyzer.generate_samples(method="morris", N=10)
    analyzer.simulation_results = np.random.rand(30, 1)
    morris_results = analyzer.analyze_morris(output_index=0)
    assert "mu_star" in morris_results
    assert "sigma" in morris_results

    # FAST
    analyzer.generate_samples(method="fast", N=100)
    analyzer.simulation_results = np.random.rand(200, 1)
    fast_results = analyzer.analyze_fast(output_index=0)
    assert "S1" in fast_results
    assert "ST" in fast_results

    # LHS
    analyzer.generate_samples(method="latin", N=100)
    analyzer.simulation_results = np.random.rand(100, 1)
    lhs_results = analyzer.analyze_lhs(output_index=0)
    assert "mean" in lhs_results
    assert "std" in lhs_results
