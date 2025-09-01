import gc
import os
from pathlib import Path

import pytest

from tricys.utils.om_utils import (
    format_parameter_value,
    get_all_parameters_details,
    get_model_parameter_names,
    get_om_session,
    load_modelica_package,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
MODEL_PATH = os.path.join(project_root, "example", "gui", "example_model", "package.mo")


@pytest.mark.build_test
def test_get_om_session():
    """Tests the get_om_session function to ensure it returns a valid OpenModelica session."""
    omc = get_om_session()

    loaded = load_modelica_package(omc, Path(MODEL_PATH).as_posix())
    if not loaded:
        pytest.fail(f"Failed to load Modelica package at {MODEL_PATH}")

    # Teardown: close the session after all tests in the module have run
    omc.sendExpression("quit()")


@pytest.mark.build_test
def test_format_parameter_value():
    """Tests the format_parameter_value function with various data types."""
    assert format_parameter_value("param", 123) == "param=123"
    assert format_parameter_value("param", 1.23) == "param=1.23"
    assert format_parameter_value("param", True) == "param=True"
    assert format_parameter_value("param", "hello") == 'param="hello"'
    assert format_parameter_value("param", [1, 2, 3]) == "param={1,2,3}"
    assert format_parameter_value("param", ["a", "b"]) == "param={a,b}"


@pytest.mark.build_test
def test_get_model_parameter_names():
    """Tests retrieving model parameter names"""
    omc_session = get_om_session()
    load_modelica_package(omc_session, Path(MODEL_PATH).as_posix())
    model_name = "example_model.Cycle"
    names = get_model_parameter_names(omc_session, model_name)

    # Check that the result is a list of strings and is not empty
    assert isinstance(names, list)
    assert len(names) > 0
    assert all(isinstance(name, str) for name in names)

    # Check for a known specific parameter
    assert "blanket.TBR" in names


@pytest.mark.build_test
def test_get_all_parameters_details():
    """Tests retrieving detailed model parameters using a real session."""
    omc_session = get_om_session()
    load_modelica_package(omc_session, Path(MODEL_PATH).as_posix())
    model_name = "example_model.Cycle"
    details = get_all_parameters_details(omc_session, model_name)
    # Check that the result is a list of dicts and is not empty
    assert isinstance(details, list)
    assert len(details) > 0
    assert all(isinstance(d, dict) for d in details)

    # Find the 'blanket.TBR' parameter and check its details
    tbr_param = next((p for p in details if p["name"] == "blanket.TBR"), None)
    assert tbr_param is not None
    assert tbr_param["type"] == "Real"
    assert "defaultValue" in tbr_param
    assert "comment" in tbr_param
    assert "dimensions" in tbr_param


@pytest.mark.build_test
def test_integrate_interceptor_model(request):
    """
    Tests the full workflow of integrating an interceptor model using
    real files from the example_model directory.
    This test requires a running OpenModelica instance.
    """
    # --- 1. Setup: Define paths and configuration ---
    # Use the globally defined MODEL_PATH for the package
    package_path = Path(MODEL_PATH)
    model_name = "example_model.Cycle"
    submodel_name = "example_model.I_ISS"
    submodel_instance_name = "i_iss"

    interception_configs = [
        {
            "submodel_name": submodel_name,
            "instance_name": submodel_instance_name,
            "csv_uri": "data/i_iss_co_sim_results.csv",
            "output_placeholder": {
                "to_SDS": "{1,2,3,4,5,6}",  # time + 5 elements
                "to_WDS": "{1,7,8,9,10,11}",
            },
        }
    ]

    # --- 2. Define cleanup actions ---
    generated_files = []

    def cleanup():
        print("\nCleaning up generated test files...")
        gc.collect()  # Ensure all file handles are released
        for f in generated_files:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed: {f}")

    request.addfinalizer(cleanup)

    # --- 3. Run the function to be tested ---
    from tricys.utils.om_utils import integrate_interceptor_model

    result = integrate_interceptor_model(
        package_path=str(package_path),
        model_name=model_name,
        interception_configs=interception_configs,
    )
    generated_files.extend(result.get("interceptor_model_paths", []))
    generated_files.append(result.get("system_model_path"))

    # --- 4. Assertions ---
    assert "interceptor_model_paths" in result
    assert "system_model_path" in result
    assert len(result["interceptor_model_paths"]) == 1

    interceptor_file = Path(result["interceptor_model_paths"][0])
    system_file = Path(result["system_model_path"])

    assert interceptor_file.exists()
    assert system_file.exists()
    assert interceptor_file.name == "I_ISS_Interceptor.mo"
    assert system_file.name == "Cycle_Intercepted.mo"

    # Check interceptor model content
    interceptor_content = interceptor_file.read_text(encoding="utf-8")
    assert "model I_ISS_Interceptor" in interceptor_content
    assert (
        'parameter String fileName = "data/i_iss_co_sim_results.csv"'
        in interceptor_content
    )
    assert (
        "Modelica.Blocks.Interfaces.RealInput physical_to_SDS[5]" in interceptor_content
    )
    assert (
        "Modelica.Blocks.Interfaces.RealInput physical_to_WDS[5]" in interceptor_content
    )
    assert "parameter Integer columns_to_SDS[6] = {1,2,3,4,5,6}" in interceptor_content
    assert (
        "parameter Integer columns_to_WDS[6] = {1,7,8,9,10,11}" in interceptor_content
    )

    # Check modified system model content
    system_content = system_file.read_text(encoding="utf-8")
    assert "model Cycle_Intercepted" in system_content
    assert "example_model.I_ISS_Interceptor i_iss_interceptor;" in system_content

    # Check rewiring for to_SDS
    assert "connect(i_iss.to_SDS, i_iss_interceptor.physical_to_SDS);" in system_content
    assert "connect(i_iss_interceptor.final_to_SDS, sds.from_I_ISS)" in system_content

    # Check rewiring for to_WDS
    assert "connect(i_iss.to_WDS, i_iss_interceptor.physical_to_WDS);" in system_content
    assert "connect(i_iss_interceptor.final_to_WDS, wds.from_I_ISS)" in system_content
