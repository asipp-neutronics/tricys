
import pytest
import os
from tricys.utils.om_utils import (
    get_om_session,
    load_modelica_package,
    get_model_parameter_names,
    get_all_parameters_details,
    format_parameter_value,
)


def test_get_om_session():
    """Tests the get_om_session function to ensure it returns a valid OpenModelica session."""
    omc = get_om_session()
    
    loaded = load_modelica_package(omc, "/tricys/example/package.mo")
    if not loaded:
        pytest.fail(f"Failed to load Modelica package at /tyicys/example/package.mo")
        
    # Teardown: close the session after all tests in the module have run
    omc.sendExpression("quit()")

def test_format_parameter_value():
    """Tests the format_parameter_value function with various data types."""
    assert format_parameter_value("param", 123) == "param=123"
    assert format_parameter_value("param", 1.23) == "param=1.23"
    assert format_parameter_value("param", True) == "param=True"
    assert format_parameter_value("param", "hello") == 'param="hello"'
    assert format_parameter_value("param", [1, 2, 3]) == "param={1,2,3}"
    assert format_parameter_value("param", ["a", "b"]) == "param={a,b}"

def test_get_model_parameter_names():
    """Tests retrieving model parameter names"""
    omc_session = get_om_session()
    load_modelica_package(omc_session,"/tricys/example/package.mo")
    model_name = "example.Cycle"
    names = get_model_parameter_names(omc_session, model_name)
    
    # Check that the result is a list of strings and is not empty
    assert isinstance(names, list)
    assert len(names) > 0
    assert all(isinstance(name, str) for name in names)
    
    # Check for a known specific parameter
    assert "blanket.TBR" in names

def test_get_all_parameters_details():
    """Tests retrieving detailed model parameters using a real session."""
    omc_session = get_om_session()
    load_modelica_package(omc_session,"/tricys/example/package.mo")
    model_name = "example.Cycle"
    details = get_all_parameters_details(omc_session, model_name)
    # Check that the result is a list of dicts and is not empty
    assert isinstance(details, list)
    assert len(details) > 0
    assert all(isinstance(d, dict) for d in details)
    
    # Find the 'blanket.TBR' parameter and check its details
    tbr_param = next((p for p in details if p['name'] == 'blanket.TBR'), None)
    assert tbr_param is not None
    assert tbr_param['type'] == 'Real'
    assert 'defaultValue' in tbr_param
    assert 'comment' in tbr_param
    assert 'dimensions' in tbr_param

