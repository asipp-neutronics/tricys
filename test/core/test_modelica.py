import os
import shutil
from pathlib import Path

import pytest

# Try to import OMPython and skip all tests if it's not available
try:
    from OMPython import OMCSessionZMQ

    OMCSessionZMQ
    OMPYTHON_AVAILABLE = True
except ImportError:
    OMPYTHON_AVAILABLE = False

if OMPYTHON_AVAILABLE:
    from tricys.core.modelica import (
        format_parameter_value,
        get_all_parameters_details,
        get_model_default_parameters,
        get_model_parameter_names,
        get_om_session,
        load_modelica_package,
    )

TEST_DIR = "temp_modelica_test"


@pytest.fixture(scope="module")
def omc_session():
    """Provides an OMCSessionZMQ instance."""
    if not OMPYTHON_AVAILABLE:
        pytest.skip("OMPython is not available")
    try:
        session = get_om_session()
        yield session
        session.sendExpression("quit()")
    except Exception:
        pytest.skip("OMCSessionZMQ could not be initialized")


@pytest.fixture(scope="module")
def model_path(omc_session):
    """Creates a dummy Modelica package and loads it."""
    package_content = """
package TestModel
  model MyModel
    parameter Real p1 = 1.0;
    parameter Real p2 = 2.0;
    inner Real x;
  end MyModel;
end TestModel;
    """
    package_path = Path(TEST_DIR) / "TestModel.mo"
    package_path.write_text(package_content, encoding="utf-8")

    if not load_modelica_package(omc_session, str(package_path)):
        pytest.skip("Failed to load Modelica package")

    return package_path


@pytest.fixture(autouse=True, scope="module")
def setup_and_teardown_test_dir():
    """Set up and tear down the test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


@pytest.mark.skipif(not OMPYTHON_AVAILABLE, reason="OMPython is not available")
class TestModelicaUtils:
    def test_get_model_parameter_names(self, omc_session, model_path):
        """Test get_model_parameter_names."""
        # This function is designed to get subcomponent parameters, so it won't find p1 and p2.
        # This is expected.
        names = get_model_parameter_names(omc_session, "TestModel.MyModel")
        assert names == []

    def test_get_all_parameters_details(self, omc_session, model_path):
        """Test get_all_parameters_details."""
        details = get_all_parameters_details(omc_session, "TestModel.MyModel")
        assert len(details) == 2
        param_names = {p["name"] for p in details}
        assert "p1" in param_names
        assert "p2" in param_names

    def test_get_model_default_parameters(self, omc_session, model_path):
        """Test get_model_default_parameters."""
        params = get_model_default_parameters(omc_session, "TestModel.MyModel")
        assert params["p1"] == 1.0
        assert params["p2"] == 2.0


def test_format_parameter_value():
    """Test format_parameter_value."""
    assert format_parameter_value("p", 1.0) == "p=1.0"
    assert format_parameter_value("p", "test") == 'p="test"'
    assert format_parameter_value("p", [1, 2, 3]) == "p={1,2,3}"
    assert format_parameter_value("p", True) == "p=true"
