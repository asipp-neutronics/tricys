import os
import shutil
from pathlib import Path

import pytest

from tricys.core.interceptor import integrate_interceptor_model

TEST_DIR = "temp_interceptor_test"


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Set up and tear down the test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def create_single_file_package(package_path: Path):
    """Creates a dummy single-file Modelica package."""
    content = """
package TestPackage
  model SubModel
    Modelica.Blocks.Interfaces.RealOutput y;
  equation
    y = 1.0;
  end SubModel;

  model SystemModel
    SubModel sub;
    Modelica.Blocks.Interfaces.RealOutput z;
  equation
    connect(sub.y, z);
  end SystemModel;
end TestPackage;
    """
    package_path.write_text(content, encoding="utf-8")


def create_multi_file_package(package_dir: Path):
    """Creates a dummy multi-file Modelica package."""
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "package.mo").write_text(
        "within ; package TestPackage end TestPackage;", encoding="utf-8"
    )
    (package_dir / "SubModel.mo").write_text(
        """
within TestPackage;
model SubModel
  Modelica.Blocks.Interfaces.RealOutput y;
equation
  y = 1.0;
end SubModel;
    """,
        encoding="utf-8",
    )
    (package_dir / "SystemModel.mo").write_text(
        """
within TestPackage;
model SystemModel
  SubModel sub;
  Modelica.Blocks.Interfaces.RealOutput z;
equation
  connect(sub.y, z);
end SystemModel;
    """,
        encoding="utf-8",
    )


@pytest.mark.skip(reason="OMCSessionZMQ not available in test environment")
def test_integrate_interceptor_single_file():
    """Test the interceptor mode with a single-file package."""
    package_path = Path(TEST_DIR) / "TestPackage.mo"
    create_single_file_package(package_path)

    interception_configs = [
        {
            "submodel_name": "TestPackage.SubModel",
            "csv_uri": "dummy.csv",
            "instance_name": "sub",
            "output_placeholder": {"y": "{1, 2}"},
            "mode": "interceptor",
        }
    ]

    result = integrate_interceptor_model(
        package_path=str(package_path),
        model_name="TestPackage.SystemModel",
        interception_configs=interception_configs,
    )

    assert os.path.exists(result["system_model_path"])

    modified_code = Path(result["system_model_path"]).read_text(encoding="utf-8")
    assert "SystemModel_Intercepted" in modified_code
    assert "SubModel_Interceptor" in modified_code
    assert "connect(sub.y, sub_interceptor.physical_y)" in modified_code


@pytest.mark.skip(reason="OMCSessionZMQ not available in test environment")
def test_integrate_interceptor_multi_file():
    """Test the interceptor mode with a multi-file package."""
    package_dir = Path(TEST_DIR) / "TestPackage"
    create_multi_file_package(package_dir)
    package_path = package_dir / "package.mo"

    interception_configs = [
        {
            "submodel_name": "TestPackage.SubModel",
            "csv_uri": "dummy.csv",
            "instance_name": "sub",
            "output_placeholder": {"y": "{1, 2}"},
            "mode": "interceptor",
        }
    ]

    result = integrate_interceptor_model(
        package_path=str(package_path),
        model_name="TestPackage.SystemModel",
        interception_configs=interception_configs,
    )

    assert os.path.exists(result["system_model_path"])
    assert len(result["interceptor_model_paths"]) == 1
    assert os.path.exists(result["interceptor_model_paths"][0])

    modified_code = Path(result["system_model_path"]).read_text(encoding="utf-8")
    assert "SystemModel_Intercepted" in modified_code
    assert "SubModel_Interceptor sub_interceptor" in modified_code
    assert "connect(sub.y, sub_interceptor.physical_y)" in modified_code


@pytest.mark.skip(reason="OMCSessionZMQ not available in test environment")
def test_integrate_replacement_single_file():
    """Test the replacement mode with a single-file package."""
    package_path = Path(TEST_DIR) / "TestPackage.mo"
    create_single_file_package(package_path)

    interception_configs = [
        {
            "submodel_name": "TestPackage.SubModel",
            "csv_uri": "dummy.csv",
            "instance_name": "sub",
            "output_placeholder": {"y": "{1, 2}"},
            "mode": "replacement",
        }
    ]

    result = integrate_interceptor_model(
        package_path=str(package_path),
        model_name="TestPackage.SystemModel",
        interception_configs=interception_configs,
    )

    assert os.path.exists(result["system_model_path"])

    modified_code = Path(result["system_model_path"]).read_text(encoding="utf-8")
    assert "y = if columns_y[2] == 1 then 0.0 else table_y.y[1];" in modified_code


@pytest.mark.skip(reason="OMCSessionZMQ not available in test environment")
def test_integrate_replacement_multi_file():
    """Test the replacement mode with a multi-file package."""
    package_dir = Path(TEST_DIR) / "TestPackage"
    create_multi_file_package(package_dir)
    package_path = package_dir / "package.mo"

    interception_configs = [
        {
            "submodel_name": "TestPackage.SubModel",
            "csv_uri": "dummy.csv",
            "instance_name": "sub",
            "output_placeholder": {"y": "{1, 2}"},
            "mode": "replacement",
        }
    ]

    result = integrate_interceptor_model(
        package_path=str(package_path),
        model_name="TestPackage.SystemModel",
        interception_configs=interception_configs,
    )

    assert len(result["replaced_models"]) == 1
    modified_submodel_path = result["replaced_models"][0]["modified_path"]
    assert os.path.exists(modified_submodel_path)

    modified_code = Path(modified_submodel_path).read_text(encoding="utf-8")
    assert "y = if columns_y[2] == 1 then 0.0 else table_y.y[1];" in modified_code
