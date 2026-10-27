import os
import shutil
from pathlib import Path

import pytest

from tricys.core.submodel_replacer import (
    PortCompatibilityError,
    extract_ports_from_text,
    extract_system_connections,
    replace_submodel_in_package,
    validate_port_and_connection_compatibility,
)

TEMP_TEST_DIR = "temp_replacer_unit_test"


@pytest.fixture(autouse=True)
def setup_teardown_test_workspace():
    if os.path.exists(TEMP_TEST_DIR):
        shutil.rmtree(TEMP_TEST_DIR)
    os.makedirs(TEMP_TEST_DIR, exist_ok=True)
    yield
    if os.path.exists(TEMP_TEST_DIR):
        shutil.rmtree(TEMP_TEST_DIR)


DUMMY_SYSTEM_PACKAGE = """package TestSystem
  model DummySub
    Modelica.Blocks.Interfaces.RealInput in_flow[5] "Inlet";
    Modelica.Blocks.Interfaces.RealOutput out_top[5] "Top product";
    Modelica.Blocks.Interfaces.RealOutput out_bottom[5] "Bottom product";
  equation
    out_top = in_flow;
    out_bottom = in_flow;
  end DummySub;

  model Cycle
    DummySub my_sub;
    Modelica.Blocks.Interfaces.RealInput sys_feed[5];
    Modelica.Blocks.Interfaces.RealOutput sys_prod[5];
    Modelica.Blocks.Interfaces.RealOutput sys_waste[5];
  equation
    connect(sys_feed, my_sub.in_flow);
    connect(my_sub.out_top, sys_prod);
    connect(my_sub.out_bottom, sys_waste);
  end Cycle;
end TestSystem;
"""

COMPATIBLE_SUBSTITUTE_MODEL = """package TestAdapters
  model GoodAdapter
    Modelica.Blocks.Interfaces.RealInput in_flow[5] "Matching inlet";
    Modelica.Blocks.Interfaces.RealOutput out_top[5] "Matching top";
    Modelica.Blocks.Interfaces.RealOutput out_bottom[5] "Matching bottom";
    Modelica.Blocks.Interfaces.RealInput optional_nbi[5] "Unconnected input";
  equation
    out_top = in_flow;
    out_bottom = in_flow;
  end GoodAdapter;
end TestAdapters;
"""

MISSING_PORT_SUBSTITUTE_MODEL = """package TestAdapters
  model BadAdapterMissingPort
    Modelica.Blocks.Interfaces.RealInput in_flow[5];
    Modelica.Blocks.Interfaces.RealOutput out_top[5];
    // Missing out_bottom!
  equation
    out_top = in_flow;
  end BadAdapterMissingPort;
end TestAdapters;
"""

DIM_MISMATCH_SUBSTITUTE_MODEL = """package TestAdapters
  model BadAdapterDimMismatch
    Modelica.Blocks.Interfaces.RealInput in_flow[5];
    Modelica.Blocks.Interfaces.RealOutput out_top[6]; // Dimension 6 instead of 5!
    Modelica.Blocks.Interfaces.RealOutput out_bottom[5];
  equation
    out_bottom = in_flow;
  end BadAdapterDimMismatch;
end TestAdapters;
"""

DIRECTION_MISMATCH_SUBSTITUTE_MODEL = """package TestAdapters
  model BadAdapterDirectionMismatch
    Modelica.Blocks.Interfaces.RealInput in_flow[5];
    Modelica.Blocks.Interfaces.RealInput out_top[5]; // Input instead of Output!
    Modelica.Blocks.Interfaces.RealOutput out_bottom[5];
  equation
    out_bottom = in_flow;
  end BadAdapterDirectionMismatch;
end TestAdapters;
"""


def test_extract_ports_and_connections():
    """Verify static extraction of port metadata and instance connections."""
    ports = extract_ports_from_text(DUMMY_SYSTEM_PACKAGE, model_name="DummySub")
    assert "in_flow" in ports
    assert ports["in_flow"].direction == "input"
    assert ports["in_flow"].dim == 5

    assert "out_top" in ports
    assert ports["out_top"].direction == "output"
    assert ports["out_top"].dim == 5

    conns = extract_system_connections(
        DUMMY_SYSTEM_PACKAGE,
        system_model_name="Cycle",
        instance_name="my_sub",
    )
    assert len(conns) == 3
    port_names = {c.port_name for c in conns}
    assert port_names == {"in_flow", "out_top", "out_bottom"}


def test_compatibility_validation_passes_when_ports_match():
    """Verify that validation cleanly passes when ports and dimensions match."""
    pkg_file = os.path.join(TEMP_TEST_DIR, "system.mo")
    sub_file = os.path.join(TEMP_TEST_DIR, "adapter.mo")
    Path(pkg_file).write_text(DUMMY_SYSTEM_PACKAGE, encoding="utf-8")
    Path(sub_file).write_text(COMPATIBLE_SUBSTITUTE_MODEL, encoding="utf-8")

    report = validate_port_and_connection_compatibility(
        target_model_name="TestSystem.DummySub",
        substitute_model_name="TestAdapters.GoodAdapter",
        instance_name="my_sub",
        parent_system_name="TestSystem.Cycle",
        system_code_or_path=pkg_file,
        substitute_code_or_path=sub_file,
    )
    assert report.is_compatible is True
    assert len(report.missing_ports) == 0
    assert len(report.dim_mismatches) == 0
    assert len(report.direction_mismatches) == 0
    assert len(report.matched_ports) == 3
    # Optional unconnected input detected
    assert len(report.unconnected_substitute_inputs) == 1
    assert report.unconnected_substitute_inputs[0]["port_name"] == "optional_nbi"


def test_compatibility_validation_fails_on_missing_connection_port():
    """Verify PortCompatibilityError is raised when a connection port is missing."""
    pkg_file = os.path.join(TEMP_TEST_DIR, "system.mo")
    sub_file = os.path.join(TEMP_TEST_DIR, "bad_adapter.mo")
    Path(pkg_file).write_text(DUMMY_SYSTEM_PACKAGE, encoding="utf-8")
    Path(sub_file).write_text(MISSING_PORT_SUBSTITUTE_MODEL, encoding="utf-8")

    with pytest.raises(PortCompatibilityError) as exc_info:
        validate_port_and_connection_compatibility(
            target_model_name="TestSystem.DummySub",
            substitute_model_name="TestAdapters.BadAdapterMissingPort",
            instance_name="my_sub",
            parent_system_name="TestSystem.Cycle",
            system_code_or_path=pkg_file,
            substitute_code_or_path=sub_file,
        )

    err = exc_info.value
    assert err.report.is_compatible is False
    assert len(err.report.missing_ports) == 1
    assert err.report.missing_ports[0]["port_name"] == "out_bottom"
    assert "Missing Required Ports" in err.report.diagnostics
    assert "out_bottom" in err.report.diagnostics


def test_compatibility_validation_fails_on_dimension_mismatch():
    """Verify PortCompatibilityError is raised when array dimension differs."""
    pkg_file = os.path.join(TEMP_TEST_DIR, "system.mo")
    sub_file = os.path.join(TEMP_TEST_DIR, "dim_mismatch.mo")
    Path(pkg_file).write_text(DUMMY_SYSTEM_PACKAGE, encoding="utf-8")
    Path(sub_file).write_text(DIM_MISMATCH_SUBSTITUTE_MODEL, encoding="utf-8")

    with pytest.raises(PortCompatibilityError) as exc_info:
        validate_port_and_connection_compatibility(
            target_model_name="TestSystem.DummySub",
            substitute_model_name="TestAdapters.BadAdapterDimMismatch",
            instance_name="my_sub",
            parent_system_name="TestSystem.Cycle",
            system_code_or_path=pkg_file,
            substitute_code_or_path=sub_file,
        )

    err = exc_info.value
    assert len(err.report.dim_mismatches) == 1
    assert err.report.dim_mismatches[0]["port_name"] == "out_top"
    assert err.report.dim_mismatches[0]["expected_dim"] == 5
    assert err.report.dim_mismatches[0]["actual_dim"] == 6


def test_compatibility_validation_fails_on_direction_mismatch():
    """Verify PortCompatibilityError is raised when port direction is reversed."""
    pkg_file = os.path.join(TEMP_TEST_DIR, "system.mo")
    sub_file = os.path.join(TEMP_TEST_DIR, "dir_mismatch.mo")
    Path(pkg_file).write_text(DUMMY_SYSTEM_PACKAGE, encoding="utf-8")
    Path(sub_file).write_text(DIRECTION_MISMATCH_SUBSTITUTE_MODEL, encoding="utf-8")

    with pytest.raises(PortCompatibilityError) as exc_info:
        validate_port_and_connection_compatibility(
            target_model_name="TestSystem.DummySub",
            substitute_model_name="TestAdapters.BadAdapterDirectionMismatch",
            instance_name="my_sub",
            parent_system_name="TestSystem.Cycle",
            system_code_or_path=pkg_file,
            substitute_code_or_path=sub_file,
        )

    err = exc_info.value
    assert len(err.report.direction_mismatches) == 1
    assert err.report.direction_mismatches[0]["port_name"] == "out_top"
    assert err.report.direction_mismatches[0]["expected"] == "output"
    assert err.report.direction_mismatches[0]["actual"] == "input"


def test_replace_submodel_in_package():
    """Verify submodel replacement in package code using extends adapter syntax."""
    pkg_file = os.path.join(TEMP_TEST_DIR, "system.mo")
    out_file = os.path.join(TEMP_TEST_DIR, "system_modified.mo")
    Path(pkg_file).write_text(DUMMY_SYSTEM_PACKAGE, encoding="utf-8")

    replace_submodel_in_package(
        package_path=pkg_file,
        target_model_name="TestSystem.DummySub",
        substitute_model_name="TestAdapters.GoodAdapter",
        unconnected_inputs=[{"port_name": "optional_nbi", "dim": 5}],
        output_package_path=out_file,
    )

    modified_content = Path(out_file).read_text(encoding="utf-8")
    assert "extends TestAdapters.GoodAdapter;" in modified_content
    assert "optional_nbi = {0, 0, 0, 0, 0};" in modified_content
    assert "model DummySub" in modified_content
