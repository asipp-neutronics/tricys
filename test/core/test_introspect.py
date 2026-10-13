"""Unit tests for Modelica model deep introspection engine (introspect.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tricys.core.introspect import (
    ModelicaASTParser,
    ModelIntrospection,
    introspect_model,
)

SAMPLE_MODELICA_CODE = """
package TestPlant
  model Reactor
    outer parameter Real time_scale "Time factor";
    parameter Real burn_fraction = 0.05 "Burn fraction";
    Real I[2](start = {10.0, 0.0}) "Tritium inventory";
    Real burn_rate "Burn rate";
  equation
    der(I[1]) = -burn_rate / time_scale;
    der(I[2]) = 0.0;
    burn_rate = burn_fraction * I[1];
  end Reactor;

  model Plant
    inner parameter Real time_scale = 3600 "Global time scale";
    parameter Real TBR = 1.15 "Tritium Breeding Ratio";
    Reactor reactor;
  annotation(
    experiment(startTime = 0, stopTime = 86400, stepSize = 100, tolerance = 1e-6, solver = "dassl")
  );
  end Plant;
end TestPlant;
"""


def test_ast_parser_basic_extraction():
    """Tests basic static AST extraction on a small model snippet."""
    parser = ModelicaASTParser(SAMPLE_MODELICA_CODE, package_path="dummy.mo")
    result: ModelIntrospection = parser.parse_model("TestPlant.Plant")

    assert result.model_name == "TestPlant.Plant"
    assert "time_scale" in result.parameters
    assert result.parameters["time_scale"].default == 3600
    assert result.parameters["time_scale"].is_inner is True

    assert "TBR" in result.parameters
    assert result.parameters["TBR"].default == 1.15

    # Check child subcomponent parameters
    assert "reactor.burn_fraction" in result.parameters
    assert result.parameters["reactor.burn_fraction"].default == 0.05

    # Check subcomponents
    assert "reactor" in result.subcomponents
    assert result.subcomponents["reactor"].type == "Reactor"

    # Check inventories
    inv_names = [inv.name for inv in result.inventories]
    assert "reactor.I" in inv_names
    reactor_inv = next(inv for inv in result.inventories if inv.name == "reactor.I")
    assert reactor_inv.is_state is True
    assert reactor_inv.start == [10.0, 0.0]

    # Check flow rates
    flow_names = [f.name for f in result.flow_rates]
    assert "reactor.burn_rate" in flow_names

    # Check experiment annotation
    assert result.experiment.startTime == 0.0
    assert result.experiment.stopTime == 86400.0
    assert result.experiment.stepSize == 100.0
    assert result.experiment.solver == "dassl"


def test_introspect_model_on_real_example_model():
    """Tests introspection on the real example_model.mo Cycle model."""
    mo_path = Path("tricys/example/example_data/example_model_single/example_model.mo")
    assert mo_path.exists(), "example_model.mo must exist"

    intro = introspect_model(mo_path, "example_model.Cycle")
    assert len(intro.parameters) > 50
    assert len(intro.inventories) >= 15
    assert len(intro.flow_rates) >= 40
    assert len(intro.subcomponents) >= 15

    # Verify key fusion fuel cycle components exist
    assert "sds" in intro.subcomponents
    assert "blanket" in intro.subcomponents
    assert "plasma" in intro.subcomponents
    assert "tes" in intro.subcomponents

    # Verify SDS inventories and TBR parameter
    assert any("blanket.TBR" in p for p in intro.parameters)
    assert any("sds.I" in i.name for i in intro.inventories)

    # Verify JSON serialization
    json_data = intro.to_dict()
    assert json_data["model_name"] == "example_model.Cycle"
    assert isinstance(json_data["parameters"], dict)
    assert isinstance(json_data["inventories"], list)


def test_cli_model_inspect():
    """Tests python -m tricys.main model inspect CLI command."""
    mo_path = "tricys/example/example_data/example_model_single/example_model.mo"
    cmd = [
        sys.executable,
        "-m",
        "tricys.main",
        "model",
        "inspect",
        mo_path,
        "example_model.Cycle",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI inspect failed: {res.stderr}"

    data = json.loads(res.stdout)
    assert data["model_name"] == "example_model.Cycle"
    assert len(data["parameters"]) > 50
    assert len(data["inventories"]) > 10
