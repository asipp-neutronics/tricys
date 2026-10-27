"""Unit tests for Diagnostics Barrier and multi-level self-healing error classifier."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, ValidationError

from tricys.core.diagnostics import (
    DiagnosticsBarrier,
    DiagnosticsClassifier,
    ErrorLevel,
)


class DummySchema(BaseModel):
    required_num: float = Field(..., gt=0)


def test_level_1_config_validation_error():
    """Tests Level 1 configuration schema validation error interception."""
    try:
        DummySchema(required_num=-5.0)
    except ValidationError as err:
        record = DiagnosticsClassifier.classify(err, {"config_path": "invalid.yaml"})
        assert record.error_level == ErrorLevel.LEVEL_1_CONFIG
        assert record.error_code == "ERR_CONFIG_INVALID"
        assert "required_num" in record.root_cause
        assert record.suggested_fix is not None
        assert record.suggested_fix.action == "modify_yaml"


def test_level_2_modelica_compilation_error():
    """Tests Level 2 Modelica compiler failure interception."""
    err = RuntimeError(
        "OpenModelica compilation failed at line 42: undefined variable TBR"
    )
    record = DiagnosticsClassifier.classify(err, {"model": "Cycle"})
    assert record.error_level == ErrorLevel.LEVEL_2_MODELICA
    assert record.error_code == "ERR_MODELICA_COMPILE"
    assert "line 42" in record.root_cause
    assert record.suggested_fix is not None
    assert record.suggested_fix.action == "patch_modelica"


def test_level_3_solver_divergence_and_underflow():
    """Tests Level 3 DASSL step underflow and negative inventory divergence."""
    # Underflow case
    err1 = RuntimeError("DASSL solver failed: step size underflow at t = 10500.0s")
    record1 = DiagnosticsClassifier.classify(
        err1, {"solver": "dassl", "tolerance": 1e-6}
    )
    assert record1.error_level == ErrorLevel.LEVEL_3_SOLVER
    assert record1.error_code == "ERR_SOLVER_STEP_UNDERFLOW"
    assert record1.suggested_fix is not None
    assert record1.suggested_fix.action == "adjust_solver"
    assert record1.suggested_fix.modifications.get("scenario.solver") == "ida"
    assert record1.suggested_fix.modifications.get(
        "scenario.tolerance"
    ) == pytest.approx(1e-5)

    # Negative inventory case
    err2 = RuntimeError(
        "Physical violation: negative inventory detected in sds.I[1] = -0.5g"
    )
    record2 = DiagnosticsClassifier.classify(err2)
    assert record2.error_level == ErrorLevel.LEVEL_3_SOLVER
    assert record2.error_code == "ERR_NEGATIVE_INVENTORY"


def test_level_4_mass_balance_conservation_violation():
    """Tests Level 4 closed-loop mass balance conservation violation."""
    err = RuntimeError(
        "Mass balance conservation violated: residual Delta M = 12.5g exceeds margin 0.5g"
    )
    record = DiagnosticsClassifier.classify(err)
    assert record.error_level == ErrorLevel.LEVEL_4_PHYSICS
    assert record.error_code == "ERR_MASS_BALANCE_VIOLATION"
    assert record.suggested_fix is not None
    assert record.suggested_fix.action == "check_streams"


def test_diagnostics_barrier_file_emission():
    """Tests that DiagnosticsBarrier generates valid diagnostics.json on intercepted failure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_json = Path(tmp_dir) / "diagnostics.json"

        with pytest.raises(ValueError):
            with DiagnosticsBarrier(
                output_path=out_json, context_snapshot={"case": "test_barrier"}
            ):
                raise ValueError(
                    "Config key 'invalid_param' not recognized in Schema 2.0"
                )

        assert out_json.exists(), "diagnostics.json must be written"
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["status"] == "FAILED"
        assert data["error_level"] == "LEVEL_1_CONFIG"
        assert data["error_code"] == "ERR_CONFIG_INVALID"
        assert data["context_snapshot"]["case"] == "test_barrier"


def test_cli_diagnose_integration():
    """Tests CLI execution with --diagnose flag on an invalid config file."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f_cfg:
        # Invalid schema structure: stopTime must be positive float or missing required keys
        f_cfg.write(
            "model:\n  fileName: dummy.mo\n  modelName: Dummy\nscenario:\n  stopTime: -100\n"
        )
        tmp_cfg = f_cfg.name

    try:
        cmd = [
            sys.executable,
            "-m",
            "tricys.main",
            "run",
            "-c",
            tmp_cfg,
            "--diagnose",
        ]
        # In current working directory, DiagnosticsBarrier writes to ./diagnostics.json
        res = subprocess.run(cmd, capture_output=True, text=True)
        # Should fail with nonzero exit code
        assert res.returncode != 0
        assert "TRICYS DIAGNOSTIC BARRIER INTERCEPTED" in res.stderr

        default_diag = Path("diagnostics.json")
        if default_diag.exists():
            content = json.loads(default_diag.read_text(encoding="utf-8"))
            assert content["status"] == "FAILED"
            assert (
                "LEVEL_1" in content["error_level"] or "ERR_" in content["error_code"]
            )
            default_diag.unlink()
    finally:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)
