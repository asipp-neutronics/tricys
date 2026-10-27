"""Unit tests for Modelica model health auditor and remediation engine (model_auditor.py)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tricys.core.model_auditor import (
    ModelAuditor,
    ModelAuditReport,
    review_model,
)

DEFECTIVE_MODEL_CONTENT = """package DefectivePlant
  model SubComp
    parameter Real decay_loss[2] = {1e-5, 0.0};
    Real I[2];
  equation
    der(I[1]) = -decay_loss[1] * I[1];
    der(I[2]) = 0.0;
  end SubComp;

  model Plant
    SubComp sub;
  end Plant;
end DefectivePlant;
"""


def test_review_model_on_healthy_example_model():
    """Tests auditing on the standard compliant example_model.mo."""
    mo_path = Path("tricys/example/example_data/example_model_single/example_model.mo")
    assert mo_path.exists(), "example_model.mo must exist"

    report: ModelAuditReport = review_model(mo_path, "example_model.Cycle")
    assert report.status == "PASS"
    assert report.summary["critical"] == 0
    assert len(report.findings) == 0


def test_auditor_detects_time_scale_and_conservation_defects():
    """Tests that ModelAuditor precisely identifies missing time_scale, streams, and start values."""
    with tempfile.NamedTemporaryFile("w", suffix=".mo", delete=False) as f:
        f.write(DEFECTIVE_MODEL_CONTENT)
        tmp_path = f.name

    try:
        auditor = ModelAuditor(tmp_path, "DefectivePlant.Plant")
        report = auditor.audit()

        assert report.status == "FAIL"
        assert report.summary["critical"] >= 1

        finding_codes = {f.code for f in report.findings}
        # 1. Top-level time_scale missing
        assert "TIME_SCALE_MISSING" in finding_codes
        # 2. Subcomponent outer time_scale missing
        assert "TIME_SCALE_OUTER_MISSING" in finding_codes
        # 3. Missing decay_rate stream
        assert "MISSING_DECAY_RATE_STREAM" in finding_codes
        # 4. Continuous state start value missing
        assert "MISSING_STATE_START_VALUE" in finding_codes

        # Test patch generation
        patch_diff = auditor.generate_remediation_patch()
        assert "inner parameter Real time_scale = 3600" in patch_diff
        assert "outer parameter Real time_scale" in patch_diff
        assert patch_diff.startswith("--- a/")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_apply_remediation_patch_and_re_audit():
    """Tests that applying generated remediation patch resolves critical defects."""
    with tempfile.NamedTemporaryFile("w", suffix=".mo", delete=False) as f:
        f.write(DEFECTIVE_MODEL_CONTENT)
        tmp_path = f.name

    try:
        auditor = ModelAuditor(tmp_path, "DefectivePlant.Plant")
        patch = auditor.generate_remediation_patch()

        # Apply remediation
        success = auditor.apply_remediation(patch, backup=False)
        assert success is True

        # Re-audit remediated model
        auditor2 = ModelAuditor(tmp_path, "DefectivePlant.Plant")
        report2 = auditor2.audit()

        finding_codes2 = {f.code for f in report2.findings}
        # Critical time_scale issues should be fully resolved
        assert "TIME_SCALE_MISSING" not in finding_codes2
        assert "TIME_SCALE_OUTER_MISSING" not in finding_codes2
        assert report2.summary["critical"] == 0

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_cli_model_review_json_and_patch():
    """Tests CLI commands python -m tricys.main model review with --json and --patch-output."""
    with (
        tempfile.NamedTemporaryFile("w", suffix=".mo", delete=False) as f_mo,
        tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as f_diff,
    ):
        f_mo.write(DEFECTIVE_MODEL_CONTENT)
        tmp_mo = f_mo.name
        tmp_diff = f_diff.name

    try:
        cmd = [
            sys.executable,
            "-m",
            "tricys.main",
            "model",
            "review",
            tmp_mo,
            "DefectivePlant.Plant",
            "--json",
            "--patch-output",
            tmp_diff,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI review failed: {res.stderr}"

        # Check stdout JSON
        data = json.loads(res.stdout)
        assert data["status"] == "FAIL"
        assert data["summary"]["critical"] >= 1

        # Check saved patch file
        diff_content = Path(tmp_diff).read_text(encoding="utf-8")
        assert "time_scale" in diff_content
        assert diff_content.startswith("--- a/")

    finally:
        if os.path.exists(tmp_mo):
            os.remove(tmp_mo)
        if os.path.exists(tmp_diff):
            os.remove(tmp_diff)


def test_auditor_normalizes_derivatives_without_any_time_scale():
    """Tests that ModelAuditor catches and normalizes models with raw unscaled der(...) equations."""
    raw_model = """package RawPlant
  model Comp
    Real x;
    Real y;
  equation
    der(x) = y * 2.5;
    der(y) = 0.0;
  end Comp;

  model Plant
    inner parameter Real time_scale = 3600;
    Comp c;
  end Plant;
end RawPlant;
"""
    with tempfile.NamedTemporaryFile("w", suffix=".mo", delete=False) as f:
        f.write(raw_model)
        tmp_mo = f.name

    try:
        auditor = ModelAuditor(tmp_mo, "RawPlant.Plant")
        report = auditor.audit()

        # Should detect that Comp lacks outer time_scale and der(x) is not normalized
        codes = {f.code for f in report.findings}
        assert "TIME_SCALE_OUTER_MISSING" in codes
        assert "TIME_SCALE_UNUSED_IN_DERIVATIVE" in codes

        # Generate patch and apply
        patch = auditor.generate_remediation_patch()
        assert "/ time_scale" in patch
        assert "outer parameter Real time_scale" in patch

        success = auditor.apply_remediation(patch, backup=False)
        assert success is True

        # Re-audit
        auditor2 = ModelAuditor(tmp_mo, "RawPlant.Plant")
        report2 = auditor2.audit()
        codes2 = {f.code for f in report2.findings}
        assert "TIME_SCALE_UNUSED_IN_DERIVATIVE" not in codes2
        assert "TIME_SCALE_OUTER_MISSING" not in codes2

        # Check patched file content directly
        content = Path(tmp_mo).read_text(encoding="utf-8")
        assert "der(x) = (y * 2.5) / time_scale;" in content
        assert "der(y) = 0.0;" in content
    finally:
        if os.path.exists(tmp_mo):
            os.remove(tmp_mo)
