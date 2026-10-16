"""Comprehensive governance tests for TRICYS 2.0 Stage 2 (Issue #99 eradication).

Covers:
- AC 2.1 (Schema 2.0 seconds first-class citizen & pass-through)
- AC 2.2 (time_base: "hour" compatibility conversion & idempotency guard)
- AC 2.5 (Unit-aware postprocessing metric operator scaling)
- AC 2.6 (Zero hardcoded / 24 or / 8760 division across production codebase)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tricys.analysis.metric import (
    calculate_single_job_metrics,
)
from tricys.core.schema import (
    MetricConfig,
    ScenarioConfig,
    TricysV2Config,
)
from tricys.simulation.simulation import _sim_time_to_seconds
from tricys.utils.config_utils import basic_prepare_config


def _make_v2_config(scenario_dict: dict) -> TricysV2Config:
    return TricysV2Config.from_dict(
        {
            "version": "2.0",
            "model": {
                "fileName": "models/Cycle.mo",
                "modelName": "example_model.Cycle",
            },
            "scenario": scenario_dict,
            "task": {
                "type": "simulate",
            },
        }
    )


def test_schema2_second_first_citizen_passthrough():
    """AC 2.1: Schema 2.0 default time_base is 'second', stopTime is passed through without x3600."""
    scenario = ScenarioConfig(stopTime=86400.0, stepSize=100.0)
    assert scenario.time_base == "second"
    assert scenario.stopTime == 86400.0
    assert scenario.stepSize == 100.0

    v2_cfg = _make_v2_config({"stopTime": 86400.0, "stepSize": 100.0})
    assert v2_cfg.simulation.time_base == "second"
    assert v2_cfg.simulation.stop_time == 86400.0

    # Ensure prepared config keeps 86400.0 seconds without 3600x inflation
    prepared, _ = basic_prepare_config(v2_cfg.model_dump())
    sim_sec = _sim_time_to_seconds(prepared["simulation"])
    assert sim_sec["stop_time"] == 86400.0
    assert sim_sec["_time_unit"] == "second"


def test_schema2_hour_compatibility_conversion_and_idempotency():
    """AC 2.2: time_base: 'hour' converts hours to seconds (x3600) with strict idempotency."""
    scenario = ScenarioConfig(stopTime=24.0, stepSize=0.5, time_base="hour")
    assert scenario.time_base == "hour"

    v2_cfg = _make_v2_config({"stopTime": 24.0, "stepSize": 0.5, "time_base": "hour"})
    assert v2_cfg.simulation.time_base == "hour"
    prepared, _ = basic_prepare_config(v2_cfg.model_dump())

    # First call: scales by 3600
    sim_sec = _sim_time_to_seconds(prepared["simulation"])
    assert sim_sec["stop_time"] == 86400.0
    assert sim_sec["step_size"] == 1800.0
    assert sim_sec["_time_unit"] == "second"

    # Second call: must be idempotent, not 86400 * 3600
    sim_sec_again = _sim_time_to_seconds(sim_sec)
    assert sim_sec_again["stop_time"] == 86400.0
    assert sim_sec_again["step_size"] == 1800.0


def test_metric_unit_aware_scaling():
    """AC 2.5: Metric extractor scales raw second-native metrics to configured units."""
    # Construct a synthetic time series where:
    # Initial inventory = 1000.0
    # Turning point occurs at t = 3600.0 s
    # Inventory doubles (reaches 2000.0) at t = 86400.0 s (exactly 1 day)
    times = np.array([0.0, 1800.0, 3600.0, 43200.0, 86400.0, 100000.0])
    invs = np.array([1000.0, 800.0, 600.0, 1500.0, 2000.0, 2200.0])
    df = pd.DataFrame({"time": times, "sds.I[1]": invs})

    # 1. unit: 's' -> 86400.0 s
    m_sec = {
        "doubling": MetricConfig(source="sds.I[1]", evaluate="doubling_time", unit="s"),
        "turning": MetricConfig(source="sds.I[1]", evaluate="turning_point", unit="s"),
    }
    res_sec = calculate_single_job_metrics(df, m_sec)
    assert res_sec["doubling"] == pytest.approx(86400.0)
    assert res_sec["turning"] == pytest.approx(3600.0)

    # 2. unit: 'day' -> 1.0 day and 3600 / 86400 day
    m_day = {
        "doubling": MetricConfig(
            source="sds.I[1]", evaluate="doubling_time", unit="day"
        ),
        "turning": MetricConfig(
            source="sds.I[1]", evaluate="turning_point", unit="day"
        ),
    }
    res_day = calculate_single_job_metrics(df, m_day)
    assert res_day["doubling"] == pytest.approx(1.0)
    assert res_day["turning"] == pytest.approx(3600.0 / 86400.0)

    # 3. unit: 'year' -> 86400 / 31536000 year
    m_year = {
        "doubling": MetricConfig(
            source="sds.I[1]", evaluate="doubling_time", unit="year"
        ),
    }
    res_year = calculate_single_job_metrics(df, m_year)
    assert res_year["doubling"] == pytest.approx(86400.0 / 31536000.0)

    # 4. explicit conversion_factor: 86400.0 -> 1.0
    m_factor = {
        "doubling": MetricConfig(
            source="sds.I[1]",
            evaluate="doubling_time",
            conversion_factor=86400.0,
        ),
    }
    res_factor = calculate_single_job_metrics(df, m_factor)
    assert res_factor["doubling"] == pytest.approx(1.0)


def test_zero_hardcoded_division_in_codebase():
    """AC 2.6: Zero hardcoded / 24 or / 8760 in production code under tricys/."""
    tricys_root = Path(__file__).resolve().parents[2] / "tricys"
    assert tricys_root.is_dir()

    hardcode_patterns = [
        re.compile(r"/\s*24(?:\.0)?\b"),
        re.compile(r"/\s*8760(?:\.0)?\b"),
    ]

    violations = []
    for py_file in tricys_root.rglob("*.py"):
        # Exclude example directories or auto-generated tests
        if "example" in py_file.parts:
            continue
        content = py_file.read_text(encoding="utf-8")
        for line_no, line in enumerate(content.splitlines(), start=1):
            # Ignore comments and docstrings
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pat in hardcode_patterns:
                if pat.search(line):
                    violations.append(f"{py_file.name}:{line_no}: {line}")

    assert (
        not violations
    ), "Found hardcoded division violations in codebase:\n" + "\n".join(violations)
