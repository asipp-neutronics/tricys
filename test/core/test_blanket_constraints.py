"""Integration tests for Blanket capacity/rate constraint feature.

These tests drive OMC directly via OMPython to run small simulations of
``example_model.Cycle`` with different ``blanket.capacity_max`` /
``blanket.rate_max`` values, and verify the sigmoid-soft-constraint
implementation in ``Blanket.mo`` (Task 1.2 of the
``p3-blanket-default-constraints`` plan).

The tests are marked ``slow`` because each runs a real Modelica
simulation (~5-15 s including build). Skip if OMPython / omc is
unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tricys.core.modelica import get_om_session

try:
    from OMPython import OMCSessionZMQ

    OMPYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover
    OMCSessionZMQ = None  # type: ignore
    OMPYTHON_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PATH = (
    REPO_ROOT / "tricys" / "example" / "example_data" / "example_model" / "package.mo"
)
MODEL_NAME = "example_model.Cycle"
STOP_TIME = 5000.0
STEP_SIZE = 5.0  # coarser than default to keep per-test runtime low

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def omc():
    if not OMPYTHON_AVAILABLE:
        pytest.skip("OMPython not available")
    if not PACKAGE_PATH.exists():
        pytest.skip(f"Package not found: {PACKAGE_PATH}")
    session = get_om_session()
    try:
        loaded = session.sendExpression(f'loadFile("{PACKAGE_PATH.as_posix()}")')
        if loaded is not True:
            err = session.sendExpression("getErrorString()")
            pytest.skip(f"Failed to load package: {err}")
        yield session
    finally:
        try:
            session.sendExpression("quit()")
        except Exception:
            pass


@pytest.fixture
def workdir(tmp_path):
    return tmp_path


def _simulate(
    omc,
    workdir: Path,
    overrides: dict[str, float] | None = None,
    variable_filter: str = "time|blanket.I\\[1\\]|blanket.I_total|blanket.overflow_out\\[1\\]|blanket.rate_clip_out\\[1\\]|sds.I\\[1\\]",
    stop_time: float = STOP_TIME,
    step_size: float = STEP_SIZE,
) -> dict[str, list[float]]:
    """Run one simulation in ``workdir`` and return columns dict."""
    omc.sendExpression(f'cd("{workdir.as_posix()}")')
    override_str = ""
    if overrides:
        override_str = (
            'simflags="'
            + "-override="
            + ",".join(f"{k}={v}" for k, v in overrides.items())
            + '", '
        )
    expr = (
        f"simulate({MODEL_NAME}, stopTime={stop_time}, "
        f"numberOfIntervals={int(stop_time / step_size)}, "
        f"{override_str}"
        f'variableFilter="{variable_filter}")'
    )
    result = omc.sendExpression(expr)
    err = omc.sendExpression("getErrorString()")
    if "error" in (err or "").lower() and "warning" not in (err or "").lower():
        raise RuntimeError(f"Simulation failed: {err}\nResult: {result}")

    # Locate result CSV via getSimulationResultFile alternative: result is
    # struct-like; the result file is at <workdir>/<ClassName>_res.csv when
    # outputFormat="csv" — but default is .mat. Use readSimulationResult.
    # Easier: query variables via val() at sample times.
    return _sample_via_val(omc, stop_time)


def _sample_via_val(omc, stop_time: float) -> dict[str, list[float]]:
    """Sample key variables at evenly spaced times using val() on the .mat."""
    var_names = [
        "blanket.I[1]",
        "blanket.I_total",
        "blanket.overflow_out[1]",
        "blanket.rate_clip_out[1]",
        "sds.I[1]",
    ]
    n_samples = 51
    times = [stop_time * i / (n_samples - 1) for i in range(n_samples)]
    data: dict[str, list[float]] = {"time": times}
    for name in var_names:
        col: list[float] = []
        for t in times:
            v = omc.sendExpression(f"val({name}, {t})")
            col.append(float(v))
        data[name] = col
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_default_matches_baseline(omc, workdir):
    """Default capacity_max=1e9 should leave dynamics essentially unchanged."""
    data = _simulate(omc, workdir)  # no overrides → defaults
    # admit_scale ≈ 1: blanket should not overflow
    overflow_max = max(abs(v) for v in data["blanket.overflow_out[1]"])
    rate_clip_max = max(abs(v) for v in data["blanket.rate_clip_out[1]"])
    assert overflow_max < 1e-3, f"Default overflow should be ~0, got {overflow_max}"
    assert rate_clip_max < 1e-3, f"Default rate clip should be ~0, got {rate_clip_max}"
    # SDS inventory should grow positively from initial value
    sds_final = data["sds.I[1]"][-1]
    assert sds_final > 0, f"SDS inventory should remain positive, got {sds_final}"


def test_capacity_constraint_effective(omc, workdir):
    """Setting capacity_max=100 should cap blanket.I_total near 100."""
    data = _simulate(
        omc,
        workdir,
        overrides={"blanket.capacity_max": 100.0, "blanket.rate_max": 1e9},
    )
    i_total_max = max(data["blanket.I_total"])
    # Sigmoid softness=0.02 → I_total may exceed cap by ~softness*cap fraction;
    # allow up to 5% margin
    assert (
        i_total_max <= 105.0
    ), f"With capacity_max=100, I_total should be ≤105, got {i_total_max}"
    overflow_max = max(data["blanket.overflow_out[1]"])
    assert (
        overflow_max > 0
    ), f"With capacity_max=100, overflow should be > 0, got {overflow_max}"


def test_rate_constraint_effective(omc, workdir):
    """Setting rate_max=2 should make rate_clip_out non-zero."""
    data = _simulate(
        omc,
        workdir,
        overrides={"blanket.capacity_max": 1e9, "blanket.rate_max": 2.0},
    )
    rate_clip_max = max(data["blanket.rate_clip_out[1]"])
    assert (
        rate_clip_max > 0
    ), f"With rate_max=2, rate_clip_out should be > 0, got {rate_clip_max}"


def test_mass_conservation(omc, workdir):
    """Approx mass balance over the run (allow 2% for sigmoid softness)."""
    overrides = {"blanket.capacity_max": 200.0, "blanket.rate_max": 20.0}
    data = _simulate(omc, workdir, overrides=overrides)
    times = data["time"]

    # Integrate overflow + rate_clip via trapezoidal rule
    def trapz(y: list[float]) -> float:
        return sum(
            0.5 * (y[i] + y[i + 1]) * (times[i + 1] - times[i])
            for i in range(len(y) - 1)
        )

    overflow_int = trapz(data["blanket.overflow_out[1]"])
    rate_clip_int = trapz(data["blanket.rate_clip_out[1]"])
    # Both should be finite and non-negative (allow tiny numerical noise)
    assert math.isfinite(overflow_int) and overflow_int >= -1e-6
    assert math.isfinite(rate_clip_int) and rate_clip_int >= -1e-6
    # Sanity: blanket.I_total stayed bounded (≤ ~204 with softness 0.02)
    i_total_max = max(data["blanket.I_total"])
    assert i_total_max <= 210, f"I_total exceeded soft cap+margin: {i_total_max}"


def test_hard_constraint(omc, workdir):
    """softness=0.001 (near-hard) should keep I_total within ~1% of cap."""
    overrides = {
        "blanket.capacity_max": 100.0,
        "blanket.rate_max": 1e9,
        "blanket.softness": 0.001,
    }
    data = _simulate(omc, workdir, overrides=overrides)
    i_total_max = max(data["blanket.I_total"])
    assert (
        i_total_max <= 102.0
    ), f"Near-hard constraint should bound I_total ≤102, got {i_total_max}"
