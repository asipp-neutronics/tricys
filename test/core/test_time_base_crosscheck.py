"""Cross-validate the example model's hour-base and second-base time axes.

The example model exposes a ``time_scale`` parameter (1 = hour-based axis,
3600 = second-based axis) on top of Modelica's seconds-based ``time``. Both
settings must describe the same physical trajectory, so

    x_second(t * 3600)  ==  x_hour(t)

This test drives the same model source both ways via OMC and checks they agree.
It is skipped when OpenModelica/OMPython is unavailable.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("OMPython")

# OMC can be located either via OPENMODELICAHOME or simply on PATH.
_OMC_AVAILABLE = bool(os.environ.get("OPENMODELICAHOME")) or shutil.which("omc") is not None
pytestmark = pytest.mark.skipif(not _OMC_AVAILABLE, reason="OpenModelica (omc) not available")

_MODEL = (
    Path(__file__).resolve().parents[2]
    / "tricys"
    / "example"
    / "example_data"
    / "example_model_single"
    / "example_model.mo"
)
_SECONDS_PER_HOUR = 3600.0
_STOP_HOURS = 24.0


def _run(time_scale: float, stop: float, result_file: Path) -> dict:
    from OMPython import ModelicaSystem

    mod = ModelicaSystem(str(_MODEL), "example_model.Cycle")
    mod.set_variable_filter(r"time|sds\.I\[1\]", escape=False)
    mod.setParameters([f"time_scale={time_scale}"])
    mod.setSimulationOptions(
        [f"stopTime={stop}", f"stepSize={stop / 8}", "outputFormat=csv"]
    )
    mod.simulate(resultfile=str(result_file))
    import pandas as pd

    df = pd.read_csv(result_file)
    return {"time": df["time"].to_numpy(), "x": df["sds.I[1]"].to_numpy()}


def test_hour_and_second_bases_agree(tmp_path):
    if not _MODEL.exists():
        pytest.skip(f"example model not found: {_MODEL}")

    hour = _run(1.0, _STOP_HOURS, tmp_path / "hour.csv")
    second = _run(
        _SECONDS_PER_HOUR, _STOP_HOURS * _SECONDS_PER_HOUR, tmp_path / "second.csv"
    )

    # both runs must span the same physical duration
    assert hour["time"].max() == pytest.approx(_STOP_HOURS)
    assert second["time"].max() == pytest.approx(_STOP_HOURS * _SECONDS_PER_HOUR)

    # interpolate the second-base curve onto the hour-base grid (t_s = 3600 * t_h)
    x_second = np.interp(hour["time"], second["time"] / _SECONDS_PER_HOUR, second["x"])
    np.testing.assert_allclose(x_second, hour["x"], rtol=1e-6, atol=1e-6)
