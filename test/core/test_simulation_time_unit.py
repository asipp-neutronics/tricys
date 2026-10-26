"""Regression tests for time unit conversion of simulation time fields (Schema 2.0).

In TRICYS 2.0:
- TRICYS user configuration time fields are authored in engineering hours.
- `time_base: "second"` declares the Modelica model's time unit is seconds,
  triggering hour -> second conversion (x3600.0) for OMC.
- `time_base: "hour"` declares the Modelica model's time unit is hours
  (e.g. Jinchuan's model), passing hour values through directly (x1.0).
- Strict idempotency via `_time_unit = "second"` prevents multiple multiplications.
"""

import pytest

from tricys.simulation.simulation import (
    HOURS_TO_SECONDS,
    _sim_time_to_seconds,
)


def test_hours_to_seconds_factor():
    assert HOURS_TO_SECONDS == 3600.0


def test_time_base_second_scales_to_seconds():
    sim = {
        "stop_time": 24.0,
        "step_size": 0.5,
        "time_base": "second",
        "model_name": "x",
    }
    result = _sim_time_to_seconds(sim)
    assert result is sim
    assert sim["stop_time"] == 24.0 * 3600.0 == 86400.0
    assert sim["step_size"] == 0.5 * 3600.0 == 1800.0
    assert sim["_time_unit"] == "second"
    # Non-time fields must be untouched
    assert sim["model_name"] == "x"


def test_time_base_hour_passes_through_unchanged():
    sim = {"stop_time": 24.0, "step_size": 0.5, "time_base": "hour"}
    result = _sim_time_to_seconds(sim)
    assert result is sim
    assert sim["stop_time"] == 24.0
    assert sim["step_size"] == 0.5
    assert sim["_time_unit"] == "second"


def test_time_base_default_is_second_scales_hours():
    sim = {"stop_time": 24.0, "step_size": 0.5}
    _sim_time_to_seconds(sim)
    assert sim["stop_time"] == 86400.0
    assert sim["step_size"] == 1800.0
    assert sim["_time_unit"] == "second"


def test_is_idempotent_on_second_base():
    sim = {"stop_time": 24.0, "step_size": 1.0, "time_base": "second"}
    _sim_time_to_seconds(sim)
    once = dict(sim)
    assert once["stop_time"] == 86400.0
    _sim_time_to_seconds(sim)
    assert sim == once, "second call must not multiply by 3600 again"


def test_is_idempotent_on_hour_base():
    sim = {"stop_time": 24.0, "step_size": 1.0, "time_base": "hour"}
    _sim_time_to_seconds(sim)
    once = dict(sim)
    assert once["stop_time"] == 24.0
    _sim_time_to_seconds(sim)
    assert sim == once


def test_none_valued_field_is_left_alone():
    sim = {"stop_time": None, "step_size": 1.0, "time_base": "second"}
    _sim_time_to_seconds(sim)
    assert sim["stop_time"] is None
    assert sim["step_size"] == 3600.0


def test_missing_field_is_safe():
    sim = {"step_size": 10.0, "time_base": "second"}
    _sim_time_to_seconds(sim)
    assert sim["step_size"] == 36_000.0
    assert sim["_time_unit"] == "second"


def test_invalid_time_base_raises():
    with pytest.raises(ValueError, match="simulation.time_base"):
        _sim_time_to_seconds({"stop_time": 1.0, "time_base": "fortnight"})
