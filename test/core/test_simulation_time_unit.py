"""Regression tests for hour→second conversion of simulation time fields (Issue #99).

TRICYS configs express ``simulation.stop_time`` / ``simulation.step_size`` in
HOURS (see docs/en/guides/tricys_basic/basic_configuration.md), while the CFEDR
family models advance Modelica's built-in ``time`` in SECONDS. The framework
converts at the boundary via ``_sim_time_to_seconds``.

These tests pin:

1. the ×3600 conversion,
2. idempotency (several entry points all read ``config["simulation"]``),
3. ``None`` / missing-key safety.
"""

from tricys.simulation.simulation import HOURS_TO_SECONDS, _sim_time_to_seconds


def test_hours_to_seconds_factor():
    assert HOURS_TO_SECONDS == 3600.0


def test_converts_stop_time_and_step_size():
    sim = {"stop_time": 1200.0, "step_size": 0.5, "model_name": "x"}
    result = _sim_time_to_seconds(sim)
    assert result is sim
    assert sim["stop_time"] == 1200.0 * 3600.0 == 4_320_000.0
    assert sim["step_size"] == 0.5 * 3600.0 == 1800.0
    assert sim["_time_unit"] == "second"
    # Non-time fields must be untouched.
    assert sim["model_name"] == "x"


def test_is_idempotent():
    sim = {"stop_time": 100.0, "step_size": 1.0}
    _sim_time_to_seconds(sim)
    once = dict(sim)
    _sim_time_to_seconds(sim)
    assert sim == once, "second call must not multiply by 3600 again"


def test_none_valued_field_is_left_alone():
    sim = {"stop_time": None, "step_size": 1.0}
    _sim_time_to_seconds(sim)
    assert sim["stop_time"] is None
    assert sim["step_size"] == 3600.0


def test_missing_field_is_safe():
    sim = {"step_size": 10.0}
    _sim_time_to_seconds(sim)
    assert sim == {"step_size": 36_000.0, "_time_unit": "second"}
