"""Regression tests for the FOC effective time base (Issue #99).

The CFEDR family models are **second-native**: Modelica's built-in ``time`` is in
seconds. These tests pin the contract that:

1. ``parse_foc_content_with_unit`` / ``parse_foc_file_with_unit`` report the
   *effective* time unit of the parsed durations (after any explicit
   ``TIME_CONVERSION``), not merely the declared ``TIME_UNIT``.
2. ``_warn_on_time_base_mismatch`` emits exactly one warning when the effective
   base differs from the model time base, and stays silent when they agree.
3. The public 2-tuple API (``parse_foc_content`` / ``parse_foc_file``) is
   unchanged.
4. The shipped example FOC resolves to seconds and stays consistent with its
   example config's ``stop_time``.
"""

import logging
from pathlib import Path

import pytest

from tricys.core import foc
from tricys.core.foc import (
    parse_foc_content,
    parse_foc_content_with_unit,
    parse_foc_file,
    parse_foc_file_with_unit,
)

# Selected by the time-base CI gate: the V2 pipeline runs
# `pytest -v -m build_test test` on ubuntu-latest (.github/workflows/build_test.yml).
# This module is pure Python (it never calls OpenModelica), so the whole module is
# safe to select there.
pytestmark = pytest.mark.build_test

# --------------------------------------------------------------------------- #
# Effective time-unit resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("content", "expected_unit"),
    [
        # No header → default seconds, no conversion.
        ("PULSE 1500 10 5 1\n", "second"),
        # TIME_UNIT alone declares but does NOT convert.
        ("TIME_UNIT hour\nPULSE 1500 0.5 0.25 1\n", "hour"),
        # Named conversions report the *target* unit.
        ("TIME_CONVERSION hour_to_second\nPULSE 1500 0.5 0.25 1\n", "second"),
        ("TIME_CONVERSION second_to_hour\nPULSE 1500 7200 3600 1\n", "hour"),
        ("TIME_CONVERSION day_to_second\nPULSE 1500 1 1 1\n", "second"),
        # Bare numeric factor → unknown scale.
        ("TIME_CONVERSION 3600\nPULSE 1500 0.5 0.25 1\n", "scaled"),
        # NONE / OFF / IDENTITY → no re-scaling, keep declared unit.
        ("TIME_CONVERSION NONE\nPULSE 1500 10 5 1\n", "second"),
        (
            "TIME_UNIT hour\nTIME_CONVERSION NONE\nPULSE 1500 10 5 1\n",
            "hour",
        ),
        # Declared unit + matching named conversion.
        (
            "TIME_UNIT hour\nTIME_CONVERSION hour_to_second\nPULSE 1500 0.5 0.25 1\n",
            "second",
        ),
    ],
)
def test_effective_time_unit_resolution(content, expected_unit):
    _, _, unit = parse_foc_content_with_unit(content)
    assert unit == expected_unit


def test_parse_foc_content_public_tuple_is_unchanged():
    """The public 2-tuple API must not change arity."""
    result = parse_foc_content("PULSE 1500 10 5 1\n")
    assert len(result) == 2
    amplitudes, durations = result
    assert amplitudes == [1500.0, 0.0]
    assert durations == [10.0, 5.0]


def test_parse_foc_file_public_tuple_is_unchanged(tmp_path):
    foc_path = tmp_path / "simple.foc"
    foc_path.write_text("PULSE 1500 10 5 1\n", encoding="utf-8")
    result = parse_foc_file(str(foc_path))
    assert len(result) == 2
    amplitudes, durations = result
    assert amplitudes == [1500.0, 0.0]
    assert durations == [10.0, 5.0]


def test_parse_foc_file_with_unit_reports_hour_base(tmp_path):
    foc_path = tmp_path / "hour.foc"
    foc_path.write_text("TIME_UNIT hour\nPULSE 1500 0.5 0.25 1\n", encoding="utf-8")
    _, durations, unit = parse_foc_file_with_unit(str(foc_path))
    assert unit == "hour"
    assert durations == [0.5, 0.25]


# --------------------------------------------------------------------------- #
# Warning behavior
# --------------------------------------------------------------------------- #


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):  # noqa: D102 - logging.Handler API
        self.records.append(record)


def _capture_warnings(unit, model_time_base="second", foc_path="/tmp/unit.foc"):
    handler = _CaptureHandler()
    foc.logger.addHandler(handler)
    previous_level = foc.logger.level
    foc.logger.setLevel(logging.WARNING)
    try:
        foc._warn_on_time_base_mismatch(unit, model_time_base, foc_path)
    finally:
        foc.logger.removeHandler(handler)
        foc.logger.setLevel(previous_level)
    return handler.records


@pytest.mark.parametrize(
    ("unit", "model_time_base"),
    [("second", "second"), ("hour", "hour")],
)
def test_no_warning_when_base_matches_model(unit, model_time_base):
    assert _capture_warnings(unit, model_time_base) == []


@pytest.mark.parametrize(
    ("unit", "model_time_base"),
    [
        ("hour", "second"),
        ("day", "second"),
        ("week", "second"),
        ("year", "second"),
        ("scaled", "second"),
        ("second", "hour"),
    ],
)
def test_warning_on_time_base_mismatch(unit, model_time_base):
    records = _capture_warnings(unit, model_time_base)
    assert len(records) == 1
    record = records[0]
    assert record.__dict__.get("effective_time_unit") == unit
    assert record.__dict__.get("model_time_base") == model_time_base
    assert model_time_base in record.getMessage()


def test_warning_message_suggests_named_conversion_for_known_unit():
    record = _capture_warnings("hour", "second")[0]
    assert "hour_to_second" in record.getMessage()


# --------------------------------------------------------------------------- #
# Shipped example consistency
# --------------------------------------------------------------------------- #


def test_shipped_example_foc_resolves_to_seconds():
    """The shipped example FOC must have a seconds effective base."""
    example = (
        Path(__file__).resolve().parents[2]
        / "tricys"
        / "example"
        / "example_data"
        / "example_foc"
        / "example_scenario_mix.foc"
    )
    assert example.exists(), f"missing shipped example: {example}"
    _, _, unit = parse_foc_file_with_unit(str(example))
    assert unit == "second", (
        "example_scenario_mix.foc must be converted to seconds "
        "(add TIME_CONVERSION hour_to_second)"
    )


def test_shipped_example_matches_config_stop_time():
    """schedule_duration (seconds) must not exceed stop_time converted to seconds."""
    example = (
        Path(__file__).resolve().parents[2]
        / "tricys"
        / "example"
        / "example_data"
        / "example_foc"
        / "example_scenario_mix.foc"
    )
    _, durations, unit = parse_foc_file_with_unit(str(example))
    assert unit == "second"
    schedule_seconds = sum(durations)
    # Config documents stop_time = 1200.0 (hours) → 4.32e6 s.
    stop_time_seconds = 1200.0 * 3600.0
    assert schedule_seconds <= stop_time_seconds
