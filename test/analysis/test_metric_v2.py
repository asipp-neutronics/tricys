from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tricys.analysis.metric import (
    apply_series_smoothing,
    build_single_job_summary_df,
    calculate_doubling_time,
    calculate_dynamic_deficit,
    calculate_required_startup_inventory,
    calculate_single_job_metrics,
    calculate_startup_inventory,
    get_final_value,
    get_integral_value,
    get_max_value,
    get_mean_value,
    get_min_value,
    get_peak_value,
    get_settling_time,
    time_of_turning_point,
)
from tricys.core.schema import MetricConfig, TargetMetricConfig


@pytest.fixture
def sample_timeseries_df():
    """Generates synthetic tritium inventory timeseries DataFrame."""
    # Inventory starts at 100, drops to min of 80 at t=30, then rises to 220 at t=100
    times = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    inventories = [
        100.0,
        92.0,
        85.0,
        80.0,
        88.0,
        105.0,
        130.0,
        160.0,
        190.0,
        210.0,
        220.0,
    ]
    return pd.DataFrame(
        {
            "time": times,
            "sds.I[1]": inventories,
            "constant_signal": [50.0] * len(times),
        }
    )


def test_v2_scalar_operators(sample_timeseries_df):
    series = sample_timeseries_df["sds.I[1]"]
    time_series = sample_timeseries_df["time"]

    # Peak / Max
    assert get_peak_value(series) == 220.0
    assert get_max_value(series) == 220.0

    # Min
    assert get_min_value(series) == 80.0

    # Mean
    expected_mean = float(np.mean(series))
    assert pytest.approx(get_mean_value(series), 1e-5) == expected_mean

    # Final value
    assert get_final_value(series) == 220.0

    # Integral (trapezoidal rule)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    expected_integral = float(trapz_fn(series.values, time_series.values))
    assert (
        pytest.approx(get_integral_value(series, time_series), 1e-5)
        == expected_integral
    )

    # Turning point (minimum at t=30.0)
    assert time_of_turning_point(series, time_series) == 30.0

    # Doubling time (initial=100, target=200; discrete grid crossing is 90.0, interpolated crossing is 85.0)
    assert calculate_doubling_time(series, time_series) == 90.0
    assert calculate_doubling_time(series, time_series, interpolate=True) == 85.0

    # Startup inventory (initial 100 - min 80 = 20)
    assert calculate_startup_inventory(series) == 20.0
    # Dynamic deficit (initial 100 - min 80 = 20)
    assert calculate_dynamic_deficit(series) == 20.0
    # Required start-up inventory with reserve Ir = 50 -> 20 + 50 = 70
    assert calculate_required_startup_inventory(series, reserve_inventory=50.0) == 70.0


def test_settling_time_operator():
    times = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # Signal oscillates then settles around final value 100
    signal = pd.Series([0.0, 120.0, 90.0, 101.0, 99.5, 100.0])

    # Within 2% tolerance band [98, 102], settles starting at t=3.0
    st = get_settling_time(signal, times, tolerance=0.02)
    assert st == 3.0

    # When always constant, settling time is initial time
    constant = pd.Series([100.0] * 6)
    assert get_settling_time(constant, times) == 0.0


def test_calculate_single_job_metrics_v2_dict(sample_timeseries_df):
    metrics_def = {
        "max_inv": {"source": "sds.I[1]", "evaluate": "peak_value"},
        "min_inv": {"source": "sds.I[1]", "evaluate": "min"},
        "final_inv": {"source": "sds.I[1]", "evaluate": "final_value"},
        "mean_inv": {"source": "sds.I[1]", "evaluate": "mean"},
        "int_inv": {"source": "sds.I[1]", "evaluate": "integral"},
        "t_turn": {"source": "sds.I[1]", "evaluate": "turning_point"},
        "t_double": {"source": "sds.I[1]", "evaluate": "doubling_time"},
        "deficit": {"source": "sds.I[1]", "evaluate": "dynamic_deficit"},
        "true_startup": {
            "source": "sds.I[1]",
            "evaluate": "required_startup_inventory",
            "reserve_inventory": 50.0,
        },
    }

    res = calculate_single_job_metrics(sample_timeseries_df, metrics_def)

    assert res["max_inv"] == 220.0
    assert res["min_inv"] == 80.0
    assert res["final_inv"] == 220.0
    assert res["t_turn"] == 30.0
    assert res["t_double"] == 90.0  # default interpolate=False
    assert res["deficit"] == 20.0
    assert res["true_startup"] == 70.0


def test_calculate_single_job_metrics_v2_metric_config_objects(sample_timeseries_df):
    metrics_def = {
        "peak_storage": MetricConfig(source="sds.I[1]", evaluate="peak_value"),
        "doubling": MetricConfig(source="sds.I[1]", evaluate="doubling_time"),
        "doubling_interp": MetricConfig(
            source="sds.I[1]", evaluate="doubling_time", interpolate=True
        ),
    }

    res = calculate_single_job_metrics(sample_timeseries_df, metrics_def)
    assert res["peak_storage"] == 220.0
    assert res["doubling"] == 90.0  # default interpolate=False
    assert res["doubling_interp"] == 85.0  # explicit interpolate=True


def test_build_single_job_summary_df_v2(sample_timeseries_df):
    metrics_def = {
        "max_inv": {"source": "sds.I[1]", "evaluate": "peak_value"},
        "final_inv": {"source": "sds.I[1]", "evaluate": "final_value"},
    }
    calculated = calculate_single_job_metrics(sample_timeseries_df, metrics_def)

    summary_df = build_single_job_summary_df(
        job_id=42,
        single_job_metrics=calculated,
        metrics_definition=metrics_def,
    )

    assert not summary_df.empty
    assert len(summary_df) == 1
    assert summary_df.iloc[0]["job_id"] == 42
    assert summary_df.iloc[0]["max_inv"] == 220.0
    assert summary_df.iloc[0]["final_inv"] == 220.0


def test_apply_series_smoothing_basic():
    # Constant or short series
    s_short = pd.Series([10.0, 20.0])
    assert apply_series_smoothing(s_short, smooth_window=5.0).equals(s_short)

    # smooth_window=None or <= 0
    s = pd.Series([10.0, 50.0, 10.0, 50.0, 10.0])
    t = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    assert apply_series_smoothing(s, time_series=t, smooth_window=None).equals(s)
    assert apply_series_smoothing(s, time_series=t, smooth_window=0.0).equals(s)

    # Smoothing reduces variance
    smoothed = apply_series_smoothing(
        s, time_series=t, smooth_window=2.0, method="mean"
    )
    assert smoothed.max() < s.max()
    assert smoothed.std() < s.std()

    # Median smoothing
    smoothed_med = apply_series_smoothing(
        s, time_series=t, smooth_window=2.0, method="median"
    )
    assert len(smoothed_med) == len(s)


def test_calculate_single_job_metrics_custom_smoothing():
    # Oscillating signal with trend:
    # Inventory drops, reaches nadir around t=20, then rises past 200 around t=70
    t = np.linspace(0, 100, 101)  # dt = 1.0
    base_inv = 100.0 - 20.0 * np.exp(-((t - 20.0) ** 2) / 100.0) + 1.5 * t
    # Add +/- 15 oscillation with period 10 and phase offset so end point has non-zero oscillation
    osc = 15.0 * np.cos(2 * np.pi * t / 10.0)
    raw_inv = base_inv + osc

    df = pd.DataFrame(
        {
            "time": t,
            "sds.I[1]": raw_inv,
            "cycle_time": [10.0] * len(t),
        }
    )

    # Case 1: Without smoothing (raw metrics)
    metrics_raw = {
        "min_raw": {"source": "sds.I[1]", "evaluate": "min"},
        "max_raw": {"source": "sds.I[1]", "evaluate": "max"},
        "final_raw": {"source": "sds.I[1]", "evaluate": "final_value"},
        "dt_raw": {"source": "sds.I[1]", "evaluate": "doubling_time"},
    }
    res_raw = calculate_single_job_metrics(df, metrics_raw)

    # Case 2: With user-defined smooth_window: 10.0 (no interpolate field specified, defaults to True)
    metrics_smooth = {
        "min_smooth": {"source": "sds.I[1]", "evaluate": "min", "smooth_window": 10.0},
        "max_smooth": {"source": "sds.I[1]", "evaluate": "max", "smooth_window": 10.0},
        "final_smooth": {
            "source": "sds.I[1]",
            "evaluate": "final_value",
            "smooth_window": 10.0,
        },
        "dt_smooth": {
            "source": "sds.I[1]",
            "evaluate": "doubling_time",
            "smooth_window": 10.0,
        },
    }
    res_smooth = calculate_single_job_metrics(df, metrics_smooth)

    # Raw min is dragged down by instantaneous negative swing
    assert res_smooth["min_smooth"] > res_raw["min_raw"]
    # Raw max is spiked by instantaneous positive swing
    assert res_smooth["max_smooth"] < res_raw["max_raw"]
    # At t=100, osc=+15.0, raw_final is base_inv[-1]+15. Smoothed final filters this swing:
    assert abs(res_smooth["final_smooth"] - base_inv[-1]) < abs(
        res_raw["final_raw"] - base_inv[-1]
    )

    # Case 3: Column reference for smooth_window
    metrics_col_ref = {
        "min_col": {
            "source": "sds.I[1]",
            "evaluate": "min",
            "smooth_window": "cycle_time",
        },
    }
    res_col = calculate_single_job_metrics(df, metrics_col_ref)
    assert pytest.approx(res_col["min_col"], 1e-4) == res_smooth["min_smooth"]


def test_schema_smooth_window_and_cycle_period_aliases():
    # MetricConfig alias normalization and default interpolate
    m1 = MetricConfig(source="sds.I[1]", cycle_period=96.0)
    assert m1.smooth_window == 96.0
    assert m1.cycle_period == 96.0
    assert m1.interpolate is False

    m2 = MetricConfig(source="sds.I[1]", smooth_window="burn_time")
    assert m2.smooth_window == "burn_time"
    assert m2.cycle_period == "burn_time"
    assert m2.interpolate is False

    # TargetMetricConfig alias normalization and default interpolate
    t1 = TargetMetricConfig(source="sds.I[1]", cycle_period=48.0)
    assert t1.smooth_window == 48.0
    assert t1.cycle_period == 48.0
    assert t1.interpolate is False

    t2 = TargetMetricConfig(source="sds.I[1]", smooth_window="pulse_period")
    assert t2.smooth_window == "pulse_period"
    assert t2.cycle_period == "pulse_period"
    assert t2.interpolate is False
