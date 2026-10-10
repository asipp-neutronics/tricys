from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tricys.analysis.metric import (
    build_single_job_summary_df,
    calculate_doubling_time,
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
from tricys.core.schema import MetricConfig


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

    # Doubling time (initial=100, target=200, first reaches 210 at t=90.0)
    assert calculate_doubling_time(series, time_series) == 90.0

    # Startup inventory (initial 100 - min 80 = 20)
    assert calculate_startup_inventory(series) == 20.0


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
    }

    res = calculate_single_job_metrics(sample_timeseries_df, metrics_def)

    assert res["max_inv"] == 220.0
    assert res["min_inv"] == 80.0
    assert res["final_inv"] == 220.0
    assert res["t_turn"] == 30.0
    assert res["t_double"] == 90.0


def test_calculate_single_job_metrics_v2_metric_config_objects(sample_timeseries_df):
    metrics_def = {
        "peak_storage": MetricConfig(source="sds.I[1]", evaluate="peak_value"),
        "doubling": MetricConfig(source="sds.I[1]", evaluate="doubling_time"),
    }

    res = calculate_single_job_metrics(sample_timeseries_df, metrics_def)
    assert res["peak_storage"] == 220.0
    assert res["doubling"] == 90.0


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
