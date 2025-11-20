import numpy as np
import pandas as pd
import pytest

from tricys.analysis.metric import (
    calculate_doubling_time,
    calculate_startup_inventory,
    extract_metrics,
    get_final_value,
    time_of_turning_point,
)


@pytest.fixture
def sample_series_data():
    """Provides a sample pandas Series for testing metric functions."""
    data = [100, 90, 85, 82, 85, 95, 110, 150, 180, 210]
    time = np.arange(0, len(data)) * 10  # Time in hours
    series = pd.Series(data, name="inventory")
    time_series = pd.Series(time, name="time")
    return series, time_series


@pytest.mark.build_test
def test_get_final_value(sample_series_data):
    series, _ = sample_series_data
    assert get_final_value(series) == 210


@pytest.mark.build_test
def test_calculate_startup_inventory(sample_series_data):
    series, _ = sample_series_data
    # initial=100, min=82 -> 100 - 82 = 18
    assert calculate_startup_inventory(series) == 18


@pytest.mark.build_test
def test_time_of_turning_point(sample_series_data):
    series, time_series = sample_series_data
    # Minimum value is 82 at index 3. Time at index 3 is 30.
    assert time_of_turning_point(series, time_series) == 30


@pytest.mark.build_test
def test_time_of_turning_point_monotonic_decrease():
    data = [100, 90, 80, 70, 60]
    time = np.arange(0, len(data))
    series = pd.Series(data)
    time_series = pd.Series(time)
    assert np.isnan(time_of_turning_point(series, time_series))


@pytest.mark.build_test
def test_calculate_doubling_time(sample_series_data):
    series, time_series = sample_series_data
    # Initial is 100, doubled is 200.
    # The value 210 (>= 200) occurs at index 9. Time at index 9 is 90.
    assert calculate_doubling_time(series, time_series) == 90


@pytest.mark.build_test
def test_calculate_doubling_time_never_doubles(sample_series_data):
    series, time_series = sample_series_data
    series_no_double = pd.Series([100, 90, 85, 82, 85, 95, 110, 150, 180, 199])
    assert np.isnan(calculate_doubling_time(series_no_double, time_series))


@pytest.mark.build_test
def test_extract_metrics():
    """Tests the extraction of multiple metrics from a sweep result DataFrame."""
    results_df = pd.DataFrame(
        {
            "time": [0, 10, 20, 30, 40],
            "sds.I[1]&blanket.TBR=1.05": [100, 90, 85, 95, 110],
            "sds.I[1]&blanket.TBR=1.10": [110, 100, 95, 105, 120],
            "another.var&blanket.TBR=1.05": [1, 2, 3, 4, 5],  # Should be ignored
        }
    )
    metrics_definition = {
        "Startup_Inventory": {
            "source_column": "sds.I[1]",
            "method": "calculate_startup_inventory",
        },
        "Self_Sufficiency_Time": {
            "source_column": "sds.I[1]",
            "method": "time_of_turning_point",
        },
        "Final_Value": {
            "source_column": "sds.I[1]",
            "method": "final_value",
        },
    }
    analysis_case = {
        "dependent_variables": [
            "Startup_Inventory",
            "Self_Sufficiency_Time",
            "Final_Value",
        ],
    }

    summary_df = extract_metrics(results_df, metrics_definition, analysis_case)

    assert not summary_df.empty
    assert len(summary_df) == 2
    assert "blanket.TBR" in summary_df.columns
    assert "Startup_Inventory" in summary_df.columns
    assert "Self_Sufficiency_Time" in summary_df.columns
    assert "Final_Value" in summary_df.columns

    # Check values for TBR=1.05
    row1 = summary_df[summary_df["blanket.TBR"] == 1.05]
    assert row1["Startup_Inventory"].iloc[0] == 15.0  # 100 - 85
    assert row1["Self_Sufficiency_Time"].iloc[0] == 20.0
    assert row1["Final_Value"].iloc[0] == 110.0

    # Check values for TBR=1.10
    row2 = summary_df[summary_df["blanket.TBR"] == 1.10]
    assert row2["Startup_Inventory"].iloc[0] == 15.0  # 110 - 95
    assert row2["Self_Sufficiency_Time"].iloc[0] == 20.0
    assert row2["Final_Value"].iloc[0] == 120.0


@pytest.mark.build_test
def test_extract_metrics_bisection_search_skip():
    """Tests that metrics with 'bisection_search' method are correctly skipped."""
    results_df = pd.DataFrame({"time": [0, 10], "sds.I[1]&param=1": [100, 90]})
    metrics_definition = {
        "Required_TBR": {"method": "bisection_search"},
        "Startup_Inventory": {
            "source_column": "sds.I[1]",
            "method": "calculate_startup_inventory",
        },
    }
    analysis_case = {"dependent_variables": ["Required_TBR", "Startup_Inventory"]}
    summary_df = extract_metrics(results_df, metrics_definition, analysis_case)
    assert "Required_TBR" not in summary_df.columns
    assert "Startup_Inventory" in summary_df.columns
    assert len(summary_df) == 1
