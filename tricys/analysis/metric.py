from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def get_final_value(
    series: pd.Series, time_series: Optional[pd.Series] = None
) -> float:
    """Gets the final value of a time series.

    Args:
        series: The time series data.
        time_series: The corresponding time data (unused).

    Returns:
        The last value in the series.

    Note:
        The time_series parameter is kept for interface consistency but is not used
        in the calculation. Only the series data is required.
    """
    return series.iloc[-1]


def calculate_startup_inventory(
    series: pd.Series, time_series: Optional[pd.Series] = None
) -> float:
    """Calculates the startup inventory.

    The startup inventory is calculated as the difference between the initial
    inventory and the minimum inventory (the turning point).

    Args:
        series: The inventory time series data.
        time_series: The corresponding time data (unused).

    Returns:
        The calculated startup inventory.

    Note:
        The time_series parameter is provided for interface consistency but is not
        used in the calculation. The startup inventory represents the amount of
        inventory consumed before reaching the minimum point.
    """
    initial_inventory = series.iloc[0]
    minimum_inventory = series.min()
    return initial_inventory - minimum_inventory


def time_of_turning_point(series: pd.Series, time_series: pd.Series) -> float:
    """Finds the time of the turning point (minimum value) in a series.

    This function identifies the time corresponding to the minimum value in the
    series, which often represents the self-sufficiency time in tritium inventory
    simulations. To handle noisy data, it first smooths the series to find the
    general trend's minimum. If the smoothed minimum is not at the boundaries,
    it returns the time of the absolute minimum from the original data.

    Args:
        series: The time series data to analyze.
        time_series: The corresponding time data.

    Returns:
        The time of the turning point, or NaN if the trend is monotonic.

    Raises:
        ValueError: If time_series is None.

    Note:
        Uses a rolling window (0.1% of data length) for smoothing to identify the
        general trend. If the smoothed minimum is within the last 30% of the series,
        the trend is considered monotonic and NaN is returned. Otherwise, returns
        the time of the absolute minimum in the original data.
    """

    print(
        f"Calculating time_of_turning_point for series with length {len(series)} and {len(time_series)} "
    )
    if time_series is None:
        raise ValueError("time_series must be provided for time_of_turning_point")

    # Define a window size for the rolling average, e.g., 5% of the data length
    # with a minimum size of 1. This helps in smoothing out local fluctuations.
    window_size = max(1, int(len(series) * 0.001))
    smoothed_series = series.rolling(
        window=window_size, center=True, min_periods=1
    ).mean()

    # Find the index label of the minimum value in the smoothed series.
    smooth_min_index = smoothed_series.idxmin()
    min_index = series.idxmin()

    # Check if the minimum of the smoothed data is within the first or last 5%
    # of the series. If so, the trend is considered monotonic.
    smooth_min_pos = series.index.get_loc(smooth_min_index)
    five_percent_threshold = int(len(series) * 0.3)

    if smooth_min_pos >= len(series) - five_percent_threshold:
        return np.nan
    else:
        # A clear turning point is identified in the overall trend.
        # Now, find the precise turning point in the original, noisy data.
        min_index = series.idxmin()
        return time_series.loc[min_index]


def calculate_doubling_time(series: pd.Series, time_series: pd.Series) -> float:
    """Calculates the time it takes for the inventory to double its initial value.

    This function finds the first time point, after the inventory's minimum
    (turning point), where the inventory level reaches or exceeds twice its
    initial value.

    Args:
        series: The inventory time series data.
        time_series: The corresponding time data.

    Returns:
        The doubling time, or NaN if the inventory never doubles.

    Raises:
        ValueError: If time_series is None.

    Note:
        Only considers the portion of the series after the turning point (minimum).
        Returns NaN if the inventory never reaches twice the initial value in the
        post-turning-point region.
    """
    if time_series is None:
        raise ValueError("time_series must be provided for calculate_doubling_time")
    initial_inventory = series.iloc[0]
    doubled_inventory = 2 * initial_inventory

    # Find the first index where the inventory is >= doubled_inventory
    # We should only consider the part of the series after the turning point
    min_index = series.idxmin()
    after_turning_point_series = series.loc[min_index:]

    doubling_indices = after_turning_point_series[
        after_turning_point_series >= doubled_inventory
    ].index

    if not doubling_indices.empty:
        doubling_index = doubling_indices[0]
        return time_series.loc[doubling_index]
    else:
        # If it never doubles, return NaN
        return np.nan


def extract_metrics(
    results_df: pd.DataFrame,
    metrics_definition: Dict[str, Any],
    analysis_case: Dict[str, Any],
) -> pd.DataFrame:
    """Extracts summary metrics from detailed simulation results.

    This function processes a DataFrame from a parameter sweep, calculates
    various metrics for each run based on a definitions dictionary, and
    pivots the results into a summary DataFrame where each row corresponds
    to a unique parameter combination.

    Args:
        results_df: DataFrame from the combined sweep results.
        metrics_definition: Dictionary defining how to calculate each metric
            (e.g., source column, method).
        analysis_case: The analysis case configuration, used to identify
            dependent variables.

    Returns:
        A pivoted DataFrame with parameters as the index and metric names as columns.

    Note:
        Parses column names in format "variable&param1=value1&param2=value2" to extract
        parameter values. Skips metrics with "bisection_search" method. Returns empty
        DataFrame if no valid metrics are found or if pivoting fails.
    """

    analysis_results = []

    source_to_metric = {}
    dependent_vars = analysis_case.get("dependent_variables", [])

    for metric_name in dependent_vars:
        definition = metrics_definition.get(metric_name)

        # If the metric is not in the definition or is calculated via optimization, skip it.
        if not definition or definition.get("method") == "bisection_search":
            continue

        source = definition["source_column"]
        if source not in source_to_metric:
            source_to_metric[source] = []
        source_to_metric[source].append(
            {
                "metric_name": metric_name,
                "method": definition["method"],
            }
        )

    for col_name in results_df.columns:
        if col_name.lower() == "time":
            continue

        source_var = None
        for var in source_to_metric.keys():
            if col_name.startswith(var):
                source_var = var
                break

        if not source_var:
            continue

        param_str = col_name[len(source_var) :].lstrip("&")

        try:
            params = dict(item.split("=") for item in param_str.split("&"))
        except ValueError:
            print(
                f"Warning: Could not parse parameters from column '{col_name}'. Skipping."
            )
            continue

        for k, v in params.items():
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v

        for metric_info in source_to_metric[source_var]:
            method_name = metric_info["method"]
            metric_name = metric_info["metric_name"]

            if method_name == "final_value":
                calculation_func = get_final_value
            elif method_name == "calculate_startup_inventory":
                calculation_func = calculate_startup_inventory
            elif method_name == "time_of_turning_point":
                calculation_func = time_of_turning_point
            elif method_name == "calculate_doubling_time":
                calculation_func = calculate_doubling_time
            else:
                print(
                    f"Warning: Calculation method '{method_name}' not implemented. Skipping."
                )
                continue

            metric_value = calculation_func(results_df[col_name], results_df["time"])

            result_row = params.copy()
            result_row["metric_name"] = metric_name
            result_row["metric_value"] = metric_value
            analysis_results.append(result_row)

    if not analysis_results:
        return pd.DataFrame()

    summary_df = pd.DataFrame(analysis_results)

    # Dynamically identify all parameter columns from the dataframe
    param_cols = [
        col for col in summary_df.columns if col not in ["metric_name", "metric_value"]
    ]

    if not param_cols:
        return pd.DataFrame()

    try:
        pivot_df = summary_df.pivot_table(
            index=param_cols, columns="metric_name", values="metric_value"
        ).reset_index()
        return pivot_df
    except Exception as e:
        print(f"Error during pivoting: {e}")
        return pd.DataFrame()
