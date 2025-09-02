from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def get_final_value(
    series: pd.Series, time_series: Optional[pd.Series] = None
) -> float:
    """
    Returns the final value of a time series.
    """
    return series.iloc[-1]


def calculate_startup_inventory(
    series: pd.Series, time_series: Optional[pd.Series] = None
) -> float:
    """
    Calculates the startup inventory as the difference between the initial
    inventory and the minimum inventory (the turning point).
    """
    initial_inventory = series.iloc[0]
    minimum_inventory = series.min()
    return initial_inventory - minimum_inventory


def time_of_turning_point(series: pd.Series, time_series: pd.Series) -> float:
    """
    Finds the time of the turning point (minimum value) in the series.
    This represents the self-sufficiency time.
    """
    if time_series is None:
        raise ValueError("time_series must be provided for time_of_turning_point")
    min_index = series.idxmin()
    return time_series.loc[min_index]


def calculate_doubling_time(series: pd.Series, time_series: pd.Series) -> float:
    """
    Calculates the time it takes for the inventory to double its initial value.
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
    """
    Extracts summary metrics from the detailed simulation results DataFrame.

    Args:
        results_df: DataFrame from the combined sweep_results.csv.
        metrics_definition: Dictionary defining how to calculate metrics.
        analysis_case: analysis case to identify independent variables.

    Returns:
        A pivoted DataFrame where index are the parameters, columns are metric names,
        and values are the calculated metric values.
    """
    all_params = set()
    all_params.add(analysis_case["independent_variable"])

    analysis_results = []

    source_to_metric = {}
    for metric_name, definition in metrics_definition.items():
        # If the metric is calculated via optimization, it's not extracted from results here.
        # The main simulation script handles it. So, we skip it.
        if definition.get("method") == "bisection_search":
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
