from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from tricys.analysis.hdf5_support import iter_hdf5_job_results
from tricys.utils.hdf5_schema import load_jobs_df


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


def net_tritium_balance(
    series: pd.Series, time_series: Optional[pd.Series] = None
) -> float:
    """Calculates the net change in tritium inventory over the simulation period.

    Args:
        series: The inventory time series data.
        time_series: The corresponding time data (unused).

    Returns:
        The difference between the final and initial inventory values,
        or NaN if the series is empty or contains only NaN values.
    """
    if series.empty or series.isna().all():
        return np.nan
    return series.iloc[-1] - series.iloc[0]


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
            elif method_name in (
                "calculate_startup_inventory",
                "startup_inventory",
                "startup",
                "startup_inv",
                "startup_inventory_deficit",
            ):
                calculation_func = calculate_startup_inventory
            elif method_name == "time_of_turning_point":
                calculation_func = time_of_turning_point
            elif method_name == "calculate_doubling_time":
                calculation_func = calculate_doubling_time
            elif method_name == "net_tritium_balance":
                calculation_func = net_tritium_balance
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


def get_peak_value(series: pd.Series, time_series: Optional[pd.Series] = None) -> float:
    """Gets the peak (maximum) value of a time series."""
    return float(series.max())


def get_min_value(series: pd.Series, time_series: Optional[pd.Series] = None) -> float:
    """Gets the minimum value of a time series."""
    return float(series.min())


def get_max_value(series: pd.Series, time_series: Optional[pd.Series] = None) -> float:
    """Gets the maximum value of a time series."""
    return float(series.max())


def get_mean_value(series: pd.Series, time_series: Optional[pd.Series] = None) -> float:
    """Gets the mean (average) value of a time series."""
    return float(series.mean())


def get_integral_value(
    series: pd.Series, time_series: Optional[pd.Series] = None
) -> float:
    """Calculates the time integral of a series using the trapezoidal rule."""
    if time_series is not None and len(series) == len(time_series) and len(series) > 1:
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        if trapz_fn:
            return float(trapz_fn(series.values, time_series.values))
    return float(series.sum())


def get_settling_time(
    series: pd.Series, time_series: Optional[pd.Series] = None, tolerance: float = 0.02
) -> float:
    """Calculates settling time when series enters and remains within a tolerance band around final value."""
    if time_series is None or len(series) == 0:
        return np.nan
    final_val = series.iloc[-1]
    band = abs(final_val * tolerance) if final_val != 0 else tolerance
    out_of_band = (series - final_val).abs() > band
    if not out_of_band.any():
        return float(time_series.iloc[0])
    last_out_idx = out_of_band[::-1].idxmax()
    loc = series.index.get_loc(last_out_idx)
    if loc + 1 < len(time_series):
        return float(time_series.iloc[loc + 1])
    return float(time_series.iloc[-1])


def calculate_single_job_metrics(
    job_df: pd.DataFrame,
    metrics_definition: Dict[str, Any],
    filter_schema: Optional[Any] = None,
) -> Dict[str, float]:
    """Calculates metrics for a single simulation job.

    Supports both TRICYS 1.0 schema (method, source_column) and
    TRICYS 2.0 schema (evaluate, source, MetricConfig).

    Args:
        job_df: A DataFrame containing the results of a single job.
            Must have 'time' column and variable columns.
        metrics_definition: Dictionary defining the metrics to calculate.
        filter_schema: Optional filter schema (ignored, kept for API compatibility).

    Returns:
        A dictionary mapping metric names to their calculated values.
    """
    results = {}
    if not metrics_definition:
        return results

    for metric_name, raw_def in metrics_definition.items():
        if isinstance(raw_def, dict):
            method_name = raw_def.get("evaluate") or raw_def.get("method")
            source_col = raw_def.get("source") or raw_def.get("source_column")
        elif hasattr(raw_def, "evaluate") and hasattr(raw_def, "source"):
            method_name = getattr(raw_def, "evaluate")
            source_col = getattr(raw_def, "source")
        else:
            continue

        if method_name == "bisection_search":
            continue

        if not source_col or source_col not in job_df.columns:
            continue

        series = job_df[source_col]
        time_series = job_df["time"] if "time" in job_df.columns else None

        calculation_func = None
        if method_name in ("final_value", "final"):
            calculation_func = get_final_value
        elif method_name in ("peak_value", "peak", "max"):
            calculation_func = get_peak_value
        elif method_name == "min":
            calculation_func = get_min_value
        elif method_name in ("mean", "average"):
            calculation_func = get_mean_value
        elif method_name == "integral":
            calculation_func = get_integral_value
        elif method_name == "settling_time":
            calculation_func = get_settling_time
        elif method_name in (
            "calculate_startup_inventory",
            "startup_inventory",
            "startup",
            "startup_inv",
            "startup_inventory_deficit",
        ):
            calculation_func = calculate_startup_inventory
        elif method_name in ("time_of_turning_point", "turning_point"):
            calculation_func = time_of_turning_point
        elif method_name in ("calculate_doubling_time", "doubling_time"):
            calculation_func = calculate_doubling_time
        elif method_name == "net_tritium_balance":
            calculation_func = net_tritium_balance

        if calculation_func:
            try:
                val = calculation_func(series, time_series)
                # Ensure JSON serializable (handle numpy types)
                if hasattr(val, "item"):
                    val = val.item()
                if pd.isna(val):
                    val = None
                results[metric_name] = val
            except Exception:
                pass

    return results


def build_single_job_summary_df(
    job_id: int,
    single_job_metrics: Dict[str, Any],
    metrics_definition: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Build a stable-schema one-row summary DataFrame for HDF5 appends."""
    summary_row = {"job_id": int(job_id)}
    has_metric_value = False

    if metrics_definition:
        for metric_name, raw_def in metrics_definition.items():
            method = (
                raw_def.get("evaluate") or raw_def.get("method")
                if isinstance(raw_def, dict)
                else getattr(raw_def, "evaluate", None)
            )
            if method == "bisection_search":
                metric_value = single_job_metrics.get(metric_name)
                if metric_value is not None:
                    try:
                        summary_row[metric_name] = float(metric_value)
                        has_metric_value = True
                    except (ValueError, TypeError):
                        summary_row[metric_name] = metric_value
                        has_metric_value = True
                continue

            metric_value = single_job_metrics.get(metric_name)
            if metric_value is None:
                summary_row[metric_name] = float("nan")
                continue

            summary_row[metric_name] = float(metric_value)
            has_metric_value = True
    else:
        for k, v in single_job_metrics.items():
            try:
                summary_row[k] = float(v)
                has_metric_value = True
            except (ValueError, TypeError):
                summary_row[k] = v

    if not has_metric_value:
        return pd.DataFrame()

    return pd.DataFrame([summary_row])


def extract_metrics_from_hdf5(
    hdf_path: str,
    metrics_definition: Dict[str, Any],
    analysis_case: Dict[str, Any],
) -> pd.DataFrame:
    """Extracts summary metrics directly from unified HDF5 results without rebuilding a wide dataframe."""
    jobs_df = load_jobs_df(hdf_path)
    if jobs_df.empty:
        return pd.DataFrame()

    required_columns = []
    for metric_name in analysis_case.get("dependent_variables", []):
        definition = metrics_definition.get(metric_name)
        if not definition or definition.get("method") == "bisection_search":
            continue
        source_col = definition.get("source_column")
        if source_col:
            required_columns.append(source_col)

    required_columns = list(dict.fromkeys(required_columns))
    rows = []

    for _, job_params, job_df in iter_hdf5_job_results(
        hdf_path, jobs_df=jobs_df, columns=required_columns
    ):
        metric_values = calculate_single_job_metrics(job_df, metrics_definition)
        if not metric_values:
            continue
        row = job_params.copy()
        row.update(metric_values)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def get_optimization_tasks(config: dict) -> list[str]:
    """Identifies all valid optimization tasks from the configuration.

    A valid optimization task is a dependent variable (or metric) that is defined
    in task.metrics, metrics_definition, or sensitivity_analysis with bisection_search.
    """
    optimization_tasks: list[str] = []

    metrics_sources: list[dict] = []
    task_metrics = (
        config.get("task", {}).get("metrics", {})
        if isinstance(config.get("task"), dict)
        else {}
    )
    if task_metrics and isinstance(task_metrics, dict):
        metrics_sources.append(task_metrics)
    if config.get("metrics_definition") and isinstance(
        config.get("metrics_definition"), dict
    ):
        metrics_sources.append(config["metrics_definition"])
    sens_metrics = config.get("sensitivity_analysis", {}).get("metrics_definition", {})
    if sens_metrics and isinstance(sens_metrics, dict):
        metrics_sources.append(sens_metrics)

    sensitivity_analysis = config.get("sensitivity_analysis", {})
    analysis_case = sensitivity_analysis.get("analysis_case")
    if analysis_case is None:
        cases = sensitivity_analysis.get("analysis_cases")
        if cases and isinstance(cases, list) and len(cases) > 0:
            analysis_case = cases[0]
        else:
            analysis_case = {}

    dependent_vars = analysis_case.get("dependent_variables", [])

    for src in metrics_sources:
        for var in dependent_vars:
            if var.startswith("Required_") and var not in optimization_tasks:
                if var in src:
                    cfg_val = src[var]
                    method = (
                        cfg_val.get("evaluate") or cfg_val.get("method")
                        if isinstance(cfg_val, dict)
                        else getattr(cfg_val, "evaluate", None)
                    )
                    if method == "bisection_search":
                        optimization_tasks.append(var)

    # Also allow direct detection from any metrics source
    if not optimization_tasks:
        for src in metrics_sources:
            for var, cfg_val in src.items():
                method = (
                    cfg_val.get("evaluate") or cfg_val.get("method")
                    if isinstance(cfg_val, dict)
                    else getattr(cfg_val, "evaluate", None)
                )
                if method == "bisection_search":
                    if var not in optimization_tasks:
                        optimization_tasks.append(var)

    return optimization_tasks
