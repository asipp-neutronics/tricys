from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from tricys.analysis.hdf5_support import iter_hdf5_job_results
from tricys.utils.hdf5_schema import load_jobs_df

TIME_UNIT_FACTORS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
    "y": 31536000.0,
    "yr": 31536000.0,
    "year": 31536000.0,
    "years": 31536000.0,
}

TIME_OPERATOR_METHODS = {
    "time_of_turning_point",
    "turning_point",
    "time_of_valley",
    "valley_time",
    "nadir_time",
    "calculate_doubling_time",
    "doubling_time",
    "settling_time",
    "get_settling_time",
}


def apply_metric_unit_scaling(val: Any, method_name: str, raw_def: Any) -> Any:
    """Scale metric value according to conversion_factor or unit definitions."""
    if val is None or pd.isna(val):
        return val

    conv_factor = None
    unit = None
    if isinstance(raw_def, dict):
        conv_factor = raw_def.get("conversion_factor")
        unit = raw_def.get("unit")
    else:
        conv_factor = getattr(raw_def, "conversion_factor", None)
        unit = getattr(raw_def, "unit", None)

    if conv_factor is not None:
        try:
            return float(val) / float(conv_factor)
        except (ValueError, TypeError, ZeroDivisionError):
            return val

    if unit is not None and isinstance(unit, str):
        normalized_unit = unit.strip().lower()
        if method_name in TIME_OPERATOR_METHODS:
            factor = TIME_UNIT_FACTORS.get(normalized_unit)
            if factor is not None:
                return float(val) / factor

    return val


def apply_series_smoothing(
    series: pd.Series,
    time_series: Optional[pd.Series] = None,
    smooth_window: Optional[Union[float, int, str]] = None,
    method: str = "mean",
    center: bool = True,
) -> pd.Series:
    """Applies temporal rolling window smoothing to a time-series.

    Used to eliminate high-frequency oscillations or pulsing cycles (e.g., reactor
    burn/dwell or availability on/off cycles) before calculating metrics such as
    doubling time, nadir/minimum, final value, turning points, or settling time.

    Args:
        series: Data series to smooth.
        time_series: Time series corresponding to data points.
        smooth_window: Physical time window width (in the same units as time_series).
            If None, non-positive, or unparseable, returns the original series untouched.
        method: Smoothing method: "mean" (rolling average) or "median" (median filter).
        center: Whether the rolling window is centered (default True).

    Returns:
        Smoothed pd.Series with the same index and length as series.
    """
    if smooth_window is None:
        return series
    try:
        w_val = float(smooth_window)
    except (ValueError, TypeError):
        return series

    if w_val <= 0 or len(series) < 3:
        return series

    if (
        time_series is not None
        and len(time_series) == len(series)
        and len(time_series) > 1
    ):
        dt = float(np.median(np.diff(time_series.values)))
        if dt <= 0:
            dt = float(time_series.iloc[-1] - time_series.iloc[0]) / max(
                1, len(time_series) - 1
            )
        if dt > 0:
            window_size = max(1, int(round(w_val / dt)))
        else:
            window_size = max(1, int(round(w_val)))
    else:
        window_size = max(1, int(round(w_val)))

    if window_size <= 1:
        return series

    roller = series.rolling(window=window_size, center=center, min_periods=1)
    if method == "median":
        return roller.median()
    return roller.mean()


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


def calculate_dynamic_deficit(
    series: pd.Series, time_series: Optional[pd.Series] = None
) -> float:
    """Calculates the dynamic inventory deficit / drawdown depth (Delta_I_V).

    In the literature (Abdou et al. 2021 Eq. 3.8), this is Delta_I_V = I(0) - min(I(t)),
    representing the transient inventory holdup deficit accumulated before recycled
    tritium returns from downstream processing systems.

    Args:
        series: The inventory time series data.
        time_series: The corresponding time data (unused).

    Returns:
        The calculated dynamic inventory deficit.
    """
    initial_inventory = series.iloc[0]
    minimum_inventory = series.min()
    return float(initial_inventory - minimum_inventory)


def calculate_startup_inventory(
    series: pd.Series, time_series: Optional[pd.Series] = None
) -> float:
    """Calculates the startup inventory deficit.

    Maintained for backward compatibility. In TRICYS legacy usage, this returns
    I(0) - min(I(t)), which corresponds physically to Delta_I_V (dynamic deficit).
    For the literature start-up inventory I_S^0 that includes reserve inventory I_r,
    use calculate_required_startup_inventory or model variable required_startup_inventory.
    """
    return calculate_dynamic_deficit(series, time_series)


def calculate_required_startup_inventory(
    series: pd.Series,
    time_series: Optional[pd.Series] = None,
    reserve_inventory: float = 0.0,
) -> float:
    """Calculates the total required start-up inventory I_S^0.

    Per Abdou et al. (2021) Eq. (3.8):
        I_S^0 = Delta_I_V + I_r
    where Delta_I_V is the dynamic transit deficit and I_r is the safety reserve inventory.

    Args:
        series: The inventory time series data.
        time_series: The corresponding time data (unused).
        reserve_inventory: Safety buffer I_r (default: 0.0).

    Returns:
        The total required start-up inventory.
    """
    delta_v = calculate_dynamic_deficit(series, time_series)
    return float(delta_v + reserve_inventory)


def time_of_turning_point(
    series: pd.Series,
    time_series: pd.Series,
    smooth_window: Optional[Union[float, int, str]] = None,
) -> float:
    """Finds the time of the turning point (minimum value) in a series.

    This function identifies the time corresponding to the minimum value in the
    series, which often represents the self-sufficiency time in tritium inventory
    simulations. To handle noisy data, it first smooths the series to find the
    general trend's minimum. If the smoothed minimum is not at the boundaries,
    it returns the time of the absolute minimum from the original data.

    Args:
        series: The time series data to analyze.
        time_series: The corresponding time data.
        smooth_window: Optional physical time window for smoothing. If provided (> 0),
            uses apply_series_smoothing; otherwise falls back to 0.1% rolling window.

    Returns:
        The time of the turning point, or NaN if the trend is monotonic.

    Raises:
        ValueError: If time_series is None.
    """
    if time_series is None:
        raise ValueError("time_series must be provided for time_of_turning_point")

    if smooth_window is not None:
        try:
            sw_val = float(smooth_window)
            if sw_val > 0:
                smoothed_series = apply_series_smoothing(
                    series, time_series=time_series, smooth_window=sw_val
                )
            else:
                window_size = max(1, int(len(series) * 0.001))
                smoothed_series = series.rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
        except (ValueError, TypeError):
            window_size = max(1, int(len(series) * 0.001))
            smoothed_series = series.rolling(
                window=window_size, center=True, min_periods=1
            ).mean()
    else:
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


def calculate_doubling_time(
    series: pd.Series,
    time_series: pd.Series,
    interpolate: bool = False,
    cycle_period: Optional[Union[float, int, str]] = None,
    smooth_window: Optional[Union[float, int, str]] = None,
) -> float:
    """Calculates the time it takes for the inventory to double its initial value.

    This function finds the first time point, after the inventory's minimum
    (turning point), where the inventory level reaches or exceeds twice its
    initial value.

    Args:
        series: The inventory time series data.
        time_series: The corresponding time data.
        interpolate: If True, uses linear interpolation between adjacent grid points
            to determine fractional doubling time. Defaults to False.
        cycle_period: Alias for smooth_window.
        smooth_window: If provided (> 0), applies a rolling-mean filter over the
            specified period to eliminate transient crest false-triggering before threshold evaluation.
            If None or <= 0, no smoothing is performed (default).

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
    initial_inventory = float(series.iloc[0])
    doubled_inventory = 2.0 * initial_inventory

    sw = smooth_window if smooth_window is not None else cycle_period
    eval_series = series
    if sw is not None:
        eval_series = apply_series_smoothing(
            series, time_series=time_series, smooth_window=sw
        )

    # Find the first index where the inventory is >= doubled_inventory
    # We should only consider the part of the series after the turning point
    min_index = eval_series.idxmin()
    after_turning_point_series = eval_series.loc[min_index:]

    doubling_indices = after_turning_point_series[
        after_turning_point_series >= doubled_inventory
    ].index

    if not doubling_indices.empty:
        idx1 = doubling_indices[0]
        if interpolate:
            loc1 = after_turning_point_series.index.get_loc(idx1)
            if loc1 > 0:
                idx0 = after_turning_point_series.index[loc1 - 1]
                y0 = float(after_turning_point_series.loc[idx0])
                y1 = float(after_turning_point_series.loc[idx1])
                t0 = float(time_series.loc[idx0])
                t1 = float(time_series.loc[idx1])
                if y1 > y0 and y0 < doubled_inventory <= y1:
                    return float(t0 + (doubled_inventory - y0) / (y1 - y0) * (t1 - t0))
        return float(time_series.loc[idx1])
    else:
        # Check if inventory has passed turning point and is growing steadily
        if len(after_turning_point_series) > 10:
            tail_n = max(5, int(len(after_turning_point_series) * 0.3))
            tail_y = after_turning_point_series.iloc[-tail_n:].values
            tail_t = time_series.loc[after_turning_point_series.index[-tail_n:]].values
            dt = tail_t[-1] - tail_t[0]
            if dt > 0:
                slope = (tail_y[-1] - tail_y[0]) / dt
                if slope > 0:
                    remaining_growth = doubled_inventory - tail_y[-1]
                    extrapolated_time = tail_t[-1] + remaining_growth / slope
                    return float(extrapolated_time)
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
                "definition": definition,
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
            metric_def = metric_info.get("definition") or {}

            if method_name in ("final_value", "last_value"):
                calculation_func = get_final_value
            elif method_name in (
                "min_value",
                "valley_inventory",
                "minimum",
                "nadir_inventory",
                "nadir",
            ):
                calculation_func = get_min_value
            elif method_name in (
                "calculate_dynamic_deficit",
                "dynamic_deficit",
                "startup_deficit",
                "inventory_drawdown",
                "calculate_startup_inventory",
                "startup_inventory",
                "startup",
                "startup_inv",
                "startup_inventory_deficit",
            ):
                calculation_func = calculate_dynamic_deficit
            elif method_name in (
                "calculate_required_startup_inventory",
                "required_startup_inventory",
                "true_startup_inventory",
            ):
                calculation_func = calculate_required_startup_inventory
            elif method_name in (
                "time_of_turning_point",
                "turning_point",
                "time_of_valley",
                "valley_time",
                "nadir_time",
            ):
                calculation_func = time_of_turning_point
            elif method_name in ("calculate_doubling_time", "doubling_time"):
                calculation_func = calculate_doubling_time
            elif method_name == "net_tritium_balance":
                calculation_func = net_tritium_balance
            else:
                print(
                    f"Warning: Calculation method '{method_name}' not implemented. Skipping."
                )
                continue

            series_to_eval = results_df[col_name]
            time_series = results_df["time"] if "time" in results_df.columns else None

            smooth_w = (
                metric_def.get("smooth_window") or metric_def.get("cycle_period")
                if isinstance(metric_def, dict)
                else getattr(metric_def, "smooth_window", None)
                or getattr(metric_def, "cycle_period", None)
            )
            smooth_method = (
                metric_def.get("smooth_method", "mean")
                if isinstance(metric_def, dict)
                else getattr(metric_def, "smooth_method", "mean") or "mean"
            )
            if smooth_w is not None and time_series is not None:
                series_to_eval = apply_series_smoothing(
                    series_to_eval,
                    time_series=time_series,
                    smooth_window=smooth_w,
                    method=smooth_method,
                )

            if calculation_func == calculate_required_startup_inventory:
                res_val = 0.0
                if isinstance(metric_def, dict):
                    res_src = (
                        metric_def.get("reserve_source")
                        or metric_def.get("threshold_source")
                        or metric_def.get("reserve_column")
                    )
                    if res_src and res_src in results_df.columns:
                        res_val = float(results_df[res_src].iloc[-1])
                    elif "comp11_Storage.I_reserve_req" in results_df.columns:
                        res_val = float(
                            results_df["comp11_Storage.I_reserve_req"].iloc[-1]
                        )
                    elif "I_reserve_req" in results_df.columns:
                        res_val = float(results_df["I_reserve_req"].iloc[-1])
                    else:
                        res_val = float(metric_def.get("reserve_inventory", 0.0))
                metric_value = calculation_func(
                    series_to_eval, time_series, reserve_inventory=res_val
                )
            elif calculation_func == calculate_doubling_time:
                interpolate = False
                if isinstance(metric_def, dict):
                    interpolate = bool(metric_def.get("interpolate", False))
                else:
                    interpolate = bool(getattr(metric_def, "interpolate", False))
                metric_value = calculation_func(
                    series_to_eval, time_series, interpolate=interpolate
                )
            elif calculation_func == time_of_turning_point:
                metric_value = calculation_func(
                    series_to_eval, time_series, smooth_window=smooth_w
                )
            else:
                metric_value = calculation_func(series_to_eval, time_series)
            metric_value = apply_metric_unit_scaling(
                metric_value, method_name, metric_def
            )

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
        elif method_name in (
            "min",
            "min_value",
            "valley_inventory",
            "minimum",
            "nadir_inventory",
            "nadir",
        ):
            calculation_func = get_min_value
        elif method_name in ("mean", "average"):
            calculation_func = get_mean_value
        elif method_name == "integral":
            calculation_func = get_integral_value
        elif method_name == "settling_time":
            calculation_func = get_settling_time
        elif method_name in (
            "calculate_dynamic_deficit",
            "dynamic_deficit",
            "startup_deficit",
            "inventory_drawdown",
            "calculate_startup_inventory",
            "startup_inventory",
            "startup",
            "startup_inv",
            "startup_inventory_deficit",
        ):
            calculation_func = calculate_dynamic_deficit
        elif method_name in (
            "calculate_required_startup_inventory",
            "required_startup_inventory",
            "true_startup_inventory",
        ):
            calculation_func = calculate_required_startup_inventory
        elif method_name in (
            "time_of_turning_point",
            "turning_point",
            "time_of_valley",
            "valley_time",
            "nadir_time",
        ):
            calculation_func = time_of_turning_point
        elif method_name in ("calculate_doubling_time", "doubling_time"):
            calculation_func = calculate_doubling_time
        elif method_name == "net_tritium_balance":
            calculation_func = net_tritium_balance

        # Extract user-defined smoothing and interpolation options
        smooth_w = (
            raw_def.get("smooth_window") or raw_def.get("cycle_period")
            if isinstance(raw_def, dict)
            else getattr(raw_def, "smooth_window", None)
            or getattr(raw_def, "cycle_period", None)
        )
        smooth_method = (
            raw_def.get("smooth_method", "mean")
            if isinstance(raw_def, dict)
            else getattr(raw_def, "smooth_method", "mean") or "mean"
        )
        interpolate = (
            bool(raw_def.get("interpolate", False))
            if isinstance(raw_def, dict)
            else bool(getattr(raw_def, "interpolate", False))
        )

        eval_series = series
        if smooth_w is not None and time_series is not None:
            # Check if smooth_w refers to a parameter column in job_df
            smooth_val = None
            if isinstance(smooth_w, str) and smooth_w in job_df.columns:
                try:
                    smooth_val = float(job_df[smooth_w].iloc[0])
                except (ValueError, TypeError):
                    smooth_val = None
            else:
                try:
                    smooth_val = float(smooth_w)
                except (ValueError, TypeError):
                    smooth_val = None

            if smooth_val is not None and smooth_val > 0:
                eval_series = apply_series_smoothing(
                    series,
                    time_series=time_series,
                    smooth_window=smooth_val,
                    method=smooth_method,
                )

        if calculation_func:
            try:
                if calculation_func == calculate_required_startup_inventory:
                    res_val = 0.0
                    if isinstance(raw_def, dict):
                        res_src = raw_def.get("reserve_source")
                        if res_src and res_src in job_df.columns:
                            res_val = float(job_df[res_src].iloc[-1])
                        else:
                            res_val = float(raw_def.get("reserve_inventory", 0.0))
                    val = calculation_func(
                        eval_series, time_series, reserve_inventory=res_val
                    )
                elif calculation_func == calculate_doubling_time:
                    val = calculation_func(
                        eval_series, time_series, interpolate=interpolate
                    )
                elif calculation_func == time_of_turning_point:
                    val = calculation_func(
                        eval_series, time_series, smooth_window=smooth_w
                    )
                else:
                    val = calculation_func(eval_series, time_series)
                # Ensure JSON serializable (handle numpy types)
                if hasattr(val, "item"):
                    val = val.item()
                if pd.isna(val):
                    val = None
                else:
                    val = apply_metric_unit_scaling(val, method_name, raw_def)
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
