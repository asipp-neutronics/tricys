import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def analyze_rise_dip(results_df: pd.DataFrame, output_dir: str, **kwargs) -> None:
    """Analyzes parameter sweep results to identify curves that fail to exhibit 'dip and rise' feature.

    A curve exhibits the 'dip and rise' feature if:
    1. It has a clear minimum point (not at boundaries)
    2. Values at both start and end are higher than the minimum (with tolerance)

    Args:
        results_df: The combined DataFrame of simulation results, including time and
            multiple parameter combinations.
        output_dir: The directory to save the analysis report.
        **kwargs: Additional parameters from config, e.g., 'output_filename'.

    Note:
        Uses 0.1% smoothing window to handle noisy data. Column names expected in
        format 'variable&param1=v1&param2=v2'. Logs ERROR for each curve without
        the feature. Always generates rise_report.json with analysis results for
        all curves, including 'rises' boolean flag.
    """
    logger.info("Starting post-processing: Analyzing curve rise/dip features...")
    all_curves_info = []
    error_count = 0

    # Iterate over each column of the DataFrame (except for the 'time' column)
    for col_name in results_df.columns:
        if col_name == "time":
            continue

        # Parse parameters from the column name 'variable&param1=v1&param2=v2'
        try:
            parts = col_name.split("&")
            if len(parts) < 2:  # Must have at least one variable name and one parameter
                logger.warning(
                    f"Column name '{col_name}' has an incorrect format, skipping."
                )
                continue

            # parts[0] is the variable name, parse parameters from parts[1:]
            param_parts = parts[1:]
            job_params = dict(item.split("=") for item in param_parts)
            job_params["variable"] = parts[
                0
            ]  # Also add the original variable name to the info

        except (ValueError, IndexError):
            logger.warning(
                f"Could not parse parameters from column name '{col_name}', skipping."
            )
            continue

        series = results_df[col_name]
        rises = False
        if len(series) > 2:
            # This logic is inspired by `time_of_turning_point` from `tricys/analysis/metric.py`.
            # It uses a smoothed series to determine if there is a 'dip and rise' trend.
            window_size = max(1, int(len(series) * 0.001))  # 0.1% smoothing window
            smoothed = series.rolling(
                window=window_size, center=True, min_periods=1
            ).mean()

            min_pos_index = smoothed.idxmin()
            min_val = smoothed.loc[min_pos_index]

            logger.info(
                f"Analyzing curve '{col_name}': min at index {min_pos_index} with value {min_val}"
            )

            # Check if the minimum is at the beginning or end of the series
            is_min_at_boundary = (min_pos_index == smoothed.index[0]) or (
                min_pos_index == smoothed.index[-1]
            )

            if not is_min_at_boundary:
                # Check if it dips from the start and rises to the end.
                # A small tolerance is used to avoid issues with noise.
                series_range = smoothed.max() - smoothed.min()
                # Avoid division by zero or NaN tolerance if series is flat
                if series_range > 1e-9:
                    tolerance = series_range * 0.001  # 0.1% of range as tolerance
                else:
                    tolerance = 0

                start_val = smoothed.iloc[0]
                end_val = smoothed.iloc[-1]

                if start_val > min_val + tolerance and end_val > min_val + tolerance:
                    rises = True

        # Record the analysis result for every curve
        info = job_params.copy()
        info["rises"] = bool(rises)
        all_curves_info.append(info)

        # If the feature is not detected, log it at the ERROR level
        if not rises:
            error_count += 1
            logger.error(
                f"Feature not detected: 'Dip and rise' feature was not found for the curve with parameters {job_params}."
            )

    # Generate a report file with all information unconditionally
    output_filename = kwargs.get("output_filename", "rise_report.json")
    report_path = os.path.join(output_dir, output_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_curves_info, f, indent=4, ensure_ascii=False)

    if error_count > 0:
        logger.info(
            f"{error_count} curves did not exhibit the expected feature. See report for details: {report_path}"
        )
    else:
        logger.info(
            f"All curves exhibit the expected 'dip and rise' feature. Report generated at: {report_path}"
        )
