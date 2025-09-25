import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_rise_dip(results_df: pd.DataFrame, output_dir: str, **kwargs):
    """
    Analyzes the combined results of a parameter sweep and reports the curves
    that fail to exhibit the 'dip and rise' feature.

    Args:
        results_df (pd.DataFrame): The combined DataFrame of simulation results, including time and multiple parameter combinations.
        output_dir (str): The directory to save the analysis report.
        **kwargs: Additional parameters from the config, e.g., 'output_filename'.
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

        data = results_df[col_name].to_numpy()
        rises = False
        if len(data) > 2:
            diffs = np.diff(data)
            mid_index = len(diffs) // 2
            has_dip = np.any(diffs[:mid_index] < 0)
            has_rise = np.any(diffs[mid_index:] > 0)
            rises = has_dip and has_rise

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
