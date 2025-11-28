import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def check_thresholds(
    results_df: pd.DataFrame, output_dir: str, rules: List[Dict[str, Any]], **kwargs
) -> None:
    """Analyzes simulation results to check if specified columns fall within threshold ranges.

    Supports both single tasks (column name as 'var') and parameter sweep tasks
    (column name as 'var&param=value').

    Args:
        results_df: Merged simulation results DataFrame.
        output_dir: Directory for saving alert reports.
        rules: List of rules, where each rule defines columns and their min/max thresholds.
            Format: [{"columns": ["var1", "var2"], "min": value, "max": value}, ...]
        **kwargs: Additional parameters from configuration, such as 'output_filename'.

    Note:
        Logs ERROR for each threshold violation with peak/dip values. Generates
        alarm_report.json with parsed parameter information and 'has_alarm' flags.
        For columns matching 'base_col_name&param=value', extracts parameters into
        separate fields in the report. Reports total alarm count in logs.
    """
    logger.info("Starting post-processing: Checking thresholds...")

    # Use a dictionary to track the alarm status of each checked column
    checked_columns_status = {}

    for i, rule in enumerate(rules):
        min_val = rule.get("min")
        max_val = rule.get("max")
        columns_to_check = rule.get("columns", [])

        if not columns_to_check:
            logger.warning(f"Rule {i+1} does not specify 'columns', skipping.")
            continue

        # Iterate over each base column name specified in the rule
        for base_col_name in columns_to_check:
            # Iterate over all actual column names in the DataFrame to find matches
            for df_col_name in results_df.columns:
                if df_col_name == base_col_name or df_col_name.startswith(
                    base_col_name + "&"
                ):

                    # Initialize status for this column if it's the first time being checked
                    if df_col_name not in checked_columns_status:
                        checked_columns_status[df_col_name] = (
                            False  # Default to no alarm
                        )

                    # Check for values exceeding the maximum threshold
                    if max_val is not None:
                        exceeded_max = results_df[results_df[df_col_name] > max_val]
                        if not exceeded_max.empty:
                            peak_value = exceeded_max[df_col_name].max()
                            logger.error(
                                f"ALARM: Column '{df_col_name}' exceeds maximum threshold (Threshold: {max_val}, Value: {peak_value})"
                            )
                            checked_columns_status[df_col_name] = True

                    # Check for values falling below the minimum threshold
                    if min_val is not None:
                        exceeded_min = results_df[results_df[df_col_name] < min_val]
                        if not exceeded_min.empty:
                            dip_value = exceeded_min[df_col_name].min()
                            logger.error(
                                f"ALARM: Column '{df_col_name}' is below minimum threshold (Threshold: {min_val}, Value: {dip_value})"
                            )
                            checked_columns_status[df_col_name] = True

    # Convert to the final report format, parsing column names to include parameters
    final_report = []
    for col, status in checked_columns_status.items():
        try:
            report_item = {}
            parts = col.split("&")

            # For single runs, the column name may not contain '&'
            if len(parts) == 1:
                report_item["variable"] = parts[0]
            else:
                variable_name = parts[0]
                param_parts = parts[1:]
                report_item = dict(item.split("=") for item in param_parts)
                report_item["variable"] = variable_name

            report_item["has_alarm"] = status
            final_report.append(report_item)

        except (ValueError, IndexError):
            logger.warning(
                f"Could not parse column name '{col}' for the report, using the original name as a fallback."
            )
            # Fallback to the old format if parsing fails
            final_report.append({"column": col, "has_alarm": status})

    output_filename = kwargs.get("output_filename", "alarm_report.json")
    report_path = os.path.join(output_dir, output_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    total_alarms = sum(1 for entry in final_report if entry["has_alarm"])
    if total_alarms > 0:
        logger.info(
            f"{total_alarms} columns with alarms were found. See logs for details. Report generated at: {report_path}"
        )
    else:
        logger.info(
            f"Threshold check complete. All checked columns are within their thresholds. Report generated at: {report_path}"
        )
