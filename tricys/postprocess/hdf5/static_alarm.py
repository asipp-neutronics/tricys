import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def check_thresholds(
    results_file_path: str, output_dir: str, rules: List[Dict[str, Any]], **kwargs
) -> None:
    """Analyzes HDF5 simulation results to check if specified columns fall within threshold ranges.

    Args:
        results_file_path: Path to HDF5 results.
        output_dir: Directory for saving alert reports.
        rules: List of rules.
        **kwargs: Additional parameters.
    """
    logger.info("Starting HDF5 post-processing: Checking thresholds...")

    output_filename = kwargs.get("output_filename", "alarm_report.json")
    report_path = os.path.join(output_dir, output_filename)
    final_report = []

    if not os.path.exists(results_file_path):
        logger.error(f"Results file not found: {results_file_path}")
        return

    try:
        with pd.HDFStore(results_file_path, mode="r") as store:
            if "/jobs" not in store.keys() or "/results" not in store.keys():
                return

            # Load jobs map
            jobs_df = store.select("jobs")
            jobs_map = jobs_df.set_index("job_id").to_dict(orient="index")

            # Check which columns exist in results
            # We can use a small select to get columns
            storer = store.get_storer("results")
            available_vars = storer.table.colnames  # This includes 'time', 'job_id'

            # Track alarms to avoid duplicates?
            # Original script produces one entry per "Curve" (Variable + Job).
            # We will generate entries for ALARMED curves.
            # For non-alarmed curves, original script lists them as has_alarm=False.
            # To match that, we theoretically need to list ALL Job x Variable combinations.
            # That could be huge (1.8M * N_vars).
            # Assuming we only care about Alarms or explicitly checked vars.

            # Optimization:
            # 1. Find all Job IDs that FAIL the check.
            # 2. Report them.
            # 3. What about passing jobs?
            #    If the user expects a report of ALL jobs, we need to iterate all jobs.
            #    If the log just says "X columns with alarms", maybe we only report alarms?
            #    The original script: `final_report.append(...)` for `checked_columns_status.items()`.
            #    `checked_columns_status` contains every column found that matches a rule.
            #    So yes, it reports everything.
            #    HDF5 approach: We CANNOT dump 1.8M lines into a JSON easily/usefully.
            #    COMPROMISE: We will only report ALARMS in the JSON to keep it manageable.
            #    (Or maybe purely stats?) -> Let's stick to reporting Alarms.
            #    (If the user insists on full report, we'd need pagination or another HDF5).

            total_alarms = 0

            for i, rule in enumerate(rules):
                min_val = rule.get("min")
                max_val = rule.get("max")
                columns_to_check = rule.get("columns", [])

                for col in columns_to_check:
                    if col not in available_vars:
                        # try looking for matching prefix if exact match fails?
                        # In HDF5 table, columns are exact.
                        continue

                    # Find violating jobs
                    alarm_job_ids = set()

                    if max_val is not None:
                        try:
                            # Select rows where col > max
                            # We only need job_id and the value
                            res = store.select(
                                "results",
                                where=f"{col} > {max_val}",
                                columns=["job_id", col],
                            )
                            if not res.empty:
                                ids = res["job_id"].unique()
                                alarm_job_ids.update(ids)
                                for j_id in ids:
                                    peak = res[res["job_id"] == j_id][col].max()
                                    logger.error(
                                        f"ALARM: Job {j_id}, Var '{col}' > {max_val} (Peak: {peak})"
                                    )
                        except Exception as e:
                            logger.error(f"Query failed for {col} > {max_val}: {e}")

                    if min_val is not None:
                        try:
                            res = store.select(
                                "results",
                                where=f"{col} < {min_val}",
                                columns=["job_id", col],
                            )
                            if not res.empty:
                                ids = res["job_id"].unique()
                                alarm_job_ids.update(ids)
                                for j_id in ids:
                                    dip = res[res["job_id"] == j_id][col].min()
                                    logger.error(
                                        f"ALARM: Job {j_id}, Var '{col}' < {min_val} (Dip: {dip})"
                                    )
                        except Exception as e:
                            logger.error(f"Query failed for {col} < {min_val}: {e}")

                    # Add to report
                    for j_id in alarm_job_ids:
                        if j_id in jobs_map:
                            item = jobs_map[j_id].copy()
                            item["variable"] = col
                            item["has_alarm"] = True
                            item["job_id"] = int(j_id)
                            final_report.append(item)
                            total_alarms += 1

    except Exception as e:
        logger.error(f"HDF5 threshold check failed: {e}", exc_info=True)

    # Save report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    if total_alarms > 0:
        logger.info(f"Found {total_alarms} alarms. Report: {report_path}")
    else:
        logger.info(f"No alarms found. Report: {report_path}")
