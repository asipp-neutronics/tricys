import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd

from tricys.utils.hdf5_schema import RESULTS_KEY, get_jobs_key

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

    final_report = []
    total_alarms = 0
    report_only_alarms = kwargs.get("report_only_alarms", False)

    if not os.path.exists(results_file_path):
        logger.error(f"Results file not found: {results_file_path}")
        return

    try:
        with pd.HDFStore(results_file_path, mode="r") as store:
            if f"/{RESULTS_KEY}" not in store.keys():
                return

            try:
                jobs_key = get_jobs_key(store)
            except KeyError:
                return

            jobs_df = store.select(jobs_key)
            if "job_id" not in jobs_df.columns:
                if jobs_df.index.name == "job_id":
                    jobs_df = jobs_df.reset_index()
                elif "index" in jobs_df.columns:
                    jobs_df = jobs_df.rename(columns={"index": "job_id"})
                else:
                    jobs_df["job_id"] = jobs_df.index + 1
            jobs_map = jobs_df.set_index("job_id").to_dict(orient="index")
            available_vars = store.get_storer(RESULTS_KEY).table.colnames

            for rule in rules:
                min_val = rule.get("min")
                max_val = rule.get("max")
                columns_to_check = rule.get("columns", [])

                for col in columns_to_check:
                    if col not in available_vars:
                        continue

                    alarm_job_ids = set()

                    if max_val is not None:
                        try:
                            res = store.select(
                                RESULTS_KEY,
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
                                RESULTS_KEY,
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

                    target_job_ids = (
                        alarm_job_ids if report_only_alarms else jobs_map.keys()
                    )

                    for j_id in target_job_ids:
                        if j_id in jobs_map:
                            has_alarm = j_id in alarm_job_ids
                            item = jobs_map[j_id].copy()
                            item["variable"] = col
                            item["has_alarm"] = has_alarm
                            item["job_id"] = int(j_id)
                            final_report.append(item)
                            if has_alarm:
                                total_alarms += 1

    except Exception as e:
        logger.error(f"HDF5 threshold check failed: {e}", exc_info=True)

    output_filename = kwargs.get("output_filename", "alarm_report.json")
    report_path = os.path.join(output_dir, output_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    if total_alarms > 0:
        logger.info(f"Found {total_alarms} alarms. Report: {report_path}")
    else:
        logger.info(f"No alarms found. Report: {report_path}")
