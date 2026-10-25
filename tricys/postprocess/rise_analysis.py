import json
import logging
import os

import pandas as pd

from tricys.utils.hdf5_schema import RESULTS_KEY, get_jobs_key

logger = logging.getLogger(__name__)


def analyze_rise_dip(results_file_path: str, output_dir: str, **kwargs) -> None:
    """Analyzes HDF5 simulation results to identify curves that fail to exhibit 'dip and rise' feature.

    Args:
        results_file_path: Path to the HDF5 file containing 'results' and 'jobs' tables.
        output_dir: The directory to save the analysis report.
        **kwargs: Additional parameters.
    """
    logger.info("Starting HDF5 post-processing: Analyzing curve rise/dip features...")
    all_curves_info = []
    error_count = 0

    if not os.path.exists(results_file_path):
        logger.error(f"Results file not found: {results_file_path}")
        return

    try:
        with pd.HDFStore(results_file_path, mode="r") as store:
            if f"/{RESULTS_KEY}" not in store.keys():
                logger.error("HDF5 file missing 'results' table.")
                return

            try:
                jobs_key = get_jobs_key(store)
            except KeyError:
                logger.error("HDF5 file missing 'jobs' table.")
                return

            jobs_df = store.select(jobs_key)
            if "job_id" not in jobs_df.columns:
                jobs_df = jobs_df.reset_index()
            if "job_id" not in jobs_df.columns:
                jobs_df["job_id"] = jobs_df.index + 1
            jobs_map = jobs_df.set_index("job_id").to_dict(orient="index")
            job_ids = sorted(jobs_map.keys())

            def check_curve(series, job_params, var_name):
                rises = False
                if len(series) > 2:
                    window_size = max(1, int(len(series) * 0.001))
                    smoothed = series.rolling(
                        window=window_size, center=True, min_periods=1
                    ).mean()

                    min_pos_index = smoothed.idxmin()
                    min_val = smoothed.loc[min_pos_index]
                    is_min_at_boundary = (min_pos_index == smoothed.index[0]) or (
                        min_pos_index == smoothed.index[-1]
                    )

                    if not is_min_at_boundary:
                        series_range = smoothed.max() - smoothed.min()
                        tolerance = series_range * 0.001 if series_range > 1e-9 else 0
                        start_val = smoothed.iloc[0]
                        end_val = smoothed.iloc[-1]

                        if (
                            start_val > min_val + tolerance
                            and end_val > min_val + tolerance
                        ):
                            rises = True

                info = job_params.copy()
                info["variable"] = var_name
                info["rises"] = rises
                return info, rises

            batch_size = 100
            total_jobs = len(job_ids)

            for i in range(0, total_jobs, batch_size):
                batch_ids = job_ids[i : i + batch_size]
                min_id = min(batch_ids)
                max_id = max(batch_ids)

                try:
                    res_batch = store.select(
                        RESULTS_KEY,
                        where=f"job_id >= {min_id} & job_id <= {max_id}",
                    )
                except Exception as e:
                    logger.warning(f"Failed to load batch {min_id}-{max_id}: {e}")
                    continue

                grouped = res_batch.groupby("job_id")

                for j_id, group in grouped:
                    if j_id not in jobs_map:
                        continue

                    params = jobs_map[j_id]

                    for col in group.columns:
                        if col in ["time", "job_id"]:
                            continue

                        info, rises = check_curve(
                            group[col].reset_index(drop=True), params, col
                        )
                        all_curves_info.append(info)

                        if not rises:
                            error_count += 1
                            logger.error(
                                f"Feature not detected for Job {j_id}, Var '{col}' (Params: {params})"
                            )

    except Exception as e:
        logger.error(f"HDF5 processing failed: {e}", exc_info=True)

    # Generate a report file with all information unconditionally
    output_filename = kwargs.get("output_filename", "rise_report.json")
    report_path = os.path.join(output_dir, output_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_curves_info, f, indent=4, ensure_ascii=False)

    if error_count > 0:
        logger.info(f"{error_count} curves failed checks. Report: {report_path}")
    else:
        logger.info(f"All curves passed. Report: {report_path}")
