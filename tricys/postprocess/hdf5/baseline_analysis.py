import logging
import os

import pandas as pd

from tricys.postprocess import baseline_analysis as legacy_analysis

logger = logging.getLogger(__name__)


def baseline_analysis(results_file_path: str, output_dir: str, **kwargs) -> None:
    """
    HDF5-compatible wrapper for baseline_analysis.

    Reads data from the HDF5 file, reconstructs the 'wide' DataFrame format
    (columns = var&param=val) expected by the legacy analysis, and then
    calls the original baseline_analysis function.

    Args:
        results_file_path: Path to the .h5 results file.
        output_dir: Directory for output.
        **kwargs: Passed to legacy baseline_analysis.
    """
    if not os.path.exists(results_file_path):
        logger.error(f"HDF5 file not found: {results_file_path}")
        return

    logger.info(f"Loading data from HDF5 for baseline analysis: {results_file_path}")

    try:
        # 1. Load Parameters (Jobs)
        jobs_df = pd.read_hdf(results_file_path, "jobs")

        # 2. Check scale
        num_jobs = len(jobs_df)
        if num_jobs > 100:
            logger.warning(
                f"Baseline analysis triggered for {num_jobs} jobs. "
                "This may consume significant memory as it reconstructs a wide DataFrame. "
                "Consider using specific statistical analysis modules for large sweeps."
            )

        # 3. Load Results
        # For baseline analysis, we typically need all data.
        # If specific variables were requested, we could filter columns here if we knew them,
        # but baseline_analysis usually plots *everything*.
        results_store = pd.HDFStore(results_file_path, mode="r")
        results_df_long = results_store.select("results")  # Read all results
        results_store.close()

        # 4. Reconstruct Wide DataFrame
        # Format: time, var1&p=v, var2&p=v ...

        # Optimize reconstruction using groupby
        grouped = results_df_long.groupby("job_id")

        all_dfs = []
        time_added = False

        # Iterate through jobs in order
        for job_id in jobs_df["job_id"]:
            if job_id not in grouped.groups:
                continue

            job_data = grouped.get_group(job_id).copy()
            job_params = jobs_df[jobs_df["job_id"] == job_id].iloc[0].to_dict()
            # Remove job_id from params
            job_params.pop("job_id", None)

            # Prepare param string
            # Replicate logic: param_string = "&".join([f"{k}={v}" for k, v in job_params.items()])
            # Need to sort keys to match simulation.py behavior if possible, though not strictly required for uniqueness
            param_string = "&".join([f"{k}={v}" for k, v in sorted(job_params.items())])

            # Handle Time
            if not time_added:
                all_dfs.append(job_data[["time"]].reset_index(drop=True))
                time_added = True

            # Drop time and job_id from data columns
            data_cols = job_data.drop(columns=["time", "job_id"], errors="ignore")

            # Rename columns
            rename_map = {
                col: f"{col}&{param_string}" if param_string else col
                for col in data_cols.columns
            }
            data_cols.rename(columns=rename_map, inplace=True)
            data_cols.reset_index(drop=True, inplace=True)

            all_dfs.append(data_cols)

        if not all_dfs:
            logger.warning("No data found to reconstruct for baseline analysis.")
            return

        # Concatenate
        logger.info("Reconstructing wide DataFrame...")
        wide_df = pd.concat(all_dfs, axis=1)

        # 5. Delegate to Legacy Analysis
        logger.info("Delegating to legacy baseline_analysis logic.")
        legacy_analysis.baseline_analysis(wide_df, output_dir, **kwargs)

    except Exception as e:
        logger.error(f"Failed to run HDF5 baseline analysis: {e}", exc_info=True)
