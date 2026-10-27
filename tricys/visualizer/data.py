import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from tricys.utils.hdf5_schema import load_jobs_df, load_summary_df

logger = logging.getLogger(__name__)


def load_h5_overview(file_path):
    overview = {
        "file_size_bytes": None,
        "modified_at": None,
        "has_results": False,
        "has_summary": False,
        "has_config": False,
        "has_log": False,
        "time_start": None,
        "time_end": None,
    }
    if not file_path or not os.path.exists(file_path):
        return overview

    file_ref = Path(file_path)
    stat_info = file_ref.stat()
    overview["file_size_bytes"] = stat_info.st_size
    overview["modified_at"] = stat_info.st_mtime

    try:
        with pd.HDFStore(file_path, mode="r") as store:
            keys = set(store.keys())
            overview["has_results"] = "/results" in keys
            overview["has_summary"] = "/summary" in keys or (
                "/jobs" in keys
                and any(
                    str(c).startswith("Required_")
                    or "_for_" in str(c)
                    or c in ("Required_TBR", "sds.I_S_0", "I_S_0")
                    for c in store.select("jobs", stop=0).columns
                )
            )
            overview["has_config"] = "/config" in keys
            overview["has_log"] = "/log" in keys

            if overview["has_results"]:
                storer = store.get_storer("results")
                row_count = getattr(storer, "nrows", 0) or 0
                if row_count > 0:
                    start_df = store.select(
                        "results", start=0, stop=1, columns=["time"]
                    )
                    end_df = store.select(
                        "results",
                        start=max(row_count - 1, 0),
                        stop=row_count,
                        columns=["time"],
                    )
                    if not start_df.empty:
                        overview["time_start"] = start_df.iloc[0]["time"]
                    if not end_df.empty:
                        overview["time_end"] = end_df.iloc[-1]["time"]
    except Exception:
        logger.exception(
            "Failed to inspect HDF5 overview", extra={"file_path": file_path}
        )

    return overview


def load_h5_data(file_path):
    """
    Loads data (jobs, results, config, log) from an HDF5 file.
    """
    try:
        # Check if file exists to avoid pandas cryptic errors
        if not os.path.exists(file_path):
            return [], [], [], [], None, None

        try:
            jobs_df = load_jobs_df(file_path)
        except KeyError:
            return [], [], [], [], None, None

        c_data = None
        l_data = None
        results_cols = []

        with pd.HDFStore(file_path, mode="r") as store:
            if "/results" in store.keys():
                # Get columns without loading entire dataset
                results_cols = store.select("results", start=0, stop=0).columns

            # Try loading config
            if "/config" in store.keys():
                try:
                    c_data = json.loads(store.select("config").iloc[0, 0])
                except Exception:
                    logger.exception(
                        "Failed to load config payload", extra={"file_path": file_path}
                    )

            # Try loading log
            if "/log" in store.keys():
                try:
                    l_data = json.loads(store.select("log").iloc[0, 0])
                except Exception:
                    logger.exception(
                        "Failed to load log payload", extra={"file_path": file_path}
                    )

        # Process Options: strictly filter out metadata identifiers and non-numerical columns
        v_opts = [col for col in results_cols if col not in ["time", "job_id"]]
        ignored_p_names = {
            "id",
            "job_id",
            "name",
            "job_name",
            "case_name",
            "job",
            "title",
            "description",
            "comment",
            "status",
        }
        # Identify known metrics to ensure they are not listed as parameters
        metric_names = set()
        if c_data:
            t_m = (
                c_data.get("task", {}).get("metrics", {})
                if isinstance(c_data.get("task"), dict)
                else {}
            )
            metric_names |= set(t_m.keys())
            metric_names |= set(c_data.get("metrics_definition", {}).keys())

        p_opts = []
        for col in jobs_df.columns:
            if str(col).strip().lower() in ignored_p_names or col in metric_names:
                continue
            if pd.to_numeric(jobs_df[col], errors="coerce").notna().any():
                p_opts.append(col)

        jobs_table = jobs_df.copy()
        if "job_id" in jobs_table.columns:
            jobs_table.rename(columns={"job_id": "id"}, inplace=True)
        if "id" in jobs_table.columns:
            cols = ["id"] + [col for col in jobs_table.columns if col != "id"]
            jobs_table = jobs_table[cols]  # Reorder to put id first

        t_cols = [
            {"name": "job_id" if col == "id" else col, "id": col}
            for col in jobs_table.columns
        ]

        return v_opts, p_opts, t_cols, jobs_table.to_dict("records"), c_data, l_data
    except Exception:
        logger.exception("Failed to load HDF5 file", extra={"file_path": file_path})
        return [], [], [], [], None, None


def load_results_subset(h5_path, job_ids, variables):
    """
    Loads a subset of results for specific jobs and variables.
    """
    if not h5_path or not job_ids or not variables:
        return []
    try:
        # Use 'where' clause for efficient partial loading
        # Note: 'where' clause in read_hdf supports 'in' operator for checking list membership
        # but syntax must be careful.
        # Ideally: "job_id in [1, 2, 3]"
        # However, passing a list variable directly is safer if supported or formatted carefully.

        # Format job_ids for query string
        job_ids_list = list(map(int, job_ids))  # Ensure ints
        where_clause = f"job_id in {job_ids_list}"

        cols = list(set(["time", "job_id"] + variables))
        df = pd.read_hdf(h5_path, "results", where=where_clause, columns=cols)
        return df.to_dict("records")
    except Exception:
        logger.exception(
            "Failed to load results subset",
            extra={
                "file_path": h5_path,
                "job_ids": list(job_ids),
                "variables": list(variables),
            },
        )
        return []


def load_baseline_data(h5_path, job_id):
    """
    Loads all data for a single baseline job.
    """
    try:
        return pd.read_hdf(h5_path, "results", where=f"job_id == {job_id}")
    except Exception:
        logger.exception(
            "Failed to load baseline data",
            extra={"file_path": h5_path, "job_id": job_id},
        )
        return None


@lru_cache(maxsize=32)
def _load_summary_data_cached(h5_path, job_ids_key):
    """
    Cached loader for summary metrics to reduce repeated disk IO.
    job_ids_key is a tuple of ints (hashable) or None.
    """
    if not h5_path or not os.path.exists(h5_path):
        return []

    try:
        where_clause = None
        if job_ids_key:
            where_clause = f"job_id in {list(job_ids_key)}"

        with pd.HDFStore(h5_path, mode="r") as store:
            if "/summary" in store.keys():
                df = load_summary_df(h5_path, where=where_clause)
                return df.to_dict("records")

            # Fallback: extract optimization metrics directly from jobs table if available
            if "/jobs" in store.keys():
                jobs_df = store.select("jobs")
                opt_cols = [
                    c
                    for c in jobs_df.columns
                    if str(c).startswith("Required_")
                    or "_for_" in str(c)
                    or c in ("Required_TBR", "sds.I_S_0", "I_S_0")
                ]
                if opt_cols:
                    if "job_id" not in jobs_df.columns and "id" in jobs_df.columns:
                        jobs_df["job_id"] = jobs_df["id"]
                    if "job_id" in jobs_df.columns:
                        cols_to_keep = ["job_id"] + [
                            c for c in opt_cols if c != "job_id"
                        ]
                        sub_df = jobs_df[
                            [c for c in cols_to_keep if c in jobs_df.columns]
                        ].copy()
                        if job_ids_key:
                            sub_df = sub_df[sub_df["job_id"].isin(list(job_ids_key))]
                        return sub_df.to_dict("records")

            return []

    except Exception:
        logger.exception(
            "Failed to load summary data",
            extra={"file_path": h5_path, "job_ids": list(job_ids_key or [])},
        )
        return []


def load_summary_data(h5_path, job_ids=None):
    """
    Loads the summary metrics table.
    Expects Wide Format: job_id, MetricA, MetricB...
    """
    if not h5_path or not os.path.exists(h5_path):
        return []

    job_ids_key = None
    if job_ids:
        try:
            job_ids_key = tuple(int(j) for j in job_ids)
        except Exception:
            job_ids_key = None

    return _load_summary_data_cached(h5_path, job_ids_key)
