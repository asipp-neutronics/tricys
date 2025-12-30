import json
import os

import pandas as pd


def load_h5_data(file_path):
    """
    Loads data (jobs, results, config, log) from an HDF5 file.
    """
    try:
        # Check if file exists to avoid pandas cryptic errors
        if not os.path.exists(file_path):
            return [], [], [], [], None, None

        # Load Jobs Metadata
        try:
            jobs_df = pd.read_hdf(file_path, "jobs")
        except KeyError:
            # Handle case where 'jobs' key might be missing or different
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
                except Exception as e:
                    print(f"Error loading config: {e}")

            # Try loading log
            if "/log" in store.keys():
                try:
                    l_data = json.loads(store.select("log").iloc[0, 0])
                except Exception as e:
                    print(f"Error loading logs: {e}")

        # Process Options
        v_opts = [col for col in results_cols if col not in ["time", "job_id"]]
        p_opts = [col for col in jobs_df.columns if col != "job_id"]

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
    except Exception as e:
        print(f"Error loading HDF5 file {file_path}: {e}")
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
    except Exception as e:
        print(f"Error loading results subset: {e}")
        return []


def load_baseline_data(h5_path, job_id):
    """
    Loads all data for a single baseline job.
    """
    try:
        return pd.read_hdf(h5_path, "results", where=f"job_id == {job_id}")
    except Exception as e:
        print(f"Error loading baseline job {job_id}: {e}")
        return None
