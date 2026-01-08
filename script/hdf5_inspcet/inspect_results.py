import argparse
import json
import os

import pandas as pd


def inspect_h5(file_path, job_id=None):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    try:
        with pd.HDFStore(file_path, mode="r") as store:
            print(f"=== HDF5 Inspect: {os.path.basename(file_path)} ===")
            keys = store.keys()

            if "/config" in keys:
                print("\n--- Config info ---")
                try:
                    config_json = store.select("config").iloc[0, 0]
                    config_data = json.loads(config_json)
                    print(json.dumps(config_data, indent=2))
                except Exception as e:
                    print(f"Error reading config: {e}")

            if "/log" in keys:
                print("\n--- Log info ---")
                try:
                    log_json = store.select("log").iloc[0, 0]
                    log_data = json.loads(log_json)

                    if isinstance(log_data, list):
                        print(f"Total log entries: {len(log_data)}")
                        error_count = sum(
                            1
                            for entry in log_data
                            if isinstance(entry, dict)
                            and entry.get("levelname") == "ERROR"
                        )
                        warning_count = sum(
                            1
                            for entry in log_data
                            if isinstance(entry, dict)
                            and entry.get("levelname") == "WARNING"
                        )
                        print(f"Errors: {error_count}, Warnings: {warning_count}")

                        print("\nFirst 5 log entries:")
                        for entry in log_data[:5]:
                            if isinstance(entry, dict):
                                print(
                                    f"[{entry.get('asctime', '')}] {entry.get('levelname', '')}: {entry.get('message', '')}"
                                )
                            else:
                                print(entry)
                    else:
                        print("Log data is not a list structure.")
                except Exception as e:
                    print(f"Error reading logs: {e}")

            if "/jobs" in keys:
                print("\n--- Jobs info ---")
                if job_id is not None:
                    # Query specific job parameters
                    try:
                        jobs_df = store.select("jobs", where=f"job_id == {job_id}")
                        if jobs_df.empty:
                            print(f"Job {job_id} not found in 'jobs' table.")
                        else:
                            print(f"Parameters for Job {job_id}:")
                            print(jobs_df.transpose().to_string())
                    except Exception as e:
                        print(f"Error querying job {job_id}: {e}")
                else:
                    try:
                        nrows = store.get_storer("jobs").nrows
                        print(f"Total jobs: {nrows}")
                        print("First 5 jobs:")
                        print(store.select("jobs", stop=5).to_string())
                    except:
                        pass

            if "/results" in keys:
                print("\n--- Results info ---")
                if job_id is not None:
                    # Query specific job results
                    try:
                        results_df = store.select(
                            "results", where=f"job_id == {job_id}"
                        )
                        if results_df.empty:
                            print(f"No results found for Job {job_id}.")
                        else:
                            print(f"Results for Job {job_id} ({len(results_df)} rows):")
                            print(results_df.head(10).to_string())
                            if len(results_df) > 10:
                                print(f"... and {len(results_df)-10} more rows")
                    except Exception as e:
                        print(f"Error querying results for job {job_id}: {e}")
                else:
                    try:
                        nrows = store.get_storer("results").nrows
                        print(f"Total result rows: {nrows}")
                        print("First 5 rows:")
                        print(store.select("results", stop=5).to_string())
                    except:
                        pass

            if not keys:
                print("File is empty.")

    except Exception as e:
        print(f"Failed to open HDF5 file: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect HDF5 simulation results.")
    parser.add_argument("file_path", help="Path to the .h5 file")
    parser.add_argument(
        "-j", "--job", type=int, help="Specific Job ID to inspect", default=None
    )

    args = parser.parse_args()
    inspect_h5(args.file_path, args.job)
