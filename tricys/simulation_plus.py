import argparse
import concurrent.futures
import importlib
import itertools
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from OMPython import ModelicaSystem

from tricys.utils.file_utils import delete_old_logs, get_unique_filename
from tricys.utils.om_utils import (
    format_parameter_value,
    get_om_session,
    integrate_interceptor_model,
    load_modelica_package,
)

# Standard logger setup
logger = logging.getLogger(__name__)


def _parse_parameter_value(value: Any) -> List[Any]:
    """Parses a parameter value which can be a single value, a list, or a range string."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and ":" in value:
        try:
            start, stop, step = map(float, value.split(":"))
            return np.arange(start, stop + step / 2, step).tolist()
        except ValueError:
            logger.error(f"Invalid range format for parameter value: {value}")
            return [value]
    return [value]


def _generate_simulation_jobs(
    simulation_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generates a list of simulation jobs from parameters, handling sweeps."""
    sweep_params = {}
    single_value_params = {}
    for name, value in simulation_params.items():
        parsed_values = _parse_parameter_value(value)
        if len(parsed_values) > 1:
            sweep_params[name] = parsed_values
        else:
            single_value_params[name] = parsed_values[0]
    if not sweep_params:
        return [single_value_params] if single_value_params else [{}]
    sweep_names = list(sweep_params.keys())
    sweep_values = list(sweep_params.values())
    jobs = []
    for combo in itertools.product(*sweep_values):
        job = single_value_params.copy()
        job.update(dict(zip(sweep_names, combo)))
        jobs.append(job)
    return jobs


def setup_logging(config: Dict[str, Any]):
    """Configures the logging module based on the application configuration."""
    log_config = config.get("logging", {})
    log_level_str = log_config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_to_console = log_config.get("log_to_console", True)
    run_timestamp = config.get("run_timestamp")

    log_dir_path = log_config.get("log_dir")
    log_count = log_config.get("log_count", 5)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers to prevent duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_dir_path:
        abs_log_dir = os.path.abspath(log_dir_path)
        os.makedirs(abs_log_dir, exist_ok=True)
        delete_old_logs(abs_log_dir, log_count)
        log_file_path = os.path.join(abs_log_dir, f"simulation_{run_timestamp}.log")

        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file_path}")


def run_co_simulation_workflow(config: dict, job_params: dict, job_id: int = 0) -> str:
    """
    Runs the full co-simulation workflow in an isolated directory to ensure thread safety.
    """
    paths_config = config["paths"]
    sim_config = config["simulation"]
    run_timestamp = config["run_timestamp"]  # Get the timestamp from config

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    run_temp_dir = os.path.join(
        base_temp_dir, run_timestamp
    )  # Create timestamped subdirectory
    job_workspace = os.path.join(run_temp_dir, f"job_{job_id}")
    os.makedirs(job_workspace, exist_ok=True)

    omc = None

    try:
        original_package_path = os.path.abspath(paths_config["package_path"])
        original_package_dir = os.path.dirname(original_package_path)
        package_dir_name = os.path.basename(original_package_dir)

        isolated_package_dir = os.path.join(job_workspace, package_dir_name)

        if os.path.exists(isolated_package_dir):
            shutil.rmtree(isolated_package_dir)
        shutil.copytree(original_package_dir, isolated_package_dir)

        isolated_package_path = os.path.join(
            isolated_package_dir, os.path.basename(original_package_path)
        )
        isolated_temp_dir = job_workspace
        results_dir = os.path.abspath(paths_config["results_dir"])
        os.makedirs(results_dir, exist_ok=True)

        co_sim_configs = config["co_simulation"]
        if not isinstance(co_sim_configs, list):
            co_sim_configs = [co_sim_configs]

        model_name = sim_config["model_name"]
        stop_time = sim_config["stop_time"]
        step_size = sim_config["step_size"]

        omc = get_om_session()
        if not load_modelica_package(omc, Path(isolated_package_path).as_posix()):
            raise RuntimeError(
                f"Failed to load Modelica package at {isolated_package_path}"
            )

        # Handle copying of any additional asset directories specified with a '_path' suffix
        for co_sim_config in co_sim_configs:
            if "params" in co_sim_config:
                # Iterate over a copy of items since we are modifying the dict
                for param_key, param_value in list(co_sim_config["params"].items()):
                    if isinstance(param_value, str) and param_key.endswith("_path"):
                        original_asset_path_str = param_value

                        # Paths in config are relative to project root. We need the absolute path.
                        original_asset_path = Path(
                            os.path.abspath(original_asset_path_str)
                        )
                        original_asset_dir = original_asset_path.parent

                        if not original_asset_dir.exists():
                            logger.warning(
                                f"Asset directory '{original_asset_dir}' for parameter '{param_key}' not found. Skipping copy."
                            )
                            continue

                        asset_dir_name = original_asset_dir.name
                        dest_dir = Path(job_workspace) / asset_dir_name

                        # Copy the directory only if it hasn't been copied already
                        if not dest_dir.exists():
                            shutil.copytree(original_asset_dir, dest_dir)
                            logger.info(
                                f"Copied asset directory '{original_asset_dir}' to '{dest_dir}' for job {job_id}"
                            )

                        # Update the path in the config to point to the new location
                        new_asset_path = dest_dir / original_asset_path.name
                        co_sim_config["params"][param_key] = new_asset_path.as_posix()
                        logger.info(
                            f"Updated parameter '{param_key}' for job {job_id} to '{co_sim_config['params'][param_key]}'"
                        )

        all_input_vars = []
        for co_sim_config in co_sim_configs:
            submodel_name = co_sim_config["submodel_name"]
            instance_name = co_sim_config["instance_name"]
            logger.info(f"Identifying input ports for submodel '{submodel_name}'...")
            components = omc.sendExpression(f"getComponents({submodel_name})")
            input_ports = [
                {"name": c[1], "dim": int(c[11][0]) if c[11] else 1}
                for c in components
                if c[0] == "Modelica.Blocks.Interfaces.RealInput"
            ]
            if not input_ports:
                logger.warning(f"No RealInput ports found in {submodel_name}.")
                continue

            logger.info(
                f"Found input ports for {instance_name}: {[p['name'] for p in input_ports]}"
            )
            for port in input_ports:
                full_name = f"{instance_name}.{port['name']}".replace(".", "\\.")
                if port["dim"] > 1:
                    full_name += f"\\[[1-{port['dim']}]\\]"
                all_input_vars.append(full_name)

        variable_filter = "time|" + "|".join(all_input_vars)

        mod = ModelicaSystem(
            fileName=Path(isolated_package_path).as_posix(),
            modelName=model_name,
            variableFilter=variable_filter,
        )
        mod.setSimulationOptions(
            [f"stopTime={stop_time}", f"stepSize={step_size}", "outputFormat=csv"]
        )

        param_settings = [
            format_parameter_value(name, value) for name, value in job_params.items()
        ]
        if param_settings:
            logger.info(f"Applying parameters for job {job_id}: {param_settings}")
            mod.setParameters(param_settings)

        primary_result_filename = get_unique_filename(
            isolated_temp_dir, "primary_inputs.csv"
        )
        mod.simulate(resultfile=Path(primary_result_filename).as_posix())

        interception_configs = []
        for co_sim_config in co_sim_configs:
            handler_module = importlib.import_module(co_sim_config["handler_module"])
            handler_function = getattr(
                handler_module, co_sim_config["handler_function"]
            )
            instance_name = co_sim_config["instance_name"]

            co_sim_output_filename = get_unique_filename(
                isolated_temp_dir, f"{instance_name}_outputs.csv"
            )

            output_placeholder = handler_function(
                temp_input_csv=primary_result_filename,
                temp_output_csv=co_sim_output_filename,
                **co_sim_config.get("params", {}),
            )

            interception_configs.append(
                {
                    "submodel_name": co_sim_config["submodel_name"],
                    "instance_name": co_sim_config["instance_name"],
                    "csv_uri": Path(os.path.abspath(co_sim_output_filename)).as_posix(),
                    "output_placeholder": output_placeholder,
                }
            )

        intercepted_model_paths = integrate_interceptor_model(
            package_path=isolated_package_path,
            model_name=model_name,
            interception_configs=interception_configs,
        )

        verif_config = config["simulation"]["variableFilter"]
        logger.info("Proceeding with Final simulation.")

        for model_path in intercepted_model_paths["interceptor_model_paths"]:
            omc.sendExpression(f"""loadFile("{Path(model_path).as_posix()}")""")
        omc.sendExpression(
            f"""loadFile("{Path(intercepted_model_paths["system_model_path"]).as_posix()}")"""
        )

        package_name, original_system_name = model_name.split(".")
        intercepted_model_full_name = (
            f"{package_name}.{original_system_name}_Intercepted"
        )

        verif_mod = ModelicaSystem(
            fileName=Path(isolated_package_path).as_posix(),
            modelName=intercepted_model_full_name,
            variableFilter=verif_config,
        )
        verif_mod.setSimulationOptions(
            [f"stopTime={stop_time}", f"stepSize={step_size}", "outputFormat=csv"]
        )
        if param_settings:
            verif_mod.setParameters(param_settings)

        default_result_path = get_unique_filename(
            job_workspace, "co_simulation_results.csv"
        )
        verif_mod.simulate(resultfile=Path(default_result_path).as_posix())

        if not os.path.exists(default_result_path):
            raise FileNotFoundError(
                f"Simulation for job {job_id} failed to produce a result file at {default_result_path}"
            )

        # Return the path to the result file inside the temporary workspace
        return Path(default_result_path).as_posix()
    except Exception as e:
        logger.error(f"Workflow for job {job_id} failed: {e}", exc_info=True)
        return ""
    finally:
        if omc:
            omc.sendExpression("quit()")
            logger.info(f"Closed OMPython session for job {job_id}.")

        if not sim_config.get("keep_temp_files", False):
            if os.path.exists(job_workspace):
                shutil.rmtree(job_workspace)
                logger.info(f"Cleaned up workspace for job {job_id}: {job_workspace}")


def run_simulation(config: Dict[str, Any]):
    """Orchestrates the simulation execution, result handling, and cleanup."""
    run_timestamp = config["run_timestamp"]
    jobs = _generate_simulation_jobs(config.get("simulation_parameters", {}))

    try:
        results_dir = os.path.abspath(config["paths"]["results_dir"])
    except KeyError as e:
        logger.error(f"Missing required path key in configuration file: {e}")
        sys.exit(1)

    simulation_results = {}
    use_concurrent = config["simulation"].get("concurrent", True)

    if use_concurrent:
        logger.info("Starting co-simulation in CONCURRENT mode.")
        max_workers = config["simulation"].get("max_workers", 4)
        logger.info(f"Using up to {max_workers} parallel processes.")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_to_job = {
                executor.submit(
                    run_co_simulation_workflow, config, job_params, job_id=i + 1
                ): job_params
                for i, job_params in enumerate(jobs)
            }

            for future in concurrent.futures.as_completed(future_to_job):
                job_params = future_to_job[future]
                try:
                    result_path = future.result()
                    if result_path:
                        job_key = tuple(sorted(job_params.items()))
                        simulation_results[job_key] = result_path
                        logger.info(
                            f"Successfully finished job for params: {job_params}"
                        )
                    else:
                        logger.warning(
                            f"Job for params {job_params} did not return a result path."
                        )
                except Exception as exc:
                    logger.error(
                        f"Job for params {job_params} generated an exception: {exc}",
                        exc_info=True,
                    )
    else:
        logger.info("Starting co-simulation in SEQUENTIAL mode.")
        for i, job_params in enumerate(jobs):
            job_id = i + 1
            logger.info(f"--- Starting Sequential Job {job_id}/{len(jobs)} ---")
            try:
                result_path = run_co_simulation_workflow(
                    config, job_params, job_id=job_id
                )
                if result_path:
                    job_key = tuple(sorted(job_params.items()))
                    simulation_results[job_key] = result_path
                    logger.info(f"Successfully finished job for params: {job_params}")
                else:
                    logger.warning(
                        f"Job for params {job_params} did not return a result path."
                    )
            except Exception as exc:
                logger.error(
                    f"Job for params {job_params} generated an exception: {exc}",
                    exc_info=True,
                )
            logger.info(f"--- Finished Sequential Job {job_id}/{len(jobs)} ---")

    # --- Result Handling ---
    # The simulation_results dictionary now contains paths to results inside temporary job workspaces.
    # Create a timestamped directory for this run's final results.
    run_results_dir = os.path.join(results_dir, run_timestamp)
    os.makedirs(run_results_dir, exist_ok=True)

    # Case 1: Single job run
    if len(jobs) == 1:
        logger.info("Single job finished. Copying result to final destination.")
        if simulation_results:
            temp_result_path = list(simulation_results.values())[0]
            if temp_result_path and os.path.exists(temp_result_path):
                final_path = get_unique_filename(
                    run_results_dir, "co_simulation_result.csv"
                )
                shutil.copy2(temp_result_path, final_path)  # copy2 preserves metadata
                logger.info(f"Result copied to {final_path}")
            else:
                logger.warning("Single job did not produce a valid result file.")

    # Case 2: Multiple jobs (sweep)
    elif len(jobs) > 1:
        logger.info("All sweep jobs completed. Combining results.")
        combined_df = None
        rises_info = []

        for job_params in jobs:
            job_key = tuple(sorted(job_params.items()))
            result_path = simulation_results.get(job_key)

            if not result_path or not os.path.exists(result_path):
                logger.warning(f"Job {job_params} produced no result file. Skipping.")
                continue

            df = pd.read_csv(result_path)
            if combined_df is None:
                combined_df = df[["time"]].copy()

            col_name = "_".join([f"{k}={v}" for k, v in job_params.items()])
            if len(df.columns) > 1:
                combined_df[col_name] = df.iloc[:, 1]
                data = combined_df[col_name].to_numpy()
                if len(data) > 2:
                    diffs = np.diff(data)
                    mid_index = len(diffs) // 2
                    has_dip = np.any(diffs[:mid_index] < 0)
                    has_rise = np.any(diffs[mid_index:] > 0)
                    rises = has_dip and has_rise
                else:
                    rises = False

                info = job_params.copy()
                info["rises"] = rises
                rises_info.append(info)

        if rises_info:
            rises_df = pd.DataFrame(rises_info)
            rises_csv_path = get_unique_filename(run_results_dir, "rises_info.csv")
            rises_df.to_csv(rises_csv_path, index=False)
            logger.info(f"Rise information saved to: {rises_csv_path}")

        if combined_df is not None:
            combined_csv_path = get_unique_filename(
                run_results_dir, "sweep_results.csv"
            )
            combined_df.to_csv(combined_csv_path, index=False)
            logger.info(f"Combined sweep results saved to: {combined_csv_path}")

    # --- Final Cleanup ---
    # The primary cleanup of job workspaces is handled by the `finally` block in `run_co_simulation_workflow`.
    # This is an additional safeguard.
    if not config["simulation"].get("keep_temp_files", True):
        logger.info("Cleaning up temporary directory...")
        temp_dir_path = os.path.abspath(config["paths"].get("temp_dir", "temp"))
        if os.path.exists(temp_dir_path):
            try:
                shutil.rmtree(temp_dir_path)
                os.makedirs(temp_dir_path)  # Recreate for next run
            except OSError as e:
                logger.error(f"Error cleaning up temp directory: {e}")


def initialize_run() -> Dict[str, Any]:
    """
    Parses command-line arguments, loads the config file, and generates a run timestamp.
    Returns a fully prepared configuration dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Run a unified simulation and co-simulation workflow in parallel."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        default="config.json",
        help="Path to the JSON configuration file.",
    )
    args = parser.parse_args()

    try:
        config_path = os.path.abspath(args.config)
        with open(config_path, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Logger is not set up yet, so print directly to stderr
        print(
            f"ERROR: Failed to load or parse config file {args.config}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Generate a single timestamp for the entire run and add it to the config
    config["run_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config


def main():
    """Main function to run the simulation from the command line."""
    # 1. Initialize and get the prepared configuration
    config = initialize_run()

    # 2. Set up logging using the prepared config
    setup_logging(config)

    # 3. Run the main simulation workflow
    run_simulation(config)

    logger.info("Script finished.")
