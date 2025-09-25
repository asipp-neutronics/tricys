import argparse
import concurrent.futures
import importlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from OMPython import ModelicaSystem

from tricys.utils.file_utils import get_unique_filename
from tricys.utils.log_utils import setup_logging
from tricys.utils.om_utils import (
    format_parameter_value,
    get_om_session,
    integrate_interceptor_model,
    load_modelica_package,
)
from tricys.utils.sim_utils import generate_simulation_jobs

# Standard logger setup
logger = logging.getLogger(__name__)


def _run_co_simulation(config: dict, job_params: dict, job_id: int = 0) -> str:
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


def _run_single_job(config: dict, job_params: dict, job_id: int = 0) -> str:
    """Executes a single simulation job in an isolated workspace."""
    paths_config = config["paths"]
    sim_config = config["simulation"]
    run_timestamp = config["run_timestamp"]

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    run_temp_dir = os.path.join(base_temp_dir, run_timestamp)
    job_workspace = os.path.join(run_temp_dir, f"job_{job_id}")
    os.makedirs(job_workspace, exist_ok=True)

    logger.info(f"Starting job {job_id} with parameters: {job_params}")
    omc = None
    try:
        omc = get_om_session()
        package_path = os.path.abspath(paths_config["package_path"])
        if not load_modelica_package(omc, Path(package_path).as_posix()):
            raise RuntimeError(f"Job {job_id}: Failed to load Modelica package.")

        mod = ModelicaSystem(
            fileName=Path(package_path).as_posix(),
            modelName=sim_config["model_name"],
            variableFilter=sim_config["variableFilter"],
        )
        mod.setSimulationOptions(
            [
                f"stopTime={sim_config['stop_time']}",
                "tolerance=1e-6",
                "outputFormat=csv",
                f"stepSize={sim_config['step_size']}",
            ]
        )
        param_settings = [
            format_parameter_value(name, value) for name, value in job_params.items()
        ]
        if param_settings:
            mod.setParameters(param_settings)

        default_result_file = f"job_{job_id}_simulation_results.csv"
        result_path = Path(job_workspace) / default_result_file

        mod.simulate(resultfile=Path(result_path).as_posix())

        if not result_path.is_file():
            raise FileNotFoundError(
                f"Simulation for job {job_id} failed to produce result file at {result_path}"
            )

        logger.info(f"Job {job_id} finished. Results at {result_path}")
        return str(result_path)
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        return ""
    finally:
        if omc:
            omc.sendExpression("quit()")


def _run_sequential_sweep(config: dict, jobs: List[Dict[str, Any]]) -> List[str]:
    """
    Executes a parameter sweep sequentially, reusing the OM session for efficiency.
    Saves intermediate results to the timestamped temporary directory.
    """
    paths_config = config["paths"]
    sim_config = config["simulation"]
    run_timestamp = config["run_timestamp"]

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    run_temp_dir = os.path.join(base_temp_dir, run_timestamp)
    os.makedirs(run_temp_dir, exist_ok=True)

    logger.info(
        f"Running sweep sequentially. Intermediate files will be in: {run_temp_dir}"
    )

    omc = None
    result_paths = []
    try:
        omc = get_om_session()
        package_path = os.path.abspath(paths_config["package_path"])
        if not load_modelica_package(omc, Path(package_path).as_posix()):
            raise RuntimeError("Failed to load Modelica package for sequential sweep.")

        mod = ModelicaSystem(
            fileName=Path(package_path).as_posix(),
            modelName=sim_config["model_name"],
            variableFilter=sim_config["variableFilter"],
        )
        mod.setSimulationOptions(
            [
                f"stopTime={sim_config['stop_time']}",
                "tolerance=1e-6",
                "outputFormat=csv",
                f"stepSize={sim_config['step_size']}",
            ]
        )
        mod.buildModel()

        for i, job_params in enumerate(jobs):
            try:
                logger.info(
                    f"Running sequential job {i+1}/{len(jobs)} with parameters: {job_params}"
                )
                param_settings = [
                    format_parameter_value(name, value)
                    for name, value in job_params.items()
                ]
                if param_settings:
                    mod.setParameters(param_settings)

                job_workspace = os.path.join(run_temp_dir, f"job_{i+1}")
                os.makedirs(job_workspace, exist_ok=True)
                result_filename = f"job_{i+1}_simulation_results.csv"
                result_file_path = os.path.join(job_workspace, result_filename)

                mod.simulate(resultfile=Path(result_file_path).as_posix())

                logger.info(
                    f"Sequential job {i+1} finished. Results at {result_file_path}"
                )
                result_paths.append(result_file_path)
            except Exception as e:
                logger.error(f"Sequential job {i+1} failed: {e}", exc_info=True)
                result_paths.append("")

        return result_paths
    except Exception as e:
        logger.error(f"Sequential sweep failed during setup: {e}", exc_info=True)
        return [""] * len(jobs)
    finally:
        if omc:
            omc.sendExpression("quit()")


def _run_post_processing(
    config: Dict[str, Any], results_df: pd.DataFrame, run_results_dir: str
):
    """
    Dynamically load and run post-processing modules based on configuration.
    """
    post_processing_configs = config.get("post_processing")
    if not post_processing_configs:
        logger.info("No post-processing task configured, skipping this step.")
        return

    logger.info("--- Start post-processing phase ---")

    post_processing_dir = os.path.join(run_results_dir, "post_processing")
    os.makedirs(post_processing_dir, exist_ok=True)
    logger.info(f"The post-processing report will be saved to:{post_processing_dir}")

    for i, task_config in enumerate(post_processing_configs):
        try:
            module_name = task_config["module"]
            function_name = task_config["function"]
            params = task_config.get("params", {})
            logger.info(
                f"Run post-processing tasks #{i+1}: {module_name}.{function_name}"
            )

            module = importlib.import_module(module_name)
            post_processing_func = getattr(module, function_name)

            post_processing_func(
                results_df=results_df, output_dir=post_processing_dir, **params
            )
        except Exception as e:
            logger.error(f"Post-processing task #{i+1} failed: {e}", exc_info=True)
    logger.info("--- The post-processing stage has ended ---")


def run_simulation(config: Dict[str, Any]):
    """Orchestrates the simulation execution, result handling, and cleanup."""
    run_timestamp = config["run_timestamp"]
    jobs = generate_simulation_jobs(config.get("simulation_parameters", {}))

    try:
        results_dir = os.path.abspath(config["paths"]["results_dir"])
    except KeyError as e:
        logger.error(f"Missing required path key in configuration file: {e}")
        sys.exit(1)

    simulation_results = {}
    use_concurrent = config["simulation"].get("concurrent", True)

    try:
        if config.get("co_simulation") is None:
            if use_concurrent:
                logger.info("Starting simulation in CONCURRENT mode.")
                max_workers = config["simulation"].get("max_workers", os.cpu_count())
                logger.info(f"Using up to {max_workers} parallel workers.")
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    future_to_job = {
                        executor.submit(
                            _run_single_job, config, job_params, i + 1
                        ): job_params
                        for i, job_params in enumerate(jobs)
                    }
                    for future in concurrent.futures.as_completed(future_to_job):
                        job_params = future_to_job[future]
                        try:
                            result_path = future.result()
                            if result_path:
                                simulation_results[
                                    tuple(sorted(job_params.items()))
                                ] = result_path
                        except Exception as exc:
                            logger.error(
                                f"Job for {job_params} generated an exception: {exc}",
                                exc_info=True,
                            )
            else:
                logger.info("Starting simulation in SEQUENTIAL mode.")
                result_paths = _run_sequential_sweep(config, jobs)
                for i, result_path in enumerate(result_paths):
                    if result_path:
                        simulation_results[tuple(sorted(jobs[i].items()))] = result_path
        else:
            if use_concurrent:
                logger.info("Starting co-simulation in CONCURRENT mode.")
                max_workers = config["simulation"].get("max_workers", 4)
                logger.info(f"Using up to {max_workers} parallel processes.")

                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    future_to_job = {
                        executor.submit(
                            _run_co_simulation, config, job_params, job_id=i + 1
                        ): job_params
                        for i, job_params in enumerate(jobs)
                    }

                    for future in concurrent.futures.as_completed(future_to_job):
                        job_params = future_to_job[future]
                        try:
                            result_path = future.result()
                            if result_path:
                                simulation_results[
                                    tuple(sorted(job_params.items()))
                                ] = result_path
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
                        result_path = _run_co_simulation(
                            config, job_params, job_id=job_id
                        )
                        if result_path:
                            simulation_results[tuple(sorted(job_params.items()))] = (
                                result_path
                            )
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
                    logger.info(f"--- Finished Sequential Job {job_id}/{len(jobs)} ---")
    except Exception as e:
        raise RuntimeError(f"Failed to run simualtion: {e}")

    # --- Result Handling ---
    # The simulation_results dictionary now contains paths to results inside temporary job workspaces.
    # Create a timestamped directory for this run's final results.
    run_results_dir = os.path.join(results_dir, run_timestamp)
    os.makedirs(run_results_dir, exist_ok=True)

    # Unified result processing for both single and multiple jobs
    logger.info(f"Processing {len(jobs)} job(s). Combining results.")
    combined_df = None

    all_dfs = []
    time_df_added = False

    for job_params in jobs:
        job_key = tuple(sorted(job_params.items()))
        result_path = simulation_results.get(job_key)

        if not result_path or not os.path.exists(result_path):
            logger.warning(f"Job {job_params} produced no result file. Skipping.")
            continue

        # Read the current job's result file
        df = pd.read_csv(result_path)

        # From the very first valid DataFrame, grab the 'time' column
        if not time_df_added and "time" in df.columns:
            all_dfs.append(df[["time"]])
            time_df_added = True

        # Prepare the parameter string for column renaming
        param_string = "&".join([f"{k}={v}" for k, v in job_params.items()])

        # Isolate the data columns (everything except 'time')
        data_columns = df.drop(columns=["time"], errors="ignore")

        # Create a dictionary to map old column names to new ones
        # e.g., {'voltage': 'voltage&param1=A&param2=B'}
        rename_mapping = {col: f"{col}&{param_string}" for col in data_columns.columns}

        # Rename the columns and add the resulting DataFrame to our list
        all_dfs.append(data_columns.rename(columns=rename_mapping))

    # Concatenate all the DataFrames in the list along the columns axis (axis=1)
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=1)
    else:
        combined_df = pd.DataFrame()  # Or None, as you had before

    if combined_df is not None and not combined_df.empty:
        if len(jobs) == 1:
            # For single job, save as simulation_result.csv
            combined_csv_path = get_unique_filename(
                run_results_dir, "simulation_result.csv"
            )
        else:
            # For multiple jobs, save as sweep_results.csv
            combined_csv_path = get_unique_filename(
                run_results_dir, "sweep_results.csv"
            )

        combined_df.to_csv(combined_csv_path, index=False)
        logger.info(f"Combined results saved to: {combined_csv_path}")
    else:
        logger.warning("No valid results found to combine.")

    # --- Post-Processing ---
    if combined_df is not None:
        _run_post_processing(config, combined_df, run_results_dir)
    else:
        logger.warning(
            "No simulation results were generated, skipping post-processing."
        )

    # --- Final Cleanup ---
    # The primary cleanup of job workspaces is handled by the `finally` block in `_run_co_simulation`.
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


def _convert_relative_paths_to_absolute(
    config: Dict[str, Any], base_dir: str
) -> Dict[str, Any]:
    """
    Recursively traverse configuration data and convert relative paths to absolute paths based on the specified base directory

    Args:
        config: Configuration dictionary
        base_dir: Base directory path

    Returns:
        Converted configuration dictionary
    """

    def _process_value(value, key_name="", parent_dict=None):
        if isinstance(value, dict):
            return {k: _process_value(v, k, value) for k, v in value.items()}
        elif isinstance(value, list):
            return [_process_value(item, parent_dict=parent_dict) for item in value]
        elif isinstance(value, str):
            # Check if it's a path-related key name (extended support for more path fields)
            path_keys = [
                "package_path",
                "db_path",
                "results_dir",
                "temp_dir",
                "log_dir",
            ]

            if key_name.endswith("_path") or key_name in path_keys:
                # If it's a relative path, convert to absolute path
                if not os.path.isabs(value):
                    abs_path = os.path.abspath(os.path.join(base_dir, value))
                    logger.debug(
                        f"Converted path: {key_name} '{value}' -> '{abs_path}'"
                    )
                    return abs_path
            return value
        else:
            return value

    return _process_value(config)


def initialize_run() -> Dict[str, Any]:
    """
    Parses command-line arguments, loads the config file, and generates a run timestamp.
    Returns a fully prepared configuration dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Run a unified simulation and co-simulation workflow in parallel."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON configuration file.",
    )

    subparsers.add_parser("example", help="Run simulation examples interactively")

    args = parser.parse_args()

    if args.command == "example":
        import importlib.util

        script_path = (
            Path(__file__).parent.parent
            / "script"
            / "example_runner"
            / "tricys_runner.py"
        )
        spec = importlib.util.spec_from_file_location("tricys_runner", script_path)
        tricys_runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tricys_runner)
        tricys_runner.main()
        sys.exit(0)

    if not args.config:
        parser.error("the following arguments are required: -c/--config")

    try:
        config_path = os.path.abspath(args.config)
        with open(config_path, "r") as f:
            base_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Logger is not set up yet, so print directly to stderr
        print(
            f"ERROR: Failed to load or parse config file {args.config}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    original_config_dir = (
        os.getcwd()
    )  # Directory where the original configuration file is located

    # First convert relative paths to absolute paths
    absolute_config = _convert_relative_paths_to_absolute(
        base_config, original_config_dir
    )
    # Deep copy the converted configuration
    config = json.loads(json.dumps(absolute_config))
    # Generate a single timestamp for the entire run and add it to the config
    config["run_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config


def main():
    """Main function to run the simulation from the command line."""
    config = initialize_run()
    setup_logging(config)
    logger.info(f"Loading configuration from: {os.path.abspath(sys.argv[-1])}")
    try:
        run_simulation(config)
        logger.info("Main execution completed successfully.")
    except Exception as e:
        logger.error(f"Main execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
