"""Core module for running single or parameter sweep simulations.

This module provides the main functionality for executing Modelica simulations
based on a JSON configuration file. It supports running a single simulation or
a parameter sweep using a thread pool for parallel execution.
"""

import argparse
import concurrent.futures
import itertools
import json
import logging
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from OMPython import ModelicaSystem

from tricys.utils.decorators_utils import record_time
from tricys.utils.file_utils import delete_old_logs, get_unique_filename
from tricys.utils.om_utils import (
    format_parameter_value,
    get_om_session,
    load_modelica_package,
)


# Standard logger setup
logger = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(config: Dict[str, Any]):
    """Configures the logging module based on the application configuration.

    Sets up logging to both console and file, with log rotation based on the
    settings provided in the configuration dictionary.

    Args:
        config (Dict[str, Any]): The application configuration dictionary,
            expected to contain a 'logging' section.
    """
    log_config = config.get("logging", {})
    log_level_str = log_config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_to_console = log_config.get("log_to_console", True)

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

    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if a log directory is specified
    if log_dir_path:
        abs_log_dir = os.path.abspath(log_dir_path)
        os.makedirs(abs_log_dir, exist_ok=True)
        delete_old_logs(abs_log_dir, log_count)
        global timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(abs_log_dir, f"simulation_{timestamp}.log")

        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file_path}")


def _parse_parameter_value(value: Any) -> List[Any]:
    """Parses a parameter value which can be a single value, a list, or a range string.

    Args:
        value (Any): The parameter value to parse.

    Returns:
        List[Any]: A list of discrete values.
    """
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
    """Generates a list of simulation jobs from parameters, handling sweeps.

    Takes a dictionary of simulation parameters and expands any parameter with multiple
    values (from a list or range string) into a Cartesian product of all possible
    parameter combinations.

    Args:
        simulation_params (Dict[str, Any]): The dictionary of parameters from the config.

    Returns:
        List[Dict[str, Any]]: A list of jobs, where each job is a dictionary
            representing a single simulation run with a unique parameter combination.
    """
    sweep_params = {}
    single_value_params = {}
    for name, value in simulation_params.items():
        parsed_values = _parse_parameter_value(value)
        if len(parsed_values) > 1:
            sweep_params[name] = parsed_values
        else:
            single_value_params[name] = parsed_values[0]
    if not sweep_params:
        return [single_value_params]
    sweep_names = list(sweep_params.keys())
    sweep_values = list(sweep_params.values())
    jobs = []
    for combo in itertools.product(*sweep_values):
        job = single_value_params.copy()
        job.update(dict(zip(sweep_names, combo)))
        jobs.append(job)
    return jobs


def _run_single_job(
    job_info: tuple,
    model_name: str,
    package_path: str,
    variable_filter: str,
    stop_time: float,
    step_size: float,
    output_dir: str,
) -> str:
    """Executes a single simulation job in a thread-safe manner.

    This function is designed to be called by a thread pool executor. It initializes
    its own OpenModelica session to ensure thread safety.

    Args:
        job_info (tuple): A tuple containing the job ID and the parameter dictionary.
        model_name (str): The name of the Modelica model to simulate.
        package_path (str): The path to the Modelica package file.
        variable_filter (str): The regex filter for result variables.
        stop_time (float): The simulation stop time.
        step_size (float): The simulation step size.
        output_dir (str): The directory to save the result file.

    Returns:
        str: The path to the simulation result CSV file, or an empty string if failed.
    """
    job_id, job_params = job_info
    logger.info(f"Starting job {job_id} with parameters: {job_params}")
    omc = None
    mod = None
    try:
        omc = get_om_session()
        # Ensure all paths passed to OMPython are in POSIX format
        if not load_modelica_package(omc, Path(package_path).as_posix()):
            raise RuntimeError(f"Job {job_id}: Failed to load Modelica package.")

        mod = ModelicaSystem(
            fileName=Path(package_path).as_posix(),
            modelName=model_name,
            variableFilter=variable_filter,
        )
        param_settings = [
            format_parameter_value(name, value) for name, value in job_params.items()
        ]
        mod.setSimulationOptions(
            [
                f"stopTime={stop_time}",
                "tolerance=1e-6",
                "outputFormat=csv",
                f"stepSize={step_size}",
            ]
        )
        if param_settings:
            mod.setParameters(param_settings)
        mod.buildModel()
        result_filename = f"{timestamp}_simulation_results_{job_id}.csv"
        result_file_path = os.path.join(output_dir, result_filename)
        mod.simulate(resultfile=Path(result_file_path).as_posix())
        logger.info(f"Job {job_id} finished. Results saved to {result_file_path}")
        return result_file_path
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        return ""
    finally:
        if mod is not None:
            del mod
        if omc is not None:
            omc.sendExpression("quit()")


def _run_sequential_sweep(
    jobs: List[Dict[str, Any]],
    model_name: str,
    package_path: str,
    variable_filter: str,
    stop_time: float,
    step_size: float,
    output_dir: str,
) -> List[str]:
    """Executes a parameter sweep sequentially in a single process.

    This function reuses the OpenModelica session and model object for
    efficiency, which is ideal for non-concurrent sweeps.

    Args:
        jobs (List[Dict[str, Any]]): A list of jobs, where each job is a
            dictionary of parameters.
        model_name (str): The name of the Modelica model to simulate.
        package_path (str): The path to the Modelica package file.
        variable_filter (str): The regex filter for result variables.
        stop_time (float): The simulation stop time.
        step_size (float): The simulation step size.
        output_dir (str): The directory to save result files.

    Returns:
        List[str]: A list of paths to the simulation result CSV files.
    """
    logger.info("Running sweep sequentially, reusing OM session and model object.")
    omc = None
    mod = None
    result_paths = []
    try:
        omc = get_om_session()
        if not load_modelica_package(omc, Path(package_path).as_posix()):
            raise RuntimeError("Failed to load Modelica package for sequential sweep.")

        mod = ModelicaSystem(
            fileName=Path(package_path).as_posix(),
            modelName=model_name,
            variableFilter=variable_filter,
        )
        mod.setSimulationOptions(
            [
                f"stopTime={stop_time}",
                "tolerance=1e-6",
                "outputFormat=csv",
                f"stepSize={step_size}",
            ]
        )
        # Build the model once before the loop, as the model structure doesn't change.
        mod.buildModel()

        for i, job_params in enumerate(jobs):
            try:
                logger.info(f"Running sequential job {i} with parameters: {job_params}")
                param_settings = [
                    format_parameter_value(name, value)
                    for name, value in job_params.items()
                ]
                if param_settings:
                    mod.setParameters(param_settings)

                result_filename = f"{timestamp}_simulation_results_{i}.csv"
                result_file_path = os.path.join(output_dir, result_filename)
                mod.simulate(resultfile=Path(result_file_path).as_posix())

                logger.info(
                    f"Sequential job {i} finished. Results saved to {result_file_path}"
                )
                result_paths.append(result_file_path)
            except Exception as e:
                logger.error(f"Sequential job {i} failed: {e}", exc_info=True)
                result_paths.append("")  # Keep list length consistent

        return result_paths
    except Exception as e:
        logger.error(f"Sequential sweep failed during setup: {e}", exc_info=True)
        # Return a list of empty strings with the correct length
        return [""] * len(jobs)
    finally:
        if mod is not None:
            del mod
        if omc is not None:
            omc.sendExpression("quit()")

@record_time
def run_simulation(
    config: Dict[str, Any],
    package_path: str,
    results_dir: str,
    temp_dir: str,
):
    """Runs single or sweep simulations based on the provided configuration.

    This function orchestrates the simulation process. It generates jobs, runs them
    (in parallel for sweeps), and combines the results.

    Args:
        config (Dict[str, Any]): The full application configuration dictionary.
        package_path (str): The absolute path to the Modelica package.
        results_dir (str): The absolute path to the directory for final results.
        temp_dir (str): The absolute path to the directory for intermediate files.
    """
    sim_config = config["simulation"]
    model_name = sim_config["model_name"]
    stop_time = sim_config["stop_time"]
    step_size = sim_config["step_size"]
    variable_filter = sim_config["variableFilter"]
    max_workers = sim_config.get("max_workers", os.cpu_count())
    concurrent_execution = sim_config.get("concurrent", True)
    keep_temp_files = sim_config.get("keep_temp_files", False)
    simulation_params = config.get("simulation_parameters", {})
    jobs = _generate_simulation_jobs(simulation_params)
    num_jobs = len(jobs)
    is_sweep = num_jobs > 1

    logger.info(f"Starting simulation run: {num_jobs} job(s) to execute.")
    if is_sweep:
        if concurrent_execution:
            logger.info(
                f"Parameter sweep mode enabled. Using up to {max_workers} parallel workers."
            )
        else:
            logger.info("Parameter sweep mode enabled. Running jobs sequentially.")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    if not is_sweep:
        job_info = (0, jobs[0])
        output_path = _run_single_job(
            job_info,
            model_name,
            package_path,
            variable_filter,
            stop_time,
            step_size,
            results_dir,
        )
        if output_path:
            final_path = os.path.join(results_dir, "simulation_results.csv")
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(output_path, final_path)
            logger.info(f"Single simulation finished. Result at {final_path}")
    else:
        if concurrent_execution:
            run_job_partial = partial(
                _run_single_job,
                model_name=model_name,
                package_path=package_path,
                variable_filter=variable_filter,
                stop_time=stop_time,
                step_size=step_size,
                output_dir=temp_dir,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                simulation_results_paths = list(
                    executor.map(run_job_partial, enumerate(jobs))
                )
        else:
            simulation_results_paths = _run_sequential_sweep(
                jobs=jobs,
                model_name=model_name,
                package_path=package_path,
                variable_filter=variable_filter,
                stop_time=stop_time,
                step_size=step_size,
                output_dir=temp_dir,
            )

        logger.info("All sweep jobs completed. Combining results.")
        combined_df = None
        rises_info = []  # Initialize list to store rise info
        for i, result_path in enumerate(simulation_results_paths):
            if not result_path:
                logger.warning(f"Job {i} produced no result file. Skipping.")
                continue
            job_params = jobs[i]
            df = pd.read_csv(result_path)
            if combined_df is None:
                combined_df = df[["time"]].copy()
            col_name = "_".join([f"{k}={v}" for k, v in job_params.items()])
            if len(df.columns) > 1:
                combined_df[col_name] = df.iloc[:, 1]
                # Check for turning point and upward trend
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
            rises_csv_path = get_unique_filename(results_dir, "rises_info.csv")
            rises_df.to_csv(rises_csv_path, index=False)
            logger.info(f"Rise information saved to: {rises_csv_path}")

        if combined_df is not None:
            combined_csv_path = get_unique_filename(results_dir, "sweep_results.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            logger.info(f"Combined sweep results saved to: {combined_csv_path}")
        if not keep_temp_files:
            for p in simulation_results_paths:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError as e:
                        logger.warning(f"Could not remove intermediate file {p}: {e}")
        logger.info("Cleaned up intermediate sweep files.")
    logger.info("Simulation run finished.")


def main():
    """Main function to run the simulation from the command line.

    Parses command-line arguments for the configuration file, loads it, sets up
    logging, and starts the simulation run.
    """
    parser = argparse.ArgumentParser(
        description="Run a Tricys simulation from a configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.json",
        help="Path to the JSON configuration file. Defaults to 'config.json' in the current directory.",
    )
    args = parser.parse_args()

    try:
        config_path = os.path.abspath(args.config)
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        # Setup a basic logger to show the critical error and exit
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logger.error(
            f"Error decoding JSON from the configuration file: {config_path} - {e}"
        )
        sys.exit(1)

    # Setup logging based on the loaded configuration
    setup_logging(config)

    logger.info(f"Loading configuration from: {config_path}")

    try:
        package_path = os.path.abspath(config["paths"]["package_path"])
        results_dir = os.path.abspath(config["paths"]["results_dir"])
        temp_dir = os.path.abspath(config["paths"]["temp_dir"])
    except KeyError as e:
        logger.error(f"Missing required path key in configuration file: {e}")
        sys.exit(1)

    try:
        run_simulation(
            config=config,
            package_path=package_path,
            results_dir=results_dir,
            temp_dir=temp_dir,
        )
        logger.info("Main execution completed successfully.")
    except Exception as e:
        logger.error(f"Main execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
