import argparse
import importlib
import json
import logging
import os
import sys
import itertools
import shutil
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


def run_co_simulation_workflow(config: dict, job_params: dict, job_id: int = 0) -> str:
    """
    Runs the full co-simulation workflow for a single set of parameters,
    handling multiple submodel interceptions.
    """
    omc = None
    result_paths = []

    try:
        paths_config = config["paths"]
        sim_config = config["simulation"]
        co_sim_configs = config["co_simulation"]
        if not isinstance(co_sim_configs, list):
            co_sim_configs = [co_sim_configs]

        package_path = os.path.abspath(paths_config["package_path"])
        results_dir = os.path.abspath(paths_config["results_dir"])
        temp_dir = os.path.abspath(paths_config["temp_dir"])
        model_name = sim_config["model_name"]
        stop_time = sim_config["stop_time"]
        step_size = sim_config["step_size"]

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        omc = get_om_session()
        if not load_modelica_package(omc, Path(package_path).as_posix()):
            raise RuntimeError(f"Failed to load Modelica package at {package_path}")

        # --- Stage 1: Primary Simulation (for ALL submodels) ---
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
        logger.info(
            f"Using combined variable filter for primary sim: {variable_filter}"
        )

        mod = ModelicaSystem(
            fileName=Path(package_path).as_posix(),
            modelName=model_name,
            variableFilter=variable_filter,
        )
        mod.setSimulationOptions(
            [f"stopTime={stop_time}", f"stepSize={step_size}", "outputFormat=csv"]
        )

        param_settings = [format_parameter_value(name, value) for name, value in job_params.items()]
        if param_settings:
            logger.info(f"Applying parameters for job {job_id}: {param_settings}")
            mod.setParameters(param_settings)

        primary_result_filename = get_unique_filename(temp_dir, f"primary_inputs.csv")
        mod.simulate(resultfile=Path(primary_result_filename).as_posix())
        logger.info(
            f"OM simulation finished. Input data for all co-sims at {primary_result_filename}"
        )

        # --- Stage 2: Dynamic Co-simulation (Loop) ---
        interception_configs = []
        for co_sim_config in co_sim_configs:
            handler_module = importlib.import_module(co_sim_config["handler_module"])
            handler_function = getattr(
                handler_module, co_sim_config["handler_function"]
            )
            instance_name = co_sim_config["instance_name"]

            co_sim_output_filename = get_unique_filename(
                temp_dir, f"{instance_name}_outputs.csv"
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

        # --- Stage 3: Generate Multi-Interceptor Model ---

        intercepted_model_paths = integrate_interceptor_model(
            package_path=package_path,
            model_name=model_name,
            interception_configs=interception_configs,
        )

        # --- Stage 4: Final Simulation ---
        verif_config = config["simulation"]["variableFilter"]
        logger.info("Proceeding with Final simulation.")

        for model_path in intercepted_model_paths["interceptor_model_paths"]:
            omc.sendExpression(f'loadFile("{Path(model_path).as_posix()}")')
        omc.sendExpression(
            f'loadFile("{Path(intercepted_model_paths["system_model_path"]).as_posix()}")'
        )

        package_name, original_system_name = model_name.split(".")
        intercepted_model_full_name = (
            f"{package_name}.{original_system_name}_Intercepted"
        )

        verif_mod = ModelicaSystem(
            fileName=Path(package_path).as_posix(),
            modelName=intercepted_model_full_name,
            variableFilter=verif_config,
        )
        verif_mod.setSimulationOptions(
            [f"stopTime={stop_time}", f"stepSize={step_size}", "outputFormat=csv"]
        )
        if param_settings:
            verif_mod.setParameters(param_settings)

        verif_result_filename = get_unique_filename(
            results_dir, f"co_simulation_results.csv"
        )
        verif_mod.simulate(resultfile=Path(verif_result_filename).as_posix())

        return Path(verif_result_filename).as_posix()
    except Exception as e:
        logger.info(f"Workflow  failed: {e}", exc_info=True)
    finally:
        if omc:
            omc.sendExpression("quit()")
            logger.info("Closed OMPython session .")


def main():
    """Main function to run the simulation from the command line.

    Parses command-line arguments for the configuration file, loads it, sets up
    logging, and starts the simulation run.
    """
    parser = argparse.ArgumentParser(
        description="Run a unified simulation and co-simulation workflow."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file.",
    )
    args = parser.parse_args()

    try:
        config_path = os.path.abspath(args.config)
        with open(config_path, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.basicConfig()
        logger.error(f"Failed to load or parse config file {args.config}: {e}")
        sys.exit(1)

    setup_logging(config)

    jobs = _generate_simulation_jobs(config.get("simulation_parameters", {}))
    
    simulation_results_paths = []
    for i, job_params in enumerate(jobs):
        logger.info(f"--- Starting Co-simulation Job {i+1}/{len(jobs)} ---")
        result_path = run_co_simulation_workflow(config, job_params, job_id=i + 1)
        simulation_results_paths.append(result_path)
        logger.info(f"--- Finished Co-simulation Job {i+1}/{len(jobs)} ---")
    
    if len(jobs) > 1:
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
            rises_csv_path = get_unique_filename(config["paths"]['results_dir'], "rises_info.csv")
            rises_df.to_csv(rises_csv_path, index=False)
            logger.info(f"Rise information saved to: {rises_csv_path}")

        if combined_df is not None:
            combined_csv_path = get_unique_filename(config["paths"]['results_dir'], "sweep_results.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            logger.info(f"Combined sweep results saved to: {combined_csv_path}")

        if not config["simulation"].get("keep_temp_files", False):
            shutil.rmtree(config["paths"].get("temp_dir", "temp"), ignore_errors=True)
            os.makedirs(config["paths"].get("temp_dir", "temp"), exist_ok=True)
            for p in simulation_results_paths:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError as e:
                        logger.warning(f"Could not remove intermediate file {p}: {e}")
        else:
            temp_dir = os.path.abspath(config["paths"].get("temp_dir", "temp"))
            for p in simulation_results_paths:
               if p and os.path.exists(p):
                   try:
                    shutil.move(p, temp_dir)
                   except OSError as e:
                       logger.warning(f"Could not move intermediate file {p}: {e}")


        logger.info("Cleaned up intermediate sweep files.")
if __name__ == "__main__":
    main()
