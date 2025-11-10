import argparse
import concurrent.futures
import importlib
import json
import logging
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from OMPython import ModelicaSystem

from tricys.core.interceptor import integrate_interceptor_model
from tricys.core.jobs import generate_simulation_jobs
from tricys.core.modelica import (
    format_parameter_value,
    get_om_session,
    load_modelica_package,
)
from tricys.utils.file_utils import get_unique_filename
from tricys.utils.log_utils import log_execution_time, setup_logging

# Standard logger setup
logger = logging.getLogger(__name__)


def _run_co_simulation(config: dict, job_params: dict, job_id: int = 0) -> str:
    """
    Runs the full co-simulation workflow in an isolated directory to ensure thread safety.
    """
    paths_config = config["paths"]
    sim_config = config["simulation"]

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    # The temp_dir from the config is now the self-contained workspace's temp folder.
    job_workspace = os.path.join(base_temp_dir, f"job_{job_id}")
    os.makedirs(job_workspace, exist_ok=True)

    omc = None

    try:
        original_package_path = os.path.abspath(paths_config["package_path"])

        # Determine if it's a single-file or multi-file package and copy accordingly.
        if os.path.isfile(original_package_path) and not original_package_path.endswith(
            "package.mo"
        ):
            # SINGLE-FILE: Copy the single .mo file into the root of the job_workspace.
            isolated_package_path = os.path.join(
                job_workspace, os.path.basename(original_package_path)
            )
            shutil.copy(original_package_path, isolated_package_path)
            logger.info(
                "Copied single-file package",
                extra={
                    "job_id": job_id,
                    "source_path": original_package_path,
                    "destination_path": isolated_package_path,
                },
            )
        else:
            # MULTI-FILE: Copy the entire package directory.
            # This handles both a directory path and a path to a package.mo file.
            if os.path.isfile(original_package_path):
                original_package_dir = os.path.dirname(original_package_path)
            else:  # It's a directory
                original_package_dir = original_package_path

            package_dir_name = os.path.basename(original_package_dir)
            isolated_package_dir = os.path.join(job_workspace, package_dir_name)

            if os.path.exists(isolated_package_dir):
                shutil.rmtree(isolated_package_dir)
            shutil.copytree(original_package_dir, isolated_package_dir)

            # Reconstruct the path to the main package file inside the new isolated directory
            if os.path.isfile(original_package_path):
                isolated_package_path = os.path.join(
                    isolated_package_dir, os.path.basename(original_package_path)
                )
            else:  # path was a directory, so we assume package.mo
                isolated_package_path = os.path.join(isolated_package_dir, "package.mo")

            logger.info(
                "Copied multi-file package",
                extra={
                    "job_id": job_id,
                    "source_dir": original_package_dir,
                    "destination_dir": isolated_package_dir,
                },
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
                                "Copied asset directory",
                                extra={
                                    "job_id": job_id,
                                    "source_dir": original_asset_dir,
                                    "destination_dir": dest_dir,
                                },
                            )

                        # Update the path in the config to point to the new location
                        new_asset_path = dest_dir / original_asset_path.name
                        co_sim_config["params"][param_key] = new_asset_path.as_posix()
                        logger.info(
                            "Updated asset parameter path",
                            extra={
                                "job_id": job_id,
                                "parameter_key": param_key,
                                "new_path": co_sim_config["params"][param_key],
                            },
                        )

        all_input_vars = []
        for co_sim_config in co_sim_configs:
            submodel_name = co_sim_config["submodel_name"]
            instance_name = co_sim_config["instance_name"]
            logger.info(
                "Identifying input ports for submodel",
                extra={
                    "job_id": job_id,
                    "submodel_name": submodel_name,
                },
            )
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
                "Found input ports for instance",
                extra={
                    "job_id": job_id,
                    "instance_name": instance_name,
                    "input_ports": [p["name"] for p in input_ports],
                },
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
            logger.info(
                "Applying parameters for job",
                extra={
                    "job_id": job_id,
                    "param_settings": param_settings,
                },
            )
            mod.setParameters(param_settings)

        primary_result_filename = get_unique_filename(
            isolated_temp_dir, "primary_inputs.csv"
        )
        mod.simulate(resultfile=Path(primary_result_filename).as_posix())

        # Clean up the simulation result file
        if os.path.exists(primary_result_filename):
            try:
                df = pd.read_csv(primary_result_filename)
                df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                df.dropna(subset=["time"], inplace=True)
                df.to_csv(primary_result_filename, index=False)
            except Exception as e:
                logger.warning(
                    "Failed to clean primary result file",
                    extra={
                        "job_id": job_id,
                        "file_path": primary_result_filename,
                        "error": str(e),
                    },
                )

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
        logger.info("Proceeding with Final simulation.", extra={"job_id": job_id})

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

        # Clean up the simulation result file
        if os.path.exists(default_result_path):
            try:
                df = pd.read_csv(default_result_path)
                df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                df.dropna(subset=["time"], inplace=True)
                df.to_csv(default_result_path, index=False)
            except Exception as e:
                logger.warning(
                    "Failed to clean final co-simulation result file",
                    extra={
                        "job_id": job_id,
                        "file_path": default_result_path,
                        "error": str(e),
                    },
                )

        if not os.path.exists(default_result_path):
            raise FileNotFoundError(
                f"Simulation for job {job_id} failed to produce a result file at {default_result_path}"
            )

        # Return the path to the result file inside the temporary workspace
        return Path(default_result_path).as_posix()
    except Exception:
        logger.error(
            "Co-simulation workflow failed", exc_info=True, extra={"job_id": job_id}
        )
        return ""
    finally:
        if omc:
            omc.sendExpression("quit()")
            logger.info("Closed OMPython session", extra={"job_id": job_id})

        if not sim_config.get("keep_temp_files", False):
            if os.path.exists(job_workspace):
                shutil.rmtree(job_workspace)
                logger.info(
                    "Cleaned up job workspace",
                    extra={"job_id": job_id, "workspace": job_workspace},
                )


def _run_single_job(config: dict, job_params: dict, job_id: int = 0) -> str:
    """Executes a single simulation job in an isolated workspace."""
    paths_config = config["paths"]
    sim_config = config["simulation"]

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    # The temp_dir from the config is now the self-contained workspace's temp folder.
    job_workspace = os.path.join(base_temp_dir, f"job_{job_id}")
    os.makedirs(job_workspace, exist_ok=True)

    logger.info(
        "Starting single job",
        extra={"job_id": job_id, "job_params": job_params},
    )
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

        # Clean up the simulation result file
        if os.path.exists(result_path):
            try:
                df = pd.read_csv(result_path)
                df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                df.dropna(subset=["time"], inplace=True)
                df.to_csv(result_path, index=False)
            except Exception as e:
                logger.warning(
                    "Failed to clean result file",
                    extra={
                        "job_id": job_id,
                        "file_path": result_path,
                        "error": str(e),
                    },
                )

        if not result_path.is_file():
            raise FileNotFoundError(
                f"Simulation for job {job_id} failed to produce result file at {result_path}"
            )

        logger.info(
            "Job finished successfully",
            extra={"job_id": job_id, "result_path": str(result_path)},
        )
        return str(result_path)
    except Exception:
        logger.error("Job failed", exc_info=True, extra={"job_id": job_id})
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

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    # The temp_dir is now the self-contained workspace's temp folder.
    os.makedirs(base_temp_dir, exist_ok=True)

    logger.info(
        "Running sequential sweep",
        extra={
            "mode": "sequential",
            "intermediate_files_dir": base_temp_dir,
        },
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
        # mod.buildModel()

        for i, job_params in enumerate(jobs):
            try:
                logger.info(
                    "Running sequential job",
                    extra={
                        "job_index": f"{i+1}/{len(jobs)}",
                        "job_params": job_params,
                    },
                )
                param_settings = [
                    format_parameter_value(name, value)
                    for name, value in job_params.items()
                ]
                if param_settings:
                    mod.setParameters(param_settings)

                job_workspace = os.path.join(base_temp_dir, f"job_{i+1}")
                os.makedirs(job_workspace, exist_ok=True)
                result_filename = f"job_{i+1}_simulation_results.csv"
                result_file_path = os.path.join(job_workspace, result_filename)

                mod.simulate(resultfile=Path(result_file_path).as_posix())

                # Clean up the simulation result file
                if os.path.exists(result_file_path):
                    try:
                        df = pd.read_csv(result_file_path)
                        df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                        df.dropna(subset=["time"], inplace=True)
                        df.to_csv(result_file_path, index=False)
                    except Exception as e:
                        logger.warning(
                            "Failed to clean result file",
                            extra={
                                "job_index": i + 1,
                                "file_path": result_file_path,
                                "error": str(e),
                            },
                        )

                logger.info(
                    "Sequential job finished successfully",
                    extra={
                        "job_index": i + 1,
                        "result_path": result_file_path,
                    },
                )
                result_paths.append(result_file_path)
            except Exception:
                logger.error(
                    "Sequential job failed", exc_info=True, extra={"job_index": i + 1}
                )
                result_paths.append("")

        return result_paths
    except Exception:
        logger.error("Sequential sweep setup failed", exc_info=True)
        return [""] * len(jobs)
    finally:
        if omc:
            omc.sendExpression("quit()")


def _run_post_processing(
    config: Dict[str, Any], results_df: pd.DataFrame, post_processing_output_dir: str
):
    """
    Dynamically load and run post-processing modules based on configuration.
    """
    post_processing_configs = config.get("post_processing")
    if not post_processing_configs:
        logger.info("No post-processing task configured, skipping this step.")
        return

    logger.info("Starting post-processing phase")

    post_processing_dir = post_processing_output_dir
    os.makedirs(post_processing_dir, exist_ok=True)
    logger.info(
        "Post-processing report will be saved",
        extra={"output_dir": post_processing_dir},
    )

    for i, task_config in enumerate(post_processing_configs):
        try:
            module_name = task_config["module"]
            function_name = task_config["function"]
            params = task_config.get("params", {})
            logger.info(
                "Running post-processing task",
                extra={
                    "task_index": i + 1,
                    "task_module": module_name,
                    "function": function_name,
                },
            )

            module = importlib.import_module(module_name)
            post_processing_func = getattr(module, function_name)

            post_processing_func(
                results_df=results_df, output_dir=post_processing_dir, **params
            )
        except Exception:
            logger.error(
                "Post-processing task failed",
                exc_info=True,
                extra={"task_index": i + 1},
            )
    logger.info("Post-processing phase ended")


@log_execution_time
def run_simulation(config: Dict[str, Any]):
    """Orchestrates the simulation execution, result handling, and cleanup."""
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
                max_workers = config["simulation"].get("max_workers", os.cpu_count())
                logger.info(
                    "Starting simulation",
                    extra={
                        "mode": "CONCURRENT",
                        "max_workers": max_workers,
                    },
                )
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
                logger.info("Starting simulation", extra={"mode": "SEQUENTIAL"})
                result_paths = _run_sequential_sweep(config, jobs)
                for i, result_path in enumerate(result_paths):
                    if result_path:
                        simulation_results[tuple(sorted(jobs[i].items()))] = result_path
        else:
            if use_concurrent:
                logger.info(
                    "Starting co-simulation",
                    extra={
                        "mode": "CONCURRENT",
                        "max_workers": max_workers,
                    },
                )

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
                                    "Successfully finished co-simulation job",
                                    extra={
                                        "job_params": job_params,
                                    },
                                )
                            else:
                                logger.warning(
                                    "Co-simulation job did not return a result path",
                                    extra={
                                        "job_params": job_params,
                                    },
                                )
                        except Exception as exc:
                            logger.error(
                                "Co-simulation job generated an exception",
                                exc_info=True,
                                extra={
                                    "job_params": job_params,
                                    "exception": str(exc),
                                },
                            )
            else:
                logger.info("Starting co-simulation", extra={"mode": "SEQUENTIAL"})
                for i, job_params in enumerate(jobs):
                    job_id = i + 1
                    logger.info(
                        "Starting Sequential Co-simulation Job",
                        extra={
                            "job_index": f"{job_id}/{len(jobs)}",
                        },
                    )
                    try:
                        result_path = _run_co_simulation(
                            config, job_params, job_id=job_id
                        )
                        if result_path:
                            simulation_results[tuple(sorted(job_params.items()))] = (
                                result_path
                            )
                            logger.info(
                                "Successfully finished co-simulation job",
                                extra={
                                    "job_params": job_params,
                                },
                            )
                        else:
                            logger.warning(
                                "Co-simulation job did not return a result path",
                                extra={
                                    "job_params": job_params,
                                },
                            )
                    except Exception as exc:
                        logger.error(
                            "Co-simulation job generated an exception",
                            exc_info=True,
                            extra={
                                "job_params": job_params,
                                "exception": str(exc),
                            },
                        )
                    logger.info(
                        "Finished Sequential Co-simulation Job",
                        extra={
                            "job_index": f"{job_id}/{len(jobs)}",
                        },
                    )
    except Exception as e:
        raise RuntimeError("Failed to run simulation", e)

    # --- Result Handling ---
    # The simulation_results dictionary now contains paths to results inside temporary job workspaces.
    # The results_dir from the config is now the self-contained workspace's results folder.
    run_results_dir = results_dir
    os.makedirs(run_results_dir, exist_ok=True)

    # Unified result processing for both single and multiple jobs
    logger.info(
        "Processing jobs and combining results",
        extra={
            "num_jobs": len(jobs),
        },
    )
    combined_df = None

    all_dfs = []
    time_df_added = False

    for job_params in jobs:
        job_key = tuple(sorted(job_params.items()))
        result_path = simulation_results.get(job_key)

        if not result_path or not os.path.exists(result_path):
            logger.warning(
                "Job produced no result file",
                extra={
                    "job_params": job_params,
                },
            )
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
        rename_mapping = {
            col: f"{col}&{param_string}" if param_string else col
            for col in data_columns.columns
        }

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
        logger.info(
            "Combined results saved",
            extra={
                "file_path": combined_csv_path,
            },
        )
    else:
        logger.warning("No valid results found to combine")

    # --- Post-Processing ---
    if combined_df is not None:
        # Calculate the top-level post-processing directory
        top_level_run_workspace = os.path.abspath(config["run_timestamp"])
        top_level_post_processing_dir = os.path.join(
            top_level_run_workspace, "post_processing"
        )
        _run_post_processing(config, combined_df, top_level_post_processing_dir)
    else:
        logger.warning("No simulation results generated, skipping post-processing")

    # --- Final Cleanup ---
    # The primary cleanup of job workspaces is handled by the `finally` block in `_run_co_simulation`.
    # This is an additional safeguard.
    if not config["simulation"].get("keep_temp_files", True):
        temp_dir_path = os.path.abspath(config["paths"].get("temp_dir", "temp"))
        logger.info(
            "Cleaning up temporary directory",
            extra={
                "directory": temp_dir_path,
            },
        )
        if os.path.exists(temp_dir_path):
            try:
                shutil.rmtree(temp_dir_path)
                os.makedirs(temp_dir_path)  # Recreate for next run
            except OSError as e:
                logger.error(
                    "Error cleaning up temporary directory",
                    extra={
                        "directory": temp_dir_path,
                        "error": str(e),
                    },
                )


def archive_simulation(timestamp: str):
    """
    Archives a simulation run by collecting all necessary files and compressing them.
    """
    # Basic logging setup for archive command
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting archive for timestamp: {timestamp}")

    if not os.path.isdir(timestamp):
        logger.error(f"Timestamp directory not found: {timestamp}")
        sys.exit(1)

    # 0. Create archive directory
    archive_root = "archive"
    if os.path.exists(archive_root):
        shutil.rmtree(archive_root)  # Clean up previous archive attempt
    os.makedirs(archive_root)
    logger.info(f"Created archive directory: {archive_root}")

    try:
        # 1. Find log file and extract configs
        log_dir = os.path.join(timestamp, "log")
        log_file = None
        if os.path.isdir(log_dir):
            for f in os.listdir(log_dir):
                if f.startswith("simulation_") and f.endswith(".log"):
                    log_file = os.path.join(log_dir, f)
                    break

        if not log_file:
            logger.error(f"Could not find log file in {log_dir}")
            return

        runtime_config_str = None
        original_config_str = None
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if "message" in log_entry:
                        if log_entry["message"].startswith(
                            "Runtime Configuration (compact JSON):"
                        ):
                            runtime_config_str = log_entry["message"].replace(
                                "Runtime Configuration (compact JSON): ", ""
                            )
                        elif log_entry["message"].startswith(
                            "Original Configuration (compact JSON):"
                        ):
                            original_config_str = log_entry["message"].replace(
                                "Original Configuration (compact JSON): ", ""
                            )

                    if runtime_config_str and original_config_str:
                        break  # Found both, no need to read further
                except json.JSONDecodeError:
                    continue  # Ignore lines that are not valid JSON

        if not runtime_config_str or not original_config_str:
            logger.error(
                "Could not find runtime and/or original configuration in log file."
            )
            return

        runtime_config = json.loads(runtime_config_str)
        original_config = json.loads(original_config_str)
        logger.info("Successfully extracted both runtime and original configurations.")

        # Create a deep copy of the original config to modify
        final_config = json.loads(json.dumps(original_config))

        # 2. Use RUNTIME config to find, copy, and update path in FINAL config
        original_package_path = runtime_config["paths"]["package_path"]
        new_package_path = ""

        if os.path.isfile(original_package_path) and not original_package_path.endswith(
            "package.mo"
        ):
            # SINGLE-FILE
            model_filename = os.path.basename(original_package_path)
            destination_model_path = os.path.join(archive_root, model_filename)
            shutil.copy(original_package_path, destination_model_path)
            logger.info(f"Copied single-file model to {destination_model_path}")
            new_package_path = model_filename
        else:
            # MULTI-FILE
            if os.path.isfile(original_package_path):
                original_package_dir = os.path.dirname(original_package_path)
            else:
                original_package_dir = original_package_path

            package_dir_name = os.path.basename(original_package_dir)
            destination_model_dir = os.path.join(archive_root, package_dir_name)
            shutil.copytree(original_package_dir, destination_model_dir)
            logger.info(f"Copied multi-file model to {destination_model_dir}")

            if os.path.isfile(original_package_path):
                new_package_path = os.path.join(
                    package_dir_name, os.path.basename(original_package_path)
                ).replace("\\", "/")
            else:
                new_package_path = os.path.join(package_dir_name, "package.mo").replace(
                    "\\", "/"
                )

        # Update the path in the final_config
        if "paths" not in final_config:
            final_config["paths"] = {}
        final_config["paths"]["package_path"] = new_package_path
        logger.info(f"Updated package_path in final config to '{new_package_path}'")

        # 3. Save the MODIFIED ORIGINAL config
        config_path_in_archive = os.path.join(archive_root, "config.json")
        with open(config_path_in_archive, "w", encoding="utf-8") as f:
            json.dump(final_config, f, indent=4, ensure_ascii=False)
        logger.info(
            f"Saved modified original configuration to {config_path_in_archive}"
        )

        # 4. Copy timestamped results directory, ignoring temp/
        destination_timestamp_dir = os.path.join(archive_root, timestamp)
        shutil.copytree(
            timestamp, destination_timestamp_dir, ignore=shutil.ignore_patterns("temp")
        )
        logger.info(
            f"Copied timestamp directory to {destination_timestamp_dir}, ignoring temp/"
        )

        # 5. Pack into a compressed file
        archive_filename = f"archive_{timestamp}"
        shutil.make_archive(archive_filename, "zip", archive_root)
        logger.info(f"Successfully created archive: {archive_filename}.zip")

    finally:
        if os.path.exists(archive_root):
            shutil.rmtree(archive_root)
            logger.info(f"Cleaned up temporary archive directory: {archive_root}")


def _relativize_paths_in_config(config_node, logger):
    """
    Recursively traverses the config and converts absolute paths to relative (basename).
    """
    # The keys that hold paths that need to be made relative.
    path_keys = ["db_path", "results_dir", "temp_dir", "log_dir"]

    if isinstance(config_node, dict):
        for key, value in config_node.items():
            if isinstance(value, str):
                # Check if the key indicates a path. We exclude 'package_path' as it's handled separately.
                is_path_key = (
                    key.endswith("_path") or key in path_keys
                ) and key != "package_path"

                if is_path_key and os.path.isabs(value):
                    original_path = value
                    new_path = os.path.basename(original_path)
                    config_node[key] = new_path
                    logger.debug(
                        f"Relativizing path for key '{key}': '{original_path}' -> '{new_path}'"
                    )
            elif isinstance(value, (dict, list)):
                _relativize_paths_in_config(value, logger)  # Recurse
    elif isinstance(config_node, list):
        for item in config_node:
            _relativize_paths_in_config(item, logger)


def unarchive_simulation(zip_file: str):
    """
    Unarchives a simulation run from a zip file.
    Extracts to a new folder if the current directory is not empty.
    """
    # Basic logging setup for unarchive command
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    if not os.path.isfile(zip_file):
        logger.error(f"Archive file not found: {zip_file}")
        sys.exit(1)

    target_dir = "."
    if os.listdir("."):  # If the list of CWD contents is not empty
        dir_name = os.path.splitext(os.path.basename(zip_file))[0]
        target_dir = dir_name
        logger.info(
            f"Current directory is not empty. Extracting to new directory: {target_dir}"
        )
        os.makedirs(target_dir, exist_ok=True)
    else:
        logger.info("Current directory is empty. Extracting to current directory.")

    # Unzip the file
    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        logger.info(
            f"Successfully unarchived '{zip_file}' to '{os.path.abspath(target_dir)}'"
        )
    except zipfile.BadZipFile:
        logger.error(f"Error: '{zip_file}' is not a valid zip file.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during unarchiving: {e}")
        sys.exit(1)


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
                        "Converted path",
                        extra={
                            "key_name": key_name,
                            "original_value": value,
                            "absolute_path": abs_path,
                        },
                    )
                    return abs_path
            return value
        else:
            return value

    return _process_value(config)


def initialize_run() -> tuple[Dict[str, Any], Dict[str, Any]]:
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

    archive_parser = subparsers.add_parser("archive", help="Archive a simulation run.")
    archive_parser.add_argument(
        "timestamp",
        type=str,
        help="Timestamp of the simulation run to archive.",
    )

    unarchive_parser = subparsers.add_parser(
        "unarchive", help="Unarchive a simulation run."
    )
    unarchive_parser.add_argument(
        "zip_file",
        type=str,
        help="Path to the archive file to unarchive.",
    )

    args = parser.parse_args()

    if args.command == "archive":
        archive_simulation(args.timestamp)
        sys.exit(0)

    if args.command == "unarchive":
        unarchive_simulation(args.zip_file)
        sys.exit(0)

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
        default_config_path = "config.json"
        if os.path.exists(default_config_path):
            args.config = default_config_path
            print(
                f"INFO: No config file specified, using default: {default_config_path}"
            )
        else:
            parser.error(
                "the following arguments are required: -c/--config, or 'config.json' must exist in the current directory."
            )

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

    # Correctly set the base directory for resolving relative paths to the config file's location
    original_config_dir = os.path.dirname(config_path)

    # First convert relative paths to absolute paths
    absolute_config = _convert_relative_paths_to_absolute(
        base_config, original_config_dir
    )
    # Deep copy the converted configuration
    config = json.loads(json.dumps(absolute_config))
    # Generate a single timestamp for the entire run and add it to the config
    config["run_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Create a self-contained workspace for this run ---
    # The workspace is a directory named after the timestamp, created in the current working directory.
    run_workspace = os.path.abspath(config["run_timestamp"])

    # Ensure the 'paths' and 'logging' keys exist
    if "paths" not in config:
        config["paths"] = {}
    if "logging" not in config:
        config["logging"] = {}

    # Override the paths in the config to point to the new workspace
    config["logging"]["log_dir"] = os.path.join(run_workspace, "log")  # Corrected path
    config["paths"]["temp_dir"] = os.path.join(run_workspace, "temp")
    config["paths"]["results_dir"] = os.path.join(run_workspace, "results")

    # Create the new directory structure
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    os.makedirs(config["paths"]["temp_dir"], exist_ok=True)
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)

    # --- End of workspace creation ---

    return config, base_config


def main():
    """Main function to run the simulation from the command line."""
    config, original_config = initialize_run()
    setup_logging(config, original_config)
    logger.info(
        "Loading configuration",
        extra={
            "config_path": os.path.abspath(sys.argv[-1]),
        },
    )
    try:
        run_simulation(config)
        logger.info("Main execution completed successfully")
    except Exception as e:
        logger.error(
            "Main execution failed", exc_info=True, extra={"exception": str(e)}
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
