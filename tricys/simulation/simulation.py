import argparse
import concurrent.futures
import glob
import importlib
import importlib.util
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import pandas as pd
from OMPython import ModelicaSystem

from tricys.core.interceptor import integrate_interceptor_model
from tricys.core.jobs import generate_simulation_jobs
from tricys.core.modelica import (
    format_parameter_value,
    get_om_session,
    load_modelica_package,
)
from tricys.utils.config_utils import basic_prepare_config
from tricys.utils.file_utils import get_unique_filename
from tricys.utils.log_utils import setup_logging

# Standard logger setup
logger = logging.getLogger(__name__)


def run_co_simulation_job(config: dict, job_params: dict, job_id: int = 0) -> str:
    """Runs the full co-simulation workflow in an isolated directory.

    This function sets up a self-contained workspace for a single co-simulation
    job to ensure thread safety. It copies the model package, handles asset
    files, generates an interceptor model, and executes a two-stage
    simulation: first to get primary inputs, and second to run the final
    simulation with the intercepted model.

    Args:
        config: The main configuration dictionary.
        job_params: A dictionary of parameters specific to this job.
        job_id: A unique identifier for the job, used for workspace naming. Defaults to 0.

    Returns:
        The path to the final simulation result file, or an empty string if the simulation failed.

    Note:
        Creates isolated workspace in temp_dir/job_{job_id}. Supports both single-file
        and multi-file Modelica packages. Handles external interceptor handlers. Cleans
        up workspace after completion. Two-stage simulation: stage 1 generates input CSV,
        stage 2 runs with interceptor model.
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

        # Parse co_simulation config - new format with mode at top level
        co_sim_config = config["co_simulation"]
        mode = co_sim_config.get("mode", "interceptor")  # Get mode from top level
        handlers = co_sim_config.get("handlers", [])  # Get handlers array

        # Validate that handlers is a list
        if not isinstance(handlers, list):
            handlers = [handlers]

        model_name = sim_config["model_name"]
        stop_time = sim_config["stop_time"]
        step_size = sim_config["step_size"]

        omc = get_om_session()
        if not load_modelica_package(omc, Path(isolated_package_path).as_posix()):
            raise RuntimeError(
                f"Failed to load Modelica package at {isolated_package_path}"
            )

        # Handle copying of any additional asset directories specified with a '_path' suffix
        for handler_config in handlers:
            if "params" in handler_config:
                # Iterate over a copy of items since we are modifying the dict
                for param_key, param_value in list(handler_config["params"].items()):
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
                        handler_config["params"][param_key] = new_asset_path.as_posix()
                        logger.info(
                            "Updated asset parameter path",
                            extra={
                                "job_id": job_id,
                                "parameter_key": param_key,
                                "new_path": handler_config["params"][param_key],
                            },
                        )

        all_input_vars = []
        for handler_config in handlers:
            submodel_name = handler_config["submodel_name"]
            instance_name = handler_config["instance_name"]
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
        for handler_config in handlers:
            handler_function_name = handler_config["handler_function"]
            module = None

            # New method: Load from a direct script path
            if "handler_script_path" in handler_config:
                script_path_str = handler_config["handler_script_path"]
                script_path = Path(script_path_str).resolve()
                module_name = script_path.stem

                logger.info(
                    "Loading co-simulation handler from script path",
                    extra={
                        "job_id": job_id,
                        "script_path": str(script_path),
                        "function": handler_function_name,
                    },
                )

                if not script_path.is_file():
                    raise FileNotFoundError(
                        f"Co-simulation handler script not found at {script_path}"
                    )

                spec = importlib.util.spec_from_file_location(module_name, script_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                else:
                    raise ImportError(
                        f"Could not create module spec from script {script_path}"
                    )

            # Old method: Load from module name (backward compatibility)
            elif "handler_module" in handler_config:
                module_name = handler_config["handler_module"]
                logger.info(
                    "Loading co-simulation handler from module",
                    extra={
                        "job_id": job_id,
                        "module_name": module_name,
                        "function": handler_function_name,
                    },
                )
                module = importlib.import_module(module_name)

            else:
                raise KeyError(
                    "Handler config must contain either 'script_path' or 'handler_module'"
                )

            if not module:
                raise ImportError("Failed to load co-simulation handler module.")

            handler_function = getattr(module, handler_function_name)
            instance_name = handler_config["instance_name"]

            co_sim_output_filename = get_unique_filename(
                isolated_temp_dir, f"{instance_name}_outputs.csv"
            )

            output_placeholder = handler_function(
                temp_input_csv=primary_result_filename,
                temp_output_csv=co_sim_output_filename,
                **handler_config.get("params", {}),
            )

            interception_config = {
                "submodel_name": handler_config["submodel_name"],
                "instance_name": handler_config["instance_name"],
                "csv_uri": Path(os.path.abspath(co_sim_output_filename)).as_posix(),
                "output_placeholder": output_placeholder,
            }

            # Add mode from top-level co_simulation config
            interception_config["mode"] = mode

            interception_configs.append(interception_config)

        intercepted_model_paths = integrate_interceptor_model(
            package_path=isolated_package_path,
            model_name=model_name,
            interception_configs=interception_configs,
        )

        verif_config = config["simulation"]["variableFilter"]
        logger.info("Proceeding with Final simulation.", extra={"job_id": job_id})

        # Use mode from top-level config
        if mode == "replacement":
            # For direct replacement, the system model name stays the same
            logger.info(
                "Using direct replacement mode, system model unchanged",
                extra={"job_id": job_id},
            )
            final_model_name = model_name
            final_model_file = isolated_package_path
        else:
            # For interceptor mode, load the interceptor models and use modified system
            for model_path in intercepted_model_paths["interceptor_model_paths"]:
                omc.sendExpression(f"""loadFile("{Path(model_path).as_posix()}")""")
            omc.sendExpression(
                f"""loadFile("{Path(intercepted_model_paths["system_model_path"]).as_posix()}")"""
            )

            package_name, original_system_name = model_name.split(".")
            final_model_name = f"{package_name}.{original_system_name}_Intercepted"
            final_model_file = (
                Path(intercepted_model_paths["system_model_path"]).as_posix()
                if os.path.isfile(isolated_package_path)
                and not original_package_path.endswith("package.mo")
                else Path(isolated_package_path).as_posix()
            )

        verif_mod = ModelicaSystem(
            fileName=final_model_file,
            modelName=final_model_name,
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
        # Cleanup ModelicaSystem instances which might have their own sessions
        if "mod" in locals() and mod and hasattr(mod, "omc"):
            try:
                mod.omc.sendExpression("quit()")
                # mod.__del__() # triggering del might be safer but explicit quit is sure
            except Exception:
                pass

        if "verif_mod" in locals() and verif_mod and hasattr(verif_mod, "omc"):
            try:
                verif_mod.omc.sendExpression("quit()")
            except Exception:
                pass

        if omc:
            try:
                omc.sendExpression("quit()")
                logger.info("Closed OMPython session", extra={"job_id": job_id})
            except Exception:
                pass

        if not sim_config.get("keep_temp_files", True):
            if os.path.exists(job_workspace):
                shutil.rmtree(job_workspace)
                logger.info(
                    "Cleaned up job workspace",
                    extra={"job_id": job_id, "workspace": job_workspace},
                )


def run_single_job(config: dict, job_params: dict, job_id: int = 0) -> str:
    """Executes a single standard simulation job in an isolated workspace.

    This function sets up a dedicated workspace for one simulation run,
    initializes an OpenModelica session, sets the specified model parameters,
    runs the simulation, and cleans up the result file.

    Args:
        config: The main configuration dictionary.
        job_params: A dictionary of parameters specific to this job.
        job_id: A unique identifier for the job. Defaults to 0.

    Returns:
        The path to the simulation result file, or an empty string on failure.

    Note:
        Creates isolated workspace in temp_dir/job_{job_id}. Cleans result file by
        removing duplicate/NaN time values. Cleans up workspace unless keep_temp_files
        is True. Uses CSV output format with configurable stopTime and stepSize.
    """
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


def run_sequential_sweep(
    config: dict,
    jobs: List[Dict[str, Any]],
    post_job_callback: Callable[[int, Dict[str, Any], str], None] = None,
) -> List[str]:
    """Executes a parameter sweep sequentially.

    This function runs a series of simulation jobs one after another, reusing
    the same OpenModelica session for efficiency. Intermediate results for each
    job are saved to a dedicated job workspace.

    Args:
        config: The main configuration dictionary.
        jobs: A list of job parameter dictionaries to execute.

    Returns:
        A list of paths to the result files for each job. Failed jobs will have
        an empty string as their path.

    Note:
        Reuses single OMPython session for all jobs. Cleans result files by removing
        duplicate/NaN time values. Cleans up workspaces unless keep_temp_files is True.
        More efficient than parallel mode for small job counts.
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

                if post_job_callback:
                    try:
                        post_job_callback(i, job_params, result_file_path)
                    except Exception as e:
                        logger.error(
                            "Post-job callback failed",
                            exc_info=True,
                            extra={"job_index": i + 1, "error": str(e)},
                        )
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


def _mp_run_co_simulation_wrapper(args):
    """Wrapper method for multiprocessing co-simulation."""
    config, job_params, job_id = args
    try:
        result_path = run_co_simulation_job(config, job_params, job_id)
        return job_id, job_params, result_path, None
    except Exception as e:
        return job_id, job_params, None, str(e)


def _mp_generic_wrapper(args):
    """Generic wrapper for multiprocessing that accepts a function."""
    func, config, job_params, job_id = args
    try:
        result = func(config, job_params, job_id)
        return job_id, job_params, result, None
    except Exception as e:
        return job_id, job_params, None, str(e)


def run_parallel_sweep(
    config: dict,
    jobs: List[Dict[str, Any]],
    max_workers: int = None,
    use_multiprocessing: bool = False,
    post_job_callback: Callable[[int, Dict[str, Any], Any], None] = None,
    custom_runner_func: Callable = None,
) -> List[Any]:
    """Executes a list of jobs in parallel using ThreadPoolExecutor or MultiprocessinPool.

    Calculates results for a list of simulation jobs concurrently.

    Args:
        config: The main configuration dictionary.
        jobs: A list of job parameter dictionaries to execute.
        max_workers: Maximum number of concurrent workers. Defaults to CPU count.
        use_multiprocessing: If True, uses multiprocessing.Pool (recommended for Co-Simulation).
                             If False, uses ThreadPoolExecutor (recommended for standard jobs).
        post_job_callback: Optional callback invoked immediately after each job completion.
                           Signature: (job_id, job_params, result) -> None.
                           The result is whatever the runner function returns.
        custom_runner_func: Optional function to execute the job.
                           Signature: (config, job_params, job_id) -> result.
                           If not provided, defaults to run_co_simulation_job or run_single_job based on config.

    Returns:
        A list of results for each job (usually paths, but can be tuples if custom runner is used),
        corresponding to the input `jobs` order.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    logger.info(
        f"Starting parallel sweep with {len(jobs)} jobs.",
        extra={
            "max_workers": max_workers,
            "use_multiprocessing": use_multiprocessing,
            "has_custom_runner": custom_runner_func is not None,
        },
    )

    # Initialize result list with empty strings to preserve order
    # Although we can fill it by index
    results_map = {}  # job_id -> result

    # Helper to resolve job runner function
    has_co_sim = config.get("co_simulation") is not None

    # Logic to choose runner
    if custom_runner_func:
        runner_func = custom_runner_func
    else:
        runner_func = run_co_simulation_job if has_co_sim else run_single_job

    try:
        if use_multiprocessing:
            # MULTIPROCESSING PATH
            if not custom_runner_func and not has_co_sim:
                logger.warning(
                    "Multiprocessing requested for standard job simulation. Using Threading instead as MP wrapper is optimized for Co-Sim."
                )
                use_multiprocessing = False
            else:
                # If custom runner is provided, we need to be careful about pickling.
                # _mp_run_co_simulation_wrapper hardcodes run_co_simulation_job.
                # If we want to support custom runner in MP, we need a generic wrapper or user must provide one.
                # For now, let's assume if custom_runner is passed, it is picklable or user knows what they are doing?
                # Check if custom runner is provided, if so use generic wrapper, else use default co-sim wrapper
                wrapper = (
                    _mp_generic_wrapper
                    if custom_runner_func
                    else _mp_run_co_simulation_wrapper
                )

                # Start MP pool
                # For generic wrapper, we need to pass the runner function in the args!
                if custom_runner_func:
                    mp_args = [
                        (custom_runner_func, config, job_params, i + 1)
                        for i, job_params in enumerate(jobs)
                    ]
                else:
                    mp_args = [
                        (config, job_params, i + 1) for i, job_params in enumerate(jobs)
                    ]

                with multiprocessing.Pool(processes=max_workers) as pool:
                    for job_id, job_params, result, error in pool.imap_unordered(
                        wrapper, mp_args
                    ):
                        if error:
                            logger.error(f"Job {job_id} failed: {error}")
                            results_map[job_id] = ""
                        else:
                            results_map[job_id] = result
                            if post_job_callback:
                                try:
                                    post_job_callback(job_id, job_params, result)
                                except Exception as exc:
                                    logger.error(
                                        f"Post-job callback failed for job {job_id}: {exc}"
                                    )

        if not use_multiprocessing:
            # THREADING PATH
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Map futures to job IDs
                future_to_job = {
                    executor.submit(runner_func, config, job_params, i + 1): (
                        i + 1,
                        job_params,
                    )
                    for i, job_params in enumerate(jobs)
                }

                for future in concurrent.futures.as_completed(future_to_job):
                    job_id, job_params = future_to_job[future]
                    try:
                        result_path = future.result()
                        results_map[job_id] = result_path

                        if post_job_callback and result_path:
                            try:
                                post_job_callback(job_id, job_params, result_path)
                            except Exception as exc:
                                logger.error(
                                    f"Post-job callback failed for job {job_id}: {exc}"
                                )

                    except Exception as exc:
                        logger.error(
                            f"Job {job_id} generated an exception: {exc}", exc_info=True
                        )
                        results_map[job_id] = ""

    except Exception as e:
        logger.error(f"Parallel sweep failed: {e}", exc_info=True)

    # Construct final list in order
    final_results = []
    for i in range(len(jobs)):
        final_results.append(results_map.get(i + 1, ""))

    return final_results


def run_post_processing(
    config: Dict[str, Any],
    results_df: pd.DataFrame,
    post_processing_output_dir: str,
    results_file_path: str = None,
) -> None:
    """Dynamically loads and runs post-processing modules.

    This function iterates through the post-processing tasks defined in the
    configuration. For each task, it dynamically loads the specified module
    (from a module name or a script path) and executes the target function,
    passing the results DataFrame and other parameters to it.

    Args:
        config: The main configuration dictionary.
        results_df: The combined DataFrame of simulation results.
        post_processing_output_dir: The directory to save any output from the tasks.
        results_file_path: Path to the HDF5 results file (optional).

    Note:
        Supports two loading methods: 'script_path' for direct .py files, or 'module'
        for installed packages. Creates output_dir if it doesn't exist. Passes results_df,
        output_dir, and user-specified params to each task function. Logs errors for
        failed tasks but continues with remaining tasks.
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
            function_name = task_config["function"]
            params = task_config.get("params", {})
            module = None

            # New method: Load from a direct script path
            if "script_path" in task_config:
                script_path_str = task_config["script_path"]
                script_path = Path(script_path_str).resolve()
                module_name = script_path.stem

                logger.info(
                    "Running post-processing task from script path",
                    extra={
                        "task_index": i + 1,
                        "script_path": str(script_path),
                        "function": function_name,
                    },
                )

                if not script_path.is_file():
                    logger.error(
                        "Post-processing script not found",
                        extra={"path": str(script_path)},
                    )
                    continue

                spec = importlib.util.spec_from_file_location(module_name, script_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                else:
                    logger.error(
                        "Could not create module spec from script",
                        extra={"path": str(script_path)},
                    )
                    continue

            # Old method: Load from module name (backward compatibility)
            elif "module" in task_config:
                module_name = task_config["module"]

                # Auto-redirect to HDF5 version if applicable
                if (
                    results_file_path
                    and "tricys.postprocess" in module_name
                    and "hdf5" not in module_name
                ):
                    base_name = module_name.split(".")[-1]
                    potential_hdf5_module = f"tricys.postprocess.hdf5.{base_name}"
                    try:
                        importlib.util.find_spec(potential_hdf5_module)
                        module_name = potential_hdf5_module
                        logger.info(
                            f"Redirecting to HDF5 post-processing module: {module_name}"
                        )
                    except ImportError:
                        pass  # Fallback to original

                logger.info(
                    "Running post-processing task from module",
                    extra={
                        "task_index": i + 1,
                        "module_name": module_name,
                        "function": function_name,
                    },
                )
                module = importlib.import_module(module_name)

            else:
                logger.warning(
                    "Post-processing task is missing 'script_path' or 'module' key. Skipping.",
                    extra={"task_index": i + 1},
                )
                continue

            if module:
                post_processing_func = getattr(module, function_name)
                if results_file_path:
                    # Pass HDF5 path instead of DataFrame
                    post_processing_func(
                        results_file_path=results_file_path,
                        output_dir=post_processing_dir,
                        **params,
                    )
                else:
                    # Pass DataFrame (Legacy)
                    post_processing_func(
                        results_df=results_df, output_dir=post_processing_dir, **params
                    )
            else:
                logger.error(
                    "Failed to load post-processing module.",
                    extra={"task_index": i + 1},
                )

        except Exception:
            logger.error(
                "Post-processing task failed",
                exc_info=True,
                extra={"task_index": i + 1},
            )
    logger.info("Post-processing phase ended")


def _build_model_only(config: dict) -> tuple[str, str, str]:
    """
    Simulates "Compile Once" strategy.
    Builds the model once and returns paths to the executable and init xml.
    """
    paths_config = config["paths"]
    sim_config = config["simulation"]

    # Use a dedicated build directory to keep things clean
    build_dir = os.path.abspath(
        os.path.join(paths_config.get("temp_dir", "temp"), "build")
    )
    os.makedirs(build_dir, exist_ok=True)

    logger.info(f"Building model in {build_dir}...")

    omc = get_om_session()
    try:
        omc.sendExpression(f'cd("{Path(build_dir).as_posix()}")')

        package_path = os.path.abspath(paths_config["package_path"])
        if not load_modelica_package(omc, Path(package_path).as_posix()):
            raise RuntimeError("Failed to load Modelica package during build.")

        model_name = sim_config["model_name"]
        build_result = omc.sendExpression(f"buildModel({model_name})")

        if not build_result or len(build_result) < 2:
            err = omc.sendExpression("getErrorString()")
            raise RuntimeError(f"Model build failed: {err}")

        exe_name = build_result[0] + ".exe"
        xml_name = build_result[1]
        exe_path = os.path.join(build_dir, exe_name)
        xml_path = os.path.join(build_dir, xml_name)

        om_home = omc.sendExpression("getInstallationDirectoryPath()")
        om_bin_path = os.path.join(om_home, "bin")

        logger.info(f"Model built successfully: {exe_path}")
        return exe_path, xml_path, om_bin_path

    finally:
        omc.sendExpression("quit()")


def _run_fast_subprocess_job(
    job_params: dict,
    job_id: int,
    exe_source: str,
    xml_source: str,
    om_bin_path: str,
    base_temp_dir: str,
    sim_config: dict,
    variable_filter: str = None,
    inplace_execution: bool = False,
) -> str:
    """
    Executes a single simulation job using the pre-compiled executable.
    """
    job_workspace = os.path.join(base_temp_dir, f"job_{job_id}")
    os.makedirs(job_workspace, exist_ok=True)

    run_cwd = ""
    run_exe_path = ""

    if inplace_execution:
        # In-place: Run directly from build dir (Sequential optimization)
        run_cwd = os.path.dirname(exe_source)
        run_exe_path = exe_source
        result_filename = os.path.abspath(
            os.path.join(job_workspace, f"job_{job_id}_res.csv")
        )
    else:
        # Isolated: Copy exe to job dir (Concurrent safe)
        run_cwd = job_workspace
        build_dir = os.path.dirname(exe_source)
        model_prefix = os.path.splitext(os.path.basename(exe_source))[0]
        artifacts = glob.glob(os.path.join(build_dir, f"{model_prefix}*"))
        ignored_extensions = {".c", ".h", ".o", ".cpp", ".log", ".makefile", ".libs"}

        run_exe_path = ""
        try:
            for src_file in artifacts:
                if not os.path.isfile(src_file):
                    continue
                _, ext = os.path.splitext(src_file)
                if ext.lower() in ignored_extensions:
                    continue

                dst_file = os.path.join(job_workspace, os.path.basename(src_file))
                shutil.copy(src_file, dst_file)
                if os.path.basename(exe_source) == os.path.basename(src_file):
                    run_exe_path = dst_file
        except IOError as e:
            logger.error(f"Job {job_id}: Copy failed: {e}")
            return ""

        result_filename = os.path.abspath(
            os.path.join(job_workspace, f"job_{job_id}_res.csv")
        )

    if not run_exe_path:
        return ""

    override_pairs = [f"{k}={v}" for k, v in job_params.items()]
    override_pairs.append(f"stopTime={sim_config['stop_time']}")
    override_pairs.append(f"stepSize={sim_config['step_size']}")
    override_pairs.append("outputFormat=csv")
    if variable_filter:
        override_pairs.append(f"variableFilter={variable_filter}")
    override_str = ",".join(override_pairs)

    cmd = [run_exe_path, "-override", override_str, "-r", result_filename]

    env = os.environ.copy()
    if sys.platform == "win32":
        env["PATH"] = om_bin_path + os.pathsep + env["PATH"]

    try:
        subprocess.run(
            cmd,
            env=env,
            cwd=run_cwd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if os.path.exists(result_filename):
            try:
                df = pd.read_csv(result_filename)
                df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                df.to_csv(result_filename, index=False)
                return result_filename
            except Exception:
                return result_filename
        else:
            return ""

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        return ""


def _mp_run_fast_subprocess_job_wrapper(args):
    """Wrapper for multiprocessing.Pool to map _run_fast_subprocess_job."""
    job_params, job_id, kwargs = args
    try:
        res = _run_fast_subprocess_job(
            job_params,
            job_id,
            kwargs["exe_source"],
            kwargs["xml_source"],
            kwargs["om_bin_path"],
            kwargs["base_temp_dir"],
            kwargs["sim_config"],
            variable_filter=kwargs.get("variable_filter"),
            inplace_execution=kwargs.get("inplace_execution", False),
        )
        return job_id, job_params, res, None
    except Exception as e:
        return job_id, job_params, None, str(e)


def _mp_run_single_job_wrapper(args):
    """Wrapper for multiprocessing.Pool to map run_single_job."""
    config, job_params, job_id = args
    try:
        res = run_single_job(config, job_params, job_id)
        return job_id, job_params, res, None
    except Exception as e:
        return job_id, job_params, None, str(e)


def _mp_run_co_simulation_job_wrapper(args):
    """Wrapper for multiprocessing.Pool to map run_co_simulation_job."""
    config, job_params, job_id = args
    try:
        res = run_co_simulation_job(config, job_params, job_id)
        return job_id, job_params, res, None
    except Exception as e:
        return job_id, job_params, None, str(e)


def run_simulation(config: Dict[str, Any]) -> None:
    """Orchestrates the main simulation workflow.

    This function serves as the primary orchestrator for running simulations.
    It generates jobs from parameters, executes them (concurrently or sequentially,
    as standard or co-simulations), merges the results into a single DataFrame,
    and triggers any configured post-processing steps.

    Args:
        config: The main configuration dictionary for the run.

    Note:
        Supports concurrent (ProcessPoolExecutor) and sequential execution modes.
        Co-simulations use ProcessPoolExecutor for better isolation. Merges all job
        results into single CSV with parameter-labeled columns. Triggers post-processing
        tasks if configured. Results saved to paths.results_dir.
    """
    jobs = generate_simulation_jobs(config.get("simulation_parameters", {}))

    try:
        results_dir = os.path.abspath(config["paths"]["results_dir"])
    except KeyError as e:
        logger.error(f"Missing required path key in configuration file: {e}")
        sys.exit(1)

    # Use HDF5 for results to prevent OOM
    run_results_dir = results_dir
    os.makedirs(run_results_dir, exist_ok=True)
    hdf_filename = "sweep_results.h5"
    hdf_path = get_unique_filename(run_results_dir, hdf_filename)

    # Helper to process a single result
    def process_result(store, job_id, params, res_path):
        if not res_path or not os.path.exists(res_path):
            return

        try:
            df = pd.read_csv(res_path)

            # Add job_id column for linkage
            df["job_id"] = job_id

            # Store result in 'results' table
            store.append(
                "results", df, format="table", index=False, data_columns=["job_id"]
            )

            # Store parameters in 'jobs' table
            param_row = params.copy()
            param_row["job_id"] = job_id
            param_df = pd.DataFrame([param_row])

            store.append(
                "jobs", param_df, format="table", index=False, data_columns=["job_id"]
            )

            # Immediate cleanup of the entire job directory to save space
            try:
                job_dir = os.path.dirname(res_path)
                if os.path.exists(job_dir) and "job_" in os.path.basename(job_dir):
                    shutil.rmtree(job_dir)
            except OSError as e:
                logger.warning(f"Failed to clean up job directory {job_dir}: {e}")

        except Exception as e:
            logger.error(f"Failed to process result for Job {job_id}: {e}")

    simulation_results = {}
    use_concurrent = config["simulation"].get("concurrent", False)

    try:
        max_workers = config["simulation"].get("max_workers", os.cpu_count())

        # HDF5 path is prepared but only used in enhanced mode

        # Check for enhanced execution mode (Global)
        execute_mode = config["simulation"].get("execute_mode", "standard")
        if config["simulation"].get("excute_mode"):
            execute_mode = config["simulation"]["excute_mode"]

        if config.get("co_simulation") is None:

            if execute_mode == "enhanced":
                logger.info(
                    "Starting Standard Simulation (Enhanced Mode: Compile Once)"
                )

                # Initialize HDFStore for streaming results (Enhanced Mode Only)
                with pd.HDFStore(
                    hdf_path, mode="w", complib="blosc", complevel=9
                ) as store:
                    # Save configuration
                    try:
                        config_df = pd.DataFrame({"config_json": [json.dumps(config)]})
                        store.put("config", config_df, format="fixed")
                    except Exception as e:
                        logger.warning(f"Failed to save config to HDF5: {e}")

                    # 1. Compile Once
                    master_exe, master_xml, om_bin = _build_model_only(config)
                    temp_dir = os.path.abspath(config["paths"].get("temp_dir", "temp"))

                    # 2. Run Many
                    if use_concurrent:
                        logger.info(
                            "Running jobs concurrently (Enhanced HDF5 Streaming)",
                            extra={"max_workers": max_workers},
                        )
                        # Prepare args
                        mp_args = []
                        for i, job_params in enumerate(jobs):
                            kwargs = {
                                "exe_source": master_exe,
                                "xml_source": master_xml,
                                "om_bin_path": om_bin,
                                "base_temp_dir": temp_dir,
                                "sim_config": config["simulation"],
                                "variable_filter": config["simulation"].get(
                                    "variableFilter"
                                ),
                                "inplace_execution": True,
                            }
                            mp_args.append((job_params, i + 1, kwargs))

                        with multiprocessing.Pool(processes=max_workers) as pool:
                            for (
                                job_id,
                                job_p,
                                result_path,
                                error,
                            ) in pool.imap_unordered(
                                _mp_run_fast_subprocess_job_wrapper, mp_args
                            ):
                                if error:
                                    logger.error(f"Job {job_id} failed: {error}")
                                else:
                                    process_result(store, job_id, job_p, result_path)
                    else:
                        logger.info(
                            "Running jobs sequentially (Enhanced HDF5 Streaming)"
                        )
                        for i, job_params in enumerate(jobs):
                            result_path = _run_fast_subprocess_job(
                                job_params,
                                i + 1,
                                master_exe,
                                master_xml,
                                om_bin,
                                temp_dir,
                                config["simulation"],
                                variable_filter=config["simulation"].get(
                                    "variableFilter"
                                ),
                                inplace_execution=True,
                            )
                            process_result(store, i + 1, job_params, result_path)

                # Mark that we used HDF5 so we skip the legacy aggregation
                simulation_results = None

            else:
                # Legacy Logic (Standard) - NO HDFStore active
                if use_concurrent:
                    logger.info(
                        "Starting simulation",
                        extra={
                            "mode": "CONCURRENT",
                            "max_workers": max_workers,
                        },
                    )

                    mp_args = [
                        (config, job_params, i + 1) for i, job_params in enumerate(jobs)
                    ]

                    with multiprocessing.Pool(processes=max_workers) as pool:
                        for (
                            job_id,
                            job_p,
                            result_path,
                            error,
                        ) in pool.imap_unordered(_mp_run_single_job_wrapper, mp_args):
                            if error:
                                logger.error(f"Job {job_id} failed: {error}")
                            elif result_path:
                                simulation_results[tuple(sorted(job_p.items()))] = (
                                    result_path
                                )
                else:
                    logger.info("Starting simulation", extra={"mode": "SEQUENTIAL"})
                    result_paths = run_sequential_sweep(config, jobs)
                    for i, result_path in enumerate(result_paths):
                        if result_path:
                            simulation_results[tuple(sorted(jobs[i].items()))] = (
                                result_path
                            )
        else:
            if execute_mode == "enhanced":
                logger.info("Starting Co-Simulation (Enhanced Mode: HDF5 Streaming)")

                with pd.HDFStore(
                    hdf_path, mode="w", complib="blosc", complevel=9
                ) as store:
                    # Save configuration
                    try:
                        config_df = pd.DataFrame({"config_json": [json.dumps(config)]})
                        store.put("config", config_df, format="fixed")
                    except Exception as e:
                        logger.warning(f"Failed to save config to HDF5: {e}")

                    if use_concurrent:
                        logger.info(
                            "Starting co-simulation (Enhanced Parallel)",
                            extra={"max_workers": max_workers},
                        )
                        mp_args = [
                            (config, job_params, i + 1)
                            for i, job_params in enumerate(jobs)
                        ]

                        with multiprocessing.Pool(processes=max_workers) as pool:
                            for (
                                job_id,
                                job_p,
                                result_path,
                                error,
                            ) in pool.imap_unordered(
                                _mp_run_co_simulation_job_wrapper, mp_args
                            ):
                                if error:
                                    logger.error(
                                        "Co-simulation job generated an exception",
                                        extra={"job_params": job_p, "exception": error},
                                    )
                                else:
                                    process_result(store, job_id, job_p, result_path)
                                    logger.info(
                                        "Successfully finished co-simulation job",
                                        extra={"job_params": job_p},
                                    )
                    else:
                        logger.info("Starting co-simulation (Enhanced Sequential)")
                        for i, job_params in enumerate(jobs):
                            job_id = i + 1
                            try:
                                result_path = run_co_simulation_job(
                                    config, job_params, job_id=job_id
                                )
                                process_result(store, job_id, job_params, result_path)
                                logger.info(
                                    "Successfully finished co-simulation job",
                                    extra={"job_params": job_params},
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

                # Mark that we used HDF5 so we skip the legacy aggregation
                simulation_results = None

            else:
                if use_concurrent:
                    logger.info(
                        "Starting co-simulation",
                        extra={
                            "mode": "CONCURRENT",
                            "max_workers": max_workers,
                        },
                    )

                    mp_args = [
                        (config, job_params, i + 1) for i, job_params in enumerate(jobs)
                    ]

                    with multiprocessing.Pool(processes=max_workers) as pool:
                        for (
                            job_id,
                            job_p,
                            result_path,
                            error,
                        ) in pool.imap_unordered(
                            _mp_run_co_simulation_job_wrapper, mp_args
                        ):
                            if error:
                                logger.error(
                                    "Co-simulation job generated an exception",
                                    extra={
                                        "job_params": job_p,
                                        "exception": error,
                                    },
                                )
                            elif result_path:
                                simulation_results[tuple(sorted(job_p.items()))] = (
                                    result_path
                                )
                                logger.info(
                                    "Successfully finished co-simulation job",
                                    extra={
                                        "job_params": job_p,
                                    },
                                )
                            else:
                                logger.warning(
                                    "Co-simulation job did not return a result path",
                                    extra={
                                        "job_params": job_p,
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
                            result_path = run_co_simulation_job(
                                config, job_params, job_id=job_id
                            )
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

    if simulation_results is not None:
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
    # Enhanced mode (HDF5) or Legacy (DataFrame)
    if (combined_df is not None and not combined_df.empty) or (
        simulation_results is None and os.path.exists(hdf_path)
    ):
        # Calculate the top-level post-processing directory
        top_level_run_workspace = os.path.abspath(config["run_timestamp"])
        top_level_post_processing_dir = os.path.join(
            top_level_run_workspace, "post_processing"
        )

        # Determine if we're using HDF5 path (Enhanced mode implies simulation_results is None)
        path_to_pass = hdf_path if simulation_results is None else None

        run_post_processing(
            config,
            combined_df,
            top_level_post_processing_dir,
            results_file_path=path_to_pass,
        )
    else:
        logger.warning("No simulation results generated, skipping post-processing")

    # --- Save Logs to HDF5 ---
    if os.path.exists(hdf_path):
        try:
            # Flush logs to ensure everything is written to disk
            for handler in logging.getLogger().handlers:
                handler.flush()

            log_dir_path = config.get("paths", {}).get("log_dir")
            run_timestamp = config.get("run_timestamp")

            if log_dir_path and run_timestamp:
                abs_log_dir = os.path.abspath(log_dir_path)
                log_file_path = os.path.join(
                    abs_log_dir, f"simulation_{run_timestamp}.log"
                )

                if os.path.exists(log_file_path):
                    log_content = []
                    # Read log file safely
                    with open(
                        log_file_path, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        # Parse each line as JSON
                        for line in f:
                            try:
                                log_content.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                # Fallback for non-JSON lines if any
                                log_content.append({"raw_message": line.strip()})

                    if log_content:
                        with pd.HDFStore(
                            hdf_path, mode="a", complib="blosc", complevel=9
                        ) as store:
                            log_df = pd.DataFrame(
                                {"log_json": [json.dumps(log_content)]}
                            )
                            store.put("log", log_df, format="fixed")
                        logger.info("Logs saved to HDF5")
        except Exception as e:
            logger.warning(f"Failed to save logs to HDF5: {e}")

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


def main(config_or_path: Union[str, Dict[str, Any]], base_dir: str = None) -> None:
    """Main entry point for the simulation runner.

    This function prepares the configuration, sets up logging, and invokes
    the main `run_simulation` orchestrator.

    Args:
        config_or_path: The path to the JSON configuration file OR a config dict.
        base_dir: Optional base directory for resolving relative paths if a dict is passed.
    """
    config, original_config = basic_prepare_config(config_or_path, base_dir=base_dir)
    setup_logging(config, original_config)
    logger.info(
        "Loading configuration",
        extra={
            "config_source": (
                os.path.abspath(config_or_path)
                if isinstance(config_or_path, str)
                else "Dictionary"
            ),
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
    main(args.config)
