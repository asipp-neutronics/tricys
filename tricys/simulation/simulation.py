import argparse
import concurrent.futures
import importlib
import importlib.util
import logging
import os
import shutil
import sys
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
from tricys.utils.config_utils import basic_prepare_config
from tricys.utils.file_utils import get_unique_filename
from tricys.utils.log_utils import setup_logging

# Standard logger setup
logger = logging.getLogger(__name__)


def _run_co_simulation(config: dict, job_params: dict, job_id: int = 0) -> str:
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
        if omc:
            omc.sendExpression("quit()")
            logger.info("Closed OMPython session", extra={"job_id": job_id})

        if not sim_config.get("keep_temp_files", True):
            if os.path.exists(job_workspace):
                shutil.rmtree(job_workspace)
                logger.info(
                    "Cleaned up job workspace",
                    extra={"job_id": job_id, "workspace": job_workspace},
                )


def _run_single_job(config: dict, job_params: dict, job_id: int = 0) -> str:
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


def _run_sequential_sweep(config: dict, jobs: List[Dict[str, Any]]) -> List[str]:
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


def run_simulation(config: Dict[str, Any]) -> None:
    """Orchestrates the main simulation workflow.

    This function serves as the primary orchestrator for running simulations.
    It generates jobs from parameters, executes them (concurrently or sequentially,
    as standard or co-simulations), merges the results into a single DataFrame,
    and triggers any configured post-processing steps.

    Args:
        config: The main configuration dictionary for the run.

    Note:
        Supports concurrent (ThreadPoolExecutor) and sequential execution modes.
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

    simulation_results = {}
    use_concurrent = config["simulation"].get("concurrent", False)

    try:
        max_workers = config["simulation"].get("max_workers", os.cpu_count())
        if config.get("co_simulation") is None:
            if use_concurrent:
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


def main(config_path: str) -> None:
    """Main entry point for a standard simulation run.

    This function prepares the configuration, sets up logging, and calls
    the main `run_simulation` orchestrator.

    Args:
        config_path (str): The path to the JSON configuration file.
    """
    config, original_config = basic_prepare_config(config_path)
    setup_logging(config, original_config)
    logger.info(
        "Loading configuration",
        extra={
            "config_path": os.path.abspath(config_path),
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
