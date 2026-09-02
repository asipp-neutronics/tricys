import argparse
import copy
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
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

# Suppress PyTables NaturalNameWarning which occurs when saving Modelica variables (with dots and brackets) to HDF5
try:
    from tables import NaturalNameWarning

    warnings.filterwarnings("ignore", category=NaturalNameWarning)
except ImportError:
    pass

import pandas as pd

from tricys.analysis.metric import (
    build_single_job_summary_df,
    calculate_single_job_metrics,
)
from tricys.core.foc import prepare_foc_simulation_package
from tricys.core.interceptor import integrate_interceptor_model
from tricys.core.jobs import generate_simulation_jobs
from tricys.core.modelica import (
    ModelicaSystem,
    build_modelica_parameter_map,
    format_parameter_value,
    get_om_session,
    load_modelica_package,
)
from tricys.utils.concurrency_utils import get_safe_max_workers
from tricys.utils.config_utils import basic_prepare_config
from tricys.utils.file_utils import get_unique_filename
from tricys.utils.filter_schema import find_filter_schema_violations
from tricys.utils.log_utils import setup_logging

# Standard logger setup
logger = logging.getLogger(__name__)


def _resolve_built_model_paths(build_result: list, build_dir: str) -> tuple[str, str]:
    executable_artifact = str(
        (build_result[0] if len(build_result) > 0 else "") or ""
    ).strip()
    xml_artifact = str((build_result[1] if len(build_result) > 1 else "") or "").strip()

    if not executable_artifact or not xml_artifact:
        raise RuntimeError(f"Model build returned invalid artifacts: {build_result!r}")

    executable_name = executable_artifact
    if sys.platform == "win32" and not executable_name.lower().endswith(".exe"):
        executable_name = f"{executable_name}.exe"

    executable_path = (
        executable_name
        if os.path.isabs(executable_name)
        else os.path.join(build_dir, executable_name)
    )
    xml_path = (
        xml_artifact
        if os.path.isabs(xml_artifact)
        else os.path.join(build_dir, xml_artifact)
    )
    return executable_path, xml_path


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
        handlers = copy.deepcopy(
            co_sim_config.get("handlers", [])
        )  # Get handlers array

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
                        if not original_asset_path.exists():
                            logger.warning(
                                f"Asset file '{original_asset_path}' for parameter '{param_key}' not found. Skipping copy."
                            )
                            continue

                        # Create a destination directory based on the asset's parent folder name
                        # to avoid collisions if multiple files have the same name.
                        asset_dir_name = original_asset_path.parent.name
                        dest_dir = Path(job_workspace) / asset_dir_name
                        os.makedirs(dest_dir, exist_ok=True)

                        dest_path = dest_dir / original_asset_path.name

                        # Copy the file only if it hasn't been copied already
                        if not dest_path.exists():
                            shutil.copy(original_asset_path, dest_path)
                            logger.info(
                                "Copied asset file",
                                extra={
                                    "job_id": job_id,
                                    "source_path": original_asset_path,
                                    "destination_path": dest_path,
                                },
                            )

                        # Update the path in the config to point to the new location
                        handler_config["params"][param_key] = dest_path.as_posix()
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
            {
                "stopTime": stop_time,
                "stepSize": step_size,
                "outputFormat": "csv",
            }
        )

        param_settings = build_modelica_parameter_map(job_params)
        if param_settings:
            logger.info(
                "Applying parameters for job",
                extra={
                    "job_id": job_id,
                    "param_settings": [
                        format_parameter_value(name, value)
                        for name, value in job_params.items()
                    ],
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
                if os.getcwd() not in sys.path:
                    sys.path.insert(0, os.getcwd())
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

            if not output_placeholder:
                raise RuntimeError(
                    f"Co-simulation handler returned no output mapping for instance {instance_name}."
                )
            if not os.path.exists(co_sim_output_filename):
                raise FileNotFoundError(
                    f"Co-simulation handler did not create output CSV: {co_sim_output_filename}"
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
            {
                "stopTime": stop_time,
                "stepSize": step_size,
                "outputFormat": "csv",
            }
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


def run_sequential_sweep(
    config: dict,
    jobs: List[Dict[str, Any]],
    post_job_callback: Callable[[int, Dict[str, Any], str], None] = None,
    filter_schema: List[Dict[str, Any]] | None = None,
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
            {
                "stopTime": sim_config["stop_time"],
                "tolerance": 1e-6,
                "outputFormat": "csv",
                "stepSize": sim_config["step_size"],
            }
        )
        # mod.buildModel()

        for i, job_params in enumerate(jobs):
            try:
                # Add standardized progress log for backend parsing (Pattern: Job X of Y)
                logger.info(f"Job {i+1} of {len(jobs)}")

                logger.info(
                    "Running sequential job",
                    extra={
                        "job_index": f"{i+1}/{len(jobs)}",
                        "job_params": job_params,
                    },
                )
                param_settings = build_modelica_parameter_map(job_params)
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

                # Calculate Summary Metrics
                metrics_definition = config.get("metrics_definition", {})

                # Pre-filter check before metric calc
                skip_metrics = False
                if filter_schema and os.path.exists(result_file_path):
                    try:
                        temp_df = pd.read_csv(result_file_path)
                        violations = find_filter_schema_violations(
                            temp_df, filter_schema
                        )
                        if violations:
                            skip_metrics = True
                            logger.info(
                                f"Job {i+1} matched filter_schema. Skipping metric calculations before HDF5 append."
                            )
                    except Exception:
                        pass

                if (
                    not skip_metrics
                    and metrics_definition
                    and os.path.exists(result_file_path)
                ):
                    try:
                        df_metric = pd.read_csv(result_file_path)
                        single_job_metrics = calculate_single_job_metrics(
                            df_metric, metrics_definition
                        )

                        if single_job_metrics:
                            # Save to JSON in the job workspace
                            metrics_file_path = os.path.join(
                                job_workspace, "job_metrics.json"
                            )
                            with open(metrics_file_path, "w") as f:
                                json.dump(single_job_metrics, f, indent=4)

                            logger.info(
                                "Calculated and saved job metrics",
                                extra={
                                    "job_index": i + 1,
                                    "metrics": single_job_metrics,
                                },
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to calculate metrics for job {i+1}: {e}"
                        )

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


def _process_h5_result(
    store: pd.HDFStore,
    job_id: int,
    params: dict,
    res_path: str,
    config: dict,
    metrics_definition: dict = None,
    filter_schema: List[Dict[str, Any]] | None = None,
):
    """Helper to process simulation result into HDF5 store."""
    if not res_path or not os.path.exists(res_path):
        return

    try:
        df = pd.read_csv(res_path)

        if filter_schema:
            violations = find_filter_schema_violations(df, filter_schema)
            if violations:
                violation_messages = ", ".join(
                    [
                        f"{item['column']} {item['kind']} {item['threshold']} (observed={item['observed']})"
                        for item in violations
                    ]
                )
                logger.warning(
                    f"Result for job {job_id} filtered out by filter_schema. Violations: {violation_messages}"
                )
                # Cleanup immediately to save disk space
                job_dir = os.path.dirname(res_path)
                keep_temp = config.get("simulation", {}).get("keep_temp_files", False)
                if (
                    not keep_temp
                    and os.path.exists(job_dir)
                    and _should_cleanup_job_dir(job_dir, config)
                    and config.get("co_simulation", {}).get("engine") != "online_oms"
                ):
                    _cleanup_job_dir(job_dir, job_id)
                return

        df["job_id"] = job_id

        # Force all numeric columns (except job_id) to float to avoid HDF5 schema conflicts
        for col in df.columns:
            if col != "job_id" and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(float)

        # Use 'append' with data_columns=["job_id"] for queryability to avoid 512 column limit
        store.append("results", df, index=False, data_columns=["job_id"])

        # Calculate and Save Summary Metrics
        if metrics_definition:
            try:
                # Reuse df since we already have it
                single_job_metrics = calculate_single_job_metrics(
                    df, metrics_definition
                )

                if single_job_metrics:
                    summary_df = build_single_job_summary_df(
                        job_id, single_job_metrics, metrics_definition
                    )
                    if not summary_df.empty:
                        # Append to HDF5
                        store.append(
                            "summary",
                            summary_df,
                            index=False,
                            data_columns=True,
                        )

            except Exception as e:
                logger.warning(
                    f"Failed to calculate/save summary metrics for job {job_id} in simulation.py: {e}"
                )

        param_df = pd.DataFrame([params])
        param_df["id"] = job_id

        # Force all numeric columns (except id) to float to avoid HDF5 schema conflicts
        for col in param_df.columns:
            if col != "id" and pd.api.types.is_numeric_dtype(param_df[col]):
                param_df[col] = param_df[col].astype(float)

        # Force object dtype only for string/object columns to avoid HDF5 issues with StringDtype
        for col in param_df.select_dtypes(include=["object", "string"]).columns:
            param_df[col] = param_df[col].astype(object)

        store.append("jobs", param_df, index=False, data_columns=True)

        # Cleanup immediately to save disk space
        job_dir = os.path.dirname(res_path)
        keep_temp = config.get("simulation", {}).get("keep_temp_files", False)
        if (
            not keep_temp
            and os.path.exists(job_dir)
            and _should_cleanup_job_dir(job_dir, config)
            and config.get("co_simulation", {}).get("engine") != "online_oms"
        ):
            _cleanup_job_dir(job_dir, job_id)
    except Exception as e:
        logger.error(f"Failed to process HDF5 result for job {job_id}: {e}")


def _should_cleanup_job_dir(job_dir: str, config: dict) -> bool:
    temp_dir = os.path.abspath(config["paths"].get("temp_dir", "temp"))
    normalized_job_dir = os.path.abspath(job_dir)

    if "job_" not in os.path.basename(normalized_job_dir):
        return False

    try:
        return os.path.commonpath([normalized_job_dir, temp_dir]) == temp_dir
    except ValueError:
        return False


def _cleanup_job_dir(job_dir: str, job_id: int, retries: int = 8, delay: float = 0.5):
    """Remove a job workspace with retries for transient Windows file locks."""

    def _on_rm_error(func, path, exc_info):
        try:
            os.chmod(path, 0o666)
            func(path)
        except Exception:
            pass

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(job_dir, onerror=_on_rm_error)
            logger.info(
                "Cleaned up job workspace after HDF5 processing",
                extra={"job_id": job_id, "workspace": job_dir, "attempt": attempt},
            )
            return
        except FileNotFoundError:
            return
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay)

    logger.warning(
        "Failed to cleanup job workspace after retries",
        extra={"job_id": job_id, "workspace": job_dir, "error": str(last_error)},
    )


def export_results_to_csv(results_dir: str, hdf_path: str):
    """Exports results from HDF5 to legacy CSV formats."""
    logger.info(f"Exporting results from {hdf_path} to CSV...")
    try:
        # Export simulation_result.csv (Time-Job Matrix)
        # Needs to pivot: Time vs [Var&Params] for each Job
        # This is memory intensive and was the reason for HDF5, but we do it if requested.

        with pd.HDFStore(hdf_path, mode="r") as store:
            if "/results" not in store.keys():
                logger.warning("No results found in HDF5 to export.")
                return

            # Read all results (chunking could be better but sticking to simple pivot for now)
            # To avoid huge memory usage, we might want to iterate jobs if possible,
            # but standard pivot requires all data.
            # Let's read full table.
            df_results = store.select("results")

            # Read jobs to get parameters
            if "/jobs" in store.keys():
                df_jobs = store.select("jobs")
            else:
                df_jobs = pd.DataFrame()

            # Read summary for export
            if "/summary" in store.keys():
                df_summary = store.select("summary")

                # Join with jobs table to include parameters in the summary CSV
                if not df_jobs.empty:
                    df_summary = pd.merge(df_jobs, df_summary, on="job_id", how="left")

                summary_csv_path = get_unique_filename(
                    results_dir, "summary_metrics.csv"
                )
                df_summary.to_csv(summary_csv_path, index=False)
                logger.info(f"Exported summary metrics to {summary_csv_path}")

        # Pivot Logic for simulation_result.csv
        # Pivot columns: time, values... where columns need to be renamed with params.

        # 1. Join params to results if needed, or just build column names.
        # We need to reconstruct the "Var&Param=Val" column format.

        job_params_map = {}
        if not df_jobs.empty:
            for _, row in df_jobs.iterrows():
                job_id = row["job_id"]
                params = row.drop("job_id").to_dict()
                # filter nulls
                params = {k: v for k, v in params.items() if pd.notna(v)}
                param_str = "&".join([f"{k}={v}" for k, v in params.items()])
                job_params_map[job_id] = param_str

        # 2. Pivot
        # df_results has: time, var1, var2, ..., job_id
        # We want: time, var1&params...

        # Group by job_id to process separate dataframes and then concat (like original logic)
        all_dfs = []
        time_df_added = False

        job_ids = df_results["job_id"].unique()
        job_ids.sort()

        for job_id in job_ids:
            job_df = df_results[df_results["job_id"] == job_id].copy()
            job_df.sort_values("time", inplace=True)

            if not time_df_added:
                all_dfs.append(job_df[["time"]].reset_index(drop=True))
                time_df_added = True

            param_string = job_params_map.get(job_id, "")
            data_cols = job_df.drop(columns=["time", "job_id"], errors="ignore")

            rename_map = {
                col: f"{col}&{param_string}" if param_string else col
                for col in data_cols.columns
            }
            all_dfs.append(data_cols.rename(columns=rename_map).reset_index(drop=True))

        if all_dfs:
            combined_df = pd.concat(all_dfs, axis=1)
            # Cleanup rows with empty time (though check above should handle it)
            combined_df.dropna(subset=["time"], inplace=True)

            csv_path = get_unique_filename(
                results_dir,
                "simulation_result.csv" if len(job_ids) == 1 else "sweep_results.csv",
            )
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"Exported simulation results to {csv_path}")

    except Exception as e:
        logger.error(f"Failed to export CSVs: {e}", exc_info=True)


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
    passing the HDF5 results path and other parameters to it.

    Args:
        config: The main configuration dictionary.
        results_df: Deprecated legacy argument retained for call compatibility.
        post_processing_output_dir: The directory to save any output from the tasks.
        results_file_path: Path to the HDF5 results file.

    Note:
        Supports two loading methods: 'script_path' for direct .py files, or 'module'
        for installed packages. Creates output_dir if it doesn't exist. Passes
        results_file_path, output_dir, and user-specified params to each task function.
        Logs errors for failed tasks but continues with remaining tasks.
    """
    post_processing_configs = config.get("post_processing")
    if not post_processing_configs:
        logger.info("No post-processing task configured, skipping this step.")
        return
    if isinstance(post_processing_configs, dict):
        post_processing_configs = post_processing_configs.get("tasks", [])
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

    if not results_file_path:
        logger.error("Post-processing requires a valid HDF5 results file path.")
        return

    for i, task_config in enumerate(post_processing_configs):
        try:
            function_name = task_config["function"]
            params = task_config.get("params", {})
            module = None

            if "llm_env" not in params and isinstance(config.get("llm_env"), dict):
                params["llm_env"] = config["llm_env"]

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
                    results_file_path=results_file_path,
                    output_dir=post_processing_dir,
                    **params,
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

        try:
            exe_path, xml_path = _resolve_built_model_paths(build_result, build_dir)
        except RuntimeError as exc:
            err = omc.sendExpression("getErrorString()")
            raise RuntimeError(f"Model build failed: {err or exc}") from exc

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

    cmd = [
        run_exe_path,
        "-r",
        result_filename,
        "-outputFormat=csv",
        f"-stopTime={sim_config['stop_time']}",
        f"-stepSize={sim_config['step_size']}",
    ]

    if override_pairs:
        override_str = ",".join(override_pairs)
        cmd.extend(["-override", override_str])
    if variable_filter:
        cmd.append(f"-variableFilter={variable_filter}")

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


def _mp_run_co_simulation_job_wrapper(args):
    """Wrapper for multiprocessing.Pool to map run_co_simulation_job."""
    config, job_params, job_id = args
    try:
        res = run_co_simulation_job(config, job_params, job_id=job_id)
        # res is result_path (str)
        return job_id, job_params, res, None
    except Exception as e:
        return job_id, job_params, None, str(e)


def run_simulation(config: Dict[str, Any], export_csv: bool = False) -> None:
    """Orchestrates the main simulation workflow.

    Simplified Mode Logic (Unified HDF5 Storage):
    1. Concurrent Mode (concurrent=True):
       - Uses "Enhanced" execution (Compile Once, Run Many).
       - Streams results directly to HDF5.

    2. Sequential Mode (concurrent=False):
       - Uses "Standard" execution (Reuse OMPython Session).
       - Also streams results directly to HDF5.

    Args:
        config: The main configuration dictionary for the run.
        export_csv: Whether to export results to CSV files at the end.
    """
    jobs = generate_simulation_jobs(config.get("simulation_parameters", {}))

    try:
        results_dir = os.path.abspath(config["paths"]["results_dir"])
        temp_dir = os.path.abspath(config["paths"].get("temp_dir", "temp"))
    except KeyError as e:
        logger.error(f"Missing required path key in configuration file: {e}")
        sys.exit(1)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    sim_config = config["simulation"]
    foc_config = config.get("foc") or {}
    foc_path = foc_config.get("foc_path")
    foc_component = foc_config.get("foc_component")

    if foc_path:
        foc_workspace = os.path.join(temp_dir, "foc_prepared")
        foc_result = prepare_foc_simulation_package(
            config["paths"]["package_path"],
            sim_config["model_name"],
            foc_path,
            foc_workspace,
            strategy="table",
            foc_component=foc_component,
        )
        config["paths"]["package_path"] = foc_result["package_path"]
        if sim_config["stop_time"] < foc_result["schedule_duration"]:
            logger.warning(
                "Configured stop_time truncates the FOC schedule",
                extra={
                    "stop_time": sim_config["stop_time"],
                    "foc_duration": foc_result["schedule_duration"],
                    "foc_path": foc_path,
                    "foc_component": foc_component,
                },
            )

    use_concurrent = sim_config.get("concurrent", False)
    maximize_workers = sim_config.get("maximize_workers", False)
    max_workers = get_safe_max_workers(
        sim_config.get("max_workers"),
        maximize=maximize_workers,
        task_count=len(jobs),
    )
    is_co_sim = config.get("co_simulation") is not None
    metrics_definition = config.get("metrics_definition", {})
    filter_schema = config.get("filter_schema")

    # HDF5 Setup (Unified)
    hdf_filename = "sweep_results.h5"
    hdf_path = get_unique_filename(results_dir, hdf_filename)

    logger.info(
        f"Starting Simulation (Concurrent={use_concurrent}, CoSim={is_co_sim}). Storage: {hdf_path}"
    )

    with pd.HDFStore(hdf_path, mode="w", complib="blosc", complevel=9) as store:
        # Save configuration
        try:
            config_df = pd.DataFrame({"config_json": [json.dumps(config)]})
            config_df = config_df.astype(object)
            store.put("config", config_df, format="fixed")
        except Exception as e:
            logger.warning(f"Failed to save config to HDF5: {e}")

        # --- Concurrent Mode ---
        if use_concurrent:
            from tricys.utils.log_capture import LogCapture

            with LogCapture() as log_handler:
                pool_args = []
                wrapper_func = None

                if not is_co_sim:
                    # 1. Compile Model Once
                    master_exe, master_xml, om_bin = _build_model_only(config)
                    for i, job_params in enumerate(jobs):
                        kwargs = {
                            "exe_source": master_exe,
                            "xml_source": master_xml,
                            "om_bin_path": om_bin,
                            "base_temp_dir": temp_dir,
                            "sim_config": sim_config,
                            "variable_filter": sim_config.get("variableFilter"),
                            "inplace_execution": True,
                        }
                        pool_args.append((job_params, i + 1, kwargs))
                    wrapper_func = _mp_run_fast_subprocess_job_wrapper
                else:
                    for i, job_params in enumerate(jobs):
                        pool_args.append((config, job_params, i + 1))
                    wrapper_func = _mp_run_co_simulation_job_wrapper

                # Execute Pool
                completed_count = 0
                total_jobs = len(jobs)
                with multiprocessing.Pool(processes=max_workers) as pool:
                    for job_id, job_p, result_path, error in pool.imap_unordered(
                        wrapper_func, pool_args
                    ):
                        completed_count += 1
                        logger.info(f"Job {completed_count} of {total_jobs}")
                        if error:
                            logger.error(f"Job {job_id} failed: {error}")
                        else:
                            _process_h5_result(
                                store,
                                job_id,
                                job_p,
                                result_path,
                                config,
                                metrics_definition,
                                filter_schema=filter_schema,
                            )

                try:
                    logs_json = log_handler.to_json()
                    log_df = pd.DataFrame({"log": [logs_json]})
                    log_df = log_df.astype(object)
                    store.put("log", log_df, format="fixed")
                except Exception as e:
                    logger.warning(f"Failed to save logs to HDF5: {e}")

        # --- Sequential Mode ---
        else:
            from tricys.utils.log_capture import LogCapture

            with LogCapture() as log_handler:
                if is_co_sim:
                    for i, job_params in enumerate(jobs):
                        job_id = i + 1
                        logger.info(f"Running job {job_id}/{len(jobs)}")
                        try:
                            result_path = run_co_simulation_job(
                                config, job_params, job_id=job_id
                            )
                            _process_h5_result(
                                store,
                                job_id,
                                job_params,
                                result_path,
                                config,
                                metrics_definition,
                                filter_schema=filter_schema,
                            )
                        except Exception as e:
                            logger.error(f"Job {job_id} failed: {e}")
                else:
                    # Standard Sequential using callback to stream to HDF5
                    logger.info("Running sequential sweep with HDF5 streaming.")

                    def h5_callback(idx, params, res_path):
                        _process_h5_result(
                            store,
                            idx + 1,
                            params,
                            res_path,
                            config,
                            metrics_definition,
                            filter_schema=filter_schema,
                        )

                    run_sequential_sweep(
                        config,
                        jobs,
                        post_job_callback=h5_callback,
                        filter_schema=filter_schema,
                    )

                try:
                    logs_json = log_handler.to_json()
                    log_df = pd.DataFrame({"log": [logs_json]})
                    log_df = log_df.astype(object)
                    store.put("log", log_df, format="fixed")
                except Exception as e:
                    logger.warning(f"Failed to save logs to HDF5: {e}")

    # Export CSV if requested
    if export_csv:
        export_results_to_csv(results_dir, hdf_path)

    # Post-processing (Unified HDF5)
    post_processing_output_dir = os.path.join(results_dir, "post_processing")
    run_post_processing(
        config, None, post_processing_output_dir, results_file_path=hdf_path
    )

    # Auditor Phase
    auditor_config_dict = config.get("auditor", {})
    if auditor_config_dict.get("enabled", False):
        logger.info("Starting offline mass balance audit...")
        try:
            from tricys.auditor.offline import (
                parse_auditor_config,
                perform_offline_audit,
            )

            auditor_cfg = parse_auditor_config(config)
            audit_result = perform_offline_audit(hdf_path, auditor_cfg)

            report_path = os.path.join(results_dir, "auditor_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(audit_result, f, indent=4)

            if audit_result.get("status") == "success":
                err_val = audit_result.get("mass_balance_error", 0.0)
                logger.info(f"Mass balance audit completed. Error: {err_val:.6e} g")
                logger.info(f"Auditor report saved to {report_path}")
            else:
                logger.warning(f"Auditor returned error: {audit_result.get('error')}")

        except Exception as e:
            logger.error(f"Failed to run offline auditor: {e}", exc_info=True)

    # Cleanup
    if not sim_config.get("keep_temp_files", True):
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                os.makedirs(temp_dir)
                logger.info("Cleaned up temp directory.")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")


def main(
    config_or_path: Union[str, Dict[str, Any]],
    base_dir: str = None,
    export_csv: bool = False,
) -> None:
    """Main entry point for the simulation runner.

    This function prepares the configuration, sets up logging, and invokes
    the main `run_simulation` orchestrator.

    Args:
        config_or_path: The path to the JSON configuration file OR a config dict.
        base_dir: Optional base directory for resolving relative paths if a dict is passed.
        export_csv: Whether to export results to CSV after HDF5 storage.
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
        run_simulation(config, export_csv=export_csv)
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
