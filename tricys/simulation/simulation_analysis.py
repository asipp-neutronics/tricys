import concurrent.futures
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from OMPython import ModelicaSystem

from tricys.analysis.metric import (
    calculate_doubling_time,
    calculate_startup_inventory,
    extract_metrics,
    time_of_turning_point,
)
from tricys.analysis.plot import generate_analysis_plots, plot_sweep_time_series
from tricys.analysis.report import (
    consolidate_reports,
    generate_analysis_cases_summary,
    retry_ai_analysis,
)
from tricys.analysis.salib import run_salib_analysis
from tricys.core.interceptor import integrate_interceptor_model
from tricys.core.jobs import generate_simulation_jobs
from tricys.core.modelica import (
    format_parameter_value,
    get_om_session,
    load_modelica_package,
)
from tricys.utils.config_utils import (
    analysis_prepare_config,
    analysis_setup_analysis_cases_workspaces,
    analysis_validate_config,
)
from tricys.utils.file_utils import get_unique_filename
from tricys.utils.log_utils import (
    restore_configs_from_log,
    setup_logging,
)

# Standard logger setup
logger = logging.getLogger(__name__)


def _get_optimization_tasks(config: dict) -> List[str]:
    """Identifies all valid optimization tasks from the configuration.

    A valid optimization task is a dependent variable that starts with "Required_",
    is defined in metrics_definition, and has all the necessary fields for bisection search.

    Args:
        config: The configuration dictionary.

    Returns:
        A list of valid optimization metric names.

    Note:
        Required fields for bisection search: method, parameter_to_optimize, search_range,
        tolerance, max_iterations. Only checks dependent_variables in analysis_case.
    """
    optimization_tasks = []
    sensitivity_analysis = config.get("sensitivity_analysis", {})
    metrics_definition = sensitivity_analysis.get("metrics_definition", {})
    analysis_case = sensitivity_analysis.get("analysis_case", {})
    dependent_vars = analysis_case.get("dependent_variables", [])
    for var in dependent_vars:
        if var.startswith("Required_") and var not in optimization_tasks:
            if var in metrics_definition:
                required_config = metrics_definition[var]
                required_fields = [
                    "method",
                    "parameter_to_optimize",
                    "search_range",
                    "tolerance",
                    "max_iterations",
                ]
                if all(field in required_config for field in required_fields):
                    optimization_tasks.append(var)
    return optimization_tasks


def _run_sensitivity_analysis(
    config: Dict[str, Any], run_results_dir: str, jobs: List[Dict[str, Any]]
) -> None:
    """Executes the sensitivity analysis workflow.

    This function orchestrates the post-simulation analysis. It checks if
    sensitivity analysis is enabled, loads the merged simulation results,
    extracts summary metrics, merges any optimization results, saves the
    final summary data to a CSV, and generates analysis plots.

    Args:
        config: The configuration dictionary for the run.
        run_results_dir: The directory where results are stored and will be saved.
        jobs: The list of simulation jobs that were executed.

    Note:
        Only runs if sensitivity_analysis.enabled is True. Loads sweep_results.csv,
        extracts metrics using metrics_definition, merges optimization results if
        present, and generates plots. Saves summary to summary_metrics.csv.
    """
    if not config.get("sensitivity_analysis", {}).get("enabled", False):
        return

    logger.info("Starting automated sensitivity analysis.")

    try:
        # Check if there are result data files
        combined_csv_path = os.path.join(run_results_dir, "sweep_results.csv")
        single_result_path = os.path.join(run_results_dir, "simulation_result.csv")

        # Determine result file path based on number of jobs
        if len(jobs) == 1 and os.path.exists(single_result_path):
            # Single task case
            results_df = pd.read_csv(single_result_path)
            logger.info(f"Loading single task result from: {single_result_path}")
        elif len(jobs) > 1 and os.path.exists(combined_csv_path):
            # Multi-task case
            results_df = pd.read_csv(combined_csv_path)
            logger.info(f"Loading sweep results from: {combined_csv_path}")
        else:
            logger.warning("No result data file found for sensitivity analysis.")
            return

        # Get analysis_case configuration
        analysis_config = config["sensitivity_analysis"]
        analysis_case = analysis_config["analysis_case"]

        if analysis_case is None:
            logger.warning("No valid analysis_case found for sensitivity analysis.")
            return

        # Extract summary metrics
        summary_df = extract_metrics(
            results_df,
            analysis_config["metrics_definition"],
            analysis_case,
        )

        optimization_tasks = _get_optimization_tasks(config)

        if summary_df.empty and not optimization_tasks:
            logger.warning("Sensitivity analysis did not produce any summary data.")
            return
        elif not summary_df.empty and not optimization_tasks:
            df_to_save = summary_df
        elif summary_df.empty and optimization_tasks:
            optimization_summary_path = os.path.join(
                run_results_dir, "requierd_tbr_summary.csv"
            )
            if not os.path.exists(optimization_summary_path):
                logger.warning(
                    "Optimization summary not found, saving analysis summary only."
                )
            df_to_save = pd.read_csv(optimization_summary_path)
        elif not summary_df.empty and optimization_tasks:
            optimization_summary_path = os.path.join(
                run_results_dir, "requierd_tbr_summary.csv"
            )
            if not os.path.exists(optimization_summary_path):
                logger.warning(
                    "Optimization summary not found, saving analysis summary only."
                )
                df_to_save = summary_df
            else:
                try:
                    optimization_df = pd.read_csv(optimization_summary_path)
                    sweep_params = list(jobs[0].keys())
                    merged_df = pd.merge(
                        summary_df,
                        optimization_df,
                        on=sweep_params,
                        how="outer",
                    )
                    logger.info(
                        "Merged optimization and sensitivity analysis summaries."
                    )
                    df_to_save = merged_df
                except Exception as e:
                    logger.error(
                        f"Failed to merge summaries: {e}. Saving analysis summary only."
                    )

        # Save summary data
        summary_csv_path = get_unique_filename(
            run_results_dir, "sensitivity_analysis_summary.csv"
        )
        df_to_save.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Sensitivity analysis summary saved to: {summary_csv_path}")

        # Generate analysis charts
        unit_map = analysis_config.get("unit_map", {})
        glossary_path = analysis_config.get("glossary_path", "")
        generate_analysis_plots(
            df_to_save,
            analysis_case,
            run_results_dir,
            unit_map=unit_map,
            glossary_path=glossary_path,
        )

    except Exception as e:
        logger.error(f"Automated sensitivity analysis failed: {e}", exc_info=True)


def _run_bisection_search_for_job(
    config: Dict[str, Any], job_id_prefix: str, optimization_metric_name: str
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Performs a bisection search to find an optimal parameter value.

    This function reads optimization parameters from the configuration and uses a
    bisection search (binary search) algorithm to find the value of a specified
    parameter that causes a target metric (e.g., 'Self_Sufficiency_Time') to
    fall below a given threshold. It supports searching for multiple threshold
    values if `metric_max_value` is a list.

    Args:
        config: The configuration for the specific job.
        job_id_prefix: A prefix for creating unique IDs for sub-tasks.
        optimization_metric_name: The name of the 'Required_***' metric that defines
            the optimization task.

    Returns:
        A tuple containing two dictionaries:
        - The first maps the required metric name(s) to the found optimal parameter value(s).
        - The second maps the resulting metric name(s) to the metric value(s) achieved
          with the optimal parameter.

    Note:
        Uses binary search algorithm with configurable tolerance and max_iterations.
        Reuses single OMPython session for all search iterations. Supports multiple
        threshold values via metric_max_value list. Falls back to stop_time if
        metric_max_value not specified.
    """
    # Read the specific optimization configuration from sensitivity_analysis
    sensitivity_analysis = config.get("sensitivity_analysis", {})
    metrics_definition = sensitivity_analysis.get("metrics_definition", {})
    optimization_config = metrics_definition.get(optimization_metric_name, {})

    if not optimization_config or "parameter_to_optimize" not in optimization_config:
        raise ValueError(
            f"Optimization config for '{optimization_metric_name}' not found or missing 'parameter_to_optimize'"
        )

    sim_config = config["simulation"]
    paths_config = config["paths"]

    param_to_optimize = optimization_config["parameter_to_optimize"]
    low_orig, high_orig = optimization_config["search_range"]
    tolerance = optimization_config.get("tolerance", 0.001)
    max_iterations = optimization_config.get("max_iterations", 10)
    stop_time = sim_config["stop_time"]
    # Get the maximum value of the metric, default to stop_time
    metric_max_value = optimization_config.get("metric_max_value", stop_time)

    # Get metric configuration, with defaults for backward compatibility
    metric_name = optimization_config.get("metric_name", "Self_Sufficiency_Time")
    default_source_column = "sds.inventory"
    source_column = optimization_config.get("source_column", default_source_column)

    if not source_column:
        raise ValueError(
            f"Missing 'source_column' in {optimization_metric_name} config for metric '{metric_name}'"
        )

    metric_max_values = (
        metric_max_value if isinstance(metric_max_value, list) else [metric_max_value]
    )
    is_list_input = isinstance(metric_max_value, list)

    all_optimal_params = {}
    all_optimal_values = {}

    # Setup for reusing the model object
    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    os.makedirs(base_temp_dir, exist_ok=True)

    omc = None
    try:
        omc = get_om_session()
        package_path = os.path.abspath(paths_config["package_path"])
        if not load_modelica_package(omc, Path(package_path).as_posix()):
            raise RuntimeError("Failed to load Modelica package for bisection search.")

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

        for current_metric_max_value in metric_max_values:
            low, high = low_orig, high_orig
            logger.info(
                "Starting bisection search",
                extra={
                    "param_to_optimize": param_to_optimize,
                    "search_range": [low, high],
                    "target_metric": metric_name,
                    "target_value": f"< {current_metric_max_value}",
                },
            )

            best_successful_param = float("inf")
            best_successful_value = float("inf")

            for i in range(max_iterations):
                if high - low < tolerance:
                    logger.info(
                        f"Search converged for {job_id_prefix}. Tolerance {tolerance} reached."
                    )
                    break

                mid_param = (low + high) / 2

                logger.info(
                    "Bisection search iteration",
                    extra={
                        "job_id_prefix": job_id_prefix,
                        "iteration": f"{i+1}/{max_iterations}",
                        "param_tested": param_to_optimize,
                        "param_value": f"{mid_param:.4f}",
                    },
                )

                job_params = config.get("simulation_parameters", {}).copy()
                job_params[param_to_optimize] = mid_param

                # Set parameters on the existing mod object
                param_settings = [
                    format_parameter_value(name, value)
                    for name, value in job_params.items()
                ]
                if param_settings:
                    mod.setParameters(param_settings)

                # Create a workspace for this iteration's results
                iter_job_id_str = f"iter{i}_{mid_param}"
                iter_temp_dir = os.path.join(base_temp_dir, job_id_prefix)
                os.makedirs(iter_temp_dir, exist_ok=True)
                job_workspace = os.path.join(iter_temp_dir, f"{iter_job_id_str}")
                os.makedirs(job_workspace, exist_ok=True)
                result_filename = f"{iter_job_id_str}_simulation_results.csv"
                result_path = os.path.join(job_workspace, result_filename)

                # Simulate
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
                            f"Failed to clean result file {result_path}: {e}"
                        )

                metric_value = float("inf")

                if not os.path.exists(result_path):
                    logger.error(
                        f"Analysis failed for params {job_params}: Simulation did not produce a result file."
                    )
                else:
                    try:
                        results_df = pd.read_csv(result_path)
                        if source_column not in results_df.columns:
                            logger.error(
                                f"Analysis failed: source column '{source_column}' not found in results."
                            )
                        else:
                            if metric_name == "Self_Sufficiency_Time":
                                metric_value = time_of_turning_point(
                                    results_df[source_column],
                                    results_df["time"],
                                )
                            elif metric_name == "Doubling_Time":
                                metric_value = calculate_doubling_time(
                                    results_df[source_column],
                                    results_df["time"],
                                )
                            elif metric_name == "Startup_Inventory":
                                metric_value = calculate_startup_inventory(
                                    results_df[source_column]
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported metric_name for bisection search: {metric_name}"
                                )
                            logger.info(
                                "Bisection analysis successful",
                                extra={
                                    "job_params": job_params,
                                    "metric_name": metric_name,
                                    "metric_value": metric_value,
                                },
                            )
                    except Exception as e:
                        logger.error(
                            f"Analysis failed for params {job_params} due to an exception: {e}",
                            exc_info=True,
                        )

                # A successful search requires the turning point to be found before both stop_time and the specified metric_max_value
                if (
                    metric_value < min(stop_time, current_metric_max_value)
                    and metric_value != np.nan
                ):
                    best_successful_param = mid_param
                    best_successful_value = metric_value
                    high = mid_param
                else:
                    low = mid_param

            if best_successful_param == float("inf"):
                logger.warning(
                    f"Bisection search for {job_id_prefix} with target < {current_metric_max_value} did not find a successful parameter."
                )
            else:
                logger.info(
                    "Bisection search finished",
                    extra={
                        "job_id_prefix": job_id_prefix,
                        "target_value": f"< {current_metric_max_value}",
                        "optimal_param": f"{best_successful_param:.4f}",
                    },
                )

            # Dynamically create the key for the resulting optimal value to ensure uniqueness.
            value_key_base = f"{metric_name}_for_{optimization_metric_name}"

            if is_list_input:
                value = current_metric_max_value
                if value >= 365 * 24 / 2:
                    unit_str = f"{value / (365 * 24):.2f} year"
                elif value >= 24:
                    unit_str = f"{value / 24:.2f} day"
                else:
                    unit_str = f"{value} h"
                param_key = f"{optimization_metric_name}({unit_str})"
                value_key = f"{value_key_base}({unit_str})"
            else:
                param_key = optimization_metric_name
                value_key = value_key_base

            all_optimal_params[param_key] = best_successful_param
            all_optimal_values[value_key] = best_successful_value

    except Exception as e:
        logger.error(
            f"Bisection search failed during setup or execution: {e}", exc_info=True
        )
    finally:
        if omc:
            omc.sendExpression("quit()")

    return all_optimal_params, all_optimal_values


def _run_co_simulation(
    config: dict, job_params: dict, job_id: int = 0
) -> tuple[Dict[str, float], Dict[str, float], str]:
    """Runs a full co-simulation workflow and any subsequent optimizations.

    This function orchestrates a complete co-simulation job in an isolated,
    thread-safe directory. After the co-simulation completes, it checks if
    any bisection search optimizations are configured and runs them for the
    completed job.

    Args:
        config (dict): The main configuration dictionary.
        job_params (dict): A dictionary of parameters specific to this job.
        job_id (int): A unique identifier for the job.

    Returns:
        A tuple containing:
        - A dictionary of optimal parameter values found during optimization.
        - A dictionary of the metric values achieved with those optimal parameters.
        - The path to the final co-simulation result file, or an empty string
          on failure.
    """
    paths_config = config["paths"]
    sim_config = config["simulation"]

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))

    job_workspace = os.path.join(base_temp_dir, f"job_{job_id}")
    os.makedirs(job_workspace, exist_ok=True)

    omc = None
    optimal_param = {}
    optimal_value = {}

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
            logger.info(f"Copied single-file package to: {isolated_package_path}")
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

            logger.info(f"Copied multi-file package to: {isolated_package_dir}")
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
                                f"Copied asset directory '{original_asset_dir}' to '{dest_dir}' for job {job_id}"
                            )

                        # Update the path in the config to point to the new location
                        new_asset_path = dest_dir / original_asset_path.name
                        handler_config["params"][param_key] = new_asset_path.as_posix()
                        logger.info(
                            f"Updated parameter '{param_key}' for job {job_id} to '{handler_config['params'][param_key]}'"
                        )

        all_input_vars = []
        for handler_config in handlers:
            submodel_name = handler_config["submodel_name"]
            instance_name = handler_config["instance_name"]
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

        # Clean up the simulation result file
        if os.path.exists(primary_result_filename):
            try:
                df = pd.read_csv(primary_result_filename)
                df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                df.dropna(subset=["time"], inplace=True)
                df.to_csv(primary_result_filename, index=False)
            except Exception as e:
                logger.warning(
                    f"Failed to clean result file {primary_result_filename}: {e}"
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
        logger.info("Proceeding with Final simulation.")

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
                    f"Failed to clean result file {default_result_path}: {e}"
                )

        if not os.path.exists(default_result_path):
            raise FileNotFoundError(
                f"Simulation for job {job_id} failed to produce a result file at {default_result_path}"
            )

        optimal_param = {}
        optimal_value = {}
        # Check if optimization is enabled, if so call _run_bisection_search_for_job for each task
        optimization_tasks = _get_optimization_tasks(config)
        for optimization_metric_name in optimization_tasks:
            logger.info(
                f"Job {job_id}: Starting optimization for metric '{optimization_metric_name}'."
            )
            job_config = config.copy()
            job_config["paths"]["package_path"] = isolated_package_path
            job_config["paths"]["temp_dir"] = base_temp_dir
            job_config["simulation_parameters"] = job_params
            job_config["simulation"]["model_name"] = final_model_name

            # Use a unique prefix for each metric to avoid workspace collision
            metric_job_id_prefix = f"job_{job_id}_{optimization_metric_name}"

            (
                current_optimal_param,
                current_optimal_value,
            ) = _run_bisection_search_for_job(
                job_config,
                job_id_prefix=metric_job_id_prefix,
                optimization_metric_name=optimization_metric_name,
            )

            # Merge results from the current optimization task
            optimal_param.update(current_optimal_param)
            optimal_value.update(current_optimal_value)

            logger.info(
                f"Job {job_id} optimization for '{optimization_metric_name}' complete. "
                f"Optimal params: {current_optimal_param}, Optimal values: {current_optimal_value}"
            )

        # Return the path to the result file inside the temporary workspace
        return optimal_param, optimal_value, Path(default_result_path).as_posix()
    except Exception:
        logger.error(
            "Co-simulation workflow failed", exc_info=True, extra={"job_id": job_id}
        )
        return optimal_param, optimal_value, ""
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


def _run_single_job(
    config: dict, job_params: dict, job_id: int = 0
) -> tuple[Dict[str, float], Dict[str, float], str]:
    """Executes a single simulation job and any subsequent optimizations.

    This function runs a standard simulation in an isolated workspace. After the
    simulation completes, it checks if any bisection search optimizations are
    configured and runs them for the completed job.

    Args:
        config (dict): The main configuration dictionary.
        job_params (dict): A dictionary of parameters specific to this job.
        job_id (int): A unique identifier for the job.

    Returns:
        A tuple containing:
        - A dictionary of optimal parameter values found during optimization.
        - A dictionary of the metric values achieved with those optimal parameters.
        - The path to the simulation result file, or an empty string on failure.
    """
    paths_config = config["paths"]
    sim_config = config["simulation"]

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    job_workspace = os.path.join(base_temp_dir, f"job_{job_id}")
    os.makedirs(job_workspace, exist_ok=True)

    logger.info(
        "Starting single job",
        extra={"job_id": job_id, "job_params": job_params},
    )
    omc = None
    optimal_param = {}
    optimal_value = {}
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
        if result_path.is_file():
            try:
                df = pd.read_csv(result_path)
                df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                df.dropna(subset=["time"], inplace=True)
                df.to_csv(result_path, index=False)
            except Exception as e:
                logger.warning(f"Failed to clean result file {result_path}: {e}")

        if not result_path.is_file():
            raise FileNotFoundError(
                f"Simulation for job {job_id} failed to produce result file at {result_path}"
            )

        logger.info(
            "Job finished successfully",
            extra={"job_id": job_id, "result_path": str(result_path)},
        )

        optimal_param = {}
        optimal_value = {}
        # Check if optimization is enabled, if so call _run_bisection_search_for_job for each task
        optimization_tasks = _get_optimization_tasks(config)
        for optimization_metric_name in optimization_tasks:
            logger.info(
                f"Job {job_id}: Starting optimization for metric '{optimization_metric_name}'."
            )
            job_config = config.copy()
            job_config["paths"]["package_path"] = package_path
            job_config["paths"]["temp_dir"] = base_temp_dir
            job_config["simulation_parameters"] = job_params

            # Use a unique prefix for each metric to avoid workspace collision
            metric_job_id_prefix = f"job_{job_id}_{optimization_metric_name}"

            (
                current_optimal_param,
                current_optimal_value,
            ) = _run_bisection_search_for_job(
                job_config,
                job_id_prefix=metric_job_id_prefix,
                optimization_metric_name=optimization_metric_name,
            )

            # Merge results from the current optimization task
            optimal_param.update(current_optimal_param)
            optimal_value.update(current_optimal_value)

            logger.info(
                f"Job {job_id} optimization for '{optimization_metric_name}' complete. "
                f"Optimal params: {current_optimal_param}, Optimal values: {current_optimal_value}"
            )

        return optimal_param, optimal_value, str(result_path)
    except Exception:
        logger.error("Job failed", exc_info=True, extra={"job_id": job_id})
        return optimal_param, optimal_value, ""
    finally:
        if omc:
            omc.sendExpression("quit()")


def _run_sequential_sweep(config: dict, jobs: List[Dict[str, Any]]) -> List[str]:
    """Executes a parameter sweep sequentially, including optimizations.

    This function runs a series of simulation jobs one after another, reusing
    the same OpenModelica session. For each job, it also runs any configured
    bisection search optimizations and saves a summary of the optimization
    results.

    Args:
        config (dict): The main configuration dictionary.
        jobs (List[Dict[str, Any]]): A list of job parameter dictionaries to execute.

    Returns:
        List[str]: A list of paths to the result files for each job. Failed jobs
                   will have an empty string as their path.
    """
    paths_config = config["paths"]
    sim_config = config["simulation"]

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    os.makedirs(base_temp_dir, exist_ok=True)

    logger.info(
        f"Running sweep sequentially. Intermediate files will be in: {base_temp_dir}"
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

        # _clear_stale_init_xml(mod,sim_config["model_name"])

        # mod.buildModel()

        final_results = []

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
                            f"Failed to clean result file {result_file_path}: {e}"
                        )

                logger.info(
                    f"Sequential job {i+1} finished. Results at {result_file_path}"
                )
                result_paths.append(result_file_path)

                job_final_optimizations = {}
                optimization_tasks = _get_optimization_tasks(config)
                for optimization_metric_name in optimization_tasks:
                    logger.info(
                        f"Job {i+1}: Starting optimization for metric '{optimization_metric_name}'."
                    )
                    job_config = config.copy()
                    job_config["simulation_parameters"] = job_params

                    # Use a unique prefix for each metric to avoid workspace collision
                    metric_job_id_prefix = f"job_{i+1}_{optimization_metric_name}"

                    (optimal_params, optimal_values) = _run_bisection_search_for_job(
                        job_config,
                        job_id_prefix=metric_job_id_prefix,
                        optimization_metric_name=optimization_metric_name,
                    )

                    # Merge results from the current optimization task
                    job_final_optimizations.update(optimal_params)
                    job_final_optimizations.update(optimal_values)

                final_result_entry = job_params.copy()
                final_result_entry.update(job_final_optimizations)
                final_results.append(final_result_entry)

            except Exception as e:
                logger.error(f"Sequential job {i+1} failed: {e}", exc_info=True)
                result_paths.append("")

        # Summarize optimization results
        if _get_optimization_tasks(config):
            results_dir = os.path.abspath(config["paths"]["results_dir"])
            os.makedirs(results_dir, exist_ok=True)
            if final_results:
                final_df = pd.DataFrame(final_results)
                output_path = os.path.join(results_dir, "requierd_tbr_summary.csv")
                final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                logger.info(f"Sweep optimization summary saved to: {output_path}")

        return result_paths
    except Exception as e:
        logger.error(f"Sequential sweep failed during setup: {e}", exc_info=True)
        return [""] * len(jobs)
    finally:
        if omc:
            omc.sendExpression("quit()")


def _execute_analysis_case(case_info: Dict[str, Any]) -> bool:
    """Executes a single analysis case in a separate process.

    This function is designed to be run in a dedicated process. It changes the
    working directory to the case's workspace, sets up logging for that
    process, and calls the main `run_simulation` orchestrator. Inner
    concurrency is disabled to prevent nested process pools.

    Args:
        case_info: A dictionary containing all information for the case, including
            its index, workspace path, configuration, and original case data.

    Returns:
        True if the case executed successfully, False otherwise.

    Note:
        Changes working directory to case workspace for duration of execution.
        Sets up separate logging for the process. Forces concurrent=False in simulation
        config to prevent nested process pools. Restores original working directory
        in finally block.
    """
    case_index = case_info["index"]
    case_workspace = case_info["workspace"]
    case_config = case_info["config"]
    case_data = case_info["case_data"]

    original_cwd = os.getcwd()
    try:
        os.chdir(case_workspace)
        # Each process will have its own logging setup.
        setup_logging(case_config)

        logger.info(
            "Executing analysis case",
            extra={
                "case_name": case_data.get("name", case_index),
                "case_index": case_index,
                "workspace": case_workspace,
                "pid": os.getpid(),
            },
        )

        # Disable inner concurrency to prevent nested process pools
        if "simulation" not in case_config:
            case_config["simulation"] = {}
        case_config["simulation"]["concurrent"] = False
        logger.info("Inner concurrency has been disabled for this case.")

        run_simulation(case_config)

        logger.info(
            "Case executed successfully",
            extra={
                "case_name": case_data.get("name", case_index),
                "case_index": case_index,
            },
        )
        return True
    except Exception:
        logger.error(
            "Case execution failed",
            exc_info=True,
            extra={
                "case_name": case_data.get("name", case_index),
                "case_index": case_index,
            },
        )
        return False
    finally:
        os.chdir(original_cwd)


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
    """Orchestrates the simulation analysis workflow.

    This is the main orchestrator for a simulation analysis run. It handles
    different execution paths based on the configuration:
    - If 'analysis_cases' are defined, it sets up and executes each case,
      potentially in parallel.
    - If a SALib analysis is defined, it delegates to the SALib workflow.
    - Otherwise, it runs a standard parameter sweep, merges results,
      generates plots, and triggers sensitivity analysis and post-processing.

    Args:
        config: The main configuration dictionary for the run.

    Note:
        Supports three modes: multi-case analysis (analysis_cases), SALib analysis
        (independent_variable as list with analyzer), or standard parameter sweep.
        For multi-case mode, creates isolated workspaces and can run cases in parallel
        with ProcessPoolExecutor. Generates summary reports and handles AI analysis retries
        if configured.
    """

    # 1. Split analysis_cases and determine salib_analysis_case
    has_analysis_cases = (
        "sensitivity_analysis" in config
        and "analysis_cases" in config["sensitivity_analysis"]
        and (
            # Support list format
            (
                isinstance(config["sensitivity_analysis"]["analysis_cases"], list)
                and len(config["sensitivity_analysis"]["analysis_cases"]) > 0
            )
            or
            # Support single object format
            isinstance(config["sensitivity_analysis"]["analysis_cases"], dict)
        )
    )

    # Check if it's a SALib analysis case (and not a multi-case analysis)
    sa_config = config.get("sensitivity_analysis", {})
    analysis_case = sa_config.get("analysis_case")

    has_salib_analysis_case = (
        not has_analysis_cases
        and isinstance(analysis_case, dict)
        and isinstance(analysis_case.get("independent_variable"), list)
        and isinstance(analysis_case.get("independent_variable_sampling"), dict)
        and "analyzer" in analysis_case
    )

    if has_analysis_cases and not has_salib_analysis_case:
        logger.info(
            "Detected analysis_cases field, starting to create independent working directories for each analysis case..."
        )

        # Create independent working directories and configuration files for each analysis_case
        case_configs = analysis_setup_analysis_cases_workspaces(config)

        if not case_configs:
            logger.error(
                "Unable to create analysis_cases working directories, stopping execution"
            )
            return

        logger.info(f"Starting execution of {len(case_configs)} analysis cases...")

        sa_config = config.get("sensitivity_analysis", {})
        run_cases_concurrently = sa_config.get("concurrent_cases", False)
        successful_cases = 0

        if run_cases_concurrently:
            logger.info(
                f"Starting execution of {len(case_configs)} analysis cases in PARALLEL."
            )
            max_workers = sa_config.get("max_case_workers", os.cpu_count())
            logger.info(
                f"Using up to {max_workers} parallel processes for analysis cases."
            )

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_case = {
                    executor.submit(_execute_analysis_case, case_info): case_info
                    for case_info in case_configs
                }
                for future in concurrent.futures.as_completed(future_to_case):
                    case_info = future_to_case[future]
                    case_name = case_info["case_data"].get("name", case_info["index"])
                    try:
                        if future.result():
                            successful_cases += 1
                            logger.info(
                                f"Parallel case '{case_name}' completed successfully."
                            )
                        else:
                            logger.warning(
                                f"Parallel case '{case_name}' completed with errors."
                            )
                    except Exception as exc:
                        logger.error(
                            f"Parallel case '{case_name}' failed in executor with: {exc}",
                            exc_info=True,
                        )
        else:
            logger.info(
                f"Starting execution of {len(case_configs)} analysis cases SEQUENTIALLY."
            )
            for case_info in case_configs:
                try:
                    case_index = case_info["index"]
                    case_workspace = case_info["workspace"]
                    case_config = case_info["config"]
                    case_data = case_info["case_data"]

                    logger.info(
                        f"\n=== Starting execution of analysis case {case_index + 1}/{len(case_configs)} ==="
                    )
                    logger.info(
                        f"Case name: {case_data.get('name', f'Case{case_index+1}')}"
                    )
                    logger.info(
                        f"Independent variable: {case_data['independent_variable']}"
                    )
                    logger.info(f"Working directory: {case_workspace}")

                    original_cwd = os.getcwd()
                    os.chdir(case_workspace)

                    try:
                        setup_logging(case_config)
                        run_simulation(case_config)
                        successful_cases += 1
                        logger.info(
                            f"✓ Analysis case {case_index + 1} executed successfully"
                        )
                    except Exception as case_e:
                        logger.error(
                            f"✗ Analysis case {case_index + 1} execution failed: {case_e}",
                            exc_info=True,
                        )
                    finally:
                        os.chdir(original_cwd)
                        setup_logging(config)

                except Exception as e:
                    logger.error(
                        f"✗ Error processing analysis case {case_index + 1}: {e}",
                        exc_info=True,
                    )

        logger.info("\n=== Analysis Cases Execution Completed ===")
        logger.info(
            f"Successfully executed: {successful_cases}/{len(case_configs)} cases"
        )

        generate_analysis_cases_summary(case_configs, config)

        return  # End analysis_cases processing
    elif has_salib_analysis_case:
        logger.info("Detected SALib analysis case, diverting to SALib workflow...")
        run_salib_analysis(config)
        return  # SALib workflow is self-contained, so we exit here.

    # 2. Core operational logic
    jobs = generate_simulation_jobs(config.get("simulation_parameters", {}))

    # --- START: Add baseline jobs based on default parameter values ---
    analysis_case = config.get("sensitivity_analysis", {}).get("analysis_case", {})
    default_values = analysis_case.get("default_simulation_values")

    if default_values:
        logger.info(
            "Found default_simulation_values, generating additional baseline jobs."
        )

        # Prepare simulation parameters for the baseline run, starting with default values
        baseline_params = default_values.copy()

        # Add the main independent variable sweep to the baseline parameters
        independent_var = analysis_case.get("independent_variable")
        independent_sampling = analysis_case.get("independent_variable_sampling")

        if independent_var and independent_sampling:
            baseline_params[independent_var] = independent_sampling

            # Generate the additional jobs using the baseline config
            default_jobs = generate_simulation_jobs(baseline_params)

            # Combine with existing jobs and deduplicate
            combined_jobs = jobs + default_jobs

            # Deduplicate the list of job dictionaries
            seen = set()
            unique_jobs = []
            for job in combined_jobs:
                job_tuple = tuple(sorted(job.items()))
                if job_tuple not in seen:
                    seen.add(job_tuple)
                    unique_jobs.append(job)

            logger.info(
                f"Original jobs: {len(jobs)}, Combined jobs: {len(combined_jobs)}, Unique jobs after deduplication: {len(unique_jobs)}"
            )
            jobs = unique_jobs
    # --- END: Add baseline jobs ---

    try:
        results_dir = os.path.abspath(config["paths"]["results_dir"])
    except KeyError as e:
        logger.error(f"Missing required path key in configuration file: {e}")
        sys.exit(1)

    simulation_results = {}
    use_concurrent = config["simulation"].get("concurrent", False)

    try:
        if config.get("co_simulation") is None:
            if use_concurrent:
                logger.info("Starting simulation in CONCURRENT mode.")
                max_workers = config["simulation"].get("max_workers", os.cpu_count())
                logger.info(f"Using up to {max_workers} parallel workers.")
                final_results = []
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
                            (
                                optimal_params,
                                optimal_values,
                                result_path,
                            ) = future.result()
                            if result_path:
                                simulation_results[
                                    tuple(sorted(job_params.items()))
                                ] = result_path
                        except Exception as exc:
                            logger.error(
                                f"Job for {job_params} generated an exception: {exc}",
                                exc_info=True,
                            )
                        final_result_entry = job_params.copy()
                        final_result_entry.update(optimal_params)
                        final_result_entry.update(optimal_values)
                        final_results.append(final_result_entry)

                if _get_optimization_tasks(config):
                    results_dir = os.path.abspath(config["paths"]["results_dir"])
                    os.makedirs(results_dir, exist_ok=True)
                    if final_results:
                        final_df = pd.DataFrame(final_results)
                        output_path = os.path.join(
                            results_dir, "requierd_tbr_summary.csv"
                        )
                        final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                        logger.info(
                            f"Sweep optimization summary saved to: {output_path}"
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

                final_results = []

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
                            (
                                optimal_params,
                                optimal_values,
                                result_path,
                            ) = future.result()
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
                        final_result_entry = job_params.copy()
                        final_result_entry.update(optimal_params)
                        final_result_entry.update(optimal_values)
                        final_results.append(final_result_entry)
            else:
                logger.info("Starting co-simulation in SEQUENTIAL mode.")
                final_results = []
                for i, job_params in enumerate(jobs):
                    job_id = i + 1
                    logger.info(f"--- Starting Sequential Job {job_id}/{len(jobs)} ---")
                    try:
                        (
                            optimal_params,
                            optimal_values,
                            result_path,
                        ) = _run_co_simulation(config, job_params, job_id=job_id)
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
                        final_result_entry = job_params.copy()
                        final_result_entry.update(optimal_params)
                        final_result_entry.update(optimal_values)
                        final_results.append(final_result_entry)
                    except Exception as exc:
                        logger.error(
                            f"Job for params {job_params} generated an exception: {exc}",
                            exc_info=True,
                        )
                    logger.info(f"--- Finished Sequential Job {job_id}/{len(jobs)} ---")

            if _get_optimization_tasks(config):
                results_dir = os.path.abspath(config["paths"]["results_dir"])
                os.makedirs(results_dir, exist_ok=True)
                if final_results:
                    final_df = pd.DataFrame(final_results)
                    output_path = os.path.join(results_dir, "requierd_tbr_summary.csv")
                    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                    logger.info(f"Sweep optimization summary saved to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to run simualtion: {e}")

    # 3. Data merging and processing
    run_results_dir = results_dir
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

        # Clean up rows where the 'time' column is blank, which often occurs as redundant rows at the end of the file.
        combined_df.dropna(subset=["time"], inplace=True)
        combined_df.to_csv(combined_csv_path, index=False)
        logger.info(f"Combined results saved to: {combined_csv_path}")
    else:
        logger.warning("No valid results found to combine.")

    # Check if sweep_time plotting is enabled
    analysis_case = config["sensitivity_analysis"].get("analysis_case", {})
    sweep_time_list = analysis_case.get("sweep_time", None)
    if sweep_time_list and len(sweep_time_list) >= 1:
        # Get parameters for plot_sweep_time_series
        independent_var = analysis_case.get("independent_variable")
        dependent_vars = analysis_case.get("dependent_variables", [])
        independent_var_alias = analysis_case.get("independent_variable_alias")

        if (
            independent_var
            and dependent_vars
            and combined_csv_path
            and os.path.exists(combined_csv_path)
        ):

            try:
                # Get default values if they exist to filter the plot
                default_values = analysis_case.get("default_simulation_values")

                plot_path = plot_sweep_time_series(
                    csv_path=combined_csv_path,
                    save_dir=run_results_dir,
                    y_var_name=sweep_time_list,
                    independent_var_name=independent_var,
                    independent_var_alias=independent_var_alias,
                    default_params=default_values,  # Pass default values to the plot function
                    glossary_path=config["sensitivity_analysis"].get(
                        "glossary_path", None
                    ),
                )
                if plot_path:
                    logger.info(f"Sweep time series plot generated: {plot_path}")
                else:
                    logger.warning("Failed to generate sweep time series plot")
            except Exception as e:
                logger.error(f"Error generating sweep time series plot: {e}")

    # 4. Sensitivity analysis
    _run_sensitivity_analysis(config, run_results_dir, jobs)

    # 5. Post-processing
    if combined_df is not None:
        # Calculate the top-level post-processing directory
        top_level_run_workspace = os.path.abspath("post_processing")
        _run_post_processing(config, combined_df, top_level_run_workspace)
    else:
        logger.warning(
            "No simulation results were generated, skipping post-processing."
        )

    # 6. Intermediate data cleaning
    if not config["simulation"].get("keep_temp_files", True):
        logger.info("Cleaning up temporary directory...")
        temp_dir_path = os.path.abspath(config["paths"].get("temp_dir", "temp"))
        if os.path.exists(temp_dir_path):
            try:
                shutil.rmtree(temp_dir_path)
                os.makedirs(temp_dir_path)  # Recreate for next run
            except OSError as e:
                logger.error(f"Error cleaning up temp directory: {e}")


def retry_analysis(timestamp: str) -> None:
    """Retries a failed AI analysis for a given run timestamp.

    This function restores the configuration from the log file of a previous
    run and re-triggers the AI-dependent parts of the analysis, including
    report generation and consolidation.

    Args:
        timestamp (str): The timestamp of the run to retry (e.g., "20230101_120000").
    """
    config, original_config = restore_configs_from_log(timestamp)
    if not config or not original_config:
        # Error is printed inside the helper function
        sys.exit(1)

    config["run_timestamp"] = timestamp

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)
    logger.info(
        f"Successfully restored configuration for timestamp {timestamp} for retry."
    )

    logger.info("Starting in AI analysis retry mode...")
    if not analysis_validate_config(config):
        sys.exit(1)

    case_configs = analysis_setup_analysis_cases_workspaces(config)
    if not case_configs:
        logger.error("Could not set up case workspaces for retry. Aborting.")
        sys.exit(1)

    retry_ai_analysis(case_configs, config)
    consolidate_reports(case_configs, config)

    logger.info("AI analysis retry and consolidation complete.")


def main(config_path: str) -> None:
    """Main entry point for a simulation analysis run.

    This function prepares the configuration for an analysis run, sets up
    logging, and calls the main `run_simulation` orchestrator for analysis.

    Args:
        config_path (str): The path to the JSON configuration file.
    """
    config, original_config = analysis_prepare_config(config_path)
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
    # To allow running this script directly, we'll do a simplified arg parsing here.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.json")
    args = parser.parse_args()
    main(args.config)
