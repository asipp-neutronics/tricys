import logging
import multiprocessing
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

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
from tricys.core.jobs import generate_simulation_jobs
from tricys.core.modelica import (
    format_parameter_value,
    get_om_session,
    load_modelica_package,
)
from tricys.simulation.simulation import (
    run_co_simulation_job,
    run_parallel_sweep,
    run_post_processing,
    run_sequential_sweep,
    run_single_job,
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


def _extract_metrics_from_hdf5(
    hdf_path: str,
    metrics_definition: Dict[str, Any],
    analysis_case: Dict[str, Any],
) -> pd.DataFrame:
    """Extracts metrics from HDF5 file iteratively to save memory."""
    try:
        jobs_df = pd.read_hdf(hdf_path, "jobs_metadata")
        all_results = []

        # We need to process job by job
        # To make it efficient, we can't easily query row by row if not indexed.
        # But 'results' table is appendable. We can iterate it in chunks or by job_id if indexed.
        # Given the structure, reading the whole table is bad.
        # We can use 'where' clause if indexed, or iterator.
        # Assuming job_id is not indexed in file creation yet (it should be for perf),
        # but let's assume we can read chunks.
        # Actually, for correctness and simplicity in this refactor:
        # We iterate job_ids. If performance is slow, we can optimize later.

        # Optimize: Read chunks of results, group by job_id in memory, process.
        # But results for one job might be split across chunks? Unlikely if appended sequentially.
        # Safe approach: Read by job_id using 'where' (slow if not indexed) or read full table in chunks?
        # If we can't fully read table, we must rely on 'select'.

        store = pd.HDFStore(hdf_path, mode="r")
        try:
            if "job_id" in store.select("results", start=0, stop=1).columns:
                # We can't easily check index status without accessing pytables object,
                # but we can just try selecting.
                pass
        except:
            pass
        store.close()

        total_jobs = len(jobs_df)
        logger.info(f"Extracting metrics from HDF5 for {total_jobs} jobs...")

        for idx, job_row in jobs_df.iterrows():
            job_id = job_row["job_id"]
            # Reconstruct params dict
            job_params = job_row.drop("job_id").to_dict()

            # Read results for this job
            # Note: This might be slow for 1.8M jobs if done one by one without index.
            # But it is memory safe.
            try:
                df = pd.read_hdf(hdf_path, "results", where=f"job_id == {job_id}")
            except ValueError:
                # Fallback if query fails (e.g. table not indexed/support query)
                # If cannot query, this strategy fails for huge files.
                # Assuming 'enhanced' mode indexing or manageable size for now.
                # Alternative: chunk iterator.
                logger.warning(f"Could not query job_id={job_id}, skipping.")
                continue

            if df.empty:
                continue

            # Create a "wide" dataframe for this single job so extract_metrics works
            # extract_metrics expects columns like 'var&param=val'
            param_string = "&".join([f"{k}={v}" for k, v in job_params.items()])

            # Drop metadata cols
            data_cols = df.drop(columns=["job_id", "time"], errors="ignore")

            rename_map = {
                col: f"{col}&{param_string}" if param_string else col
                for col in data_cols.columns
            }
            renamed_df = data_cols.rename(columns=rename_map)
            renamed_df["time"] = df["time"]  # Put time back for metric calc

            # Extract metrics for this single job
            # extract_metrics returns a pivoted DataFrame (1 row usually)
            # We want the raw rows before pivot? No, extract_metrics does pivot.
            # But extract_metrics implementation aggregates a list then makes DF.
            # If we call it on 1 job, we get 1 row DF.
            job_metrics_df = extract_metrics(
                renamed_df, metrics_definition, analysis_case
            )

            if not job_metrics_df.empty:
                all_results.append(job_metrics_df)

            if idx % 100 == 0:
                logger.debug(f"Processed {idx}/{total_jobs} jobs from HDF5")

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Failed to extract metrics from HDF5: {e}")
        return pd.DataFrame()


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
        # Get analysis_case configuration first
        analysis_config = config["sensitivity_analysis"]
        analysis_case = analysis_config["analysis_case"]

        # Check if there are result data files
        combined_csv_path = os.path.join(run_results_dir, "sweep_results.csv")
        single_result_path = os.path.join(run_results_dir, "simulation_result.csv")
        hdf_path = os.path.join(run_results_dir, "sweep_results.h5")

        # Determine result file path based on number of jobs
        if os.path.exists(hdf_path):
            logger.info(f"Loading results from HDF5: {hdf_path}")
            # Use special HDF5 extraction to avoid OOM
            summary_df = _extract_metrics_from_hdf5(
                hdf_path, analysis_config["metrics_definition"], analysis_case
            )
            # Skip the extract_metrics call below as we already have summary_df
            # We set results_df to None to signal this
            results_df = None

        elif len(jobs) == 1 and os.path.exists(single_result_path):
            # Single task case
            results_df = pd.read_csv(single_result_path)
            logger.info(f"Loading single task result from: {single_result_path}")
            summary_df = None  # Will be calculated below
        elif len(jobs) > 1 and os.path.exists(combined_csv_path):
            # Multi-task case
            results_df = pd.read_csv(combined_csv_path)
            logger.info(f"Loading sweep results from: {combined_csv_path}")
            summary_df = None  # Will be calculated below
        else:
            logger.warning("No result data file found for sensitivity analysis.")
            return

        if analysis_case is None:
            logger.warning("No valid analysis_case found for sensitivity analysis.")
            return

        # Extract summary metrics (if not already done via HDF5)
        if summary_df is None:
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

        # Generate sweep time series plots (inventory evolution)
        results_path = os.path.join(run_results_dir, "sweep_results.h5")
        if not os.path.exists(results_path):
            results_path = os.path.join(run_results_dir, "sweep_results.csv")

        if os.path.exists(results_path):
            plot_sweep_time_series(
                results_path,
                run_results_dir,
                "sds.inventory",  # Default to SDS inventory
                analysis_case["independent_variable"],
                default_params=analysis_case.get("default_simulation_values"),
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
        if "mod" in locals() and mod and hasattr(mod, "omc"):
            try:
                mod.omc.sendExpression("quit()")
            except Exception:
                pass

        if omc:
            try:
                omc.sendExpression("quit()")
            except Exception:
                pass

    return all_optimal_params, all_optimal_values


def _resolve_isolated_package_path(
    job_workspace: str, original_package_path: str
) -> str:
    """Helper to determine the path of the copied package in the isolated workspace."""
    if os.path.isfile(original_package_path) and not original_package_path.endswith(
        "package.mo"
    ):
        return os.path.join(job_workspace, os.path.basename(original_package_path))
    else:
        if os.path.isfile(original_package_path):
            original_dir = os.path.dirname(original_package_path)
            base_name = os.path.basename(original_package_path)
            dir_name = os.path.basename(original_dir)
            return os.path.join(job_workspace, dir_name, base_name)
        else:
            dir_name = os.path.basename(original_package_path)
            return os.path.join(job_workspace, dir_name, "package.mo")


def _run_optimization_tasks(
    config: dict, job_params: dict, job_id: int, package_path_override: str = None
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Runs all configured bisection search optimization tasks for a job."""
    optimal_param = {}
    optimal_value = {}

    optimization_tasks = _get_optimization_tasks(config)
    paths_config = config["paths"]
    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))

    # Determine the package path to use for optimization
    package_path = (
        package_path_override
        if package_path_override
        else os.path.abspath(paths_config["package_path"])
    )

    for optimization_metric_name in optimization_tasks:
        logger.info(
            f"Job {job_id}: Starting optimization for metric '{optimization_metric_name}'."
        )
        job_config = config.copy()
        job_config["paths"] = config["paths"].copy()
        job_config["paths"]["package_path"] = package_path
        job_config["paths"]["temp_dir"] = base_temp_dir
        job_config["simulation_parameters"] = job_params
        # Note: model_name logic was specific in co-sim (using final_model_name).
        # But _run_bisection_search_for_job uses sim_config["model_name"].
        # We need to ensure job_config["simulation"]["model_name"] is correct.
        # However, _run_bisection_search_for_job uses config["simulation"]["model_name"].
        # In co-simulation, the model name changes to ..._Intercepted.
        # We should pass the correct model name if it changed.

        # Unique prefix
        metric_job_id_prefix = f"job_{job_id}_{optimization_metric_name}"

        (
            current_optimal_param,
            current_optimal_value,
        ) = _run_bisection_search_for_job(
            job_config,
            job_id_prefix=metric_job_id_prefix,
            optimization_metric_name=optimization_metric_name,
        )

        optimal_param.update(current_optimal_param)
        optimal_value.update(current_optimal_value)

        logger.info(
            f"Job {job_id} optimization for '{optimization_metric_name}' complete. "
            f"Optimal params: {current_optimal_param}, Optimal values: {current_optimal_value}"
        )

    return optimal_param, optimal_value


def _run_co_simulation(
    config: dict, job_params: dict, job_id: int = 0
) -> tuple[Dict[str, float], Dict[str, float], str]:
    """Runs a full co-simulation workflow and any subsequent optimizations."""

    # Force keep_temp_files to True so we can use the workspace for optimization
    original_keep = config["simulation"].get("keep_temp_files", True)
    config["simulation"]["keep_temp_files"] = True

    result_path = ""
    try:
        result_path = run_co_simulation_job(config, job_params, job_id)
    except Exception as e:
        logger.error(f"Co-simulation failed: {e}", exc_info=True)
        # Ensure we restore the config even if failed
    finally:
        config["simulation"]["keep_temp_files"] = original_keep

    if not result_path:
        # If run_co_simulation_job failed, it might have kept the files. We should clean up if needed.
        if not original_keep:
            paths_config = config["paths"]
            base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
            job_workspace = os.path.join(base_temp_dir, f"job_{job_id}")
            if os.path.exists(job_workspace):
                shutil.rmtree(job_workspace)
        return {}, {}, ""

    # Run Optimization
    paths_config = config["paths"]
    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    job_workspace = os.path.join(base_temp_dir, f"job_{job_id}")
    original_package_path = os.path.abspath(paths_config["package_path"])

    isolated_package_path = _resolve_isolated_package_path(
        job_workspace, original_package_path
    )

    # Identify final model name logic (duplicated from simulation.py logic technically,
    # but we need it to set the correct model name for optimization)
    # The interceptor logic changes the model name.
    co_sim_config = config.get("co_simulation", {})
    mode = co_sim_config.get("mode", "interceptor")
    model_name = config["simulation"]["model_name"]

    if mode == "replacement":
        final_model_name = model_name
    else:
        package_name, original_system_name = model_name.split(".")
        final_model_name = f"{package_name}.{original_system_name}_Intercepted"

    # We need to temporarily update the model name in config for optimization
    original_model_name = config["simulation"]["model_name"]
    config["simulation"]["model_name"] = final_model_name

    try:
        optimal_params, optimal_values = _run_optimization_tasks(
            config, job_params, job_id, package_path_override=isolated_package_path
        )
    finally:
        # Restore model name
        config["simulation"]["model_name"] = original_model_name

        # Cleanup if needed
        if not original_keep and os.path.exists(job_workspace):
            shutil.rmtree(job_workspace)
            logger.info(f"Cleaned up job workspace {job_workspace}")

    return optimal_params, optimal_values, result_path


def _run_single_job(
    config: dict, job_params: dict, job_id: int = 0
) -> tuple[Dict[str, float], Dict[str, float], str]:
    """Executes a single simulation job and any subsequent optimizations."""
    result_path = run_single_job(config, job_params, job_id)

    if not result_path:
        return {}, {}, ""

    optimal_params, optimal_values = _run_optimization_tasks(config, job_params, job_id)

    return optimal_params, optimal_values, result_path


def _run_sequential_sweep(config: dict, jobs: List[Dict[str, Any]]) -> List[str]:
    """Executes a parameter sweep sequentially, including optimizations."""

    final_results = []

    def post_job_callback(index: int, params: Dict[str, Any], result_path: str):
        if not result_path:
            return

        logger.info(f"Starting optimization for sequential job {index+1}")
        try:
            optimal_params, optimal_values = _run_optimization_tasks(
                config, params, index + 1
            )

            final_result_entry = params.copy()
            final_result_entry.update(optimal_params)
            final_result_entry.update(optimal_values)
            final_results.append(final_result_entry)
        except Exception as e:
            logger.error(
                f"Optimization failed for sequential job {index+1}: {e}", exc_info=True
            )

    result_paths = run_sequential_sweep(
        config, jobs, post_job_callback=post_job_callback
    )

    # Summarize optimization results
    if final_results and _get_optimization_tasks(config):
        results_dir = os.path.abspath(config["paths"]["results_dir"])
        os.makedirs(results_dir, exist_ok=True)
        final_df = pd.DataFrame(final_results)
        output_path = os.path.join(results_dir, "requierd_tbr_summary.csv")
        final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Sweep optimization summary saved to: {output_path}")

    return result_paths


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
    """Wrapper for simulation.run_post_processing."""
    run_post_processing(config, results_df, post_processing_output_dir)


def _mp_execute_analysis_case_wrapper(case_info):
    """Wrapper for multiprocessing.Pool to map _execute_analysis_case."""
    try:
        res = _execute_analysis_case(case_info)
        return case_info, res, None
    except Exception as e:
        return case_info, None, str(e)


def _mp_run_co_simulation_wrapper(args):
    """Wrapper for multiprocessing.Pool to map _run_co_simulation."""
    config, job_params, job_id = args
    try:
        res = _run_co_simulation(config, job_params, job_id)
        # res is (optimal_params, optimal_values, result_path)
        return job_id, job_params, res, None
    except Exception as e:
        return job_id, job_params, None, str(e)


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

        # Force sequential cases for Enhanced Mode to prevent ProcessPoolExecutor conflicts
        # User requested this safeguard.
        # Prevent nested multiprocessing pools (Co-Sim uses MP, Cases use MP)
        # If Co-Simulation is active, force sequential cases.
        # Otherwise (Standard Job -> Threads), allowing parallel cases is safe.
        is_co_sim = config.get("co_simulation") is not None
        if is_co_sim and run_cases_concurrently:
            logger.warning(
                "Co-Simulation detected: Forcing sequential execution of Analysis Cases "
                "to prevent nested multiprocessing pools."
            )
            run_cases_concurrently = False

        successful_cases = 0

        if run_cases_concurrently:
            logger.info(
                f"Starting execution of {len(case_configs)} analysis cases in PARALLEL."
            )
            max_workers = sa_config.get("max_case_workers", os.cpu_count())
            logger.info(
                f"Using up to {max_workers} parallel processes for analysis cases."
            )

            with multiprocessing.Pool(processes=max_workers) as pool:
                for case_info, success, error in pool.imap_unordered(
                    _mp_execute_analysis_case_wrapper, case_configs
                ):
                    case_name = case_info["case_data"].get("name", case_info["index"])
                    if error:
                        logger.error(
                            f"Parallel case '{case_name}' failed in executor with: {error}"
                        )
                    else:
                        if success:
                            successful_cases += 1
                            logger.info(
                                f"Parallel case '{case_name}' completed successfully."
                            )
                        else:
                            logger.warning(
                                f"Parallel case '{case_name}' completed with errors."
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

    # Use HDF5 for results to prevent OOM
    run_results_dir = results_dir
    os.makedirs(run_results_dir, exist_ok=True)
    hdf_filename = "sweep_results.h5"
    hdf_path = get_unique_filename(run_results_dir, hdf_filename)

    execute_mode = config["simulation"].get("execute_mode", "standard")

    # helper for hdf5 processing
    def _process_h5_result(store, job_id, params, res_path):
        if not res_path or not os.path.exists(res_path):
            return
        try:
            df = pd.read_csv(res_path)
            df["job_id"] = job_id

            # Append to store using Table format
            store.append(
                "results", df, index=False, data_columns=True
            )  # data_columns=True allows indexing?
            # Note: Do not enable index here for speed, better to index later if needed.

            # Store job params
            param_df = pd.DataFrame([params])
            param_df["job_id"] = job_id
            store.append("jobs", param_df, index=False, data_columns=True)

            # Cleanup
            job_dir = os.path.dirname(res_path)
            if os.path.exists(job_dir) and "job_" in os.path.basename(job_dir):
                shutil.rmtree(job_dir)
        except Exception as e:
            logger.error(f"Failed to process HDF5 result for job {job_id}: {e}")

    simulation_results = {}
    use_concurrent = config["simulation"].get("concurrent", False)

    try:
        # Define runners based on config
        is_co_sim = config.get("co_simulation") is not None
        runner_func = _run_co_simulation if is_co_sim else _run_single_job

        # Common cleanup/prep
        final_results = []

        if execute_mode == "enhanced":
            logger.info("Running simulation in ENHANCED mode (HDF5 Streaming).")

            # Using HDFStore
            with pd.HDFStore(hdf_path, mode="w", complib="blosc", complevel=9) as store:
                # Save jobs metadata
                meta_df = pd.DataFrame(jobs)
                meta_df["job_id"] = range(1, len(jobs) + 1)
                store.put("jobs_metadata", meta_df, format="table", data_columns=True)

                use_mp = is_co_sim
                max_workers = config["simulation"].get("max_workers", os.cpu_count())

                # Callback for Enhanced Mode
                import threading

                store_lock = threading.Lock()

                def enhanced_callback(job_id, params, result):
                    if not result:
                        return
                    optimal_params, optimal_values, result_path = result

                    if result_path:
                        if use_mp:
                            _process_h5_result(store, job_id, params, result_path)
                        else:
                            with store_lock:
                                _process_h5_result(store, job_id, params, result_path)

                    entry = params.copy()
                    entry.update(optimal_params)
                    entry.update(optimal_values)

                    if use_mp:
                        final_results.append(entry)
                    else:
                        with store_lock:
                            final_results.append(entry)

                logger.info(
                    f"Starting sweep (Enhanced). MP={use_mp}, Workers={max_workers if use_concurrent else 1}"
                )

                run_parallel_sweep(
                    config,
                    jobs,
                    max_workers=max_workers if use_concurrent else 1,
                    use_multiprocessing=use_mp if use_concurrent else False,
                    post_job_callback=enhanced_callback,
                    custom_runner_func=runner_func,
                )

            if _get_optimization_tasks(config) and final_results:
                final_df = pd.DataFrame(final_results)
                output_path = os.path.join(run_results_dir, "requierd_tbr_summary.csv")
                final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                logger.info(f"Sweep optimization summary saved to: {output_path}")

        else:
            # STANDARD MODE
            logger.info("Running simulation in STANDARD mode.")
            simulation_results = {}  # Needed for post-processing

            use_mp = False  # Standard mode prefers Threading usually?
            # Original code: Simple -> Threading, Co-Sim -> MP?
            # Original code:
            # if config.get("co_simulation") is None: ThreadPoolExecutor
            # else: multiprocessing.Pool
            # So YES, Co-Sim uses MP in Standard Mode too.
            use_mp = is_co_sim

            max_workers = config["simulation"].get("max_workers", os.cpu_count())

            import threading

            res_lock = threading.Lock()

            def standard_callback(job_id, params, result):
                if not result:
                    return
                optimal_params, optimal_values, result_path = result

                entry = params.copy()
                entry.update(optimal_params)
                entry.update(optimal_values)

                if use_mp:
                    # Serial callback in main process
                    if result_path:
                        simulation_results[tuple(sorted(params.items()))] = result_path
                    final_results.append(entry)
                else:
                    # Threaded callback
                    with res_lock:
                        if result_path:
                            simulation_results[tuple(sorted(params.items()))] = (
                                result_path
                            )
                        final_results.append(entry)

            logger.info(
                f"Starting sweep (Standard). MP={use_mp}, Workers={max_workers if use_concurrent else 1}"
            )

            run_parallel_sweep(
                config,
                jobs,
                max_workers=max_workers if use_concurrent else 1,
                use_multiprocessing=use_mp if use_concurrent else False,
                post_job_callback=standard_callback,
                custom_runner_func=runner_func,
            )

            if _get_optimization_tasks(config):
                results_dir = os.path.abspath(config["paths"]["results_dir"])
                # final_results populated by callback
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

    if execute_mode != "enhanced":
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
    elif execute_mode == "enhanced":
        # Enhanced mode skips combined_df
        combined_df = None
    else:
        combined_df = pd.DataFrame()  # Or None, as you had before

    combined_csv_path = None

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
    elif execute_mode == "enhanced":
        logger.info("Skipping CSV combination in Enhanced Mode.")
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
            and execute_mode
            != "enhanced"  # Sweep time plot for enhanced mode not implemented yet
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
    elif execute_mode == "enhanced":
        top_level_run_workspace = os.path.abspath("post_processing")
        # Pass None as df, but update simulation.py's run_post_processing to handle HDF5 path?
        # simulation_analysis.py imports run_post_processing from simulation.py.
        # In Step 283 I updated run_post_processing to accept results_file_path.
        # So I should pass it here.
        from tricys.simulation.simulation import run_post_processing as run_pp

        run_pp(config, None, top_level_run_workspace, results_file_path=hdf_path)
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


def main(config_or_path: Union[str, Dict[str, Any]], base_dir: str = None) -> None:
    """Main entry point for a simulation analysis run.

    This function prepares the configuration for an analysis run, sets up
    logging, and calls the main `run_simulation` orchestrator for analysis.

    Args:
        config_or_path: The path to the JSON configuration file OR a config dict.
        base_dir: Optional base directory for resolving relative paths.
    """
    config, original_config = analysis_prepare_config(config_or_path, base_dir=base_dir)
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
    # To allow running this script directly, we'll do a simplified arg parsing here.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.json")
    args = parser.parse_args()
    main(args.config)
