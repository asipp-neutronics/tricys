import logging
import multiprocessing
import os
import shutil
import sys
from typing import Any, Dict, List, Union

import pandas as pd

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
from tricys.simulation.simulation import (
    _build_model_only,
    _run_fast_subprocess_job,
    run_co_simulation_job,
    run_post_processing,
)
from tricys.utils.concurrency_utils import get_safe_max_workers
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


def _run_bisection_search_fast(
    config: Dict[str, Any],
    job_id_prefix: str,
    optimization_metric_name: str,
    fast_context: Dict[str, Any],
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Performs bisection search using pre-compiled executable (Fast Mode)."""
    sensitivity_analysis = config.get("sensitivity_analysis", {})
    metrics_definition = sensitivity_analysis.get("metrics_definition", {})
    optimization_config = metrics_definition.get(optimization_metric_name, {})

    if not optimization_config or "parameter_to_optimize" not in optimization_config:
        raise ValueError(
            f"Optimization config for '{optimization_metric_name}' invalid."
        )

    param_to_optimize = optimization_config["parameter_to_optimize"]
    low_orig, high_orig = optimization_config["search_range"]
    tolerance = optimization_config.get("tolerance", 0.001)
    max_iterations = optimization_config.get("max_iterations", 10)
    stop_time = config["simulation"]["stop_time"]
    metric_max_value = optimization_config.get("metric_max_value", stop_time)
    metric_name = optimization_config.get("metric_name", "Self_Sufficiency_Time")
    source_column = optimization_config.get("source_column", "sds.inventory")

    metric_max_values = (
        metric_max_value if isinstance(metric_max_value, list) else [metric_max_value]
    )
    is_list_input = isinstance(metric_max_value, list)

    all_optimal_params = {}
    all_optimal_values = {}

    for current_metric_max_value in metric_max_values:
        low, high = low_orig, high_orig
        best_successful_param = float("inf")
        best_successful_value = float("inf")

        for i in range(max_iterations):
            if high - low < tolerance:
                break

            mid_param = (low + high) / 2

            # Prepare Job Params
            job_params = config.get("simulation_parameters", {}).copy()
            job_params[param_to_optimize] = mid_param

            # Generate unique ID for this iteration to avoid collision in temp dirs
            iter_id = f"{job_id_prefix}_iter{i}_{int(mid_param*1000)}"

            # Run Fast Subprocess
            try:
                # We use a hash or simplified ID for the folder to be safe
                # Note: _run_fast_subprocess_job creates job_{job_id} folder.
                # We need to pass a unique job_id or handle base_temp_dir carefully.
                # We can pass a unique sub-directory as base_temp_dir?
                # Actually, let's just make job_id unique.
                # _run_fast_subprocess_job expects int job_id usually, but uses str(job_id) in path.
                # Let's pass the iter_id string.
                result_path = _run_fast_subprocess_job(
                    job_params,
                    iter_id,
                    fast_context["exe"],
                    fast_context["xml"],
                    fast_context["om_bin"],
                    fast_context["temp_dir"],
                    config["simulation"],
                    inplace_execution=False,  # Always isolated for safety in parallel/iterative
                )

                metric_value = float("inf")
                if result_path and os.path.exists(result_path):
                    try:
                        results_df = pd.read_csv(result_path)
                        if source_column in results_df.columns:
                            if metric_name == "Self_Sufficiency_Time":
                                metric_value = time_of_turning_point(
                                    results_df[source_column], results_df["time"]
                                )
                            elif metric_name == "Doubling_Time":
                                metric_value = calculate_doubling_time(
                                    results_df[source_column], results_df["time"]
                                )
                            elif metric_name == "Startup_Inventory":
                                metric_value = calculate_startup_inventory(
                                    results_df[source_column]
                                )

                        # Clean up this iteration's result immediately
                        job_dir = os.path.dirname(result_path)
                        shutil.rmtree(job_dir, ignore_errors=True)

                    except Exception:
                        pass

                if metric_value < min(stop_time, current_metric_max_value):
                    best_successful_param = mid_param
                    best_successful_value = metric_value
                    high = mid_param
                else:
                    low = mid_param

            except Exception as e:
                logger.warning(f"Fast bisection iteration failed: {e}")

        # Store results (Logic identical to original)
        value_key_base = f"{metric_name}_for_{optimization_metric_name}"
        if is_list_input:
            unit_str = f"{current_metric_max_value} units"  # Simplified
            param_key = f"{optimization_metric_name}({unit_str})"
            value_key = f"{value_key_base}({unit_str})"
        else:
            param_key = optimization_metric_name
            value_key = value_key_base

        all_optimal_params[param_key] = best_successful_param
        all_optimal_values[value_key] = best_successful_value

    return all_optimal_params, all_optimal_values


def _run_optimization_tasks_fast(
    config: dict, job_params: dict, job_id: int, fast_context: Dict[str, Any]
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Runs optimization tasks using fast subprocess execution."""
    optimal_param = {}
    optimal_value = {}
    optimization_tasks = _get_optimization_tasks(config)

    for metric_name in optimization_tasks:
        job_config = config.copy()
        job_config["simulation_parameters"] = job_params

        prefix = f"job_{job_id}_{metric_name}"
        p, v = _run_bisection_search_fast(job_config, prefix, metric_name, fast_context)
        optimal_param.update(p)
        optimal_value.update(v)

    return optimal_param, optimal_value


def _enhanced_runner_wrapper(args):
    """Worker function for Enhanced Mode (Standard/Non-CoSim)."""
    job_params, job_id, config, fast_context = args
    try:
        # 1. Run Main Simulation
        result_path = _run_fast_subprocess_job(
            job_params,
            job_id,
            fast_context["exe"],
            fast_context["xml"],
            fast_context["om_bin"],
            fast_context["temp_dir"],
            config["simulation"],
            variable_filter=config["simulation"].get("variableFilter"),
            inplace_execution=False,  # Concurrent safe
        )

        # 2. Run Optimization
        opt_params, opt_values = {}, {}
        if result_path and os.path.exists(result_path):
            opt_params, opt_values = _run_optimization_tasks_fast(
                config, job_params, job_id, fast_context
            )

        return job_id, job_params, (opt_params, opt_values, result_path), None
    except Exception as e:
        return job_id, job_params, None, str(e)


def _co_sim_runner_wrapper(args):
    """Worker function for Co-Simulation (Enhanced HDF5 Mode)."""
    config, job_params, job_id = args
    try:
        # 1. Run Co-Simulation (Using Standard Logic)
        result_path = run_co_simulation_job(config, job_params, job_id)

        # 2. Run Optimization
        opt_params, opt_values = {}, {}
        # Optimization for Co-Simulation is temporarily disabled in Enhanced Mode refactor
        # as it relied on legacy ModelicaSystem execution which has been removed.
        # Future work: Implement fast optimization for Co-Sim or restore legacy support.

        return job_id, job_params, (opt_params, opt_values, result_path), None
    except Exception as e:
        return job_id, job_params, None, str(e)


def run_simulation(config: Dict[str, Any]) -> None:
    """Orchestrates the simulation analysis workflow (Default: Enhanced Mode)."""

    # [Analysis Cases Handling - Preserved]
    if _handle_analysis_cases(config):
        return

    # Core Execution Logic
    jobs = generate_simulation_jobs(config.get("simulation_parameters", {}))
    _add_baseline_jobs(config, jobs)  # Helper to de-clutter (implied logic moved/kept)

    try:
        results_dir = os.path.abspath(config["paths"]["results_dir"])
        temp_dir = os.path.abspath(config["paths"].get("temp_dir", "temp"))
    except KeyError as e:
        logger.error(f"Missing config path: {e}")
        sys.exit(1)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    hdf_filename = "sweep_results.h5"
    hdf_path = get_unique_filename(results_dir, hdf_filename)

    is_co_sim = config.get("co_simulation") is not None
    use_concurrent = config["simulation"].get("concurrent", False)
    maximize_workers = config["simulation"].get("maximize_workers", False)
    max_workers = get_safe_max_workers(
        config["simulation"].get("max_workers"),
        maximize=maximize_workers,
        task_count=len(jobs),
    )

    logger.info(
        f"Starting Simulation Analysis (Enhanced Mode). CoSim={is_co_sim}, Concurrent={use_concurrent}"
    )

    # Prepare Context
    fast_context = {}
    if not is_co_sim:
        # Build Once for Standard Jobs
        try:
            exe, xml, om_bin = _build_model_only(config)
            fast_context = {
                "exe": exe,
                "xml": xml,
                "om_bin": om_bin,
                "temp_dir": temp_dir,
            }
        except Exception as e:
            logger.error(f"Build failed: {e}")
            sys.exit(1)

    # Run Sweep
    final_results = []

    try:
        with pd.HDFStore(hdf_path, mode="w", complib="blosc", complevel=9) as store:
            # Metadata
            meta_df = pd.DataFrame(jobs)
            if not meta_df.empty:
                meta_df["job_id"] = range(1, len(jobs) + 1)
                store.put("jobs_metadata", meta_df, format="table", data_columns=True)

            # Results Handler
            def _handle_result(job_id, params, result_data):
                opts, vals, res_path = result_data

                # Save trace to HDF5
                if res_path and os.path.exists(res_path):
                    try:
                        df = pd.read_csv(res_path)
                        df["job_id"] = job_id
                        store.append("results", df, index=False, data_columns=True)

                        param_df = pd.DataFrame([params])
                        param_df["job_id"] = job_id
                        store.append("jobs", param_df, index=False, data_columns=True)

                        # Cleanup Job Workspace
                        job_dir = os.path.dirname(res_path)
                        if "job_" in os.path.basename(job_dir):
                            shutil.rmtree(job_dir, ignore_errors=True)
                    except Exception as e:
                        logger.error(f"HDF5 write failed for job {job_id}: {e}")

                # Collect Summary
                entry = params.copy()
                entry.update(opts)
                entry.update(vals)
                final_results.append(entry)

            # Execution
            if not is_co_sim:
                # STANDARD (FAST) PATH
                pool_args = [
                    (job, i + 1, config, fast_context) for i, job in enumerate(jobs)
                ]
                wrapper = _enhanced_runner_wrapper
            else:
                # CO-SIM PATH
                pool_args = [(config, job, i + 1) for i, job in enumerate(jobs)]
                wrapper = _co_sim_runner_wrapper

            if use_concurrent:
                with multiprocessing.Pool(max_workers) as pool:
                    for job_id, parsed_params, res_data, err in pool.imap_unordered(
                        wrapper, pool_args
                    ):
                        if err:
                            logger.error(f"Job {job_id} failed: {err}")
                        else:
                            _handle_result(job_id, parsed_params, res_data)
            else:
                # Sequential Loop
                for args in pool_args:
                    job_id, parsed_params, res_data, err = wrapper(args)
                    if err:
                        logger.error(f"Job {job_id} failed: {err}")
                    else:
                        _handle_result(job_id, parsed_params, res_data)

    except Exception as e:
        logger.error(f"Sweep failed: {e}", exc_info=True)

    # Post Analysis
    if final_results and _get_optimization_tasks(config):
        pd.DataFrame(final_results).to_csv(
            os.path.join(results_dir, "requierd_tbr_summary.csv"), index=False
        )

    # Sensitivity & Plots
    _run_sensitivity_analysis(config, results_dir, jobs)

    # General Post Processing
    run_post_processing(
        config,
        None,
        os.path.join(results_dir, "post_processing"),
        results_file_path=hdf_path,
    )

    # Final Cleanup
    if not config["simulation"].get("keep_temp_files", True):
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)


def _handle_analysis_cases(config: Dict[str, Any]) -> bool:
    """Handles logic for Analysis Cases and SALib. Returns True if handled."""
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
            return True

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
            max_workers = get_safe_max_workers(
                sa_config.get("max_case_workers"),
                maximize=config["simulation"].get("maximize_workers", False),
            )
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

        return True  # End analysis_cases processing
    elif has_salib_analysis_case:
        logger.info("Detected SALib analysis case, diverting to SALib workflow...")
        run_salib_analysis(config)
        return True  # SALib workflow is self-contained, so we exit here.

    return False


def _add_baseline_jobs(config, jobs):
    """Adds baseline jobs from default_simulation_values if present."""
    # ... (Logic from original run_simulation)
    analysis_case = config.get("sensitivity_analysis", {}).get("analysis_case", {})
    defaults = analysis_case.get("default_simulation_values")
    if defaults:
        baseline = defaults.copy()
        setup = analysis_case.get("independent_variable_sampling")
        var = analysis_case.get("independent_variable")
        if var and setup:
            baseline[var] = setup
            new_jobs = generate_simulation_jobs(baseline)
            # Deduplicate
            seen = set(tuple(sorted(j.items())) for j in jobs)
            for j in new_jobs:
                t = tuple(sorted(j.items()))
                if t not in seen:
                    jobs.append(j)
                    seen.add(t)


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
