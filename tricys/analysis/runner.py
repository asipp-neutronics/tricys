from __future__ import annotations

import json
import logging
import multiprocessing
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from tricys.analysis.cases import AnalysisCasesManager
from tricys.analysis.metric import (
    build_single_job_summary_df,
    calculate_single_job_metrics,
)
from tricys.analysis.optimizer import (
    BisectionOptimizer,
    get_optimization_tasks,
)
from tricys.analysis.report import (
    consolidate_reports,
    retry_ai_analysis,
)
from tricys.core.jobs import generate_simulation_jobs
from tricys.plugins.sensitivity_plugin import execute_sensitivity_analysis
from tricys.simulation.engines.offline_cosim import run_co_simulation_job
from tricys.simulation.engines.standard import run_fast_subprocess_job
from tricys.simulation.simulation import (
    _build_model_only,
    run_post_processing,
)
from tricys.utils.concurrency_utils import get_safe_max_workers
from tricys.utils.config_utils import (
    analysis_setup_analysis_cases_workspaces,
    analysis_validate_config,
)
from tricys.utils.file_utils import get_unique_filename
from tricys.utils.log_capture import LogCapture
from tricys.utils.log_utils import (
    restore_configs_from_log,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
_run_fast_subprocess_job = run_fast_subprocess_job
_get_optimization_tasks = get_optimization_tasks


def _run_bisection_search_fast(
    config: Dict[str, Any],
    job_id_prefix: str,
    optimization_metric_name: str,
    fast_context: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    return BisectionOptimizer.run_bisection_search_fast(
        config, job_id_prefix, optimization_metric_name, fast_context
    )


def _run_optimization_tasks_fast(
    config: dict, job_params: dict, job_id: int, fast_context: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    return BisectionOptimizer.run_optimization_tasks_fast(
        config, job_params, job_id, fast_context
    )


def _handle_analysis_cases(config: Dict[str, Any]) -> bool:
    return AnalysisCasesManager.handle_analysis_cases(config, run_simulation)


def _run_sensitivity_analysis(
    config: Dict[str, Any], run_results_dir: str, jobs: List[Dict[str, Any]]
) -> None:
    execute_sensitivity_analysis(config, run_results_dir, jobs)


def _add_baseline_jobs(config: Dict[str, Any], jobs: List[Dict[str, Any]]) -> None:
    """Adds baseline jobs from default_simulation_values if present."""
    analysis_case = config.get("sensitivity_analysis", {}).get("analysis_case", {})
    defaults = analysis_case.get("default_simulation_values")
    if defaults:
        baseline = defaults.copy()
        setup = analysis_case.get("independent_variable_sampling")
        var = analysis_case.get("independent_variable")
        if var and setup:
            baseline[var] = setup
            new_jobs = generate_simulation_jobs(baseline)
            seen = set(tuple(sorted(j.items())) for j in jobs)
            for j in new_jobs:
                t = tuple(sorted(j.items()))
                if t not in seen:
                    jobs.append(j)
                    seen.add(t)


def _enhanced_runner_wrapper(
    args: Tuple[Dict[str, Any], int, Dict[str, Any], Dict[str, Any]],
) -> Tuple[int, Dict[str, Any], Optional[Tuple[Any, Any, str]], Optional[str]]:
    """Worker function for Enhanced Mode (Standard/Non-CoSim)."""
    job_params, job_id, config, fast_context = args
    try:
        result_path = run_fast_subprocess_job(
            job_params,
            job_id,
            fast_context["exe"],
            fast_context["xml"],
            fast_context["om_bin"],
            fast_context["temp_dir"],
            config["simulation"],
            variable_filter=config["simulation"].get("variableFilter"),
            inplace_execution=False,
        )

        opt_params, opt_values = {}, {}
        if result_path and os.path.exists(result_path):
            opt_params, opt_values = BisectionOptimizer.run_optimization_tasks_fast(
                config, job_params, job_id, fast_context
            )

        return job_id, job_params, (opt_params, opt_values, result_path), None
    except Exception as e:
        return job_id, job_params, None, str(e)


def _co_sim_runner_wrapper(
    args: Tuple[Dict[str, Any], Dict[str, Any], int],
) -> Tuple[int, Dict[str, Any], Optional[Tuple[Any, Any, str]], Optional[str]]:
    """Worker function for Co-Simulation."""
    config, job_params, job_id = args
    try:
        result_path = run_co_simulation_job(config, job_params, job_id)
        return job_id, job_params, ({}, {}, result_path), None
    except Exception as e:
        return job_id, job_params, None, str(e)


def run_simulation(config: Dict[str, Any]) -> None:
    """[Deprecated] Orchestrates the simulation analysis workflow. Forwarded to simulation.run_simulation."""
    import warnings

    warnings.warn(
        "tricys.analysis.runner.run_simulation is deprecated. "
        "Use tricys.simulation.simulation.run_simulation.",
        DeprecationWarning,
        stacklevel=2,
    )
    from tricys.simulation.simulation import run_simulation as pipeline_run_sim

    pipeline_run_sim(config)

    jobs = generate_simulation_jobs(config.get("simulation_parameters", {}))
    _add_baseline_jobs(config, jobs)

    try:
        results_dir = os.path.abspath(config["paths"]["results_dir"])
        temp_dir = os.path.abspath(config["paths"].get("temp_dir", "temp"))
    except KeyError as e:
        logger.error(f"Missing config path: {e}")
        sys.exit(1)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    hdf_path = get_unique_filename(results_dir, "sweep_results.h5")
    is_co_sim = config.get("co_simulation") is not None
    use_concurrent = config["simulation"].get("concurrent", False)
    maximize_workers = config["simulation"].get("maximize_workers", False)
    max_workers = get_safe_max_workers(
        config["simulation"].get("max_workers"),
        maximize=maximize_workers,
        task_count=len(jobs),
    )

    logger.info(
        f"Starting Simulation Analysis. CoSim={is_co_sim}, Concurrent={use_concurrent}"
    )

    fast_context = {}
    if not is_co_sim:
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

    final_results = []
    metrics_definition = config.get("sensitivity_analysis", {}).get(
        "metrics_definition", {}
    )

    with LogCapture() as log_handler:
        try:
            with pd.HDFStore(hdf_path, mode="w", complib="blosc", complevel=9) as store:
                meta_df = pd.DataFrame(jobs)
                if not meta_df.empty:
                    meta_df["job_id"] = range(1, len(jobs) + 1)
                    for col in meta_df.select_dtypes(
                        include=["object", "string"]
                    ).columns:
                        meta_df[col] = meta_df[col].astype(object)
                    store.put("jobs", meta_df, format="table", data_columns=True)

                def _handle_result(job_id, params, result_data):
                    opts, vals, res_path = result_data
                    if res_path and os.path.exists(res_path):
                        try:
                            df = pd.read_csv(res_path)
                            df["job_id"] = job_id
                            store.append("results", df, index=False, data_columns=True)
                            job_dir = os.path.dirname(res_path)
                            if "job_" in os.path.basename(job_dir):
                                shutil.rmtree(job_dir, ignore_errors=True)
                        except Exception as write_err:
                            logger.error(
                                f"HDF5 write failed for job {job_id}: {write_err}"
                            )

                    if res_path and os.path.exists(res_path) and metrics_definition:
                        try:
                            df_metric = pd.read_csv(res_path)
                            single_metrics = calculate_single_job_metrics(
                                df_metric, metrics_definition
                            )
                            if single_metrics:
                                summary_df = build_single_job_summary_df(
                                    job_id, single_metrics, metrics_definition
                                )
                                if not summary_df.empty:
                                    store.append(
                                        "summary",
                                        summary_df,
                                        index=False,
                                        data_columns=True,
                                    )
                        except Exception as m_err:
                            logger.warning(
                                f"Failed to calculate/save summary metrics for job {job_id}: {m_err}"
                            )

                    entry = params.copy()
                    entry.update(opts)
                    entry.update(vals)
                    final_results.append(entry)

                if not is_co_sim:
                    pool_args = [
                        (job, i + 1, config, fast_context) for i, job in enumerate(jobs)
                    ]
                    wrapper = _enhanced_runner_wrapper
                else:
                    pool_args = [(config, job, i + 1) for i, job in enumerate(jobs)]
                    wrapper = _co_sim_runner_wrapper

                completed_count = 0
                total_jobs = len(jobs)

                if use_concurrent:
                    with multiprocessing.Pool(max_workers) as pool:
                        for job_id, parsed_params, res_data, err in pool.imap_unordered(
                            wrapper, pool_args
                        ):
                            completed_count += 1
                            if err:
                                logger.error(f"Job {job_id} failed: {err}")
                            else:
                                _handle_result(job_id, parsed_params, res_data)
                            logger.info(f"Job {completed_count} of {total_jobs}")
                else:
                    for args in pool_args:
                        job_id, parsed_params, res_data, err = wrapper(args)
                        completed_count += 1
                        if err:
                            logger.error(f"Job {job_id} failed: {err}")
                        else:
                            _handle_result(job_id, parsed_params, res_data)
                        logger.info(f"Job {completed_count} of {total_jobs}")

                try:
                    config_df = pd.DataFrame(
                        {"config_json": [json.dumps(config)]}
                    ).astype(object)
                    store.put("config", config_df, format="fixed")

                    logs_json = log_handler.to_json()
                    log_df = pd.DataFrame({"log": [logs_json]}).astype(object)
                    store.put("log", log_df, format="fixed")
                except Exception as save_err:
                    logger.warning(f"Failed to save finalize data to HDF5: {save_err}")

        except Exception as e:
            logger.error(f"Sweep failed: {e}", exc_info=True)

    if final_results and get_optimization_tasks(config):
        pd.DataFrame(final_results).to_csv(
            os.path.join(results_dir, "requierd_tbr_summary.csv"), index=False
        )

    execute_sensitivity_analysis(config, results_dir, jobs)

    run_post_processing(
        config,
        None,
        os.path.join(results_dir, "post_processing"),
        results_file_path=hdf_path,
    )

    if not config["simulation"].get("keep_temp_files", True):
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)


def retry_analysis(timestamp: str) -> None:
    """Retries a failed AI analysis for a given run timestamp."""
    config, original_config = restore_configs_from_log(timestamp)
    if not config or not original_config:
        sys.exit(1)

    config["run_timestamp"] = timestamp
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger.info(
        f"Successfully restored configuration for timestamp {timestamp} for retry."
    )

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
    """Main entry point for a simulation analysis run (delegates to unified simulation.main)."""
    from tricys.simulation.simulation import main as simulation_main

    simulation_main(config_or_path, base_dir=base_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.json")
    args = parser.parse_args()
    main(args.config)
