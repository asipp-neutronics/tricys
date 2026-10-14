from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional, Union

# Suppress PyTables NaturalNameWarning
try:
    import warnings

    from tables import NaturalNameWarning

    warnings.filterwarnings("ignore", category=NaturalNameWarning)
except ImportError:
    pass

import pandas as pd

from tricys.plugins.report_plugin import LegacyTaskPlugin
from tricys.plugins.storage_plugin import (
    export_hdf5_to_csv,
    process_and_store_job_result,
)
from tricys.preprocessing.compiler import (
    build_model_artifacts,
    resolve_built_model_paths,
)
from tricys.simulation.engines.offline_table import (
    mp_cosim_job_worker,
    run_co_simulation_job,
)
from tricys.simulation.engines.standard import (
    mp_fast_job_worker,
    run_fast_subprocess_job,
)
from tricys.utils.config_utils import basic_prepare_config
from tricys.utils.log_utils import setup_logging
from tricys.workflow.builder import PipelineBuilder
from tricys.workflow.context import PipelineContext

logger = logging.getLogger(__name__)

# Backward-compatibility aliases for existing imports across repository and test suites
_resolve_built_model_paths = resolve_built_model_paths
_build_model_only = build_model_artifacts
_run_fast_subprocess_job = run_fast_subprocess_job
_mp_run_fast_subprocess_job_wrapper = mp_fast_job_worker
_mp_run_co_simulation_job_wrapper = mp_cosim_job_worker
_process_h5_result = process_and_store_job_result
export_results_to_csv = export_hdf5_to_csv

HOURS_TO_SECONDS = 3600.0
SUPPORTED_TIME_BASES = ("second", "hour")


def _sim_time_to_seconds(sim_config: dict) -> dict:
    """Normalise simulation time fields to SI seconds.

    In TRICYS 2.0 (Schema 2.0), time fields (stop_time, step_size) are native SI seconds
    by default (time_base='second').
    When time_base='hour' is configured, the hour-authored values are multiplied by 3600.0.

    Idempotent: a private `_time_unit` marker records that conversion has already
    been applied ('second'), preventing multiple multiplications by 3600.0.
    """
    if not isinstance(sim_config, dict):
        return sim_config
    if sim_config.get("_time_unit") == "second":
        return sim_config

    time_base = str(sim_config.get("time_base", "second")).strip().lower()
    if time_base not in SUPPORTED_TIME_BASES:
        raise ValueError(
            f"Unsupported simulation.time_base {time_base!r}; "
            f"expected one of {SUPPORTED_TIME_BASES}."
        )

    factor = HOURS_TO_SECONDS if time_base == "hour" else 1.0
    for key in ("stop_time", "step_size"):
        if key in sim_config and sim_config[key] is not None:
            sim_config[key] = float(sim_config[key]) * factor

    sim_config["_time_unit"] = "second"
    return sim_config


__all__ = [
    "HOURS_TO_SECONDS",
    "SUPPORTED_TIME_BASES",
    "_sim_time_to_seconds",
    "run_simulation",
    "main",
    "run_sequential_sweep",
    "run_post_processing",
    "run_co_simulation_job",
    "run_fast_subprocess_job",
    "export_results_to_csv",
    "_resolve_built_model_paths",
    "_build_model_only",
    "_run_fast_subprocess_job",
    "_process_h5_result",
    "_cleanup_job_dir",
    "_should_cleanup_job_dir",
]


def _should_cleanup_job_dir(job_dir: str, config: dict) -> bool:
    """Helper to determine whether a job directory is safe to clean up."""
    temp_dir = os.path.abspath(config["paths"].get("temp_dir", "temp"))
    normalized_job_dir = os.path.abspath(job_dir)

    if "job_" not in os.path.basename(normalized_job_dir):
        return False

    try:
        return os.path.commonpath([normalized_job_dir, temp_dir]) == temp_dir
    except ValueError:
        return False


def _cleanup_job_dir(
    job_dir: str, job_id: int, retries: int = 8, delay: float = 0.5
) -> None:
    """Remove a job workspace with retries for transient Windows file locks."""
    import time

    def _on_rm_error(func, path, exc_info):
        try:
            os.chmod(path, 0o666)
            func(path)
        except Exception:
            pass

    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(job_dir, onerror=_on_rm_error)
            logger.info(
                "Cleaned up job workspace after HDF5 processing",
                extra={"job_id": job_id, "workspace": job_dir, "attempt": attempt},
            )
            return
        except Exception:
            time.sleep(delay)


def run_post_processing(
    config: Dict[str, Any],
    results_df: Optional[pd.DataFrame],
    post_processing_output_dir: str,
    results_file_path: Optional[str] = None,
) -> None:
    """Backward-compatible entry point for executing legacy post-processing tasks."""
    legacy_plugin = LegacyTaskPlugin()
    context = PipelineContext.from_config(config)
    context.results_store_path = results_file_path
    context.results_dir = (
        os.path.dirname(post_processing_output_dir)
        if post_processing_output_dir
        else context.results_dir
    )
    legacy_plugin.run(context)


def run_sequential_sweep(
    config: dict,
    jobs: List[Dict[str, Any]],
    post_job_callback: Optional[Callable[[int, Dict[str, Any], str], None]] = None,
    filter_schema: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Executes a parameter sweep sequentially using ModelicaSystem session reuse."""
    from pathlib import Path

    from tricys.core.modelica import (
        ModelicaSystem,
        build_modelica_parameter_map,
        get_om_session,
        load_modelica_package,
    )

    paths_config = config["paths"]
    sim_config = config["simulation"]
    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    os.makedirs(base_temp_dir, exist_ok=True)

    omc = get_om_session()
    result_paths = []
    try:
        package_path = os.path.abspath(paths_config["package_path"])
        if not load_modelica_package(omc, Path(package_path).as_posix()):
            raise RuntimeError("Failed to load Modelica package for sequential sweep.")

        mod = ModelicaSystem(
            fileName=Path(package_path).as_posix(),
            modelName=sim_config["model_name"],
            variableFilter=sim_config.get("variableFilter", ".*"),
        )
        mod.setSimulationOptions(
            {
                "stopTime": sim_config["stop_time"],
                "tolerance": 1e-6,
                "outputFormat": "csv",
                "stepSize": sim_config["step_size"],
            }
        )

        for i, job_params in enumerate(jobs):
            logger.info(f"Job {i + 1} of {len(jobs)}")
            job_workspace = os.path.join(base_temp_dir, f"job_{i + 1}")
            os.makedirs(job_workspace, exist_ok=True)
            result_filename = os.path.abspath(
                os.path.join(job_workspace, f"job_{i + 1}_res.csv")
            )

            param_settings = build_modelica_parameter_map(job_params)
            if param_settings:
                mod.setParameters(param_settings)

            try:
                mod.simulate(resultfile=Path(result_filename).as_posix())
                if os.path.exists(result_filename):
                    df = pd.read_csv(result_filename)
                    df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                    df.dropna(subset=["time"], inplace=True)
                    df.to_csv(result_filename, index=False)

                    if post_job_callback:
                        post_job_callback(i, job_params, result_filename)
                    result_paths.append(result_filename)
            except Exception as e:
                logger.error(f"Sequential job {i + 1} failed: {e}")
                result_paths.append("")
    finally:
        omc.sendExpression("quit()")

    return result_paths


def _run_single_pipeline(
    config: Dict[str, Any],
    export_csv: bool = False,
    base_dir: Optional[str] = None,
) -> PipelineContext:
    """Executes a single pipeline workflow."""
    logger.info("Starting simulation pipeline execution...")

    # 1. Initialize PipelineContext
    context = PipelineContext.from_config(config, base_dir=base_dir)

    # 2. Build Pipeline (inferred or explicit)
    pipeline = PipelineBuilder.build(config, export_csv=export_csv, base_dir=base_dir)

    # 3. Execute Pipeline
    pipeline.run(context)

    # 4. Optional Temp Directory Cleanup
    sim_cfg = config.get("simulation", {})
    if not sim_cfg.get("keep_temp_files", True):
        temp_dir = context.temp_dir
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
                logger.info("Cleaned up temporary workspace directory.")
            except Exception as exc:
                logger.warning(f"Failed to cleanup temp directory: {exc}")

    return context


def _run_multi_case_pipeline(
    config: Dict[str, Any],
    export_csv: bool = False,
    base_dir: Optional[str] = None,
) -> List[PipelineContext]:
    """Executes multiple analysis cases natively using isolated pipeline workspaces."""
    from tricys.analysis.report import generate_analysis_cases_summary
    from tricys.utils.config_utils import analysis_setup_analysis_cases_workspaces

    logger.info(
        "Orchestrating multi-case simulation workflow through Pipeline architecture..."
    )
    case_configs = analysis_setup_analysis_cases_workspaces(config)
    if not case_configs:
        logger.error("Unable to create analysis_cases working directories, aborting.")
        return []

    completed_contexts: List[PipelineContext] = []
    total_cases = len(case_configs)

    for i, case_info in enumerate(case_configs):
        case_name = case_info["case_data"].get("name", f"case_{i}")
        case_workspace = case_info["workspace"]
        case_config = case_info["config"]

        logger.info(
            f"=== [Case {i + 1}/{total_cases}] Running case: '{case_name}' in {case_workspace} ==="
        )
        try:
            ctx = _run_single_pipeline(
                case_config, export_csv=export_csv, base_dir=case_workspace
            )
            completed_contexts.append(ctx)
            logger.info(f"✓ Case '{case_name}' executed successfully.")
        except Exception as exc:
            logger.error(f"✗ Case '{case_name}' execution failed: {exc}", exc_info=True)

    # Cross-case comparison report and visualization
    try:
        generate_analysis_cases_summary(case_configs, config)
        logger.info("Generated multi-case cross-comparison summary reports.")
    except Exception as exc:
        logger.warning(f"Failed to generate analysis cases summary: {exc}")

    return completed_contexts


def run_simulation(
    config: Dict[str, Any],
    export_csv: bool = False,
    base_dir: Optional[str] = None,
) -> Union[PipelineContext, List[PipelineContext]]:
    """Orchestrates simulation execution using the pluggable pipeline architecture.

    Supports:
    - Single simulation runs
    - Parameter sweeps (Cartesian grid)
    - Statistical parameter sampling (Sobol, Latin Hypercube, Morris, FAST)
    - Multi-case analysis workflows (sequential or unified pipeline batch)
    """
    sa_config = config.get("sensitivity_analysis", {})
    if isinstance(sa_config, dict):
        cases = sa_config.get("analysis_cases")
        if cases and isinstance(cases, list) and len(cases) > 1:
            return _run_multi_case_pipeline(
                config, export_csv=export_csv, base_dir=base_dir
            )

    return _run_single_pipeline(config, export_csv=export_csv, base_dir=base_dir)


def run_cosimulation(
    config: Dict[str, Any], export_csv: bool = False
) -> PipelineContext:
    """Unified entry point for co-simulation workflows."""
    co_simulation_config = config.get("co_simulation")
    if not isinstance(co_simulation_config, dict):
        raise ValueError("Co-simulation entry requires a 'co_simulation' config block")

    return run_simulation(config, export_csv=export_csv)


def main(
    config_or_path: Union[str, Dict[str, Any]],
    base_dir: Optional[str] = None,
    export_csv: bool = False,
) -> None:
    """Main CLI entry point for the simulation runner."""
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
        context = run_simulation(config, export_csv=export_csv, base_dir=base_dir)
        logger.info("Main execution completed successfully")
        return context
    except Exception as exc:
        logger.error(
            "Main execution failed", exc_info=True, extra={"exception": str(exc)}
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
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV format.",
    )
    args = parser.parse_args()
    main(args.config, export_csv=args.csv)
