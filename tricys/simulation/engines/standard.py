from __future__ import annotations

import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd

from tricys.core.modelica import (
    ModelicaSystem,
    build_modelica_parameter_map,
    get_om_session,
    load_modelica_package,
)
from tricys.utils.concurrency_utils import get_safe_max_workers
from tricys.workflow.base import BaseSimulationEngine
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_simulation_engine

logger = logging.getLogger(__name__)


def run_fast_subprocess_job(
    job_params: dict,
    job_id: int,
    exe_source: str,
    xml_source: str,
    om_bin_path: str,
    base_temp_dir: str,
    sim_config: dict,
    variable_filter: Optional[str] = None,
    inplace_execution: bool = False,
) -> str:
    """Executes a single simulation job using pre-compiled Modelica executable."""
    import glob
    import shutil
    import subprocess
    import sys

    import pandas as pd

    job_workspace = os.path.join(base_temp_dir, f"job_{job_id}")
    os.makedirs(job_workspace, exist_ok=True)

    if inplace_execution:
        run_cwd = os.path.dirname(exe_source)
        run_exe_path = exe_source
        result_filename = os.path.abspath(
            os.path.join(job_workspace, f"job_{job_id}_res.csv")
        )
    else:
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

    override_pairs = [
        f"{k}={v}"
        for k, v in job_params.items()
        if k not in ("name", "job_id") and not str(k).startswith("_")
    ]

    from tricys.simulation.simulation import _sim_time_to_seconds

    _sim_time_to_seconds(sim_config)

    output_fmt = sim_config.get("outputFormat", "csv")
    cmd = [
        run_exe_path,
        "-r",
        result_filename,
        f"-outputFormat={output_fmt}",
        f"-stopTime={sim_config['stop_time']}",
        f"-stepSize={sim_config['step_size']}",
    ]

    if "tolerance" in sim_config and sim_config["tolerance"] is not None:
        cmd.append(f"-tolerance={sim_config['tolerance']}")

    if "start_time" in sim_config and float(sim_config.get("start_time", 0.0)) > 0:
        cmd.append(f"-startTime={sim_config['start_time']}")

    if "method" in sim_config and sim_config["method"]:
        cmd.append(f"-s={sim_config['method']}")

    override_file = sim_config.get("override_file") or (
        job_params.get("override_file") if isinstance(job_params, dict) else None
    )
    if override_file and os.path.exists(override_file):
        cmd.append(f"-overrideFile={override_file}")
    elif override_pairs:
        override_str = ",".join(override_pairs)
        cmd.extend(["-override", override_str])
    if variable_filter:
        cmd.append(f"-variableFilter={variable_filter}")

    if sim_config.get("simflags"):
        import shlex

        cmd.extend(shlex.split(sim_config["simflags"]))

    env = os.environ.copy()
    if sys.platform == "win32":
        env["PATH"] = om_bin_path + os.pathsep + env.get("PATH", "")

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
        return ""
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        return ""


def mp_fast_job_worker(
    args: Tuple[dict, int, dict],
) -> Tuple[int, dict, str, Optional[str]]:
    job_params, job_id, kwargs = args
    try:
        kwargs_copy = kwargs.copy()
        config = kwargs_copy.pop("config", None)
        fast_context = kwargs_copy.pop("fast_context", None)

        res = run_fast_subprocess_job(job_params, job_id, **kwargs_copy)
        if not res or not os.path.exists(res):
            return job_id, job_params, "", f"Job {job_id} generated no output"

        # Controlled bisection / nested optimization in execution layer
        if config and fast_context:
            try:
                from tricys.analysis.optimizer import OptimizationEngine

                if OptimizationEngine.has_optimization(config):
                    opt_params, opt_values = OptimizationEngine.execute_tasks(
                        config, job_params, job_id, fast_context
                    )
                    job_params = {**job_params, **opt_params, **opt_values}
            except Exception as opt_exc:
                logger.warning(f"Job {job_id}: Optimization task failed: {opt_exc}")

        return job_id, job_params, res, None
    except Exception as exc:
        return job_id, job_params, "", str(exc)


@register_simulation_engine("standard")
class StandardSimulationEngine(BaseSimulationEngine):
    """Core numerical simulation engine for single runs, sweeps, and multi-process batch execution."""

    name = "standard"

    def __init__(
        self,
        stop_time: Optional[float] = None,
        step_size: Optional[float] = None,
        start_time: float = 0.0,
        time_base: Optional[str] = None,
        solver: Optional[str] = None,
        tolerance: Optional[float] = None,
        output_format: Optional[str] = None,
        variable_filter: Optional[str] = None,
        parallel: Optional[bool] = None,
        max_workers: Optional[int] = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.stop_time = stop_time
        self.step_size = step_size
        self.start_time = start_time
        self.time_base = time_base
        self.solver = solver
        self.tolerance = tolerance
        self.output_format = output_format
        self.variable_filter = variable_filter
        self.parallel = parallel
        self.max_workers = max_workers

    def execute(self, context: PipelineContext) -> None:
        sim_cfg = dict(context.config.get("simulation", {}))
        scenario_cfg = context.config.get("scenario", {})
        task_cfg = context.config.get("task", {})
        jobs = context.jobs or [{}]

        time_base = (
            str(
                self.time_base
                or sim_cfg.get("time_base")
                or scenario_cfg.get("time_base")
                or getattr(context, "time_base", "second")
                or "second"
            )
            .strip()
            .lower()
        )
        sim_cfg["time_base"] = time_base
        factor = 3600.0 if time_base == "second" else 1.0

        # Explicit-First with Context-Fallback parameter resolution
        if self.stop_time is not None:
            sim_cfg["stop_time"] = float(self.stop_time) * factor
            sim_cfg["_time_unit"] = "second"
        elif "stop_time" not in sim_cfg and "stopTime" in scenario_cfg:
            sim_cfg["stop_time"] = float(scenario_cfg["stopTime"]) * factor
            sim_cfg["_time_unit"] = "second"

        if self.step_size is not None:
            sim_cfg["step_size"] = float(self.step_size) * factor
        elif "step_size" not in sim_cfg and "stepSize" in scenario_cfg:
            sim_cfg["step_size"] = float(scenario_cfg["stepSize"]) * factor

        if self.start_time is not None and float(self.start_time) > 0:
            sim_cfg["start_time"] = float(self.start_time) * factor
        elif "start_time" not in sim_cfg and "startTime" in scenario_cfg:
            sim_cfg["start_time"] = float(scenario_cfg["startTime"]) * factor

        if self.solver is not None:
            sim_cfg["method"] = self.solver
        elif "method" not in sim_cfg and "method" in scenario_cfg:
            sim_cfg["method"] = scenario_cfg["method"]

        if self.tolerance is not None:
            sim_cfg["tolerance"] = float(self.tolerance)
        elif "tolerance" not in sim_cfg and "tolerance" in scenario_cfg:
            sim_cfg["tolerance"] = float(scenario_cfg["tolerance"])

        if self.output_format is not None:
            sim_cfg["outputFormat"] = self.output_format
        elif "outputFormat" not in sim_cfg and "outputFormat" in scenario_cfg:
            sim_cfg["outputFormat"] = scenario_cfg["outputFormat"]

        if self.variable_filter is not None:
            sim_cfg["variableFilter"] = self.variable_filter
        elif "variableFilter" not in sim_cfg and "variableFilter" in scenario_cfg:
            sim_cfg["variableFilter"] = scenario_cfg["variableFilter"]

        # Keep context time bounds synchronized
        if "stop_time" in sim_cfg:
            context.stop_time = float(sim_cfg["stop_time"])
        if "step_size" in sim_cfg:
            context.step_size = float(sim_cfg["step_size"])

        if self.parallel is not None:
            use_concurrent = bool(self.parallel)
        else:
            from tricys.analysis.optimizer import OptimizationEngine

            use_concurrent = (
                sim_cfg.get("concurrent", False)
                or (isinstance(task_cfg, dict) and task_cfg.get("parallel", False))
                or OptimizationEngine.has_optimization(context.config)
            )
        maximize_workers = sim_cfg.get("maximize_workers", False)
        configured_workers = (
            self.max_workers
            or sim_cfg.get("max_workers")
            or (task_cfg.get("max_workers") if isinstance(task_cfg, dict) else None)
        )
        max_workers = get_safe_max_workers(
            configured_workers,
            maximize=maximize_workers,
            task_count=len(jobs),
        )

        logger.info(
            f"StandardSimulationEngine: Executing {len(jobs)} jobs (concurrent={use_concurrent}, workers={max_workers})"
        )

        raw_results = []

        if use_concurrent:
            # Ensure compilation artifacts exist
            artifacts = context.compiled_artifacts
            if not artifacts or "exe" not in artifacts:
                from tricys.preprocessing.compiler import build_model_artifacts

                exe, xml, om_bin = build_model_artifacts(
                    context.package_path,
                    context.model_name,
                    context.temp_dir,
                )
                artifacts = {"exe": exe, "xml": xml, "om_bin": om_bin}
                context.compiled_artifacts = artifacts

            fast_context = {
                "exe": artifacts["exe"],
                "xml": artifacts["xml"],
                "om_bin": artifacts["om_bin"],
                "temp_dir": context.temp_dir,
            }

            pool_args = []
            for i, job_params in enumerate(jobs, 1):
                kwargs = {
                    "exe_source": artifacts["exe"],
                    "xml_source": artifacts["xml"],
                    "om_bin_path": artifacts["om_bin"],
                    "base_temp_dir": context.temp_dir,
                    "sim_config": sim_cfg,
                    "variable_filter": sim_cfg.get("variableFilter"),
                    "inplace_execution": True,
                    "config": context.config,
                    "fast_context": fast_context,
                }
                pool_args.append((job_params, i, kwargs))

            completed_count = 0
            total_jobs = len(jobs)

            with multiprocessing.Pool(processes=max_workers) as pool:
                for job_id, job_p, result_path, error in pool.imap_unordered(
                    mp_fast_job_worker, pool_args
                ):
                    completed_count += 1
                    logger.info(f"Job {completed_count} of {total_jobs}")
                    if error:
                        logger.error(f"Job {job_id} failed: {error}")
                    else:
                        df_res = None
                        if result_path and os.path.exists(result_path):
                            try:
                                df_res = pd.read_csv(result_path)
                            except Exception:
                                pass
                        raw_results.append(
                            {
                                "job_id": job_id,
                                "params": job_p,
                                "result_path": result_path,
                                "dataframe": df_res,
                            }
                        )

            # Record optimization results if optimization tasks were active
            from tricys.analysis.optimizer import OptimizationEngine

            if OptimizationEngine.has_optimization(context.config):
                opt_entries = [r["params"] for r in raw_results if r.get("params")]
                if opt_entries:
                    results_dir = os.path.abspath(context.results_dir)
                    os.makedirs(results_dir, exist_ok=True)
                    opt_df = pd.DataFrame(opt_entries)
                    # Write both modern correct spelling and legacy typo spelling for compatibility
                    opt_df.to_csv(
                        os.path.join(results_dir, "required_tbr_summary.csv"),
                        index=False,
                    )
                    opt_df.to_csv(
                        os.path.join(results_dir, "requierd_tbr_summary.csv"),
                        index=False,
                    )
                    logger.info(
                        f"Exported execution-layer optimization summary to {results_dir}"
                    )
                    context.artifacts["optimization_summary"] = opt_df
        else:
            # Sequential execution using ModelicaSystem session reuse
            omc = get_om_session()
            try:
                if not load_modelica_package(
                    omc, Path(context.package_path).as_posix()
                ):
                    raise RuntimeError(
                        f"Failed to load Modelica package at {context.package_path}"
                    )

                mod = ModelicaSystem(
                    fileName=Path(context.package_path).as_posix(),
                    modelName=context.model_name,
                )

                sim_options = {
                    "stopTime": context.stop_time,
                    "tolerance": float(sim_cfg.get("tolerance", 1e-6)),
                    "outputFormat": sim_cfg.get("outputFormat", "csv"),
                    "stepSize": context.step_size,
                    "solver": sim_cfg.get("method", "dassl"),
                }
                if float(sim_cfg.get("start_time", 0.0)) > 0:
                    sim_options["startTime"] = float(sim_cfg["start_time"])
                override_file = sim_cfg.get("override_file")
                if override_file and os.path.exists(override_file):
                    sim_options["simflags"] = (
                        f"{sim_options.get('simflags', '')} -overrideFile {override_file}".strip()
                    )
                mod.setSimulationOptions(sim_options)

                for i, job_params in enumerate(jobs, 1):
                    logger.info(f"Job {i} of {len(jobs)}")
                    job_workspace = os.path.join(context.temp_dir, f"job_{i}")
                    os.makedirs(job_workspace, exist_ok=True)
                    result_filename = os.path.abspath(
                        os.path.join(job_workspace, f"job_{i}_res.csv")
                    )

                    param_settings = build_modelica_parameter_map(job_params)
                    if param_settings:
                        mod.setParameters(param_settings)

                    try:
                        simargs = {}
                        if sim_cfg.get("variableFilter"):
                            simargs["variableFilter"] = sim_cfg["variableFilter"]
                        mod.simulate(
                            resultfile=Path(result_filename).as_posix(),
                            simargs=simargs if simargs else None,
                        )
                        df_res = None
                        if os.path.exists(result_filename):
                            try:
                                df_res = pd.read_csv(result_filename)
                            except Exception:
                                pass
                        raw_results.append(
                            {
                                "job_id": i,
                                "params": job_params,
                                "result_path": result_filename,
                                "dataframe": df_res,
                            }
                        )
                    except Exception as exc:
                        logger.error(f"Sequential job {i} failed: {exc}")
            finally:
                omc.sendExpression("quit()")

        context.raw_results = raw_results
        context.sync_typed_results()
        logger.info(
            f"StandardSimulationEngine: Completed {len(raw_results)}/{len(jobs)} jobs successfully"
        )


# Backward compatibility import
