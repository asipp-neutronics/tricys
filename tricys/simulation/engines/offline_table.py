from __future__ import annotations

import copy
import importlib
import importlib.util
import logging
import multiprocessing
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from tricys.core.interceptor import integrate_interceptor_model
from tricys.core.modelica import (
    ModelicaSystem,
    build_modelica_parameter_map,
    get_om_session,
    load_modelica_package,
)
from tricys.utils.concurrency_utils import get_safe_max_workers
from tricys.utils.file_utils import get_unique_filename
from tricys.workflow.base import BaseSimulationEngine
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_simulation_engine

logger = logging.getLogger(__name__)


def run_co_simulation_job(config: dict, job_params: dict, job_id: int = 0) -> str:
    """Executes the full two-stage interceptor co-simulation workflow in an isolated workspace."""
    paths_config = config["paths"]
    sim_config = config["simulation"]

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    job_workspace = os.path.join(base_temp_dir, f"job_{job_id}")
    os.makedirs(job_workspace, exist_ok=True)

    omc = None
    try:
        original_package_path = os.path.abspath(paths_config["package_path"])

        if os.path.isfile(original_package_path) and not original_package_path.endswith(
            "package.mo"
        ):
            isolated_package_path = os.path.join(
                job_workspace, os.path.basename(original_package_path)
            )
            shutil.copy(original_package_path, isolated_package_path)
        else:
            original_package_dir = (
                os.path.dirname(original_package_path)
                if os.path.isfile(original_package_path)
                else original_package_path
            )
            package_dir_name = os.path.basename(original_package_dir)
            isolated_package_dir = os.path.join(job_workspace, package_dir_name)

            if os.path.exists(isolated_package_dir):
                shutil.rmtree(isolated_package_dir)
            shutil.copytree(original_package_dir, isolated_package_dir)

            if os.path.isfile(original_package_path):
                isolated_package_path = os.path.join(
                    isolated_package_dir, os.path.basename(original_package_path)
                )
            else:
                isolated_package_path = os.path.join(isolated_package_dir, "package.mo")

        isolated_temp_dir = job_workspace
        results_dir = os.path.abspath(paths_config["results_dir"])
        os.makedirs(results_dir, exist_ok=True)

        co_sim_config = config["co_simulation"]
        mode = co_sim_config.get("mode", "interceptor")
        handlers = copy.deepcopy(co_sim_config.get("handlers", []))
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

        # Copy asset files specified in handler params
        for handler_config in handlers:
            if "params" in handler_config:
                for param_key, param_value in list(handler_config["params"].items()):
                    if isinstance(param_value, str) and param_key.endswith("_path"):
                        original_asset_path = Path(os.path.abspath(param_value))
                        if not original_asset_path.exists():
                            logger.warning(
                                f"Asset file '{original_asset_path}' not found. Skipping copy."
                            )
                            continue
                        asset_dir_name = original_asset_path.parent.name
                        dest_dir = Path(job_workspace) / asset_dir_name
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_path = dest_dir / original_asset_path.name
                        if not dest_path.exists():
                            shutil.copy(original_asset_path, dest_path)
                        handler_config["params"][param_key] = dest_path.as_posix()

        # Step 1: Detect input ports for submodels
        all_input_vars = []
        for handler_config in handlers:
            submodel_name = handler_config["submodel_name"]
            instance_name = handler_config["instance_name"]
            components = omc.sendExpression(f"getComponents({submodel_name})")
            input_ports = [
                {"name": c[1], "dim": int(c[11][0]) if c[11] else 1}
                for c in (components or [])
                if c[0] == "Modelica.Blocks.Interfaces.RealInput"
            ]
            for port in input_ports:
                full_name = f"{instance_name}.{port['name']}".replace(".", "\\.")
                if port["dim"] > 1:
                    full_name += f"\\[[1-{port['dim']}]\\]"
                all_input_vars.append(full_name)

        variable_filter = (
            "time|" + "|".join(all_input_vars) if all_input_vars else "time"
        )

        # Step 2: Primary baseline simulation to obtain boundary inputs
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
            mod.setParameters(param_settings)

        primary_result_filename = get_unique_filename(
            isolated_temp_dir, "primary_inputs.csv"
        )
        mod.simulate(resultfile=Path(primary_result_filename).as_posix())

        if os.path.exists(primary_result_filename):
            try:
                df = pd.read_csv(primary_result_filename)
                df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                df.dropna(subset=["time"], inplace=True)
                df.to_csv(primary_result_filename, index=False)
            except Exception as e:
                logger.warning(f"Failed to clean primary result file: {e}")

        # Step 3: Run external handlers
        interception_configs = []
        for handler_config in handlers:
            handler_function_name = handler_config["handler_function"]
            module = None

            if "handler_script_path" in handler_config:
                script_path = Path(handler_config["handler_script_path"]).resolve()
                if not script_path.is_file():
                    raise FileNotFoundError(
                        f"Handler script not found at {script_path}"
                    )
                spec = importlib.util.spec_from_file_location(
                    script_path.stem, script_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
            elif "handler_module" in handler_config:
                if os.getcwd() not in sys.path:
                    sys.path.insert(0, os.getcwd())
                module = importlib.import_module(handler_config["handler_module"])

            if not module:
                raise ImportError(f"Failed to load handler for {handler_config}")

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

            if not os.path.exists(co_sim_output_filename):
                raise FileNotFoundError(
                    f"Co-simulation handler did not create output CSV: {co_sim_output_filename}"
                )

            interception_config = {
                "submodel_name": handler_config["submodel_name"],
                "instance_name": handler_config["instance_name"],
                "csv_uri": Path(os.path.abspath(co_sim_output_filename)).as_posix(),
                "output_placeholder": output_placeholder,
                "mode": mode,
            }
            interception_configs.append(interception_config)

        # Step 4: Inject interceptors or replace submodels
        intercepted_model_paths = integrate_interceptor_model(
            package_path=isolated_package_path,
            model_name=model_name,
            interception_configs=interception_configs,
        )

        verif_config = config["simulation"].get("variableFilter", ".*")

        if mode == "replacement":
            final_model_name = model_name
            final_model_file = isolated_package_path
        else:
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

        if os.path.exists(default_result_path):
            try:
                df = pd.read_csv(default_result_path)
                df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                df.dropna(subset=["time"], inplace=True)
                df.to_csv(default_result_path, index=False)
            except Exception:
                pass

        if not os.path.exists(default_result_path):
            raise FileNotFoundError(
                f"Co-simulation failed to produce {default_result_path}"
            )

        return Path(default_result_path).as_posix()
    except Exception as e:
        logger.error(
            f"Co-simulation workflow failed for job {job_id}: {e}", exc_info=True
        )
        return ""
    finally:
        if "mod" in locals() and mod and hasattr(mod, "omc"):
            try:
                mod.omc.sendExpression("quit()")
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
            except Exception:
                pass


def mp_cosim_job_worker(
    args: Tuple[dict, dict, int],
) -> Tuple[int, dict, str, Optional[str]]:
    config, job_params, job_id = args
    try:
        res = run_co_simulation_job(config, job_params, job_id=job_id)
        if not res or not os.path.exists(res):
            return job_id, job_params, "", f"Job {job_id} produced no result"
        return job_id, job_params, res, None
    except Exception as exc:
        return job_id, job_params, "", str(exc)


@register_simulation_engine("offline_table")
@register_simulation_engine("offline_cosim")
class OfflineTableEngine(BaseSimulationEngine):
    """Core simulation engine for offline table / interceptor co-simulation."""

    name = "offline_table"

    def execute(self, context: PipelineContext) -> None:
        jobs = context.jobs or [{}]
        sim_cfg = context.config.get("simulation", {})
        use_concurrent = sim_cfg.get("concurrent", False)
        max_workers = get_safe_max_workers(
            sim_cfg.get("max_workers"),
            maximize=sim_cfg.get("maximize_workers", False),
            task_count=len(jobs),
        )

        logger.info(
            f"OfflineTableEngine: Running {len(jobs)} co-simulation jobs (concurrent={use_concurrent})"
        )

        raw_results = []
        if use_concurrent and len(jobs) > 1:
            pool_args = [
                (context.config, job_params, i) for i, job_params in enumerate(jobs, 1)
            ]
            with multiprocessing.Pool(processes=max_workers) as pool:
                for job_id, job_p, res_path, err in pool.imap_unordered(
                    mp_cosim_job_worker, pool_args
                ):
                    if err:
                        logger.error(f"CoSim Job {job_id} failed: {err}")
                    else:
                        raw_results.append(
                            {"job_id": job_id, "params": job_p, "result_path": res_path}
                        )
        else:
            for i, job_params in enumerate(jobs, 1):
                logger.info(f"Running co-simulation job {i}/{len(jobs)}")
                res_path = run_co_simulation_job(context.config, job_params, job_id=i)
                if res_path:
                    raw_results.append(
                        {"job_id": i, "params": job_params, "result_path": res_path}
                    )

        context.raw_results = raw_results
        logger.info(
            f"OfflineTableEngine: Completed {len(raw_results)}/{len(jobs)} jobs successfully"
        )


# Backward compatibility alias
OfflineCoSimEngine = OfflineTableEngine
