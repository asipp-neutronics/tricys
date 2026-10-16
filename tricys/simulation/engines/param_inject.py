from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List

import pandas as pd

from tricys.core.modelica import get_om_session
from tricys.core.submodel_replacer import (
    execute_subsystem_handler,
    replace_submodel_in_package,
    validate_port_and_connection_compatibility,
)
from tricys.workflow.base import BaseSimulationEngine
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_simulation_engine

logger = logging.getLogger(__name__)


@register_simulation_engine("param_inject")
@register_simulation_engine("parameter_injection")
class ParamInjectSimulationEngine(BaseSimulationEngine):
    """Simulation engine dedicated to high-fidelity parameter injection co-simulation.

    Workflow:
    1. Workspace isolation to prevent modifying original user models.
    2. Optional unified task-level solver handler execution (e.g. h2iso MESH distillation solver).
    3. Strict port & connection compatibility check between target and substitute submodels.
    4. Seamless submodel replacement in the isolated Modelica package.
    5. Native -overrideFile parameter injection for high-speed dynamic simulation.
    """

    name = "param_inject"

    def execute(self, context: PipelineContext) -> None:
        sim_cfg = context.config.get("simulation", {})
        task_cfg = context.config.get("task", {})
        workspace_dir = os.path.abspath(context.temp_dir)
        os.makedirs(workspace_dir, exist_ok=True)

        logger.info(
            "ParamInjectSimulationEngine: Initializing parameter injection co-simulation",
            extra={"workspace": workspace_dir, "model": context.model_name},
        )

        # Step 1: Ensure package is isolated in workspace
        orig_package_path = os.path.abspath(context.package_path)
        isolated_package_path = os.path.join(
            workspace_dir, os.path.basename(orig_package_path)
        )
        if orig_package_path != isolated_package_path:
            shutil.copy2(orig_package_path, isolated_package_path)
            context.package_path = isolated_package_path
            logger.info(f"Isolated Modelica package copied to {isolated_package_path}")

        # Step 2: Task-level external parameter injection handler execution (if injection_type == "handler")
        injection_cfg = task_cfg.get("injection") or {}
        injection_type = (
            injection_cfg.get("type")
            or task_cfg.get("injection_type")
            or (
                "handler"
                if (
                    task_cfg.get("handler_script")
                    or injection_cfg.get("handler_script")
                )
                else "file"
            )
        )
        task_handler_script = injection_cfg.get("handler_script") or task_cfg.get(
            "handler_script"
        )
        task_handler_params = (
            injection_cfg.get("params") or task_cfg.get("handler_params") or {}
        )

        if injection_type == "handler" and task_handler_script:
            pkg_dir = os.path.dirname(orig_package_path)
            if not os.path.isabs(task_handler_script):
                handler_cands = [
                    os.path.join(pkg_dir, task_handler_script),
                    os.path.abspath(task_handler_script),
                    os.path.join(workspace_dir, task_handler_script),
                ]
                for cand in handler_cands:
                    if os.path.exists(cand):
                        task_handler_script = cand
                        break
            if os.path.exists(task_handler_script):
                logger.info(
                    f"ParamInjectSimulationEngine: Executing unified task-level solver handler: {os.path.basename(task_handler_script)}"
                )
                execute_subsystem_handler(
                    task_handler_script,
                    task_handler_params,
                    workspace_dir=workspace_dir,
                )
            else:
                logger.warning(
                    f"ParamInjectSimulationEngine: Task handler script not found at {task_handler_script}"
                )

        # Step 3: Iterate subsystems for port validation and model replacement
        subsystems = task_cfg.get("subsystems", [])
        if not subsystems and "co_simulation" in context.config:
            subsystems = context.config["co_simulation"].get("handlers", [])

        loaded_aux_files: List[str] = []

        for sub in subsystems:
            target_model = sub.get("target") or sub.get("submodel_name")
            instance_name = sub.get("instance") or sub.get("instance_name")
            substitute_model = sub.get("substitute_model") or target_model
            substitute_file = sub.get("substitute_file") or sub.get("model_file")
            aux_files = sub.get("aux_files", [])

            # Resolve paths relative to original workspace / package directory
            pkg_dir = os.path.dirname(orig_package_path)
            if substitute_file and not os.path.isabs(substitute_file):
                sub_candidates = [
                    os.path.join(pkg_dir, substitute_file),
                    os.path.abspath(substitute_file),
                    os.path.join(workspace_dir, substitute_file),
                ]
                for cand in sub_candidates:
                    if os.path.exists(cand):
                        substitute_file = cand
                        break

            # 3.1 Copy and track auxiliary dependent Modelica models
            resolved_aux: List[str] = []
            for aux_item in aux_files:
                aux_path = (
                    aux_item
                    if os.path.isabs(aux_item)
                    else os.path.join(pkg_dir, aux_item)
                )
                if os.path.exists(aux_path):
                    dst_aux = os.path.join(workspace_dir, os.path.basename(aux_path))
                    if aux_path != dst_aux:
                        shutil.copy2(aux_path, dst_aux)
                    resolved_aux.append(dst_aux)
                    if dst_aux not in loaded_aux_files:
                        loaded_aux_files.append(dst_aux)

            # Copy substitute file to workspace if it exists
            resolved_sub_file = None
            if substitute_file and os.path.exists(substitute_file):
                dst_sub = os.path.join(workspace_dir, os.path.basename(substitute_file))
                if substitute_file != dst_sub:
                    shutil.copy2(substitute_file, dst_sub)
                resolved_sub_file = dst_sub
                if dst_sub not in loaded_aux_files:
                    loaded_aux_files.append(dst_sub)

            # 3.2 Strict Port & Connection Compatibility Check
            logger.info(
                f"Validating submodel port and connection compatibility for '{instance_name}' ({target_model} -> {substitute_model})"
            )
            report = validate_port_and_connection_compatibility(
                target_model_name=target_model,
                substitute_model_name=substitute_model,
                instance_name=instance_name,
                parent_system_name=context.model_name,
                system_code_or_path=context.package_path,
                substitute_code_or_path=resolved_sub_file or context.package_path,
            )
            context.set_metadata(f"submodel_compat_{instance_name}", report)

            # 3.3 Submodel replacement in package code
            if substitute_model != target_model or resolved_sub_file:
                replace_submodel_in_package(
                    package_path=context.package_path,
                    target_model_name=target_model,
                    substitute_model_name=substitute_model,
                    substitute_code_or_file=resolved_sub_file,
                    instance_name=instance_name,
                    unconnected_inputs=report.unconnected_substitute_inputs,
                )

        # Step 4: Resolve override file for parameter injection
        override_file = task_cfg.get("override_file") or sim_cfg.get("override_file")
        if override_file:
            resolved_ovr = None
            if os.path.isabs(override_file) and os.path.exists(override_file):
                resolved_ovr = override_file
            else:
                candidates = [
                    os.path.abspath(override_file),
                    os.path.join(workspace_dir, os.path.basename(override_file)),
                    os.path.join(
                        os.path.dirname(orig_package_path),
                        os.path.basename(override_file),
                    ),
                    os.path.join(os.path.dirname(orig_package_path), override_file),
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        resolved_ovr = os.path.abspath(cand)
                        break
            if resolved_ovr and os.path.exists(resolved_ovr):
                dst_ovr = os.path.join(workspace_dir, os.path.basename(resolved_ovr))
                if os.path.abspath(resolved_ovr) != os.path.abspath(dst_ovr):
                    shutil.copy2(resolved_ovr, dst_ovr)
                override_file = dst_ovr
                logger.info(
                    f"Resolved and staged parameter override file: {override_file}"
                )
            else:
                logger.warning(f"Override file not found: {override_file}")
                override_file = None

        # Step 5: Run OpenModelica simulation with -overrideFile
        omc = get_om_session()
        raw_results = []
        try:
            omc.sendExpression(f'cd("{Path(workspace_dir).as_posix()}")')

            # Ensure Modelica standard library and user-specified libraries are loaded
            omc.sendExpression("loadModel(Modelica)")
            custom_libs = context.config.get("model", {}).get(
                "loadModel"
            ) or sim_cfg.get("loadModel", [])
            for lib in custom_libs:
                omc.sendExpression(f"loadModel({lib})")

            # Load auxiliary/substitute libraries first
            for aux_mo in loaded_aux_files:
                load_res = omc.sendExpression(f'loadFile("{Path(aux_mo).as_posix()}")')
                logger.info(
                    f"Loaded auxiliary library {os.path.basename(aux_mo)}: {load_res}"
                )

            # Load main system package
            pkg_load_res = omc.sendExpression(
                f'loadFile("{Path(context.package_path).as_posix()}")'
            )
            logger.info(f"Loaded main system package: {pkg_load_res}")

            err_check = omc.sendExpression("getErrorString()")
            if err_check and "Error" in err_check:
                logger.warning(f"OMC library load messages: {err_check}")

            start_t = float(sim_cfg.get("start_time", 0.0))
            stop_t = float(context.stop_time)
            step_sz = float(context.step_size)
            tol = float(sim_cfg.get("tolerance", 1e-6))
            solv_method = sim_cfg.get("method", "dassl")
            var_filter = sim_cfg.get("variableFilter", ".*")

            sim_flags = ""
            if override_file and os.path.exists(override_file):
                sim_flags = f"-overrideFile {Path(override_file).as_posix()}"
                logger.info(f"Injected parameter override flags: {sim_flags}")

            sim_cmd = (
                f"simulate({context.model_name}, "
                f"startTime={start_t}, "
                f"stopTime={stop_t}, "
                f"stepSize={step_sz}, "
                f"tolerance={tol}, "
                f'method="{solv_method}", '
                f'outputFormat="csv", '
                f'fileNamePrefix="param_inject_results", '
                f'simflags="{sim_flags}", '
                f'variableFilter="{var_filter}")'
            )
            logger.info(f"Executing OMC simulation: {sim_cmd}")
            sim_res = omc.sendExpression(sim_cmd)
            omc_err = omc.sendExpression("getErrorString()")
            logger.info(f"OMC simulation response: {sim_res}")
            if omc_err:
                logger.info(f"OMC message log: {omc_err}")

            result_filename = os.path.abspath(
                os.path.join(workspace_dir, "param_inject_results_res.csv")
            )
            alt_result_filename = os.path.abspath(
                os.path.join(workspace_dir, "param_inject_results.csv")
            )

            actual_res_path = None
            if os.path.exists(result_filename):
                actual_res_path = result_filename
            elif os.path.exists(alt_result_filename):
                actual_res_path = alt_result_filename

            if actual_res_path:
                df = pd.read_csv(actual_res_path)
                df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                final_csv_path = os.path.abspath(
                    os.path.join(workspace_dir, "param_inject_results.csv")
                )
                df.to_csv(final_csv_path, index=False)
                raw_results.append(
                    {
                        "job_id": 1,
                        "params": {"override_file": override_file},
                        "result_path": final_csv_path,
                    }
                )
                logger.info(
                    f"ParamInjectSimulationEngine: Simulation completed successfully. Output: {final_csv_path}"
                )
            else:
                raise RuntimeError(
                    f"Simulation run produced no output file. OMC log: {omc_err}"
                )
        finally:
            omc.sendExpression("quit()")

        context.raw_results = raw_results
        context.sync_typed_results()


# Backward compatibility aliases
ParameterInjectionEngine = ParamInjectSimulationEngine
ParamInjectEngine = ParamInjectSimulationEngine
