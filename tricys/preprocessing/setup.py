from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from tricys.core.sampling import SamplingEngine
from tricys.workflow.base import PreprocessPlugin
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_preprocess_plugin

logger = logging.getLogger(__name__)


@register_preprocess_plugin("setup")
@register_preprocess_plugin("modelica_setup")
@register_preprocess_plugin("simulation_setup")
@register_preprocess_plugin("isolation")
@register_preprocess_plugin("job_generator")
class SimulationSetupPlugin(PreprocessPlugin):
    """Functional plugin for simulation environment setup and parameter job expansion.

    Combines:
    1. Modelica workspace and package isolation to prevent polluting source repositories.
    2. Parameter sweep matrix expansion into discrete simulation execution jobs.
    Supports explicit parameters injection for pure pipeline mode with context fallback.
    """

    name = "setup"

    def __init__(
        self,
        package_path: Optional[str] = None,
        model_name: Optional[str] = None,
        parameters: Optional[Any] = None,
        task_type: Optional[str] = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.package_path = package_path
        self.model_name = model_name
        self.parameters = parameters
        self.task_type = task_type

    def run(self, context: PipelineContext) -> None:
        # Sync explicit parameters to context if provided
        if self.package_path:
            p = self.package_path
            if not os.path.isabs(p) and context.base_dir:
                p = os.path.abspath(os.path.join(context.base_dir, p))
            context.package_path = p
        elif not context.package_path and "model" in context.config:
            raw_pkg = context.config.get("model", {}).get("fileName", "")
            if raw_pkg:
                p = raw_pkg
                if not os.path.isabs(p) and context.base_dir:
                    p = os.path.abspath(os.path.join(context.base_dir, p))
                context.package_path = p

        if self.model_name:
            context.model_name = self.model_name
        elif not context.model_name and "model" in context.config:
            context.model_name = context.config.get("model", {}).get("modelName", "")

        self._isolate_workspace(context)
        self._expand_simulation_jobs(context)

    def _isolate_workspace(self, context: PipelineContext) -> None:
        original_package_path = context.package_path
        if not original_package_path or not os.path.exists(original_package_path):
            logger.debug(
                f"Package path does not exist or empty: {original_package_path}"
            )
            return

        workspace = os.path.join(context.temp_dir, "isolated_workspace")
        os.makedirs(workspace, exist_ok=True)

        if os.path.isfile(original_package_path) and not original_package_path.endswith(
            "package.mo"
        ):
            # Single-file .mo
            isolated_package_path = os.path.join(
                workspace, os.path.basename(original_package_path)
            )
            shutil.copy(original_package_path, isolated_package_path)
            logger.info(f"Isolated single-file package to: {isolated_package_path}")
        else:
            # Multi-file directory package
            if os.path.isfile(original_package_path):
                original_package_dir = os.path.dirname(original_package_path)
            else:
                original_package_dir = original_package_path

            package_dir_name = os.path.basename(original_package_dir)
            isolated_package_dir = os.path.join(workspace, package_dir_name)

            if os.path.exists(isolated_package_dir):
                shutil.rmtree(isolated_package_dir)
            shutil.copytree(original_package_dir, isolated_package_dir)

            if os.path.isfile(original_package_path):
                isolated_package_path = os.path.join(
                    isolated_package_dir, os.path.basename(original_package_path)
                )
            else:
                isolated_package_path = os.path.join(isolated_package_dir, "package.mo")

            logger.info(f"Isolated multi-file package to: {isolated_package_path}")

        context.package_path = os.path.abspath(isolated_package_path)
        context.set_metadata("isolated_workspace", workspace)

    def _expand_simulation_jobs(self, context: PipelineContext) -> None:
        if context.jobs:
            logger.debug("Simulation jobs already generated in context, skipping")
            return

        # Explicit parameter injection (Pure Pipeline Mode priority)
        if self.parameters is not None:
            task_type = (self.task_type or "simulate").lower()
            if isinstance(self.parameters, dict):
                # Check if dictionary contains sweep lists, e.g. {"Cycle.sds.V": [1.2, 1.5]}
                has_list_values = any(
                    isinstance(v, (list, tuple)) for v in self.parameters.values()
                )
                if has_list_values:
                    from tricys.core.jobs import generate_simulation_jobs

                    jobs = generate_simulation_jobs(self.parameters)
                    context.jobs = jobs
                    context.set_metadata("sampling_method", "grid_sweep")
                    context.set_metadata("total_jobs", len(jobs))
                    logger.info(
                        f"Generated {len(jobs)} grid sweep job(s) from explicit parameters"
                    )
                    return
                else:
                    jobs = [dict(self.parameters)]
                    context.jobs = jobs
                    context.set_metadata("sampling_method", task_type)
                    context.set_metadata("total_jobs", 1)
                    logger.info(
                        f"Generated 1 single-point simulation job from explicit parameters for '{task_type}'"
                    )
                    return
            elif isinstance(self.parameters, list):
                jobs = [dict(c) for c in self.parameters]
                context.jobs = jobs
                context.set_metadata("sampling_method", "discrete_sweep")
                context.set_metadata("total_jobs", len(jobs))
                logger.info(
                    f"Generated {len(jobs)} discrete sweep job(s) from explicit parameters"
                )
                return

        # Schema 2.0 Task Handling (Domain Context Fallback)
        task_cfg = context.config.get("task", {})
        if isinstance(task_cfg, dict):
            task_type = task_cfg.get("type", "").lower()
            if task_type in ("simulate", "optimize"):
                jobs = [dict(task_cfg.get("parameters", {}))]
                context.jobs = jobs
                context.set_metadata("sampling_method", task_type)
                context.set_metadata("total_jobs", 1)
                logger.info(
                    f"Generated 1 single-point simulation job for '{task_type}' task"
                )
                return
            elif task_type == "sweep":
                params = task_cfg.get("parameters")
                grid_params = task_cfg.get("grid") or (
                    params
                    if isinstance(params, dict) and "file" not in params
                    else None
                )
                if isinstance(params, list):
                    jobs = [dict(c) for c in params]
                    context.jobs = jobs
                    context.set_metadata("sampling_method", "discrete_sweep")
                    context.set_metadata("total_jobs", len(jobs))
                    logger.info(
                        f"Generated {len(jobs)} discrete sweep job(s) from task.parameters"
                    )
                    return
                elif grid_params and isinstance(grid_params, dict):
                    from tricys.core.jobs import generate_simulation_jobs

                    jobs = generate_simulation_jobs(grid_params, auto_name=True)
                    context.jobs = jobs
                    context.set_metadata("sampling_method", "grid_sweep")
                    context.set_metadata("total_jobs", len(jobs))
                    logger.info(
                        f"Generated {len(jobs)} multi-parameter grid sweep job(s) for {len(grid_params)} variable(s)"
                    )
                    return

        engine = SamplingEngine.from_config(context.config)
        if engine is not None:
            jobs = engine.generate_jobs()
            context.set_metadata("sampling_method", engine.method)
            if engine.is_statistical() and engine.problem:
                context.set_metadata("sampling_problem", engine.problem)
        else:
            sweep_params = context.config.get("simulation_parameters", {})
            jobs = (
                generate_simulation_jobs(sweep_params, auto_name=True)
                if sweep_params
                else [{}]
            )
            context.set_metadata("sampling_method", "none")

        if not jobs:
            # If no parameter sweep configured, create a single baseline job with default/empty overrides
            jobs = [{}]

        context.jobs = jobs
        context.set_metadata("total_jobs", len(jobs))
        logger.info(
            f"Generated {len(jobs)} simulation job(s) for execution "
            f"(method: {context.get_metadata('sampling_method', 'default')})"
        )

    def _extract_sampling_jobs(self, context: PipelineContext) -> List[Dict[str, Any]]:
        """Legacy helper for extracting sampling jobs from config."""
        engine = SamplingEngine.from_config(context.config)
        if engine is not None and engine.is_statistical():
            return engine.generate_jobs()
        return []


# Aliases for backwards compatibility and nomenclature flexibility
ModelicaSetupPlugin = SimulationSetupPlugin
PackageIsolationPlugin = SimulationSetupPlugin
JobGeneratorPlugin = SimulationSetupPlugin
