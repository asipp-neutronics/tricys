from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, List

from tricys.core.jobs import generate_simulation_jobs
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
    """

    name = "setup"

    def run(self, context: PipelineContext) -> None:
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
        # Schema 2.0 Task Handling
        task_cfg = context.config.get("task", {})
        if isinstance(task_cfg, dict):
            task_type = task_cfg.get("type", "").lower()
            if task_type == "simulate":
                jobs = [dict(task_cfg.get("parameters", {}))]
                context.jobs = jobs
                context.set_metadata("sampling_method", "simulate")
                context.set_metadata("total_jobs", 1)
                logger.info(
                    "Generated 1 single-point simulation job for 'simulate' task"
                )
                return
            elif task_type == "sweep" and isinstance(task_cfg.get("parameters"), list):
                jobs = [dict(c) for c in task_cfg["parameters"]]
                context.jobs = jobs
                context.set_metadata("sampling_method", "discrete_sweep")
                context.set_metadata("total_jobs", len(jobs))
                logger.info(
                    f"Generated {len(jobs)} discrete sweep job(s) from task.parameters"
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
            jobs = generate_simulation_jobs(sweep_params) if sweep_params else [{}]
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
