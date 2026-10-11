from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from tricys.core.foc import prepare_foc_simulation_package
from tricys.workflow.base import PreprocessPlugin
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_preprocess_plugin

logger = logging.getLogger(__name__)


@register_preprocess_plugin("foc")
@register_preprocess_plugin("foc_injector")
class FOCInjectorPlugin(PreprocessPlugin):
    """Parses Fusion Operation Cycle (.foc) scenario DSL and injects dynamic pulse blocks."""

    name = "foc"

    def __init__(
        self,
        foc_config: Optional[Any] = None,
        foc_path: Optional[str] = None,
        foc_component: Optional[str] = None,
        strategy: str = "table",
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.foc_config = foc_config
        self.foc_path = foc_path
        self.foc_component = foc_component
        self.strategy = strategy

    def _resolve_foc_dict(self, context: PipelineContext) -> Dict[str, Any]:
        if self.foc_config is not None:
            if hasattr(self.foc_config, "model_dump"):
                return self.foc_config.model_dump()
            elif isinstance(self.foc_config, dict):
                return self.foc_config

        foc_cfg = context.config.get("foc") or self.params.get("foc") or {}
        if not isinstance(foc_cfg, dict):
            foc_cfg = {}

        if self.foc_path:
            foc_cfg["foc_path"] = self.foc_path
        if self.foc_component:
            foc_cfg["foc_component"] = self.foc_component
        if self.strategy:
            foc_cfg.setdefault("strategy", self.strategy)
        return foc_cfg

    def validate(self, context: PipelineContext) -> bool:
        foc_cfg = self._resolve_foc_dict(context)
        return bool(foc_cfg and foc_cfg.get("foc_path"))

    def run(self, context: PipelineContext) -> None:
        foc_cfg = self._resolve_foc_dict(context)
        foc_path = foc_cfg.get("foc_path")
        foc_component = foc_cfg.get("foc_component")
        strategy = foc_cfg.get("strategy", "table")

        if not foc_path:
            return

        if not os.path.isabs(foc_path):
            foc_path = os.path.abspath(os.path.join(context.base_dir, foc_path))

        foc_workspace = os.path.join(context.temp_dir, "foc_prepared")
        os.makedirs(foc_workspace, exist_ok=True)

        logger.info(
            f"Injecting FOC schedule from: {foc_path} into model: {context.model_name}"
        )
        model_time_base = getattr(context, "time_base", "second") or "second"
        foc_result = prepare_foc_simulation_package(
            context.package_path,
            context.model_name,
            foc_path,
            foc_workspace,
            strategy=strategy,
            foc_component=foc_component,
            model_time_base=model_time_base,
        )

        resolved_component = (
            foc_result.get("foc_component") or foc_component or "pulseSource"
        )
        context.package_path = foc_result["package_path"]
        context.set_metadata(
            "foc_schedule_duration", foc_result.get("schedule_duration")
        )
        context.set_metadata("foc_output_names", foc_result.get("output_names"))
        context.set_metadata("foc_component", resolved_component)
        context.set_metadata("foc_injected", True)

        if context.stop_time < foc_result.get("schedule_duration", 0.0):
            logger.warning(
                f"Configured stop_time ({context.stop_time}s) is shorter than FOC schedule "
                f"duration ({foc_result.get('schedule_duration')}s)"
            )

        self._check_foc_parameter_conflicts(context, resolved_component)

    def _check_foc_parameter_conflicts(
        self, context: PipelineContext, component_name: str
    ) -> List[str]:
        """Detect conflicts between FOC component replacement and scheduled parameter overrides."""
        incompatible_suffixes = ("width", "period", "nperiod", "shift", "startTime")
        candidate_params = set()

        sim_params = context.config.get("simulation_parameters", {})
        if isinstance(sim_params, dict):
            candidate_params.update(sim_params.keys())

        sens_cfg = context.config.get("sensitivity_analysis", {})
        cases = sens_cfg.get("analysis_cases") or [sens_cfg.get("analysis_case")]
        for c in cases:
            if isinstance(c, dict):
                indep = c.get("independent_variable")
                if isinstance(indep, list):
                    candidate_params.update(indep)
                elif isinstance(indep, str):
                    candidate_params.add(indep)
                problem = c.get("problem", {})
                if isinstance(problem, dict) and "names" in problem:
                    candidate_params.update(problem.get("names", []))

        for job in context.jobs:
            if isinstance(job, dict):
                candidate_params.update(job.keys())

        warnings = []
        for param in candidate_params:
            if not isinstance(param, str):
                continue
            is_matching = param.startswith(f"{component_name}.") or param.startswith(
                "pulseSource."
            )
            if not is_matching:
                continue

            prop = param.split(".", 1)[1]
            if prop in incompatible_suffixes or prop == "amplitude":
                msg = (
                    f"FOC injection replaces '{component_name}' with FOC_TablePulse / FOC_ArrayPulse. "
                    f"Parameter '{param}' cannot be applied because FOC schedule governs pulse timing. "
                    f"To scale pulse power / fusion rate dynamically, please use '{component_name}.amplitude_scale'."
                )
                logger.warning(msg)
                warnings.append(msg)

        if warnings:
            context.set_metadata("foc_parameter_warnings", warnings)
        return warnings
