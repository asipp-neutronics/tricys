from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tricys.postprocess.baseline_analysis import baseline_analysis
from tricys.workflow.base import PostprocessPlugin
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_postprocess_plugin

logger = logging.getLogger(__name__)


@register_postprocess_plugin("report")
@register_postprocess_plugin("analysis_report")
@register_postprocess_plugin("tasks")
@register_postprocess_plugin("legacy_tasks")
@register_postprocess_plugin("baseline_plot")
class AnalysisReportPlugin(PostprocessPlugin):
    """Functional plugin for system condition analysis, reporting, and custom post-processing tasks.

    Combines:
    1. Execution of configured analysis tasks (e.g., baseline inventory analysis, rise-time analysis).
    2. Generation of system operational charts and LLM summary reports.
    """

    name = "report"

    def __init__(self, run_baseline_plot: bool = False, **params: Any) -> None:
        super().__init__(**params)
        self.run_baseline_plot = run_baseline_plot

    def validate(self, context: PipelineContext) -> bool:
        has_tasks = bool(
            context.config.get("post_processing") or self.params.get("tasks")
        )
        return has_tasks or self.run_baseline_plot

    def run(self, context: PipelineContext) -> None:
        if not context.results_store_path or not os.path.exists(
            context.results_store_path
        ):
            logger.warning(
                "AnalysisReportPlugin: No HDF5 results file available for analysis"
            )
            return

        output_dir = os.path.join(context.results_dir, "post_processing")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Execute task list from post_processing or params
        tasks = self._resolve_task_list(context)
        if tasks:
            self._execute_tasks(context, tasks, output_dir)

        # 2. Execute direct baseline plot if requested
        if self.run_baseline_plot or self.params.get("run_baseline_plot", False):
            self._execute_baseline_plot(context, output_dir)

    def _resolve_task_list(self, context: PipelineContext) -> List[Dict[str, Any]]:
        post_cfg = self.params.get("tasks") or context.config.get("post_processing")
        if not post_cfg:
            return []

        if isinstance(post_cfg, dict):
            return list(post_cfg.get("tasks", []))
        elif isinstance(post_cfg, list):
            return list(post_cfg)
        return []

    def _execute_tasks(
        self,
        context: PipelineContext,
        task_list: List[Dict[str, Any]],
        output_dir: str,
    ) -> None:
        results_file_path = context.results_store_path

        for i, task_config in enumerate(task_list, 1):
            try:
                function_name = task_config.get("function")
                params = task_config.get("params", {}).copy()

                if "llm_env" not in params and isinstance(
                    context.config.get("llm_env"), dict
                ):
                    params["llm_env"] = context.config["llm_env"]

                module = self._resolve_task_module(task_config, context.base_dir)
                if module and function_name and hasattr(module, function_name):
                    func = getattr(module, function_name)
                    logger.info(
                        f"Executing task {i}: {function_name} in {module.__name__}"
                    )
                    func(
                        results_file_path=results_file_path,
                        output_dir=output_dir,
                        **params,
                    )
                else:
                    logger.error(
                        f"Failed to resolve function '{function_name}' for task {i}"
                    )
            except Exception as exc:
                logger.error(f"Post-processing task {i} failed: {exc}", exc_info=True)

    def _resolve_task_module(
        self, task_config: Dict[str, Any], base_dir: str
    ) -> Optional[Any]:
        if "script_path" in task_config:
            script_path_str = task_config["script_path"]
            if not os.path.isabs(script_path_str):
                script_path_str = os.path.abspath(
                    os.path.join(base_dir, script_path_str)
                )
            script_path = Path(script_path_str).resolve()

            if not script_path.is_file():
                logger.error(f"Script not found at {script_path}")
                return None

            spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        elif "module" in task_config:
            return importlib.import_module(task_config["module"])
        return None

    def _execute_baseline_plot(self, context: PipelineContext, output_dir: str) -> None:
        params = self.params.copy()
        logger.info(
            f"AnalysisReportPlugin: Generating baseline inventory plots in {output_dir}"
        )
        try:
            baseline_analysis(
                results_file_path=context.results_store_path,
                output_dir=output_dir,
                **params,
            )
        except Exception as exc:
            logger.error(f"Baseline plot generation failed: {exc}", exc_info=True)


# Backwards compatibility shims
LegacyTaskPlugin = AnalysisReportPlugin
BaselinePlotPlugin = AnalysisReportPlugin
