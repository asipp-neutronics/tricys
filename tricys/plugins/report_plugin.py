from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tricys.analysis.baseline_analysis import baseline_analysis
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
        report_cfg = (
            self.params.get("report_config")
            or context.config.get("report")
            or context.config.get("visualizer", {}).get("report")
        )
        has_report = bool(report_cfg and report_cfg.get("enabled", True))
        return has_tasks or has_report or self.run_baseline_plot

    def run(self, context: PipelineContext) -> None:
        if not context.results_store_path or not os.path.exists(
            context.results_store_path
        ):
            logger.warning(
                "AnalysisReportPlugin: No HDF5 results file available for analysis"
            )
            return

        # 1. Execute automated engineering report generation if configured
        report_cfg = (
            self.params.get("report_config")
            or context.config.get("report")
            or context.config.get("visualizer", {}).get("report")
        )
        if report_cfg and report_cfg.get("enabled", True):
            self._generate_engineering_report(context, report_cfg)

        # 2. Execute legacy task list from post_processing or params
        tasks = self._resolve_task_list(context)
        if tasks:
            output_dir = os.path.join(context.results_dir, "post_processing")
            os.makedirs(output_dir, exist_ok=True)
            self._execute_tasks(context, tasks, output_dir)

        # 3. Execute direct baseline plot if requested
        if self.run_baseline_plot or self.params.get("run_baseline_plot", False):
            output_dir = os.path.join(context.results_dir, "post_processing")
            os.makedirs(output_dir, exist_ok=True)
            self._execute_baseline_plot(context, output_dir)

    def _generate_engineering_report(
        self, context: PipelineContext, report_cfg: Dict[str, Any]
    ) -> None:
        import json

        import pandas as pd

        from tricys.visualizer.report.exporter import ReportExporter
        from tricys.visualizer.report.figures import generate_all_report_figures
        from tricys.visualizer.report.templates import (
            normalize_template_id,
            render_report,
        )

        results_h5 = context.results_store_path
        if not results_h5 or not os.path.exists(results_h5):
            logger.warning(
                "AnalysisReportPlugin: HDF5 results file not found for report generation"
            )
            return

        out_rel = report_cfg.get("output_dir", "reports")
        report_dir = os.path.abspath(os.path.join(context.results_dir, out_rel))
        os.makedirs(report_dir, exist_ok=True)

        jobs_df = None
        metrics_df = None
        results_df = None
        try:
            with pd.HDFStore(results_h5, mode="r") as store:
                if "/jobs" in store.keys():
                    jobs_df = store.select("jobs")
                if "/summary" in store.keys():
                    metrics_df = store.select("summary")
                if "/results" in store.keys():
                    results_df = store.select("results")
        except Exception as exc:
            logger.error(f"AnalysisReportPlugin: Failed to read data from HDF5: {exc}")

        jobs_list = jobs_df.to_dict(orient="records") if jobs_df is not None else []

        task_type = context.config.get("task", {}).get("type", "sweep")
        figures = generate_all_report_figures(
            results_df=results_df,
            jobs_df=jobs_df,
            metrics_df=metrics_df,
            task_type=task_type,
            output_dir=report_dir,
            task_config=context.config,
        )

        # 默认不填跟 task.type 一致，并规范化至三大核心模板之一
        template_id = normalize_template_id(report_cfg.get("template"), task_type)
        title = report_cfg.get(
            "title", f"TRICYS 聚变氚循环工程仿真分析报告 ({template_id.upper()})"
        )
        formats = report_cfg.get("formats", ["markdown"])

        report_md = render_report(
            template_id=template_id,
            config=context.config,
            jobs=jobs_list,
            metrics_df=metrics_df,
            figures=figures,
            results_df=results_df,
        )

        exporter = ReportExporter()
        slug = f"engineering_report_{template_id}"
        if "markdown" in formats or "md" in formats:
            md_path = os.path.join(report_dir, f"{slug}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(report_md)
            logger.info(f"AnalysisReportPlugin: Exported Markdown report to {md_path}")

        if "docx" in formats or "word" in formats:
            try:
                docx_bytes = exporter.export_docx(report_md, title=title)
                docx_path = os.path.join(report_dir, f"{slug}.docx")
                with open(docx_path, "wb") as f:
                    f.write(docx_bytes)
                logger.info(
                    f"AnalysisReportPlugin: Exported Word report to {docx_path}"
                )
            except Exception as e:
                logger.warning(f"AnalysisReportPlugin: DOCX export skipped/failed: {e}")

        if "pdf" in formats:
            try:
                pdf_bytes = exporter.export_pdf(report_md, title=title)
                pdf_path = os.path.join(report_dir, f"{slug}.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)
                logger.info(f"AnalysisReportPlugin: Exported PDF report to {pdf_path}")
            except Exception as e:
                logger.warning(f"AnalysisReportPlugin: PDF export skipped/failed: {e}")

        try:
            with pd.HDFStore(results_h5, mode="a") as store:
                report_payload = {
                    "markdown": report_md,
                    "template": template_id,
                    "title": title,
                }
                report_df = pd.DataFrame(
                    {"report_json": [json.dumps(report_payload)]}
                ).astype(object)
                store.put("report", report_df, format="fixed")
                logger.info("AnalysisReportPlugin: Stored report in HDF5 /report")
        except Exception as e:
            logger.warning(
                f"AnalysisReportPlugin: Failed to persist report in HDF5: {e}"
            )

        context.set_metadata("report_markdown", report_md)

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
            mod_str = task_config["module"]
            if mod_str == "tricys.postprocess.static_alarm":
                mod_str = "tricys.plugins.static_alarm"
            elif mod_str == "tricys.postprocess.baseline_analysis":
                mod_str = "tricys.analysis.baseline_analysis"
            return importlib.import_module(mod_str)
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
