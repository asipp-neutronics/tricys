from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from tricys.analysis.metric import (
    extract_metrics,
    extract_metrics_from_hdf5,
    get_optimization_tasks,
)
from tricys.analysis.plot import generate_analysis_plots, plot_sweep_time_series
from tricys.utils.file_utils import get_unique_filename
from tricys.workflow.base import PostprocessPlugin
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_postprocess_plugin

logger = logging.getLogger(__name__)


def execute_sensitivity_analysis(
    config: Dict[str, Any],
    run_results_dir: str,
    jobs: Optional[List[Dict[str, Any]]] = None,
) -> Optional[pd.DataFrame]:
    """Core function executing sensitivity analysis, metric consolidation, and chart generation."""
    if not config.get("sensitivity_analysis", {}).get("enabled", False):
        return None

    logger.info("Starting automated sensitivity analysis.")
    jobs = jobs or []

    try:
        analysis_config = config.get("sensitivity_analysis", {})
        analysis_case = analysis_config.get("analysis_case")
        if analysis_case is None:
            cases = analysis_config.get("analysis_cases")
            if cases and isinstance(cases, list) and len(cases) > 0:
                analysis_case = cases[0]

        metrics_definition = analysis_config.get("metrics_definition", {})

        combined_csv_path = os.path.join(run_results_dir, "sweep_results.csv")
        single_result_path = os.path.join(run_results_dir, "simulation_result.csv")
        hdf_path = os.path.join(run_results_dir, "sweep_results.h5")

        summary_df: Optional[pd.DataFrame] = None
        results_df: Optional[pd.DataFrame] = None

        if os.path.exists(hdf_path):
            logger.info(f"Loading results from HDF5: {hdf_path}")
            try:
                summary_df = extract_metrics_from_hdf5(
                    hdf_path, metrics_definition, analysis_case
                )
            except Exception as e:
                logger.error(f"Failed to extract metrics from HDF5: {e}")
                summary_df = pd.DataFrame()
        elif len(jobs) == 1 and os.path.exists(single_result_path):
            results_df = pd.read_csv(single_result_path)
            logger.info(f"Loading single task result from: {single_result_path}")
        elif len(jobs) > 1 and os.path.exists(combined_csv_path):
            results_df = pd.read_csv(combined_csv_path)
            logger.info(f"Loading sweep results from: {combined_csv_path}")
        else:
            logger.warning("No result data file found for sensitivity analysis.")
            return None

        if analysis_case is None:
            logger.warning("No valid analysis_case found for sensitivity analysis.")
            return None

        if summary_df is None and results_df is not None:
            summary_df = extract_metrics(
                results_df,
                metrics_definition,
                analysis_case,
            )

        if summary_df is None:
            summary_df = pd.DataFrame()

        optimization_tasks = get_optimization_tasks(config)

        def _find_opt_df() -> Optional[pd.DataFrame]:
            for fname in ("required_tbr_summary.csv", "requierd_tbr_summary.csv"):
                p = os.path.join(run_results_dir, fname)
                if os.path.exists(p):
                    try:
                        return pd.read_csv(p)
                    except Exception:
                        pass
            return None

        opt_df = _find_opt_df() if optimization_tasks else None

        if summary_df.empty and not optimization_tasks:
            logger.warning("Sensitivity analysis did not produce any summary data.")
            return None
        elif not summary_df.empty and not optimization_tasks:
            df_to_save = summary_df
        elif summary_df.empty and optimization_tasks:
            if opt_df is None or opt_df.empty:
                logger.warning("Optimization summary not found.")
                return None
            df_to_save = opt_df
        else:
            if opt_df is None or opt_df.empty:
                logger.warning(
                    "Optimization summary not found, saving analysis summary only."
                )
                df_to_save = summary_df
            else:
                try:
                    optimization_df = opt_df
                    sweep_params = list(jobs[0].keys()) if jobs else []
                    if sweep_params:
                        df_to_save = pd.merge(
                            summary_df,
                            optimization_df,
                            on=sweep_params,
                            how="outer",
                        )
                    else:
                        df_to_save = pd.concat([summary_df, optimization_df], axis=1)
                    logger.info(
                        "Merged optimization and sensitivity analysis summaries."
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to merge summaries: {e}. Saving analysis summary only."
                    )
                    df_to_save = summary_df

        summary_csv_path = get_unique_filename(
            run_results_dir, "sensitivity_analysis_summary.csv"
        )
        df_to_save.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Sensitivity analysis summary saved to: {summary_csv_path}")

        # Update HDF5 summary store
        if os.path.exists(hdf_path):
            try:
                with pd.HDFStore(
                    hdf_path, mode="a", complib="blosc", complevel=9
                ) as store:
                    hdf_df = df_to_save.copy()
                    if "job_id" not in hdf_df.columns and jobs:
                        try:
                            job_map = [
                                dict(job, job_id=i + 1) for i, job in enumerate(jobs)
                            ]
                            job_map_df = pd.DataFrame(job_map)
                            merge_on = [
                                c
                                for c in job_map_df.columns
                                if c != "job_id" and c in hdf_df.columns
                            ]
                            if merge_on:
                                hdf_df = pd.merge(
                                    hdf_df, job_map_df, on=merge_on, how="left"
                                )
                        except Exception as merge_e:
                            logger.warning(
                                f"Failed to restore job_id for HDF5 summary: {merge_e}"
                            )

                    for col in hdf_df.select_dtypes(
                        include=["object", "string"]
                    ).columns:
                        hdf_df[col] = hdf_df[col].astype(object)

                    store.put("summary", hdf_df, format="table", data_columns=True)
                    logger.info(
                        f"Sensitivity analysis summary saved to HDF5: {hdf_path}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to save sensitivity analysis summary to HDF5: {e}"
                )

        # Generate analysis plots
        unit_map = analysis_config.get("unit_map", {})
        glossary_path = analysis_config.get("glossary_path", "")
        generate_analysis_plots(
            df_to_save,
            analysis_case,
            run_results_dir,
            unit_map=unit_map,
            glossary_path=glossary_path,
        )

        # Generate sweep time series plots
        results_path = hdf_path if os.path.exists(hdf_path) else combined_csv_path
        if os.path.exists(results_path) and "independent_variable" in analysis_case:
            plot_sweep_time_series(
                results_path,
                run_results_dir,
                "sds.inventory",
                analysis_case["independent_variable"],
                default_params=analysis_case.get("default_simulation_values"),
                glossary_path=glossary_path,
            )

        return df_to_save
    except Exception as e:
        logger.error(f"Automated sensitivity analysis failed: {e}", exc_info=True)
        return None


@register_postprocess_plugin("sensitivity_analysis")
@register_postprocess_plugin("sensitivity")
class SensitivityAnalysisPlugin(PostprocessPlugin):
    """Functional postprocessing plugin for sensitivity analysis, metric extraction, and plotting."""

    name = "sensitivity_analysis"

    def __init__(
        self,
        enabled: bool = True,
        sensitivity_config: Optional[Any] = None,
        **params: Any,
    ) -> None:
        super().__init__(enabled=enabled, **params)
        self.enabled = enabled
        self.sensitivity_config = sensitivity_config

    def _resolve_sensitivity_config(self, context: PipelineContext) -> Dict[str, Any]:
        if self.sensitivity_config is not None:
            if hasattr(self.sensitivity_config, "model_dump"):
                return self.sensitivity_config.model_dump()
            elif isinstance(self.sensitivity_config, dict):
                return self.sensitivity_config
        return context.config.get("sensitivity_analysis", {})

    def validate(self, context: PipelineContext) -> bool:
        cfg = self._resolve_sensitivity_config(context)
        return bool(self.enabled and cfg.get("enabled", False))

    def run(self, context: PipelineContext) -> None:
        if not self.enabled:
            return

        res_dir = context.results_dir
        if not res_dir:
            return

        cfg = self._resolve_sensitivity_config(context)
        analysis_input = (
            context.config
            if "sensitivity_analysis" in context.config
            else {"sensitivity_analysis": cfg}
        )

        summary_df = execute_sensitivity_analysis(
            config=analysis_input,
            run_results_dir=res_dir,
            jobs=context.jobs,
        )
        if summary_df is not None:
            context.artifacts["sensitivity_summary"] = summary_df

    execute = run
