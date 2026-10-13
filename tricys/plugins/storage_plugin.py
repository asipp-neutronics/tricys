from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Dict, Optional

import pandas as pd

from tricys.analysis.metric import (
    build_single_job_summary_df,
    calculate_single_job_metrics,
)
from tricys.utils.file_utils import get_unique_filename
from tricys.workflow.base import PostprocessPlugin
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_postprocess_plugin

logger = logging.getLogger(__name__)


def process_and_store_job_result(
    store: pd.HDFStore,
    job_id: int,
    params: dict,
    res_path: str,
    config: dict,
    metrics_definition: Optional[dict] = None,
    filter_schema: Optional[list] = None,
    jobs_min_itemsize: Optional[dict] = None,
) -> None:
    """Processes a single job's CSV result, appends to HDF5, and cleans up transient workspace."""
    if not res_path or not os.path.exists(res_path):
        return

    try:
        df = pd.read_csv(res_path)
        df["job_id"] = job_id
        store.append("results", df, index=False, data_columns=True)

        # Summary calculations
        if metrics_definition:
            try:
                metrics_res = (
                    calculate_single_job_metrics(
                        df,
                        metrics_definition,
                        filter_schema=filter_schema,
                    )
                    or {}
                )
                if params and isinstance(params, dict):
                    for m_name in metrics_definition.keys():
                        if m_name in params and m_name not in metrics_res:
                            metrics_res[m_name] = params[m_name]

                if metrics_res:
                    summary_df = build_single_job_summary_df(
                        job_id, metrics_res, metrics_definition
                    )
                    for col in summary_df.columns:
                        if col != "job_id" and pd.api.types.is_numeric_dtype(
                            summary_df[col]
                        ):
                            summary_df[col] = summary_df[col].astype(float)
                        elif col != "job_id":
                            summary_df[col] = summary_df[col].astype(object)

                    sum_str_cols = {
                        col: 256
                        for col in summary_df.select_dtypes(
                            include=["object", "string"]
                        ).columns
                    }
                    if "/summary" not in store.keys() and sum_str_cols:
                        store.append(
                            "summary",
                            summary_df,
                            index=False,
                            data_columns=True,
                            min_itemsize=sum_str_cols,
                        )
                    else:
                        store.append(
                            "summary",
                            summary_df,
                            index=False,
                            data_columns=True,
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to calculate/save summary metrics for job {job_id}: {e}"
                )

        clean_params = {
            k: v
            for k, v in params.items()
            if not metrics_definition or k not in metrics_definition
        }
        param_df = pd.DataFrame([clean_params])
        param_df["job_id"] = job_id
        for col in param_df.columns:
            if col != "job_id" and pd.api.types.is_numeric_dtype(param_df[col]):
                param_df[col] = param_df[col].astype(float)
        for col in param_df.select_dtypes(include=["object", "string"]).columns:
            param_df[col] = param_df[col].astype(object)

        param_str_cols = {
            col: 256
            for col in param_df.select_dtypes(include=["object", "string"]).columns
        }
        if jobs_min_itemsize:
            param_str_cols.update(
                {k: v for k, v in jobs_min_itemsize.items() if k in param_df.columns}
            )

        if "/jobs" not in store.keys() and param_str_cols:
            store.append(
                "jobs",
                param_df,
                index=False,
                data_columns=True,
                min_itemsize=param_str_cols,
            )
        else:
            store.append("jobs", param_df, index=False, data_columns=True)

        # Cleanup intermediate job files if keep_temp_files is False
        keep_temp = config.get("simulation", {}).get("keep_temp_files", False)
        job_dir = os.path.dirname(res_path)
        if (
            not keep_temp
            and os.path.exists(job_dir)
            and "job_" in os.path.basename(job_dir)
        ):
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass
    except Exception as exc:
        logger.error(f"Failed to process HDF5 result for job {job_id}: {exc}")


def export_hdf5_to_csv(results_dir: str, hdf_path: str) -> None:
    """Exports structured simulation traces and summary metrics from HDF5 to CSV format."""
    if not hdf_path or not os.path.exists(hdf_path):
        return

    logger.info(f"Exporting results from {hdf_path} to CSV...")
    try:
        with pd.HDFStore(hdf_path, mode="r") as store:
            if "/results" not in store.keys():
                logger.warning("No results found in HDF5 to export.")
                return

            df_results = store.select("results")
            df_jobs = (
                store.select("jobs") if "/jobs" in store.keys() else pd.DataFrame()
            )

            if "/summary" in store.keys():
                df_summary = store.select("summary")
                if not df_jobs.empty:
                    df_summary = pd.merge(df_jobs, df_summary, on="job_id", how="left")
                summary_csv_path = get_unique_filename(
                    results_dir, "summary_metrics.csv"
                )
                df_summary.to_csv(summary_csv_path, index=False)
                logger.info(f"Exported summary metrics to {summary_csv_path}")

        job_params_map = {}
        if not df_jobs.empty:
            for _, row in df_jobs.iterrows():
                job_id = row["job_id"]
                params = row.drop("job_id").to_dict()
                params = {k: v for k, v in params.items() if pd.notna(v)}
                param_str = "&".join([f"{k}={v}" for k, v in params.items()])
                job_params_map[job_id] = param_str

        all_times = df_results[["time"]].drop_duplicates().sort_values("time")
        pivoted_dfs = [all_times]

        metric_cols = [c for c in df_results.columns if c not in ["time", "job_id"]]

        for job_id, group in df_results.groupby("job_id"):
            param_str = job_params_map.get(job_id, f"job={job_id}")
            group_pivoted = group.drop(columns=["job_id"]).copy()

            rename_dict = {}
            for col in metric_cols:
                if param_str:
                    rename_dict[col] = f"{col}&{param_str}"
                else:
                    rename_dict[col] = col

            group_pivoted.rename(columns=rename_dict, inplace=True)
            pivoted_dfs.append(group_pivoted)

        final_df = pivoted_dfs[0]
        for df_part in pivoted_dfs[1:]:
            final_df = pd.merge_asof(
                final_df,
                df_part.sort_values("time"),
                on="time",
                direction="nearest",
                tolerance=1e-5,
            )

        sim_csv_path = get_unique_filename(results_dir, "simulation_result.csv")
        final_df.to_csv(sim_csv_path, index=False)
        logger.info(f"Successfully exported simulation results to {sim_csv_path}")
    except Exception as exc:
        logger.error(f"Failed to export CSV results: {exc}")


@register_postprocess_plugin("storage")
@register_postprocess_plugin("data_storage")
@register_postprocess_plugin("simulation_storage")
@register_postprocess_plugin("hdf5_storage")
class SimulationStoragePlugin(PostprocessPlugin):
    """Functional plugin for simulation data persistence, metric evaluation, and export.

    Combines:
    1. Metrics definition registration and calculation.
    2. Consolidated HDF5 storage with high-performance compression.
    3. Optional CSV export (simulation trajectories and summary metrics).
    """

    name = "storage"

    def __init__(
        self,
        export_csv: Optional[bool] = None,
        metrics_definition: Optional[Dict[str, Any]] = None,
        results_dir: Optional[str] = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.export_csv = export_csv
        self.metrics_definition = metrics_definition
        self.results_dir = results_dir

    def run(self, context: PipelineContext) -> None:
        if self.results_dir:
            if not os.path.isabs(self.results_dir) and context.base_dir:
                context.results_dir = os.path.abspath(
                    os.path.join(context.base_dir, self.results_dir)
                )
            else:
                context.results_dir = os.path.abspath(self.results_dir)

        # 1. Register metrics definition metadata if available
        metrics_def = (
            self.metrics_definition
            or self.params.get("metrics_definition")
            or getattr(context, "metrics_definition", None)
            or context.config.get("metrics_definition")
            or (
                context.config.get("task", {}).get("metrics")
                if isinstance(context.config.get("task"), dict)
                else None
            )
            or context.config.get("sensitivity_analysis", {}).get(
                "metrics_definition", {}
            )
        )
        if metrics_def:
            context.metrics_definition = metrics_def
            context.set_metadata("metrics_definition", metrics_def)

        # 2. Persist simulation results into HDF5 store
        if not context.results_store_path or not os.path.exists(
            context.results_store_path
        ):
            results_dir = os.path.abspath(context.results_dir)
            os.makedirs(results_dir, exist_ok=True)
            hdf_path = get_unique_filename(results_dir, "sweep_results.h5")
            filter_schema = context.config.get("filter_schema")

            logger.info(f"SimulationStoragePlugin: Storing results in {hdf_path}")
            with pd.HDFStore(hdf_path, mode="w", complib="blosc", complevel=9) as store:
                # Save configuration
                try:
                    config_df = pd.DataFrame(
                        {"config_json": [json.dumps(context.config)]}
                    ).astype(object)
                    store.put("config", config_df, format="fixed")
                except Exception as e:
                    logger.warning(f"Failed to save config to HDF5: {e}")

                # Precompute string column sizes across all jobs to prevent PyTables overflow
                jobs_min_itemsize = {}
                for item in context.raw_results:
                    for k, v in item.get("params", {}).items():
                        if isinstance(v, str):
                            jobs_min_itemsize[k] = max(
                                jobs_min_itemsize.get(k, 256),
                                len(v) * 2,
                                256,
                            )

                # Save raw results and calculate job metrics
                for item in context.raw_results:
                    job_id = item["job_id"]
                    params = item.get("params", {})
                    res_path = item.get("result_path", "")
                    process_and_store_job_result(
                        store=store,
                        job_id=job_id,
                        params=params,
                        res_path=res_path,
                        config=context.config,
                        metrics_definition=metrics_def,
                        filter_schema=filter_schema,
                        jobs_min_itemsize=jobs_min_itemsize,
                    )

            context.results_store_path = hdf_path
            logger.info(
                f"SimulationStoragePlugin: Results saved to {context.results_store_path}"
            )

        # 3. Export to CSV if requested via flag, params, or config
        should_export_csv = (
            self.export_csv is True
            or self.params.get("export_csv", False)
            or context.config.get("export_csv", False)
        )
        if should_export_csv:
            export_hdf5_to_csv(context.results_dir, context.results_store_path)


# Backward-compatibility shims
@register_postprocess_plugin("csv_export")
class CSVExportPlugin(PostprocessPlugin):
    name = "csv_export"

    def run(self, context: PipelineContext) -> None:
        if not context.results_store_path:
            logger.warning(
                "CSVExportPlugin: No HDF5 results store path found in context"
            )
            return
        export_hdf5_to_csv(context.results_dir, context.results_store_path)


@register_postprocess_plugin("metric")
class MetricCalculatorPlugin(PostprocessPlugin):
    name = "metric"

    def run(self, context: PipelineContext) -> None:
        metrics_def = context.config.get("metrics_definition") or context.config.get(
            "sensitivity_analysis", {}
        ).get("metrics_definition", {})
        if metrics_def:
            context.set_metadata("metrics_definition", metrics_def)


HDF5StoragePlugin = SimulationStoragePlugin
