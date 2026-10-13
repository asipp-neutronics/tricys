"""TRICYS Example Plugin: TritiumReserveAuditorPlugin

Demonstrates:
1. Agent-Native 3.0 Explicit-First Contract with Physics Semantic Bus integration.
2. In-memory typed result traversal across simulation jobs.
3. Decoupled extraction of fusion storage dynamics and metrics emission.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd

from tricys.workflow.base import PluginMetadata, PostprocessPlugin
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_postprocess_plugin

logger = logging.getLogger(__name__)


@register_postprocess_plugin("tritium_reserve_auditor")
class TritiumReserveAuditorPlugin(PostprocessPlugin):
    """聚变在役储氢系统 (SDS) 氚自持储备裕度与倍增动力学审计插件。"""

    name = "tritium_reserve_auditor"
    requires = ["results_store_path"]
    produces = ["tritium_reserve_audit_report"]

    def __init__(
        self,
        baseline_loading_g: float = 6000.0,
        safety_valley_redline_g: float = 1000.0,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.baseline_loading_g = float(baseline_loading_g)
        self.safety_valley_redline_g = float(safety_valley_redline_g)

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name=cls.name,
            category="postprocess",
            description="聚变堆储氢系统 (SDS) 储备裕度与安全谷底动力学审计插件",
            version="1.0.0",
            author="TRICYS Plugin Team",
            parameters_schema={
                "baseline_loading_g": {
                    "type": "number",
                    "default": 6000.0,
                    "description": "初始装料基准 (g)",
                },
                "safety_valley_redline_g": {
                    "type": "number",
                    "default": 1000.0,
                    "description": "最低存量安全谷底警戒红线 (g)",
                },
            },
            input_requirements=["results"],
            output_artifacts=["tritium_reserve_audit.json"],
        )

    def validate(self, context: PipelineContext) -> bool:
        """确保已有仿真结果或结果落盘文件。"""
        if context.results and len(context.results) > 0:
            return True
        if context.results_store_path and os.path.exists(context.results_store_path):
            return True
        logger.warning(f"[{self.name}] 未找到仿真结果，跳过审计。")
        return False

    def run(self, context: PipelineContext) -> None:
        """遍历各仿真工况，提取 SDS 存量并执行定量自持与安全评估。"""
        logger.info(
            f"[{self.name}] 正在执行储氢系统自持储备审计 (初始={self.baseline_loading_g}g, 红线={self.safety_valley_redline_g}g)..."
        )
        audit_records: List[Dict[str, Any]] = []

        # 优先加载已由 storage 插件落盘的 HDF5 完整时序数据
        h5_results_df = None
        if context.results_store_path and os.path.exists(context.results_store_path):
            try:
                h5_results_df = pd.read_hdf(context.results_store_path, "results")
            except Exception as e:
                logger.debug(f"无法直接读取 HDF5 results 数据集: {e}")

        if context.results and len(context.results) > 0:
            for job in context.results:
                df = job.get_dataframe()
                if (df is None or df.empty) and h5_results_df is not None:
                    if "job_id" in h5_results_df.columns:
                        df = h5_results_df[h5_results_df["job_id"] == job.job_id]
                    else:
                        df = h5_results_df

                if df is None or df.empty:
                    continue

                # 通过物理语义总线解析在役储罐存量时序 (跨模型自动匹配 sds.inventory, sds.I[1] 等)
                sds_series = context.resolve_physics_variable(df, "sds_inventory")
                if sds_series is None:
                    # 备用回退列名搜索
                    candidates = [
                        c
                        for c in df.columns
                        if "sds" in c.lower() and ("inventory" in c or "i" in c)
                    ]
                    sds_col = candidates[0] if candidates else df.columns[1]
                    sds_series = df[sds_col]

                min_storage = float(sds_series.min())
                final_storage = float(sds_series.iloc[-1])
                reserve_margin_pct = float(
                    (min_storage / self.baseline_loading_g) * 100.0
                )

                # 判定自持状态
                if min_storage <= 0.0:
                    status = "DEPLETED (枯竭断供)"
                elif min_storage < self.safety_valley_redline_g:
                    status = "WARNING (突破警戒红线)"
                else:
                    status = "SAFE (自持充裕)"

                record = {
                    "job_id": job.job_id,
                    "min_storage_g": round(min_storage, 2),
                    "final_storage_g": round(final_storage, 2),
                    "reserve_margin_pct": round(reserve_margin_pct, 2),
                    "audit_status": status,
                }
                job.metrics["reserve_margin_pct"] = record["reserve_margin_pct"]
                job.metrics["audit_status"] = status
                audit_records.append(record)

        # 回写全局上下文
        context.metadata["tritium_reserve_audit"] = audit_records
        context.metrics["tritium_reserve_audited_jobs"] = len(audit_records)

        # 导出结构化审计 JSON
        out_dir = context.results_dir or "./results"
        os.makedirs(out_dir, exist_ok=True)
        report_file = os.path.join(out_dir, "tritium_reserve_audit.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "plugin": self.name,
                    "baseline_loading_g": self.baseline_loading_g,
                    "safety_redline_g": self.safety_valley_redline_g,
                    "records": audit_records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        context.artifacts["tritium_reserve_audit_report"] = report_file
        logger.info(f"[{self.name}] 审计完成，交付件落盘至: {report_file}")
