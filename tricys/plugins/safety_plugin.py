from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from tricys.plugins.auditor.offline import parse_auditor_config, perform_offline_audit
from tricys.plugins.static_alarm import check_thresholds
from tricys.workflow.base import PostprocessPlugin
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_postprocess_plugin

logger = logging.getLogger(__name__)


@register_postprocess_plugin("safety")
@register_postprocess_plugin("safety_auditor")
@register_postprocess_plugin("audit")
@register_postprocess_plugin("auditor")
@register_postprocess_plugin("mass_conservation")
@register_postprocess_plugin("offline_auditor")
@register_postprocess_plugin("static_alarm")
class SafetyAuditorPlugin(PostprocessPlugin):
    """Functional plugin for plant-wide mass conservation auditing and safety threshold monitoring.

    Combines:
    1. Offline plant-wide mass balance conservation audit across tritium subsystems.
    2. Static safety threshold monitoring (inventory limits, release rates, concentration bounds).
    """

    name = "safety"

    def __init__(
        self,
        auditor_config: Optional[Any] = None,
        alarm_rules: Optional[List[Dict[str, Any]]] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.auditor_config = auditor_config
        self.alarm_rules = alarm_rules or rules

    def _resolve_auditor_config(self, context: PipelineContext) -> Dict[str, Any]:
        if self.auditor_config is not None:
            if hasattr(self.auditor_config, "model_dump"):
                return self.auditor_config.model_dump()
            elif isinstance(self.auditor_config, dict):
                return self.auditor_config
        return (
            self.params.get("auditor_config")
            or (
                context.config.get("task", {}).get("auditor")
                if isinstance(context.config.get("task"), dict)
                else None
            )
            or context.config.get("auditor", {})
        )

    def _resolve_rules(self, context: PipelineContext) -> List[Dict[str, Any]]:
        if self.alarm_rules is not None:
            return self.alarm_rules
        return (
            self.params.get("rules")
            or self.params.get("alarm_rules")
            or context.config.get("static_alarm", {}).get("rules", [])
        )

    def validate(self, context: PipelineContext) -> bool:
        auditor_cfg = self._resolve_auditor_config(context)
        rules = self._resolve_rules(context)
        auditor_enabled = bool(auditor_cfg.get("enabled", False))
        return auditor_enabled or bool(rules)

    def run(self, context: PipelineContext) -> None:
        if not context.results_store_path or not os.path.exists(
            context.results_store_path
        ):
            logger.warning(
                "SafetyAuditorPlugin: No result store available for safety auditing"
            )
            return

        self._run_mass_balance_audit(context)
        self._run_static_alarm_checks(context)

    def _run_mass_balance_audit(self, context: PipelineContext) -> None:
        auditor_cfg_raw = self._resolve_auditor_config(context)
        if not auditor_cfg_raw.get("enabled", False) and not self.params.get(
            "auditor_enabled", False
        ):
            return

        logger.info("SafetyAuditorPlugin: Running mass balance conservation audit...")
        try:
            auditor_cfg = parse_auditor_config(
                {"auditor": auditor_cfg_raw}
                if "auditor" not in auditor_cfg_raw
                else auditor_cfg_raw
            )
            audit_result = perform_offline_audit(
                context.results_store_path, auditor_cfg
            )

            report_path = os.path.join(context.results_dir, "auditor_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(audit_result, f, indent=4)

            context.set_metadata("auditor_report", audit_result)

            if audit_result.get("status") == "success":
                err_val = audit_result.get("mass_balance_error", 0.0)
                logger.info(
                    f"Mass balance audit completed. Error: {err_val:.6e} g (Report: {report_path})"
                )
            else:
                logger.warning(f"Auditor returned error: {audit_result.get('error')}")
        except Exception as exc:
            logger.error(f"Failed to run offline auditor: {exc}", exc_info=True)

    def _run_static_alarm_checks(self, context: PipelineContext) -> None:
        rules: List[Dict[str, Any]] = (
            self.params.get("rules")
            or self.params.get("alarm_rules")
            or context.config.get("static_alarm", {}).get("rules", [])
        )

        if not rules:
            return

        output_dir = os.path.join(context.results_dir, "post_processing")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"SafetyAuditorPlugin: Checking {len(rules)} threshold rule(s)")
        try:
            check_thresholds(
                results_file_path=context.results_store_path,
                output_dir=output_dir,
                rules=rules,
                **self.params,
            )
        except Exception as exc:
            logger.error(
                f"SafetyAuditorPlugin alarm checks failed: {exc}", exc_info=True
            )


# Backwards compatibility shims
OfflineAuditorPlugin = SafetyAuditorPlugin
StaticAlarmPlugin = SafetyAuditorPlugin
