"""Visualizer Report Module for TRICYS 2.0.

Provides automated engineering report generation and AI-assisted insights,
natively integrated into the visualizer pillar.
"""

from __future__ import annotations

from tricys.analysis.report import (
    call_openai_analysis_api,
    generate_analysis_cases_summary,
    generate_sensitivity_academic_report,
)
from tricys.visualizer.report.ai_insight import generate_ai_insight
from tricys.visualizer.report.exporter import ReportExporter
from tricys.visualizer.report.templates import (
    TEMPLATES_CATALOG,
    get_recommended_template_for_task,
    normalize_template_id,
    render_report,
)

__all__ = [
    "generate_sensitivity_academic_report",
    "generate_analysis_cases_summary",
    "call_openai_analysis_api",
    "generate_ai_insight",
    "TEMPLATES_CATALOG",
    "get_recommended_template_for_task",
    "normalize_template_id",
    "render_report",
    "ReportExporter",
]
