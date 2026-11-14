"""AI Insight Generation Module for TRICYS 2.0.

Provides pluggable LLM prompt templates and automated insight generation for
simulation KPI summaries and sensitivity ranking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from tricys.analysis.report import call_openai_analysis_api
from tricys.utils.config_utils import get_llm_env

logger = logging.getLogger(__name__)


def generate_ai_insight(
    summary_df: pd.DataFrame,
    report_content: str,
    config: Dict[str, Any],
    case_name: str = "default",
    independent_variable: str = "",
) -> Optional[str]:
    """Generates automated AI analysis insights based on simulation summary data.

    Args:
        summary_df: Summary metrics DataFrame.
        report_content: Markdown or text report content.
        config: TRICYS configuration dictionary.
        case_name: Case identifier.
        independent_variable: Primary sweep or sensitivity parameter name.

    Returns:
        Generated AI insight text if LLM credentials are configured, else None.
    """
    llm_env = get_llm_env(config)
    api_key = llm_env.get("API_KEY")
    base_url = llm_env.get("BASE_URL")
    ai_model = llm_env.get("AI_MODEL") or "gpt-4"

    if not api_key or not base_url:
        logger.info(
            "AI insight skipped: LLM environment (API_KEY / BASE_URL) not configured."
        )
        return None

    try:
        return call_openai_analysis_api(
            case_name=case_name,
            df=summary_df,
            api_key=api_key,
            base_url=base_url,
            ai_model=ai_model,
            independent_variable=independent_variable,
            report_content=report_content,
            original_config=config,
            case_data={},
        )
    except Exception as exc:
        logger.warning(f"Failed to generate AI insight: {exc}")
        return None
