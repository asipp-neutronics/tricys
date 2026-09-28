from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

from tricys.utils.log_utils import setup_logging

logger = logging.getLogger(__name__)


def _mp_execute_case_worker(
    args: Tuple[Dict[str, Any], Callable[[Dict[str, Any]], None]],
) -> Tuple[Dict[str, Any], Optional[bool], Optional[str]]:
    case_info, runner_func = args
    case_index = case_info["index"]
    case_workspace = case_info["workspace"]
    case_config = case_info["config"]
    case_data = case_info["case_data"]

    original_cwd = os.getcwd()
    try:
        os.chdir(case_workspace)
        setup_logging(case_config)

        logger.info(
            "Executing analysis case",
            extra={
                "case_name": case_data.get("name", case_index),
                "case_index": case_index,
                "workspace": case_workspace,
                "pid": os.getpid(),
            },
        )

        runner_func(case_config)

        logger.info(
            "Case executed successfully",
            extra={
                "case_name": case_data.get("name", case_index),
                "case_index": case_index,
            },
        )
        return case_info, True, None
    except Exception as e:
        logger.error(
            "Case execution failed",
            exc_info=True,
            extra={
                "case_name": case_data.get("name", case_index),
                "case_index": case_index,
            },
        )
        return case_info, False, str(e)
    finally:
        os.chdir(original_cwd)


class AnalysisCasesManager:
    """Manages multi-case analysis workflows, including workspace isolation,

    concurrent/sequential dispatching, and summary report generation.
    """

    @staticmethod
    def is_analysis_cases(config: Dict[str, Any]) -> bool:
        sa_config = config.get("sensitivity_analysis", {})
        cases = sa_config.get("analysis_cases")
        return bool(
            cases
            and (
                (isinstance(cases, list) and len(cases) > 0) or isinstance(cases, dict)
            )
        )

    @staticmethod
    def is_salib_analysis(config: Dict[str, Any]) -> bool:
        sa_config = config.get("sensitivity_analysis", {})
        analysis_case = sa_config.get("analysis_case")
        return bool(
            isinstance(analysis_case, dict)
            and isinstance(analysis_case.get("independent_variable"), list)
            and isinstance(analysis_case.get("independent_variable_sampling"), dict)
            and "analyzer" in analysis_case
        )

    @classmethod
    def handle_analysis_cases(
        cls,
        config: Dict[str, Any],
        runner_func: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        """[Deprecated] Determines if the configuration is a multi-case or SALib analysis,

        and delegates to the unified Pipeline architecture.
        """
        has_cases = cls.is_analysis_cases(config)
        has_salib = cls.is_salib_analysis(config)

        if not (has_cases or has_salib):
            return False

        import warnings

        warnings.warn(
            "AnalysisCasesManager is deprecated and will be removed in a future release. "
            "Multi-case and sensitivity workflows are now orchestrated natively by PipelineBuilder "
            "and simulation.run_simulation.",
            DeprecationWarning,
            stacklevel=2,
        )

        from tricys.simulation.simulation import run_simulation as pipeline_run_sim

        pipeline_run_sim(config)
        return True
