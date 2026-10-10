from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from tricys.analysis.metric import (
    calculate_doubling_time,
    calculate_startup_inventory,
    get_optimization_tasks,
    time_of_turning_point,
)
from tricys.simulation.engines.standard import run_fast_subprocess_job

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Holds outcome and convergence trace for an optimization task."""

    metric_name: str
    target_parameter: str
    optimal_parameter_value: float
    optimal_metric_value: float
    converged: bool = False
    iterations: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)


class BisectionOptimizer:
    """Performs bisection search optimization using pre-compiled fast subprocesses."""

    @staticmethod
    def run_bisection_search_fast(
        config: Dict[str, Any],
        job_id_prefix: str,
        optimization_metric_name: str,
        fast_context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Performs bisection search using pre-compiled executable (Fast Mode)."""
        # 1. Resolve optimization_config from multiple possible locations (V2 task.metrics, metrics_definition, sensitivity_analysis)
        task_metrics = (
            config.get("task", {}).get("metrics", {})
            if isinstance(config.get("task"), dict)
            else {}
        )
        optimization_config = (
            task_metrics.get(optimization_metric_name)
            or config.get("metrics_definition", {}).get(optimization_metric_name)
            or config.get("sensitivity_analysis", {})
            .get("metrics_definition", {})
            .get(optimization_metric_name)
            or {}
        )
        if hasattr(optimization_config, "model_dump"):
            optimization_config = optimization_config.model_dump()
        elif hasattr(optimization_config, "dict"):
            optimization_config = optimization_config.dict()

        param_to_optimize = optimization_config.get(
            "parameter"
        ) or optimization_config.get("parameter_to_optimize")
        bounds = optimization_config.get("search_bounds") or optimization_config.get(
            "search_range"
        )

        if not optimization_config or not param_to_optimize or not bounds:
            raise ValueError(
                f"Optimization config for '{optimization_metric_name}' invalid: must specify parameter and search_bounds/search_range."
            )

        low_orig, high_orig = float(bounds[0]), float(bounds[1])
        tolerance = float(optimization_config.get("tolerance", 0.001))
        max_iterations = int(optimization_config.get("max_iterations", 15))
        stop_time = float(config.get("simulation", {}).get("stop_time", 1000.0))

        # 2. Target metric extraction: Support both V2 target_metric dict and legacy flat fields
        target_metric = optimization_config.get("target_metric")
        if isinstance(target_metric, dict):
            source_column = (
                target_metric.get("source")
                or optimization_config.get("source_column")
                or "sds.I[1]"
            )
            t_eval = (
                target_metric.get("evaluate") or target_metric.get("operator") or ""
            ).lower()
            t_thresh = (
                target_metric.get("threshold")
                if target_metric.get("threshold") is not None
                else target_metric.get("target_value")
            )
            if t_thresh is None:
                t_thresh = target_metric.get("max_value")

            if "doubling" in t_eval or "倍增" in t_eval:
                metric_name = "Doubling_Time"
            elif (
                "turn" in t_eval
                or "self" in t_eval
                or "拐点" in t_eval
                or "自持" in t_eval
            ):
                metric_name = "Self_Sufficiency_Time"
            elif "min" in t_eval or "startup" in t_eval or "亏空" in t_eval:
                metric_name = "Startup_Inventory"
            else:
                metric_name = optimization_config.get("metric_name", "Doubling_Time")

            metric_max_value = (
                float(t_thresh)
                if t_thresh is not None
                else float(optimization_config.get("metric_max_value", stop_time))
            )
        else:
            source_column = optimization_config.get("source_column") or "sds.I[1]"
            metric_name = optimization_config.get(
                "metric_name", "Self_Sufficiency_Time"
            )
            metric_max_value = float(
                optimization_config.get("metric_max_value", stop_time)
            )

        metric_max_values = (
            metric_max_value
            if isinstance(metric_max_value, list)
            else [metric_max_value]
        )
        is_list_input = isinstance(metric_max_value, list)

        all_optimal_params = {}
        all_optimal_values = {}

        for current_metric_max_value in metric_max_values:
            low, high = low_orig, high_orig
            best_successful_param = float("inf")
            best_successful_value = float("inf")

            for i in range(max_iterations):
                if high - low < tolerance:
                    break

                mid_param = (low + high) / 2

                # Prepare Job Params
                job_params = config.get("simulation_parameters", {}).copy()
                job_params[param_to_optimize] = mid_param

                iter_id = f"{job_id_prefix}_iter{i}_{int(mid_param * 1000)}"

                try:
                    result_path = run_fast_subprocess_job(
                        job_params,
                        iter_id,
                        fast_context["exe"],
                        fast_context["xml"],
                        fast_context["om_bin"],
                        fast_context["temp_dir"],
                        config["simulation"],
                        inplace_execution=False,
                    )

                    metric_value = float("inf")
                    if result_path and os.path.exists(result_path):
                        try:
                            results_df = pd.read_csv(result_path)
                            target_col = (
                                source_column
                                if source_column in results_df.columns
                                else None
                            )
                            if not target_col:
                                from tricys.analysis.tritium_cycle import (
                                    detect_sds_column,
                                )

                                auto_col = detect_sds_column(results_df)
                                if auto_col and auto_col in results_df.columns:
                                    target_col = auto_col

                            if target_col and target_col in results_df.columns:
                                if metric_name == "Self_Sufficiency_Time":
                                    metric_value = time_of_turning_point(
                                        results_df[target_col], results_df["time"]
                                    )
                                elif metric_name == "Doubling_Time":
                                    metric_value = calculate_doubling_time(
                                        results_df[target_col], results_df["time"]
                                    )
                                elif metric_name == "Startup_Inventory":
                                    metric_value = calculate_startup_inventory(
                                        results_df[target_col]
                                    )

                            job_dir = os.path.dirname(result_path)
                            shutil.rmtree(job_dir, ignore_errors=True)

                        except Exception:
                            pass

                    upper_limit = (
                        current_metric_max_value
                        if current_metric_max_value is not None
                        else stop_time
                    )
                    if (
                        metric_value is not None
                        and not pd.isna(metric_value)
                        and metric_value <= upper_limit
                    ):
                        best_successful_param = mid_param
                        best_successful_value = metric_value
                        high = mid_param
                    else:
                        low = mid_param

                except Exception as e:
                    logger.warning(f"Fast bisection iteration failed: {e}")

            value_key_base = f"{metric_name}_for_{optimization_metric_name}"
            if is_list_input:
                unit_str = f"{current_metric_max_value} units"
                param_key = f"{optimization_metric_name}({unit_str})"
                value_key = f"{value_key_base}({unit_str})"
            else:
                param_key = optimization_metric_name
                value_key = value_key_base

            all_optimal_params[param_key] = best_successful_param
            all_optimal_values[value_key] = best_successful_value

        return all_optimal_params, all_optimal_values

    @classmethod
    def run_optimization_tasks_fast(
        cls,
        config: dict,
        job_params: dict,
        job_id: Union[int, str],
        fast_context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Runs all configured optimization tasks for a given parameter point."""
        optimal_param = {}
        optimal_value = {}
        tasks = get_optimization_tasks(config)

        for metric_name in tasks:
            job_config = config.copy()
            job_config["simulation_parameters"] = job_params

            prefix = f"job_{job_id}_{metric_name}"
            p, v = cls.run_bisection_search_fast(
                job_config, prefix, metric_name, fast_context
            )
            optimal_param.update(p)
            optimal_value.update(v)

        return optimal_param, optimal_value


class OptimizationEngine:
    """First-class citizen optimization execution engine for closed-loop parameter search.

    Encapsulates goal seeking, convergence criteria, and iteration tracking
    under standard execution layer governance.
    """

    @classmethod
    def execute_tasks(
        cls,
        config: Dict[str, Any],
        job_params: Dict[str, Any],
        job_id: Union[int, str],
        fast_context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Executes all active optimization goals (e.g. Required_TBR bisection) for a job."""
        return BisectionOptimizer.run_optimization_tasks_fast(
            config=config,
            job_params=job_params,
            job_id=job_id,
            fast_context=fast_context,
        )
