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
        sensitivity_analysis = config.get("sensitivity_analysis", {})
        metrics_definition = sensitivity_analysis.get("metrics_definition", {})
        optimization_config = metrics_definition.get(optimization_metric_name, {})

        if (
            not optimization_config
            or "parameter_to_optimize" not in optimization_config
        ):
            raise ValueError(
                f"Optimization config for '{optimization_metric_name}' invalid."
            )

        param_to_optimize = optimization_config["parameter_to_optimize"]
        low_orig, high_orig = optimization_config["search_range"]
        tolerance = optimization_config.get("tolerance", 0.001)
        max_iterations = optimization_config.get("max_iterations", 10)
        stop_time = config["simulation"]["stop_time"]
        metric_max_value = optimization_config.get("metric_max_value", stop_time)
        metric_name = optimization_config.get("metric_name", "Self_Sufficiency_Time")
        source_column = optimization_config.get("source_column", "sds.inventory")

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
                            if source_column in results_df.columns:
                                if metric_name == "Self_Sufficiency_Time":
                                    metric_value = time_of_turning_point(
                                        results_df[source_column], results_df["time"]
                                    )
                                elif metric_name == "Doubling_Time":
                                    metric_value = calculate_doubling_time(
                                        results_df[source_column], results_df["time"]
                                    )
                                elif metric_name == "Startup_Inventory":
                                    metric_value = calculate_startup_inventory(
                                        results_df[source_column]
                                    )

                            job_dir = os.path.dirname(result_path)
                            shutil.rmtree(job_dir, ignore_errors=True)

                        except Exception:
                            pass

                    if metric_value < min(stop_time, current_metric_max_value):
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
