from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from tricys.analysis.metric import (
    apply_series_smoothing,
    calculate_doubling_time,
    calculate_startup_inventory,
    get_final_value,
    get_min_value,
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
        time_base = (
            str(config.get("simulation", {}).get("time_base", "second")).strip().lower()
        )
        time_factor = 3600.0 if time_base == "second" else 1.0
        stop_time = float(config.get("simulation", {}).get("stop_time", 1000.0))
        stop_time_in_time_base = stop_time / time_factor

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

            comparator = (
                target_metric.get("comparator")
                or target_metric.get("operator_type")
                or optimization_config.get("comparator")
            )
            if "doubling" in t_eval or "倍增" in t_eval:
                metric_name = "Doubling_Time"
                if not comparator:
                    comparator = "<="
            elif (
                "turn" in t_eval
                or "self" in t_eval
                or "拐点" in t_eval
                or "自持" in t_eval
                or "valley" in t_eval
                or "谷" in t_eval
            ):
                metric_name = "Self_Sufficiency_Time"
                if not comparator:
                    comparator = "<="
            elif "deficit" in t_eval or "drawdown" in t_eval or "亏空" in t_eval:
                metric_name = "Dynamic_Deficit"
                if not comparator:
                    comparator = "<="
            elif "startup" in t_eval:
                metric_name = "Startup_Inventory"
                if not comparator:
                    comparator = "<="
            elif (
                "final" in t_eval
                or "last" in t_eval
                or "residual" in t_eval
                or "残差" in t_eval
            ):
                metric_name = "Final_Value"
                if not comparator:
                    comparator = (
                        ">="
                        if (
                            "residual" in source_column.lower()
                            or "margin" in source_column.lower()
                        )
                        else "<="
                    )
            elif "min" in t_eval or "nadir" in t_eval:
                metric_name = "Min_Value"
                if not comparator:
                    comparator = ">="
            else:
                metric_name = optimization_config.get(
                    "metric_name", "Self_Sufficiency_Time"
                )
                if not comparator:
                    comparator = "<="

            if isinstance(t_thresh, list):
                metric_max_value = [float(v) for v in t_thresh]
            elif t_thresh is not None:
                metric_max_value = float(t_thresh)
            else:
                cfg_max = optimization_config.get(
                    "metric_max_value", stop_time_in_time_base
                )
                if isinstance(cfg_max, list):
                    metric_max_value = [float(v) for v in cfg_max]
                else:
                    metric_max_value = float(cfg_max)
        else:
            source_column = optimization_config.get("source_column") or "sds.I[1]"
            metric_name = optimization_config.get(
                "metric_name", "Self_Sufficiency_Time"
            )
            cfg_max = optimization_config.get(
                "metric_max_value", stop_time_in_time_base
            )
            if isinstance(cfg_max, list):
                metric_max_value = [float(v) for v in cfg_max]
            else:
                metric_max_value = float(cfg_max)

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
                    var_filt = config.get("simulation", {}).get(
                        "variableFilter"
                    ) or config.get("scenario", {}).get("variableFilter")
                    result_path = run_fast_subprocess_job(
                        job_params,
                        iter_id,
                        fast_context["exe"],
                        fast_context["xml"],
                        fast_context["om_bin"],
                        fast_context["temp_dir"],
                        config["simulation"],
                        variable_filter=var_filt,
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
                                time_series = results_df["time"] / time_factor
                                smooth_w = (
                                    target_metric.get("smooth_window")
                                    or target_metric.get("cycle_period")
                                    or optimization_config.get("smooth_window")
                                    or optimization_config.get("cycle_period")
                                )
                                smooth_method = (
                                    target_metric.get("smooth_method")
                                    or optimization_config.get("smooth_method", "mean")
                                    or "mean"
                                )

                                smooth_val = None
                                if smooth_w is not None:
                                    if (
                                        isinstance(smooth_w, str)
                                        and smooth_w in job_params
                                    ):
                                        try:
                                            smooth_val = float(job_params[smooth_w])
                                        except (ValueError, TypeError):
                                            smooth_val = None
                                    else:
                                        try:
                                            smooth_val = float(smooth_w)
                                        except (ValueError, TypeError):
                                            smooth_val = None

                                eval_series = results_df[target_col]
                                if smooth_val is not None and smooth_val > 0:
                                    eval_series = apply_series_smoothing(
                                        eval_series,
                                        time_series=time_series,
                                        smooth_window=smooth_val,
                                        method=smooth_method,
                                    )

                                if metric_name == "Self_Sufficiency_Time":
                                    try:
                                        metric_value = time_of_turning_point(
                                            eval_series,
                                            time_series,
                                            smooth_window=smooth_val,
                                        )
                                    except TypeError:
                                        metric_value = time_of_turning_point(
                                            eval_series, time_series
                                        )
                                elif metric_name == "Doubling_Time":
                                    interpolate = bool(
                                        target_metric.get("interpolate", False)
                                    )
                                    try:
                                        metric_value = calculate_doubling_time(
                                            eval_series,
                                            time_series,
                                            interpolate=interpolate,
                                        )
                                    except TypeError:
                                        metric_value = calculate_doubling_time(
                                            eval_series, time_series
                                        )
                                elif metric_name in (
                                    "Startup_Inventory",
                                    "Dynamic_Deficit",
                                ):
                                    metric_value = calculate_startup_inventory(
                                        eval_series
                                    )
                                elif metric_name == "Final_Value":
                                    metric_value = get_final_value(eval_series)
                                elif metric_name == "Min_Value":
                                    metric_value = get_min_value(eval_series)

                            job_dir = os.path.dirname(result_path)
                            shutil.rmtree(job_dir, ignore_errors=True)

                        except Exception:
                            pass

                    upper_limit = (
                        current_metric_max_value
                        if current_metric_max_value is not None
                        else stop_time
                    )
                    is_successful = False
                    if metric_value is not None and not pd.isna(metric_value):
                        if comparator in (">=", "ge", ">"):
                            is_successful = metric_value >= upper_limit
                        else:
                            is_successful = metric_value <= upper_limit

                    if is_successful:
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


class NestedBisectionOptimizer:
    """Performs bi-level / nested bisection search for coupled self-sufficiency parameters."""

    @classmethod
    def run_nested_bisection_fast(
        cls,
        config: Dict[str, Any],
        job_params: Dict[str, Any],
        job_id_prefix: str,
        nested_cfg: Dict[str, Any],
        fast_context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Executes coupled outer-inner bisection optimization."""
        if hasattr(nested_cfg, "model_dump"):
            nested_cfg = nested_cfg.model_dump()
        elif hasattr(nested_cfg, "dict"):
            nested_cfg = nested_cfg.dict()

        outer_cfg = nested_cfg.get("outer", {})
        inner_cfg = nested_cfg.get("inner", {})

        if hasattr(outer_cfg, "model_dump"):
            outer_cfg = outer_cfg.model_dump()
        elif hasattr(outer_cfg, "dict"):
            outer_cfg = outer_cfg.dict()

        if hasattr(inner_cfg, "model_dump"):
            inner_cfg = inner_cfg.model_dump()
        elif hasattr(inner_cfg, "dict"):
            inner_cfg = inner_cfg.dict()

        o_bounds = outer_cfg.get("search_bounds")
        i_bounds = inner_cfg.get("search_bounds")

        if (
            not o_bounds
            or len(o_bounds) != 2
            or float(o_bounds[0]) >= float(o_bounds[1])
        ):
            raise ValueError(
                "Outer loop must explicitly specify valid 'search_bounds' [low, high] with low < high."
            )
        if (
            not i_bounds
            or len(i_bounds) != 2
            or float(i_bounds[0]) >= float(i_bounds[1])
        ):
            raise ValueError(
                "Inner loop must explicitly specify valid 'search_bounds' [low, high] with low < high."
            )

        outer_param = (
            outer_cfg.get("parameter")
            or outer_cfg.get("parameter_to_optimize")
            or "I_S_0"
        )
        o_low_orig, o_high_orig = float(o_bounds[0]), float(o_bounds[1])
        o_tolerance = float(outer_cfg.get("tolerance", 1.0))
        o_max_iter = int(outer_cfg.get("max_iterations", 15))

        outer_target = outer_cfg.get("target_metric", {})
        if hasattr(outer_target, "model_dump"):
            outer_target = outer_target.model_dump()
        elif hasattr(outer_target, "dict"):
            outer_target = outer_target.dict()

        outer_source = (
            outer_target.get("source") or outer_cfg.get("source_column") or "sds.I[1]"
        )
        outer_eval = (
            outer_target.get("evaluate") or outer_target.get("operator") or "min_value"
        ).lower()
        case_thresh = job_params.get("I_reserve_req")
        if case_thresh is None:
            case_thresh = job_params.get("reserve_threshold")
        if case_thresh is None and outer_target.get("threshold") is not None:
            case_thresh = outer_target.get("threshold")
        if case_thresh is None and "f_b" in job_params and "eta_f" in job_params:
            # Physically derive I_r = (N_dot / (eta_f * f_b)) * t_r * q_fail (Abdou2021 Eq. 3.9)
            p_fus = float(job_params.get("P_fus", 3000.0))
            f_b_val = float(job_params["f_b"])
            eta_f_val = float(job_params["eta_f"])
            t_r_val = float(job_params.get("t_r", 24.0))
            q_fail_val = float(job_params.get("q_fail", 0.25))
            n_dot = 19.125 * (p_fus / 3000.0)
            if f_b_val * eta_f_val > 0:
                case_thresh = (n_dot / (eta_f_val * f_b_val)) * t_r_val * q_fail_val
        if case_thresh is None:
            case_thresh = outer_target.get("target_value", 0.0)
        outer_thresh = float(case_thresh)
        outer_comparator = (
            outer_target.get("comparator") or outer_target.get("operator_type") or ">="
        )

        inner_param = (
            inner_cfg.get("parameter")
            or inner_cfg.get("parameter_to_optimize")
            or (
                "blanket.TBR"
                if "blanket.TBR" in job_params or "Lambda" not in job_params
                else "Lambda"
            )
        )
        i_low_orig, i_high_orig = float(i_bounds[0]), float(i_bounds[1])
        i_tolerance = float(inner_cfg.get("tolerance", 0.001))
        i_max_iter = int(inner_cfg.get("max_iterations", 15))
        inner_metric_name = inner_cfg.get("metric_name", "Required_TBR")

        inner_target = inner_cfg.get("target_metric", {})
        if hasattr(inner_target, "model_dump"):
            inner_target = inner_target.model_dump()
        elif hasattr(inner_target, "dict"):
            inner_target = inner_target.dict()

        inner_source = (
            inner_target.get("source")
            or inner_cfg.get("source_column")
            or outer_source
            or "sds.I[1]"
        )
        inner_eval = (
            inner_target.get("evaluate")
            or inner_target.get("operator")
            or "doubling_time"
        ).lower()
        inner_comparator = (
            inner_target.get("comparator") or inner_target.get("operator_type") or "<="
        )

        sim_config = config.get("simulation") or {}
        if not sim_config and "scenario" in config:
            sc = config.get("scenario", {})
            tb = str(sc.get("time_base", "second")).strip().lower()
            stop_t = float(sc.get("stopTime", 1000.0))
            step_s = float(sc.get("stepSize", 1.0))
            if tb == "second":
                stop_t *= 3600.0
                step_s *= 3600.0
            sim_config = {
                "stop_time": stop_t,
                "step_size": step_s,
                "time_base": tb,
                "method": sc.get("method", "dassl"),
                "tolerance": sc.get("tolerance", 1e-6),
                "outputFormat": "csv",
            }

        time_base = (
            str(
                sim_config.get("time_base")
                or config.get("scenario", {}).get("time_base", "second")
            )
            .strip()
            .lower()
        )
        time_factor = 3600.0 if time_base == "second" else 1.0

        raw_inner_thresh = (
            inner_target.get("threshold")
            if inner_target.get("threshold") is not None
            else inner_target.get("target_value")
        )
        if raw_inner_thresh is None:
            raw_inner_thresh = job_params.get("t_d")
        if raw_inner_thresh is None:
            raw_inner_thresh = inner_target.get("max_value")

        if isinstance(raw_inner_thresh, list):
            inner_thresholds = [float(v) for v in raw_inner_thresh]
            is_list_input = True
        elif raw_inner_thresh is not None:
            inner_thresholds = [float(raw_inner_thresh)]
            is_list_input = False
        else:
            stop_time = float(sim_config.get("stop_time", 1000.0))
            inner_thresholds = [stop_time / time_factor]
            is_list_input = False

        all_optimal_params = {}
        all_optimal_values = {}

        for curr_td in inner_thresholds:
            o_low, o_high = o_low_orig, o_high_orig
            best_successful_outer_param = float("inf")
            best_successful_inner_param = float("inf")
            best_successful_outer_val = float("inf")
            best_successful_inner_val = float("inf")

            for o_iter in range(o_max_iter):
                if o_high - o_low < o_tolerance:
                    break
                o_mid = (o_low + o_high) / 2.0

                current_outer_job_params = job_params.copy()
                current_outer_job_params[outer_param] = o_mid

                i_low, i_high = i_low_orig, i_high_orig
                inner_best_param = float("inf")
                inner_best_val = float("inf")
                inner_best_df = None

                for i_iter in range(i_max_iter):
                    if i_high - i_low < i_tolerance:
                        break
                    i_mid = (i_low + i_high) / 2.0

                    sub_job_params = current_outer_job_params.copy()
                    sub_job_params[inner_param] = i_mid
                    iter_id = f"{job_id_prefix}_td{int(curr_td)}_out{o_iter}_in{i_iter}"

                    nested_var_filter = sim_config.get("variableFilter") or config.get(
                        "scenario", {}
                    ).get("variableFilter")
                    res_path = run_fast_subprocess_job(
                        sub_job_params,
                        iter_id,
                        fast_context["exe"],
                        fast_context["xml"],
                        fast_context["om_bin"],
                        fast_context["temp_dir"],
                        sim_config,
                        variable_filter=nested_var_filter,
                        inplace_execution=False,
                    )

                    i_metric_val = float("inf")
                    sub_df = None
                    if res_path and os.path.exists(res_path):
                        try:
                            sub_df = pd.read_csv(res_path)
                            target_col = (
                                inner_source if inner_source in sub_df.columns else None
                            )
                            if not target_col:
                                from tricys.analysis.tritium_cycle import (
                                    detect_sds_column,
                                )

                                auto_c = detect_sds_column(sub_df)
                                if auto_c and auto_c in sub_df.columns:
                                    target_col = auto_c

                            if target_col and target_col in sub_df.columns:
                                time_series = sub_df["time"] / time_factor
                                inner_smooth_w = inner_target.get(
                                    "smooth_window"
                                ) or inner_target.get("cycle_period")
                                inner_smooth_method = (
                                    inner_target.get("smooth_method", "mean") or "mean"
                                )

                                smooth_val = None
                                if inner_smooth_w is not None:
                                    if (
                                        isinstance(inner_smooth_w, str)
                                        and inner_smooth_w in sub_job_params
                                    ):
                                        try:
                                            smooth_val = float(
                                                sub_job_params[inner_smooth_w]
                                            )
                                        except (ValueError, TypeError):
                                            smooth_val = None
                                    else:
                                        try:
                                            smooth_val = float(inner_smooth_w)
                                        except (ValueError, TypeError):
                                            smooth_val = None

                                inner_series = sub_df[target_col]
                                if smooth_val is not None and smooth_val > 0:
                                    inner_series = apply_series_smoothing(
                                        inner_series,
                                        time_series=time_series,
                                        smooth_window=smooth_val,
                                        method=inner_smooth_method,
                                    )

                                if "doubling" in inner_eval or "倍增" in inner_eval:
                                    interpolate = bool(
                                        inner_target.get("interpolate", False)
                                    )
                                    try:
                                        i_metric_val = calculate_doubling_time(
                                            inner_series,
                                            time_series,
                                            interpolate=interpolate,
                                        )
                                    except TypeError:
                                        i_metric_val = calculate_doubling_time(
                                            inner_series, time_series
                                        )
                                elif "final" in inner_eval or "last" in inner_eval:
                                    i_metric_val = get_final_value(inner_series)
                                elif "min" in inner_eval or "nadir" in inner_eval:
                                    i_metric_val = get_min_value(inner_series)
                                elif (
                                    "turn" in inner_eval
                                    or "self" in inner_eval
                                    or "拐点" in inner_eval
                                ):
                                    i_metric_val = time_of_turning_point(
                                        inner_series,
                                        time_series,
                                        smooth_window=smooth_val,
                                    )

                            job_dir = os.path.dirname(res_path)
                            if job_dir:
                                shutil.rmtree(job_dir, ignore_errors=True)
                        except Exception as exc:
                            logger.debug(
                                f"Failed to read inner iteration result: {exc}"
                            )

                    if "final" in inner_eval:
                        inner_target_threshold = 2.0 * o_mid
                    else:
                        inner_target_threshold = curr_td

                    is_inner_success = False
                    if i_metric_val is not None and not pd.isna(i_metric_val):
                        if inner_comparator in ("<=", "le", "<"):
                            is_inner_success = i_metric_val <= inner_target_threshold
                        else:
                            is_inner_success = i_metric_val >= inner_target_threshold

                    if is_inner_success:
                        inner_best_param = i_mid
                        inner_best_val = i_metric_val
                        inner_best_df = sub_df
                        i_high = i_mid
                    else:
                        i_low = i_mid

                o_metric_val = float("-inf")
                if inner_best_df is not None:
                    target_col = (
                        outer_source if outer_source in inner_best_df.columns else None
                    )
                    if not target_col:
                        from tricys.analysis.tritium_cycle import detect_sds_column

                        auto_c = detect_sds_column(inner_best_df)
                        if auto_c and auto_c in inner_best_df.columns:
                            target_col = auto_c
                    if target_col and target_col in inner_best_df.columns:
                        outer_time_series = (
                            inner_best_df["time"] / time_factor
                            if "time" in inner_best_df.columns
                            else None
                        )
                        outer_smooth_w = outer_target.get(
                            "smooth_window"
                        ) or outer_target.get("cycle_period")
                        outer_smooth_method = (
                            outer_target.get("smooth_method", "mean") or "mean"
                        )

                        smooth_val = None
                        if outer_smooth_w is not None:
                            if (
                                isinstance(outer_smooth_w, str)
                                and outer_smooth_w in job_params
                            ):
                                try:
                                    smooth_val = float(job_params[outer_smooth_w])
                                except (ValueError, TypeError):
                                    smooth_val = None
                            else:
                                try:
                                    smooth_val = float(outer_smooth_w)
                                except (ValueError, TypeError):
                                    smooth_val = None

                        outer_series = inner_best_df[target_col]
                        if (
                            smooth_val is not None
                            and smooth_val > 0
                            and outer_time_series is not None
                        ):
                            outer_series = apply_series_smoothing(
                                outer_series,
                                time_series=outer_time_series,
                                smooth_window=smooth_val,
                                method=outer_smooth_method,
                            )

                        if "min" in outer_eval or "nadir" in outer_eval:
                            o_metric_val = get_min_value(outer_series)
                        elif "final" in outer_eval:
                            o_metric_val = get_final_value(outer_series)
                        elif "doubling" in outer_eval:
                            interpolate = bool(outer_target.get("interpolate", False))
                            try:
                                o_metric_val = calculate_doubling_time(
                                    outer_series,
                                    outer_time_series,
                                    interpolate=interpolate,
                                )
                            except TypeError:
                                o_metric_val = calculate_doubling_time(
                                    outer_series, outer_time_series
                                )

                # Dynamically resolve effective outer threshold from simulation outputs if available
                effective_outer_thresh = outer_thresh
                if inner_best_df is not None:
                    thresh_col = (
                        outer_target.get("threshold_source")
                        or outer_target.get("reserve_source")
                        or outer_cfg.get("threshold_source")
                        or outer_cfg.get("reserve_column")
                        or job_params.get("reserve_column")
                    )
                    if thresh_col and thresh_col in inner_best_df.columns:
                        effective_outer_thresh = float(
                            inner_best_df[thresh_col].iloc[-1]
                        )
                    elif "comp11_Storage.I_reserve_req" in inner_best_df.columns:
                        effective_outer_thresh = float(
                            inner_best_df["comp11_Storage.I_reserve_req"].iloc[-1]
                        )
                    elif "I_reserve_req" in inner_best_df.columns:
                        effective_outer_thresh = float(
                            inner_best_df["I_reserve_req"].iloc[-1]
                        )
                    elif "sds.I_reserve_req" in inner_best_df.columns:
                        effective_outer_thresh = float(
                            inner_best_df["sds.I_reserve_req"].iloc[-1]
                        )
                    elif "Cycle.I_reserve_req" in inner_best_df.columns:
                        effective_outer_thresh = float(
                            inner_best_df["Cycle.I_reserve_req"].iloc[-1]
                        )

                is_outer_success = False
                if o_metric_val is not None and not pd.isna(o_metric_val):
                    if outer_comparator in (">=", "ge", ">"):
                        is_outer_success = o_metric_val >= effective_outer_thresh
                    else:
                        is_outer_success = o_metric_val <= effective_outer_thresh

                logger.info(
                    f"[NestedBisection] Target td={curr_td}: Outer iter {o_iter} -> "
                    f"{outer_param}={o_mid:.2f}, Inner {inner_metric_name}={inner_best_param:.4f}, "
                    f"Nadir={o_metric_val:.2f} g (Req {outer_comparator} {effective_outer_thresh} g) -> "
                    f"{'SUCCESS' if is_outer_success else 'DEFICIT'}"
                )

                if is_outer_success:
                    best_successful_outer_param = o_mid
                    best_successful_inner_param = inner_best_param
                    best_successful_outer_val = o_metric_val
                    best_successful_inner_val = inner_best_val
                    o_high = o_mid
                else:
                    o_low = o_mid

            if best_successful_outer_param == float("inf"):
                best_successful_outer_param = o_high
            if best_successful_inner_param == float("inf"):
                best_successful_inner_param = i_high

            suffix = f"({curr_td} units)" if is_list_input else ""
            all_optimal_params[f"{outer_param}{suffix}"] = best_successful_outer_param
            all_optimal_params[f"{inner_metric_name}{suffix}"] = (
                best_successful_inner_param
            )
            all_optimal_values[f"Min_Value_for_{outer_param}{suffix}"] = (
                best_successful_outer_val
            )
            all_optimal_values[f"Doubling_Time_for_{inner_metric_name}{suffix}"] = (
                best_successful_inner_val
            )

        return all_optimal_params, all_optimal_values


class OptimizationEngine:
    """First-class citizen optimization execution engine for closed-loop parameter search.

    Encapsulates goal seeking, convergence criteria, and iteration tracking
    under standard execution layer governance.
    """

    @classmethod
    def has_optimization(cls, config: Dict[str, Any]) -> bool:
        """Determines if any optimization tasks (single or nested) are configured."""
        if not isinstance(config, dict):
            return False
        task_cfg = (
            config.get("task", {}) if isinstance(config.get("task"), dict) else {}
        )
        if task_cfg.get("type") == "optimize":
            return True
        if task_cfg.get("optimization"):
            return True
        from tricys.analysis.metric import get_optimization_tasks

        return bool(get_optimization_tasks(config))

    @classmethod
    def execute_tasks(
        cls,
        config: Dict[str, Any],
        job_params: Dict[str, Any],
        job_id: Union[int, str],
        fast_context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Executes active optimization goals (nested bisection or single Required_TBR bisection) for a job."""
        task_cfg = (
            config.get("task", {}) if isinstance(config.get("task"), dict) else {}
        )
        nested_opt_cfg = task_cfg.get("optimization") or (
            task_cfg if task_cfg.get("type") == "optimize" else None
        )
        if hasattr(nested_opt_cfg, "model_dump"):
            nested_opt_cfg = nested_opt_cfg.model_dump()
        elif hasattr(nested_opt_cfg, "dict"):
            nested_opt_cfg = nested_opt_cfg.dict()

        if nested_opt_cfg and (
            nested_opt_cfg.get("algorithm") in ("nested_bisection", "dual_basis")
            or "outer" in nested_opt_cfg
            or "track" in nested_opt_cfg
            or "track1" in nested_opt_cfg
        ):
            track = str(nested_opt_cfg.get("track") or "").strip().lower()

            if track in ("track1", "track1_basis", "basis", "dual_basis"):
                prefix = f"job_{job_id}_track1"
                from tricys.analysis.dual_basis import DualBasisOptimizer

                return DualBasisOptimizer.run_dual_basis_fast(
                    config=config,
                    job_params=job_params,
                    job_id_prefix=prefix,
                    opt_cfg=nested_opt_cfg,
                    fast_context=fast_context,
                )
            elif track in ("track2", "track2_bisection", "bisection"):
                prefix = f"job_{job_id}_nested"
                return NestedBisectionOptimizer.run_nested_bisection_fast(
                    config=config,
                    job_params=job_params,
                    job_id_prefix=prefix,
                    nested_cfg=nested_opt_cfg,
                    fast_context=fast_context,
                )
            elif track == "auto":
                # Auto-routing based on model linearity & threshold presence
                model_cfg = config.get("model", {})
                model_file = str(model_cfg.get("fileName", "")).lower()
                model_name = str(model_cfg.get("modelName", "")).lower()

                has_thresholds = (
                    any(
                        k in job_params
                        for k in ("I_th", "threshold_inv", "retention_thresh")
                    )
                    or "cfedr" in model_file
                    or "cfedr" in model_name
                )

                if not has_thresholds and (
                    "outer" not in nested_opt_cfg
                    or not nested_opt_cfg.get("outer", {}).get("search_bounds")
                ):
                    prefix = f"job_{job_id}_track1"
                    from tricys.analysis.dual_basis import DualBasisOptimizer

                    return DualBasisOptimizer.run_dual_basis_fast(
                        config=config,
                        job_params=job_params,
                        job_id_prefix=prefix,
                        opt_cfg=nested_opt_cfg,
                        fast_context=fast_context,
                    )
                else:
                    prefix = f"job_{job_id}_nested"
                    return NestedBisectionOptimizer.run_nested_bisection_fast(
                        config=config,
                        job_params=job_params,
                        job_id_prefix=prefix,
                        nested_cfg=nested_opt_cfg,
                        fast_context=fast_context,
                    )

            # Default fallback: preserve existing NestedBisectionOptimizer behavior
            prefix = f"job_{job_id}_nested"
            return NestedBisectionOptimizer.run_nested_bisection_fast(
                config=config,
                job_params=job_params,
                job_id_prefix=prefix,
                nested_cfg=nested_opt_cfg,
                fast_context=fast_context,
            )

        return BisectionOptimizer.run_optimization_tasks_fast(
            config=config,
            job_params=job_params,
            job_id=job_id,
            fast_context=fast_context,
        )
