#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TRICYS Track 1 Dual-Basis Closed-Form Mapping & Direct Extremum Optimization Engine.

Implements the patented Track 1 algorithm:
"双基函数预积分-闭式映射-解析极值直解/代数二分法"
for linear / unconstrained tritium fuel cycle dynamic systems.
"""

import logging
import os
import shutil
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from tricys.analysis.tritium_cycle import detect_sds_column
from tricys.simulation.engines.standard import run_fast_subprocess_job

logger = logging.getLogger(__name__)


class DualBasisOptimizer:
    """Performs Track 1 dual-basis closed-form direct solving and algebraic bisection."""

    @classmethod
    def run_dual_basis_fast(
        cls,
        config: Dict[str, Any],
        job_params: Dict[str, Any],
        job_id_prefix: str,
        opt_cfg: Dict[str, Any],
        fast_context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Executes Track 1 dual-basis pre-integration and analytical closed-form solving.

        Args:
            config: Full pipeline/simulation configuration.
            job_params: Parameter dictionary for current job.
            job_id_prefix: Prefix string for subprocess job isolation.
            opt_cfg: Optimization configuration dictionary.
            fast_context: Precompiled Modelica simulation context.

        Returns:
            Tuple of (optimal_params, optimal_values).
        """
        if hasattr(opt_cfg, "model_dump"):
            opt_cfg = opt_cfg.model_dump()
        elif hasattr(opt_cfg, "dict"):
            opt_cfg = opt_cfg.dict()

        outer_cfg = opt_cfg.get("outer", {}) or {}
        inner_cfg = opt_cfg.get("inner", {}) or {}
        track1_cfg = opt_cfg.get("track1", {}) or {}

        if hasattr(outer_cfg, "model_dump"):
            outer_cfg = outer_cfg.model_dump()
        elif hasattr(outer_cfg, "dict"):
            outer_cfg = outer_cfg.dict()

        if hasattr(inner_cfg, "model_dump"):
            inner_cfg = inner_cfg.model_dump()
        elif hasattr(inner_cfg, "dict"):
            inner_cfg = inner_cfg.dict()

        if hasattr(track1_cfg, "model_dump"):
            track1_cfg = track1_cfg.model_dump()
        elif hasattr(track1_cfg, "dict"):
            track1_cfg = track1_cfg.dict()

        outer_param = (
            outer_cfg.get("parameter")
            or outer_cfg.get("parameter_to_optimize")
            or "I_S_0"
        )
        inner_param = (
            inner_cfg.get("parameter")
            or inner_cfg.get("parameter_to_optimize")
            or "Lambda"
        )
        inner_metric_name = inner_cfg.get("metric_name", "Required_TBR")

        outer_target = outer_cfg.get("target_metric", {}) or {}
        inner_target = inner_cfg.get("target_metric", {}) or {}

        sim_config = config.get("simulation") or {}
        if not sim_config and "scenario" in config:
            from tricys.utils.config_utils import normalize_config

            config = normalize_config(config)
            sim_config = config.get("simulation", {})

        time_base = (
            str(
                sim_config.get("time_base")
                or config.get("scenario", {}).get("time_base", "second")
            )
            .strip()
            .lower()
        )
        time_factor = 3600.0 if time_base == "second" else 1.0

        default_decay = 1.7838e-9 if time_base == "second" else 6.422e-6
        lambda_decay = float(job_params.get("lambda_decay", default_decay))

        # Inner doubling times will be determined below, outer threshold resolved from df1

        # Determine target doubling time(s)
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
            stop_time = float(sim_config.get("stop_time", 50000.0))
            inner_thresholds = [stop_time / time_factor]
            is_list_input = False

        # Execute 2 black-box pre-integration subprocess runs
        # Run 1: Lambda = 0, I_S_0 = 0 -> yields Phi_burn(t)
        sub_job_1 = job_params.copy()
        sub_job_1[outer_param] = 0.0
        sub_job_1[inner_param] = 0.0
        iter_id_1 = f"{job_id_prefix}_basis_burn"

        res_path_1 = run_fast_subprocess_job(
            sub_job_1,
            iter_id_1,
            fast_context["exe"],
            fast_context["xml"],
            fast_context["om_bin"],
            fast_context["temp_dir"],
            sim_config,
            inplace_execution=False,
        )

        df1 = None
        if res_path_1 and os.path.exists(res_path_1):
            try:
                df1 = pd.read_csv(res_path_1)
            finally:
                job_dir_1 = os.path.dirname(res_path_1)
                if job_dir_1:
                    shutil.rmtree(job_dir_1, ignore_errors=True)

        if df1 is None or df1.empty:
            raise RuntimeError(
                f"Track 1 Pre-integration 1 (burn) failed for job {job_id_prefix}"
            )

        # Run 2: Lambda = 1, I_S_0 = 0 -> yields Phi_breed(t) + Phi_burn(t)
        sub_job_2 = job_params.copy()
        sub_job_2[outer_param] = 0.0
        sub_job_2[inner_param] = 1.0
        iter_id_2 = f"{job_id_prefix}_basis_breed"

        res_path_2 = run_fast_subprocess_job(
            sub_job_2,
            iter_id_2,
            fast_context["exe"],
            fast_context["xml"],
            fast_context["om_bin"],
            fast_context["temp_dir"],
            sim_config,
            inplace_execution=False,
        )

        df2 = None
        if res_path_2 and os.path.exists(res_path_2):
            try:
                df2 = pd.read_csv(res_path_2)
            finally:
                job_dir_2 = os.path.dirname(res_path_2)
                if job_dir_2:
                    shutil.rmtree(job_dir_2, ignore_errors=True)

        if df2 is None or df2.empty:
            raise RuntimeError(
                f"Track 1 Pre-integration 2 (breed) failed for job {job_id_prefix}"
            )

        # Locate storage column
        target_col = (
            track1_cfg.get("sds_column")
            or outer_target.get("source")
            or inner_target.get("source")
        )
        if not target_col or target_col not in df1.columns:
            auto_c = detect_sds_column(df1)
            if auto_c and auto_c in df1.columns:
                target_col = auto_c
            elif "sds.I[1]" in df1.columns:
                target_col = "sds.I[1]"
            else:
                target_col = "comp11_Storage.inventory"

        if target_col not in df1.columns or target_col not in df2.columns:
            raise ValueError(
                f"Storage column '{target_col}' not found in simulation outputs."
            )

        t_arr = np.array(df1["time"]) / time_factor
        Phi_burn = np.array(df1[target_col], dtype=np.float64)
        Phi_breed = np.array(df2[target_col], dtype=np.float64) - Phi_burn

        # Determine target reserve inventory threshold from Modelica simulation dataframe or configuration
        case_thresh = None

        # 1. Primary: read directly from Modelica simulation dataframe via specified or default column
        thresh_col = (
            outer_target.get("threshold_source")
            or outer_target.get("reserve_source")
            or outer_cfg.get("threshold_source")
            or outer_cfg.get("reserve_column")
            or track1_cfg.get("reserve_column")
            or job_params.get("reserve_column")
        )
        if thresh_col and thresh_col in df1.columns:
            case_thresh = float(df1[thresh_col].iloc[-1])
        elif "comp11_Storage.I_reserve_req" in df1.columns:
            case_thresh = float(df1["comp11_Storage.I_reserve_req"].iloc[-1])
        elif "I_reserve_req" in df1.columns:
            case_thresh = float(df1["I_reserve_req"].iloc[-1])
        elif "sds.I_reserve_req" in df1.columns:
            case_thresh = float(df1["sds.I_reserve_req"].iloc[-1])
        elif "Cycle.I_reserve_req" in df1.columns:
            case_thresh = float(df1["Cycle.I_reserve_req"].iloc[-1])

        # 2. Secondary: parameter override in job_params
        if case_thresh is None:
            case_thresh = job_params.get("I_reserve_req")
        if case_thresh is None:
            case_thresh = job_params.get("reserve_threshold")

        # 3. Tertiary: fixed threshold in outer_target
        if case_thresh is None and outer_target.get("threshold") is not None:
            case_thresh = outer_target.get("threshold")

        # 4. Fallback: target_value or 0.0
        if case_thresh is None:
            case_thresh = outer_target.get("target_value", 0.0)

        outer_thresh = float(case_thresh)

        exp_t = np.exp(-lambda_decay * t_arr)
        mode = str(track1_cfg.get("mode", "direct")).strip().lower()
        interp_method = (
            str(track1_cfg.get("interpolation_method", "pchip")).strip().lower()
        )
        subgrid_refine = bool(track1_cfg.get("subgrid_nadir_refinement", True))
        rel_tol = float(track1_cfg.get("relative_tolerance", 1e-5))

        def interpolate_signal(y_arr: np.ndarray, t_target: float) -> float:
            if interp_method == "pchip":
                try:
                    from scipy.interpolate import PchipInterpolator

                    pchip_obj = PchipInterpolator(t_arr, y_arr)
                    return float(pchip_obj(t_target))
                except Exception:
                    pass
            elif interp_method == "cubic":
                try:
                    from scipy.interpolate import CubicSpline

                    cs_obj = CubicSpline(t_arr, y_arr)
                    return float(cs_obj(t_target))
                except Exception:
                    pass
            return float(np.interp(t_target, t_arr, y_arr))

        all_optimal_params = {}
        all_optimal_values = {}

        for curr_td in inner_thresholds:
            # Interpolate values at target doubling time curr_td (pure storage_only basis)
            Phi_breed_td = interpolate_signal(Phi_breed, curr_td)
            Phi_burn_td = interpolate_signal(Phi_burn, curr_td)

            if abs(Phi_breed_td) < 1e-12:
                raise ZeroDivisionError(
                    f"Track 1: Phi_breed at doubling time td={curr_td} is zero, cannot solve."
                )

            exp_td = float(np.exp(-lambda_decay * curr_td))

            # Transient mask: search for nadir within t <= curr_td
            mask = (t_arr <= curr_td) & (t_arr > 0.0)
            if not np.any(mask):
                mask = t_arr <= curr_td

            if mode == "direct":
                # Direct analytical extremum solution (Mode A3-1, 0 iterations)
                factor_td = (2.0 - exp_td) / Phi_breed_td
                Psi = exp_t + factor_td * Phi_breed
                Omega = Phi_burn - (Phi_burn_td / Phi_breed_td) * Phi_breed

                Psi_mask = Psi[mask]
                Omega_mask = Omega[mask]
                t_mask = t_arr[mask]

                quotient = (outer_thresh - Omega_mask) / Psi_mask
                best_idx = int(np.argmax(quotient))
                I_S_0 = float(quotient[best_idx])

                # Subgrid quadratic vertex refinement for lowest storage nadir
                if subgrid_refine and 0 < best_idx < len(quotient) - 1:
                    t0_pt, t1_pt, t2_pt = (
                        t_mask[best_idx - 1],
                        t_mask[best_idx],
                        t_mask[best_idx + 1],
                    )
                    y0_pt, y1_pt, y2_pt = (
                        quotient[best_idx - 1],
                        quotient[best_idx],
                        quotient[best_idx + 1],
                    )

                    dt0 = t0_pt - t1_pt
                    dt2 = t2_pt - t1_pt
                    dy0 = y0_pt - y1_pt
                    dy2 = y2_pt - y1_pt
                    det = dt0 * dt2 * (dt0 - dt2)

                    if abs(det) > 1e-12:
                        a_quad = (dt2 * dy0 - dt0 * dy2) / det
                        b_quad = (dt0**2 * dy2 - dt2**2 * dy0) / det
                        if a_quad < 0.0:
                            dt_star = -b_quad / (2.0 * a_quad)
                            if dt0 <= dt_star <= dt2:
                                q_star = y1_pt - (b_quad**2) / (4.0 * a_quad)
                                if q_star >= y1_pt:
                                    I_S_0 = float(q_star)

                if I_S_0 < outer_thresh:
                    I_S_0 = outer_thresh

                Required_TBR = float(
                    ((2.0 - exp_td) * I_S_0 - Phi_burn_td) / Phi_breed_td
                )
            else:
                # Vectorized in-memory algebraic bisection (Mode A3-2)
                o_bounds = outer_cfg.get("search_bounds")
                if o_bounds and len(o_bounds) == 2:
                    low, high = float(o_bounds[0]), float(o_bounds[1])
                else:
                    low, high = outer_thresh, max(300000.0, outer_thresh * 10.0)

                tol = max(
                    outer_thresh * rel_tol, float(outer_cfg.get("tolerance", 0.05))
                )
                for _ in range(60):
                    if (high - low) / max(low, 1.0) < rel_tol or (high - low) < tol:
                        break
                    mid = (low + high) / 2.0
                    lam = ((2.0 - exp_td) * mid - Phi_burn_td) / Phi_breed_td
                    traj = mid * exp_t + lam * Phi_breed + Phi_burn
                    if np.min(traj[mask]) >= outer_thresh:
                        high = mid
                    else:
                        low = mid
                I_S_0 = high
                Required_TBR = float(
                    ((2.0 - exp_td) * I_S_0 - Phi_burn_td) / Phi_breed_td
                )

            # Synthesize final trajectory metrics
            final_traj = I_S_0 * exp_t + Required_TBR * Phi_breed + Phi_burn
            min_nadir = float(np.min(final_traj[mask]))

            suffix = f"({curr_td} units)" if is_list_input else ""
            all_optimal_params[f"{outer_param}{suffix}"] = I_S_0
            all_optimal_params[f"{inner_metric_name}{suffix}"] = Required_TBR
            all_optimal_values[f"Min_Value_for_{outer_param}{suffix}"] = min_nadir
            all_optimal_values[f"Doubling_Time_for_{inner_metric_name}{suffix}"] = (
                curr_td
            )

            logger.info(
                f"[Track 1 DualBasis] Solved ({mode}) td={curr_td}: {outer_param}={I_S_0:.2f} g, "
                f"{inner_metric_name}={Required_TBR:.4f}, Nadir={min_nadir:.2f} g (Req {outer_thresh:.2f} g)"
            )

        return all_optimal_params, all_optimal_values
