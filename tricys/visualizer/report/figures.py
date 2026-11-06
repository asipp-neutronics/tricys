"""Publication-grade scientific chart generation utilities for TRICYS engineering documentation.

Adheres strictly to fusion nuclear engineering standards (CFEDR 2000MW, ITER, DEMO):
1. fig1_inventory_trajectories: Full 2-Year / Timeline Storage Inventory Trajectories
2. fig2_dt_aligned_valley_comparison: Startup Valley & Self-Sufficiency Turning Points
3. fig3_subsystem_dynamics: Two-panel Linked Subsystems Paving & SDS Depletion Dynamics
4. fig4_metrics_barchart: Startup Tritium Deficit & Self-Sufficiency Time Dual Bar Chart
5. fig_mass_conservation: Timeseries Mass Conservation Relative Residual & Threshold
6. fig_sobol_sensitivity: Sobol First-Order & Total-Order Variance Decomposition
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tricys.analysis.tritium_cycle import (
    HOURS_PER_DAY,
    detect_sds_column,
    extract_startup_inventory_metrics,
    load_glossary_mapping,
    resolve_subsystem_metadata,
    resolve_time_factor_to_hours,
)

logger = logging.getLogger(__name__)

# Standard CFEDR Publication Palette
PALETTE = ["#D9534F", "#F0AD4E", "#2E7D32", "#1F77B4", "#9467BD", "#8C564B", "#17BECF"]


def resolve_plot_time_scale(
    t_max: float, config: Optional[Dict[str, Any]] = None
) -> Tuple[str, float]:
    """Resolves truthful time unit and scale factor based on config and data span.

    TRICYS schema time fields (stopTime, stepSize, t_dd, etc.) are always in hours.
    time_base specifies the Modelica model's time unit ('second' or 'hour').
    The plotted time unit is always hours ('h'), and the divisor scales raw simulation
    time (seconds or hours) to hours.
    """
    scale = resolve_time_factor_to_hours(t_max, config)
    return "h", scale


def _init_plot_style() -> None:
    """Sets publication-grade matplotlib typography and rendering parameters."""
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "WenQuanYi Micro Hei",
                "WenQuanYi Zen Hei",
                "Noto Sans CJK SC",
                "Source Han Sans CN",
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
                "Arial",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "lines.linewidth": 1.8,
        }
    )


def _fig_to_result(
    fig: plt.Figure,
    title: str,
    description: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Encodes matplotlib figure into clean Base64 data URI and saves file if requested."""
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, format="png", dpi=200, bbox_inches="tight")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"

    return {
        "title": title,
        "description": description,
        "image_data": data_uri,
        "file_path": output_path,
    }


def generate_inventory_trajectories_plot(
    results_df: pd.DataFrame,
    jobs_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    target_var: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fig 1: Full-timeline storage inventory trajectories with valley & end-point annotations."""
    if results_df is None or results_df.empty or "time" not in results_df.columns:
        return None

    plot_var = target_var or detect_sds_column(results_df)
    if not plot_var or plot_var not in results_df.columns:
        return None

    job_info_map: Dict[Any, Dict[str, Any]] = {}
    if jobs_df is not None and not jobs_df.empty:
        jid_col = "job_id" if "job_id" in jobs_df.columns else jobs_df.columns[0]
        for _, r in jobs_df.iterrows():
            jid = r[jid_col]
            name = r.get("name") or r.get("job_name") or f"工况 {jid}"
            t_dd = r.get("t_DD_end") or r.get("t_dd") or 0.0
            job_info_map[jid] = {"name": name, "t_dd": float(t_dd)}

    # Determine truthful time unit
    t_max = float(results_df["time"].max()) if "time" in results_df.columns else 0.0
    time_unit, time_scale = resolve_plot_time_scale(t_max, config=config)

    _init_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.5), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    job_ids = (
        sorted(results_df["job_id"].unique()) if "job_id" in results_df.columns else [1]
    )

    all_y_min, all_y_max = float("inf"), float("-inf")
    i_0_ref = None

    for i, jid in enumerate(job_ids):
        color = PALETTE[i % len(PALETTE)]
        subset = (
            results_df[results_df["job_id"] == jid].sort_values("time")
            if "job_id" in results_df.columns
            else results_df.sort_values("time")
        )
        if subset.empty:
            continue

        t = subset["time"].values / time_scale
        y = subset[plot_var].values
        info = job_info_map.get(jid, {"name": f"工况 {jid}", "t_dd": 0.0})
        label = info["name"]
        ax.plot(t, y, label=label, color=color, linewidth=2.0)

        all_y_min = min(all_y_min, float(np.min(y)))
        all_y_max = max(all_y_max, float(np.max(y)))
        if i_0_ref is None and len(y) > 0:
            i_0_ref = float(y[0])

        # Valley turning point
        min_idx = int(np.argmin(y))
        ax.plot(t[min_idx], y[min_idx], marker="o", markersize=7, color=color)

        # End point annotation
        t_end, y_end = t[-1], y[-1]
        ax.plot(t_end, y_end, marker="s", markersize=6, color=color)
        ax.annotate(
            f"{y_end:.0f} g\n(期末)",
            xy=(t_end, y_end),
            xytext=(
                t_end - (t[-1] - t[0]) * 0.12,
                y_end + (all_y_max - all_y_min) * 0.04,
            ),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            fontsize=8.5,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9),
        )

        # D-D phase background shading if present
        t_dd = float(info["t_dd"])
        if t_dd > 0:
            if time_unit == "d":
                t_dd_scaled = t_dd / HOURS_PER_DAY
            elif time_unit == "s":
                t_dd_scaled = t_dd * 3600.0
            else:  # "h"
                t_dd_scaled = t_dd
            ax.axvspan(
                0, t_dd_scaled, color=color, alpha=0.06, label=f"{label} (调试阶段)"
            )

    # Reference limits
    ax.axhline(
        0, color="gray", linestyle="--", linewidth=1.2, label="安全零底限 ($I=0$)"
    )
    if i_0_ref is not None and i_0_ref > 0:
        ax.axhline(
            i_0_ref,
            color="navy",
            linestyle=":",
            linewidth=1.2,
            alpha=0.75,
            label=f"初始装填基准 ($I_0 = {i_0_ref:.0f}$ g)",
        )

    ax.set_title(
        "全生命周期在役储罐存量演化轨迹对比 (Storage Inventory Trajectories)",
        color="#1E3A8A",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(f"仿真时间 ({time_unit})", color="#1F2937", fontsize=11)
    ax.set_ylabel("SDS 储罐在役存量 (g)", color="#1F2937", fontsize=11)
    ax.grid(True, linestyle=":", color="#E2E8F0", alpha=0.9)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=2)
    plt.tight_layout()

    return _fig_to_result(
        fig,
        title="全生命周期储罐存量演化轨迹对比图",
        description="多工况下在役储罐全生命周期的自持回升曲线、启动谷底转折点与期末存量标注",
        output_path=output_path,
    )


def generate_valley_comparison_plot(
    results_df: pd.DataFrame,
    jobs_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    target_var: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fig 2: Zoomed-in startup valley aligned at transient start."""
    if results_df is None or results_df.empty or "time" not in results_df.columns:
        return None

    plot_var = target_var or detect_sds_column(results_df)
    if not plot_var or plot_var not in results_df.columns:
        return None

    job_ids = (
        sorted(results_df["job_id"].unique()) if "job_id" in results_df.columns else [1]
    )

    # Physical verification: Check if an actual startup drawdown valley exists across any job
    has_valley = False
    for jid in job_ids:
        subset = (
            results_df[results_df["job_id"] == jid].sort_values("time")
            if "job_id" in results_df.columns
            else results_df.sort_values("time")
        )
        if not subset.empty and plot_var in subset.columns:
            arr = pd.to_numeric(subset[plot_var], errors="coerce").dropna().values
            if len(arr) > 0 and (np.min(arr) < arr[0] - 1e-4):
                has_valley = True
                break

    if not has_valley:
        # If storage inventory never drops below initial, no startup valley exists!
        return None

    job_info_map: Dict[Any, Dict[str, Any]] = {}
    if jobs_df is not None and not jobs_df.empty:
        jid_col = "job_id" if "job_id" in jobs_df.columns else jobs_df.columns[0]
        for _, r in jobs_df.iterrows():
            jid = r[jid_col]
            name = r.get("name") or r.get("job_name") or f"工况 {jid}"
            t_dd = r.get("t_DD_end") or r.get("t_dd") or 0.0
            job_info_map[jid] = {"name": name, "t_dd": float(t_dd)}

    # Determine truthful time unit
    t_max = float(results_df["time"].max()) if "time" in results_df.columns else 0.0
    time_unit, time_scale = resolve_plot_time_scale(t_max, config=config)
    raw_to_hours = 1.0 / time_scale

    _init_plot_style()
    fig, ax = plt.subplots(figsize=(9.8, 5.2), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    i_0_ref = None
    has_ignition_offset = False

    for i, jid in enumerate(job_ids):
        color = PALETTE[i % len(PALETTE)]
        subset = (
            results_df[results_df["job_id"] == jid].sort_values("time")
            if "job_id" in results_df.columns
            else results_df.sort_values("time")
        )
        if subset.empty:
            continue

        info = job_info_map.get(jid, {"name": f"工况 {jid}", "t_dd": 0.0})
        t_ign = float(info["t_dd"])
        if t_ign > 0:
            has_ignition_offset = True

        t_raw = subset["time"].values
        t_hours = t_raw * raw_to_hours

        # Align time comparison and subtraction in hours
        mask = t_hours >= t_ign
        if np.any(mask):
            t_dt_h = t_hours[mask]
            y = subset[plot_var].values[mask]
        else:
            t_dt_h = t_hours
            y = subset[plot_var].values
            t_ign = 0.0

        t_rel_h = t_dt_h - t_ign
        if time_unit == "d":
            t_rel = t_rel_h / HOURS_PER_DAY
        elif time_unit == "s":
            t_rel = t_rel_h * 3600.0
        else:
            t_rel = t_rel_h

        if i_0_ref is None and len(y) > 0:
            i_0_ref = float(y[0])

        ax.plot(t_rel, y, label=info["name"], color=color, linewidth=2.0)

        # Min turning point
        min_idx = int(np.argmin(y))
        t_min_rel = t_rel[min_idx]
        y_min = y[min_idx]

        ax.plot(t_min_rel, y_min, marker="*", markersize=10, color=color)
        ax.annotate(
            f"谷底: {y_min:.1f} g\nΔt: {t_min_rel:.1f} {time_unit}",
            xy=(t_min_rel, y_min),
            xytext=(
                t_min_rel + max(t_rel[-1] * 0.04, 5.0),
                y_min + 300 * (1 if i % 2 == 0 else -1),
            ),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
            fontsize=8.8,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9),
        )

    ax.axhline(
        0, color="crimson", linestyle="--", linewidth=1.3, label="零储量限 ($I=0$)"
    )
    if i_0_ref is not None:
        ax.axhline(
            i_0_ref,
            color="navy",
            linestyle=":",
            linewidth=1.1,
            alpha=0.7,
            label=f"初始装填基准 ({i_0_ref:.0f} g)",
        )

    x_title = (
        f"点火后流逝时间: $t - t_{{ignition}}$ ({time_unit})"
        if has_ignition_offset
        else f"仿真流逝时间: $t$ ({time_unit})"
    )
    ax.set_title(
        "储罐启动瞬态谷底局部特写与自持转折点",
        color="#1E3A8A",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(x_title, color="#1F2937", fontsize=11)
    ax.set_ylabel("储罐在役存量 (g)", color="#1F2937", fontsize=11)
    ax.grid(True, linestyle=":", color="#E2E8F0", alpha=0.9)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    return _fig_to_result(
        fig,
        title="储罐启动瞬态谷底特写对比图",
        description="各工况主储罐在下探自持拐点区间的局部特写与启动外供亏空评估",
        output_path=output_path,
    )


def generate_subsystem_dynamics_plot(
    results_df: pd.DataFrame,
    jobs_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    target_job_id: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fig 3: Two-panel linked figure (Upper: Subsystem Paving; Lower: SDS Depletion)."""
    if results_df is None or results_df.empty or "time" not in results_df.columns:
        return None

    sds_col = detect_sds_column(results_df)
    if not sds_col:
        return None

    # Pick target job
    jid = target_job_id
    if jid is None:
        jid = results_df["job_id"].iloc[0] if "job_id" in results_df.columns else 1

    if "job_id" in results_df.columns:
        match_mask = results_df["job_id"] == jid
        if not match_mask.any():
            match_mask = results_df["job_id"].astype(str) == str(jid)
        subset = results_df[match_mask].sort_values("time")
    else:
        subset = results_df.sort_values("time")
    if subset.empty:
        return None

    # Resolve human-readable case label & ignition time
    case_name = ""
    t_ign = 0.0
    if jobs_df is not None and not jobs_df.empty:
        jid_col = "job_id" if "job_id" in jobs_df.columns else jobs_df.columns[0]
        match = jobs_df[jobs_df[jid_col] == jid]
        if match.empty:
            match = jobs_df[jobs_df[jid_col].astype(str) == str(jid)]
        if not match.empty:
            cname = match.iloc[0].get("name") or match.iloc[0].get("job_name")
            if cname and pd.notnull(cname) and str(cname).strip():
                case_name = str(cname).strip()
            else:
                case_name = f"工况 {jid}"
            t_ign = float(match.iloc[0].get("t_DD_end", match.iloc[0].get("t_dd", 0.0)))
    elif target_job_id is not None:
        case_name = f"工况 {target_job_id}"

    t_max_raw = float(subset["time"].max()) if not subset.empty else 0.0
    time_unit, time_scale = resolve_plot_time_scale(t_max_raw, config=config)
    raw_to_hours = 1.0 / time_scale

    t_hours = subset["time"].values * raw_to_hours
    mask = t_hours >= t_ign
    if np.any(mask):
        sub_post = subset[mask].copy()
        t_rel_h = t_hours[mask] - t_ign
    else:
        sub_post = subset.copy()
        t_rel_h = t_hours
        t_ign = 0.0

    if time_unit == "d":
        t_plot = t_rel_h / HOURS_PER_DAY
    elif time_unit == "s":
        t_plot = t_rel_h * 3600.0
    else:
        t_plot = t_rel_h

    sub_plot = sub_post

    _init_plot_style()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 7.5), sharex=True, facecolor="#ffffff"
    )
    ax1.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")

    # Collect target subsystem columns
    target_cols: List[str] = []

    # Priority A: Extract explicitly configured variables from scenario.variableFilter
    if config:
        vf = config.get("scenario", {}).get("variableFilter", "")
        if vf:
            items = vf if isinstance(vf, list) else str(vf).split("|")
            for item in items:
                v = str(item).strip()
                if v and not v.startswith(".*") and v in sub_plot.columns:
                    if v not in ["time", "job_id", sds_col] and v not in target_cols:
                        if np.issubdtype(sub_plot[v].dtype, np.number):
                            target_cols.append(v)

    glossary_map = load_glossary_mapping(config=config)

    # Priority B: If variableFilter omitted or incomplete, discover all numerical non-time, non-primary-sds columns
    if not target_cols:
        for c in sub_plot.columns:
            if c not in ["time", "job_id", sds_col] and np.issubdtype(
                sub_plot[c].dtype, np.number
            ):
                if c not in target_cols:
                    target_cols.append(c)
    else:
        # Also append columns that exist in glossary if present in sub_plot
        if glossary_map:
            for c in sub_plot.columns:
                if c not in ["time", "job_id", sds_col] and np.issubdtype(
                    sub_plot[c].dtype, np.number
                ):
                    if c not in target_cols and (
                        c in glossary_map or c.lower() in glossary_map
                    ):
                        target_cols.append(c)

    turning_rel = float(t_plot[np.argmin(sub_plot[sds_col].values)])

    for idx, col in enumerate(target_cols):
        meta = resolve_subsystem_metadata(
            col, index=idx, config=config, glossary_map=glossary_map
        )
        arr = sub_plot[col].values
        color = meta["color"]
        ax1.plot(t_plot, arr, label=meta["name"], color=color, linewidth=1.8)

        # 99% mark
        tail = max(int(len(arr) * 0.1), 1)
        ss_val = np.mean(arr[-tail:])
        if ss_val > 0.5:
            idx_99 = np.where(arr >= ss_val * 0.99)[0]
            if len(idx_99) > 0:
                t_99 = t_plot[idx_99[0]]
                ax1.plot(t_99, arr[idx_99[0]], marker="o", markersize=5, color=color)

    # Vertical line indicating turning point
    ax1.axvline(turning_rel, color="crimson", linestyle=":", linewidth=1.5, alpha=0.85)
    ax1.set_ylabel("过程设备滞留量 (g)", fontsize=11, color="#1F2937")
    ax1_title = (
        f"各工艺子系统在役滞留演化与铺底时序 ({case_name})"
        if (target_job_id is not None and case_name)
        else "各工艺子系统在役滞留演化与铺底时序 (Process Holdup Dynamics)"
    )
    ax1.set_title(ax1_title, fontsize=12, color="#1E3A8A", pad=8)
    ax1.grid(True, linestyle=":", color="#E2E8F0", alpha=0.9)
    if target_cols:
        ax1.legend(
            loc="upper right",
            framealpha=0.9,
            fontsize=8.5,
            ncol=2 if len(target_cols) > 3 else 1,
        )

    # Lower panel: SDS Storage Depletion
    sds_label = (
        f"SDS 储罐在役存量 ({case_name})"
        if (target_job_id is not None and case_name)
        else f"SDS 储罐在役存量 (Job {jid})"
    )
    ax2.plot(
        t_plot,
        sub_plot[sds_col].values,
        label=sds_label,
        color="#D9534F",
        linewidth=2.2,
    )
    sds_min = np.min(sub_plot[sds_col].values)
    i_0_job = float(sub_plot[sds_col].iloc[0])
    turning_rel = float(t_plot[np.argmin(sub_plot[sds_col].values)])
    ax2.plot(turning_rel, sds_min, marker="o", markersize=7, color="crimson")

    t_max = float(t_plot.max()) if len(t_plot) > 0 else 0.0
    offset_x = max(t_max * 0.03, 0.5)

    if i_0_job > sds_min + 1e-3:
        ax2.axvline(
            turning_rel,
            color="crimson",
            linestyle=":",
            linewidth=1.5,
            alpha=0.85,
            label=f"自持拐点 Δt_ss={turning_rel:.1f} {time_unit}",
        )
        offset_y = max((i_0_job - sds_min) * 0.35, 300.0)
        ax2.annotate(
            f"启动外供亏空: {i_0_job - sds_min:.1f} g\n拐点时刻: {turning_rel:.1f} {time_unit}",
            xy=(turning_rel, sds_min),
            xytext=(turning_rel + offset_x, sds_min + offset_y),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.1),
            fontsize=9.0,
            fontweight="bold",
            color="crimson",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="crimson", alpha=0.85),
        )
    else:
        ax2.axvline(
            turning_rel,
            color="navy",
            linestyle=":",
            linewidth=1.5,
            alpha=0.85,
            label=f"最低存量时刻 t={turning_rel:.1f} {time_unit}",
        )
        ax2.annotate(
            f"初始装填: {i_0_job:.1f} g\n运行期未见外供亏空",
            xy=(turning_rel, sds_min),
            xytext=(turning_rel + offset_x, sds_min + 200),
            arrowprops=dict(arrowstyle="->", color="navy", lw=1.1),
            fontsize=9.0,
            fontweight="bold",
            color="navy",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="navy", alpha=0.85),
        )

    ax2.axhline(
        0, color="gray", linestyle="--", linewidth=1.1, label="安全零限 ($I=0$)"
    )
    ax2.axhline(
        i_0_job,
        color="navy",
        linestyle=":",
        linewidth=1.1,
        alpha=0.7,
        label=f"初始装填 ($I_0={i_0_job:.0f}$ g)",
    )

    x_sub_label = (
        f"点火后流逝时间: $t - t_{{ignition}}$ ({time_unit})"
        if t_ign > 0
        else f"仿真流逝时间: $t$ ({time_unit})"
    )
    ax2.set_xlabel(x_sub_label, fontsize=11, color="#1F2937")
    ax2.set_ylabel("SDS 储罐在役存量 (g)", fontsize=11, color="#1F2937")
    ax2.grid(True, linestyle=":", color="#E2E8F0", alpha=0.9)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    fig_title = (
        f"各工艺子组件铺底与储罐联动演化双子图 ({case_name})"
        if (target_job_id is not None and case_name)
        else "各工艺子组件铺底与储罐联动演化双子图"
    )
    fig_desc = (
        f"工况 [{case_name}] 下过程设备滞留铺底建立过程与储存供料系统下探自持拐点的物理时间对齐关系"
        if (target_job_id is not None and case_name)
        else "上下联动展示各过程设备滞留铺底建立过程与储存供料系统下探自持拐点的物理时间对齐关系"
    )
    return _fig_to_result(
        fig,
        title=fig_title,
        description=fig_desc,
        output_path=output_path,
    )


def generate_metrics_comparison_plot(
    metrics_df: pd.DataFrame,
    jobs_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Fig 4: Dynamic multi-metric bar chart comparing real evaluated metrics across cases."""
    if metrics_df is None or metrics_df.empty:
        return None

    known_metrics = set()
    if config:
        task_metrics = (
            config.get("task", {}).get("metrics", {})
            if isinstance(config.get("task"), dict)
            else {}
        )
        known_metrics |= set(task_metrics.keys())
        known_metrics |= set(config.get("metrics_definition", {}).keys())

    ignored_cols = {
        "job_id",
        "id",
        "name",
        "job_name",
        "case_name",
        "job",
        "status",
        "工况编号",
        "工况名称",
    }
    if jobs_df is not None and not jobs_df.empty:
        for c in jobs_df.columns:
            if c.lower() not in ignored_cols and c not in known_metrics:
                ignored_cols.add(c)

    metric_cols = []
    for c in metrics_df.columns:
        if (
            c.lower() not in ignored_cols and c not in ignored_cols
        ) or c in known_metrics:
            if c.lower() in {"job_id", "id", "工况编号"}:
                continue
            num = pd.to_numeric(metrics_df[c], errors="coerce")
            if num.notna().sum() >= 1 and c not in metric_cols:
                metric_cols.append(c)

    if not metric_cols:
        return None

    # Limit to top 4 metrics for clear publication-grade presentation
    plot_metric_cols = metric_cols[:4]
    n_metrics = len(plot_metric_cols)

    glossary_map = load_glossary_mapping(config=config)

    # Build friendly labels for each case
    labels = []
    case_name_map = {}
    if jobs_df is not None and not jobs_df.empty:
        jid_col = "job_id" if "job_id" in jobs_df.columns else jobs_df.columns[0]
        for _, r in jobs_df.iterrows():
            jid = r[jid_col]
            name = (
                r.get("name")
                or r.get("job_name")
                or r.get("case_name")
                or f"工况 {jid}"
            )
            case_name_map[jid] = str(name)

    for idx, r in metrics_df.iterrows():
        jid = (
            r.get("job_id")
            if "job_id" in r
            else (r.get("id") if "id" in r else idx + 1)
        )
        name = (
            r.get("工况名称")
            or r.get("name")
            or case_name_map.get(jid)
            or f"工况 {jid}"
        )
        c_str = str(name).replace("_", "\n", 2) if len(str(name)) > 15 else str(name)
        labels.append(c_str)

    _init_plot_style()
    if n_metrics == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7.5, 4.8), facecolor="#ffffff")
        axes = [axes]
    elif n_metrics == 2:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), facecolor="#ffffff")
    elif n_metrics == 3:
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), facecolor="#ffffff")
    else:
        fig, axes_arr = plt.subplots(2, 2, figsize=(11.5, 8.5), facecolor="#ffffff")
        axes = axes_arr.flatten()

    x = np.arange(len(labels))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    for ax, col in zip(axes, plot_metric_cols):
        ax.set_facecolor("#ffffff")
        vals = pd.to_numeric(metrics_df[col], errors="coerce").values

        bars = ax.bar(
            x,
            vals,
            color=colors,
            width=0.48,
            alpha=0.88,
            edgecolor="#374151",
            linewidth=0.9,
        )

        item = glossary_map.get(col) or glossary_map.get(col.lower())
        if item and item.get("zh"):
            col_title = f"{item['zh']} ({col})"
        else:
            col_title = col

        unit_label = "指标量值"
        if (
            item
            and item.get("alias")
            and any(u in item["alias"] for u in ["g", "kg", "h", "s", "%"])
        ):
            unit_label = item["alias"]
        elif (
            "inventory" in col.lower()
            or "存量" in col
            or "装料" in col
            or "亏空" in col
        ):
            unit_label = "存量数值 (g)"
        elif "time" in col.lower() or "时滞" in col or "时间" in col:
            unit_label = "时间数值"
        elif "ratio" in col.lower() or "tbr" in col.lower():
            unit_label = "比值 / 产额比"

        ax.set_title(col_title, fontweight="bold", fontsize=11, color="#1E3A8A", pad=10)
        ax.set_ylabel(unit_label, fontsize=10.5, color="#1F2937")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.grid(True, axis="y", linestyle=":", color="#E2E8F0", alpha=0.9)

        for b in bars:
            h = b.get_height()
            if np.isnan(h):
                continue
            v_str = f"{h:.4g}" if abs(h) < 10000 else f"{h:.1f}"
            ax.annotate(
                v_str,
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 4 if h >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8.5,
                fontweight="bold",
                color="#1F2937",
            )

    plt.tight_layout()
    return _fig_to_result(
        fig,
        title="多工况核心性能指标横向对比柱状图",
        description="各评估工况在各项核心输出指标维度下的横向定量对比直方图",
        output_path=output_path,
    )


def generate_sobol_sensitivity_plot(
    sobol_records: Optional[List[Dict[str, Any]]] = None,
    jobs_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    task_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Generates sensitivity indices or variance contribution bar chart using real calculation data."""
    if not sobol_records:
        from tricys.analysis.tritium_cycle import calculate_real_sensitivity_indices

        sens_df = calculate_real_sensitivity_indices(
            jobs_df=jobs_df, metrics_df=metrics_df, task_config=task_config
        )
        if not sens_df.empty:
            if "全阶敏感度 (ST)" in sens_df.columns:
                sobol_records = [
                    {
                        "param": str(r["敏感参数"]),
                        "s1": float(r["一阶敏感度 (S1)"]),
                        "st": float(r["全阶敏感度 (ST)"]),
                    }
                    for _, r in sens_df.iterrows()
                ]
            elif "方差贡献率 (R²)" in sens_df.columns:
                sobol_records = [
                    {
                        "param": str(r["敏感参数"]),
                        "r2": float(r["方差贡献率 (R²)"]),
                        "r": float(r.get("Pearson 线性相关度 (r)", 0.0)),
                    }
                    for _, r in sens_df.iterrows()
                ]

    if not sobol_records:
        return None

    _init_plot_style()
    glossary_map = load_glossary_mapping(config=task_config)
    labels = []
    for r in sobol_records:
        p = r["param"]
        if glossary_map:
            item = glossary_map.get(p) or glossary_map.get(p.lower())
            if item and item.get("zh"):
                labels.append(f"{item['zh']}\n({p})")
            elif item and item.get("en"):
                labels.append(f"{item['en']}\n({p})")
            else:
                labels.append(p)
        else:
            labels.append(p)
    x = np.arange(len(labels))

    if "st" in sobol_records[0]:
        fig, ax = plt.subplots(figsize=(9.8, 4.8), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        width = 0.35
        s1_vals = [r["s1"] for r in sobol_records]
        st_vals = [r["st"] for r in sobol_records]

        b1 = ax.bar(
            x - width / 2,
            s1_vals,
            width,
            label="一阶敏感度指数 ($S_1$)",
            color="#1F77B4",
            alpha=0.85,
        )
        b2 = ax.bar(
            x + width / 2,
            st_vals,
            width,
            label="全阶敏感度指数 ($S_T$)",
            color="#D9534F",
            alpha=0.85,
        )

        ax.set_title(
            "Sobol 全局敏感性一阶与全阶方差分解对比图",
            color="#1E3A8A",
            fontsize=12.5,
            fontweight="bold",
            pad=10,
        )
        ax.set_ylabel("敏感度指数", fontsize=11, color="#1F2937")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.grid(True, axis="y", linestyle=":", color="#E2E8F0", alpha=0.9)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

        for b in list(b1) + list(b2):
            h = b.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

        plt.tight_layout()
        return _fig_to_result(
            fig,
            title="Sobol 全局敏感性指数对比柱状图",
            description="各工艺参数对目标指标的一阶主效应指数 S1 与包含交互作用的全阶指数 ST 分解",
            output_path=output_path,
        )
    else:
        fig, ax = plt.subplots(figsize=(9.8, 4.8), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        width = 0.35
        r2_vals = [r["r2"] for r in sobol_records]
        r_vals = [r["r"] for r in sobol_records]

        b1 = ax.bar(
            x - width / 2,
            r2_vals,
            width,
            label="方差贡献率 ($R^2$)",
            color="#2E7D32",
            alpha=0.85,
        )
        b2 = ax.bar(
            x + width / 2,
            r_vals,
            width,
            label="线性相关度 ($r$)",
            color="#1F77B4",
            alpha=0.85,
        )

        ax.set_title(
            "参数敏感度与方差贡献率对比图",
            color="#1E3A8A",
            fontsize=12.5,
            fontweight="bold",
            pad=10,
        )
        ax.set_ylabel("指标数值", fontsize=11, color="#1F2937")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.grid(True, axis="y", linestyle=":", color="#E2E8F0", alpha=0.9)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

        for b in list(b1) + list(b2):
            h = b.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

        plt.tight_layout()
        return _fig_to_result(
            fig,
            title="参数敏感度与方差贡献率对比图",
            description="各参数对目标指标的回归方差贡献率 R^2 与线性相关度 r 对比",
            output_path=output_path,
        )


def generate_optimization_report_figures(
    results_df: Optional[pd.DataFrame] = None,
    jobs_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generates Abdou 2021 publication-grade scientific optimization charts.

    1. fig_opt1_tbr_response: Required TBR vs Design Parameter(s) (Abdou Fig 4-8 style)
    2. fig_opt2_inventory_response: Startup Inventory vs Design Parameter(s) (Abdou Fig 10-13 style)
    3. fig_opt3_tradeoff: Required TBR vs Startup Inventory Pareto Trade-off (Abdou Fig 14-15 style)
    4. fig_opt4_trajectory_verification: Optimal Trajectory with Nadir & Doubling Time Verification
    """
    _init_plot_style()
    figures: List[Dict[str, Any]] = []

    # Merge jobs_df and metrics_df
    df = pd.DataFrame()
    if jobs_df is not None and not jobs_df.empty:
        df = jobs_df.copy()
    if metrics_df is not None and not metrics_df.empty:
        if df.empty:
            df = metrics_df.copy()
        else:
            for c in metrics_df.columns:
                if c not in df.columns:
                    df[c] = metrics_df[c]

    if df.empty:
        return figures

    opt_cfg = None
    if config:
        task_block = (
            config.get("task", {}) if isinstance(config.get("task"), dict) else {}
        )
        opt_cfg = (
            task_block.get("optimization")
            or task_block.get("optimizer")
            or config.get("optimization")
            or config.get("optimizer")
        )
    opt_outer_param = (
        opt_cfg.get("outer", {}).get("parameter") if isinstance(opt_cfg, dict) else None
    )
    opt_inner_metric = (
        opt_cfg.get("inner", {}).get("metric_name")
        if isinstance(opt_cfg, dict)
        else None
    )
    opt_inner_param = (
        opt_cfg.get("inner", {}).get("parameter") if isinstance(opt_cfg, dict) else None
    )

    # Detect optimal TBR column
    tbr_col = None
    if opt_inner_metric and opt_inner_metric in df.columns:
        tbr_col = opt_inner_metric
    elif opt_inner_param and opt_inner_param in df.columns:
        tbr_col = opt_inner_param
    if not tbr_col:
        tbr_col = next(
            (
                c
                for c in df.columns
                if "required_tbr" in c.lower() or c.startswith("Required_")
            ),
            None,
        )

    # Detect optimal startup inventory column
    inv_col = None
    if opt_outer_param and opt_outer_param in df.columns:
        inv_col = opt_outer_param
    if not inv_col:
        inv_col = next(
            (
                c
                for c in df.columns
                if "i_s_0" in c.lower()
                or "startup_inventory" in c.lower()
                or "必需首炉外供氚量" in c
                or "初始装填" in c
            ),
            None,
        )

    ignored_cols = {
        "id",
        "job_id",
        "name",
        "job_name",
        "case_name",
        "job",
        "title",
        "description",
        "comment",
        "status",
        "condition",
    }
    metric_cols = {
        c
        for c in df.columns
        if str(c).startswith("Required_")
        or "_for_" in str(c)
        or "min_value" in str(c).lower()
        or "doubling_time" in str(c).lower()
        or c in (tbr_col, inv_col)
    }
    candidate_params = [
        c
        for c in df.columns
        if str(c).strip().lower() not in ignored_cols
        and c not in metric_cols
        and c not in (inv_col, tbr_col)
        and pd.to_numeric(df[c], errors="coerce").notna().any()
    ]
    varying_params = [
        c
        for c in candidate_params
        if len(pd.to_numeric(df[c], errors="coerce").dropna().unique()) > 1
    ]
    primary_param = (
        varying_params[0]
        if varying_params
        else (candidate_params[0] if candidate_params else None)
    )
    secondary_param = varying_params[1] if len(varying_params) > 1 else None

    # Chart 1: Required TBR Response Curve (Abdou Fig 4-8 Style, Multi-curve for 2D params)
    if tbr_col and primary_param and len(df) > 1:
        try:
            p_df = df.dropna(subset=[tbr_col, primary_param]).copy()
            p_df[primary_param] = pd.to_numeric(p_df[primary_param], errors="coerce")
            p_df[tbr_col] = pd.to_numeric(p_df[tbr_col], errors="coerce")
            p_df = p_df.dropna(subset=[tbr_col, primary_param])
            if len(p_df) > 1:
                p1 = (
                    os.path.join(output_dir, "fig_opt1_tbr_response.png")
                    if output_dir
                    else None
                )
                fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)

                markers = ["o", "s", "^", "D", "v", "p", "*", "X"]
                if secondary_param:
                    # 2D Parameter Group: Draw multiple curves grouped by secondary_param
                    try:
                        p_df[secondary_param] = pd.to_numeric(p_df[secondary_param])
                    except Exception:
                        pass
                    sec_unique = p_df[secondary_param].unique()
                    try:
                        sec_sorted = sorted(sec_unique, key=lambda x: float(x))
                    except Exception:
                        sec_sorted = sorted(sec_unique, key=lambda x: str(x))

                    for idx, s_val in enumerate(sec_sorted):
                        sub = p_df[p_df[secondary_param] == s_val].sort_values(
                            by=primary_param
                        )
                        if not sub.empty:
                            ax.plot(
                                sub[primary_param],
                                sub[tbr_col],
                                color=PALETTE[idx % len(PALETTE)],
                                marker=markers[idx % len(markers)],
                                markersize=6.8,
                                linewidth=2.0,
                                label=f"{secondary_param} = {s_val}",
                            )
                    ax.set_title(
                        f"临界产氚比响应规律 (Required TBR vs {primary_param} 分组: {secondary_param})",
                        fontsize=12,
                        pad=10,
                        loc="left",
                        fontweight="bold",
                    )
                else:
                    p_df = p_df.sort_values(by=primary_param)
                    ax.plot(
                        p_df[primary_param],
                        p_df[tbr_col],
                        color=PALETTE[0],
                        marker="o",
                        markersize=6.8,
                        linewidth=2.2,
                        label=f"临界产氚比 ({tbr_col})",
                    )
                    ax.set_title(
                        f"临界产氚比响应规律 (Required TBR vs {primary_param})",
                        fontsize=12,
                        pad=10,
                        loc="left",
                        fontweight="bold",
                    )

                ax.set_xlabel(f"主设计变量: {primary_param}", fontsize=11)
                ax.set_ylabel(f"临界产氚比 {tbr_col}", fontsize=11)
                ax.grid(True, linestyle=":", alpha=0.5, color="gray")
                ax.legend(loc="best", frameon=True, framealpha=0.9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                figures.append(
                    _fig_to_result(
                        fig,
                        title=f"临界产氚比响应规律曲线 (Required TBR vs {primary_param})",
                        description="展示不同设计工况下满足自持倍增约束所需的临界产氚比变化趋势；若包含二维参数组则绘制多条参数对比曲线（Abdou 2021 理论曲线风格）。",
                        output_path=p1,
                    )
                )
        except Exception as chart1_err:
            logger.warning(
                f"Failed to generate optimization chart 1 (TBR response): {chart1_err}"
            )

    # Chart 2: Startup Inventory Response Curve (Abdou Fig 10-13 Style, Multi-curve for 2D params)
    if inv_col and primary_param and len(df) > 1:
        try:
            p_df = df.dropna(subset=[inv_col, primary_param]).copy()
            p_df[primary_param] = pd.to_numeric(p_df[primary_param], errors="coerce")
            p_df[inv_col] = pd.to_numeric(p_df[inv_col], errors="coerce")
            p_df = p_df.dropna(subset=[inv_col, primary_param])
            if len(p_df) > 1:
                p2 = (
                    os.path.join(output_dir, "fig_opt2_inventory_response.png")
                    if output_dir
                    else None
                )
                fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)

                markers = ["s", "o", "D", "^", "v", "p", "*", "X"]
                if secondary_param:
                    # 2D Parameter Group: Draw multiple curves grouped by secondary_param
                    try:
                        p_df[secondary_param] = pd.to_numeric(p_df[secondary_param])
                    except Exception:
                        pass
                    sec_unique = p_df[secondary_param].unique()
                    try:
                        sec_sorted = sorted(sec_unique, key=lambda x: float(x))
                    except Exception:
                        sec_sorted = sorted(sec_unique, key=lambda x: str(x))

                    for idx, s_val in enumerate(sec_sorted):
                        sub = p_df[p_df[secondary_param] == s_val].sort_values(
                            by=primary_param
                        )
                        if not sub.empty:
                            ax.plot(
                                sub[primary_param],
                                sub[inv_col],
                                color=PALETTE[(idx + 1) % len(PALETTE)],
                                marker=markers[idx % len(markers)],
                                markersize=6.8,
                                linewidth=2.0,
                                label=f"{secondary_param} = {s_val}",
                            )
                    ax.set_title(
                        f"自持初始启动库存响应规律 ({inv_col} vs {primary_param} 分组: {secondary_param})",
                        fontsize=12,
                        pad=10,
                        loc="left",
                        fontweight="bold",
                    )
                else:
                    p_df = p_df.sort_values(by=primary_param)
                    ax.plot(
                        p_df[primary_param],
                        p_df[inv_col],
                        color=PALETTE[1],
                        marker="s",
                        markersize=6.8,
                        linewidth=2.2,
                        label=f"初始装填量 ({inv_col})",
                    )
                    ax.set_title(
                        f"自持初始启动库存响应规律 ({inv_col} vs {primary_param})",
                        fontsize=12,
                        pad=10,
                        loc="left",
                        fontweight="bold",
                    )

                ax.set_xlabel(f"主设计变量: {primary_param}", fontsize=11)
                ax.set_ylabel(f"初始装填库存 {inv_col} [g]", fontsize=11)
                ax.grid(True, linestyle=":", alpha=0.5, color="gray")
                ax.legend(loc="best", frameon=True, framealpha=0.9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                figures.append(
                    _fig_to_result(
                        fig,
                        title=f"最优初始启动库存响应曲线 ({inv_col} vs {primary_param})",
                        description="展示确保系统瞬态最低存量不低于安全储备极限所需的最优初始库存启动量随工况的变化；若包含二维参数组则绘制多条参数对比曲线。",
                        output_path=p2,
                    )
                )
        except Exception as chart2_err:
            logger.warning(
                f"Failed to generate optimization chart 2 (Inventory response): {chart2_err}"
            )

    # Chart 3: Pareto Trade-off Curve (Abdou Fig 14-15 Style)
    if tbr_col and inv_col and len(df) > 1:
        try:
            p_df = df.dropna(subset=[tbr_col, inv_col]).copy()
            p_df[tbr_col] = pd.to_numeric(p_df[tbr_col], errors="coerce")
            p_df[inv_col] = pd.to_numeric(p_df[inv_col], errors="coerce")
            p_df = p_df.dropna(subset=[tbr_col, inv_col])
            p_df = p_df.sort_values(by=inv_col)
            if len(p_df[inv_col].unique()) > 1:
                p3 = (
                    os.path.join(output_dir, "fig_opt3_tradeoff.png")
                    if output_dir
                    else None
                )
                fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
                ax.plot(
                    p_df[inv_col],
                    p_df[tbr_col],
                    color="#8C564B",
                    marker="D",
                    markersize=7.0,
                    linewidth=2.2,
                    label="帕累托自持前沿 (Pareto Frontier)",
                )
                ax.fill_between(
                    p_df[inv_col],
                    p_df[tbr_col],
                    p_df[tbr_col].max() * 1.12,
                    color="#2E7D32",
                    alpha=0.12,
                    label="可行自持设计区 (Feasible Region)",
                )
                ax.set_title(
                    "产氚比与启动库存帕累托权衡前沿 (Required TBR vs Startup Inventory)",
                    fontsize=12,
                    pad=10,
                    loc="left",
                    fontweight="bold",
                )
                ax.set_xlabel(f"初始装填库存 {inv_col} [g]", fontsize=11)
                ax.set_ylabel(f"临界产氚比 {tbr_col}", fontsize=11)
                ax.grid(True, linestyle=":", alpha=0.5, color="gray")
                ax.legend(loc="best", frameon=True, framealpha=0.9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                figures.append(
                    _fig_to_result(
                        fig,
                        title="产氚比与初始装填库存帕累托权衡前沿 (Pareto Trade-off Curve)",
                        description="反映聚变堆核心设计权衡：较高的初始装填量能缓冲瞬态下探亏空，降低自持所需的临界产氚比要求。",
                        output_path=p3,
                    )
                )
        except Exception as chart3_err:
            logger.warning(
                f"Failed to generate optimization chart 3 (Pareto tradeoff): {chart3_err}"
            )

    # Chart 4: Trajectory Verification (Dual Constraints Verification)
    if results_df is not None and not results_df.empty and "time" in results_df.columns:
        target_col = detect_sds_column(results_df)
        if target_col and target_col in results_df.columns:
            p4 = (
                os.path.join(output_dir, "fig_opt4_trajectory_verification.png")
                if output_dir
                else None
            )
            fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=200)

            # 筛选最具代表性/最优解工况轨迹，防止多工况数据直接拼合导致索引重复或连线折叠错乱
            sub_df = results_df
            target_jid = None
            if "job_id" in results_df.columns:
                if jobs_df is not None and not jobs_df.empty:
                    jid_col = (
                        "job_id"
                        if "job_id" in jobs_df.columns
                        else (jobs_df.columns[0] if "id" in jobs_df.columns else None)
                    )
                    # 优先选取临界产氚比或自持指标最优的工况
                    val_col = (
                        tbr_col if (tbr_col and tbr_col in jobs_df.columns) else None
                    )
                    if val_col:
                        try:
                            valid_jobs = jobs_df.dropna(subset=[val_col]).sort_values(
                                by=val_col
                            )
                            if not valid_jobs.empty and jid_col in valid_jobs.columns:
                                target_jid = valid_jobs[jid_col].iloc[0]
                        except Exception:
                            pass
                    if target_jid is None and jid_col and jid_col in jobs_df.columns:
                        target_jid = jobs_df[jid_col].iloc[0]

                if target_jid is None:
                    target_jid = results_df["job_id"].iloc[0]

                mask = results_df["job_id"] == target_jid
                if not mask.any():
                    mask = results_df["job_id"].astype(str) == str(target_jid)
                matched_df = results_df[mask]
                if not matched_df.empty:
                    sub_df = matched_df

            sub_df = sub_df.sort_values(by="time").reset_index(drop=True)

            t_unit, t_factor = resolve_plot_time_scale(
                float(sub_df["time"].iloc[-1]), config
            )
            time_axis = sub_df["time"] / t_factor
            inv_series = pd.to_numeric(sub_df[target_col], errors="coerce").fillna(0.0)

            job_label_suffix = f" (工况 {target_jid})" if target_jid is not None else ""
            ax.plot(
                time_axis,
                inv_series,
                color="#1F77B4",
                linewidth=2.2,
                label=f"存储罐存量动态 ({target_col}){job_label_suffix}",
            )
            i_0 = float(inv_series.iloc[0]) if len(inv_series) > 0 else 0.0
            i_double = 2.0 * i_0
            ax.axhline(
                i_0,
                color="#64748B",
                linestyle=":",
                linewidth=1.2,
                label=f"初始装填量 $I_0$ ({i_0:.1f} g)",
            )
            ax.axhline(
                i_double,
                color="#2E7D32",
                linestyle="--",
                linewidth=1.5,
                label=f"目标自持倍增阈值 $2I_0$ ({i_double:.1f} g)",
            )

            if len(inv_series) > 0:
                min_loc = int(np.argmin(inv_series.values))
                t_nadir = float(time_axis.iloc[min_loc])
                i_nadir = float(inv_series.iloc[min_loc])
                ax.scatter(
                    [t_nadir],
                    [i_nadir],
                    color="#D9534F",
                    s=70,
                    zorder=5,
                    label=f"最低存储谷底 Nadir ({i_nadir:.1f} g)",
                )

            reserve_th = None
            if isinstance(config, dict):
                task_sec = (
                    config.get("task") if isinstance(config.get("task"), dict) else {}
                )
                opt_sec = (
                    task_sec.get("optimization")
                    or task_sec.get("optimizer")
                    or config.get("optimization")
                    or config.get("optimizer")
                )
                if isinstance(opt_sec, dict):
                    outer_sec = opt_sec.get("outer")
                    if isinstance(outer_sec, dict):
                        tm = outer_sec.get("target_metric")
                        if isinstance(tm, dict):
                            outer_th = tm.get("threshold")
                            if outer_th is not None:
                                try:
                                    reserve_th = float(outer_th)
                                except (ValueError, TypeError):
                                    pass

                if reserve_th is None:
                    params_sec = task_sec.get("parameters")
                    if isinstance(params_sec, dict):
                        val = params_sec.get("wds.threshold")
                        if val is not None:
                            try:
                                reserve_th = float(val)
                            except (ValueError, TypeError):
                                pass
                    elif isinstance(params_sec, list):
                        for p in params_sec:
                            if isinstance(p, dict) and "wds.threshold" in p:
                                try:
                                    reserve_th = float(p["wds.threshold"])
                                    break
                                except (ValueError, TypeError):
                                    pass
            if reserve_th is not None:
                ax.axhline(
                    reserve_th,
                    color="#D9534F",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"安全储备警戒线 ({reserve_th:.1f} g)",
                )

            ax.set_title(
                "最优解工况全厂时序动力学与自持双重约束核验",
                fontsize=12,
                pad=10,
                loc="left",
                fontweight="bold",
            )
            ax.set_xlabel(f"运行时间 [{t_unit}]", fontsize=11)
            ax.set_ylabel("存储罐在役氚存量 [g]", fontsize=11)
            ax.grid(True, linestyle=":", alpha=0.5, color="gray")
            ax.legend(loc="best", frameon=True, framealpha=0.9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            figures.append(
                _fig_to_result(
                    fig,
                    title="最优解工况全系统瞬态动力学与双重自持约束核验曲线",
                    description="严格核验二分寻优求解出的系统响应：谷底存量严格保持在安全红线之上，并在设定倍增时间内成功完成增殖自持积累。",
                    output_path=p4,
                )
            )

    return figures


def generate_all_report_figures(
    results_df: Optional[pd.DataFrame] = None,
    jobs_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    task_type: str = "sweep",
    output_dir: Optional[str] = None,
    task_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Task-aware generator providing full publication-grade figures tailored to each task."""
    figures = []
    fig_dir = os.path.join(output_dir, "figures") if output_dir else None

    # Calculate real startup metrics if not provided
    if (
        (metrics_df is None or metrics_df.empty)
        and results_df is not None
        and not results_df.empty
    ):
        metrics_df = extract_startup_inventory_metrics(results_df, jobs_df)

    # 1. Sweep / Simulate Task
    if task_type in ["sweep", "bisection", "cosim"] or task_type == "simulate":
        if results_df is not None and not results_df.empty:
            p1 = (
                os.path.join(fig_dir, "fig1_inventory_trajectories.png")
                if fig_dir
                else None
            )
            f1 = generate_inventory_trajectories_plot(
                results_df, jobs_df, output_path=p1, config=task_config
            )
            if f1:
                figures.append(f1)

            p2 = (
                os.path.join(fig_dir, "fig2_dt_aligned_valley_comparison.png")
                if fig_dir
                else None
            )
            f2 = generate_valley_comparison_plot(
                results_df, jobs_df, output_path=p2, config=task_config
            )
            if f2:
                figures.append(f2)

            # Subsystem Dynamics Dual Plots (Multi-Job aware)
            unique_jobs = []
            if "job_id" in results_df.columns:
                for j in results_df["job_id"]:
                    if j not in unique_jobs:
                        unique_jobs.append(j)

            if not unique_jobs or len(unique_jobs) <= 1:
                # Single run / simulate task
                p3 = (
                    os.path.join(fig_dir, "fig3_subsystem_dynamics.png")
                    if fig_dir
                    else None
                )
                f3 = generate_subsystem_dynamics_plot(
                    results_df, jobs_df, output_path=p3, config=task_config
                )
                if f3:
                    figures.append(f3)
            else:
                # Parameter sweep / multi-case study: render dual plot for each case
                selected_jobs = unique_jobs
                if len(unique_jobs) > 5:
                    selected_jobs = [
                        unique_jobs[0],
                        unique_jobs[1],
                        unique_jobs[len(unique_jobs) // 2],
                        unique_jobs[-2],
                        unique_jobs[-1],
                    ]
                    selected_jobs = list(dict.fromkeys(selected_jobs))

                for idx, jid in enumerate(selected_jobs, 1):
                    slug = (
                        str(jid).replace(" ", "_").replace("/", "_").replace("\\", "_")
                    )
                    p3 = (
                        os.path.join(
                            fig_dir, f"fig3_{idx}_subsystem_dynamics_{slug}.png"
                        )
                        if fig_dir
                        else None
                    )
                    f3 = generate_subsystem_dynamics_plot(
                        results_df,
                        jobs_df,
                        output_path=p3,
                        target_job_id=jid,
                        config=task_config,
                    )
                    if f3:
                        figures.append(f3)

        if metrics_df is not None and not metrics_df.empty and len(metrics_df) > 1:
            p4 = os.path.join(fig_dir, "fig4_metrics_barchart.png") if fig_dir else None
            f4 = generate_metrics_comparison_plot(
                metrics_df, jobs_df, output_path=p4, config=task_config
            )
            if f4:
                figures.append(f4)

    # 3. Sensitivity Task: Sobol Bar Chart
    if task_type in ["sensitivity", "gsa", "sobol"]:
        p_sobol = (
            os.path.join(fig_dir, "fig_sobol_sensitivity.png") if fig_dir else None
        )
        f_sobol = generate_sobol_sensitivity_plot(
            jobs_df=jobs_df,
            metrics_df=metrics_df,
            output_path=p_sobol,
            task_config=task_config,
        )
        if f_sobol:
            figures.append(f_sobol)

    # 4. Optimization Task or Sweep with Optimization Metrics (Abdou 2021 Style)
    is_opt_task = task_type in [
        "optimize",
        "optimization",
        "nested_bisection",
        "dual_basis",
    ]
    has_opt_metrics = False
    has_opt_config = False
    if task_config:
        task_block = (
            task_config.get("task", {})
            if isinstance(task_config.get("task"), dict)
            else {}
        )
        has_opt_config = bool(
            task_block.get("optimization")
            or task_block.get("optimizer")
            or task_config.get("optimization")
            or task_config.get("optimizer")
        )

    merged_data_df = (
        jobs_df.copy() if jobs_df is not None and not jobs_df.empty else pd.DataFrame()
    )
    if metrics_df is not None and not metrics_df.empty:
        if merged_data_df.empty:
            merged_data_df = metrics_df.copy()
        else:
            for c in metrics_df.columns:
                if c not in merged_data_df.columns:
                    merged_data_df[c] = metrics_df[c]

    if not merged_data_df.empty:
        has_opt_metrics = any(
            str(c).startswith("Required_")
            or "_for_" in str(c)
            or c in ("Required_TBR", "sds.I_S_0", "I_S_0")
            for c in merged_data_df.columns
        )

    if is_opt_task or has_opt_metrics or has_opt_config:
        try:
            opt_figs = generate_optimization_report_figures(
                results_df=results_df,
                jobs_df=merged_data_df,
                metrics_df=metrics_df,
                output_dir=fig_dir,
                config=task_config,
            )
            figures.extend(opt_figs)
        except Exception as opt_err:
            logger.warning(
                f"Failed to generate optimization figures: {opt_err}",
                exc_info=True,
            )

    return figures
