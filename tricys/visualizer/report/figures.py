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
import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tricys.analysis.tritium_cycle import (
    detect_sds_column,
    extract_startup_inventory_metrics,
    load_glossary_mapping,
    resolve_subsystem_metadata,
)

# Standard CFEDR Publication Palette
PALETTE = ["#D9534F", "#F0AD4E", "#2E7D32", "#1F77B4", "#9467BD", "#8C564B", "#17BECF"]


def _init_plot_style() -> None:
    """Sets publication-grade matplotlib typography and rendering parameters."""
    plt.rcParams.update(
        {
            "font.sans-serif": [
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

        t = subset["time"].values
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
            ax.axvspan(0, t_dd, color=color, alpha=0.06, label=f"{label} (调试阶段)")

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
    ax.set_xlabel("仿真时间 (hours)", color="#1F2937", fontsize=11)
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
    if t_max > 86400 * 2:
        time_unit = "d"
        time_scale = 86400.0
    elif t_max > 7200:
        time_unit = "h"
        time_scale = 3600.0
    else:
        time_unit = "s"
        time_scale = 1.0

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
        df_dt = subset[subset["time"] >= t_ign].copy()
        if df_dt.empty:
            df_dt = subset.copy()
            t_ign = 0.0

        t_rel = (df_dt["time"].values - t_ign) / time_scale
        y = df_dt[plot_var].values

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

    # Filter post ignition and clip to 600h
    sub_post = subset[subset["time"] >= t_ign].copy()
    if sub_post.empty:
        sub_post = subset.copy()
        t_ign = 0.0

    t_rel = sub_post["time"].values - t_ign
    mask = t_rel <= 600.0 if t_rel[-1] > 600 else np.ones_like(t_rel, dtype=bool)

    t_plot = t_rel[mask]
    sub_plot = sub_post[mask]

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
            for item in vf.split("|"):
                v = item.strip()
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

    # Time unit detection
    t_max = float(t_plot.max()) if len(t_plot) > 0 else 0.0
    time_unit = "s" if t_max <= 7200 else "h"

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
            xytext=(turning_rel + max(t_max * 0.03, 10.0), sds_min + offset_y),
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
            xytext=(turning_rel + max(t_max * 0.03, 10.0), sds_min + 200),
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
                results_df, jobs_df, output_path=p1
            )
            if f1:
                figures.append(f1)

            p2 = (
                os.path.join(fig_dir, "fig2_dt_aligned_valley_comparison.png")
                if fig_dir
                else None
            )
            f2 = generate_valley_comparison_plot(results_df, jobs_df, output_path=p2)
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

    return figures
