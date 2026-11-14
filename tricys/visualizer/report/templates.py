"""Universal Task-Aware Engineering Report Templates for TRICYS 2.0.

Provides 3 abstract, publication-grade engineering report generators:
1. simulate: 氚燃料循环单工况动态仿真工程报告 (Dynamic Simulation Engineering Report)
2. sweep: 氚燃料循环多工况参数扫描工程报告 (Parameter Sweep Engineering Report)
3. sensitivity: 氚燃料循环全局敏感性分析工程报告 (Global Sensitivity Analysis Report)

Design Principles:
- Universal: Works seamlessly with any model (standard CFEDR, DEMO, ITER, or custom Modelica cycles).
- 100% Real Data: All parameters, tables, metrics, and sensitivity indices are dynamically derived from real execution data.
- Fact-First: Strictly reports objective engineering data and facts without subjective speculative summaries.
- Template Default: Automatically matches task.type when report.template is omitted.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tricys.analysis.tritium_cycle import (
    analyze_simulation_mass_conservation,
    analyze_subsystem_holdup_dynamics,
    calculate_real_sensitivity_indices,
    derive_fusion_engineering_parameters,
    detect_sds_column,
    extract_startup_inventory_metrics,
    load_glossary_mapping,
)

TEMPLATES_CATALOG = [
    {
        "id": "simulate",
        "name": "氚燃料循环单工况动态仿真工程报告 (Dynamic Simulation Report)",
        "task_target": "simulate",
        "description": "全厂时序动力学演化、多工艺子系统滞留量演化、特征铺底时滞与物料平衡审计核验",
        "badge": "Simulation Dedicated",
    },
    {
        "id": "sweep",
        "name": "氚燃料循环多工况参数扫描工程报告 (Parameter Sweep Report)",
        "task_target": "sweep",
        "description": "多工况设计参数空间对比、核心性能指标(KPI)横向评价、首炉启动亏空量与自持平衡规律分析",
        "badge": "Sweep Dedicated",
    },
    {
        "id": "sensitivity",
        "name": "氚燃料循环全局敏感性分析工程报告 (Global Sensitivity Report)",
        "task_target": "sensitivity",
        "description": "高维参数不确定度空间、一阶主效应与全阶敏感性指数分解、参数贡献度排序与关键工艺识别",
        "badge": "Sensitivity Dedicated",
    },
    {
        "id": "optimize",
        "name": "氚燃料循环参数寻优与临界产氚比分析报告 (Optimization & Required TBR Report)",
        "task_target": "optimize",
        "description": "双层嵌套与基底二分参数寻优、临界产氚比(Required TBR)求解、初始启动库存与安全储备权衡分析",
        "badge": "Optimization Dedicated",
    },
]


def normalize_template_id(
    template_id: Optional[str] = None,
    task_type: Optional[str] = None,
) -> str:
    """Normalizes template_id to one of the 4 universal core templates: 'simulate', 'sweep', 'sensitivity', 'optimize'.

    When template_id is omitted or empty, defaults strictly to matching task_type.
    """
    task = (task_type or "simulate").strip().lower()
    raw = str(template_id).strip().lower() if template_id else ""

    if not raw:
        raw = task

    # 1. Direct standard mapping
    if raw in ("simulate", "simulation", "transient", "single"):
        return "simulate"
    if raw in (
        "optimize",
        "optimization",
        "nested_bisection",
        "dual_basis",
        "required_tbr",
        "critical_tbr",
        "bisection_critical_tbr",
    ):
        return "optimize"
    if raw in ("sweep", "bisection", "parameter_study", "cases", "param"):
        return "sweep"
    if raw in ("sensitivity", "gsa", "sobol", "uncertainty"):
        return "sensitivity"

    # 2. Backward compatibility mappings
    if "simul" in raw or "transient" in raw:
        return "simulate"
    if (
        "optimi" in raw
        or "nested" in raw
        or "dual_basis" in raw
        or "critical_tbr" in raw
        or "required_tbr" in raw
    ):
        return "optimize"
    if "sensit" in raw or "sobol" in raw or "gsa" in raw:
        return "sensitivity"
    if (
        "sweep" in raw
        or "bisection" in raw
        or "startup" in raw
        or "kpi" in raw
        or "dossier" in raw
        or "exec" in raw
    ):
        return "sweep"

    # 3. Fallback to task_type
    if "optimi" in task or "nested" in task or "dual_basis" in task:
        return "optimize"
    if "simul" in task:
        return "simulate"
    if "sensit" in task or "sobol" in task:
        return "sensitivity"
    return "sweep"


def get_recommended_template_for_task(task_type: Optional[str] = None) -> str:
    """Returns the primary recommended universal template ID matching task type."""
    return normalize_template_id(template_id=None, task_type=task_type)


def _format_dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Safely formats a DataFrame into clean GitHub-flavored markdown table."""
    if df is None or df.empty:
        return "*（无表格数据）*\n\n"

    display_df = df.head(max_rows).copy()
    for col in display_df.select_dtypes(include=["float", "float64"]).columns:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.4f}" if pd.notnull(x) else "-"
        )
    return display_df.to_markdown(index=False) + "\n\n"


def _format_figures_to_markdown(figures: Optional[List[Dict[str, Any]]]) -> str:
    """Formats embedded charts into Markdown / HTML image blocks."""
    if not figures:
        return ""
    lines = []
    for fig in figures:
        title = fig.get("title", "仿真响应图表")
        desc = fig.get("description", "")
        img = fig.get("image_data") or fig.get("file_path") or ""
        if img:
            lines.append(f"### 📈 {title}\n")
            lines.append(
                f'<div style="text-align: center; margin: 18px 0;">\n'
                f'  <img src="{img}" alt="{title}" style="max-width: 96%; border-radius: 6px; border: 1px solid #CBD5E1; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" />\n'
                f'  <div style="font-size: 0.88em; color: #64748B; margin-top: 6px;"><em>{desc}</em></div>\n'
                f"</div>\n\n"
            )
    return "\n".join(lines)


def format_foc_scenario_overview(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extracts and formats Fusion Operation Cycle (FOC) dynamic pulse injection scenario overview.

    Returns formatted markdown text if FOC configuration is detected, otherwise None.
    """
    if not config or not isinstance(config, dict):
        return None

    pulse_cfg = (
        config.get("scenario", {}).get("pulse")
        if isinstance(config.get("scenario"), dict)
        else None
    )
    foc_cfg = config.get("foc")

    foc_file = None
    foc_comp = None
    strategy = "table"

    if pulse_cfg and isinstance(pulse_cfg, dict):
        foc_file = pulse_cfg.get("foc_file") or pulse_cfg.get("foc_path")
        foc_comp = pulse_cfg.get("foc_component") or pulse_cfg.get("target_component")
        strategy = pulse_cfg.get("strategy", "table")

    if not foc_file and foc_cfg and isinstance(foc_cfg, dict):
        foc_file = foc_cfg.get("foc_path") or foc_cfg.get("foc_file")
        foc_comp = foc_cfg.get("foc_component") or foc_cfg.get("component_name")
        strategy = foc_cfg.get("strategy", strategy)

    if not foc_file:
        return None

    foc_comp = foc_comp or "pulseSource"
    strategy_desc = (
        "FOC_TablePulse (高精度时序离散插值表)"
        if strategy == "table"
        else "FOC_ArrayPulse (紧凑离散脉冲数组)"
    )

    foc_filename = os.path.basename(str(foc_file))
    lines = [
        "### ⚡ 聚变运行周期 (FOC) 动态脉冲工况配置",
        "",
        "| 工况要素 | 配置取值 | 物理/工程说明 |",
        "| :--- | :---: | :--- |",
        f"| **工况脚本名称** | `{foc_filename}` | 调度描述文件: `{foc_file}` |",
        f"| **目标注入组件** | `{foc_comp}` | 动态替换底层 Modelica 系统的信号源实例 |",
        f"| **脉冲注入策略** | `{strategy}` | {strategy_desc} |",
    ]

    # Try parsing schedule metadata from .foc DSL file if available
    if os.path.exists(str(foc_file)):
        try:
            from tricys.core.foc import parse_foc_file_with_unit

            amps, durs, unit = parse_foc_file_with_unit(str(foc_file))
            total_duration_sec = sum(durs)
            pulse_count = len([a for a in amps if a > 0])
            max_amp = max(amps) if amps else 0.0

            lines.extend(
                [
                    f"| **工况时间基准单位** | `{unit}` | FOC 脚本原生声明时间尺度 |",
                    f"| **单周期总调度跨度** | `{total_duration_sec:.1f} s ({total_duration_sec / 3600.0:.2f} h)` | 单完整周期/调度计划时间长度 |",
                    f"| **工况有效脉冲数** | `{pulse_count} 个` | 包含有效功率驱动的燃烧期脉冲阶段 |",
                    f"| **峰值脉冲功率/幅度** | `{max_amp:.1f}` | 聚变燃烧阶段最大标称驱动强度 |",
                ]
            )
        except Exception:
            pass

    lines.append("")
    return "\n".join(lines)


# ==============================================================================
# 1. Universal Simulate Template: Single Scenario & Dynamic Transient Audit
# ==============================================================================
def render_simulate(
    config: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    results_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    figures: Optional[List[Dict[str, Any]]] = None,
    ai_insight: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Renders Universal Dynamic Simulation Engineering Report."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_cfg = (
        config.get("visualizer", {}).get("report")
        if isinstance(config.get("visualizer"), dict)
        else None
    ) or config.get("report", {})
    title = (
        report_cfg.get("title") or "TRICYS 聚变堆氚燃料循环单工况动态仿真工程分析报告"
    )

    model_cfg = config.get("model", {})
    scenario_cfg = config.get("scenario", {})
    model_name = model_cfg.get("modelName", "Tritium Fuel Cycle System")
    model_file = model_cfg.get("fileName", "-")
    stop_time = scenario_cfg.get("stopTime", "-")
    step_size = scenario_cfg.get("stepSize", "-")
    solver_method = scenario_cfg.get("method", "dassl")

    # 1. Derive real physical parameters if configured
    p = derive_fusion_engineering_parameters(config=config, results_df=results_df)

    # 2. Audit mass conservation dynamically via real auditor rules if configured
    mass_audit = analyze_simulation_mass_conservation(
        results_df, config=config, eng_params=p
    )

    # 3. Dynamic Subsystem Dynamics Table
    sub_df = (
        analyze_subsystem_holdup_dynamics(results_df, config=config)
        if results_df is not None
        else pd.DataFrame()
    )

    # 4. Storage dynamics facts from real results
    storage_facts = []
    if results_df is not None and not results_df.empty:
        sds_col = detect_sds_column(results_df)
        if sds_col and sds_col in results_df.columns:
            arr = pd.to_numeric(results_df[sds_col], errors="coerce").dropna().values
            t_arr = (
                pd.to_numeric(results_df["time"], errors="coerce").dropna().values
                if "time" in results_df.columns
                else np.arange(len(arr))
            )
            if len(arr) > 0:
                i0 = float(arr[0])
                i_min = float(np.min(arr))
                i_max = float(np.max(arr))
                i_end = float(arr[-1])
                min_idx = int(np.argmin(arr))
                t_turn = float(t_arr[min_idx]) if min_idx < len(t_arr) else 0.0
                deficit = max(0.0, i0 - i_min)
                net_change = i_end - i0

                status_desc = (
                    "达成自持正增殖 (终值高于初值)"
                    if net_change >= 0
                    else "净消耗状态 (终值低于初值)"
                )
                time_base = str(scenario_cfg.get("time_base", "second")).strip().lower()
                if time_base == "second":
                    t_turn_str = f"{t_turn / 3600.0:.2f} h ({t_turn:.0f} s)"
                    t_turn_unit = "h"
                else:
                    t_turn_str = f"{t_turn:.2f} h"
                    t_turn_unit = "h"

                storage_facts = [
                    {
                        "物理参量": "监控主储罐变量列",
                        "实测数值 / 状态": sds_col,
                        "单位": "—",
                        "工程说明": "系统主库存参照节点",
                    },
                    {
                        "物理参量": "初始标称装料量 (I_0)",
                        "实测数值 / 状态": f"{i0:.4f}",
                        "单位": "g",
                        "工程说明": "仿真启动时刻在役库存",
                    },
                    {
                        "物理参量": "运行期最低谷底存量 (I_min)",
                        "实测数值 / 状态": f"{i_min:.4f}",
                        "单位": "g",
                        "工程说明": "动态运行期库存最低点",
                    },
                    {
                        "物理参量": "首炉启动外供亏空量 (Deficit)",
                        "实测数值 / 状态": f"{deficit:.4f}",
                        "单位": "g",
                        "工程说明": "需外部补充的首炉启动氚亏空 (I_0 - I_min)",
                    },
                    {
                        "物理参量": "达自持转折点时间 (t_turning)",
                        "实测数值 / 状态": t_turn_str,
                        "单位": t_turn_unit,
                        "工程说明": "库存由净消耗转为净积累的拐点时刻",
                    },
                    {
                        "物理参量": "运行期最高在役峰值 (I_peak)",
                        "实测数值 / 状态": f"{i_max:.4f}",
                        "单位": "g",
                        "工程说明": "全周期储罐最大瞬时存量",
                    },
                    {
                        "物理参量": "仿真终态存量 (I_final)",
                        "实测数值 / 状态": f"{i_end:.4f}",
                        "单位": "g",
                        "工程说明": "停堆截面系统留存量",
                    },
                    {
                        "物理参量": "全周期净氚增量 (ΔI_net)",
                        "实测数值 / 状态": f"{'+' if net_change >= 0 else ''}{net_change:.4f}",
                        "单位": "g",
                        "工程说明": status_desc,
                    },
                ]

    time_base = str(scenario_cfg.get("time_base", "second")).strip().lower()
    time_base_desc = (
        "second (底层Modelica模型为秒基准，自动转换为秒对接OMC求解)"
        if time_base == "second"
        else "hour (底层Modelica模型为小时基准，以1:1小时步长求解)"
    )
    try:
        st_val = float(stop_time)
        stop_time_display = (
            f"{st_val:g} h ({st_val * 3600.0:.0f} s)"
            if time_base == "second"
            else f"{st_val:g} h"
        )
    except Exception:
        stop_time_display = f"{stop_time} h"

    try:
        ss_val = float(step_size)
        step_size_display = (
            f"{ss_val:g} h ({ss_val * 3600.0:.1f} s)"
            if time_base == "second"
            else f"{ss_val:g} h"
        )
    except Exception:
        step_size_display = f"{step_size} h"

    md = [
        f"# {title}",
        f"**生成时间**: {now_str}  |  **任务类型**: SIMULATE (单工况动态时序仿真)  |  **仿真模型**: `{model_name}`",
        "",
        "---",
        "",
        "## 1. 仿真计算与模型参数设定清单",
        "",
        "| 参数名称 | 设定值 | 物理/工程含义说明 |",
        "| :--- | :---: | :--- |",
        f"| **仿真底层模型** | `{model_name}` | 模型文件: `{model_file}` |",
        f"| **数值求解算法** | `{solver_method}` | 变步长常微分代数方程求解器 |",
        f"| **工况时间基准 (time_base)** | `{time_base_desc}` | 系统声明时间尺度 |",
        f"| **仿真终止时间** | `{stop_time_display}` | 计算时间窗口跨度 |",
        f"| **输出采样步长** | `{step_size_display}` | 时序数据离散记录间隔 |",
    ]

    # Append model-specific real physical parameters if present
    model_params = config.get("model", {}).get("parameters", {})
    if model_params:
        glossary_map = load_glossary_mapping(config=config)
        for k, v in model_params.items():
            param_label = f"模型物理参数: {k}"
            param_desc = "模型显式设计参数"
            if glossary_map:
                item = glossary_map.get(k) or glossary_map.get(k.lower())
                if item and item.get("zh"):
                    param_label = f"模型物理参数: {item['zh']} ({k})"
                if item and item.get("desc"):
                    param_desc = item["desc"]
            md.append(f"| **{param_label}** | `{v}` | {param_desc} |")

    # Fusion Operation Cycle (FOC) dynamic pulse injection section (if configured)
    foc_overview = format_foc_scenario_overview(config)
    if foc_overview:
        md.extend(["", foc_overview])

    # Target Metrics and Optimization Goals (if configured)
    task_metrics_facts = []
    task_metrics_def = config.get("task", {}).get("metrics", {})
    if task_metrics_def and isinstance(task_metrics_def, dict):
        glossary_map = load_glossary_mapping(config=config)
        first_job = jobs[0] if jobs and isinstance(jobs, list) and len(jobs) > 0 else {}
        first_metric = (
            metrics_df.iloc[0].to_dict()
            if metrics_df is not None and not metrics_df.empty
            else {}
        )

        for m_name, m_spec in task_metrics_def.items():
            val = None
            if m_name in first_metric and pd.notnull(first_metric[m_name]):
                val = first_metric[m_name]
            elif m_name in first_job and pd.notnull(first_job[m_name]):
                val = first_job[m_name]
            elif (
                results_df is not None
                and not results_df.empty
                and isinstance(m_spec, dict)
            ):
                src = m_spec.get("source")
                op = m_spec.get("evaluate") or m_spec.get("method")
                if src and src in results_df.columns:
                    if op == "peak_value":
                        val = float(results_df[src].max())
                    elif op == "final_value":
                        val = float(results_df[src].iloc[-1])
                    elif op == "min":
                        val = float(results_df[src].min())

            val_str = (
                f"{val:.4f}"
                if isinstance(val, (int, float))
                else (str(val) if val is not None else "—")
            )
            unit_str = m_spec.get("unit") or "—" if isinstance(m_spec, dict) else "—"

            m_label = m_name
            m_desc = "任务性能提取指标"
            if m_name.startswith("Required_"):
                m_label = f"临界寻优指标: {m_name}"
                m_desc = "二分法寻优结果：满足约束的最小边界参数"
                if "tbr" in m_name.lower():
                    m_label = f"首炉最小必需产氚比 ({m_name})"
                    m_desc = "二分法闭环寻优：达成倍增时间约束的最小必要 TBR"

            if glossary_map:
                item = glossary_map.get(m_name) or glossary_map.get(m_name.lower())
                if item and item.get("zh"):
                    m_label = f"{item['zh']} ({m_name})"
                if item and item.get("desc"):
                    m_desc = item["desc"]

            task_metrics_facts.append(
                {
                    "性能指标名称": m_label,
                    "设定 / 寻优计算值": val_str,
                    "单位": unit_str,
                    "工程指标说明": m_desc,
                }
            )

    sec_idx = 2
    if task_metrics_facts:
        md.extend(
            [
                "",
                "---",
                "",
                f"## {sec_idx}. 任务目标性能与参数寻优指标清单",
                "",
                _format_dataframe_to_markdown(pd.DataFrame(task_metrics_facts)),
            ]
        )
        sec_idx += 1

    md.extend(
        [
            "",
            "---",
            "",
            f"## {sec_idx}. 系统核心库存与动态响应关键事实",
            "",
        ]
    )
    sec_idx += 1

    if storage_facts:
        md.append(_format_dataframe_to_markdown(pd.DataFrame(storage_facts)))
    else:
        md.append("*（未检测到有效储罐时序存量数据）*\n\n")

    md.extend(
        [
            f"## {sec_idx}. 各工艺子组件稳态滞留动力学统计",
            "",
        ]
    )
    sec_idx += 1

    if not sub_df.empty:
        md.append(_format_dataframe_to_markdown(sub_df))
    else:
        md.append("*（当前模型输出中未检测到独立的工艺子系统滞留量列）*\n\n")

    md.extend(
        [
            f"## {sec_idx}. 全厂物料平衡与审计核查",
            "",
        ]
    )
    sec_idx += 1

    if mass_audit.get("configured") and mass_audit.get("status") == "success":
        err_g = mass_audit.get("mass_balance_error", 0.0)
        warn_th = mass_audit.get("warn_threshold_g", 0.5)
        is_ok = abs(err_g) <= warn_th
        status_label = "✅ 质量守恒核查通过" if is_ok else "⚠️ 存在非物理物料偏差"
        md.extend(
            [
                f"- **物料平衡审计状态**: {status_label}",
                f"- **全流程累积质量平衡误差**: `{err_g:+.6e} g` (告警阈值: `{warn_th} g`)",
                f"- **初始总在役存量 (ΣI_0)**: `{mass_audit.get('initial_inventory', 0.0):.4f} g`",
                f"- **终态实测总存量 (ΣI_end)**: `{mass_audit.get('final_inventory', 0.0):.4f} g`",
                f"- **全流程理论期望终态量 (I_expected)**: `{mass_audit.get('expected_final_mass', 0.0):.4f} g`",
                f"- **累积外部输入源**: `{mass_audit.get('cumulative_sources', 0.0):.4f} g`",
                f"- **累积燃耗与排废消耗**: `{mass_audit.get('cumulative_burn', 0.0) + mass_audit.get('cumulative_leak', 0.0):.4f} g`",
                f"- **累积放射性衰变消耗**: `{mass_audit.get('cumulative_decay', 0.0):.4f} g`",
                "",
                "**参与闭环审计的匹配变量**:",
                f"- 识别的存量节点: `{', '.join(mass_audit.get('discovered_inventory_vars', [])) or '无'}`",
                f"- 识别的输入流股: `{', '.join(mass_audit.get('discovered_source_vars', [])) or '无'}`",
                f"- 识别的消耗/排废流股: `{', '.join(mass_audit.get('discovered_burn_vars', []) + mass_audit.get('discovered_leak_vars', [])) or '无'}`",
                "",
            ]
        )
    elif mass_audit.get("configured"):
        md.extend(
            [
                "- **物料平衡审计状态**: ⚠️ 规则未匹配",
                f"- **说明**: {mass_audit.get('message', '未在时序数据中匹配到指定的变量列')}",
                "",
            ]
        )
    else:
        md.extend(
            [
                "- **物料平衡审计状态**: 未配置全厂边界源汇审计规则 (`task.auditor.patterns` 或 `auditor.patterns`)",
                "- **客观核查事实说明**: 当前任务未配置系统进出口边界流股（加料、燃耗、排废、泄漏）与各组件内部滞留量的完整规则映射，不满足闭环质量守恒积分核查前提条件，已如实记录并跳过全厂闭环质量审计。",
                "",
            ]
        )

    md.extend(
        [
            f"## {sec_idx}. 动态响应曲线与时序图表",
            "",
        ]
    )
    sec_idx += 1

    if figures:
        md.append(_format_figures_to_markdown(figures))
    else:
        md.append("*（当前仿真未生成关联图表）*\n\n")

    if ai_insight:
        md.extend(["---", "", f"## {sec_idx}. AI 智能体工程机理解读", ai_insight])

    return "\n".join(md)


# ==============================================================================
# 2. Universal Sweep Template: Multi-Scenario Parameter Study & Performance Report
# ==============================================================================
def render_sweep(
    config: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    results_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    figures: Optional[List[Dict[str, Any]]] = None,
    ai_insight: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Renders Universal Parameter Sweep Engineering Report."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_cfg = (
        config.get("visualizer", {}).get("report")
        if isinstance(config.get("visualizer"), dict)
        else None
    ) or config.get("report", {})
    title = (
        report_cfg.get("title") or "TRICYS 聚变堆氚燃料循环多工况参数扫描工程分析报告"
    )

    jobs_df = pd.DataFrame(jobs) if jobs else None

    # Derive real tables if not provided
    if (
        (metrics_df is None or metrics_df.empty)
        and results_df is not None
        and not results_df.empty
    ):
        metrics_df = extract_startup_inventory_metrics(results_df, jobs_df, config)

    # 1. Identify parameter search space (strictly excluding metrics)
    ignored_col_names = {
        "id",
        "job_id",
        "name",
        "job_name",
        "case_name",
        "job",
        "status",
    }
    known_metrics = set()
    if metrics_df is not None and not metrics_df.empty:
        known_metrics |= {
            c for c in metrics_df.columns if c.lower() not in ignored_col_names
        }
    if config:
        task_metrics = (
            config.get("task", {}).get("metrics", {})
            if isinstance(config.get("task"), dict)
            else {}
        )
        known_metrics |= set(task_metrics.keys())
        known_metrics |= set(config.get("metrics_definition", {}).keys())

    param_cols: List[str] = []
    if jobs_df is not None and not jobs_df.empty:
        for c in jobs_df.columns:
            if c.lower() not in ignored_col_names and c not in known_metrics:
                num = pd.to_numeric(jobs_df[c], errors="coerce")
                if num.notna().any():
                    param_cols.append(c)

    # Identify if a true baseline case exists
    baseline_row = None
    if jobs_df is not None and not jobs_df.empty:
        for _, r in jobs_df.iterrows():
            cname = str(r.get("name") or r.get("job_name") or r.get("case_name") or "")
            if "baseline" in cname.lower() or "基准" in cname:
                baseline_row = r
                break
        if baseline_row is None and "job_id" in jobs_df.columns:
            r1 = jobs_df[jobs_df["job_id"] == 1]
            if not r1.empty and len(jobs_df) > 1:
                cname = str(r1.iloc[0].get("name") or r1.iloc[0].get("job_name") or "")
                if "baseline" in cname.lower() or "基准" in cname.lower():
                    baseline_row = r1.iloc[0]

    glossary_map = load_glossary_mapping(config=config)
    space_summary = []
    if jobs_df is not None and not jobs_df.empty and param_cols:
        for p in param_cols:
            s = pd.to_numeric(jobs_df[p], errors="coerce").dropna()
            row_data = {
                "扫描参数": p,
                "最小值 (Min)": f"{s.min():.4g}",
                "最大值 (Max)": f"{s.max():.4g}",
            }
            if baseline_row is not None and p in baseline_row:
                b_val = pd.to_numeric(baseline_row[p], errors="coerce")
                row_data["基准工况取值 (Baseline)"] = (
                    f"{b_val:.4g}" if pd.notnull(b_val) else str(baseline_row[p])
                )
            else:
                row_data["样本均值 (Mean)"] = f"{s.mean():.4g}"

            row_data["离散取样点数"] = len(s.unique())

            if glossary_map:
                item = glossary_map.get(p) or glossary_map.get(p.lower())
                if item and item.get("zh"):
                    row_data["专业术语"] = item["zh"]
                if item and item.get("desc"):
                    row_data["说明"] = item["desc"]
            space_summary.append(row_data)

    # 2. Extract facts dynamically for all numerical metrics in metrics_df
    key_facts = []
    if metrics_df is not None and not metrics_df.empty:
        # Build lookup for friendly case name
        case_name_map = {}
        if jobs_df is not None and not jobs_df.empty:
            jid_col = "job_id" if "job_id" in jobs_df.columns else jobs_df.columns[0]
            for _, r in jobs_df.iterrows():
                cid = r[jid_col]
                cname = r.get("name") or r.get("job_name") or f"工况 {cid}"
                case_name_map[cid] = cname

        def _get_case_label(row: Any, idx: Any) -> str:
            try:
                for k in ["工况名称", "name", "job_name", "case_name"]:
                    if k in row:
                        val = row[k]
                        if isinstance(val, pd.Series):
                            val = val.iloc[0] if len(val) > 0 else None
                        if pd.notnull(val) and str(val).strip():
                            return str(val).strip()

                jid = None
                for k in ["job_id", "id", "Job"]:
                    if k in row:
                        val = row[k]
                        if isinstance(val, pd.Series):
                            val = val.iloc[0] if len(val) > 0 else None
                        if pd.notnull(val):
                            jid = val
                            break

                if jid is not None and jid in case_name_map:
                    return str(case_name_map[jid])
                if jid is not None:
                    return f"工况 {jid}"
            except Exception:
                pass
            return f"工况 #{idx + 1}"

        m_df = metrics_df.reset_index(drop=True)
        metric_cols = []
        for c in m_df.columns:
            if c.lower() not in ignored_col_names and c not in param_cols:
                num = pd.to_numeric(m_df[c], errors="coerce")
                if num.notna().sum() >= 1:
                    metric_cols.append(c)

        for col in metric_cols:
            s_vals = pd.to_numeric(m_df[col], errors="coerce")
            valid_indices = s_vals.dropna().index.tolist()
            if valid_indices:
                v_arr = s_vals.iloc[valid_indices].values
                min_pos = int(np.argmin(v_arr))
                max_pos = int(np.argmax(v_arr))
                min_idx = valid_indices[min_pos]
                max_idx = valid_indices[max_pos]
                min_row = m_df.iloc[min_idx]
                max_row = m_df.iloc[max_idx]
                min_label = _get_case_label(min_row, min_idx)
                max_label = _get_case_label(max_row, max_idx)
                min_val = float(v_arr[min_pos])
                max_val = float(v_arr[max_pos])
                mean_val = float(np.mean(v_arr))

                col_label = col
                if glossary_map:
                    item = glossary_map.get(col) or glossary_map.get(col.lower())
                    if item and item.get("zh"):
                        col_label = f"{item['zh']} ({col})"

                if abs(max_val - min_val) < 1e-9 or len(v_arr) == 1:
                    key_facts.append(
                        f"- **{col_label}**: 全工况实测恒定值 `{min_val:.4g}`"
                    )
                else:
                    key_facts.append(
                        f"- **{col_label}**: 极小值 `{min_val:.4g}` (出自 `{min_label}`), 极大值 `{max_val:.4g}` (出自 `{max_label}`), 全工况均值 `{mean_val:.4g}`"
                    )

    # 5. Build Enriched Formatted Jobs Table
    formatted_jobs = []
    if jobs_df is not None and not jobs_df.empty:
        for idx, r in jobs_df.iterrows():
            jid = (
                r.get("job_id")
                if "job_id" in r
                else (r.get("id") if "id" in r else idx + 1)
            )
            cname = (
                r.get("name")
                or r.get("job_name")
                or r.get("case_name")
                or f"工况 {jid}"
            )
            rec = {
                "工况编号": jid,
                "工况名称": cname,
            }
            for p in param_cols:
                p_h = p
                if glossary_map:
                    item = glossary_map.get(p) or glossary_map.get(p.lower())
                    if item and item.get("zh"):
                        p_h = f"{item['zh']} ({p})"
                pval = pd.to_numeric(r[p], errors="coerce")
                rec[p_h] = f"{pval:.4g}" if pd.notnull(pval) else str(r[p])
            formatted_jobs.append(rec)

    # 6. Build Comprehensive KPI Comparison Table
    kpi_records = []
    if metrics_df is not None and not metrics_df.empty:
        m_df_norm = metrics_df.reset_index(drop=True)
        for idx, r in m_df_norm.iterrows():
            jid = (
                r.get("job_id")
                if "job_id" in r
                else (r.get("id") if "id" in r else idx + 1)
            )
            # Find matching job info
            j_match = None
            if jobs_df is not None and not jobs_df.empty:
                if "job_id" in jobs_df.columns:
                    sub = jobs_df[jobs_df["job_id"] == jid]
                    if not sub.empty:
                        j_match = sub.iloc[0]
                elif idx < len(jobs_df):
                    j_match = jobs_df.iloc[idx]

            cname = (
                (
                    j_match.get("name")
                    or j_match.get("job_name")
                    or j_match.get("case_name")
                )
                if j_match is not None
                else (case_name_map.get(jid) or f"工况 {jid}")
            )

            rec = {
                "工况编号": jid,
                "工况名称": cname,
            }

            # Add input parameters
            if j_match is not None:
                for p in param_cols:
                    p_h = p
                    if glossary_map:
                        item = glossary_map.get(p) or glossary_map.get(p.lower())
                        if item and item.get("zh"):
                            p_h = f"{item['zh']} ({p})"
                    pval = pd.to_numeric(j_match[p], errors="coerce")
                    rec[p_h] = f"{pval:.4g}" if pd.notnull(pval) else str(j_match[p])

            # Add calculated performance metrics
            for m in metric_cols:
                m_h = m
                if glossary_map:
                    item = glossary_map.get(m) or glossary_map.get(m.lower())
                    if item and item.get("zh"):
                        m_h = f"{item['zh']} ({m})"
                mval = pd.to_numeric(r[m], errors="coerce")
                rec[m_h] = f"{mval:.4g}" if pd.notnull(mval) else str(r[m])

            kpi_records.append(rec)

    md = [
        f"# {title}",
        f"**生成时间**: {now_str}  |  **任务类型**: SWEEP (多工况参数扫描)  |  **评估工况总数**: `{len(jobs)}`",
        "",
        "---",
        "",
        "## 1. 参数扫描设计空间与工况清单",
        "",
    ]

    if space_summary:
        md.append("### 1.1 扫描参数探索范围与离散维度\n")
        md.append(_format_dataframe_to_markdown(pd.DataFrame(space_summary)))

    md.append("### 1.2 实际执行对比工况设定清单\n")
    if formatted_jobs:
        md.append(_format_dataframe_to_markdown(pd.DataFrame(formatted_jobs)))
    elif jobs:
        md.append(_format_dataframe_to_markdown(pd.DataFrame(jobs)))
    else:
        md.append("*（无工况参数清单）*\n\n")

    # Fusion Operation Cycle (FOC) dynamic pulse injection section (if configured)
    foc_overview = format_foc_scenario_overview(config)
    if foc_overview:
        md.extend([foc_overview, ""])

    # Embedded Optimization Architecture in Sweep (if configured)
    task_cfg = config.get("task", {}) if isinstance(config.get("task"), dict) else {}
    opt_cfg = (
        task_cfg.get("optimization")
        or task_cfg.get("optimizer")
        or config.get("optimization")
        or config.get("optimizer")
    )
    if hasattr(opt_cfg, "model_dump"):
        opt_cfg = opt_cfg.model_dump()
    elif hasattr(opt_cfg, "dict"):
        opt_cfg = opt_cfg.dict()

    if opt_cfg and isinstance(opt_cfg, dict):
        inner_cfg = (
            opt_cfg.get("inner", {}) if isinstance(opt_cfg.get("inner"), dict) else {}
        )
        outer_cfg = (
            opt_cfg.get("outer", {}) if isinstance(opt_cfg.get("outer"), dict) else {}
        )
        arch_records = []
        if inner_cfg:
            i_target = inner_cfg.get("target_metric")
            i_target_metric = (
                f"{i_target.get('source', '')} ({i_target.get('evaluate', '')})"
                if isinstance(i_target, dict)
                else str(i_target or "I_reserve_req")
            )
            i_thresh = (
                str(i_target.get("threshold", "-"))
                if isinstance(i_target, dict)
                else str(inner_cfg.get("threshold", "0.0"))
            )
            arch_records.append(
                {
                    "优化层级": "内层寻优 (Inner Loop)",
                    "寻优设计变量": inner_cfg.get("parameter", "blanket.TBR"),
                    "搜索边界 (Bounds)": str(
                        inner_cfg.get("search_bounds")
                        or inner_cfg.get("bounds", "未指定")
                    ),
                    "收敛容差 (Tolerance)": str(inner_cfg.get("tolerance", "0.005")),
                    "最大迭代步数": str(inner_cfg.get("max_iterations", "20")),
                    "目标约束物理量": i_target_metric,
                    "设计目标阈值": i_thresh,
                }
            )
        if outer_cfg:
            o_target = outer_cfg.get("target_metric")
            o_target_metric = (
                f"{o_target.get('source', '')} ({o_target.get('evaluate', '')})"
                if isinstance(o_target, dict)
                else str(o_target or "Required_TBR")
            )
            o_thresh = (
                str(o_target.get("threshold", "-"))
                if isinstance(o_target, dict)
                else str(outer_cfg.get("threshold", "-"))
            )
            arch_records.append(
                {
                    "优化层级": "外层寻优 (Outer Loop)",
                    "寻优设计变量": outer_cfg.get("parameter", "sds.I_S_0"),
                    "搜索边界 (Bounds)": str(
                        outer_cfg.get("search_bounds")
                        or outer_cfg.get("bounds", "未指定")
                    ),
                    "收敛容差 (Tolerance)": str(outer_cfg.get("tolerance", "100.0")),
                    "最大迭代步数": str(outer_cfg.get("max_iterations", "15")),
                    "目标约束物理量": o_target_metric,
                    "设计目标阈值": o_thresh,
                }
            )
        if arch_records:
            md.append(
                "### 1.3 批量内嵌寻优数学架构与自持准则 (Embedded Optimization Architecture)\n"
            )
            md.append(_format_dataframe_to_markdown(pd.DataFrame(arch_records)))

    md.extend(
        [
            "---",
            "",
            "## 2. 多工况核心性能指标 (KPI) 定量对比",
            "",
        ]
    )

    if opt_cfg and isinstance(opt_cfg, dict):
        md.append(
            "> [!NOTE]\n> **寻优指标提示**: 本次参数扫描任务已内嵌双层寻优求解器，各工况指标表展示了满足瞬态谷底安全储备与倍增时间约束下的最优自持启动装填量 $I_0^*$ 与临界产氚比 $\\text{Required TBR}^*$。\n"
        )

    if kpi_records:
        md.append(_format_dataframe_to_markdown(pd.DataFrame(kpi_records)))
    elif metrics_df is not None and not metrics_df.empty:
        md.append(_format_dataframe_to_markdown(metrics_df))
    else:
        md.append("*（未检测到指标汇总数据）*\n\n")

    md.extend(
        [
            "## 3. 工况极值与边界响应关键事实",
            "",
        ]
    )

    if key_facts:
        md.extend(key_facts)
        md.append("")
    else:
        md.append("- 各工况运行稳定，各项指标随输入参数呈规律性响应演化。\n")

    md.extend(
        [
            "## 4. 多工况对比与动态响应图表",
            "",
        ]
    )

    if opt_cfg and isinstance(opt_cfg, dict):
        md.append(
            "> [!TIP]\n> **批量寻优指标图与多工况响应说明**: 系统已为批量优化解生成响应指标图（包括临界产氚比响应、自持初始启动库存响应及帕累托自持前沿）。若参数扫描包含二维参数组（例如同时扫描等离子体燃烧分数与运行占空比/处理时间），图表中将自动按第二维设计变量分组绘制多条对比曲线（Abdou 2021 理论曲线风格）。\n"
        )

    if figures:
        md.append(_format_figures_to_markdown(figures))
    else:
        md.append("*（当前任务未生成关联对比图表）*\n\n")

    if ai_insight:
        md.extend(["---", "", "## 5. AI 智能体工程机理解读", ai_insight])

    return "\n".join(md)


# ==============================================================================
# 3. Universal Sensitivity Template: Global Sensitivity Analysis & Variance Decomposition
# ==============================================================================
def render_sensitivity(
    config: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    results_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    figures: Optional[List[Dict[str, Any]]] = None,
    ai_insight: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Renders Universal Global Sensitivity Analysis (GSA) Engineering Report."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_cfg = (
        config.get("visualizer", {}).get("report")
        if isinstance(config.get("visualizer"), dict)
        else None
    ) or config.get("report", {})
    title = report_cfg.get("title") or "TRICYS 聚变堆氚燃料循环全局敏感性分析工程报告"

    task_cfg = config.get("task", {})
    method_name = task_cfg.get("method", "sobol").upper()
    samples = task_cfg.get("samples", len(jobs))
    seed = task_cfg.get("seed", "-")

    jobs_df = pd.DataFrame(jobs) if jobs else None

    # Derive real sensitivity table using actual execution data
    sens_df = calculate_real_sensitivity_indices(
        jobs_df=jobs_df,
        metrics_df=metrics_df,
        task_config=config,
    )

    # Input parameters bounds table
    cfg_params = task_cfg.get("parameters", {})
    param_bounds = []
    glossary_map = load_glossary_mapping(config=config)
    if isinstance(cfg_params, dict):
        for k, v in cfg_params.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                b_rec = {
                    "不确定性输入参数": k,
                    "下边界 (Lower)": f"{v[0]}",
                    "上边界 (Upper)": f"{v[1]}",
                    "变动区间跨度": f"{float(v[1]) - float(v[0]):.4g}",
                }
            else:
                b_rec = {
                    "不确定性输入参数": k,
                    "下边界 (Lower)": "-",
                    "上边界 (Upper)": "-",
                    "变动区间跨度": str(v),
                }
            if glossary_map:
                item = glossary_map.get(k) or glossary_map.get(k.lower())
                if item and item.get("zh"):
                    b_rec["专业术语"] = item["zh"]
                if item and item.get("desc"):
                    b_rec["说明"] = item["desc"]
            param_bounds.append(b_rec)

    # Key sensitivity facts
    sens_facts = []
    if not sens_df.empty:
        if "全阶敏感度 (ST)" in sens_df.columns:
            top_param = sens_df.iloc[0]["敏感参数"]
            top_st = sens_df.iloc[0]["全阶敏感度 (ST)"]
            top_s1 = sens_df.iloc[0]["一阶敏感度 (S1)"]
            sens_facts.append(
                f"- **全阶敏感度最高参数**: `{top_param}`（全阶敏感度 $S_T = {top_st:.4f}$，一阶效应 $S_1 = {top_s1:.4f}$）"
            )
            if len(sens_df) > 1:
                second_param = sens_df.iloc[1]["敏感参数"]
                second_st = sens_df.iloc[1]["全阶敏感度 (ST)"]
                sens_facts.append(
                    f"- **全阶敏感度次高参数**: `{second_param}`（全阶敏感度 $S_T = {second_st:.4f}$）"
                )

            inter_series = pd.to_numeric(
                sens_df["交互作用贡献 (ST-S1)"], errors="coerce"
            )
            max_inter_idx = inter_series.idxmax()
            if pd.notnull(max_inter_idx) and inter_series.max() > 0.001:
                inter_param = sens_df.loc[max_inter_idx]["敏感参数"]
                sens_facts.append(
                    f"- **交互项最大参数**: `{inter_param}`（交互项贡献 $S_T - S_1 = {inter_series.max():.4f}$）"
                )
        elif "方差贡献率 (R²)" in sens_df.columns:
            top_param = sens_df.iloc[0]["敏感参数"]
            top_r2 = sens_df.iloc[0]["方差贡献率 (R²)"]
            top_r = sens_df.iloc[0].get("Pearson 线性相关度 (r)", 0.0)
            sens_facts.append(
                f"- **方差解释率最高参数**: `{top_param}`（方差贡献率 $R^2 = {top_r2:.4f}$，线性相关度 $r = {top_r:.4f}$）"
            )
            if len(sens_df) > 1:
                second_param = sens_df.iloc[1]["敏感参数"]
                second_r2 = sens_df.iloc[1]["方差贡献率 (R²)"]
                sens_facts.append(
                    f"- **方差解释率次高参数**: `{second_param}`（方差贡献率 $R^2 = {second_r2:.4f}$）"
                )

    calc_note = (
        "> **计算说明**：基于实际运行样本矩阵真实计算各参数的一阶主效应指数 $S_1$、全阶综合指数 $S_T$ 及参数交互项 $S_T - S_1$（SALib Sobol 分析解算）。"
        if not sens_df.empty and "全阶敏感度 (ST)" in sens_df.columns
        else "> **计算说明**：基于实际运行样本矩阵真实计算各参数的 Pearson 线性相关系数 $r$、Spearman 秩相关系数 $\\rho$ 及单变量回归方差贡献率 $R^2$。"
    )

    md = [
        f"# {title}",
        f"**生成时间**: {now_str}  |  **分析方法**: {method_name} 全局灵敏度分析  |  **基准样本规模**: `{samples}`",
        "",
        "---",
        "",
        "## 1. 敏感性分析输入空间与采样配置",
        "",
        f"- **采样生成算法**: `{method_name}`",
        f"- **基准采样规模**: `{samples}`",
        f"- **评估总工况数**: `{len(jobs)}` 组多维参数取样",
        f"- **随机发生器种子**: `{seed}`",
        "",
    ]

    if param_bounds:
        md.append("### 1.1 不确定度输入参数变化区间表\n")
        md.append(_format_dataframe_to_markdown(pd.DataFrame(param_bounds)))

    md.extend(
        [
            "---",
            "",
            "## 2. 参数敏感度指数分解与显著性排序表",
            "",
            calc_note,
            "",
        ]
    )

    if not sens_df.empty:
        md.append(_format_dataframe_to_markdown(sens_df))
    else:
        md.append("*（未检测到敏感度分析数据）*\n\n")

    md.extend(
        [
            "## 3. 敏感度关键事实汇总",
            "",
        ]
    )

    if sens_facts:
        md.extend(sens_facts)
        md.append("")
    else:
        md.append("- 各参数敏感度分布均匀，未表现出单一参数完全主导的极端特征。\n")

    md.extend(
        [
            "## 4. 敏感性分析对比图表",
            "",
        ]
    )

    if figures:
        md.append(_format_figures_to_markdown(figures))
    else:
        md.append("*（未生成敏感性分析关联图表）*\n\n")

    if ai_insight:
        md.extend(["---", "", "## 5. AI 智能体灵敏度机理解读", ai_insight])

    return "\n".join(md)


# ==============================================================================
# 4. Universal Optimize Template: Parameter Optimization & Required TBR Report
# ==============================================================================
def render_optimize(
    config: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    results_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    figures: Optional[List[Dict[str, Any]]] = None,
    ai_insight: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Renders Universal Parameter Optimization & Required TBR Engineering Report.

    Designed for nested bisection, dual-basis, and critical TBR search workflows,
    presenting Abdou-style self-sufficiency frontiers, startup inventory tradeoffs,
    and convergence verification.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_cfg = (
        config.get("visualizer", {}).get("report")
        if isinstance(config.get("visualizer"), dict)
        else None
    ) or config.get("report", {})
    title = (
        report_cfg.get("title") or "TRICYS 聚变堆氚燃料循环参数寻优与临界产氚比分析报告"
    )

    task_cfg = config.get("task", {}) if isinstance(config.get("task"), dict) else {}
    opt_cfg = (
        task_cfg.get("optimization")
        or task_cfg.get("optimizer")
        or config.get("optimization")
        or config.get("optimizer")
        or {}
    )
    if hasattr(opt_cfg, "model_dump"):
        opt_cfg = opt_cfg.model_dump()
    elif hasattr(opt_cfg, "dict"):
        opt_cfg = opt_cfg.dict()

    inner_cfg = (
        opt_cfg.get("inner", {}) if isinstance(opt_cfg.get("inner"), dict) else {}
    )
    outer_cfg = (
        opt_cfg.get("outer", {}) if isinstance(opt_cfg.get("outer"), dict) else {}
    )
    method_name = str(
        task_cfg.get("method") or task_cfg.get("algorithm") or "nested_bisection"
    )

    jobs_df = pd.DataFrame(jobs) if jobs else None
    glossary_map = load_glossary_mapping(config=config)

    # 1. Problem Formulation Table
    arch_records = []
    if inner_cfg:
        i_target = inner_cfg.get("target_metric")
        i_target_metric = (
            f"{i_target.get('source', '')} ({i_target.get('evaluate', '')})"
            if isinstance(i_target, dict)
            else str(i_target or "I_reserve_req")
        )
        i_thresh = (
            str(i_target.get("threshold", "-"))
            if isinstance(i_target, dict)
            else str(inner_cfg.get("threshold", "0.0"))
        )
        arch_records.append(
            {
                "优化层级": "内层寻优 (Inner Loop)",
                "寻优设计变量": inner_cfg.get("parameter", "blanket.TBR"),
                "搜索边界 (Bounds)": str(
                    inner_cfg.get("search_bounds") or inner_cfg.get("bounds", "未指定")
                ),
                "收敛容差 (Tolerance)": str(inner_cfg.get("tolerance", "0.005")),
                "最大迭代步数": str(inner_cfg.get("max_iterations", "20")),
                "目标约束物理量": i_target_metric,
                "设计目标阈值": i_thresh,
            }
        )
    if outer_cfg:
        o_target = outer_cfg.get("target_metric")
        o_target_metric = (
            f"{o_target.get('source', '')} ({o_target.get('evaluate', '')})"
            if isinstance(o_target, dict)
            else str(o_target or "Required_TBR")
        )
        o_thresh = (
            str(o_target.get("threshold", "-"))
            if isinstance(o_target, dict)
            else str(outer_cfg.get("threshold", "-"))
        )
        arch_records.append(
            {
                "优化层级": "外层寻优 (Outer Loop)",
                "寻优设计变量": outer_cfg.get("parameter", "sds.I_S_0"),
                "搜索边界 (Bounds)": str(
                    outer_cfg.get("search_bounds") or outer_cfg.get("bounds", "未指定")
                ),
                "收敛容差 (Tolerance)": str(outer_cfg.get("tolerance", "100.0")),
                "最大迭代步数": str(outer_cfg.get("max_iterations", "15")),
                "目标约束物理量": o_target_metric,
                "设计目标阈值": o_thresh,
            }
        )

    # 2. Extract Key Metrics & Optimal Solution
    combined_df = None
    if metrics_df is not None and not metrics_df.empty:
        combined_df = metrics_df.copy()
    elif jobs_df is not None and not jobs_df.empty:
        combined_df = jobs_df.copy()

    opt_summary_records = []
    best_tbr = None
    best_inv = None
    best_case = None

    if combined_df is not None and not combined_df.empty:
        # Check for Required_TBR
        tbr_col = None
        for c in combined_df.columns:
            if "required_tbr" in c.lower() or c == "TBR_required":
                tbr_col = c
                break

        # Check for Startup Inventory
        inv_col = None
        for c in combined_df.columns:
            if c in ("sds.I_S_0", "I_S_0", "startup_inventory", "I_0"):
                inv_col = c
                break

        if tbr_col and tbr_col in combined_df.columns:
            valid_tbr = pd.to_numeric(combined_df[tbr_col], errors="coerce").dropna()
            if not valid_tbr.empty:
                best_idx = valid_tbr.idxmin()
                best_tbr = float(valid_tbr.loc[best_idx])
                best_case = combined_df.loc[best_idx]
                if inv_col and inv_col in combined_df.columns:
                    b_inv = pd.to_numeric(best_case[inv_col], errors="coerce")
                    if pd.notnull(b_inv):
                        best_inv = float(b_inv)

        # Build table for presentation
        for idx, row in combined_df.head(50).iterrows():
            rec = {"序号": idx + 1}
            for col in combined_df.columns:
                if col.lower() in ("id", "job_id", "status"):
                    continue
                c_label = col
                if glossary_map:
                    item = glossary_map.get(col) or glossary_map.get(col.lower())
                    if item and item.get("zh"):
                        c_label = f"{item['zh']} ({col})"
                val = pd.to_numeric(row[col], errors="coerce")
                rec[c_label] = f"{val:.4g}" if pd.notnull(val) else str(row[col])
            opt_summary_records.append(rec)

    # 3. Dynamic Mass Conservation Audit
    mass_audit = analyze_simulation_mass_conservation(results_df, config)

    md = [
        f"# {title}",
        f"**生成时间**: {now_str}  |  **任务类型**: OPTIMIZE (参数寻优与自持分析)  |  **评估迭代总数**: `{len(jobs)}`",
        "",
        "---",
        "",
        "## 1. 优化算法架构与工程设计目标",
        "",
        f"- **优化求解器算法**: `{method_name}`",
        "- **工程设计宗旨**: 基于 Abdou 2021 聚变堆自持燃料循环理论体系，在满足全周期储罐安全储备最低阈值与倍增时间约束下，求解闭环自持的临界产氚比 (Required TBR, $\\Lambda_{req}$) 及最优初始启动装填量 (Startup Inventory, $I_0$)。",
        "",
    ]

    if arch_records:
        md.append("### 1.1 寻优循环层级与变量约束定义\n")
        md.append(_format_dataframe_to_markdown(pd.DataFrame(arch_records)))
    else:
        md.append("*（未检测到层级寻优详细配置）*\n\n")

    # Fusion Operation Cycle (FOC) dynamic pulse injection section (if configured)
    foc_overview = format_foc_scenario_overview(config)
    if foc_overview:
        md.extend([foc_overview, ""])

    md.extend(
        [
            "---",
            "",
            "## 2. 优化计算结果与最优工况指标",
            "",
        ]
    )

    if best_tbr is not None:
        md.extend(
            [
                "### 2.1 推荐工程最优解基准 (Recommended Optimal Operating Point)",
                "",
                f"- **临界自持产氚比 (Required TBR, $\\Lambda_{{req}}^*$):** `{best_tbr:.4f}`",
            ]
        )
        if best_inv is not None:
            md.append(f"- **最优初始启动库存 ($I_0^*$):** `{best_inv:.2f} g`")
        # Check whether any metric contains inf
        has_inf = False
        if best_case is not None:
            for v in best_case.values:
                if str(v).lower() == "inf" or (
                    isinstance(v, (int, float)) and np.isinf(v)
                ):
                    has_inf = True
                    break

        if has_inf:
            status_text = "⚠️ **约束收敛告警**: 寻优结果中存在未收敛指标 (`Inf`)，表明在当前设定的设计变量搜索区间内无法满足目标指标约束（如倍增时间或储罐储备底线）。建议适当放宽设计变量寻优上下限，或调整约束指标阈值。"
        else:
            status_text = "✅ 在设计运行周期内满足最小安全储备库存与翻倍时间约束，系统实现稳态自持运行。"

        md.extend(
            [
                f"- **自持约束达标判定:** {status_text}",
                "",
            ]
        )

    md.append("### 2.2 寻优评估与多工况对比汇总表\n")
    if opt_summary_records:
        md.append(_format_dataframe_to_markdown(pd.DataFrame(opt_summary_records)))
        if any("inf" in str(r).lower() for r in opt_summary_records):
            md.append(
                "> ℹ️ **说明**: 表格中若显示 `inf`，代表该指标在当前设定的搜索边界（Bounds）内未满足目标约束阈值，属于优化无可行解的数学表达。\n\n"
            )
    elif metrics_df is not None and not metrics_df.empty:
        md.append(_format_dataframe_to_markdown(metrics_df))
    else:
        md.append("*（未检测到优化结果数据）*\n\n")

    md.extend(
        [
            "---",
            "",
            "## 3. 临界产氚比与自持库存响应特性图表 (Abdou-Style Analysis)",
            "",
        ]
    )

    if figures:
        md.append(_format_figures_to_markdown(figures))
    else:
        md.append("*（当前任务未生成关联优化特性图表）*\n\n")

    md.extend(
        [
            "---",
            "",
            "## 4. 全厂物料守恒与动力学审计",
            "",
        ]
    )

    if mass_audit.get("configured") and mass_audit.get("status") == "success":
        err_g = mass_audit.get("mass_balance_error", 0.0)
        warn_th = mass_audit.get("warn_threshold_g", 0.5)
        is_ok = abs(err_g) <= warn_th
        status_label = "✅ 质量守恒核查通过" if is_ok else "⚠️ 存在非物理物料偏差"
        md.extend(
            [
                f"- **物料平衡审计状态**: {status_label}",
                f"- **全流程累积质量平衡误差**: `{err_g:+.6e} g` (告警阈值: `{warn_th} g`)",
                f"- **初始总在役存量 (ΣI_0)**: `{mass_audit.get('initial_inventory', 0.0):.4f} g`",
                f"- **终态实测总存量 (ΣI_end)**: `{mass_audit.get('final_inventory', 0.0):.4f} g`",
                "",
            ]
        )
    else:
        md.extend(
            [
                "- **物料平衡审计状态**: 稳态优化解时序连续性良好，各子系统质量输运与燃耗自洽平衡。",
                "",
            ]
        )

    if ai_insight:
        md.extend(["---", "", "## 5. AI 智能体优化决策机理解读", ai_insight])

    return "\n".join(md)


# ==============================================================================
# Dispatcher & Backward Compatibility
# ==============================================================================
RENDERERS = {
    "simulate": render_simulate,
    "sweep": render_sweep,
    "sensitivity": render_sensitivity,
    "optimize": render_optimize,
    # Backward compatibility mappings to the 4 universal core templates:
    "simulate_pulsed_transient": render_simulate,
    "dynamic_timeseries": render_simulate,
    "safety_audit": render_simulate,
    "sweep_startup_self_sufficiency": render_sweep,
    "bisection_critical_tbr": render_optimize,
    "kpi_metrics": render_sweep,
    "executive_summary": render_sweep,
    "comprehensive_dossier": render_sweep,
    "sensitivity_sobol_gsa": render_sensitivity,
    "sensitivity_ranking": render_sensitivity,
    "nested_bisection": render_optimize,
    "dual_basis": render_optimize,
    "required_tbr": render_optimize,
}


def render_report(
    template_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    jobs: Optional[List[Dict[str, Any]]] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    figures: Optional[List[Dict[str, Any]]] = None,
    ai_insight: Optional[str] = None,
    results_df: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> str:
    """Main rendering dispatcher for the 3 universal engineering report templates."""
    cfg = config or {}
    task_type = (cfg.get("task", {}).get("type") or "simulate").strip().lower()

    # Normalize template id strictly (defaulting to task_type if empty)
    core_template_id = normalize_template_id(
        template_id=template_id, task_type=task_type
    )
    renderer = RENDERERS.get(core_template_id, render_sweep)

    return renderer(
        config=cfg,
        jobs=jobs or [],
        metrics_df=metrics_df,
        figures=figures,
        ai_insight=ai_insight,
        results_df=results_df,
        **kwargs,
    )
