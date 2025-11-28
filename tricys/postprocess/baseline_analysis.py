"This module provides functions for plotting simulation results."

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


_english_glossary_map = {}
_chinese_glossary_map = {}
_use_chinese_labels = False

# Add a dictionary for UI text translations
_ui_text = {
    "en": {
        "overall_view": "Overall View (Data exceeding 2x initial value is hidden)",
        "detailed_view": "Detailed View (t=0 to Post-Minimum)",
        "time_days": "Time (days)",
        "days": "days",
        "kg": "kg",
        "g": "g",
        "hours": "hours",
        "years": "years",
        "y_label": "Tritium Inventory",
        "overall_view_title": "Overall View",
        "detailed_view_zoom_title": "Detailed View (Zoomed on '{detailed_var}' Self Sufficiency Point)",
        "final_values_bar_chart_title": "Tritium Inventory in Each Submodule",
        "final_value": "Tritium Inventory (g)",
    },
    "cn": {
        "overall_view": "全局视图 (超出初始值2倍的数据已隐藏)",
        "detailed_view": "细节视图 (t=0到最小值后)",
        "time_days": "时间 (天)",
        "days": "天",
        "kg": "千克",
        "g": "克",
        "hours": "小时",
        "years": "年",
        "y_label": "氚盘存量",
        "overall_view_title": "全局视图",
        "detailed_view_zoom_title": "细节视图 (放大“{detailed_var}”自持点)",
        "final_values_bar_chart_title": "各子模块氚盘存量",
        "final_value": "氚盘存量 (g)",
    },
}


def _get_text(key: str) -> str:
    """Helper to get text based on the current language setting.

    Args:
        key: The text key to retrieve.

    Returns:
        The translated text string, or the key itself if not found.

    Note:
        Falls back to the key if translation not found. Useful for dynamic keys
        from unit_map that may not be in _ui_text dictionary.
    """
    # Fallback to key itself if not found, useful for units from unit_map
    lang = "cn" if _use_chinese_labels else "en"
    return _ui_text[lang].get(key, key)


def set_plot_language(lang: str = "en") -> None:
    """Sets the preferred language for plot labels.

    Args:
        lang: 'en' for English (default), 'cn' for Chinese.

    Note:
        For Chinese, sets font to SimHei and adjusts unicode_minus. For English,
        restores matplotlib defaults. Changes apply globally to all subsequent plots.
    """
    global _use_chinese_labels
    _use_chinese_labels = lang.lower() == "cn"

    if _use_chinese_labels:
        # To display Chinese characters correctly, specify a list of fallback fonts.
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 替换成你电脑上有的字体
        plt.rcParams["axes.unicode_minus"] = False  # To display minus sign correctly.
        plt.rcParams["font.family"] = "sans-serif"  # 确保字体家族设置生效
    else:
        # Restore default settings
        plt.rcParams["font.sans-serif"] = plt.rcParamsDefault["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = plt.rcParamsDefault["axes.unicode_minus"]


def load_glossary(glossary_path: str) -> None:
    """Loads glossary data from the specified CSV path into global dictionaries.

    Args:
        glossary_path: Path to the glossary CSV file.

    Note:
        Expected columns: "模型参数 (Model Parameter)", "英文术语 (English Term)",
        "中文翻译 (Chinese Translation)". Clears existing glossaries on error.
        Updates global _english_glossary_map and _chinese_glossary_map.
    """
    global _english_glossary_map, _chinese_glossary_map

    if not glossary_path or not os.path.exists(glossary_path):
        logger.warning(
            f"Glossary file not found at {glossary_path}. No labels will be loaded."
        )
        _english_glossary_map = {}
        _chinese_glossary_map = {}
        return

    try:
        df = pd.read_csv(glossary_path)
        if (
            "模型参数 (Model Parameter)" in df.columns
            and "英文术语 (English Term)" in df.columns
            and "中文翻译 (Chinese Translation)" in df.columns
        ):
            df.dropna(subset=["模型参数 (Model Parameter)"], inplace=True)
            _english_glossary_map = pd.Series(
                df["英文术语 (English Term)"].values,
                index=df["模型参数 (Model Parameter)"],
            ).to_dict()
            _chinese_glossary_map = pd.Series(
                df["中文翻译 (Chinese Translation)"].values,
                index=df["模型参数 (Model Parameter)"],
            ).to_dict()
            logger.info(f"Successfully loaded glossary from {glossary_path}.")
        else:
            logger.warning("Glossary CSV does not contain expected columns.")
            _english_glossary_map = {}
            _chinese_glossary_map = {}
    except Exception as e:
        logger.warning(f"Failed to load or parse glossary file. Error: {e}")
        _english_glossary_map = {}
        _chinese_glossary_map = {}


def _format_label(label: str) -> str:
    """Formats a label for display using glossary or simple formatting.

    Args:
        label: The raw label string to format.

    Returns:
        The formatted label string.

    Note:
        First checks glossary for professional term (language-specific). If not found,
        replaces underscores with spaces and removes dots (except in numbers). Returns
        non-string inputs unchanged.
    """
    global _english_glossary_map, _chinese_glossary_map, _use_chinese_labels

    if not isinstance(label, str):
        return label

    glossary_map = (
        _chinese_glossary_map if _use_chinese_labels else _english_glossary_map
    )

    if glossary_map and label in glossary_map:
        term = glossary_map[label]
        if pd.notna(term) and str(term).strip():
            return str(term)

    # Fallback to simple formatting
    formatted_label = label.replace("_", " ")
    formatted_label = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", formatted_label)
    return formatted_label


def _calculate_startup_inventory(
    series: pd.Series, time_series: Optional[pd.Series] = None
) -> float:
    """
    Calculates the startup inventory as the difference between the initial
    inventory and the minimum inventory (the turning point).
    """
    initial_inventory = series.iloc[0]
    minimum_inventory = series.min()
    return initial_inventory - minimum_inventory


def _time_of_turning_point(series: pd.Series, time_series: pd.Series) -> float:
    """
    Finds the time of the turning point (minimum value) in the series.
    This represents the self-sufficiency time.
    To handle noisy data, it first smooths the series to determine if an
    overall turning point exists. If the minimum of the smoothed trend is at
    the beginning or end of the series (indicating a monotonic trend),
    it returns NaN. Otherwise, it returns the time of the minimum value from
    the original, unsmoothed data.
    """

    logger.debug(
        f"Calculating time_of_turning_point for series with length {len(series)} and {len(time_series)} "
    )
    if time_series is None:
        raise ValueError("time_series must be provided for time_of_turning_point")

    # Define a window size for the rolling average, e.g., 5% of the data length
    # with a minimum size of 1. This helps in smoothing out local fluctuations.
    window_size = max(1, int(len(series) * 0.001))
    smoothed_series = series.rolling(
        window=window_size, center=True, min_periods=1
    ).mean()

    # Find the index label of the minimum value in the smoothed series.
    smooth_min_index = smoothed_series.idxmin()

    # Check if the minimum of the smoothed data is within the first or last 5%
    # of the series. If so, the trend is considered monotonic.
    smooth_min_pos = series.index.get_loc(smooth_min_index)
    five_percent_threshold = int(len(series) * 0.3)

    if smooth_min_pos >= len(series) - five_percent_threshold:
        return np.nan
    else:
        # A clear turning point is identified in the overall trend.
        # Now, find the precise turning point in the original, noisy data.
        min_index = series.idxmin()
        return time_series.loc[min_index]


def _calculate_doubling_time(series: pd.Series, time_series: pd.Series) -> float:
    """
    Calculates the time it takes for the inventory to double its initial value.
    """
    if time_series is None:
        raise ValueError("time_series must be provided for calculate_doubling_time")
    initial_inventory = series.iloc[0]
    doubled_inventory = 2 * initial_inventory

    # Find the first index where the inventory is >= doubled_inventory
    # We should only consider the part of the series after the turning point
    min_index = series.idxmin()
    after_turning_point_series = series.loc[min_index:]

    doubling_indices = after_turning_point_series[
        after_turning_point_series >= doubled_inventory
    ].index

    if not doubling_indices.empty:
        doubling_index = doubling_indices[0]
        return time_series.loc[doubling_index]
    else:
        # If it never doubles, return NaN
        return np.nan


def _plot_time_series_with_zoom(df: pd.DataFrame, output_dir: str, **kwargs) -> None:
    """Helper to generate the time-series plot with a detailed zoom view."""
    detailed_var = kwargs.get("detailed_var", "sds.I[1]")
    color_map = kwargs.get("color_map", {})

    time_days = df["time"] / 24
    all_plot_columns = sorted([col for col in df.columns if col != "time"])
    primary_var_columns = [
        col for col in all_plot_columns if col.startswith(detailed_var)
    ]

    min_x_global = float("inf")
    if primary_var_columns:
        # Find the column with the absolute minimum value among primary variables
        min_col = None
        min_val_for_col = float("inf")

        for col in primary_var_columns:
            y_data = df[col]
            if not y_data.empty:
                current_min = y_data.min()
                if current_min < min_val_for_col:
                    min_val_for_col = current_min
                    min_col = col

        if min_col:
            y_data = df[min_col]
            min_idx = y_data.idxmin()
            min_x_global = time_days.loc[min_idx]

    sns.set_theme(style="whitegrid")

    original_lang_is_chinese = _use_chinese_labels
    for lang in ["en", "cn"]:
        set_plot_language(lang)

        y_label = kwargs.get("y_label", _get_text("y_label"))

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=kwargs.get("figsize", (14, 18)),
            sharex=False,
            gridspec_kw={"height_ratios": [2, 1]},
        )

        for column in all_plot_columns:
            color = color_map.get(column, "blue")  # Default to blue if not in map
            ax1.plot(
                time_days,
                df[column],
                label=_format_label(column),
                color=color,
                linewidth=1.2,
                alpha=0.85,
            )
            ax2.plot(
                time_days,
                df[column],
                label=_format_label(column),
                color=color,
                linewidth=1.5,
                alpha=0.9,
            )

        ax1.set_ylabel(_format_label(y_label), fontsize=14)
        ax1.set_title(_get_text("overall_view_title"), fontsize=12)
        if len(all_plot_columns) <= 20:
            ax1.legend(loc="best", fontsize="x-small")
        ax1.grid(True)
        ax1.set_xlabel(_get_text("time_days"), fontsize=14)

        ax2.set_ylabel(_format_label(y_label), fontsize=14)
        detailed_view_title = _get_text("detailed_view_zoom_title").format(
            detailed_var=_format_label(detailed_var)
        )
        ax2.set_title(detailed_view_title, fontsize=12)
        ax2.grid(True, linestyle="--")
        ax2.set_xlabel(_get_text("time_days"), fontsize=14)
        if len(all_plot_columns) <= 20:
            ax2.legend(loc="best", fontsize="x-small")

        if np.isfinite(min_x_global):
            x1_zoom, x2_zoom = 0, min_x_global + 5
            ax2.set_xlim(x1_zoom, x2_zoom)
            y_min_in_range = (
                df.loc[
                    (time_days >= x1_zoom) & (time_days <= x2_zoom), all_plot_columns
                ]
                .min()
                .min()
            )
            y_max_in_range = (
                df.loc[
                    (time_days >= x1_zoom) & (time_days <= x2_zoom), all_plot_columns
                ]
                .max()
                .max()
            )
            y_padding = (y_max_in_range - y_min_in_range) * 0.1
            ax2.set_ylim(y_min_in_range - y_padding, y_max_in_range + y_padding)
            y1_ax1, y2_ax1 = ax1.get_ylim()
            ax1.add_patch(
                patches.Rectangle(
                    (x1_zoom, y1_ax1),
                    x2_zoom - x1_zoom,
                    y2_ax1 - y1_ax1,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="red",
                    linestyle="--",
                    alpha=0.1,
                )
            )
        else:
            logger.warning(
                f"Could not determine a zoom range for the detailed view. "
                f"The primary variable for zooming '{detailed_var}' was not found or had no data. "
                "The detailed view will be hidden."
            )
            ax2.set_visible(False)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        base_filename = kwargs.get(
            "output_filename", "simulation_all_curves_detailed.svg"
        )
        name, ext = os.path.splitext(base_filename)
        suffix = "_zh" if lang == "cn" else ""
        output_filename = f"{name}{suffix}{ext}"

        save_path = os.path.join(output_dir, output_filename)
        try:
            plt.rcParams["svg.fonttype"] = "path"
            plt.savefig(save_path, format="svg", bbox_inches="tight")
            logger.info(f"Successfully generated plot with all curves: {save_path}")
        finally:
            plt.close(fig)

    set_plot_language("cn" if original_lang_is_chinese else "en")


def _plot_final_values_bar_chart(df: pd.DataFrame, output_dir: str, **kwargs) -> None:
    """Helper to generate a bar chart of the final values for each column."""
    last_values = df.drop(columns=["time"]).iloc[-1].sort_values(ascending=False)
    color_map = kwargs.get("color_map", {})

    # Create a list of colors ordered according to the sorted values
    bar_colors = [color_map.get(col, "blue") for col in last_values.index]

    original_lang_is_chinese = _use_chinese_labels
    for lang in ["en", "cn"]:
        plt.style.use("seaborn-v0_8-whitegrid")
        set_plot_language(lang)

        fig, ax = plt.subplots(figsize=kwargs.get("bar_chart_figsize", (12, 8)))

        x_labels = [_format_label(col) for col in last_values.index]
        sns.barplot(x=x_labels, y=last_values.values, ax=ax, palette=bar_colors)

        title = kwargs.get("bar_chart_title", _get_text("final_values_bar_chart_title"))
        ax.set_title(_format_label(title), fontsize=16, fontweight="bold")
        y_label = kwargs.get("y_label", _get_text("final_value"))
        ax.set_ylabel(_format_label(y_label), fontsize=12)
        ax.set_xlabel("", fontsize=12)

        plt.xticks(rotation=45, ha="right", fontsize=10)

        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2e}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )

        fig.tight_layout()
        base_filename = kwargs.get("bar_chart_filename", "final_values_bar_chart.svg")
        name, ext = os.path.splitext(base_filename)
        suffix = "_zh" if lang == "cn" else ""
        output_filename = f"{name}{suffix}{ext}"
        save_path = os.path.join(output_dir, output_filename)
        try:
            plt.rcParams["svg.fonttype"] = "path"
            plt.savefig(save_path, format="svg", bbox_inches="tight")
            logger.info(
                f"Successfully generated bar chart of final values: {save_path}"
            )
        finally:
            plt.close(fig)

    set_plot_language("cn" if original_lang_is_chinese else "en")


def _call_openai_for_postprocess_analysis(
    api_key: str,
    base_url: str,
    ai_model: str,
    report_content: str,
    **kwargs,
) -> Optional[str]:
    """
    Constructs a prompt for post-simulation analysis, calls the OpenAI API, and returns the result.
    """
    try:
        logger.info("Proceeding with LLM analysis for post-simulation report.")

        detailed_var = kwargs.get("detailed_var", "sds.I[1]")

        # --- New Prompt ---
        role_prompt = """**角色：** 你是一名聚变反应堆氚燃料循环领域的专家。

**任务：** 请仔细审查并解读以下单次模拟运行的**数据**。由于无法查看图表，你的分析必须**完全基于**报告中提供的**数据表**，包括“最终值数据表”和“关键阶段抽样数据”。请遵循以下结构，对关键指标进行分析，并总结本次模拟的发现。
"""

        analysis_prompt = f"""**分析数据：**

{report_content}
"""

        points_prompt = f"""
**分析要点：**

1.  **总体趋势分析 (基于抽样数据和最终值):**
    *   结合 **初始阶段**、**转折点阶段** 和 **结束阶段** 的抽样数据，描述主要变量（特别是 `{detailed_var}` 和其他关键库存）随时间变化的总体趋势。
    *   `{detailed_var}` 的值是如何从初始阶段变化到转折点，再到结束阶段的？这揭示了什么物理过程？

2.  **关键事件分析 (基于转折点数据):**
    *   详细分析 **转折点阶段数据**。`{detailed_var}` 在这个阶段达到最小值，这个值大约是多少？对应的时间点是什么？
    *   这个转折点在氚燃料循环中通常意味着什么？（例如：它是否代表了系统从氚消耗主导转向氚增殖主导的时刻，即接近或达到氚自持？）

3.  **关键性能指标分析 (基于关键性能指标数据表):**
    *   分析报告中的 **“关键性能指标”** 数据表。
    *   **启动库存 (Startup Inventory)** 的值是多少？它在氚经济性方面说明了什么？
    *   **自持时间 (Self-Sufficiency Time)** 是多少天？这个时间点对于评估氚燃料循环的性能有何重要意义？
    *   **倍增时间 (Doubling Time)** 是多少天？如果这个值存在（非 N/A），它揭示了系统氚增殖的速率和潜力。如果为 N/A，可能的原因是什么？

4.  **最终状态评估 (基于最终值数据表):**
    *   分析报告中的 **“最终值数据表”**。模拟结束时，哪些变量的值最大？这说明氚主要存储在哪个子系统中？
    *   最终的氚总库存量是多少？与初始库存相比有何变化？

5.  **结论：**
    *   根据以上对**数据表**的分析，总结本次模拟运行的主要发现。
    *   基于这些数据驱动的发现，对模型或操作参数有什么初步的建议或观察？
"""

        # 2. Call API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                logger.info(
                    f"Sending request to OpenAI API for post-simulation analysis (Attempt {attempt + 1}/{max_retries})..."
                )

                full_text_prompt = "\n\n".join(
                    [role_prompt, analysis_prompt, points_prompt]
                )

                response = client.chat.completions.create(
                    model=ai_model,
                    messages=[{"role": "user", "content": full_text_prompt}],
                    max_tokens=4000,
                )
                analysis_result = response.choices[0].message.content

                logger.info("LLM analysis successful for post-simulation report.")
                return (
                    role_prompt
                    + points_prompt
                    + "\n```\n\n"
                    + "\n\n---\n\n# AI模型分析结果\n\n"
                    + analysis_result
                )

            except Exception as e:
                logger.error(f"Error calling OpenAI API on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    logger.error(
                        f"Failed to call OpenAI API after {max_retries} attempts."
                    )
                    return None

    except Exception as e:
        logger.error(
            f"Error in _call_openai_for_postprocess_analysis: {e}",
            exc_info=True,
        )
        return None


def _generate_postprocess_report(
    df: pd.DataFrame, output_dir: str, **kwargs
) -> Tuple[Optional[str], Optional[str]]:
    """Generates a Markdown report for the post-simulation analysis."""
    try:
        logger.info("Starting to generate post-process report.")
        # File paths for plots (point to Chinese versions for the Chinese report)
        ts_filename_base = kwargs.get(
            "output_filename", "simulation_all_curves_detailed.svg"
        )
        ts_name, ts_ext = os.path.splitext(ts_filename_base)
        time_series_plot_filename = f"{ts_name}_zh{ts_ext}"

        bar_filename_base = kwargs.get(
            "bar_chart_filename", "final_values_bar_chart.svg"
        )
        bar_name, bar_ext = os.path.splitext(bar_filename_base)
        bar_chart_filename = f"{bar_name}_zh{bar_ext}"

        report_filename = kwargs.get(
            "report_filename", "baseline_condition_analysis_report.md"
        )
        report_path = os.path.join(output_dir, report_filename)

        # --- Markdown Generation ---
        report_lines = [
            "# 基准工况分析报告\n\n",
            f"生成时间: {pd.Timestamp.now()}\n\n",
        ]

        # --- Configuration Details ---
        config_to_report = {k: v for k, v in kwargs.items() if k not in ["color_map"]}
        if config_to_report:
            report_lines.extend(
                [
                    "## 后处理配置详情\n\n",
                    "本次后处理任务的具体配置如下：\n\n",
                    "```json\n",
                    json.dumps(config_to_report, indent=4, ensure_ascii=False),
                    "\n```\n\n",
                ]
            )

        # --- Key Metrics Calculation ---
        report_lines.append("## 关键性能指标\n\n")
        detailed_var = kwargs.get("detailed_var", "sds.I[1]")
        primary_var_columns = sorted(
            [col for col in df.columns if col.startswith(detailed_var)]
        )

        metrics_data = []
        if not primary_var_columns:
            report_lines.append(
                f"未找到与主要变量 '{detailed_var}' 相关的列，无法计算关键指标。\n\n"
            )
        else:
            for col in primary_var_columns:
                series = df[col]
                time_series = df["time"]

                startup_inventory = _calculate_startup_inventory(series, time_series)
                turning_point_time = _time_of_turning_point(series, time_series)
                doubling_time = _calculate_doubling_time(series, time_series)

                metrics_data.append(
                    {
                        "变量 (Variable)": col,
                        "启动库存 (Startup Inventory)": f"{startup_inventory:.4f}",
                        "自持时间 (Self-Sufficiency Time)": (
                            f"{turning_point_time/24:.2f} 天"
                            if pd.notna(turning_point_time)
                            else "N/A"
                        ),
                        "倍增时间 (Doubling Time)": (
                            f"{doubling_time/24:.2f} 天"
                            if pd.notna(doubling_time)
                            else "N/A"
                        ),
                    }
                )

            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                report_lines.append(metrics_df.to_markdown(index=False))
                report_lines.append("\n\n")

        # --- Plotting Sections ---
        report_lines.extend(
            [
                "## 模拟结果时序图\n\n",
                "下图展示了所有变量随时间变化的曲线，并对关键转折点进行了放大。\n\n",
                f"![时序图]({time_series_plot_filename})\n\n",
                "## 模拟结束时各变量最终值\n\n",
                "下图通过条形图展示了模拟结束时各个变量的最终值，并按大小排序。\n\n",
                f"![最终值条形图]({bar_chart_filename})\n\n",
            ]
        )

        # --- Data Table (Final Values) ---
        report_lines.append("## 最终值数据表\n\n")
        last_values = df.drop(columns=["time"]).iloc[-1].sort_values(ascending=False)
        report_lines.append(
            last_values.to_frame(
                name=kwargs.get("y_label", "Final Value")
            ).to_markdown()
        )
        report_lines.append("\n\n")
        logger.info("Added final values table to report.")

        # --- Data Sampling Section ---
        logger.info("Starting data sampling for the report.")
        detailed_var = kwargs.get("detailed_var", "sds.I[1]")
        primary_var_columns = [
            col for col in df.columns if col.startswith(detailed_var)
        ]

        min_idx = -1
        if primary_var_columns:
            min_col_val = float("inf")
            min_col_name = None
            for col in primary_var_columns:
                if df[col].min() < min_col_val:
                    min_col_val = df[col].min()
                    min_col_name = col
            if min_col_name:
                min_idx = df[min_col_name].idxmin()

        # New sampling logic
        num_points = 20
        interval = 2
        window_size = (num_points - 1) * interval + 1

        start_data = df.iloc[:window_size:interval]
        end_data = df.iloc[-(window_size)::interval]

        report_lines.append("## 关键阶段抽样数据\n\n")
        report_lines.append(
            f"这是从完整时间序列数据中抽样的三个关键阶段的表格，每个阶段包含约 {num_points} 个数据点 (采样间隔 {interval})。\n\n"
        )
        report_lines.append(f"### 1. 初始阶段数据 (前 {num_points} 个数据点)\n")
        report_lines.append(start_data.to_markdown(index=False) + "\n\n")

        if min_idx != -1:
            # Window of ~20 points with interval 2 around the turning point
            window_radius_indices = (num_points // 2) * interval

            start_idx = max(0, min_idx - window_radius_indices)
            end_idx = min(len(df), min_idx + window_radius_indices)

            turning_point_data = df.iloc[start_idx:end_idx:interval]

            report_lines.append(
                f"### 2. 转折点阶段数据 (围绕 '{detailed_var}' 的最小值)\n"
            )
            report_lines.append(turning_point_data.to_markdown(index=False) + "\n\n")
        else:
            report_lines.append(
                f"### 2. 转折点阶段数据\n在提供的抽样中未找到 '{detailed_var}' 的明确转折点。\n\n"
            )

        report_lines.append(f"### 3. 结束阶段数据 (后 {num_points} 个数据点)\n")
        report_lines.append(end_data.to_markdown(index=False) + "\n\n")
        logger.info("Finished data sampling and added to report lines.")

        report_content = "".join(report_lines)
        logger.info(f"Final report content length: {len(report_content)}")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Post-process analysis report generated: {report_path}")
        return report_path, report_content

    except Exception as e:
        logger.error(f"Error generating post-process report: {e}", exc_info=True)
        return None, None


def generate_academic_report(output_dir: str, ai_model: str, **kwargs) -> None:
    """
    Generates a professional academic analysis summary by sending the existing report
    and a glossary of terms to an LLM.
    """
    try:
        logger.info(
            f"Starting generation of the academic analysis summary for model {ai_model}."
        )

        # 1. Read the existing report
        report_filename = kwargs.get(
            "report_filename", "baseline_condition_analysis_report.md"
        )
        report_path = os.path.join(output_dir, report_filename)
        if not os.path.exists(report_path):
            logger.error(
                f"Cannot generate academic summary: Original report '{report_path}' not found."
            )
            return
        with open(report_path, "r", encoding="utf-8") as f:
            original_report_content = f.read()

        # 2. Read the glossary
        glossary_path = kwargs.get("glossary_path", "sheets.csv")
        if not os.path.exists(glossary_path):
            logger.error(
                f"Cannot generate academic summary: Glossary file '{glossary_path}' not found."
            )
            return
        with open(glossary_path, "r", encoding="utf-8") as f:
            glossary_content = f.read()

        # 3. Check for API credentials
        load_dotenv()
        api_key = os.environ.get("API_KEY")
        base_url = os.environ.get("BASE_URL")

        if not all([api_key, base_url, ai_model]):
            logger.warning(
                "API_KEY, BASE_URL, or AI_MODEL not found. Skipping academic summary generation."
            )
            return

        # 4. Construct the prompt
        role_prompt = """**角色：** 您是一位在核聚变工程，特别是氚燃料循环领域，具有深厚学术背景的资深科学家。

**任务：** 您收到了由程序自动生成的初步分析报告和一份专业术语表。请您基于这两份文件，撰写一份更加专业、正式、符合学术发表标准的深度分析总结报告。
"""

        # Find all plots to instruct the LLM to include them
        all_plots = [f for f in os.listdir(output_dir) if f.endswith((".svg", ".png"))]
        plot_list_str = "\n".join([f"    *   `{plot}`" for plot in all_plots])
        instructions_prompt = f"""**指令：**

1.  **专业化语言：** 将初步报告中的模型参数/缩写（例如 `sds.I[1]`, `detailed_var`）替换为术语表中对应的“中文翻译”或“英文术语”。例如，应将“`sds`的库存”表述为“储存与输送系统 (SDS) 的氚库存量 (Tritium Inventory)”。
2.  **学术化重述：** 用严谨、客观的学术语言重新组织和阐述初步报告中的发现。避免使用“看起来”、“好像”等模糊词汇。
3.  **图表和表格的呈现与引用：**
    *   **显示图表：** 在报告的“结果与讨论”部分，您**必须**使用Markdown语法 `![图表标题](图表文件名)` 来**直接嵌入**和显示初步报告中包含的所有图表。可用的图表文件如下：
{plot_list_str}
    *   **引用图表：** 在正文中分析和讨论图表内容时，请使用“如图1所示...”等方式对图表进行编号和文字引用。
    *   **显示表格：** 当呈现数据时（例如，关键阶段的抽样数据或最终值），您**必须**使用Markdown的管道表格（pipe-table）格式来清晰地展示它们。您可以直接复用或重新格式化初步报告中的数据表格。
4.  **结构化报告：** 您的报告是关于一个**基准工况（Baseline Operating Condition）**的模拟分析。报告应包含以下部分：
    *   **摘要 (Abstract):** 简要概括本次**基准工况**模拟的目的、关键发现和核心结论。
    *   **引言 (Introduction):** 描述**基准工况**模拟的背景和目标，提及关键的输入参数。
    *   **结果与讨论 (Results and Discussion):** 这是报告的核心。分点详细论述：
        *   关键性能指标（如氚自持时间、倍增时间等，如果数据可用）的总体趋势。
        *   对关键转折点（例如氚库存的最低点）的物理意义进行深入分析。
        *   评估系统在模拟结束时的最终状态，并讨论氚在各子系统中的分布情况。
    *   **结论 (Conclusion):** 总结本次模拟研究得出的主要学术结论。
5.  **输出格式：** 请直接输出完整的学术分析报告正文，确保所有内容（包括图表和表格）都遵循正确的Markdown语法。

**输入文件：**
"""

        analysis_prompt = f"""
---
### 1. 初步分析报告 (`baseline_condition_analysis_report.md`)
---
{original_report_content}

---
### 2. 专业术语表 (`sheets.csv`)
---
{glossary_content}
"""

        # 5. Call the API
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                logger.info(
                    f"Sending request to OpenAI API for academic summary for model {ai_model} (Attempt {attempt + 1}/{max_retries})..."
                )

                full_text_prompt = "\n\n".join(
                    [role_prompt, instructions_prompt, analysis_prompt]
                )

                response = client.chat.completions.create(
                    model=ai_model,
                    messages=[{"role": "user", "content": full_text_prompt}],
                    max_tokens=4000,
                )
                academic_summary = response.choices[0].message.content

                # 6. Save the result
                sanitized_model_name = "".join(
                    c for c in ai_model if c.isalnum() or c in ("-", "_")
                ).rstrip()
                summary_filename = (
                    f"academic_analysis_summary_{sanitized_model_name}.md"
                )
                summary_path = os.path.join(output_dir, summary_filename)
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(academic_summary)

                logger.info(
                    f"Successfully generated academic analysis summary: {summary_path}"
                )
                return  # Exit after success

            except Exception as e:
                logger.error(
                    f"Error calling OpenAI API for academic summary on attempt {attempt + 1}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    logger.error(
                        f"Failed to generate academic summary for {ai_model} after {max_retries} attempts."
                    )
                    return  # Exit after all retries failed

    except Exception as e:
        logger.error(
            f"Error in generate_academic_report for model {ai_model}: {e}",
            exc_info=True,
        )


def baseline_analysis(results_df: pd.DataFrame, output_dir: str, **kwargs) -> None:
    """Generates baseline analysis plots and reports.

    Creates three outputs:
    1. A time-series plot with overall view and detailed zoom around turning point
    2. A bar chart showing final values of all variables, sorted
    3. An optional Markdown report with AI analysis (if 'ai' flag is True)

    Args:
        results_df: The combined DataFrame of simulation results.
        output_dir: The directory to save the plots and report.
        **kwargs: Additional parameters from config, including 'ai' flag, 'detailed_var',
            'glossary_path', and AI model settings.

    Note:
        Removes duplicate rows before processing. Creates bilingual plots (English and
        Chinese). If AI analysis enabled, requires API_KEY, BASE_URL, and AI_MODELS/AI_MODEL
        environment variables. Generates both initial LLM analysis and academic summary.
    """
    if "time" not in results_df.columns:
        logger.error("Plotting failed: 'time' column not found in results DataFrame.")
        return

    if "glossary_path" in kwargs:
        load_glossary(kwargs["glossary_path"])

    os.removedirs(output_dir) if os.path.exists(output_dir) else None
    p = Path(output_dir)
    output_dir = p.parent / "report"
    os.makedirs(output_dir, exist_ok=True)

    df = results_df.copy()
    # Remove duplicate rows before processing
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create a unified color map for all variables
    all_plot_columns = sorted([col for col in df.columns if col != "time"])
    colors = sns.color_palette("turbo", len(all_plot_columns))
    color_map = dict(zip(all_plot_columns, colors))

    # Add the color map to kwargs to pass it to the helper functions
    plot_kwargs = kwargs.copy()
    plot_kwargs["color_map"] = color_map

    # Generate the time-series plot with zoom
    _plot_time_series_with_zoom(df, output_dir, **plot_kwargs)

    # Generate the bar chart of final values
    _plot_final_values_bar_chart(df, output_dir, **plot_kwargs)

    # --- Report Generation and AI Analysis ---
    base_report_path, base_report_content = _generate_postprocess_report(
        df, output_dir, **kwargs
    )

    if base_report_path and kwargs.get("ai", False):
        load_dotenv()
        api_key = os.environ.get("API_KEY")
        base_url = os.environ.get("BASE_URL")

        # Prioritize AI_MODELS, fallback to AI_MODEL
        ai_models_str = os.environ.get("AI_MODELS")
        if not ai_models_str:
            ai_models_str = os.environ.get("AI_MODEL")

        if not api_key or not base_url or not ai_models_str:
            logger.warning(
                "API_KEY, BASE_URL, or AI_MODELS/AI_MODEL not found in environment variables. Skipping LLM analysis."
            )
            return

        ai_models = [model.strip() for model in ai_models_str.split(",")]

        for ai_model in ai_models:
            logger.info(f"Generating AI analysis for model: {ai_model}")

            sanitized_model_name = "".join(
                c for c in ai_model if c.isalnum() or c in ("-", "_")
            ).rstrip()

            model_report_filename = (
                f"analysis_report_baseline_condition_{sanitized_model_name}.md"
            )
            model_report_path = os.path.join(output_dir, model_report_filename)

            with open(model_report_path, "w", encoding="utf-8") as f:
                f.write(base_report_content)

            llm_analysis = _call_openai_for_postprocess_analysis(
                api_key=api_key,
                base_url=base_url,
                ai_model=ai_model,
                report_content=base_report_content,
                **kwargs,
            )

            if llm_analysis:
                with open(model_report_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n---\n\n# AI模型分析提示词 ({ai_model})\n\n")
                    f.write("```markdown\n")
                    f.write(llm_analysis)
                    f.write("\n```\n")
                logger.info(
                    f"Appended LLM analysis for model {ai_model} to {model_report_path}"
                )

                # --- ADDED: Second AI call for academic summary ---
                academic_kwargs = kwargs.copy()
                academic_kwargs["report_filename"] = model_report_filename
                generate_academic_report(
                    output_dir, ai_model=ai_model, **academic_kwargs
                )


# {
# #   "module": "tricys.postprocess.baseline_analysis",
# #   "function": "baseline_analysis",
# #   "params": {
# #        "detailed_var": "sds.I[1]",
# #        "glossary_path": "./sheets.csv"
# #    }
# #}
