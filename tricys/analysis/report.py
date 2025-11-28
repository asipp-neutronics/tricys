import json
import logging
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional

import openai
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def call_openai_analysis_api(
    case_name: str,
    df: pd.DataFrame,
    api_key: str,
    base_url: str,
    ai_model: str,
    independent_variable: str,
    report_content: str,
    original_config: dict,
    case_data: dict,
    reference_col_for_turning_point: str = None,
) -> Optional[str]:
    """Constructs a text-only prompt, calls the OpenAI API for analysis, and returns the result string.

    Args:
        case_name: Name of the analysis case.
        df: DataFrame containing summary data.
        api_key: OpenAI API key.
        base_url: Base URL for the OpenAI API.
        ai_model: Model name to use for analysis.
        independent_variable: Name of the independent variable.
        report_content: The report content to analyze.
        original_config: Original configuration dictionary.
        case_data: Case-specific data dictionary.
        reference_col_for_turning_point: Optional reference column for turning point analysis.

    Returns:
        The combined prompt and LLM analysis result, or None if failed.

    Note:
        Constructs dynamic prompts based on case configuration. Includes sections for
        global sensitivity analysis, interaction effects (if simulation parameters present),
        and dynamic process analysis (if reference column provided). Retries up to 3 times
        on failure with 5-second delays.
    """
    try:
        logger.info(f"Proceeding with LLM analysis for case {case_name}.")

        # 1. Construct the prompt for the API
        role_prompt = """**角色：** 你是一名聚变反应堆氚燃料循环领域的专家。

**任务：** 请**完全基于**下方提供的**两类数据表格**，对聚变堆燃料循环模型的**敏感性分析**结果进行深度解读。
"""

        analysis_prompt = f"""
**分析数据：**(注意：分析中不可使用任何图表信息，所有结论必须源于数据表格。)

{report_content}
"""

        # --- Dynamic Prompt Construction ---

        # 1. Detect analysis scenario
        has_sim_params = bool(case_data.get("simulation_parameters"))

        # 2. Build prompt sections dynamically
        prompt_sections = []

        # Section 1: Global Sensitivity Analysis
        global_sensitivity_points = [
            "1.  **全局敏感性分析 (参考“性能指标总表”) :**",
            "    *   分析性能指标总表（ `Startup_Inventory`, `Doubling_Time` 以及以 `Required_` 开头的求解指标等）呈现出怎样的**总体趋势**？请进行量化描述。",
            f"    *   如果存在多个性能指标，分析哪个性能指标对独立变量 `{independent_variable}` 的变化最为敏感？哪个最不敏感？\n",
        ]

        # Interaction effect analysis, with refined description
        if has_sim_params:
            param_names_list = []
            for p in case_data["simulation_parameters"].keys():
                if p == "Required_TBR":
                    label = "`Required_TBR约束值 (hour)`"
                else:
                    label = f"`{p}`"
                param_names_list.append(label)
            param_names = ", ".join(param_names_list)

            interaction_text = (
                f"2.  **交互效应分析：** 本次分析包含了多变量的交互效应。请分析独立变量 `{independent_variable}` "
                f"与背景扫描参数 ({param_names}) 之间的交互作用对各项性能指标的影响。"
                "请注意，独立变量或背景扫描参数中，可能包含常规的模型参数，也可能包含为满足特定性能目标（限制倍增时间Double_Time达到倍增）而求解出的特殊变量（约束限制变量Double_Time）。"
                "请讨论在不同的变量组合下，性能指标的敏感性有何不同？是否存在显著的交互效应？"
            )
            global_sensitivity_points.append(interaction_text)

        prompt_sections.append("\n".join(global_sensitivity_points))

        # Section 2: Dynamic Process Analysis
        if reference_col_for_turning_point:
            dynamic_process_points = [
                "3.  **动态过程分析 (参考“关键动态数据切片：过程数据”) :**",
                "    *   观察过程数据切片：系统在“初始阶段”和“结束阶段”的行为有何不同？",
                f"    *   以 `{reference_col_for_turning_point}` 为参考，其“转折点阶段”的数据揭示了什么物理过程？（例如，它是否是氚库存由消耗转为净增长的关键时刻？）",
            ]
            prompt_sections.append("\n".join(dynamic_process_points))

        # Section 3: Overall Conclusion (renumbered from 4)
        conclusion_points = ["3.  **综合结论：**"]
        conclusion_intro = "结合所有分析（包括主趋势"
        if has_sim_params:
            conclusion_intro += "、背景参数交互效应"
        conclusion_intro += "），"

        conclusion_points.append(
            conclusion_intro
            + f"总结在不同的运行场景下，调整 `{independent_variable}` 对整个氚燃料循环系统的综合影响和潜在的利弊权衡。"
        )
        conclusion_points.append(
            "    *   基于这些发现，可以得出哪些关于系统设计或运行优化的初步建议？"
        )
        prompt_sections.append("\n".join(conclusion_points))

        # Assemble the final prompt
        points_prompt = "\n\n".join(prompt_sections)
        points_prompt = (
            "\n**分析要点 (必须严格依据数据表格作答)：**\n\n" + points_prompt
        )

        # 2. Call API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                logger.info(
                    f"Sending request to OpenAI API for case {case_name} (Attempt {attempt + 1}/{max_retries})..."
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

                logger.info(f"LLM analysis successful for case {case_name}.")
                return (
                    role_prompt
                    + points_prompt
                    + "\n```\n\n"
                    + "\n\n---\n\n# AI模型分析结果\n\n"
                    + analysis_result
                )  # Return the result string

            except Exception as e:
                logger.error(f"Error calling OpenAI API on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    logger.error(
                        f"Failed to call OpenAI API after {max_retries} attempts."
                    )
                    return None  # Return None on failure

    except Exception as e:
        logger.error(
            f"Error in call_openai_analysis_api for case {case_name}: {e}",
            exc_info=True,
        )
        return None


def generate_sensitivity_academic_report(
    case_name: str,
    case_workspace: str,
    independent_variable: str,
    original_config: dict,
    case_data: dict,
    ai_model: str,
    report_path: str,
) -> None:
    """Generates a professional academic analysis summary for a sensitivity analysis case.

    Sends the existing report and a glossary of terms to an LLM for academic formatting.

    Args:
        case_name: Name of the analysis case.
        case_workspace: Path to the case workspace directory.
        independent_variable: Name of the independent variable.
        original_config: Original configuration dictionary.
        case_data: Case-specific data dictionary.
        ai_model: Model name to use for generating the report.
        report_path: Path to the existing report file.

    Note:
        Requires report file and glossary file to exist. Loads API credentials from
        environment variables. Generates academic report with proper structure including
        title, abstract, introduction, methodology, results & discussion, and conclusion.
        Retries up to 3 times on API failure. Saves result to academic_report_{case_name}_{model}.md.
    """
    try:
        logger.info(
            f"Starting generation of the academic analysis summary for case {case_name} with model {ai_model}."
        )

        # 1. Read the existing report
        results_dir = os.path.join(case_workspace, "results")
        report_filename = os.path.basename(report_path)

        if not os.path.exists(report_path):
            logger.error(
                f"Cannot generate academic summary: Original report '{report_path}' not found."
            )
            return
        with open(report_path, "r", encoding="utf-8") as f:
            original_report_content = f.read()

        # 2. Read the glossary
        glossary_path = original_config.get("sensitivity_analysis", {}).get(
            "glossary_path", "./sheets.csv"
        )
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

**任务：** 您收到了一个关于**敏感性分析**的程序自动生成的初步报告和一份专业术语表。请您基于这两份文件，撰写一份更加专业、正式、符合学术发表标准的深度分析总结报告。
"""

        # Extract relevant details from case_data for the prompt
        sampling_range = case_data.get("independent_variable_sampling", {})
        simulation_parameters = case_data.get("simulation_parameters", {})
        dependent_variables = case_data.get("dependent_variables", [])

        simulation_params_str = ""
        if simulation_parameters:
            params_list = []
            # Access metric definitions from the original config
            metrics_definitions = original_config.get("sensitivity_analysis", {}).get(
                "metrics_definition", {}
            )

            for k, v in simulation_parameters.items():
                # Check if the parameter is a 'Required_' metric and has configurations defined
                if (
                    k.startswith("Required_")
                    and k in metrics_definitions
                    and "configurations" in metrics_definitions[k]
                ):
                    try:
                        metric_configs = metrics_definitions[k]["configurations"]
                        # Look up the metric_max_value for each configuration key in the scan list `v`
                        actual_values = [
                            metric_configs.get(conf_name, {}).get(
                                "metric_max_value", conf_name
                            )
                            for conf_name in v
                        ]
                        params_list.append(f"`{k}` (约束扫描值): {actual_values}")
                    except Exception:
                        # If lookup fails for any reason, fall back to the original representation
                        params_list.append(f"`{k}`: {v}")
                else:
                    # For regular parameters
                    params_list.append(f"`{k}`: {v}")

            simulation_params_str = (
                "\n        *   **背景扫描参数 (Simulation Parameters):** "
                + ", ".join(params_list)
            )

        dependent_vars_str = ""
        if dependent_variables:
            dependent_vars_str = (
                "\n        *   **因变量 (Dependent Variables):** "
                + ", ".join([f"`{v}`" for v in dependent_variables])
            )

        # Find all plots to instruct the LLM to include them, prioritizing Chinese versions
        all_files = [f for f in os.listdir(results_dir) if f.endswith((".svg", ".png"))]

        plot_map = {}
        # Handle SVGs, prioritizing _zh versions
        svg_plots = sorted([f for f in all_files if f.endswith(".svg")], reverse=True)
        for plot in svg_plots:
            base_name = plot.replace("_zh.svg", ".svg")
            if base_name not in plot_map:
                plot_map[base_name] = plot

        # Add PNGs (which are not bilingual)
        png_plots = [f for f in all_files if f.endswith(".png")]
        for plot in png_plots:
            plot_map[plot] = plot  # Use plot name as key for uniqueness

        all_plots = list(plot_map.values())
        plot_list_str = "\n".join([f"    *   `{plot}`" for plot in all_plots])

        # Dynamically build the "Results and Discussion" section for the prompt
        results_and_discussion_points = []

        # 1. Main Effect Analysis (always included)
        main_effect_text = (
            f"           *   **主效应分析：** 详细分析独立变量 **`{independent_variable}`** 的变化对主要性能指标（如 `Startup_Inventory`, `Doubling_Time` 等）的总体影响趋势。"
            "评估不同指标对自变量变化的敏感度，并讨论指标间的**权衡关系 (Trade-offs)**。"
        )
        results_and_discussion_points.append(main_effect_text)

        # 2. Interaction Effect Analysis (conditional)
        if simulation_parameters:
            interaction_text = (
                f"           *   **交互效应分析：** 深入探讨独立变量与背景参数间的**交互效应**。"
                "背景参数可能包含常规的模型参数，也可能包含约束相关的变量（例如 `Required_TBR`）。"
                f"请阐述在不同的背景参数组合下，`{independent_variable}` 对性能指标的敏感性是否存在显著差异（例如，是被放大还是减弱）。"
                "请特别关注当独立变量与约束类背景参数交互时，对系统性能和达成工程目标的影响。"
            )
            results_and_discussion_points.append(interaction_text)

        # 3. Dynamic Behavior Analysis (conditional)
        if "关键动态数据切片" in original_report_content:
            dynamic_text = (
                f"           *   **动态行为分析：** 解读系统在“初始”、“转折点”和“结束”阶段的行为变化。"
                f"分析 **`{independent_variable}`** 的变化如何影响系统的动态过程，如达到平衡的时间、库存的转折点等。"
            )
            results_and_discussion_points.append(dynamic_text)

        results_and_discussion_str = "\n".join(results_and_discussion_points)

        # Dynamically create the title instruction
        if simulation_parameters:
            title_instruction_text = "请在标题中明确指出，本次分析是关于“独立变量”与“背景扫描参数”的【交互敏感性分析】。"
        else:
            title_instruction_text = (
                "请在标题中明确指出，本次分析是关于“独立变量”的【敏感性分析】。"
            )

        instructions_prompt = f"""**指令：**

1.  **专业化语言：** 将初步报告中的模型参数/缩写（例如 `sds.I[1]`, `Startup_Inventory`）替换为术语表中对应的“中文翻译”或“英文术语”。例如，应将“`sds`的库存”表述为“储存与输送系统 (SDS) 的氚库存量 (Tritium Inventory)”。
2.  **学术化重述：** 用严谨、客观的学术语言重新组织和阐述初步报告中的发现。避免使用“看起来”、“好像”等模糊词汇。
3.  **图表和表格的呈现与引用：**
    *   **显示图表：** 在报告的“结果与讨论”部分，您**必须**使用Markdown语法 `![图表标题](图表文件名)` 来**直接嵌入**和显示初步报告中包含的所有图表。可用的图表文件如下：
{plot_list_str}
    *   **引用图表：** 在正文中分析和讨论图表内容时，请使用“如图1所示...”等方式对图表进行编号和文字引用。
    *   **显示表格：** 当呈现数据时（例如，性能指标总表或关键动态数据切片），您**必须**使用Markdown的管道表格（pipe-table）格式来清晰地展示它们。您可以直接复用或重新格式化初步报告中的数据表格。
4.  **结构化报告：** 您的报告是关于一项**敏感性分析**。报告应包含以下部分：
    *   **标题 (Title):** {title_instruction_text}
    *   **摘要 (Abstract):** 简要概括本次敏感性研究的目的，明确指明独立变量是 **`{independent_variable}`** 以及背景扫描参数，总结其对哪些关键性能指标（如启动库存、增殖时间等）影响最显著，并陈述核心结论。
    *   **引言 (Introduction):** 描述进行这项关于 **`{independent_variable}`** 的敏感性分析的背景和重要性。阐述研究目标，即量化评估 **`{independent_variable}`** 的变化对氚燃料循环系统性能的影响。
        *   **独立变量采样 (Independent Variable Sampling):** 本次分析中，独立变量 `{independent_variable}` 扫描范围为 `{sampling_range}`。
{simulation_params_str}{dependent_vars_str}
    *   **方法 (Methodology):** 简要说明分析方法，包括提及 **`{independent_variable}`** 的扫描范围和被评估的关键性能指标。
    *   **结果与讨论 (Results and Discussion):** 这是报告的核心。请结合所有图表和数据表格，并根据分析内容，组织分点详细论述：
{results_and_discussion_str}
    *   **结论 (Conclusion):** 总结本次敏感性分析得出的主要学术结论，并对反应堆设计或未来运行策略提出具体建议。
5.  **输出格式：** 请直接输出完整的学术分析报告正文，确保所有内容（包括图表和表格）都遵循正确的Markdown语法。

**输入文件：**
"""

        analysis_prompt = f"""
---
### 1. 初步分析报告 (`{report_filename}`)
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
                    f"Sending request to OpenAI API for academic summary for case {case_name} with model {ai_model} (Attempt {attempt + 1}/{max_retries})..."
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
                    f"academic_report_{case_name}_{sanitized_model_name}.md"
                )
                summary_path = os.path.join(results_dir, summary_filename)
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
                        f"Failed to generate academic summary for {case_name} after {max_retries} attempts."
                    )
                    return  # Exit after all retries failed

    except Exception as e:
        logger.error(
            f"Error in generate_sensitivity_academic_report for case {case_name}: {e}",
            exc_info=True,
        )


def generate_prompt_templates(
    case_configs: List[Dict[str, Any]], original_config: Dict[str, Any]
) -> None:
    """Generate detailed Markdown analysis reports for each analysis case.

    Args:
        case_configs: List of case configuration dictionaries containing case data and workspace info.
        original_config: Original configuration dictionary with sensitivity analysis settings.

    Note:
        Skips SALib cases (those with analyzer.method defined). For each case, generates
        a detailed Markdown report including configuration details, optimization configs,
        time-series plots, performance metric plots, and data tables. Supports AI-enhanced
        reporting if API credentials are available. Creates bilingual plots prioritizing
        Chinese versions (_zh suffix).
    """

    def _find_unit_config(var_name: str, unit_map: dict) -> dict | None:
        """
        Finds the unit configuration for a variable name from the unit_map.
        1. Checks for an exact match.
        2. Checks if the last part of a dot-separated name matches.
        3. Checks for a simple substring containment as a fallback, matching longest keys first.
        """
        if not unit_map or not var_name:
            return None
        if var_name in unit_map:
            return unit_map[var_name]
        components = var_name.split(".")
        if len(components) > 1 and components[-1] in unit_map:
            return unit_map[components[-1]]
        for key in sorted(unit_map.keys(), key=len, reverse=True):
            if key in var_name:
                return unit_map[key]
        return None

    def _format_label(label: str) -> str:
        """Formats a label for display, replacing underscores/dots with spaces and capitalizing each word."""
        if not isinstance(label, str):
            return label
        label = label.replace("_", " ")
        label = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", label)
        return label  # .title()

    try:
        sensitivity_analysis_config = original_config.get("sensitivity_analysis", {})
        unit_map = sensitivity_analysis_config.get("unit_map", {})

        for case_info in case_configs:
            case_data = case_info["case_data"]

            if "analyzer" in case_data and case_data.get("analyzer", {}).get("method"):
                logger.info(
                    f"Skipping default report generation for SALib case: {case_data.get('name', 'Unknown')}"
                )
                continue

            case_workspace = case_info["workspace"]
            case_name = case_data.get("name", f"Case{case_info['index']+1}")

            case_results_dir = os.path.join(case_workspace, "results")
            if not os.path.exists(case_results_dir):
                continue

            summary_csv_path = os.path.join(
                case_results_dir, "sensitivity_analysis_summary.csv"
            )
            sweep_csv_path = os.path.join(case_results_dir, "sweep_results.csv")

            if not os.path.exists(summary_csv_path):
                logger.warning(
                    f"summary_csv not found for case {case_name}, skipping report generation."
                )
                continue

            summary_df = pd.read_csv(summary_csv_path)
            independent_variable = case_data.get("independent_variable", "燃烧率")

            # Use a dictionary to ensure we only get one version of each plot, prioritizing Chinese
            all_plots_all_langs = [
                f for f in os.listdir(case_results_dir) if f.endswith(".svg")
            ]
            plot_map = {}
            for plot in sorted(
                all_plots_all_langs, reverse=True
            ):  # Process _zh.svg first
                base_name = plot.replace("_zh.svg", ".svg")
                if base_name not in plot_map:
                    plot_map[base_name] = plot

            all_plots = list(plot_map.values())
            sweep_plots = [f for f in all_plots if f.startswith("sweep_")]
            combined_plots = [f for f in all_plots if f.startswith("combined_")]
            multi_metric_plots = [
                f
                for f in all_plots
                if f.startswith("multi_") and f.endswith("_analysis_by_param.svg")
            ]
            all_individual_plots = [
                f
                for f in all_plots
                if not f.startswith("sweep_")
                and not f.startswith("combined_")
                and not f.startswith("multi_")
            ]
            required_individual_plots = [
                f for f in all_individual_plots if f.startswith("line_Required_")
            ]
            standard_individual_plots = [
                f for f in all_individual_plots if not f.startswith("line_Required_")
            ]

            # --- Markdown Generation (with dynamic title) ---
            sim_params = case_data.get("simulation_parameters")
            if sim_params:
                main_var_label = _format_label(independent_variable)

                other_vars_labels_list = []
                for p in sim_params.keys():
                    if p == "Required_TBR":
                        label = "Required_TBR约束值"
                    else:
                        label = _format_label(p)
                    other_vars_labels_list.append(label)
                other_vars_labels = "、".join(other_vars_labels_list)

                report_title = (
                    f"# {main_var_label} 与 {other_vars_labels} 交互敏感性分析报告\n\n"
                )
            else:
                report_title = (
                    f"# {_format_label(independent_variable)} 敏感性分析报告\n\n"
                )

            prompt_lines = [
                report_title,
                f"生成时间: {pd.Timestamp.now()}\n\n",
            ]

            config_details_lines = [
                "## 分析案例配置详情\n\n",
                "本分析案例的具体配置如下，这决定了仿真的扫描方式和分析的重点：\n\n",
                "| 配置项 | 值 | 说明 |",
                "| :--- | :--- | :--- |",
            ]

            def format_for_md(value):
                return f"`{json.dumps(value, ensure_ascii=False)}`".replace("|", "\\|")

            config_details_lines.extend(
                [
                    f"| **`name`** | {format_for_md(case_name)} | 本次分析案例的名称。 |",
                    f"| **`independent_variable`** | {format_for_md(independent_variable)} | 独立扫描变量，即本次分析中主要改变的参数。 |",
                    f"| **`independent_variable_sampling`** | {format_for_md(case_data.get('independent_variable_sampling'))} | 独立变量的采样方法和范围。 |",
                ]
            )
            if "default_independent_values" in case_data:
                config_details_lines.append(
                    f"| **`default_independent_values`** | {format_for_md(case_data['default_independent_values'])} | 独立扫描变量在模型中的原始默认值。 |"
                )
            if (
                "simulation_parameters" in case_data
                and case_data["simulation_parameters"]
            ):
                config_details_lines.append(
                    f"| **`simulation_parameters`** | {format_for_md(case_data['simulation_parameters'])} | 背景扫描参数，与独立变量组合形成多维扫描。 |"
                )
            if (
                "default_simulation_values" in case_data
                and case_data["default_simulation_values"]
            ):
                config_details_lines.append(
                    f"| **`default_simulation_values`** | {format_for_md(case_data['default_simulation_values'])} | 背景扫描参数在模型中的原始默认值。 |"
                )
            config_details_lines.append(
                f"| **`dependent_variables`** | {format_for_md(case_data.get('dependent_variables'))} | 因变量，即我们关心的、随自变量变化的性能指标。 |"
            )
            config_details_lines.append("\n")
            prompt_lines.extend(config_details_lines)

            optimization_metrics = [
                v
                for v in case_data.get("dependent_variables", [])
                if v.startswith("Required_")
            ]
            if optimization_metrics:
                for metric_name in optimization_metrics:
                    metric_config = (
                        original_config.get("sensitivity_analysis", {})
                        .get("metrics_definition", {})
                        .get(metric_name)
                    )
                    if metric_config:
                        details_lines = [
                            f"## “{metric_name}”优化配置\n",
                            f"当“{metric_name}”作为因变量时，系统会启用一个二分查找算法来寻找满足特定性能指标的最小`{metric_config.get('parameter_to_optimize', 'N/A')}`值。以下是本次优化任务的具体配置：\n\n",
                            "| 配置项 | 值 | 说明 |",
                            "| :--- | :--- | :--- |",
                        ]
                        config_map = {
                            "source_column": "限制条件的数据源列。",
                            "parameter_to_optimize": "优化的目标参数。",
                            "search_range": "参数的搜索范围。",
                            "tolerance": "搜索的收敛精度。",
                            "max_iterations": "最大迭代次数。",
                            "metric_name": "限制条件的性能指标。",
                            "metric_max_value": "限制条件满足的上限值。（hour）",
                        }
                        for key, description in config_map.items():
                            if key in metric_config:
                                value = metric_config[key]
                                details_lines.append(
                                    f"| **`{key}`** | {format_for_md(value)} | {description} |"
                                )

                            metric_config_sim = case_data.get(
                                "simulation_parameters", {}
                            ).get("Required_TBR", {})
                            if metric_config_sim and key in metric_config_sim:
                                value = metric_config_sim[key]
                                details_lines.append(
                                    f"| **`{key} (from simulation_parameters)`** | {format_for_md(value)} | {description} |"
                                )

                        details_lines.append("\n")
                        prompt_lines.extend(details_lines)

            for plot in sweep_plots:
                prompt_lines.extend(
                    [
                        "## SDS Inventory 的时间曲线图:\n\n",
                        f"![SDS Inventory 的时间曲线图]({plot})\n\n",
                    ]
                )
                if "default_simulation_values" in case_data and case_data.get(
                    "default_simulation_values"
                ):
                    default_values_str = json.dumps(
                        case_data["default_simulation_values"],
                        ensure_ascii=False,
                        indent=4,
                    )
                    note = (
                        "**筛选说明**：当存在多个背景扫描参数 (`simulation_parameters`) 时，为突出重点，上图默认仅显示与原始默认值 "
                        f"(`default_simulation_values`) 相匹配的基准情景曲线。本次分析中用于筛选的默认值为：\n\n"
                        f"```json\n{default_values_str}\n```\n\n"
                        "此方法有助于在固定的基准条件下，清晰地观察独立变量变化带来的影响。\n"
                    )
                    prompt_lines.append(note)

            if combined_plots:
                for plot in combined_plots:
                    title = "性能指标趋势曲线图"
                    prompt_lines.extend([f"## {title}\n\n", f"![{title}]({plot})\n"])
            elif standard_individual_plots:
                prompt_lines.append("## 性能指标分析图\n\n")
                for plot in standard_individual_plots:
                    title = _format_label(
                        os.path.splitext(plot)[0].replace("line_", "")
                    )
                    prompt_lines.extend([f"### {title}\n", f"![{title}]({plot})\n\n"])

            if multi_metric_plots or required_individual_plots:
                prompt_lines.append("## 约束求解性能指标分析图\n\n")
                for plot_file in multi_metric_plots:
                    try:
                        base_metric_name = plot_file.replace("multi_", "").replace(
                            "_analysis_by_param.svg", ""
                        )
                        friendly_name = _format_label(base_metric_name)
                    except Exception:
                        friendly_name = "Optimization"
                    prompt_lines.extend(
                        [
                            f"### 不同约束值下的“{friendly_name}”分析 (按参数分组)\n",
                            f"下图展示了“{friendly_name}”指标随独立变量变化的趋势。每个子图对应一组特定的背景扫描参数组合，子图内的每条曲线代表一个具体的约束值。\n\n",
                            f"![不同约束值下的{friendly_name}分析]({plot_file})\n\n",
                        ]
                    )
                for plot_file in required_individual_plots:
                    title = _format_label(
                        os.path.splitext(plot_file)[0].replace("line_", "")
                    )
                    prompt_lines.extend(
                        [f"### {title}\n", f"![{title}]({plot_file})\n\n"]
                    )

            def _format_df_to_md(
                sub_df: pd.DataFrame,
                ind_var: str,
                case_data: dict,
                current_unit_map: dict,
            ) -> str:
                if sub_df.empty:
                    return "无数据。"
                all_markdown_lines = []
                all_cols = sub_df.columns.tolist()
                if ind_var in all_cols:
                    all_cols.remove(ind_var)
                standard_cols = [
                    c
                    for c in all_cols
                    if not (c.startswith("Required_") or "_for_Required_" in c)
                ]
                required_groups = {}
                required_base_names = [
                    v
                    for v in case_data.get("dependent_variables", [])
                    if v.startswith("Required_")
                ]
                for base_name in required_base_names:
                    group_cols = []
                    # pattern = re.compile(f"_for_{re.escape(base_name)}(?:\\(.*\\))?$")
                    for col in all_cols:
                        if col == base_name or col.startswith(base_name + "("):
                            group_cols.append(col)
                    if group_cols:
                        required_groups[base_name] = group_cols

                def _format_slice_to_md(df_slice: pd.DataFrame, umap: dict) -> str:
                    if df_slice.empty:
                        return ""
                    df_formatted = df_slice.copy()
                    new_columns = {}
                    for col_name in df_formatted.columns:
                        unit_config = _find_unit_config(col_name, umap)
                        new_col_name = _format_label(col_name)
                        if unit_config:
                            unit = unit_config.get("unit")
                            factor = unit_config.get("conversion_factor")
                            if factor and pd.api.types.is_numeric_dtype(
                                df_formatted[col_name]
                            ):
                                df_formatted[col_name] = df_formatted[col_name] / float(
                                    factor
                                )
                            if unit:
                                new_col_name = f"{new_col_name} ({unit})"
                        new_columns[col_name] = new_col_name
                    df_formatted.rename(columns=new_columns, inplace=True)
                    format_map = {}
                    for original_col_name in df_slice.columns:
                        if original_col_name.startswith("Required_"):
                            format_map[new_columns[original_col_name]] = "{:.4f}"
                    default_format = "{:.2f}"
                    for col in df_formatted.columns:
                        if pd.api.types.is_numeric_dtype(df_formatted[col]):
                            formatter = format_map.get(col, default_format)
                            df_formatted[col] = df_formatted[col].apply(
                                lambda x: formatter.format(x) if pd.notnull(x) else x
                            )
                    return df_formatted.to_markdown(index=False)

                if standard_cols:
                    all_markdown_lines.append("##### 性能指标\n")
                    std_df_slice = sub_df[[ind_var] + sorted(standard_cols)]
                    all_markdown_lines.append(
                        _format_slice_to_md(std_df_slice, current_unit_map)
                    )
                    all_markdown_lines.append("\n")
                if required_groups:
                    for base_name, cols in required_groups.items():
                        existing_cols = [c for c in cols if c in sub_df.columns]
                        if not existing_cols:
                            continue

                        all_markdown_lines.append(
                            f"##### “{_format_label(base_name)}” 相关数据\n"
                        )
                        req_df_slice = sub_df[[ind_var] + sorted(existing_cols)]

                        try:
                            # --- PIVOT LOGIC to transform data from wide to long format ---
                            # e.g., from [A, B(v1), B(v2)] to [A, new_col, B]

                            # Columns to unpivot, e.g., ['Required_TBR(7.0)', 'Required_TBR(10.0)']
                            value_vars = [
                                c
                                for c in req_df_slice.columns
                                if c.startswith(base_name)
                                and "(" in c
                                and c.endswith(")")
                            ]

                            # If no columns are in the format B(v), pivot is not applicable.
                            if not value_vars:
                                all_markdown_lines.append(
                                    _format_slice_to_md(req_df_slice, current_unit_map)
                                )
                                all_markdown_lines.append("\n")
                                continue

                            # Melt the dataframe from wide to long format
                            melted_df = req_df_slice.melt(
                                id_vars=[ind_var],
                                value_vars=value_vars,
                                var_name="variable_col",
                                value_name=base_name,
                            )

                            # Determine the name for the new column from config (e.g., 'Doubling_Time')
                            new_col_name = "Constraint"  # Default name
                            metric_def = case_data.get("simulation_parameters").get(
                                "Required_TBR"
                            )
                            if metric_def and metric_def.get("metric_name"):
                                new_col_name = "Constraint " + metric_def["metric_name"]

                            # Extract constraint value from old column name, e.g., '7.0' from 'Required_TBR(7.0)'
                            pattern_str = f"{re.escape(base_name)}\\((.*)\\)"
                            melted_df[new_col_name] = melted_df[
                                "variable_col"
                            ].str.extract(pat=pattern_str)

                            # Create the final dataframe with the desired columns: [A, new_col, B]
                            final_df = melted_df[
                                [ind_var, new_col_name, base_name]
                            ].copy()
                            final_df.dropna(subset=[base_name], inplace=True)

                            all_markdown_lines.append(final_df.to_markdown(index=False))
                            all_markdown_lines.append("\n")

                        except Exception as e:
                            logger.warning(
                                f"Could not pivot data for '{base_name}', displaying in wide format. Error: {e}"
                            )
                            all_markdown_lines.append(
                                _format_slice_to_md(req_df_slice, current_unit_map)
                            )
                            all_markdown_lines.append("\n")
                return "\n".join(all_markdown_lines)

            reference_col_for_turning_point = None
            if case_data.get("sweep_time") and os.path.exists(sweep_csv_path):
                try:
                    logger.info("Loading sweep_results.csv for dynamic slicing.")
                    sweep_df = pd.read_csv(sweep_csv_path)
                    if "time" in sweep_df.columns and len(sweep_df.columns) > 1:
                        reference_col_for_turning_point = sweep_df.columns[
                            len(sweep_df.columns) // 2
                        ]
                    if reference_col_for_turning_point:
                        data_to_slice_df = sweep_df.copy()
                        data_to_slice_df.reset_index(drop=True, inplace=True)
                        if not data_to_slice_df.empty:
                            prompt_lines.append("## 关键动态数据切片：过程数据\n\n")
                            prompt_lines.append(
                                f"下表展示了过程数据中，以 `{reference_col_for_turning_point}` 为参考变量，在关键阶段的数据切片。**注意：下表中的默认单位为：时间(h), 库存(g), 功率(MW)。**\n\n"
                            )
                            base_var_name = reference_col_for_turning_point.split("&")[
                                0
                            ]
                            cols_to_rename = [
                                c for c in data_to_slice_df.columns if c != "time"
                            ]
                            rename_map = {
                                col: f"C{i+1}" for i, col in enumerate(cols_to_rename)
                            }
                            legend_lines = [
                                "**表格图例说明**：",
                                "| 简称 | 参数组合 |",
                                "| :--- | :--- |",
                            ]
                            for original_name, abbr in rename_map.items():
                                param_parts = original_name.split("&", 1)
                                param_str = (
                                    param_parts[1] if len(param_parts) > 1 else "无"
                                )
                                param_str_formatted = (
                                    "`" + "`, `".join(param_str.split("&")) + "`"
                                )
                                legend_lines.append(
                                    f"| **{abbr}** | {param_str_formatted} |"
                                )
                            base_var_info = f"**注**：表格中所有简称列（C1, C2, ...）的数据均代表变量 `{base_var_name}` 在不同参数组合下的值。\n"
                            legend_md = base_var_info + "\n".join(legend_lines) + "\n\n"
                            prompt_lines.append(legend_md)
                            primary_y_var = reference_col_for_turning_point
                            min_idx = -1
                            if primary_y_var in data_to_slice_df.columns:
                                y_data = data_to_slice_df[primary_y_var]
                                if not y_data.empty:
                                    min_idx = y_data.idxmin()
                            num_points, interval = 20, 2
                            window_size = (num_points - 1) * interval + 1
                            start_data = data_to_slice_df.iloc[:window_size:interval]
                            end_data = data_to_slice_df.iloc[-(window_size)::interval]
                            prompt_lines.append(
                                f"### 1. 初始阶段 (前 {num_points} 个数据点, 间隔 {interval})\n"
                            )
                            prompt_lines.append(
                                start_data.rename(columns=rename_map).to_markdown(
                                    index=False
                                )
                                + "\n\n"
                            )
                            if min_idx != -1:
                                window_radius_indices = (num_points // 2) * interval
                                start_idx = max(0, min_idx - window_radius_indices)
                                end_idx = min(
                                    len(data_to_slice_df),
                                    min_idx + window_radius_indices,
                                )
                                turning_point_data = data_to_slice_df.iloc[
                                    start_idx:end_idx:interval
                                ]
                                prompt_lines.append(
                                    f"### 2. 转折点阶段 (围绕 '{primary_y_var}' 最小值)\n"
                                )
                                prompt_lines.append(
                                    turning_point_data.rename(
                                        columns=rename_map
                                    ).to_markdown(index=False)
                                    + "\n\n"
                                )
                            prompt_lines.append(
                                f"### 3. 结束阶段 (后 {num_points} 个数据点, 间隔 {interval})\n"
                            )
                            prompt_lines.append(
                                end_data.rename(columns=rename_map).to_markdown(
                                    index=False
                                )
                                + "\n\n"
                            )
                except Exception as e:
                    logger.warning(
                        f"Could not generate dynamic data slices for case {case_name}: {e}"
                    )

            grouping_vars = list(case_data.get("default_simulation_values", {}).keys())
            if not grouping_vars:
                prompt_lines.append("## 性能指标总表\n\n")
                prompt_lines.append(
                    _format_df_to_md(
                        summary_df, independent_variable, case_data, unit_map
                    )
                )
            else:
                prompt_lines.append(
                    f"## 性能指标总表 (分组: `{'`, `'.join(grouping_vars)}`)\n\n"
                )
                groups = dict(list(summary_df.groupby(grouping_vars)))
                default_values = case_data.get("default_simulation_values")
                default_group_key = None
                if default_values:
                    try:
                        default_group_key = tuple(
                            default_values[key] for key in grouping_vars
                        )
                    except KeyError:
                        logger.warning(
                            "Mismatch between default_simulation_values and grouping_vars. Cannot find default group."
                        )
                        default_group_key = None
                if default_group_key and default_group_key in groups:
                    default_group_df = groups.pop(default_group_key)
                    header = " & ".join(
                        f"`{var}={val}`"
                        for var, val in zip(grouping_vars, default_group_key)
                    )
                    prompt_lines.append(f"#### 数据子表 (原始默认值: {header})\n")
                    sub_df_to_format = default_group_df.drop(
                        columns=grouping_vars, errors="ignore"
                    )
                    prompt_lines.append(
                        _format_df_to_md(
                            sub_df_to_format, independent_variable, case_data, unit_map
                        )
                    )
                    prompt_lines.append("\n---\n")
                if groups:
                    prompt_lines.append("> 其他参数组合下的数据子表：\n")
                for group_name, group_df in groups.items():
                    header = (
                        " & ".join(
                            f"`{var}={val}`"
                            for var, val in zip(grouping_vars, group_name)
                        )
                        if isinstance(group_name, tuple)
                        else f"`{grouping_vars[0]}={group_name}`"
                    )
                    prompt_lines.append(f"#### 数据子表 (当 {header} 时)\n")
                    sub_df_to_format = group_df.drop(
                        columns=grouping_vars, errors="ignore"
                    )
                    prompt_lines.append(
                        _format_df_to_md(
                            sub_df_to_format, independent_variable, case_data, unit_map
                        )
                    )
                    prompt_lines.append("\n")

            base_report_content = "\n".join(prompt_lines)

            # --- AI Analysis and Report Writing ---
            if not case_data.get("ai", False):
                # AI is off: write a single, simple report
                report_path = os.path.join(
                    case_results_dir, f"analysis_report_{case_name}.md"
                )
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(base_report_content)
                logger.info(
                    f"Detailed analysis report generated for {case_name}: {report_path}"
                )
                continue  # Go to next case

            # AI is ON: go into multi-model logic
            load_dotenv()
            api_key = os.environ.get("API_KEY")
            base_url = os.environ.get("BASE_URL")

            ai_models_str = os.environ.get("AI_MODELS")
            if not ai_models_str:
                ai_models_str = os.environ.get("AI_MODEL")

            if not all((api_key, base_url, ai_models_str)):
                logger.warning(
                    "API_KEY, BASE_URL, or AI_MODELS/AI_MODEL not found in environment variables. Skipping LLM analysis."
                )
                # Also write the base report here so something is generated
                report_path = os.path.join(
                    case_results_dir, f"analysis_report_{case_name}.md"
                )
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(base_report_content)
                logger.info(
                    f"Wrote base report for {case_name} because AI credentials were not found: {report_path}"
                )
                continue

            ai_models = [model.strip() for model in ai_models_str.split(",")]

            for ai_model in ai_models:
                logger.info(
                    f"Generating AI analysis for case '{case_name}' with model '{ai_model}'."
                )

                sanitized_model_name = "".join(
                    c for c in ai_model if c.isalnum() or c in ("-", "_")
                ).rstrip()
                model_report_filename = (
                    f"analysis_report_{case_name}_{sanitized_model_name}.md"
                )
                model_report_path = os.path.join(
                    case_results_dir, model_report_filename
                )

                with open(model_report_path, "w", encoding="utf-8") as f:
                    f.write(base_report_content)
                logger.info(
                    f"Generated base report for model {ai_model}: {model_report_path}"
                )

                llm_analysis = call_openai_analysis_api(
                    case_name=case_name,
                    df=summary_df,
                    api_key=api_key,
                    base_url=base_url,
                    ai_model=ai_model,
                    independent_variable=independent_variable,
                    report_content=base_report_content,
                    original_config=original_config,
                    case_data=case_data,
                    reference_col_for_turning_point=reference_col_for_turning_point,
                )

                if llm_analysis:
                    with open(model_report_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"\n\n---\n\n# AI模型分析提示词 ({ai_model})\n\n```markdown\n"
                        )
                        f.write(llm_analysis)
                        f.write("\n```\n")
                    logger.info(f"Appended LLM analysis to {model_report_path}")

                    generate_sensitivity_academic_report(
                        case_name=case_name,
                        case_workspace=case_workspace,
                        independent_variable=independent_variable,
                        original_config=original_config,
                        case_data=case_data,
                        ai_model=ai_model,
                        report_path=model_report_path,
                    )

    except Exception as e:
        logger.error(f"Error generating detailed analysis reports: {e}", exc_info=True)


def _retry_salib_case(
    case_info: Dict[str, Any], original_config: Dict[str, Any]
) -> None:
    """Retries AI analysis for a single SALib case."""
    from tricys.analysis.salib import (
        call_llm_for_academic_report,
        call_llm_for_salib_analysis,
    )

    case_data = case_info["case_data"]
    case_name = case_data.get("name", f"SALibCase{case_info['index']+1}")

    results_dir = os.path.join(case_info["workspace"], "results")

    if not results_dir:
        logger.error(
            "'paths.results_dir' not found in config. Cannot retry SALib case."
        )
        return

    salib_report_dir = results_dir
    if not os.path.isdir(salib_report_dir):
        logger.warning(
            f"SALib report directory '{salib_report_dir}' not found for case {case_name}, skipping retry."
        )
        return

    load_dotenv()
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    ai_models_str = os.environ.get("AI_MODELS") or os.environ.get("AI_MODEL")

    if not all((api_key, base_url, ai_models_str)):
        logger.warning(
            f"API credentials not found. Skipping SALib AI analysis retry for case {case_name}."
        )
        return

    ai_model_to_use = [model.strip() for model in ai_models_str.split(",")][0]
    logger.info(
        f"Checking for SALib retry: case '{case_name}' with model '{ai_model_to_use}'."
    )

    main_report_path = os.path.join(salib_report_dir, "analysis_report.md")
    academic_report_path = os.path.join(salib_report_dir, "academic_report.md")

    if not os.path.exists(main_report_path):
        logger.warning(
            f"SALib base report '{main_report_path}' not found. Cannot retry."
        )
        return

    with open(main_report_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    llm_summary = None
    if "AI模型分析结果" not in report_content:
        logger.info(
            f"SALib AI analysis result not found in '{main_report_path}'. Retrying..."
        )
        method = case_data["analyzer"]["method"]
        wrapper_prompt, llm_summary = call_llm_for_salib_analysis(
            report_content=report_content,
            api_key=api_key,
            base_url=base_url,
            ai_model=ai_model_to_use,
            method=method,
        )
        if wrapper_prompt and llm_summary:
            with open(main_report_path, "a", encoding="utf-8") as f:
                f.write("\n\n---\n\n# AI模型分析提示词\n\n")
                f.write("```markdown\n")
                f.write(wrapper_prompt)
                f.write("\n```\n\n")
                f.write("\n\n---\n\n# AI模型分析结果\n\n")
                f.write(llm_summary)
            logger.info(f"Successfully appended LLM analysis to {main_report_path}")
            with open(main_report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
        else:
            logger.error(
                f"Failed to generate LLM analysis for {main_report_path} on retry."
            )
            return

    if not os.path.exists(academic_report_path):
        if llm_summary is None:
            match = re.search(r"# AI模型分析结果\n\n(.*)", report_content, re.S)
            if match:
                llm_summary = match.group(1).strip()
            else:
                logger.warning(
                    f"Could not find existing LLM summary in {main_report_path} to generate academic report."
                )
                return

        logger.info(
            f"SALib academic report '{academic_report_path}' not found. Generating..."
        )
        glossary_path = original_config.get("sensitivity_analysis", {}).get(
            "glossary_path"
        )
        if not glossary_path or not os.path.exists(glossary_path):
            logger.warning(
                f"Glossary file not found at {glossary_path}, skipping academic report generation."
            )
            return
        with open(glossary_path, "r", encoding="utf-8") as f:
            glossary_content = f.read()

        analysis_case = case_data
        param_names = analysis_case.get("independent_variable")
        sampling_details = analysis_case.get("independent_variable_sampling")
        param_bounds = {
            name: sampling_details[name]["bounds"]
            for name in param_names
            if name in sampling_details
        }
        param_dists = {
            name: sampling_details[name].get("distribution", "unif")
            for name in param_names
            if name in sampling_details
        }
        problem_details = {
            "num_vars": len(param_bounds),
            "names": list(param_bounds.keys()),
            "bounds": list(param_bounds.values()),
            "dists": [param_dists.get(name, "unif") for name in param_bounds.keys()],
        }
        metric_names = case_data.get("dependent_variables", [])
        method = case_data["analyzer"]["method"]

        (
            academic_wrapper_prompt,
            academic_report,
        ) = call_llm_for_academic_report(
            analysis_report=llm_summary,
            glossary_content=glossary_content,
            api_key=api_key,
            base_url=base_url,
            ai_model=ai_model_to_use,
            problem_details=problem_details,
            metric_names=metric_names,
            method=method,
            save_dir=salib_report_dir,
        )
        if academic_wrapper_prompt and academic_report:
            with open(academic_report_path, "w", encoding="utf-8") as f:
                f.write(academic_report)
            logger.info(
                f"Successfully generated academic report: {academic_report_path}"
            )
    else:
        logger.info(
            f"SALib academic report '{academic_report_path}' already exists. Skipping generation."
        )


def _retry_standard_case(
    case_info: Dict[str, Any], original_config: Dict[str, Any]
) -> None:
    """Retries AI analysis for a single standard case."""
    case_data = case_info["case_data"]
    case_workspace = case_info["workspace"]
    case_name = case_data.get("name", f"Case{case_info['index']+1}")

    if not case_data.get("ai", False):
        logger.debug(f"AI is disabled for case {case_name}, skipping retry.")
        return

    case_results_dir = os.path.join(case_workspace, "results")
    if not os.path.isdir(case_results_dir):
        logger.warning(
            f"Results directory not found for case {case_name}, skipping retry."
        )
        return

    load_dotenv()
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    ai_models_str = os.environ.get("AI_MODELS") or os.environ.get("AI_MODEL")

    if not all((api_key, base_url, ai_models_str)):
        logger.warning(
            "API credentials not found. Skipping AI analysis retry for all cases."
        )
        return

    ai_models = [model.strip() for model in ai_models_str.split(",")]
    independent_variable = case_data.get("independent_variable", "燃烧率")

    for ai_model in ai_models:
        logger.info(f"Checking for retry: case '{case_name}' with model '{ai_model}'.")
        sanitized_model_name = "".join(
            c for c in ai_model if c.isalnum() or c in ("-", "_")
        ).rstrip()
        model_report_filename = f"analysis_report_{case_name}_{sanitized_model_name}.md"
        model_report_path = os.path.join(case_results_dir, model_report_filename)

        if not os.path.exists(model_report_path):
            logger.warning(
                f"Base report '{model_report_path}' not found. Cannot retry. Please run the full analysis first."
            )
            continue

        with open(model_report_path, "r", encoding="utf-8") as f:
            report_content = f.read()

        if "AI模型分析结果" not in report_content:
            logger.info(
                f"AI analysis result not found in '{model_report_path}'. Retrying generation..."
            )
            summary_csv_path = os.path.join(
                case_results_dir, "sensitivity_analysis_summary.csv"
            )
            if not os.path.exists(summary_csv_path):
                logger.error(
                    f"Summary CSV not found for case {case_name}, cannot retry."
                )
                continue
            summary_df = pd.read_csv(summary_csv_path)
            reference_col_for_turning_point = None
            sweep_csv_path = os.path.join(case_results_dir, "sweep_results.csv")
            if case_data.get("sweep_time") and os.path.exists(sweep_csv_path):
                try:
                    sweep_df = pd.read_csv(sweep_csv_path)
                    if "time" in sweep_df.columns and len(sweep_df.columns) > 1:
                        reference_col_for_turning_point = sweep_df.columns[
                            len(sweep_df.columns) // 2
                        ]
                except Exception as e:
                    logger.warning(
                        f"Could not determine reference_col_for_turning_point for retry: {e}"
                    )
            llm_analysis = call_openai_analysis_api(
                case_name=case_name,
                df=summary_df,
                api_key=api_key,
                base_url=base_url,
                ai_model=ai_model,
                independent_variable=independent_variable,
                report_content=report_content,
                original_config=original_config,
                case_data=case_data,
                reference_col_for_turning_point=reference_col_for_turning_point,
            )
            if llm_analysis:
                with open(model_report_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n\n---\n\n# AI模型分析提示词 ({ai_model})\n\n```markdown\n"
                    )
                    f.write(llm_analysis)
                    f.write("\n```\n")
                logger.info(
                    f"Successfully appended LLM analysis to {model_report_path}"
                )
                with open(model_report_path, "r", encoding="utf-8") as f:
                    report_content = f.read()
            else:
                logger.error(
                    f"Failed to generate LLM analysis for {model_report_path} on retry."
                )
                continue

        academic_report_filename = (
            f"academic_report_{case_name}_{sanitized_model_name}.md"
        )
        academic_report_path = os.path.join(case_results_dir, academic_report_filename)
        if not os.path.exists(academic_report_path):
            if "AI模型分析结果" in report_content:
                logger.info(
                    f"Academic report '{academic_report_path}' not found. Generating..."
                )
                generate_sensitivity_academic_report(
                    case_name=case_name,
                    case_workspace=case_workspace,
                    independent_variable=independent_variable,
                    original_config=original_config,
                    case_data=case_data,
                    ai_model=ai_model,
                    report_path=model_report_path,
                )
            else:
                logger.warning(
                    f"Skipping academic report for {case_name} as main AI analysis is still missing after retry."
                )
        else:
            logger.info(
                f"Academic report '{academic_report_path}' already exists. Skipping generation."
            )


def retry_ai_analysis(
    case_configs: List[Dict[str, Any]], original_config: Dict[str, Any]
) -> None:
    """Retries AI analysis for cases where it might have failed due to network issues.

    Checks for existing reports and re-runs only the AI-dependent parts if they are missing.
    This function can be triggered by setting an environment variable.

    Args:
        case_configs: List of case configuration dictionaries.
        original_config: Original configuration dictionary.

    Note:
        Routes to _retry_salib_case for SALib cases or _retry_standard_case for standard cases.
        Only regenerates missing AI analysis and academic reports. Does not re-run simulations.
        Logs all retry attempts and failures.
    """
    logger.info("Starting AI analysis retry process...")
    try:
        for case_info in case_configs:
            case_data = case_info["case_data"]
            if "analyzer" in case_data and case_data.get("analyzer", {}).get("method"):
                _retry_salib_case(case_info, original_config)
            else:
                _retry_standard_case(case_info, original_config)
    except Exception as e:
        logger.error(f"Error during AI analysis retry process: {e}", exc_info=True)


def consolidate_reports(
    case_configs: List[Dict[str, Any]], original_config: Dict[str, Any]
) -> None:
    """Consolidates generated reports and their images into a 'report' directory for each case.

    Args:
        case_configs: List of case configuration dictionaries.
        original_config: Original configuration dictionary.

    Note:
        Moves analysis reports, academic reports, and plot images from results directory
        to report directory. Uses move operation (not copy). Creates report directory
        if it doesn't exist. Skips cases where source directory not found.
    """
    logger.info("Consolidating analysis reports...")
    try:
        for case_info in case_configs:
            case_workspace = case_info["workspace"]
            source_dir = os.path.join(case_workspace, "results")
            dest_dir = os.path.join(case_workspace, "report")

            if not os.path.isdir(source_dir):
                logger.warning(
                    f"Source directory not found, skipping consolidation for case: {case_workspace}"
                )
                continue

            # Find files to copy
            files_to_copy = []
            for filename in os.listdir(source_dir):
                if (
                    filename.startswith("analysis_report")
                    or filename.startswith("academic_report")
                ) and filename.endswith(".md"):
                    files_to_copy.append(filename)
                elif filename.endswith((".svg", ".png")):
                    files_to_copy.append(filename)

            if not files_to_copy:
                logger.info(
                    f"No reports or images found in {source_dir}, skipping consolidation."
                )
                continue

            # Create destination directory and copy files
            os.makedirs(dest_dir, exist_ok=True)
            logger.info(f"Consolidating reports into: {dest_dir}")

            for filename in files_to_copy:
                source_path = os.path.join(source_dir, filename)
                shutil.move(source_path, dest_dir)
                logger.info(f"Moved {filename} to {dest_dir}")

    except Exception as e:
        logger.error(f"Error during report consolidation: {e}", exc_info=True)


def generate_analysis_cases_summary(
    case_configs: List[Dict[str, Any]], original_config: Dict[str, Any]
) -> None:
    """Generate summary report for analysis_cases.

    Args:
        case_configs: List of case configuration dictionaries.
        original_config: Original configuration dictionary containing run timestamp.

    Note:
        Creates an execution report with basic information, case details, and status.
        Saves report to {run_timestamp}/execution_report_{run_timestamp}.md in current
        working directory. Also triggers generate_prompt_templates and consolidate_reports.
        Logs summary of successfully executed cases.
    """
    try:
        run_timestamp = original_config["run_timestamp"]
        # Generate report in current working directory
        current_dir = os.getcwd()

        # Create summary report
        summary_data = []
        for case_info in case_configs:
            case_data = case_info["case_data"]
            case_workspace = case_info["workspace"]

            # Check if case results exist
            case_results_dir = os.path.join(case_workspace, "results")
            has_results = (
                os.path.exists(case_results_dir)
                and len(os.listdir(case_results_dir)) > 0
            )

            summary_entry = {
                "case_name": case_data.get("name", f"Case{case_info['index']+1}"),
                "independent_variable": case_data["independent_variable"],
                "independent_variable_sampling": case_data[
                    "independent_variable_sampling"
                ],
                "workspace_path": case_workspace,
                "has_results": has_results,
                "config_file": case_info["config_path"],
            }
            summary_data.append(summary_entry)

        # Generate text report
        report_lines = [
            "# Analysis Cases Execution Report",
            "\n## Basic Information",
            f"- Execution time: {run_timestamp}",
            f"- Total cases: {len(case_configs)}",
            f"- Successfully executed: {sum(1 for entry in summary_data if entry['has_results'])}",
            f"- Working directory: {current_dir}",
            "\n## Case Details",
        ]

        for i, entry in enumerate(summary_data, 1):
            status = "✓ Success" if entry["has_results"] else "✗ Failed"
            report_lines.extend(
                [
                    f"\n### {i}. {entry['case_name']}",
                    f"- Status: {status}",
                    f"- Independent variable: {entry['independent_variable']}",
                    f"- Sampling method: {entry['independent_variable_sampling']}",
                    f"- Working directory: {entry['workspace_path']}",
                    f"- Configuration file: {entry['config_file']}",
                ]
            )

        # Save report to current directory
        report_path = os.path.join(
            current_dir,
            run_timestamp,
            f"execution_report_{run_timestamp}.md",
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info("Summary report generated:")
        logger.info(f"  - Detailed report: {report_path}")

        # Generate prompt engineering template for each case
        generate_prompt_templates(case_configs, original_config)

        # Consolidate all generated reports
        consolidate_reports(case_configs, original_config)

    except Exception as e:
        logger.error(f"Error generating summary report: {e}", exc_info=True)
