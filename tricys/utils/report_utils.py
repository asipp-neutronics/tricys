import logging
import os
import shutil
import time
from typing import Any, Dict, List

import openai
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def call_openai_analysis_api(
    case_name: str,
    case_results_dir: str,
    df: pd.DataFrame,
    api_key: str,
    base_url: str,
    independent_variable: str,
):
    """
    Constructs a text-only prompt, calls the OpenAI API for analysis, and returns the result string.
    """
    try:
        logger.info(f"Proceeding with LLM analysis for case {case_name}.")

        # 1. Construct the prompt for the API
        content_parts = []
        content_parts.append(
            {
                "type": "text",
                "text": """**角色：** 你是一名聚变反应堆氚燃料循环领域的专家。

**任务：** 基于我提供的数据，对聚变堆燃料循环模型的模拟结果进行详细的敏感性分析。请遵循以下结构，分析各项重要参数对关键性能指标的影响，并得出结论。

**分析数据：**
""",
            }
        )

        # NOTE: Image data has been removed from the prompt as per the user's request
        # to create a text-only prompt.

        # Add data table and analysis points as text
        analysis_prompt = f"""(Note: The plot images are not available for analysis. Please perform the analysis based on the data table provided below.)

* **相关指标的数据表:**
{df.to_markdown(index=False)}

**分析要点：**

1.  **总体趋势：** 描述随着{independent_variable}的提高（例如，从2%增加到9%），总氚库存（Inventory）的增长速率有何变化。
2.  **关键指标影响：**
   * **首炉氚量 (Start-up Inventory, Unit: gram)：** 分析其如何随{independent_variable}变化，并量化其降幅。
   * **倍增时间 (Doubling Time, Unit: hour)：** 分析其变化趋势，并量化其降幅。
   * **自持时间 (Self-sufficiency Time, Unit: hour)：** 分析其变化趋势，通常变化较小，请指出。
   * **所需氚增殖比 (Required TBR, $TBR_r$)：** 描述其与{independent_variable}的关系。
3.  **结论：** 总结提高{independent_variable}对于实现氚自持、减少初始投资（首炉氚）和加速氚增殖的有效性。
"""
        content_parts.append({"type": "text", "text": analysis_prompt})

        # 2. Call API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                logger.info(
                    f"Sending request to OpenAI API for case {case_name} (Attempt {attempt + 1}/{max_retries})..."
                )

                full_text_prompt = "\n\n".join([part["text"] for part in content_parts])

                response = client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.1",
                    messages=[{"role": "user", "content": full_text_prompt}],
                    max_tokens=4000,
                )
                analysis_result = response.choices[0].message.content

                logger.info(f"LLM analysis successful for case {case_name}.")
                return analysis_result  # Return the result string

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


def generate_prompt_templates(
    case_configs: List[Dict[str, Any]], original_config: Dict[str, Any]
):
    """Generate detailed Markdown analysis reports for each analysis case."""
    try:
        for case_info in case_configs:
            case_data = case_info["case_data"]

            # Skip SALib cases, as they are handled by salib_utils.py
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

            # File discovery
            sweep_plots = [
                f
                for f in os.listdir(case_results_dir)
                if f.startswith("sweep_") and f.endswith(".png")
            ]
            combined_plots = [
                f
                for f in os.listdir(case_results_dir)
                if f.startswith("combined_") and f.endswith(".png")
            ]
            summary_csv_path = os.path.join(
                case_results_dir, "sensitivity_analysis_summary.csv"
            )

            if not os.path.exists(summary_csv_path):
                logger.warning(
                    f"summary_csv not found for case {case_name}, skipping report generation."
                )
                continue

            # Data analysis
            df = pd.read_csv(summary_csv_path)
            independent_variable = case_data.get("independent_variable", "燃烧率")

            # Markdown Generation
            prompt_lines = [
                "**角色：** 你是一名聚变反应堆氚燃料循环领域的专家。",
                "",
                "**任务：** 基于我提供的数据，对聚变堆燃料循环模型的模拟结果进行详细的敏感性分析。请遵循以下结构，分析各项重要参数对关键性能指标的影响，并得出结论。",
                "",
                "**分析数据：**",
                "",
            ]

            # Add plots
            for plot in sweep_plots:
                prompt_lines.append(
                    f"* **不同{independent_variable}下Inventory随时间变化 的曲线图:**\n"
                    f"![不同{independent_variable}下Inventory随时间变化 的曲线图]({plot})"
                )
            for plot in combined_plots:
                prompt_lines.append(
                    f"* **不同{independent_variable}下首炉氚、自持时间、倍增时间变化，最小TBR 的柱状图/折线图:**\n"
                    f"![不同{independent_variable}下首炉氚、自持时间、倍增时间变化，最小TBR 的柱状图/折线图]({plot})"
                )

            # Add data table
            prompt_lines.append("* **相关指标的数据表:**\n")
            prompt_lines.append(df.to_markdown(index=False))

            prompt_lines.extend(
                [
                    "",
                    "**分析要点：**\n",
                    f"1.  **总体趋势：** 描述随着{independent_variable}的提高，总氚库存（Inventory）的增长速率有何变化。",
                    "2.  **关键指标影响：**",
                    f"   * **首炉氚量 (Start-up Inventory, Unit: gram)：** 分析其如何随{independent_variable}变化，并量化其降幅。",
                    "   * **倍增时间 (Doubling Time, Unit: hour)：** 分析其变化趋势，并量化其降幅。",
                    "   * **自持时间 (Self-sufficiency Time, Unit: hour)：** 分析其变化趋势，通常变化较小，请指出。",
                    f"   * **所需氚增殖比 (Required TBR, $TBR_r$)：** 描述其与{independent_variable}的关系。",
                    f"3.  **结论：** 总结提高{independent_variable}对于实现氚自持、减少初始投资（首炉氚）和加速氚增殖的有效性。",
                ]
            )

            # Save the detailed report
            report_path = os.path.join(
                case_results_dir, f"analysis_report_{case_name}.md"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(prompt_lines))

            logger.info(
                f"Detailed analysis report generated for {case_name}: {report_path}"
            )

            # --- OpenAI API Call for Automated Analysis ---
            load_dotenv()

            llm_analysis = call_openai_analysis_api(
                case_name=case_name,
                case_results_dir=case_results_dir,
                df=df,
                api_key=os.environ.get("SILICONFLOW_API_KEY"),
                base_url=os.environ.get("SILICONFLOW_BASE_URL"),
                independent_variable=independent_variable,
            )

            # Append LLM analysis to the report
            if llm_analysis:
                with open(report_path, "a", encoding="utf-8") as f:
                    f.write("\n\n---\n\n# AI模型分析结果\n\n")
                    f.write(llm_analysis)
                logger.info(f"Appended LLM analysis to {report_path}")

    except Exception as e:
        logger.error(f"Error generating detailed analysis reports: {e}", exc_info=True)


def consolidate_reports(
    case_configs: List[Dict[str, Any]], original_config: Dict[str, Any]
):
    """
    Consolidates generated reports and their images into a 'report' directory for each case.
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
                if filename.startswith("analysis_report") and filename.endswith(".md"):
                    files_to_copy.append(filename)
                elif filename.endswith(".png"):
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
                shutil.copy(source_path, dest_dir)
                logger.info(f"Copied {filename} to {dest_dir}")

    except Exception as e:
        logger.error(f"Error during report consolidation: {e}", exc_info=True)


def generate_analysis_cases_summary(
    case_configs: List[Dict[str, Any]], original_config: Dict[str, Any]
):
    """Generate summary report for analysis_cases"""
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
            "analysis_cases",
            run_timestamp,
            f"execution_report_{run_timestamp}.md",
        )
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
