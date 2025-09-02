import argparse
import concurrent.futures
import importlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from OMPython import ModelicaSystem

from tricys.utils.file_utils import get_unique_filename
from tricys.utils.log_utils import setup_logging
from tricys.utils.metric_utils import (
    extract_metrics,
    time_of_turning_point,
)
from tricys.utils.om_utils import (
    format_parameter_value,
    get_om_session,
    integrate_interceptor_model,
    load_modelica_package,
)
from tricys.utils.plot_utils import generate_analysis_plots, plot_sweep_time_series
from tricys.utils.sim_utils import generate_simulation_jobs

# Standard logger setup
logger = logging.getLogger(__name__)


def _validate_analysis_cases_config(config: Dict[str, Any]) -> bool:
    """Validate analysis_cases configuration format, supporting both list and single object formats

    This function validates:
    1. Basic structure and required fields of analysis_cases
    2. Simulation parameters compatibility (single job requirement)
    3. Required_TBR configuration completeness if used in dependent_variables

    Args:
        config: Configuration dictionary to validate

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if "sensitivity_analysis" not in config:
        logger.error("Missing sensitivity_analysis")
        return False

    sensitivity_analysis = config["sensitivity_analysis"]
    if "analysis_cases" not in sensitivity_analysis:
        logger.error("Missing analysis_cases")
        return False

    analysis_cases = sensitivity_analysis["analysis_cases"]

    # Support both single object and list formats
    if isinstance(analysis_cases, dict):
        # Single analysis_case object
        cases_to_check = [analysis_cases]
    elif isinstance(analysis_cases, list) and len(analysis_cases) > 0:
        # analysis_cases list
        cases_to_check = analysis_cases
    else:
        logger.error("analysis_cases must be a non-empty list or a single object")
        return False

    # Check required fields for each analysis_case
    required_fields = ["name", "independent_variable", "independent_variable_sampling"]
    for i, case in enumerate(cases_to_check):
        if not isinstance(case, dict):
            logger.error(f"analysis_cases[{i}] must be an object")
            return False
        for field in required_fields:
            if field not in case:
                logger.error(f"Missing required field '{field}' in analysis_cases[{i}]")
                return False

    # Check if simulation_parameters would generate more than 1 job
    try:
        jobs = generate_simulation_jobs(config.get("simulation_parameters", {}))
        if len(jobs) > 1:
            logger.error(
                f"simulation_parameters would generate {len(jobs)} jobs, but analysis mode only supports single job execution"
            )
            return False
    except Exception as e:
        logger.error(f"Error validating simulation_parameters: {e}")
        return False

    # Check Required_TBR configuration completeness if it exists in dependent_variables
    metrics_definition = sensitivity_analysis.get("metrics_definition", {})
    for i, case in enumerate(cases_to_check):
        dependent_vars = case.get("dependent_variables", [])
        if "Required_TBR" in dependent_vars:
            # Check if Required_TBR exists in metrics_definition
            if "Required_TBR" not in metrics_definition:
                logger.error(
                    f"Required_TBR is in dependent_variables of analysis_cases[{i}] but missing from metrics_definition"
                )
                return False

            # Check if Required_TBR configuration is complete
            required_tbr_config = metrics_definition["Required_TBR"]
            required_fields = [
                "method",
                "parameter_to_optimize",
                "search_range",
                "tolerance",
                "max_iterations",
            ]
            missing_fields = [
                field for field in required_fields if field not in required_tbr_config
            ]
            if missing_fields:
                logger.error(
                    f"Required_TBR configuration in metrics_definition is incomplete. Missing fields: {missing_fields}"
                )
                return False

    return True


def _convert_relative_paths_to_absolute(
    config: Dict[str, Any], base_dir: str
) -> Dict[str, Any]:
    """
    Recursively traverse configuration data and convert relative paths to absolute paths based on the specified base directory

    Args:
        config: Configuration dictionary
        base_dir: Base directory path

    Returns:
        Converted configuration dictionary
    """

    def _process_value(value, key_name="", parent_dict=None):
        if isinstance(value, dict):
            return {k: _process_value(v, k, value) for k, v in value.items()}
        elif isinstance(value, list):
            return [_process_value(item, parent_dict=parent_dict) for item in value]
        elif isinstance(value, str):
            # Check if it's a path-related key name (extended support for more path fields)
            path_keys = [
                "package_path",
                "db_path",
                "results_dir",
                "temp_dir",
                "log_dir",
            ]

            # Special case: when independent_variable="file", independent_variable_sampling is also a file path
            is_file_sampling = (
                key_name == "independent_variable_sampling"
                and parent_dict is not None
                and parent_dict.get("independent_variable") == "file"
            )

            if key_name.endswith("_path") or key_name in path_keys or is_file_sampling:
                # If it's a relative path, convert to absolute path
                if not os.path.isabs(value):
                    abs_path = os.path.abspath(os.path.join(base_dir, value))
                    logger.debug(
                        f"Converted path: {key_name} '{value}' -> '{abs_path}'"
                    )
                    return abs_path
            return value
        else:
            return value

    return _process_value(config)


def _create_standard_config_for_case(
    base_config: Dict[str, Any], analysis_case: Dict[str, Any], i: int
) -> Dict[str, Any]:
    """Create a standard format configuration file for a single analysis case and handle path conversion"""
    # Get the base directory where the original configuration file is located
    original_config_dir = (
        os.getcwd()
    )  # Directory where the original configuration file is located

    # First convert relative paths to absolute paths
    absolute_config = _convert_relative_paths_to_absolute(
        base_config, original_config_dir
    )
    # Deep copy the converted configuration
    standard_config = json.loads(json.dumps(absolute_config))

    # Create simulation_parameters
    # Support both single object and list formats for analysis_cases
    analysis_cases = standard_config["sensitivity_analysis"]["analysis_cases"]
    if isinstance(analysis_cases, dict):
        independent_var = analysis_cases["independent_variable"]
        independent_sampling = analysis_cases["independent_variable_sampling"]
    elif isinstance(analysis_cases, list):
        independent_var = analysis_cases[i]["independent_variable"]
        independent_sampling = analysis_cases[i]["independent_variable_sampling"]
    else:
        raise ValueError("analysis_cases must be a dict or a list")
    logger.debug(f"independent_sampling configuration: {independent_sampling}")

    # Ensure simulation_parameters exists
    if "simulation_parameters" not in standard_config:
        standard_config["simulation_parameters"] = {}

    # Add independent_variable_sampling to simulation_parameters
    standard_config["simulation_parameters"][independent_var] = independent_sampling

    # Update sensitivity_analysis configuration
    sensitivity_analysis = standard_config["sensitivity_analysis"]

    # Remove analysis_cases and replace with single analysis_case
    if "analysis_cases" in sensitivity_analysis:
        del sensitivity_analysis["analysis_cases"]

    sensitivity_analysis["analysis_case"] = analysis_case.copy()

    return standard_config


def _setup_analysis_cases_workspaces(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Set up independent working directories and configuration files for multiple analysis_cases

    This function will:
    1. Create independent working directories for each analysis_case in the current working directory
    2. Convert relative paths in the original configuration to absolute paths
    3. Convert analysis_cases format to standard analysis_case format
    4. Generate independent config.json files for each case

    Args:
        config: Original configuration dictionary containing analysis_cases

    Returns:
        List containing information for each case, each element contains:
        - index: Case index
        - workspace: Working directory path
        - config_path: Configuration file path
        - config: Configuration applicable to this case
        - case_data: Original case data
    """

    analysis_cases_raw = config["sensitivity_analysis"]["analysis_cases"]

    # Unified processing into list format
    if isinstance(analysis_cases_raw, dict):
        # Single analysis_case object
        analysis_cases = [analysis_cases_raw]
        logger.info(
            "Detected single analysis_case object, converting to list format for processing"
        )
    else:
        # Already in list format
        analysis_cases = analysis_cases_raw

    # Create case working directories in current working directory
    current_dir = os.getcwd()

    logger.info(
        f"Detected {len(analysis_cases)} analysis cases, creating independent working directories in current directory..."
    )
    logger.info(f"Current working directory: {current_dir}")

    case_configs = []

    for i, analysis_case in enumerate(analysis_cases):
        try:
            # Generate case working directory name
            workspace_name = analysis_case.get("name", f"case_{i}")
            # Get runtime timestamp
            run_timestamp = config["run_timestamp"]
            # Create timestamped analysis_cases directory in current directory, then create case directory within it
            case_workspace = os.path.join(
                current_dir, "analysis_cases", run_timestamp, workspace_name
            )
            os.makedirs(case_workspace, exist_ok=True)

            # Create standard configuration
            standard_config = _create_standard_config_for_case(config, analysis_case, i)

            # Update paths in configuration to be relative to case working directory
            case_config = standard_config.copy()
            case_config["paths"]["results_dir"] = os.path.join(
                case_workspace, "results"
            )
            case_config["paths"]["temp_dir"] = os.path.join(case_workspace, "temp")
            case_config["paths"]["db_path"] = os.path.join(
                case_workspace, "data", "parameters.db"
            )

            # If there's logging configuration, also update log directory
            if "logging" in case_config and "log_dir" in case_config["logging"]:
                case_config["logging"]["log_dir"] = os.path.join(case_workspace, "log")

            # Save standard configuration file to case working directory
            config_file_path = os.path.join(case_workspace, "config.json")
            with open(config_file_path, "w", encoding="utf-8") as f:
                json.dump(standard_config, f, indent=4, ensure_ascii=False)

            # Record case information
            case_info = {
                "index": i,
                "workspace": case_workspace,
                "config_path": config_file_path,
                "config": case_config,
                "case_data": analysis_case,
            }
            case_configs.append(case_info)

            logger.info(f"✓ Working directory for case {i+1} created:")
            logger.info(f"  - Case name: {analysis_case.get('name', f'Case{i+1}')}")
            logger.info(
                f"  - Independent variable: {analysis_case['independent_variable']}"
            )
            logger.info(
                f"  - Sampling method: {analysis_case['independent_variable_sampling']}"
            )
            logger.info(f"  - Working directory: {case_workspace}")
            logger.info(f"  - Configuration file: {config_file_path}")

        except Exception as e:
            logger.error(f"✗ Error processing case {i}: {e}", exc_info=True)
            continue

    logger.info(
        f"Successfully created independent working directories for {len(case_configs)} analysis cases"
    )
    logger.info(f"{case_configs}")
    return case_configs


def _is_optimization_enabled(config: dict) -> bool:
    """
    Check if optimization functionality is enabled
    Determined by checking if Required_TBR is included in the sensitivity_analysis field
    Also check metrics_definition and dependent_variables, and validate Required_TBR configuration completeness

    Args:
        config: Configuration dictionary

    Returns:
        bool: Returns True if optimization is enabled, otherwise False
    """
    sensitivity_analysis = config.get("sensitivity_analysis", {})

    # Check if metrics_definition contains Required_TBR and has complete configuration
    metrics_definition = sensitivity_analysis.get("metrics_definition", {})
    has_required_tbr_in_metrics = "Required_TBR" in metrics_definition

    # Check if Required_TBR configuration is complete
    has_complete_tbr_config = False
    if has_required_tbr_in_metrics:
        required_tbr_config = metrics_definition["Required_TBR"]
        required_fields = [
            "method",
            "parameter_to_optimize",
            "search_range",
            "tolerance",
            "max_iterations",
        ]
        has_complete_tbr_config = all(
            field in required_tbr_config for field in required_fields
        )

    # Check if dependent_variables in analysis_cases contains Required_TBR
    has_required_tbr_in_dependent_vars = False
    analysis_cases_raw = sensitivity_analysis.get("analysis_cases", [])

    # Unified processing into list format
    if isinstance(analysis_cases_raw, dict):
        analysis_cases = [analysis_cases_raw]
    elif isinstance(analysis_cases_raw, list):
        analysis_cases = analysis_cases_raw
    else:
        analysis_cases = []

    if analysis_cases:
        for case in analysis_cases:
            dependent_vars = case.get("dependent_variables", [])
            if "Required_TBR" in dependent_vars:
                has_required_tbr_in_dependent_vars = True
                break
    else:
        # If there's no analysis_cases, check single analysis_case
        analysis_case = sensitivity_analysis.get("analysis_case", {})
        dependent_vars = analysis_case.get("dependent_variables", [])
        has_required_tbr_in_dependent_vars = "Required_TBR" in dependent_vars

    # Only enable optimization when Required_TBR exists in metrics_definition with complete configuration
    # and also exists in dependent_variables
    return (
        has_required_tbr_in_metrics
        and has_complete_tbr_config
        and has_required_tbr_in_dependent_vars
    )


def _run_sensitivity_analysis(
    config: Dict[str, Any], run_results_dir: str, jobs: List[Dict[str, Any]]
) -> None:
    """
    Execute the complete workflow of sensitivity analysis

    This function is responsible for:
    1. Checking if sensitivity analysis is enabled
    2. Extracting metrics from merged result data
    3. Merging optimization results (if they exist)
    4. Saving summary data
    5. Generating analysis charts

    Args:
        config: Configuration dictionary containing sensitivity analysis related configuration
        run_results_dir: Results save directory
        jobs: Simulation job list
    """
    if not config.get("sensitivity_analysis", {}).get("enabled", False):
        return

    logger.info("Starting automated sensitivity analysis.")

    try:
        # Check if there are result data files
        combined_csv_path = os.path.join(run_results_dir, "sweep_results.csv")
        single_result_path = os.path.join(run_results_dir, "simulation_result.csv")

        # Determine result file path based on number of jobs
        if len(jobs) == 1 and os.path.exists(single_result_path):
            # Single task case
            results_df = pd.read_csv(single_result_path)
            logger.info(f"Loading single task result from: {single_result_path}")
        elif len(jobs) > 1 and os.path.exists(combined_csv_path):
            # Multi-task case
            results_df = pd.read_csv(combined_csv_path)
            logger.info(f"Loading sweep results from: {combined_csv_path}")
        else:
            logger.warning("No result data file found for sensitivity analysis.")
            return

        # Get analysis_case configuration
        analysis_config = config["sensitivity_analysis"]
        analysis_case = analysis_config["analysis_case"]

        if analysis_case is None:
            logger.warning("No valid analysis_case found for sensitivity analysis.")
            return

        # Extract summary metrics
        summary_df = extract_metrics(
            results_df,
            analysis_config["metrics_definition"],
            analysis_case,
        )

        if summary_df.empty:
            logger.warning("Sensitivity analysis did not produce any summary data.")
            return

        # Merge optimization results (if they exist)
        if not _is_optimization_enabled(config):
            df_to_save = summary_df
        else:
            optimization_summary_path = os.path.join(
                run_results_dir, "requierd_tbr_summary.csv"
            )
            if not os.path.exists(optimization_summary_path):
                logger.warning(
                    "Optimization summary not found, saving analysis summary only."
                )
                df_to_save = summary_df
            else:
                try:
                    optimization_df = pd.read_csv(optimization_summary_path)
                    sweep_params = list(jobs[0].keys())
                    merged_df = pd.merge(
                        summary_df,
                        optimization_df,
                        on=sweep_params,
                        how="outer",
                    )
                    logger.info(
                        "Merged optimization and sensitivity analysis summaries."
                    )
                    df_to_save = merged_df
                except Exception as e:
                    logger.error(
                        f"Failed to merge summaries: {e}. Saving analysis summary only."
                    )

        # Save summary data
        summary_csv_path = get_unique_filename(
            run_results_dir, "sensitivity_analysis_summary.csv"
        )
        df_to_save.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Sensitivity analysis summary saved to: {summary_csv_path}")

        # Generate analysis charts
        generate_analysis_plots(df_to_save, analysis_case, run_results_dir)

    except Exception as e:
        logger.error(f"Automated sensitivity analysis failed: {e}", exc_info=True)


def _save_optimization_summary(
    config: dict, final_results: List[Dict[str, Any]]
) -> None:
    """
    Save optimization results summary

    Args:
        config: Configuration dictionary
        final_results: Optimization results list
    """
    if _is_optimization_enabled(config):
        results_dir = os.path.abspath(config["paths"]["results_dir"])
        os.makedirs(results_dir, exist_ok=True)
        if final_results:
            final_df = pd.DataFrame(final_results)
            output_path = os.path.join(results_dir, "requierd_tbr_summary.csv")
            final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"Sweep optimization summary saved to: {output_path}")


def _generate_analysis_cases_summary(
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
            current_dir, "analysis_cases", run_timestamp, f"report_{run_timestamp}.md"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info("Summary report generated:")
        logger.info(f"  - Detailed report: {report_path}")

    except Exception as e:
        logger.error(f"Error generating summary report: {e}", exc_info=True)


def _run_bisection_search_for_job(
    config: Dict[str, Any], job_id_prefix: str
) -> tuple[float, float]:
    """
    Execute bisection search for a single job configuration and return results.
    Read optimization parameters directly from sensitivity_analysis.metrics_definition.Required_TBR.

    Args:
        config: Specific job configuration containing fixed parameters.
        job_id_prefix: Prefix for creating unique IDs for subtasks.

    Returns:
        A tuple containing:
        - Found optimal successful parameter value (float).
        - Optimal successful metric value (float).
    """
    # Read Required_TBR configuration directly from sensitivity_analysis
    sensitivity_analysis = config.get("sensitivity_analysis", {})
    metrics_definition = sensitivity_analysis.get("metrics_definition", {})
    required_tbr_config = metrics_definition.get("Required_TBR", {})

    if not required_tbr_config or "parameter_to_optimize" not in required_tbr_config:
        raise ValueError(
            "Required_TBR configuration not found or missing parameter_to_optimize"
        )

    sim_config = config["simulation"]
    paths_config = config["paths"]

    param_to_optimize = required_tbr_config["parameter_to_optimize"]
    low, high = required_tbr_config["search_range"]
    tolerance = required_tbr_config.get("tolerance", 0.001)
    max_iterations = required_tbr_config.get("max_iterations", 10)

    stop_time = sim_config["stop_time"]
    # Hardcoded for Self_Sufficiency_Time analysis
    metric_source_column = "sds.I[1]"
    metric_name = "Self_Sufficiency_Time"

    logger.info(
        f"Starting bisection search for '{param_to_optimize}' in range [{low}, {high}]"
    )

    best_successful_param = float("inf")
    best_successful_value = float("inf")

    # Setup for reusing the model object
    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    os.makedirs(base_temp_dir, exist_ok=True)

    omc = None
    try:
        omc = get_om_session()
        package_path = os.path.abspath(paths_config["package_path"])
        if not load_modelica_package(omc, Path(package_path).as_posix()):
            raise RuntimeError("Failed to load Modelica package for bisection search.")

        mod = ModelicaSystem(
            fileName=Path(package_path).as_posix(),
            modelName=sim_config["model_name"],
            variableFilter=sim_config["variableFilter"],
        )
        mod.setSimulationOptions(
            [
                f"stopTime={sim_config['stop_time']}",
                "tolerance=1e-6",
                "outputFormat=csv",
                f"stepSize={sim_config['step_size']}",
            ]
        )
        mod.buildModel()

        for i in range(max_iterations):
            if high - low < tolerance:
                logger.info(
                    f"Search converged for {job_id_prefix}. Tolerance {tolerance} reached."
                )
                break

            mid_param = (low + high) / 2

            logger.info(
                f"--- [{job_id_prefix}] Iteration {i+1}/{max_iterations}: Testing {param_to_optimize} = {mid_param:.4f} ---"
            )

            job_params = config.get("simulation_parameters", {}).copy()
            job_params[param_to_optimize] = mid_param

            # Set parameters on the existing mod object
            param_settings = [
                format_parameter_value(name, value)
                for name, value in job_params.items()
            ]
            if param_settings:
                mod.setParameters(param_settings)

            # Create a workspace for this iteration's results
            iter_job_id_str = f"iter{i}_{mid_param}"
            iter_temp_dir = os.path.join(base_temp_dir, job_id_prefix)
            os.makedirs(iter_temp_dir, exist_ok=True)
            job_workspace = os.path.join(iter_temp_dir, f"{iter_job_id_str}")
            os.makedirs(job_workspace, exist_ok=True)
            result_filename = f"{iter_job_id_str}_simulation_results.csv"
            result_path = os.path.join(job_workspace, result_filename)

            # Simulate
            mod.simulate(resultfile=Path(result_path).as_posix())

            metric_value = float("inf")

            if not os.path.exists(result_path):
                logger.error(
                    f"Analysis failed for params {job_params}: Simulation did not produce a result file."
                )
            else:
                try:
                    results_df = pd.read_csv(result_path)
                    if metric_source_column not in results_df.columns:
                        logger.error(
                            f"Analysis failed: source column '{metric_source_column}' not found in results."
                        )
                    else:
                        metric_value = time_of_turning_point(
                            results_df[metric_source_column], results_df["time"]
                        )
                        logger.info(
                            f"Analysis for params {job_params} successful. Metric '{metric_name}': {metric_value}"
                        )
                except Exception as e:
                    logger.error(
                        f"Analysis failed for params {job_params} due to an exception: {e}",
                        exc_info=True,
                    )

            if metric_value < stop_time:
                best_successful_param = mid_param
                best_successful_value = metric_value
                high = mid_param
            else:
                low = mid_param

    except Exception as e:
        logger.error(
            f"Bisection search failed during setup or execution: {e}", exc_info=True
        )
    finally:
        if omc:
            omc.sendExpression("quit()")

    if best_successful_param == float("inf"):
        logger.warning(
            f"Bisection search for {job_id_prefix} did not find a successful parameter."
        )
    else:
        logger.info(
            f"Bisection search for {job_id_prefix} finished. Optimal value: {best_successful_param:.4f}"
        )

    return best_successful_param, best_successful_value


def _run_co_simulation(
    config: dict, job_params: dict, job_id: int = 0
) -> tuple[float, float, str]:
    """
    Runs the full co-simulation workflow in an isolated directory to ensure thread safety.
    """
    paths_config = config["paths"]
    sim_config = config["simulation"]

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))

    job_workspace = os.path.join(base_temp_dir, f"job_{job_id}")
    os.makedirs(job_workspace, exist_ok=True)

    omc = None
    optimal_param = 0.0
    optimal_value = 0.0

    try:
        original_package_path = os.path.abspath(paths_config["package_path"])
        original_package_dir = os.path.dirname(original_package_path)
        package_dir_name = os.path.basename(original_package_dir)

        isolated_package_dir = os.path.join(job_workspace, package_dir_name)

        if os.path.exists(isolated_package_dir):
            shutil.rmtree(isolated_package_dir)
        shutil.copytree(original_package_dir, isolated_package_dir)

        isolated_package_path = os.path.join(
            isolated_package_dir, os.path.basename(original_package_path)
        )
        isolated_temp_dir = job_workspace
        results_dir = os.path.abspath(paths_config["results_dir"])
        os.makedirs(results_dir, exist_ok=True)

        co_sim_configs = config["co_simulation"]
        if not isinstance(co_sim_configs, list):
            co_sim_configs = [co_sim_configs]

        model_name = sim_config["model_name"]
        stop_time = sim_config["stop_time"]
        step_size = sim_config["step_size"]

        omc = get_om_session()
        if not load_modelica_package(omc, Path(isolated_package_path).as_posix()):
            raise RuntimeError(
                f"Failed to load Modelica package at {isolated_package_path}"
            )

        # Handle copying of any additional asset directories specified with a '_path' suffix
        for co_sim_config in co_sim_configs:
            if "params" in co_sim_config:
                # Iterate over a copy of items since we are modifying the dict
                for param_key, param_value in list(co_sim_config["params"].items()):
                    if isinstance(param_value, str) and param_key.endswith("_path"):
                        original_asset_path_str = param_value

                        # Paths in config are relative to project root. We need the absolute path.
                        original_asset_path = Path(
                            os.path.abspath(original_asset_path_str)
                        )
                        original_asset_dir = original_asset_path.parent

                        if not original_asset_dir.exists():
                            logger.warning(
                                f"Asset directory '{original_asset_dir}' for parameter '{param_key}' not found. Skipping copy."
                            )
                            continue

                        asset_dir_name = original_asset_dir.name
                        dest_dir = Path(job_workspace) / asset_dir_name

                        # Copy the directory only if it hasn't been copied already
                        if not dest_dir.exists():
                            shutil.copytree(original_asset_dir, dest_dir)
                            logger.info(
                                f"Copied asset directory '{original_asset_dir}' to '{dest_dir}' for job {job_id}"
                            )

                        # Update the path in the config to point to the new location
                        new_asset_path = dest_dir / original_asset_path.name
                        co_sim_config["params"][param_key] = new_asset_path.as_posix()
                        logger.info(
                            f"Updated parameter '{param_key}' for job {job_id} to '{co_sim_config['params'][param_key]}'"
                        )

        all_input_vars = []
        for co_sim_config in co_sim_configs:
            submodel_name = co_sim_config["submodel_name"]
            instance_name = co_sim_config["instance_name"]
            logger.info(f"Identifying input ports for submodel '{submodel_name}'...")
            components = omc.sendExpression(f"getComponents({submodel_name})")
            input_ports = [
                {"name": c[1], "dim": int(c[11][0]) if c[11] else 1}
                for c in components
                if c[0] == "Modelica.Blocks.Interfaces.RealInput"
            ]
            if not input_ports:
                logger.warning(f"No RealInput ports found in {submodel_name}.")
                continue

            logger.info(
                f"Found input ports for {instance_name}: {[p['name'] for p in input_ports]}"
            )
            for port in input_ports:
                full_name = f"{instance_name}.{port['name']}".replace(".", "\\.")
                if port["dim"] > 1:
                    full_name += f"\\[[1-{port['dim']}]\\]"
                all_input_vars.append(full_name)

        variable_filter = "time|" + "|".join(all_input_vars)

        mod = ModelicaSystem(
            fileName=Path(isolated_package_path).as_posix(),
            modelName=model_name,
            variableFilter=variable_filter,
        )
        mod.setSimulationOptions(
            [f"stopTime={stop_time}", f"stepSize={step_size}", "outputFormat=csv"]
        )

        param_settings = [
            format_parameter_value(name, value) for name, value in job_params.items()
        ]
        if param_settings:
            logger.info(f"Applying parameters for job {job_id}: {param_settings}")
            mod.setParameters(param_settings)

        primary_result_filename = get_unique_filename(
            isolated_temp_dir, "primary_inputs.csv"
        )
        mod.simulate(resultfile=Path(primary_result_filename).as_posix())

        interception_configs = []
        for co_sim_config in co_sim_configs:
            handler_module = importlib.import_module(co_sim_config["handler_module"])
            handler_function = getattr(
                handler_module, co_sim_config["handler_function"]
            )
            instance_name = co_sim_config["instance_name"]

            co_sim_output_filename = get_unique_filename(
                isolated_temp_dir, f"{instance_name}_outputs.csv"
            )

            output_placeholder = handler_function(
                temp_input_csv=primary_result_filename,
                temp_output_csv=co_sim_output_filename,
                **co_sim_config.get("params", {}),
            )

            interception_configs.append(
                {
                    "submodel_name": co_sim_config["submodel_name"],
                    "instance_name": co_sim_config["instance_name"],
                    "csv_uri": Path(os.path.abspath(co_sim_output_filename)).as_posix(),
                    "output_placeholder": output_placeholder,
                }
            )

        intercepted_model_paths = integrate_interceptor_model(
            package_path=isolated_package_path,
            model_name=model_name,
            interception_configs=interception_configs,
        )

        verif_config = config["simulation"]["variableFilter"]
        logger.info("Proceeding with Final simulation.")

        for model_path in intercepted_model_paths["interceptor_model_paths"]:
            omc.sendExpression(f"""loadFile("{Path(model_path).as_posix()}")""")
        omc.sendExpression(
            f"""loadFile("{Path(intercepted_model_paths["system_model_path"]).as_posix()}")"""
        )

        package_name, original_system_name = model_name.split(".")
        intercepted_model_full_name = (
            f"{package_name}.{original_system_name}_Intercepted"
        )

        verif_mod = ModelicaSystem(
            fileName=Path(isolated_package_path).as_posix(),
            modelName=intercepted_model_full_name,
            variableFilter=verif_config,
        )
        verif_mod.setSimulationOptions(
            [f"stopTime={stop_time}", f"stepSize={step_size}", "outputFormat=csv"]
        )
        if param_settings:
            verif_mod.setParameters(param_settings)

        default_result_path = get_unique_filename(
            job_workspace, "co_simulation_results.csv"
        )
        verif_mod.simulate(resultfile=Path(default_result_path).as_posix())

        if not os.path.exists(default_result_path):
            raise FileNotFoundError(
                f"Simulation for job {job_id} failed to produce a result file at {default_result_path}"
            )

        optimal_param = 0.0
        optimal_value = 0.0
        # Done START: Check if optimization is enabled, if so call _run_bisection_search_for_job for optimization
        if _is_optimization_enabled(config):
            job_config = config.copy()
            job_config["paths"]["package_path"] = isolated_package_path
            job_config["paths"]["temp_dir"] = base_temp_dir
            job_config["simulation_parameters"] = job_params
            job_config["simulation"]["model_name"] = intercepted_model_full_name
            optimal_param, optimal_value = _run_bisection_search_for_job(
                job_config, job_id_prefix=f"job_{job_id}"
            )
            logger.info(
                f"Job {job_id} optimization complete. Optimal blanket.TBR: {optimal_param} , Optimal Self_Sufficiency_Time: {optimal_value}"
            )
        # Done END

        # Return the path to the result file inside the temporary workspace
        return optimal_param, optimal_value, Path(default_result_path).as_posix()
    except Exception as e:
        logger.error(f"Workflow for job {job_id} failed: {e}", exc_info=True)
        return optimal_param, optimal_value, ""
    finally:
        if omc:
            omc.sendExpression("quit()")
            logger.info(f"Closed OMPython session for job {job_id}.")

        if not sim_config.get("keep_temp_files", False):
            if os.path.exists(job_workspace):
                shutil.rmtree(job_workspace)
                logger.info(f"Cleaned up workspace for job {job_id}: {job_workspace}")


def _run_single_job(
    config: dict, job_params: dict, job_id: int = 0
) -> tuple[float, float, str]:
    """Executes a single simulation job in an isolated workspace."""
    paths_config = config["paths"]
    sim_config = config["simulation"]

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    job_workspace = os.path.join(base_temp_dir, f"job_{job_id}")
    os.makedirs(job_workspace, exist_ok=True)

    logger.info(f"Starting job {job_id} with parameters: {job_params}")
    omc = None
    try:
        omc = get_om_session()
        package_path = os.path.abspath(paths_config["package_path"])
        if not load_modelica_package(omc, Path(package_path).as_posix()):
            raise RuntimeError(f"Job {job_id}: Failed to load Modelica package.")

        mod = ModelicaSystem(
            fileName=Path(package_path).as_posix(),
            modelName=sim_config["model_name"],
            variableFilter=sim_config["variableFilter"],
        )
        mod.setSimulationOptions(
            [
                f"stopTime={sim_config['stop_time']}",
                "tolerance=1e-6",
                "outputFormat=csv",
                f"stepSize={sim_config['step_size']}",
            ]
        )
        param_settings = [
            format_parameter_value(name, value) for name, value in job_params.items()
        ]
        if param_settings:
            mod.setParameters(param_settings)

        default_result_file = f"job_{job_id}_simulation_results.csv"
        result_path = Path(job_workspace) / default_result_file

        mod.simulate(resultfile=Path(result_path).as_posix())

        if not result_path.is_file():
            raise FileNotFoundError(
                f"Simulation for job {job_id} failed to produce result file at {result_path}"
            )

        logger.info(f"Job {job_id} finished. Results at {result_path}")

        optimal_param = 0.0

        # DONE START: Check if optimization is enabled, if so call _run_bisection_search_for_job for optimization
        if _is_optimization_enabled(config):
            job_config = config.copy()
            job_config["paths"]["package_path"] = package_path
            job_config["paths"]["temp_dir"] = base_temp_dir
            job_config["simulation_parameters"] = job_params
            optimal_param, optimal_value = _run_bisection_search_for_job(
                job_config, job_id_prefix=f"job_{job_id}"
            )
            logger.info(
                f"Job {job_id} optimization complete. Optimal blanket.TBR: {optimal_param} , Optimal Self_Sufficiency_Time: {optimal_value}"
            )
        # DONE END

        return optimal_param, optimal_value, str(result_path)
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        return optimal_param, optimal_value, ""
    finally:
        if omc:
            omc.sendExpression("quit()")


def _run_sequential_sweep(config: dict, jobs: List[Dict[str, Any]]) -> List[str]:
    """
    Executes a parameter sweep sequentially, reusing the OM session for efficiency.
    Saves intermediate results to the timestamped temporary directory.
    """
    paths_config = config["paths"]
    sim_config = config["simulation"]

    base_temp_dir = os.path.abspath(paths_config.get("temp_dir", "temp"))
    os.makedirs(base_temp_dir, exist_ok=True)

    logger.info(
        f"Running sweep sequentially. Intermediate files will be in: {base_temp_dir}"
    )

    omc = None
    result_paths = []
    try:
        omc = get_om_session()
        package_path = os.path.abspath(paths_config["package_path"])
        if not load_modelica_package(omc, Path(package_path).as_posix()):
            raise RuntimeError("Failed to load Modelica package for sequential sweep.")

        mod = ModelicaSystem(
            fileName=Path(package_path).as_posix(),
            modelName=sim_config["model_name"],
            variableFilter=sim_config["variableFilter"],
        )
        mod.setSimulationOptions(
            [
                f"stopTime={sim_config['stop_time']}",
                "tolerance=1e-6",
                "outputFormat=csv",
                f"stepSize={sim_config['step_size']}",
            ]
        )
        mod.buildModel()

        final_results = []

        for i, job_params in enumerate(jobs):
            try:
                logger.info(
                    f"Running sequential job {i+1}/{len(jobs)} with parameters: {job_params}"
                )
                param_settings = [
                    format_parameter_value(name, value)
                    for name, value in job_params.items()
                ]
                if param_settings:
                    mod.setParameters(param_settings)

                job_workspace = os.path.join(base_temp_dir, f"job_{i+1}")
                os.makedirs(job_workspace, exist_ok=True)
                result_filename = f"job_{i+1}_simulation_results.csv"
                result_file_path = os.path.join(job_workspace, result_filename)

                mod.simulate(resultfile=Path(result_file_path).as_posix())

                logger.info(
                    f"Sequential job {i+1} finished. Results at {result_file_path}"
                )
                result_paths.append(result_file_path)

                if _is_optimization_enabled(config):
                    job_config = config.copy()
                    job_config["simulation_parameters"] = job_params
                    optimal_param, optimal_value = _run_bisection_search_for_job(
                        job_config, job_id_prefix=f"job_{i+1}"
                    )

                    final_result_entry = job_params.copy()
                    final_result_entry["Required_TBR"] = optimal_param
                    final_result_entry["Required_Self_Sufficiency_Time"] = optimal_value
                    final_results.append(final_result_entry)

            except Exception as e:
                logger.error(f"Sequential job {i+1} failed: {e}", exc_info=True)
                result_paths.append("")

        # Summarize optimization results
        _save_optimization_summary(config, final_results)

        return result_paths
    except Exception as e:
        logger.error(f"Sequential sweep failed during setup: {e}", exc_info=True)
        return [""] * len(jobs)
    finally:
        if omc:
            omc.sendExpression("quit()")


def run_simulation(config: Dict[str, Any]):
    """Orchestrates the simulation execution, result handling, and cleanup."""

    # --- START: Check if analysis_cases need to be processed ---

    has_analysis_cases = (
        "sensitivity_analysis" in config
        and "analysis_cases" in config["sensitivity_analysis"]
        and (
            # Support list format
            (
                isinstance(config["sensitivity_analysis"]["analysis_cases"], list)
                and len(config["sensitivity_analysis"]["analysis_cases"]) > 0
            )
            or
            # Support single object format
            isinstance(config["sensitivity_analysis"]["analysis_cases"], dict)
        )
    )

    if has_analysis_cases:
        logger.info(
            "Detected analysis_cases field, starting to create independent working directories for each analysis case..."
        )

        # Create independent working directories and configuration files for each analysis_case
        case_configs = _setup_analysis_cases_workspaces(config)

        if not case_configs:
            logger.error(
                "Unable to create analysis_cases working directories, stopping execution"
            )
            return

        logger.info(f"Starting execution of {len(case_configs)} analysis cases...")

        # Execute simulation for each case
        successful_cases = 0
        for case_info in case_configs:
            try:
                case_index = case_info["index"]
                case_workspace = case_info["workspace"]
                case_config = case_info["config"]
                case_data = case_info["case_data"]

                logger.info(
                    f"\n=== Starting execution of analysis case {case_index + 1}/{len(case_configs)} ==="
                )
                logger.info(
                    f"Case name: {case_data.get('name', f'Case{case_index+1}')}"
                )
                logger.info(
                    f"Independent variable: {case_data['independent_variable']}"
                )
                logger.info(f"Working directory: {case_workspace}")

                # Set current working directory as Python's working directory
                original_cwd = os.getcwd()
                os.chdir(case_workspace)

                try:
                    # Recursively call run_simulation to execute single case
                    run_simulation(case_config)
                    successful_cases += 1
                    logger.info(
                        f"✓ Analysis case {case_index + 1} executed successfully"
                    )

                except Exception as case_e:
                    logger.error(
                        f"✗ Analysis case {case_index + 1} execution failed: {case_e}",
                        exc_info=True,
                    )

                finally:
                    # Restore original working directory
                    os.chdir(original_cwd)

            except Exception as e:
                logger.error(
                    f"✗ Error processing analysis case {case_index + 1}: {e}",
                    exc_info=True,
                )

        logger.info("\n=== Analysis Cases Execution Completed ===")
        logger.info(
            f"Successfully executed: {successful_cases}/{len(case_configs)} cases"
        )

        # Generate summary report
        _generate_analysis_cases_summary(case_configs, config)

        return  # End analysis_cases processing

    jobs = generate_simulation_jobs(config.get("simulation_parameters", {}))

    try:
        results_dir = os.path.abspath(config["paths"]["results_dir"])
    except KeyError as e:
        logger.error(f"Missing required path key in configuration file: {e}")
        sys.exit(1)

    simulation_results = {}
    use_concurrent = config["simulation"].get("concurrent", True)

    try:
        if config.get("co_simulation") is None:
            if use_concurrent:
                logger.info("Starting simulation in CONCURRENT mode.")
                max_workers = config["simulation"].get("max_workers", os.cpu_count())
                logger.info(f"Using up to {max_workers} parallel workers.")
                final_results = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    future_to_job = {
                        executor.submit(
                            _run_single_job, config, job_params, i + 1
                        ): job_params
                        for i, job_params in enumerate(jobs)
                    }
                    for future in concurrent.futures.as_completed(future_to_job):
                        job_params = future_to_job[future]
                        try:
                            optimal_param, optimal_value, result_path = future.result()
                            if result_path:
                                simulation_results[
                                    tuple(sorted(job_params.items()))
                                ] = result_path
                        except Exception as exc:
                            logger.error(
                                f"Job for {job_params} generated an exception: {exc}",
                                exc_info=True,
                            )
                        final_result_entry = job_params.copy()
                        final_result_entry["Required_TBR"] = optimal_param
                        final_result_entry["Required_Self_Sufficiency_Time"] = (
                            optimal_value
                        )
                        final_results.append(final_result_entry)

                _save_optimization_summary(config, final_results)
            else:
                logger.info("Starting simulation in SEQUENTIAL mode.")
                result_paths = _run_sequential_sweep(config, jobs)
                for i, result_path in enumerate(result_paths):
                    if result_path:
                        simulation_results[tuple(sorted(jobs[i].items()))] = result_path
        else:
            if use_concurrent:
                logger.info("Starting co-simulation in CONCURRENT mode.")
                max_workers = config["simulation"].get("max_workers", 4)
                logger.info(f"Using up to {max_workers} parallel processes.")

                final_results = []

                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    future_to_job = {
                        executor.submit(
                            _run_co_simulation, config, job_params, job_id=i + 1
                        ): job_params
                        for i, job_params in enumerate(jobs)
                    }

                    for future in concurrent.futures.as_completed(future_to_job):
                        job_params = future_to_job[future]
                        try:
                            optimal_param, optimal_value, result_path = future.result()
                            if result_path:
                                simulation_results[
                                    tuple(sorted(job_params.items()))
                                ] = result_path
                                logger.info(
                                    f"Successfully finished job for params: {job_params}"
                                )
                            else:
                                logger.warning(
                                    f"Job for params {job_params} did not return a result path."
                                )
                        except Exception as exc:
                            logger.error(
                                f"Job for params {job_params} generated an exception: {exc}",
                                exc_info=True,
                            )
                        final_result_entry = job_params.copy()
                        final_result_entry["Required_TBR"] = optimal_param
                        final_result_entry["Required_Self_Sufficiency_Time"] = (
                            optimal_value
                        )
                        final_results.append(final_result_entry)
            else:
                logger.info("Starting co-simulation in SEQUENTIAL mode.")
                for i, job_params in enumerate(jobs):
                    job_id = i + 1
                    logger.info(f"--- Starting Sequential Job {job_id}/{len(jobs)} ---")
                    try:
                        optimal_param, optimal_value, result_path = _run_co_simulation(
                            config, job_params, job_id=job_id
                        )
                        if result_path:
                            simulation_results[tuple(sorted(job_params.items()))] = (
                                result_path
                            )
                            logger.info(
                                f"Successfully finished job for params: {job_params}"
                            )
                        else:
                            logger.warning(
                                f"Job for params {job_params} did not return a result path."
                            )
                        final_result_entry = job_params.copy()
                        final_result_entry["Required_TBR"] = optimal_param
                        final_result_entry["Required_Self_Sufficiency_Time"] = (
                            optimal_value
                        )
                        final_results.append(final_result_entry)
                    except Exception as exc:
                        logger.error(
                            f"Job for params {job_params} generated an exception: {exc}",
                            exc_info=True,
                        )
                    logger.info(f"--- Finished Sequential Job {job_id}/{len(jobs)} ---")

            _save_optimization_summary(config, final_results)
    except Exception as e:
        raise RuntimeError(f"Failed to run simualtion: {e}")

    # --- Result Handling ---
    run_results_dir = results_dir
    os.makedirs(run_results_dir, exist_ok=True)

    # Unified result processing for both single and multiple jobs
    logger.info(f"Processing {len(jobs)} job(s). Combining results.")
    combined_df = None

    all_dfs = []
    time_df_added = False

    for job_params in jobs:
        job_key = tuple(sorted(job_params.items()))
        result_path = simulation_results.get(job_key)

        if not result_path or not os.path.exists(result_path):
            logger.warning(f"Job {job_params} produced no result file. Skipping.")
            continue

        # Read the current job's result file
        df = pd.read_csv(result_path)

        # From the very first valid DataFrame, grab the 'time' column
        if not time_df_added and "time" in df.columns:
            all_dfs.append(df[["time"]])
            time_df_added = True

        # Prepare the parameter string for column renaming
        param_string = "&".join([f"{k}={v}" for k, v in job_params.items()])

        # Isolate the data columns (everything except 'time')
        data_columns = df.drop(columns=["time"], errors="ignore")

        # Create a dictionary to map old column names to new ones
        # e.g., {'voltage': 'voltage&param1=A&param2=B'}
        rename_mapping = {col: f"{col}&{param_string}" for col in data_columns.columns}

        # Rename the columns and add the resulting DataFrame to our list
        all_dfs.append(data_columns.rename(columns=rename_mapping))

    # Concatenate all the DataFrames in the list along the columns axis (axis=1)
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=1)
    else:
        combined_df = pd.DataFrame()  # Or None, as you had before

    if combined_df is not None and not combined_df.empty:
        if len(jobs) == 1:
            # For single job, save as simulation_result.csv
            combined_csv_path = get_unique_filename(
                run_results_dir, "simulation_result.csv"
            )
        else:
            # For multiple jobs, save as sweep_results.csv
            combined_csv_path = get_unique_filename(
                run_results_dir, "sweep_results.csv"
            )

        combined_df.to_csv(combined_csv_path, index=False)
        logger.info(f"Combined results saved to: {combined_csv_path}")
    else:
        logger.warning("No valid results found to combine.")

    # Check if sweep_time plotting is enabled
    analysis_case = config["sensitivity_analysis"].get("analysis_case", {})
    if analysis_case.get("sweep_time", False):
        # Get parameters for plot_sweep_time_series
        independent_var = analysis_case.get("independent_variable")
        dependent_vars = analysis_case.get("dependent_variables", [])
        independent_var_alias = analysis_case.get("independent_variable_alias")

        if (
            independent_var
            and dependent_vars
            and combined_csv_path
            and os.path.exists(combined_csv_path)
        ):

            try:
                plot_path = plot_sweep_time_series(
                    csv_path=combined_csv_path,
                    save_dir=run_results_dir,
                    y_var_name="sds.I[1]",
                    independent_var_name=independent_var,
                    independent_var_alias=independent_var_alias,
                )
                if plot_path:
                    logger.info(f"Sweep time series plot generated: {plot_path}")
                else:
                    logger.warning("Failed to generate sweep time series plot")
            except Exception as e:
                logger.error(f"Error generating sweep time series plot: {e}")

    # --- Sensitivity Analysis ---
    _run_sensitivity_analysis(config, run_results_dir, jobs)

    # --- Final Cleanup ---
    if not config["simulation"].get("keep_temp_files", True):
        logger.info("Cleaning up temporary directory...")
        temp_dir_path = os.path.abspath(config["paths"].get("temp_dir", "temp"))
        if os.path.exists(temp_dir_path):
            try:
                shutil.rmtree(temp_dir_path)
                os.makedirs(temp_dir_path)  # Recreate for next run
            except OSError as e:
                logger.error(f"Error cleaning up temp directory: {e}")


def initialize_run() -> Dict[str, Any]:
    """
    Parses command-line arguments, loads the config file, and generates a run timestamp.
    Returns a fully prepared configuration dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Run a unified simulation and co-simulation workflow in parallel."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        default="config.json",
        help="Path to the JSON configuration file.",
    )
    args = parser.parse_args()

    try:
        config_path = os.path.abspath(args.config)
        with open(config_path, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Logger is not set up yet, so print directly to stderr
        print(
            f"ERROR: Failed to load or parse config file {args.config}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Generate a single timestamp for the entire run and add it to the config
    config["run_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config


def main():
    """Main function to run the simulation from the command line."""
    config = initialize_run()
    setup_logging(config)
    logger.info(f"Loading configuration from: {os.path.abspath(sys.argv[-1])}")
    if _validate_analysis_cases_config(config):
        try:
            run_simulation(config)
            logger.info("Main execution completed successfully.")
        except Exception as e:
            logger.error(f"Main execution failed: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
