"""Configuration utility functions for tricys."""

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from tricys.core.modelica import (
    get_model_default_parameters,
    get_om_session,
    load_modelica_package,
)

# Standard logger setup
logger = logging.getLogger(__name__)


def _search_dict(d: Any, key: str, value: Any) -> bool:
    """Recursively search for a key-value pair in a dictionary or list."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key and v == value:
                return True
            if isinstance(v, (dict, list)):
                if _search_dict(v, key, value):
                    return True
    elif isinstance(d, list):
        for item in d:
            if _search_dict(item, key, value):
                return True
    return False


def check_ai_config(config: Dict[str, Any]) -> None:
    """
    Checks for AI-related environment variables if 'ai: true' is found in the config.

    Args:
        config: The configuration dictionary.

    Raises:
        SystemExit: If AI is enabled in the config but required environment
                    variables are missing.

    Note:
        If any part of the configuration contains `"ai": true`, this function verifies
        that `API_KEY`, `BASE_URL`, and either `AI_MODEL` or `AI_MODELS` are set as
        environment variables.
    """
    if _search_dict(config, "ai", True):
        logger.info(
            "AI feature enabled in config, checking for required environment variables..."
        )
        load_dotenv()
        api_key = os.environ.get("API_KEY")
        base_url = os.environ.get("BASE_URL")
        ai_model = os.environ.get("AI_MODEL")
        ai_models = os.environ.get("AI_MODELS")

        missing_vars = []
        if not api_key:
            missing_vars.append("API_KEY")
        if not base_url:
            missing_vars.append("BASE_URL")
        if not ai_model and not ai_models:
            missing_vars.append("AI_MODEL or AI_MODELS")

        if missing_vars:
            print(
                f"ERROR: 'ai: true' is set in the configuration, but the following required environment variables are missing: {', '.join(missing_vars)}. "
                "Please set them in your environment or a .env file.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            logger.info("All required AI environment variables are present.")


# Define the structure of required configuration keys and their expected types
BASIC_REQUIRED_CONFIG_KEYS = {
    "paths": {
        "package_path": str,
    },
    "simulation": {
        "model_name": str,
        "stop_time": (int, float),
        "step_size": (int, float),
        "variableFilter": str,
    },
}


ANALYSIS_REQUIRED_CONFIG_KEYS = {
    "paths": {
        "package_path": str,
    },
    "simulation": {
        "model_name": str,
        "stop_time": (int, float),
        "step_size": (int, float),
        "variableFilter": str,
    },
    "sensitivity_analysis": {
        "enabled": bool,
    },
}


def convert_relative_paths_to_absolute(
    config: Dict[str, Any], base_dir: str
) -> Dict[str, Any]:
    """Recursively converts relative paths to absolute paths in configuration.

    Args:
        config: Configuration dictionary to process.
        base_dir: Base directory path for resolving relative paths.

    Returns:
        Configuration dictionary with converted absolute paths.

    Note:
        Processes path keys including package_path, db_path, results_dir, temp_dir,
        log_dir, glossary_path, and any key ending with '_path'. Converts relative
        paths to absolute using base_dir. Handles nested dictionaries and lists recursively.
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
                "glossary_path",
            ]

            if key_name.endswith("_path") or key_name in path_keys:
                # If it's a relative path, convert to absolute path
                if not os.path.isabs(value) and value:
                    abs_path = os.path.abspath(os.path.join(base_dir, value))
                    logger.debug(
                        "Converted path",
                        extra={
                            "key_name": key_name,
                            "original_value": value,
                            "absolute_path": abs_path,
                        },
                    )
                    return abs_path
            return value
        else:
            return value

    return _process_value(config)


def basic_validate_config(
    config: Dict[str, Any],
    required_keys: Dict = BASIC_REQUIRED_CONFIG_KEYS,
    parent_key: str = "",
) -> None:
    """Recursively validates the configuration against required structure.

    Args:
        config: Configuration dictionary to validate.
        required_keys: Dictionary defining required keys and their expected types.
        parent_key: Parent key path for nested validation (used internally).

    Raises:
        SystemExit: If validation fails (exits with code 1).

    Note:
        Performs structural validation (required keys and types) and value validation
        (path existence, variableFilter format). Uses regex to validate variableFilter
        against Modelica identifier patterns. Only validates values on top-level call.
    """
    # --- Structural Validation ---
    for key, expected_type_or_dict in required_keys.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key

        if key not in config:
            print(
                f"ERROR: Missing required configuration key: '{full_key_path}'",
                file=sys.stderr,
            )
            sys.exit(1)

        if isinstance(expected_type_or_dict, dict):
            if not isinstance(config[key], dict):
                print(
                    f"ERROR: Configuration key '{full_key_path}' must be a dictionary.",
                    file=sys.stderr,
                )
                sys.exit(1)
            # Recurse for nested dictionaries
            basic_validate_config(
                config[key], expected_type_or_dict, parent_key=full_key_path
            )
        else:
            # Perform type checking for leaf keys
            if not isinstance(config[key], expected_type_or_dict):
                print(
                    f"ERROR: Configuration key '{full_key_path}' has incorrect type. "
                    f"Expected {expected_type_or_dict}, but got {type(config[key])}.",
                    file=sys.stderr,
                )
                sys.exit(1)

    # --- Value Validation (only on top-level call) ---
    if not parent_key:
        # 1. Check if package_path exists
        package_path = config.get("paths", {}).get("package_path")
        if package_path and not os.path.exists(package_path):
            print(
                f"ERROR: File specified in 'paths.package_path' not found: {package_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 2. Validate variableFilter format
        variable_filter = config.get("simulation", {}).get("variableFilter")
        if variable_filter:
            # Regex for a valid Modelica identifier (simplified)
            ident = r"[a-zA-Z_][a-zA-Z0-9_]*"
            # Regex for a valid substring in the filter:
            # - time
            # - class.name
            # - class.name[index]
            # - class.name[start-end]
            valid_substring_re = re.compile(
                rf"^time$|^{ident}\.{ident}(\[\d+(-\d+)?\])?$"
            )

            substrings = variable_filter.split("|")
            for sub in substrings:
                if not valid_substring_re.match(sub):
                    print(
                        f"ERROR: Invalid format in 'simulation.variableFilter'. Substring '{sub}' does not match required format. "
                        f"Valid formats are 'time', 'classname.typename', 'classname.typename[1]', or 'classname.typename[1-5]'.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        check_ai_config(config)


def basic_prepare_config(config_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads and prepares the configuration from the given path.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A tuple of (runtime_config, original_config).

    Raises:
        SystemExit: If config file loading/parsing fails or validation fails.

    Note:
        Converts relative paths to absolute, validates config structure, adds run_timestamp,
        creates workspace directories, and processes variableFilter for regex escaping.
        Sets up log_dir, temp_dir, and results_dir within run workspace.
    """
    try:
        config_path = os.path.abspath(config_path)
        with open(config_path, "r") as f:
            base_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Logger is not set up yet, so print directly to stderr
        print(
            f"ERROR: Failed to load or parse config file {config_path}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    original_config_dir = os.path.dirname(config_path)

    absolute_config = convert_relative_paths_to_absolute(
        base_config, original_config_dir
    )

    # Perform all validation on the config with absolute paths
    basic_validate_config(absolute_config)

    config = json.loads(json.dumps(absolute_config))
    config["run_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_workspace = os.path.abspath(config["run_timestamp"])

    if "paths" not in config:
        config["paths"] = {}

    original_string = config["simulation"]["variableFilter"]
    config["simulation"]["variableFilter"] = original_string.replace(
        "[", "\\[["
    ).replace("]", "]\\]")

    config["paths"]["log_dir"] = os.path.join(
        run_workspace, base_config["paths"].get("log_dir", "log")
    )
    config["paths"]["temp_dir"] = os.path.join(
        run_workspace, base_config["paths"].get("temp_dir", "temp")
    )
    config["paths"]["results_dir"] = os.path.join(
        run_workspace, base_config["paths"].get("results_dir", "results")
    )

    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    os.makedirs(config["paths"]["temp_dir"], exist_ok=True)
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)

    return config, base_config


def analysis_validate_analysis_cases_config(config: Dict[str, Any]) -> bool:
    """Validates analysis_cases configuration format supporting both list and single object.

    This function validates:
    1. Basic structure and required fields of analysis_cases
    2. Simulation parameters compatibility (single job requirement)
    3. Required_TBR configuration completeness if used in dependent_variables

    Args:
        config: Configuration dictionary to validate.

    Returns:
        True if configuration is valid, False otherwise.

    Note:
        Supports both single analysis_case dict or list of cases. Required fields per case:
        name, independent_variable, independent_variable_sampling. Validates simulation_parameters
        contain only single job (no sweep). Checks Required_TBR completeness in metrics_definition.
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

    # Check if top-level simulation_parameters are used, which is disallowed in analysis_cases mode
    if config.get("simulation_parameters"):
        logger.error(
            "The top-level 'simulation_parameters' field cannot be used when 'analysis_cases' is defined. "
            "Please move any shared or case-specific parameters into the 'simulation_parameters' field "
            "inside each object within the 'analysis_cases' list."
        )
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


def analysis_validate_config(
    config: Dict[str, Any],
    required_keys: Dict = ANALYSIS_REQUIRED_CONFIG_KEYS,
    parent_key: str = "",
) -> None:
    """
    Recursively validates the configuration's structure and values.
    """
    # --- Structural Validation ---
    for key, expected in required_keys.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in config:
            print(
                f"ERROR: Missing required configuration key: '{full_key_path}'",
                file=sys.stderr,
            )
            sys.exit(1)

        if isinstance(expected, dict):
            if not isinstance(config[key], dict):
                print(
                    f"ERROR: Configuration key '{full_key_path}' must be a dictionary.",
                    file=sys.stderr,
                )
                sys.exit(1)
            analysis_validate_config(config[key], expected, parent_key=full_key_path)
        elif not isinstance(config[key], expected):
            print(
                f"ERROR: Configuration key '{full_key_path}' has incorrect type. Expected {expected}, got {type(config[key])}.",
                file=sys.stderr,
            )
            sys.exit(1)

    # 2. Validate variableFilter format
    variable_filter = config.get("simulation", {}).get("variableFilter")
    if variable_filter:
        # Regex for a valid Modelica identifier (simplified)
        ident = r"[a-zA-Z_][a-zA-Z0-9_]*"
        # Regex for a valid substring in the filter:
        # - time
        # - class.name
        # - class.name[index]
        # - class.name[start-end]
        valid_substring_re = re.compile(rf"^time$|^{ident}\.{ident}(\[\d+(-\d+)?\])?$")

        substrings = variable_filter.split("|")
        for sub in substrings:
            if not valid_substring_re.match(sub):
                print(
                    f"ERROR: Invalid format in 'simulation.variableFilter'. Substring '{sub}' does not match required format. "
                    f"Valid formats are 'time', 'classname.typename', 'classname.typename[1]', or 'classname.typename[1-5]'.",
                    file=sys.stderr,
                )
                sys.exit(1)

    # --- Value and Conditional Validation (only on top-level call) ---
    if not parent_key:
        # Check for package_path existence
        package_path = config.get("paths", {}).get("package_path")
        if package_path and not os.path.exists(package_path):
            print(
                f"ERROR: File specified in 'paths.package_path' not found: {package_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Analysis-specific validation
        sa_config = config.get("sensitivity_analysis", {})
        if sa_config.get("enabled", False):
            has_sim_params = (
                "simulation_parameters" in config and config["simulation_parameters"]
            )
            has_analysis_cases = (
                "analysis_cases" in sa_config and sa_config["analysis_cases"]
            )

            if not has_sim_params and not has_analysis_cases:
                print(
                    "ERROR: When 'sensitivity_analysis' is enabled, either 'simulation_parameters' or 'sensitivity_analysis.analysis_cases' must be defined.",
                    file=sys.stderr,
                )
                sys.exit(1)

            if has_analysis_cases:
                if not analysis_validate_analysis_cases_config(config):
                    # The original function uses a logger which is not yet configured.
                    # Add a print statement to ensure the user sees an error.
                    print(
                        "ERROR: 'analysis_cases' configuration is invalid. See previous logs for details.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        check_ai_config(config)


def analysis_setup_analysis_cases_workspaces(
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
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

    # The main run workspace is the timestamped directory, already created by initialize_run.
    # We will create the case workspaces inside it.
    run_workspace = os.path.abspath(config["run_timestamp"])

    # Determine the main log file path to be shared with all cases
    main_log_file_name = f"simulation_{config['run_timestamp']}.log"
    main_log_path = os.path.join(run_workspace, main_log_file_name)

    logger.info(
        f"Detected {len(analysis_cases)} analysis cases, creating independent workspaces inside: {run_workspace}"
    )

    case_configs = []

    for i, analysis_case in enumerate(analysis_cases):
        try:
            # Generate case working directory name
            workspace_name = analysis_case.get("name", f"case_{i}")
            # Create the case workspace directly inside the main run workspace
            case_workspace = os.path.join(run_workspace, workspace_name)
            os.makedirs(case_workspace, exist_ok=True)

            # Create standard configuration (inlined from _create_standard_config_for_case)
            base_config = config
            original_config_dir = os.path.dirname(
                base_config.get("paths", {}).get("package_path", os.getcwd())
            )
            absolute_config = convert_relative_paths_to_absolute(
                base_config, original_config_dir
            )
            standard_config = json.loads(json.dumps(absolute_config))

            # if analysis_case.get("name") == "SALib_Analysis":
            if isinstance(
                analysis_case.get("independent_variable"), list
            ) and isinstance(analysis_case.get("independent_variable_sampling"), dict):
                sensitivity_analysis = standard_config["sensitivity_analysis"]
                if "analysis_cases" in sensitivity_analysis:
                    del sensitivity_analysis["analysis_cases"]
                sensitivity_analysis["analysis_case"] = analysis_case.copy()
            else:
                # Get independent variable and sampling from the current analysis case
                independent_var = analysis_case["independent_variable"]
                independent_sampling = analysis_case["independent_variable_sampling"]
                logger.debug(
                    f"independent_sampling configuration: {independent_sampling}"
                )

                # Ensure simulation_parameters exists at the top level
                if "simulation_parameters" not in standard_config:
                    standard_config["simulation_parameters"] = {}

                # If the specific analysis_case has its own simulation_parameters, merge them into the top-level ones
                # This allows for case-specific parameter overrides or additions
                if "simulation_parameters" in analysis_case:
                    case_sim_params = analysis_case.get("simulation_parameters", {})

                    # Identify and handle virtual parameters (e.g., Required_TBR) used for metric configuration
                    virtual_params = {
                        k: v
                        for k, v in case_sim_params.items()
                        if k.startswith("Required_") and isinstance(v, dict)
                    }

                    if virtual_params:
                        # Merge virtual parameter config into the case's metrics_definition
                        metrics_def = standard_config.setdefault(
                            "sensitivity_analysis", {}
                        ).setdefault("metrics_definition", {})
                        for key, value in virtual_params.items():
                            if key in metrics_def:
                                metrics_def[key].update(value)
                            else:
                                metrics_def[key] = value

                    # Get real parameters by excluding virtual ones
                    real_params = {
                        k: v
                        for k, v in case_sim_params.items()
                        if k not in virtual_params
                    }

                    # Update standard_config's simulation_parameters with only real parameters for job generation
                    standard_config["simulation_parameters"].update(real_params)

                # Fetch default values for both independent and simulation parameters
                omc = None
                try:
                    # Get all sim params from the case, which may include virtual parameters
                    all_case_sim_params = analysis_case.get("simulation_parameters", {})
                    # Filter out virtual parameters before fetching default values
                    sim_param_keys = [
                        k
                        for k, v in all_case_sim_params.items()
                        if not (k.startswith("Required_") and isinstance(v, dict))
                    ]
                    # Ensure independent_var is a list for consistent processing, as it can be a list in SALib cases
                    ind_param_keys = (
                        [independent_var]
                        if isinstance(independent_var, str)
                        else independent_var
                    )

                    param_keys_to_fetch = sim_param_keys + ind_param_keys

                    if param_keys_to_fetch:
                        logger.info(
                            f"Fetching default values for parameters: {param_keys_to_fetch}"
                        )
                        omc = get_om_session()
                        if load_modelica_package(
                            omc,
                            Path(standard_config["paths"]["package_path"]).as_posix(),
                        ):
                            all_defaults = get_model_default_parameters(
                                omc, standard_config["simulation"]["model_name"]
                            )

                            # Helper function to handle array access like 'param[1]'
                            def get_specific_default(key, defaults):
                                if key in defaults:
                                    return defaults[key]
                                if "[" in key and key.endswith("]"):
                                    try:
                                        base_name, index_str = key.rsplit("[", 1)
                                        # Modelica is 1-based, Python is 0-based
                                        index = int(index_str[:-1]) - 1
                                        if base_name in defaults:
                                            default_array = defaults[base_name]
                                            if isinstance(
                                                default_array, list
                                            ) and 0 <= index < len(default_array):
                                                return default_array[index]
                                    except (ValueError, IndexError):
                                        pass  # Malformed index or out of bounds
                                return "N/A"

                            # Get defaults for simulation_parameters
                            default_sim_values = {
                                p: get_specific_default(p, all_defaults)
                                for p in sim_param_keys
                            }
                            analysis_case["default_simulation_values"] = (
                                default_sim_values
                            )

                            # Get defaults for independent_variable
                            default_ind_values = {
                                p: get_specific_default(p, all_defaults)
                                for p in ind_param_keys
                            }
                            analysis_case["default_independent_values"] = (
                                default_ind_values
                            )

                except Exception as e:
                    logger.warning(
                        f"Could not fetch default parameter values. Defaults will be empty. Error: {e}"
                    )
                    analysis_case["default_simulation_values"] = {}
                    analysis_case["default_independent_values"] = {}
                finally:
                    if omc:
                        omc.sendExpression("quit()")

                # Add the primary independent_variable_sampling for the current analysis case
                standard_config["simulation_parameters"][
                    independent_var
                ] = independent_sampling

                # Update sensitivity_analysis configuration
                sensitivity_analysis = standard_config["sensitivity_analysis"]

                # Remove analysis_cases and replace with single analysis_case
                if "analysis_cases" in sensitivity_analysis:
                    del sensitivity_analysis["analysis_cases"]

                sensitivity_analysis["analysis_case"] = analysis_case.copy()

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
            if "paths" in case_config and "log_dir" in case_config["paths"]:
                case_config["paths"]["log_dir"] = os.path.join(case_workspace, "log")
                # Inject the main log path for dual logging
                if "logging" in case_config:
                    case_config["logging"]["main_log_path"] = main_log_path

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

            logger.info(
                f"Workspace for case {i+1} created successfully",
                extra={
                    "case_index": i,
                    "case_name": analysis_case.get("name", f"case_{i}"),
                    "workspace": case_workspace,
                    "config_path": config_file_path,
                },
            )

        except Exception as e:
            logger.error(f"✗ Error processing case {i}: {e}", exc_info=True)
            continue

    logger.info(
        f"Successfully created independent working directories for {len(case_configs)} analysis cases"
    )
    return case_configs


def analysis_prepare_config(config_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads, validates, and prepares the configuration from the given path."""
    try:
        config_path = os.path.abspath(config_path)
        with open(config_path, "r") as f:
            base_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(
            f"ERROR: Failed to load or parse config file {config_path}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    original_config_dir = os.path.dirname(config_path)
    absolute_config = convert_relative_paths_to_absolute(
        base_config, original_config_dir
    )

    # Perform all validation on the config with absolute paths
    analysis_validate_config(absolute_config)

    config = json.loads(json.dumps(absolute_config))
    config["run_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_workspace = os.path.abspath(config["run_timestamp"])

    config["paths"]["log_dir"] = run_workspace
    if "paths" not in config:
        config["paths"] = {}

    original_string = config["simulation"]["variableFilter"]
    config["simulation"]["variableFilter"] = original_string.replace(
        "[", "\\[["
    ).replace("]", "]\\]")

    return config, base_config
