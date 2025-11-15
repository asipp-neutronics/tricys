"""Utilities for interacting with OpenModelica via OMPython.

This module provides a set of functions to manage an OpenModelica session,
load models, retrieve parameter details, and format parameter values for
simulation.
"""

import logging
import os
from typing import Any, Dict, List

from OMPython import ModelicaSystem, OMCSessionZMQ

logger = logging.getLogger(__name__)


def get_om_session() -> OMCSessionZMQ:
    """Initializes and returns a new OMCSessionZMQ session.

    Returns:
        An active OpenModelica session object.

    Note:
        Creates a new ZMQ-based connection to OpenModelica Compiler. Each call
        creates an independent session that should be properly closed after use.
    """
    logger.debug("Initializing new OMCSessionZMQ session")
    return OMCSessionZMQ()


def load_modelica_package(omc: OMCSessionZMQ, package_path: str) -> bool:
    """Loads a Modelica package into the OpenModelica session.

    Args:
        omc: The active OpenModelica session object.
        package_path: The file path to the Modelica package (`package.mo`).

    Returns:
        True if the package was loaded successfully, False otherwise.

    Note:
        Uses sendExpression('loadFile(...)') command. Logs error if loading fails.
        The package must be a valid Modelica package file.
    """
    logger.info("Loading package", extra={"package_path": package_path})
    load_result = omc.sendExpression(f'loadFile("{package_path}")')
    if not load_result:
        logger.error("Failed to load package", extra={"package_path": package_path})
        return False
    return True


def get_model_parameter_names(omc: OMCSessionZMQ, model_name: str) -> List[str]:
    """Parses and returns all subcomponent parameter names for a given model.

    Args:
        omc: The active OpenModelica session object.
        model_name: The full name of the model (e.g., 'example.Cycle').

    Returns:
        A list of all available parameter names in hierarchical format
        (e.g., ['blanket.TBR', 'divertor.heatLoad']).

    Note:
        Only traverses components whose type starts with the package name.
        Returns empty list if model is not found or has no components. Uses
        getComponents() and getParameterNames() OMC API calls.
    """
    logger.info("Getting parameter names for model", extra={"model_name": model_name})
    all_params = []
    try:
        if not omc.sendExpression(f"isModel({model_name})"):
            logger.warning(
                "Model not found in package", extra={"model_name": model_name}
            )
            return []

        components = omc.sendExpression(f"getComponents({model_name})")
        if not components:
            logger.warning(
                "No components found for model", extra={"model_name": model_name}
            )
            return []

        for comp in components:
            comp_type, comp_name = comp[0], comp[1]
            if comp_type.startswith(model_name.split(".")[0]):
                params = omc.sendExpression(f"getParameterNames({comp_type})")
                for param in params:
                    full_param = f"{comp_name}.{param}"
                    if full_param not in all_params:
                        all_params.append(full_param)

        logger.info("Found parameter names", extra={"count": len(all_params)})
        return all_params

    except Exception as e:
        logger.error(
            "Failed to get parameter names", exc_info=True, extra={"error": str(e)}
        )
        return []


def _recursive_get_parameters(
    omc: OMCSessionZMQ,
    class_name: str,
    path_prefix: str,
    params_list: List[Dict[str, Any]],
) -> None:
    """A private helper function to recursively traverse a model and collect parameters.

    Args:
        omc: The active OpenModelica session object.
        class_name: The name of the class/model to inspect.
        path_prefix: The hierarchical path prefix for the current component.
        params_list: The list to which parameter details are appended (modified in-place).

    Note:
        Recursively explores models and blocks that belong to the same package.
        Collects parameter name, type, default value, comment, and dimensions.
        Only descends into components from the same package (checked by prefix matching).
    """
    logger.debug(
        "Recursively exploring model",
        extra={
            "class_name": class_name,
            "path_prefix": path_prefix,
        },
    )
    components = omc.sendExpression(f"getComponents({class_name})")
    if not components:
        return

    for comp in components:
        comp_type, comp_name, comp_comment = comp[0], comp[1], comp[2]
        comp_variability = comp[8]
        comp_dimensions = str(comp[11])  # Extract dimensions

        full_name = f"{path_prefix}.{comp_name}" if path_prefix else comp_name

        if comp_variability == "parameter":
            logger.debug(
                "Found parameter",
                extra={
                    "full_name": full_name,
                    "type": comp_type,
                },
            )
            param_value = omc.sendExpression(
                f'getParameterValue(stringTypeName("{class_name}"), "{comp_name}")'
            )
            params_list.append(
                {
                    "name": full_name,
                    "type": comp_type,
                    "defaultValue": param_value,
                    "comment": comp_comment,
                    "dimensions": comp_dimensions,  # Add dimensions to the dictionary
                }
            )
        elif comp_variability != "parameter" and omc.sendExpression(
            f"isModel({comp_type})"
        ):
            if comp_type.startswith(class_name.split(".")[0]):
                logger.debug(f"Descending into component: {full_name} ({comp_type})")
                _recursive_get_parameters(omc, comp_type, full_name, params_list)
            else:
                logger.debug(
                    f"Skipping non-example component: {full_name} ({comp_type})"
                )
        elif comp_variability != "parameter" and omc.sendExpression(
            f"isBlock({comp_type})"
        ):
            if comp_type.startswith(class_name.split(".")[0]):
                logger.debug(f"Descending into component: {full_name} ({comp_type})")
                _recursive_get_parameters(omc, comp_type, full_name, params_list)
            else:
                logger.debug(
                    "Skipping non-example component",
                    extra={
                        "full_name": full_name,
                        "type": comp_type,
                    },
                )


def get_all_parameters_details(
    omc: OMCSessionZMQ, model_name: str
) -> List[Dict[str, Any]]:
    """Recursively retrieves detailed information for all parameters in a given model.

    Args:
        omc: The active OpenModelica session object.
        model_name: The full name of the model.

    Returns:
        A list of dictionaries, where each dictionary contains the detailed
        information of a single parameter including name, type, defaultValue,
        comment, and dimensions.

    Note:
        Uses _recursive_get_parameters() to traverse the model hierarchy.
        Returns empty list if model is not found or on error. Each parameter
        dict includes 'name' (hierarchical), 'type', 'defaultValue', 'comment',
        and 'dimensions' fields.
    """
    logger.info(
        "Getting detailed parameters via recursion", extra={"model_name": model_name}
    )
    all_params_details = []
    try:
        if not omc.sendExpression(f"isModel({model_name})"):
            logger.error("Model not found in package", extra={"model_name": model_name})
            return []
        _recursive_get_parameters(omc, model_name, "", all_params_details)
        logger.info(
            "Successfully found parameter details",
            extra={"count": len(all_params_details)},
        )
        return all_params_details
    except Exception as e:
        logger.error(
            "Failed to get detailed parameters via recursion",
            exc_info=True,
            extra={"error": str(e)},
        )
        return []


def format_parameter_value(name: str, value: Any) -> str:
    """Formats a parameter value into a string recognized by OpenModelica.

    Args:
        name: The name of the parameter.
        value: The value of the parameter (can be number, string, list, or bool).

    Returns:
        A formatted string for use in simulation overrides (e.g., "p=1.0",
        "name={1,2,3}", or 'path="value"').

    Note:
        Lists are formatted as {v1,v2,...}. Strings are quoted with double quotes.
        Numbers and booleans use direct string conversion.
    """
    if isinstance(value, list):
        # Format lists as {v1,v2,...}
        return f"{name}={{{','.join(map(str, value))}}}"
    elif isinstance(value, str):
        # Format strings as "value"
        return f'{name}="{value}"'
    # For numbers and booleans, direct string conversion is fine
    return f"{name}={value}"


def _parse_om_value(value_str: str) -> Any:
    """Parses a string value from OpenModelica into a Python type.

    Args:
        value_str: The string value from OpenModelica to parse.

    Returns:
        Parsed value as appropriate Python type (float, bool, str, list, or original).

    Note:
        Handles OpenModelica formats: arrays "{v1,v2,...}", booleans "true"/"false",
        quoted strings '"..."', and numeric values. Recursively parses array elements.
        Returns original value if not a string or if parsing fails.
    """
    if not isinstance(value_str, str):
        return value_str  # Already parsed or not a string

    value_str = value_str.strip()

    # Handle lists/arrays: "{v1,v2,...}"
    if value_str.startswith("{") and value_str.endswith("}"):
        elements_str = value_str[1:-1]
        if not elements_str:
            return []
        # Split and recursively parse each element
        return [_parse_om_value(elem) for elem in elements_str.split(",")]

    # Handle booleans: "true" or "false"
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Handle strings: '"...some string..."'
    if value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]

    # Handle numbers (try float conversion)
    try:
        return float(value_str)
    except (ValueError, TypeError):
        # If all parsing fails, return the original string
        return value_str


def get_model_default_parameters(omc: OMCSessionZMQ, model_name: str) -> Dict[str, Any]:
    """Retrieves the default values for all parameters in a given model.

    This function leverages get_all_parameters_details to fetch detailed
    parameter information and then extracts and parses the name and default value
    into a dictionary.

    Args:
        omc: The active OpenModelica session object.
        model_name: The full name of the model.

    Returns:
        A dictionary mapping parameter names to their default values
        (e.g., float, list, bool, str). Returns an empty dictionary if
        the model is not found or has no parameters.

    Note:
        Values are parsed from OpenModelica string format to Python types using
        _parse_om_value(). Handles arrays, booleans, strings, and numeric values.
    """
    logger.info(
        "Getting and parsing default parameter values", extra={"model_name": model_name}
    )

    # Use the existing detailed function to get all parameter info
    all_params_details = get_all_parameters_details(omc, model_name)

    if not all_params_details:
        logger.warning(
            "No parameters found for model", extra={"model_name": model_name}
        )
        return {}

    # Convert the list of dicts into a single dict of name: parsed_defaultValue
    default_params = {
        param["name"]: _parse_om_value(param["defaultValue"])
        for param in all_params_details
    }

    logger.info(
        "Found and parsed default parameters",
        extra={
            "count": len(default_params),
            "model_name": model_name,
        },
    )
    return default_params


def _clear_stale_init_xml(mod: ModelicaSystem, model_name: str) -> None:
    """Find and delete residual <model_name>_init.xml file to prevent GUID mismatch errors.

    Args:
        mod: An instance object of OMPython.ModelicaSystem.
        model_name: The name of the model (e.g., "CFEDR.Cycle").

    Note:
        Locates ModelicaSystem's working directory and removes stale initialization
        XML files that can cause GUID errors. Tries getWorkDirectory() method first,
        falls back to _workDir attribute. Errors are logged but not raised as they
        may not be critical.
    """
    try:
        work_dir = ""
        try:
            work_dir = mod.getWorkDirectory()
        except AttributeError:
            logger.warning("getWorkDirectory() not found, trying ._workDir")
            work_dir = mod._workDir  # 很多 OMPython 版本使用这个

        if not work_dir or not os.path.isdir(work_dir):
            raise RuntimeError(
                f"Could not get a valid work_dir from mod object: {work_dir}"
            )

        logger.info("ModelicaSystem working directory", extra={"directory": work_dir})

        xml_file_name = f"{model_name}_init.xml"
        xml_file_path = os.path.join(work_dir, xml_file_name)

        logger.info("Checking for stale init file", extra={"file_path": xml_file_path})
        if os.path.exists(xml_file_path):
            logger.warning(
                "Found and removing stale init file (old GUID)",
                extra={"file_path": xml_file_path},
            )
            os.remove(xml_file_path)
        else:
            logger.info("No stale init file found. Proceeding to build.")

    except Exception as e:
        logger.error(
            "Error during stale init file cleanup",
            exc_info=True,
            extra={
                "error": str(e),
                "note": "This might not be critical, but GUID errors may occur.",
            },
        )
