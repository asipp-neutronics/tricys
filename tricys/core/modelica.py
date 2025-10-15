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
        OMCSessionZMQ: An active OpenModelica session object.
    """
    logger.debug("Initializing new OMCSessionZMQ session.")
    return OMCSessionZMQ()


def load_modelica_package(omc: OMCSessionZMQ, package_path: str) -> bool:
    """Loads a Modelica package into the OpenModelica session.

    Args:
        omc (OMCSessionZMQ): The active OpenModelica session object.
        package_path (str): The file path to the Modelica package (`package.mo`).

    Returns:
        bool: True if the package was loaded successfully, False otherwise.
    """
    logger.info(f"Loading package: {package_path}")
    load_result = omc.sendExpression(f'loadFile("{package_path}")')
    if not load_result:
        logger.error(f"Failed to load package: {package_path}")
        return False
    return True


def get_model_parameter_names(omc: OMCSessionZMQ, model_name: str) -> List[str]:
    """Parses and returns all subcomponent parameter names for a given model.

    Args:
        omc (OMCSessionZMQ): The active OpenModelica session object.
        model_name (str): The full name of the model (e.g., 'example.Cycle').

    Returns:
        List[str]: A list of all available parameter names (e.g., ['blanket.TBR']).
    """
    logger.info(f"Getting parameter names for model '{model_name}'")
    all_params = []
    try:
        if not omc.sendExpression(f"isModel({model_name})"):
            logger.warning(f"Model '{model_name}' not found in package.")
            return []

        components = omc.sendExpression(f"getComponents({model_name})")
        if not components:
            logger.warning(f"No components found for {model_name}")
            return []

        for comp in components:
            comp_type, comp_name = comp[0], comp[1]
            if comp_type.startswith(model_name.split(".")[0]):
                params = omc.sendExpression(f"getParameterNames({comp_type})")
                for param in params:
                    full_param = f"{comp_name}.{param}"
                    if full_param not in all_params:
                        all_params.append(full_param)

        logger.info(f"Found {len(all_params)} parameter names.")
        return all_params

    except Exception as e:
        logger.error(f"Failed to get parameter names: {e}", exc_info=True)
        return []


def _recursive_get_parameters(
    omc: OMCSessionZMQ, class_name: str, path_prefix: str, params_list: list
):
    """A private helper function to recursively traverse a model and collect parameters.

    Args:
        omc (OMCSessionZMQ): The active OpenModelica session object.
        class_name (str): The name of the class/model to inspect.
        path_prefix (str): The hierarchical path prefix for the current component.
        params_list (list): The list to which parameter details are appended.
    """
    logger.debug(f"Recursively exploring: {class_name} with prefix: '{path_prefix}'")
    components = omc.sendExpression(f"getComponents({class_name})")
    if not components:
        return

    for comp in components:
        comp_type, comp_name, comp_comment = comp[0], comp[1], comp[2]
        comp_variability = comp[8]
        comp_dimensions = str(comp[11])  # Extract dimensions

        full_name = f"{path_prefix}.{comp_name}" if path_prefix else comp_name

        if comp_variability == "parameter":
            logger.debug(f"Found parameter: {full_name} of type {comp_type}")
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
                    f"Skipping non-example component: {full_name} ({comp_type})"
                )


def get_all_parameters_details(
    omc: OMCSessionZMQ, model_name: str
) -> List[Dict[str, Any]]:
    """Recursively retrieves detailed information for all parameters in a given model.

    Args:
        omc (OMCSessionZMQ): The active OpenModelica session object.
        model_name (str): The full name of the model.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            contains the detailed information of a single parameter.
    """
    logger.info(f"Getting detailed parameters for model '{model_name}' via recursion.")
    all_params_details = []
    try:
        if not omc.sendExpression(f"isModel({model_name})"):
            logger.error(f"Model '{model_name}' not found in package.")
            return []
        _recursive_get_parameters(omc, model_name, "", all_params_details)
        logger.info(
            f"Successfully found details for {len(all_params_details)} parameters."
        )
        return all_params_details
    except Exception as e:
        logger.error(
            f"Failed to get detailed parameters via recursion: {e}", exc_info=True
        )
        return []


def format_parameter_value(name: str, value: Any) -> str:
    """Formats a parameter value into a string recognized by OpenModelica.

    Args:
        name (str): The name of the parameter.
        value (Any): The value of the parameter.

    Returns:
        str: A formatted string for use in simulation overrides (e.g., "p=1.0").
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
    """Parses a string value from OpenModelica into a Python type."""
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
    """Retrieves the default values for all parameters in a given model,
    parsing them into appropriate Python types.

    This function leverages get_all_parameters_details to fetch detailed
    parameter information and then extracts and parses the name and default value
    into a dictionary.

    Args:
        omc (OMCSessionZMQ): The active OpenModelica session object.
        model_name (str): The full name of the model.

    Returns:
        Dict[str, Any]: A dictionary mapping parameter names to their
            default values (e.g., float, list, bool, str). Returns an empty
            dictionary if the model is not found or has no parameters.
    """
    logger.info(
        f"Getting and parsing default parameter values for model '{model_name}'."
    )

    # Use the existing detailed function to get all parameter info
    all_params_details = get_all_parameters_details(omc, model_name)

    if not all_params_details:
        logger.warning(f"No parameters found for model '{model_name}'.")
        return {}

    # Convert the list of dicts into a single dict of name: parsed_defaultValue
    default_params = {
        param["name"]: _parse_om_value(param["defaultValue"])
        for param in all_params_details
    }

    logger.info(
        f"Found and parsed {len(default_params)} default parameters for model '{model_name}'."
    )
    return default_params


def _clear_stale_init_xml(mod: ModelicaSystem, model_name: str):
    """
    查找 ModelicaSystem 的工作目录并删除残留的 <model_name>_init.xml 文件，
    以防止 GUID 不匹配的错误。

    Args:
        mod: OMPython.ModelicaSystem 的实例对象。
        model_name: 模型的名称 (例如 "CFEDR.Cycle")。
        logger: 用于记录日志的 logging.Logger 对象。
    """
    try:
        work_dir = ""
        try:
            # 1. 尝试使用标准方法获取工作目录
            work_dir = mod.getWorkDirectory()
        except AttributeError:
            # 2. 如果方法不存在，尝试访问 "私有" 属性
            logger.warning("'.getWorkDirectory()' not found, trying '._workDir'")
            work_dir = mod._workDir  # 很多 OMPython 版本使用这个

        if not work_dir or not os.path.isdir(work_dir):
            raise RuntimeError(
                f"Could not get a valid work_dir from mod object: {work_dir}"
            )

        logger.info(f"ModelicaSystem working directory is: {work_dir}")

        # 3. 构造旧文件的完整路径
        xml_file_name = f"{model_name}_init.xml"
        xml_file_path = os.path.join(work_dir, xml_file_name)

        # 4. 检查并删除它
        logger.info(f"Checking for stale init file at: {xml_file_path}")
        if os.path.exists(xml_file_path):
            logger.warning(
                f"Found and removing stale init file (old GUID): {xml_file_path}"
            )
            os.remove(xml_file_path)
        else:
            logger.info("No stale init file found. Proceeding to build.")

    except Exception as e:
        logger.error(
            f"Error during stale init file cleanup: {e}. "
            "This might not be critical, but GUID errors may occur.",
            exc_info=True,
        )
