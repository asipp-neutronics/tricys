"""Utilities for interacting with OpenModelica via OMPython.

This module provides a set of functions to manage an OpenModelica session,
load models, retrieve parameter details, and format parameter values for
simulation.
"""

import logging
from typing import Any, Dict, List

from OMPython import OMCSessionZMQ

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
