"""本模块提供与OpenModelica交互的实用功能。"""

import logging
from typing import Any, Dict, List

from OMPython import OMCSessionZMQ

from tricys.manager.logger_manager import logger_manager
logger = logging.getLogger(__name__)


def get_om_session() -> OMCSessionZMQ:
    """初始化并返回一个新的OMCSessionZMQ会话。"""
    logger.debug("Initializing new OMCSessionZMQ session.")
    return OMCSessionZMQ()


def load_modelica_package(omc: OMCSessionZMQ, package_path) -> bool:
    """
    加载全局配置中指定的Modelica包。

    参数:
        omc (OMCSessionZMQ): OpenModelica会话对象。
        package_path (str): Modelica包的路径。

    返回:
        bool: 如果包加载成功，则为True，否则为False。
    """
    logger.info(f"Loading package: {package_path}")
    load_result = omc.sendExpression(f'loadFile("{package_path}")')
    if not load_result:
        logger.error(f"Failed to load package: {package_path}")
        return False
    return True


def get_model_parameter_names(omc: OMCSessionZMQ, model_name: str) -> List[str]:
    """
    解析并返回给定模型的所有子组件参数名称。

    参数:
        omc (OMCSessionZMQ): OpenModelica会话对象。
        model_name (str): 模型的全名（例如，'example.Cycle'）。

    返回:
        List[str]: 所有可用参数名称的列表（例如，['blanket.TBR']）。
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
            if comp_type.startswith(model_name.split('.')[0]):
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
    """
    一个私有辅助函数，用于递归遍历模型并收集参数详细信息。
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
            if comp_type.startswith(class_name.split('.')[0]):
                _recursive_get_parameters(omc, comp_type, full_name, params_list)
            else:
                logger.debug(f"Skipping non-example component: {full_name} ({comp_type})")


def get_all_parameters_details(
    omc: OMCSessionZMQ, model_name: str
) -> List[Dict[str, Any]]:
    """
    使用递归获取给定模型中所有参数的详细信息。

    参数:
        omc (OMCSessionZMQ): OpenModelica会话对象。
        model_name (str): 模型的全名。

    返回:
        List[Dict[str, Any]]: 一个字典列表，每个字典包含一个参数的详细信息。
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
    """将参数值格式化为OpenModelica可识别的字符串。"""
    if isinstance(value, list):
        # Format lists as {v1,v2,...}
        return f"{name}={{{','.join(map(str, value))}}}"
    elif isinstance(value, str):
        # Format strings as "value"
        return f'{name}="{value}"'
    # For numbers and booleans, direct string conversion is fine
    return f"{name}={value}"
