from OMPython import OMCSessionZMQ
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OMPython')

def get_available_parameters(package_path, model_name):
    """
    从 Modelica 包中解析指定模型的所有子组件参数。
    
    参数:
    - package_path: Modelica 包路径，例如 "path/to/FFCAS/package.mo"
    - model_name: 模型名称，例如 "FFCAS.Cycle"
    
    返回:
    - 可用参数列表，例如 ['i_iss.T', 'blanket.TBR', ...]
    """
    available_params = ['i_iss.T']  # 默认参数
    omc = OMCSessionZMQ()
    try:
        logger.info(f"Loading package: {package_path}")
        omc.sendExpression(f'loadFile("{package_path}")')
        if omc.sendExpression(f"isModel({model_name})"):
            logger.info(f"Parsing components for {model_name}")
            components = omc.sendExpression(f"getComponents({model_name})")
            for comp in components:
                comp_type = comp[0]
                comp_name = comp[1]
                if comp_type.startswith("FFCAS."):
                    params = omc.sendExpression(f"getParameterNames({comp_type})")
                    for param in params:
                        full_param = f"{comp_name}.{param}"
                        if full_param not in available_params:
                            available_params.append(full_param)
            logger.info(f"Available parameters for {model_name}: {available_params}")
            print(f"Available parameters for {model_name}: {available_params}")
        else:
            logger.warning(f"{model_name} not found in package.")
        return available_params
    except Exception as e:
        logger.error(f"Failed to load parameters: {str(e)}")
        return available_params
    finally:
        omc.sendExpression("quit()")