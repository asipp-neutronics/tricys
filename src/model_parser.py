import logging
from typing import List, Optional
from OMPython import OMCSessionZMQ
from .file_utils import get_unique_filename  # 假设 file_utils.py 存在

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ModelicaParameterParser:
    """
    A class to parse parameters from a Modelica model using OMPython.

    Attributes:
        package_path (str): Path to the Modelica package file (e.g., 'path/to/package.mo').
        model_name (str): Name of the model (e.g., 'FFCAS.Cycle').
        default_params (List[str]): Default parameters to return if parsing fails.
    """
    
    def __init__(self, package_path: str, model_name: str, default_params: Optional[List[str]] = None):
        """
        Initialize the parser with package path, model name, and optional default parameters.

        Args:
            package_path (str): Path to the Modelica package file.
            model_name (str): Name of the model.
            default_params (List[str], optional): Default parameters if parsing fails. Defaults to ['i_iss.T'].
        """
        self.package_path = package_path
        self.model_name = model_name
        self.default_params = default_params if default_params is not None else ['i_iss.T']
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate package path and model name."""
        if not self.package_path or not isinstance(self.package_path, str):
            logger.error("Package path must be a non-empty string.")
            raise ValueError("Package path must be a non-empty string.")
        if not self.model_name or not isinstance(self.model_name, str):
            logger.error("Model name must be a non-empty string.")
            raise ValueError("Model name must be a non-empty string.")
        if not self.package_path.endswith('.mo'):
            logger.warning(f"Package path '{self.package_path}' does not end with '.mo'. Ensure it is a valid Modelica file.")

    def get_available_parameters(self) -> List[str]:
        """
        Parse all subcomponent parameters from the specified Modelica model.

        Returns:
            List[str]: List of available parameters (e.g., ['i_iss.T', 'blanket.TBR', ...]).

        Raises:
            RuntimeError: If the package or model cannot be loaded.
        """
        available_params = self.default_params.copy()
        try:
            with OMCSessionZMQ() as omc:
                logger.info(f"Loading package: {self.package_path}")
                load_result = omc.sendExpression(f'loadFile("{self.package_path}")')
                if not load_result:
                    logger.error(f"Failed to load package: {self.package_path}")
                    raise RuntimeError(f"Failed to load package: {self.package_path}")

                logger.info(f"Checking model: {self.model_name}")
                if not omc.sendExpression(f"isModel({self.model_name})"):
                    logger.warning(f"Model '{self.model_name}' not found in package.")
                    return available_params

                logger.info(f"Parsing components for {self.model_name}")
                components = omc.sendExpression(f"getComponents({self.model_name})")
                if not components:
                    logger.warning(f"No components found for {self.model_name}")
                    return available_params

                for comp in components:
                    comp_type = comp[0]
                    comp_name = comp[1]
                    if comp_type.startswith("FFCAS."):
                        logger.debug(f"Processing component: {comp_name} ({comp_type})")
                        params = omc.sendExpression(f"getParameterNames({comp_type})")
                        for param in params:
                            full_param = f"{comp_name}.{param}"
                            if full_param not in available_params:
                                available_params.append(full_param)

                logger.info(f"Available parameters for {self.model_name}: {available_params}")
                return available_params

        except Exception as e:
            logger.error(f"Failed to parse parameters: {str(e)}")
            return available_params