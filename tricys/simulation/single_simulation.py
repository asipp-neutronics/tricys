"""本模块提供运行单次模拟的功能。"""

import logging
import os
from typing import Any, Dict

from OMPython import ModelicaSystem

from tricys.manager.config_manager import config_manager
from tricys.manager.logger_manager import logger_manager

from tricys.utils.db_utils import (
    create_parameters_table,
    get_parameters_from_db,
    store_parameters_in_db,
)
from tricys.utils.file_utils import get_unique_filename
from tricys.utils.om_utils import (
    format_parameter_value,
    get_all_parameters_details,
    get_om_session,
    load_modelica_package,
)

# Add project root to sys.path to allow absolute imports from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
logger = logging.getLogger(__name__)


def simulation(
    package_path: str,
    model_name: str,
    stop_time: float,
    step_size: float,
    results_dir: str,
    param_values: Dict[str, Any] ,
    variableFilter: str
) -> str:
    """
    运行单次模拟，将模型参数与数据库同步。
    """
    os.makedirs(results_dir, exist_ok=True)

    original_dir = os.getcwd()
    os.chdir(results_dir)

    mod = None
    omc = None
    output_csv = ""

    try:
        logger.info("Initializing database and OpenModelica session.")
        create_parameters_table()
        omc = get_om_session()

        if not load_modelica_package(omc, package_path):
            raise RuntimeError(f"Failed to load Modelica package at {package_path}")

        logger.info(
            f"Fetching parameter details from model '{model_name}' for DB sync."
        )
        params_details = get_all_parameters_details(omc, model_name)
        if params_details:
            logger.info(f"params_details:{params_details}")
            store_parameters_in_db(params_details)
        else:
            logger.warning(
                f"No parameters found for model {model_name}. Database not updated."
            )

        logger.info("Loading parameters from the database for the simulation run.")
        db_params = get_parameters_from_db()
        if not db_params:
            logger.warning(
                "No parameters loaded from the database. Using model defaults."
            )

        logger.info(f"Instantiating model: {model_name}")
        mod = ModelicaSystem(fileName=package_path, modelName=model_name,variableFilter=variableFilter)

        param_settings = []
        if param_values:
            logger.info(f"Applying {len(param_values)} parameter overrides.")
            for name, value in param_values.items():
                logger.debug(f"Setting parameter '{name}' to value: {value}")
                param_settings.append(format_parameter_value(name, value))

        mod.setSimulationOptions(
            [
                f"stopTime={stop_time}",
                "tolerance=1e-6",
                "outputFormat=csv",
                f"stepSize={step_size}",
            ]
        )
        if param_settings:
            logger.info("Setting model parameters for simulation.")
            mod.setParameters(param_settings)

        logger.info("Building and running simulation...")
        mod.buildModel()
        base_filename = "simulation_results.csv"
        output_csv = get_unique_filename(os.getcwd(), base_filename)
        mod.simulate(resultfile=output_csv)
        logger.info(f"Simulation finished. Results saved to {output_csv}")

    except Exception as e:
        logger.error(f"An error occurred during simulation: {e}", exc_info=True)
        raise
    finally:
        # Clean up
        if mod is not None:
            del mod
        if omc is not None:
            logger.info("Closing OpenModelica session.")
            omc.sendExpression("quit()")
            del omc
        os.chdir(original_dir)

    return output_csv


if __name__ == "__main__":
    package_path = os.path.join(PROJECT_ROOT, config_manager.get("paths.package_path"))
    results_dir = os.path.join(PROJECT_ROOT, config_manager.get("paths.results_dir"))
    model_name = config_manager.get("simulation.model_name", "example.Cycle")
    stop_time = config_manager.get("simulation.stop_time", 5000.0)
    step_size = config_manager.get("simulation.step_size", 1)
    param_overrides = config_manager.get("overrides_parameter")
    variableFilter = config_manager.get("simulation.variableFilter", "time|sds.I[1]")

    logger.info(f"Starting simulation for model: {model_name}")
    try:
        result_path = simulation(
            package_path=package_path,
            model_name=model_name,
            stop_time=stop_time,
            step_size=step_size,
            results_dir=results_dir,
            param_values=param_overrides,
            variableFilter=variableFilter
        )
        logger.info(
            f"Simulation run completed successfully. Result path: {result_path}"
        )
    except Exception:
        logger.critical("Simulation failed to complete.")
