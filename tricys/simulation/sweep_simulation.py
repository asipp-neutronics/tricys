"""本模块提供运行参数扫描模拟的功能。"""

import logging
import os

import numpy as np
import pandas as pd
from OMPython import ModelicaSystem, OMCSessionZMQ

from tricys.manager.config_manager import config_manager
from tricys.manager.logger_manager import logger_manager

from tricys.utils.file_utils import get_unique_filename

# Add project root to sys.path to allow absolute imports from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
logger = logging.getLogger(__name__)


def run_parameter_sweep(
    package_path,
    model_name,
    param_A_values,
    param_B_sweep,
    stop_time,
    step_size,
    temp_dir,
    results_dir,
    variableFilter
):
    """
    运行双参数扫描模拟。

    此函数使用固定的参数A值和扫描的参数B值运行模拟。参数A通常是值变化较少的参数，而参数B是值变化较多的参数。

    参数:
        package_path (str): Modelica包的路径。
        model_name (str): 要模拟的模型的名称。
        param_A_values (dict): 包含参数A的名称及其值列表的字典（例如，{"A_name": [1.0, 1.1, 1.2]}）。
        param_B_sweep (dict): 包含参数B的名称及其扫描范围的字典（例如，{"B_name": np.linspace(1.05, 1.15, 20)}）。
        stop_time (float): 模拟停止时间。
        step_size (float): 模拟步长。
        temp_dir (str): 临时文件的目录。
        results_dir (str): 保存结果的目录。
        variableFilter：输出过滤器，用于指定要输出的变量。

    返回:
        str: 组合结果文件的路径。
    """

    if len(param_A_values) != 1 or len(param_B_sweep) != 1:
        raise ValueError(
            "Exactly one parameter A and one parameter B should be provided."
        )

    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)

    omc = OMCSessionZMQ()
    omc.sendExpression(f'loadFile("{package_path}")')

    mod = ModelicaSystem(
        fileName=package_path,
        modelName=model_name,
        variableFilter=variableFilter,
    )
    mod.buildModel()

    param_A_name = list(param_A_values.keys())[0]
    param_A_vals = param_A_values[param_A_name]
    param_B_name = list(param_B_sweep.keys())[0]
    param_B_vals = param_B_sweep[param_B_name]

    counter = 0
    for param_A_val in param_A_vals:
        for param_B_val in param_B_vals:
            mod.setParameters(
                [f"{param_A_name}={param_A_val}", f"{param_B_name}={param_B_val}"]
            )
            mod.setSimulationOptions(
                [
                    f"stopTime={stop_time}",
                    "tolerance=1e-6",
                    "outputFormat=csv",
                    f"stepSize={step_size}",
                ]
            )

            base_filename = f"simulation_results_{counter}.csv"
            output_csv = get_unique_filename(temp_dir, base_filename)
            mod.simulate(resultfile=output_csv)
            counter += 1

    combined_df = None
    original_csv_files = []
    counter = 0
    for param_A_val in param_A_vals:
        for param_B_val in param_B_vals:
            csv_file = os.path.join(temp_dir, f"simulation_results_{counter}.csv")
            if os.path.exists(csv_file):
                original_csv_files.append(csv_file)
                df = pd.read_csv(csv_file)
                if combined_df is None:
                    combined_df = df[["time"]].copy()
                column_name = (
                    f"{param_A_name}={param_A_val:.3f}_{param_B_name}={param_B_val:.3f}"
                )
                combined_df[column_name] = df["sds.I[1]"]
            else:
                print(f"Warning: CSV file {csv_file} not found.")
            counter += 1

    base_combined_filename = f"{param_A_name}_{param_B_name}.csv"
    combined_csv_path = get_unique_filename(results_dir, base_combined_filename)
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined CSV file saved to: {combined_csv_path}")

    for csv_file in original_csv_files:
        os.remove(csv_file)

    del mod
    del omc

    return combined_csv_path


if __name__ == "__main__":
    package_path = os.path.join(PROJECT_ROOT, config_manager.get("paths.package_path"))
    temp_dir = os.path.join(PROJECT_ROOT, config_manager.get("paths.temp_dir"))
    results_dir = os.path.join(PROJECT_ROOT, config_manager.get("paths.results_dir"))
    
    variableFilter = config_manager.get("simulation.variableFilter", "time|sds.I[1]")
    model_name = config_manager.get("simulation.model_name", "example.Cycle")
    param_A_name = config_manager.get("sweep_parameter.parameter_A.name")
    param_B_name = config_manager.get("sweep_parameter.parameter_B.name")
    min_val = config_manager.get("sweep_parameter.parameter_B.min_value")
    max_val = config_manager.get("sweep_parameter.parameter_B.max_value")
    steps = config_manager.get("sweep_parameter.parameter_B.num_steps")

    param_A_vals = config_manager.get("sweep_parameter.parameter_A.values")
    param_A_values = {param_A_name: param_A_vals}
    param_B_vals = np.linspace(min_val, max_val, steps)
    param_B_sweep = {param_B_name: param_B_vals}

    stop_time = config_manager.get("simulation.stop_time", 5000.0)
    step_size = config_manager.get("simulation.step_size", 1)

    #param_A_values = {"blanket.T": [1.0, 1.1, 1.2]}  # 示例参数 A（变化较少）
    #param_B_sweep = {
    #    "blanket.TBR": np.linspace(1.05, 1.15, 5)
    #}  # 示例参数 B（变化较多）

    result_path = run_parameter_sweep(
        package_path,
        model_name,
        param_A_values,
        param_B_sweep,
        stop_time,
        step_size,
        temp_dir,
        results_dir,
        variableFilter
    )
    print(f"Result path: {result_path}")