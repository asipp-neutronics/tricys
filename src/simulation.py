import os
import sqlite3
import json
import pandas as pd
from OMPython import OMCSessionZMQ, ModelicaSystem
from .file_utils import get_unique_filename

# 读取 config.json
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)


def get_parameters_from_db(db_path: str) -> dict:
    """从数据库读取参数"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT name,  default_value FROM parameters')
        params = {}
        for name, default_value in cursor.fetchall():
            default_value = json.loads(default_value)
            params[name] = {
                'default_value': default_value
            }
    return params


def format_parameter_value(name: str, value: str) -> str:
    """格式化参数值为 OpenModelica 格式"""
    return f"{name}={value}"


def run_simulation(
        package_path: str,
        model_name: str,
        db_path: str,
        stop_time: float,
        step_size: float,
        temp_dir: str,
        param_values: dict = None) -> str:
    """运行单次仿真，使用数据库默认参数或指定参数"""
    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)

    # 加载参数
    params = get_parameters_from_db(db_path)
    param_names = list(params.keys())
    param_types = [params[name]['type'] for name in param_names]

    # 使用指定参数或默认值
    if param_values is None:
        param_vals = [params[name]['default_value'] for name in param_names]
    else:
        param_vals = [param_values.get(
            name, params[name]['default_value']) for name in param_names]
    # 初始化 OpenModelica
    omc = OMCSessionZMQ()
    omc.sendExpression(f'loadFile("{package_path}")')
    mod = ModelicaSystem(fileName=package_path, modelName=model_name)
    mod.buildModel()

    # 设置仿真选项
    mod.setSimulationOptions([
        f"stopTime={stop_time}",
        "tolerance=1e-6",
        "outputFormat=csv",
        "variableFilter=time|sds\\.I\\[1\\]"
        f"stepSize={step_size}"
    ])

    try:
        # 设置参数
        param_settings = [
            format_parameter_value(
                param_names[j], param_vals[j], param_types[j])
            for j in range(len(param_names))
        ]
        mod.setParameters(param_settings)

        # 运行仿真
        base_filename = "simulation_results.csv"
        output_csv = get_unique_filename(temp_dir, base_filename)
        mod.simulate(resultfile=output_csv)

        # 后处理 CSV
        df = pd.read_csv(output_csv)
        columns = df.columns.tolist()
        if 'time' in columns and 'sds.I[1]' in columns:
            df = df[['time', 'sds.I[1]']]
            df.to_csv(output_csv, index=False)
        else:
            raise ValueError(
                f"Expected columns 'time' and 'sds.I[1]', got {columns}")

    except Exception as e:
        raise

    finally:
        del mod
        del omc

    return output_csv


if __name__ == "__main__":
    # 测试用例
    package_path = "./example/package.mo"
    model_name = "example.Cycle"
    result_path = run_simulation(
        package_path, model_name, 500.0, 0.1, "./temp")
    print(f"Result path: {result_path}")
