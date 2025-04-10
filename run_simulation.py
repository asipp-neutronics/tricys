import os
import shutil
from OMPython import OMCSessionZMQ
from OMPython import ModelicaSystem
import pandas as pd
import numpy as np

def run_parameter_sweep(package_path, model_name, param_sweep, stop_time, step_size, temp_dir):
    """
    运行单参数扫描的仿真。
    
    参数:
    - package_path: Modelica 包路径
    - model_name: 模型名称
    - param_sweep: 字典，包含一个参数名和其值范围，例如 {"blanket.TBR": np.linspace(1.05, 1.15, 20)}
    - stop_time: 仿真停止时间
    - step_size: 输出步长
    - temp_dir: 临时文件目录
    """
    # 确保只传入一个参数
    if len(param_sweep) != 1:
        raise ValueError("Only one parameter should be provided for simulation.")

    # 清理并创建临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)

    # 启动 OpenModelica 会话
    omc = OMCSessionZMQ()
    omc.sendExpression(f'loadFile("{package_path}")')

    # 创建并编译模型
    mod = ModelicaSystem(fileName=package_path, modelName=model_name)
    mod.buildModel()

    # 获取参数名和值
    param_name = list(param_sweep.keys())[0]
    param_values = param_sweep[param_name]

    # 运行仿真
    for i, param_value in enumerate(param_values):
        # 设置参数
        mod.setParameters([f"{param_name}={param_value}"])

        # 设置仿真选项
        mod.setSimulationOptions([
            f"stopTime={stop_time}",
            "tolerance=1e-6",
            "outputFormat=csv",
            "variableFilter=time|sds\\.I\\[1\\]",
            f"stepSize={step_size}"
        ])

        # 运行仿真
        output_csv = os.path.join(temp_dir, f"simulation_results_{i}.csv")
        mod.simulate(resultfile=output_csv)

    # 整合结果
    combined_df = None
    original_csv_files = []
    for i, param_value in enumerate(param_values):
        csv_file = os.path.join(temp_dir, f"simulation_results_{i}.csv")
        if os.path.exists(csv_file):
            original_csv_files.append(csv_file)
            df = pd.read_csv(csv_file)
            if combined_df is None:
                combined_df = df[['time']].copy()
            column_name = f"sds.I[1]_{param_name}={param_value:.3f}"
            combined_df[column_name] = df['sds.I[1]']
        else:
            print(f"Warning: CSV file {csv_file} not found.")

    # 保存整合结果
    combined_csv_path = os.path.join(temp_dir, "combined_simulation_results.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined CSV file saved to: {combined_csv_path}")

    # 清理临时文件
    for csv_file in original_csv_files:
        os.remove(csv_file)

    # 关闭会话
    del mod
    del omc

if __name__ == "__main__":
    # 测试用例
    package_path = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/FFCAS/package.mo"
    model_name = "FFCAS.Cycle"
    param_sweep = {
        "blanket.TBR": np.linspace(1.05, 1.15, 5)
    }
    run_parameter_sweep(package_path, model_name, param_sweep, 5000.0, 1.0, "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp")