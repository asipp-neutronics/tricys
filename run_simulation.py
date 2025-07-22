import os
from OMPython import OMCSessionZMQ
from OMPython import ModelicaSystem
import pandas as pd
import numpy as np

def get_unique_filename(base_path, filename):
    base_name, ext = os.path.splitext(filename)
    counter = 0
    new_filename = filename
    new_filepath = os.path.join(base_path, new_filename)
    
    while os.path.exists(new_filepath):
        counter += 1
        new_filename = f"{base_name}_{counter}{ext}"
        new_filepath = os.path.join(base_path, new_filename)
    
    return new_filepath

def run_parameter_sweep(package_path, model_name, param_A_values, param_B_sweep, stop_time, step_size, temp_dir):
    """
    运行双参数扫描的仿真：外层参数 A 固定值，内层参数 B 扫描。
    外层参数 A 通常是变化较少的参数（值的数量少），内层参数 B 是变化较多的参数（值的数量多）。
    
    参数:
    - package_path: Modelica 包路径
    - model_name: 模型名称
    - param_A_values: 字典，包含参数 A 的名称和值列表，例如 {"A_name": [1.0, 1.1, 1.2]}
    - param_B_sweep: 字典，包含参数 B 的名称和扫描范围，例如 {"B_name": np.linspace(1.05, 1.15, 20)}
    - stop_time: 仿真停止时间
    - step_size: 输出步长
    - temp_dir: 临时文件目录
    
    返回:
    - 整合结果文件的保存路径
    """
    # 确保参数 A 和参数 B 各只有一个
    if len(param_A_values) != 1 or len(param_B_sweep) != 1:
        raise ValueError("Exactly one parameter A and one parameter B should be provided.")

    # 创建临时目录（如果不存在）
    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)

    # 启动 OpenModelica 会话
    omc = OMCSessionZMQ()
    omc.sendExpression(f'loadFile("{package_path}")')

    # 创建并编译模型
    mod = ModelicaSystem(fileName=package_path, modelName=model_name, variableFilter="time|sds\\.I\\[1\\]")
    mod.buildModel()

    # 获取参数 A 和参数 B 的名称和值
    param_A_name = list(param_A_values.keys())[0]
    param_A_vals = param_A_values[param_A_name]
    param_B_name = list(param_B_sweep.keys())[0]
    param_B_vals = param_B_sweep[param_B_name]

    # 运行仿真：外层循环参数 A（变化较少），内层循环参数 B（变化较多）
    counter = 0
    for param_A_val in param_A_vals:
        for param_B_val in param_B_vals:
            # 设置参数 A 和参数 B
            mod.setParameters([f"{param_A_name}={param_A_val}", f"{param_B_name}={param_B_val}"])
            # 设置仿真选项
            mod.setSimulationOptions([
                f"stopTime={stop_time}",
                "tolerance=1e-6",
                "outputFormat=csv",
                f"stepSize={step_size}"
            ])

            # 运行仿真并保存到唯一文件名
            base_filename = f"simulation_results_{counter}.csv"
            output_csv = get_unique_filename(temp_dir, base_filename)
            mod.simulate(resultfile=output_csv)
            counter += 1

    # 整合结果
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
                    combined_df = df[['time']].copy()
                column_name = f"{param_A_name}={param_A_val:.3f}_{param_B_name}={param_B_val:.3f}"
                combined_df[column_name] = df['sds.I[1]']
            else:
                print(f"Warning: CSV file {csv_file} not found.")
            counter += 1

    # 保存整合结果，使用参数 A 和参数 B 的名称构造文件名
    base_combined_filename = f"{param_A_name}_{param_B_name}.csv"
    combined_csv_path = get_unique_filename(temp_dir, base_combined_filename)
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined CSV file saved to: {combined_csv_path}")

    # 清理临时文件
    for csv_file in original_csv_files:
        os.remove(csv_file)

    # 关闭会话
    del mod
    del omc

    return combined_csv_path

if __name__ == "__main__":
    # 测试用例
    package_path = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/FFCAS/package.mo"
    model_name = "FFCAS.Cycle"
    param_A_values = {"blanket.T": [1.0, 1.1, 1.2]}  # 示例参数 A（变化较少）
    param_B_sweep = {"blanket.TBR": np.linspace(1.05, 1.15, 5)}  # 示例参数 B（变化较多）
    result_path = run_parameter_sweep(package_path, model_name, param_A_values, param_B_sweep, 5000.0, 0.1, "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp")
    print(f"Result path: {result_path}")