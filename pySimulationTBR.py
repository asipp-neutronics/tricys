import os
import shutil
from OMPython import OMCSessionZMQ
from OMPython import ModelicaSystem
import numpy as np
import pandas as pd
import glob

# 设置自定义临时目录
custom_temp_dir = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp"
if os.path.exists(custom_temp_dir):
    shutil.rmtree(custom_temp_dir)  # 清理旧的临时文件
os.makedirs(custom_temp_dir, exist_ok=True)

# 切换到自定义工作目录
os.chdir(custom_temp_dir)

# 启动 OpenModelica 会话
omc = OMCSessionZMQ()

# 指定包路径
package_path = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/FFCAS/package.mo"

# 加载包
omc.sendExpression(f'loadFile("{package_path}")')

# 创建 ModelicaSystem 对象，指定顶层模型
model_name = "FFCAS.Cycle"
mod = ModelicaSystem(fileName=package_path, modelName=model_name)

# 编译模型（只需编译一次）
mod.buildModel()

# 定义需要修改的参数及其取值范围
parameter_name = "blanket.TBR"
parameter_values = np.linspace(1.05, 1.15, 20)  # 生成 20 个参数值，从 1.05 到 1.15

# 循环仿真
for i, param_value in enumerate(parameter_values):
    # 修改参数
    mod.setParameters([f"{parameter_name}={param_value}"])

    # 设置仿真参数，并指定只输出 time 和 sds.I[1]
    mod.setSimulationOptions([
        "stopTime=5000.0",
        "tolerance=1e-6",
        "outputFormat=csv",
        "variableFilter=time|sds\\.I\\[1\\]",  # 使用正则表达式过滤变量
        "stepSize=1.0"  # 设置输出步长为 1.0 秒
    ])

    # 运行仿真，指定结果文件为 CSV 格式
    output_csv_path = os.path.join(custom_temp_dir, f"simulation_results_{i}.csv")
    mod.simulate(resultfile=output_csv_path)

# 关闭会话
del mod
del omc

# 整合 CSV 文件
combined_df = None
original_csv_files = []  # 用于存储原始 CSV 文件路径，以便后续删除
for i, param_value in enumerate(parameter_values):
    csv_file = os.path.join(custom_temp_dir, f"simulation_results_{i}.csv")
    if os.path.exists(csv_file):
        original_csv_files.append(csv_file)  # 记录文件路径
        df = pd.read_csv(csv_file)
        if combined_df is None:
            # 初始化 combined_df，使用第一个 CSV 文件的 time 列
            combined_df = df[['time']].copy()
        # 添加 sds.I[1] 列，列名包含 TBR 值
        column_name = f"sds.I[1]_TBR_{param_value:.3f}"
        combined_df[column_name] = df['sds.I[1]']
    else:
        print(f"Warning: CSV file {csv_file} not found.")

# 保存整合后的 CSV 文件
combined_csv_path = os.path.join(custom_temp_dir, "combined_simulation_results.csv")
combined_df.to_csv(combined_csv_path, index=False)
print(f"Combined CSV file saved to: {combined_csv_path}")

# 删除原始的 20 个 CSV 文件
for csv_file in original_csv_files:
    try:
        os.remove(csv_file)
    except Exception as e:
        print(f"Warning: Could not delete {csv_file}. Error: {e}")