import os
import shutil
import glob
from OMPython import OMCSessionZMQ
from OMPython import ModelicaSystem

# 设置自定义临时目录
custom_temp_dir = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp"
if os.path.exists(custom_temp_dir):
    shutil.rmtree(custom_temp_dir)  # 清理旧的临时文件
os.makedirs(custom_temp_dir, exist_ok=True)

# 切换到自定义工作目录
os.chdir(custom_temp_dir)
print(f"Current working directory: {os.getcwd()}")

# 启动 OpenModelica 会话
omc = OMCSessionZMQ()

# 指定包路径
package_path = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/FFCAS/package.mo"

# 加载包
if not omc.sendExpression(f'loadFile("{package_path}")'):
    raise RuntimeError(f"Failed to load {package_path}: {omc.sendExpression('getErrorString()')}")

# 创建 ModelicaSystem 对象，指定顶层模型
model_name = "FFCAS.Cycle"
mod = ModelicaSystem(fileName=package_path, modelName=model_name)

# 编译模型
mod.buildModel()

# 设置仿真参数，正确使用字符串列表形式
mod.setSimulationOptions([
    "stopTime=5000.0",
    "tolerance=1e-6",
    "outputFormat=csv"
])

# 运行仿真，指定结果文件为 CSV 格式
output_csv_path = os.path.join(custom_temp_dir, "simulation_results.csv")
mod.simulate(resultfile=output_csv_path)

# 检查生成的 CSV 文件
csv_files = glob.glob(os.path.join(custom_temp_dir, "*.csv"))
if csv_files:
    print(f"Generated CSV files: {csv_files}")
else:
    print("No CSV files were generated. Check if simulation completed successfully.")

# 获取仿真结果
time = mod.getSolutions("time")
variable1 = mod.getSolutions("sds.I[1]")  # 使用小写 sds

# 打印结果
print(f"Time: {time}")
print(f"sds.I[1]: {variable1}")

# 关闭会话
del mod
del omc