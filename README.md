
# FFCAS - Fusion Fuel Cycle Analysis System

## 项目简介

FFCAS（Fusion Fuel Cycle Analysis System）是一个用于分析聚变燃料循环的仿真系统，基于 OpenModelica 平台开发。该项目旨在通过建模和仿真，研究聚变反应堆燃料循环的动态行为，特别关注参数（如 `blanket.TBR`）对系统性能的影响。

当前版本提供了一个图形用户界面（GUI）和模块化的参数扫描功能，方便用户配置仿真参数、运行多次仿真并分析结果。

## 功能特点

- **图形用户界面**：
  - 通过 Tkinter 提供直观的交互界面，支持输入模型名称、参数范围（如 `blanket.TBR`）、仿真时间和步长。
  - 显示仿真成功或错误提示，适合初学者和专业用户。
- **参数扫描仿真**：
  - 支持单参数扫描，自动运行多组仿真，整合结果为单一 CSV 文件（`temp/combined_simulation_results.csv`）。
  - 输出限制为关键变量（如 `time` 和 `sds.I[1]`），提高效率。
- **OpenModelica 集成**：
  - 使用 `OMPython` 加载和运行 Modelica 模型（如 `FFCAS.Cycle`）。
- **文件管理**：
  - 动态创建和清理 `temp/` 目录中的临时文件，优化磁盘使用。
  - 通过 `.gitignore` 忽略 `temp/`、`__pycache__/` 和 `*.pyc`，保持仓库整洁。
- **跨平台支持**：
  - 基于 Python 和 OpenModelica，支持 Windows、Linux 和 macOS。

## 安装与依赖

### 环境要求

- **操作系统**：Windows、Linux 或 macOS
- **OpenModelica**：最新版本（建议从 [OpenModelica 官网](https://openmodelica.org/) 下载）。
- **Python**：Python 3.8 或更高版本。

### 安装步骤

1. **安装 OpenModelica**：
   - 下载并安装 OpenModelica，确保 `omc` 可执行文件在系统路径中。
   - 参考 [OpenModelica 安装指南](https://openmodelica.org/).

2. **安装 Python 依赖**：
   - 克隆本仓库：
     ```bash
     git clone https://github.com/Valtro1s/FFCAS_FusionFuelCycleAnalysisSystem.git
     cd FFCAS_FusionFuelCycleAnalysisSystem
     ```
   - 安装所需库：
     ```bash
     pip install OMPython pandas numpy
     ```
   - 确保 `tkinter` 可用（通常随 Python 安装，若缺失可运行 `sudo apt install python3-tk` 或类似命令）。

3. **准备模型文件**：
   - 将 `FFCAS/package.mo` 放置在项目目录或指定路径。
   - *注*：当前代码使用固定路径（`D:/FusionSimulationProgram/...`），请根据实际环境调整。

## 使用方法

### 通过图形界面运行仿真

1. **启动程序**：
   ```bash
   python simulation_ui.py
   ```

2. **配置参数**：
   - **Model Name**：输入模型名称（默认 `FFCAS.Cycle`）。
   - **Parameter Name**：输入参数（如 `blanket.TBR`）。
   - **Min Value** / **Max Value**：设置范围（如 1.05 到 1.15）。
   - **Number of Steps**：设置步数（如 20）。
   - **Stop Time** / **Step Size**：设置仿真时间（如 5000 秒）和步长（如 1 秒）。

3. **运行仿真**：
   - 点击“Run Simulation”按钮。
   - 成功后，结果保存为 `temp/combined_simulation_results.csv`，并显示成功提示。
   - 失败时会显示错误信息。

### 通过脚本运行仿真

1. **运行参数扫描**：
   ```python
   from run_parameter_sweep import run_parameter_sweep
   import numpy as np

   package_path = "path/to/FFCAS/package.mo"  # 替换为实际路径
   model_name = "FFCAS.Cycle"
   param_sweep = {"blanket.TBR": np.linspace(1.05, 1.15, 5)}
   stop_time = 5000.0
   step_size = 1.0
   temp_dir = "temp"

   run_parameter_sweep(package_path, model_name, param_sweep, stop_time, step_size, temp_dir)
   ```

2. **查看结果**：
   - 结果保存在 `temp/combined_simulation_results.csv`。
   - 示例内容：
     ```
     time,sds.I[1]_blanket.TBR=1.050,sds.I[1]_blanket.TBR=1.075,...
     0.0,0.0,0.0,...
     1.0,0.123,0.125,...
     ...
     5000.0,0.1234,0.1240,...
     ```

## 文件结构

```
FFCAS_FusionFuelCycleAnalysisSystem/
├── FFCAS/                  # Modelica 模型文件目录
│   ├── package.mo          # 主包文件
│   └── ...                 # 其他模型文件
├── run_parameter_sweep.py  # 参数扫描仿真模块
├── simulation_ui.py        # 图形用户界面模块
├── temp/                   # 临时文件目录（运行时生成，.gitignore 忽略）
│   └── combined_simulation_results.csv  # 整合后的仿真结果
└── README.md               # 项目说明文件
```

## 开发与贡献

本项目处于活跃开发阶段，欢迎提出建议或贡献代码。在 `uers-features-dev` 分支中，我们新增了以下功能：
- 图形用户界面（`simulation_ui.py`），简化参数配置。
- 模块化的参数扫描（`run_parameter_sweep.py`），支持灵活仿真。
- 修复了 `.gitignore`，确保 `temp/` 目录不被跟踪。

请按照以下步骤参与：
1. Fork 本仓库。
2. 在你的副本中修改，提交到分支（建议命名为 `feature/xxx` 或 `fix/xxx`）。
3. 创建 Pull Request，描述改动内容。

提交问题请使用 GitHub Issues。

## 注意事项

- **模型路径**：确保 `package.mo` 路径正确，当前代码的硬编码路径需手动调整。
- **变量名**：确认 `sds.I[1]` 是模型中的有效变量，可通过以下代码检查：
  ```python
  from OMPython import ModelicaSystem
  mod = ModelicaSystem("FFCAS/package.mo", "FFCAS.Cycle")
  print(mod.getSolutions())
  ```
- **磁盘空间**：多次仿真可能生成大量数据，请确保 `temp/` 目录有足够空间。
- **OpenModelica 兼容性**：建议使用最新版本以支持 `variableFilter` 等功能。

## 许可证

本项目尚未指定许可证。请根据需要选择合适的开源许可证（如 MIT）。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
```
