# FFCAS v0 - Fusion Fuel Cycle Analysis System

## 项目简介

FFCAS（Fusion Fuel Cycle Analysis System）是一个用于分析聚变燃料循环的仿真系统，基于 OpenModelica 平台开发。该项目旨在通过建模和仿真，研究聚变反应堆燃料循环的动态行为，特别关注参数（如 `blanket.TBR`）对系统性能的影响。

本项目提供了一个 Python 脚本，用于执行多次仿真、生成结果文件并进行后处理，方便用户对比不同参数条件下的仿真结果。

## 功能特点

- **多次仿真**：支持通过修改参数（例如 `blanket.TBR`）执行多次仿真，生成独立的仿真结果。
- **结果限制**：限制仿真结果输出为特定变量（如 `time` 和 `sds.I[1]`），提高效率和可读性。
- **结果整合**：将多次仿真的结果整合为单一 CSV 文件，便于对比分析。
- **文件管理**：自动清理临时文件，优化磁盘使用。
- **跨平台支持**：基于 OpenModelica 和 Python，支持 Windows、Linux 和 macOS。

## 安装与依赖

### 环境要求

- **操作系统**：Windows、Linux 或 macOS
- **OpenModelica**：需要安装 OpenModelica 编译器（建议使用最新版本）。
- **Python**：Python 3.6 或更高版本。

### 安装步骤

1. **安装 OpenModelica**：
   - 从 [OpenModelica 官网](https://openmodelica.org/) 下载并安装 OpenModelica。
   - 确保 `omc` 可执行文件已添加到系统路径。

2. **安装 Python 依赖**：
   - 克隆本仓库到本地：
     ```bash
     git clone https://github.com/Valtro1s/FFCAS_v0_FusionFuelCycleAnalysisSystem.git
     cd FFCAS_v0_FusionFuelCycleAnalysisSystem
     ```
   - 安装所需的 Python 包：
     ```bash
     pip install -r requirements.txt
     ```
   - 如果 `requirements.txt` 不存在，可以手动安装以下依赖：
     ```bash
     pip install OMPython numpy pandas
     ```

## 使用方法

### 运行仿真

1. **准备模型文件**：
   - 确保 `FFCAS` 目录下的 `package.mo` 文件已正确配置，包含所需的 Modelica 模型。

2. **修改脚本参数**（可选）：
   - 打开 `simulation.py`，根据需要修改以下参数：
     - `custom_temp_dir`：临时文件存储目录。
     - `package_path`：Modelica 包路径。
     - `parameter_name` 和 `parameter_values`：要修改的参数及其取值范围。

3. **执行脚本**：
   - 在命令行或 VS Code 终端中运行脚本：
     ```bash
     python simulation.py
     ```
   - 脚本将执行 20 次仿真，生成结果文件，并整合为一个 CSV 文件（`combined_simulation_results.csv`）。

4. **查看结果**：
   - 仿真结果保存在 `temp` 目录下的 `combined_simulation_results.csv` 文件中。
   - 该文件包含 `time` 列和 20 个 `sds.I[1]` 列（对应不同的 `blanket.TBR` 值）。

### 示例输出

`combined_simulation_results.csv` 的内容示例：

```
time,sds.I[1]_TBR_1.050,sds.I[1]_TBR_1.055,...,sds.I[1]_TBR_1.150
0.0,0.0,0.0,...,0.0
1.0,0.0123,0.0125,...,0.0145
2.0,0.0245,0.0248,...,0.0290
...
5000.0,0.1234,0.1240,...,0.1300
```

## 文件结构

```
FFCAS_v0_FusionFuelCycleAnalysisSystem/
├── FFCAS/                  # Modelica 模型文件目录
│   ├── package.mo          # 主包文件
│   └── ...                 # 其他模型文件
├── simulation.py           # 主仿真脚本
├── temp/                   # 临时文件目录（运行时生成）
│   └── combined_simulation_results.csv  # 整合后的仿真结果
├── requirements.txt        # Python 依赖文件（可选）
└── README.md               # 项目说明文件
```

## 开发与贡献

本项目目前处于开发阶段，欢迎提出建议或贡献代码。请按照以下步骤参与开发：

1. **Fork 本仓库**：
   - 点击 GitHub 页面右上角的“Fork”按钮，创建你的仓库副本。

2. **提交 Pull Request**：
   - 在你的副本中进行修改，提交到你的仓库。
   - 创建 Pull Request，描述你的改动内容。

3. **提交问题**：
   - 如果发现 bug 或有功能建议，请在 GitHub 的 Issues 页面提交。

## 注意事项

- **变量名确认**：确保 `sds.I[1]` 是模型中实际存在的变量。如果不确定，可以在 `simulation.py` 中添加调试代码，打印所有输出变量：
  ```python
  output_vars = mod.getSolutions()
  print("Available output variables:", output_vars)
  ```
- **磁盘空间**：多次仿真可能生成大量数据，请确保磁盘空间充足。
- **OpenModelica 版本**：某些功能（例如 `variableFilter`）可能依赖于 OpenModelica 的版本，建议使用最新版本。

## 许可证

本项目尚未指定许可证。请根据需要选择合适的开源许可证（例如 MIT、GPL 等）。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系，或发送邮件至 [你的邮箱地址]。
```