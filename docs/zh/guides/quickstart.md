# 快速开始

本指南将引导您完成 **Windows** 环境下 `tricys` 的安装过程，并运行一个基本的命令行仿真。

## 1. 环境要求

在开始之前，请确保您的系统满足以下要求：

- **Python**: 版本 3.8 或更高。
- **Git**: 用于克隆项目仓库。
- **OpenModelica**: 需要安装 OpenModelica，并确保其命令行工具 (`omc.exe`) 已添加到系统的 `PATH` 环境变量中。

!!! tip "提示"
    在 Windows 上安装 Python 时，请务必勾选 **"Add Python to PATH"** 选项，以确保 `python` 和 `pip` 命令在终端中可用。

## 2. 安装步骤

### a. 克隆项目仓库

打开您的终端（如 PowerShell 或 Cmd），使用 `git` 克隆 `tricys` 的源代码。

```shell
git clone https://github.com/asipp-neutronics/tricys.git
cd tricys
```

### b. 创建并激活虚拟环境

在项目根目录下，创建一个独立的 Python 虚拟环境，以隔离项目依赖。

```shell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\activate
```

激活后，您会看到终端提示符前出现 `(venv)` 字样。

### c. 安装项目依赖

使用 `pip` 安装 `tricys` 及其所有开发依赖项。`-e` 参数表示以“可编辑”模式安装，这意味着您对源代码的任何修改都会立即生效。

```shell
pip install -e ".[win]"

or # 或使用Makefile.bat脚本安装依赖

Makefile.bat win-install
```


## 3. 运行示例

`tricys` 提供了一个交互式的示例运行器，可以帮助您快速探索和运行所有可用的示例，包括基础仿真和高级分析任务。这是验证安装并了解 `tricys` 功能的最简单方法。

### a. 启动示例运行器

在激活了虚拟环境的终端中，执行以下命令来启动示例运行器：

```shell
tricys example
```

### b. 选择并运行示例

该命令会启动一个统一的示例运行器，扫描并列出 `example/basic` 和 `example/analysis` 目录下的所有可用示例。

您会看到一个类似下面的菜单：

```text
============================================================
         TRICYS 统一示例运行器
============================================================

  1. [BASIC] Basic Configuration
     描述: A basic simulation with a single run
     配置: basic_configuration.json

  2. [BASIC] Parameter Sweep
     描述: A multi-run simulation with parameter sweeps
     配置: parameter_sweep.json

  ...

  6. [ANALYSIS] Baseline Condition Analysis
     描述: Baseline condition analysis for TBR search
     配置: baseline_condition_analysis.json

  ...

  0. 退出程序
  h. 显示帮助信息
  s. 重新扫描示例目录

============================================================
```

- **输入数字** (例如 `1`) 并按回车，即可运行对应的示例。
- 程序会自动将示例文件复制到项目根目录下的 `test_example` 文件夹中，并在该目录中执行任务。
- 这种设计可以确保原始示例文件不被修改，并保持工作区整洁。

### c. 查看结果

任务完成后，结果将保存在 `test_example` 目录中，位于相应示例的子文件夹内。

例如，如果您运行了 `Basic Configuration` 示例，结果将位于 `test_example/basic/1_basic_configuration/` 目录下。您可以在其中找到：

- `simulation_result.csv`: 包含所有输出变量随时间变化的数据。
- `simualtion_{timestamp}.log`: 本次运行的详细日志文件。
- `basic_configuation.json`: 本次运行所使用的完整配置的备份。

## 4. 运行图形用户界面 (GUI)

如果您更喜欢图形化操作，可以启动 `tricys` 的 GUI。

```shell
tricys gui
```

GUI 提供了一个交互式界面，用于加载模型、设置参数、定义扫描范围和启动仿真。

---

## 5. TRICYS 相关命令

除了通过示例运行器，您也可以直接使用 `tricys` 的各种子命令来执行特定任务。下面是所有可用命令的详细列表：

| 命令 / 用法 | 解释 |
| :--- | :--- |
| **主命令 / 用法** | |
| `tricys` | 需要通过 `-c` 参数指定配置文件，程序会自动检测配置文件内容并执行相应的工作流（标准模拟或分析）。 |
| `tricys basic` | 运行一个标准的模拟。需要通过 `-c` 参数指定配置文件，或者在当前目录下存在默认的 `config.json`。 |
| `tricys analysis` | 运行一个模拟分析。需要通过 `-c` 参数指定配置文件，或者在当前目录下存在默认的 `config.json`。 |
| `tricys gui` | 启动交互式图形用户界面 (GUI)。 |
| `tricys example` | 运行一个交互式的示例运行器，可以启动所有类型的示例。 |
| `tricys archive <timestamp>` | 将指定时间戳 (`timestamp`) 的模拟或分析运行结果归档成一个 zip 文件。 |
| `tricys unarchive <zip_file>` | 解压一个之前归档的运行文件 (`zip_file`)。 |
| **`basic` 子命令** | |
| `tricys basic example` | 运行交互式的基本功能示例。 |
| **`analysis` 子命令** | |
| `tricys analysis example` | 运行交互式的分析示例。 |
| `tricys analysis retry <timestamp>` | 针对一个已存在的、AI 分析失败的报告，根据其时间戳 (`timestamp`) 重新尝试进行分析。 |
| **通用参数** | |
| `-h, --help` | 显示帮助信息并退出。 |
| **隐式行为** | |
| `(无命令)` | 若未提供 `-c` 或其他子命令，程序会查找并使用当前目录下的 `config.json` 文件。 |

![tricys_command.svg](../../assets/tricys_command.svg)


恭喜！您已经成功安装并运行了 `tricys`。接下来，您可以探索更高级的功能，例如[参数扫描](tricys_basic/parameter_sweep.md)或[协同仿真](tricys_basic/co_simulation_module.md)。
