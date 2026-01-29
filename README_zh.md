# TRICYS - 氚燃料循环集成仿真平台

[![license](https://img.shields.io/badge/license-Apache--2.0-green)](./LICENSE)
[![python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![docs](https://img.shields.io/badge/docs-中文-brightgreen.svg)](https://asipp-neutronics.github.io/tricys/)

**TRICYS** (**TR**itium **I**ntegrated **CY**cle **S**imulation) 是一个开源、模块化、多尺度的聚变堆氚燃料循环仿真器，旨在提供基于物理的动态闭环分析，并严格遵守全厂范围的质量守恒原则。

我们的目标是为研究人员和工程师提供一个灵活且强大的平台，用于探索各种氚管理策略、优化系统设计，并深入理解聚变反应堆环境中氚的流动与库存动态。

![Tritium Fuel Cycle System](./docs/zh/assets/cycle_system.png)

## 功能特性

- **参数扫描与并发**: 系统地研究多个参数对系统性能的影响，支持并发运行和大规模批量仿真。
- **子模块协同仿真**: 支持与外部工具（如 Aspen Plus）进行数据交换，完成子模块系统集成。
- **自动化报告生成**: 自动生成标准化的 Markdown 分析报告，包含图表、统计数据和可视化结果。
- **高级敏感性分析**: 支持系统参数的自定义敏感性分析，并集成SALib库量化参数对输出的影响
- **AI 增强分析**: 集成大型语言模型（LLM），能够将原始的图表和数据自动转化为结构化的学术风格报告。

## 快速开始：Windows 本地安装

为确保与 Aspen Plus 等外部 Windows 软件的协同仿真功能完全兼容，我们优先推荐 Windows 本地安装。

### 1. 环境要求
1.  **Python**: 3.8 或更高版本 (建议安装时勾选 "Add Python to PATH")。
2.  **Git**: 用于克隆代码仓库。
3.  **OpenModelica**: 确保其命令行工具 (`omc.exe`) 已添加到系统的 `PATH` 环境变量中。

### 2. 安装步骤

a. **克隆项目仓库**
   打开终端（如 PowerShell），使用 `git` 克隆源代码。
   ```shell
   git clone https://github.com/asipp-neutronics/tricys.git
   cd tricys
   ```

b. **创建并激活虚拟环境**
   为了隔离项目依赖，建议创建一个独立的 Python 虚拟环境。
   ```shell
   # 创建虚拟环境
   py -m venv venv
   # 激活虚拟环境
   .\venv\Scripts\activate
   ```

c. **安装项目依赖**
   使用 `pip` 以“可编辑”模式安装 `tricys` 及其所有依赖项。
   ```shell
   pip install -e ".[win]"
   ```
   或者，您也可以使用项目提供的便捷脚本：
   ```shell
   Makefile.bat win-install
   ```

### 3. 运行一个示例

安装完成后，您可以启动交互式示例运行器来快速体验 `tricys` 的核心功能。

```shell
tricys example
```
该命令会扫描并列出所有可用的基础和高级分析示例。您只需根据提示输入数字，即可自动运行对应的示例任务。


## 备选方案：Docker（标准 0D 仿真）

如果您不需要与外部 Windows 软件进行协同仿真，为了简化开发环境的配置，本项目维护了两个容器镜像, 支持 **VSCode & Dev Containers** 在容器化环境中运行和测试代码，：
1. [ghcr.io/asipp-neutronics/tricys_openmodelica_gui:docker_dev](https://github.com/orgs/asipp-neutronics/packages/container/tricys_openmodelica_ompython/476218036?tag=docker_dev)：带有OMEdit可视化应用
2. [ghcr.io/asipp-neutronics/tricys_openmodelica_ompython:docker_dev](https://github.com/orgs/asipp-neutronics/packages/container/tricys_openmodelica_gui/476218102?tag=docker_dev)：不带有OMEdit可视化应用

**如需切换dev container请删除原容器并修改docker-compose.yml**
```
image: ghcr.io/asipp-neutronics/tricys_openmodelica_gui:docker_dev
```

### 1. 环境要求
- **Docker**: 最新版本。
- **VSCode**: 最新版本，并已安装 [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 插件。

### 2. 一键启动开发环境

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/asipp-neutronics/tricys.git
    cd tricys
    ```

2.  **在 VSCode 中打开**:
    ```bash
    code .
    ```

3.  **在容器中重新打开**: VSCode 会检测到 `.devcontainer` 目录并提示“在容器中重新打开 (Reopen in Container)”，点击该按钮。
    > 首次构建容器时，需要下载指定的 Docker 镜像，可能需要一些时间。

4.  **安装项目依赖**: 容器成功启动后，在 VSCode 的终端中执行以下命令来安装项目所需的 Python 库。
    ```bash
    make dev-install
    ```


##  文档

获取更详细的功能介绍、配置指南和高级教程，请访问我们的[在线文档](https://asipp-neutronics.github.io/tricys/)。

##  贡献

我们欢迎社区的任何贡献！如果您希望参与 `tricys` 的开发，请遵循以下规范：

- **代码风格**: 使用 `black` 进行代码格式化，`ruff` 进行风格检查和修复。
- **命名规范**: 遵循 `snake_case` (变量/函数) 和 `PascalCase` (类) 的约定。
- **文档字符串**: 所有公共模块、类和函数都必须包含 Google 风格的文档字符串。
- **测试**: 使用 `pytest` 编写单元测试，并确保高覆盖率。
- **Git 提交**: 遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范，使提交历史清晰可读。

##  许可证

本项目采用 [Apache-2.0](./LICENSE) 许可证。
