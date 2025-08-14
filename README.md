
# tricys - TRitium Integrated CYcle Simulation

## 项目简介

**tricys (TRitium Integrated CYcle Simulation)** 是一个用于分析聚变燃料循环的仿真系统，基于 OpenModelica 平台开发。该项目旨在通过建模和仿真，研究聚变反应堆燃料循环的动态行为，特别关注参数（如 `blanket.TBR`）对系统性能的影响。是一个用于分析聚变燃料循环的仿真系统，基于 OpenModelica 平台开发。该项目旨在通过建模和仿真，研究聚变反应堆燃料循环的动态行为，特别关注参数（如 `blanket.TBR`）对系统性能的影响。

为满足不同用户的需求，`tricys` 提供了两种操作模式：

*   **图形用户界面 (GUI)**: 提供一个直观的交互界面，用户可以方便地加载模型、设置仿真参数、定义参数扫描范围并启动仿真。
*   **命令行界面 (CLI)**: 通过配置文件驱动，支持复杂的参数扫描和批量仿真任务，适合进行大规模的自动化计算和集成到其他工作流程中。


## Docker环境 安装与使用
为了简化开发环境的配置，本项目维护了两个容器镜像, 支持 **VSCode & Dev Containers** 在容器化环境中运行和测试代码，：
1. [ghcr.io/asipp-neutronics/tricys_openmodelica_gui:docker_dev](https://github.com/orgs/asipp-neutronics/packages/container/tricys_openmodelica_ompython/476218036?tag=docker_dev)：带有OMEdit可视化应用
2. [ghcr.io/asipp-neutronics/tricys_openmodelica_ompython:docker_dev](https://github.com/orgs/asipp-neutronics/packages/container/tricys_openmodelica_gui/476218102?tag=docker_dev)：不带有OMEdit可视化应用

**如需切换dev container请删除原容器并修改docker-compose.yml**
```
image: ghcr.io/asipp-neutronics/tricys_openmodelica_gui:docker_dev
```

### 环境要求
- **Docker**: 最新版本（从 [Docker 官网](https://www.docker.com/) 下载）。
- **Docker Compose**: 最新版本（通常随 Docker Desktop 一起安装）。
- **VSCode**: 最新版本（安装Dev Containers插件）。
- **Windows 11**: 最新版本（安装 WSL2 并默认启用 WSLg 功能）。
- **Linux**: 需要运行 **`xhost +local:`** 命令。

| 系统测试| tricys_openmodelica_ompython | tricys_openmodelica_gui（OMEdit） |
| :--- | :--- | :--- |
| Windows11 (WSL2) | ✅ | ✅ |
| Ubuntu 24.04 | ✅ | ✅ |
| Rocky 10 | ✅ | ✅ |
| CentOS 7 | ✅ | ❌|

**注意事项**：经测试，CentOS7考虑到版本较旧，无法在tricys_openmodelica_gui容器中运行`OMEdit`可视化应用


### 安装步骤

1.  **克隆仓库**: 打开终端，克隆本项目到本地。
    ```bash
    git clone https://github.com/asipp-neutronics/tricys.git
    cd tricys
    ```

2.  **在 VSCode 中打开**:
    ```bash
    code .
    ```

3.  **在容器中重新打开**: VSCode 会检测到 `.devcontainer` 目录并提示“在容器中重新打开 (Reopen in Container)”，点击该按钮。
    ```
    注意: 首次构建容器时，需要下载指定的 Docker 镜像，可能需要一些时间。
    ```

4.  **安装项目依赖**: 容器成功启动后进入容器的终端，在终端中执行以下命令来安装项目所需的 Python 库。
    ```bash
    make dev-install
    ```
### 使用方法

安装完成后，在容器终端中，您可以使用以下命令：

*   **运行图形用户界面 (GUI)**:
    ```shell
    tricys-gui
    ```

*   **运行命令行 (CLI) 仿真**:
    ```shell
    tricys -c example_config.json
    ```

## Windows环境 安装与使用

### 环境要求

1.  **Python**: 安装 Python 3.8 或更高版本。您可以从 [Python 官网](https://www.python.org/downloads/) 下载或通过 Microsoft Store 安装。**重要提示**：在安装过程中，请务必勾选“Add Python to PATH”选项。
2.  **Git**: 从 [Git 官网](https://git-scm.com/download/win) 下载并安装 Git for Windows。
3.  **OpenModelica**: 需要安装 OpenModelica。请确保其命令行工具（如 `omc.exe`）已添加到系统的 `PATH` 环境变量中。

### 安装步骤

1.  **克隆仓库**:
    ```shell
    git clone https://github.com/asipp-neutronics/tricys.git
    cd tricys
    ```

2.  **创建并激活虚拟环境**:
    ```shell
    py -m venv venv
    .\venv\Scripts\activate
    ```

3.  **安装项目依赖**: 以可编辑模式安装项目及所有开发工具，请运行：
    ```shell
    pip install -e ".[dev]"
    ```

### 使用方法

安装完成后，在激活虚拟环境的终端中，您可以使用以下命令：

*   **运行图形用户界面 (GUI)**:
    ```shell
    tricys-gui
    ```

*   **运行命令行 (CLI) 仿真**:
    ```shell
    tricys -c example_config.json
    ```

*   **开发任务 (便捷脚本)**: 项目提供了 `Makefile.bat` 脚本，方便在 Windows 上执行常见的开发任务：
    *   运行测试: `Makefile.bat test`
    *   检查代码格式与风格: `Makefile.bat check`
    *   清理生成的文件: `Makefile.bat clean`

## 项目说明
### 文件结构
```
tricys/
├── docs/                        # 项目文档
│   └── TricysUsersGuide.md      # Tricys用户手册
├── example/                     # 示范案例
│   ├── example_model/           # Openmodelica示例模型
│   └── example_config.json      # 示例配置文件
├── docker/                      # docker镜像构建
├── script/                      # 辅助脚本
├── tricys/                      # Python 源代码
│   ├── simulation.py            # 仿真模拟命令
│   ├── simulation_gui.py        # 界面仿真模拟命令
│   └── utils/                   # 通用工具模块
├── tests/                       # 测试代码
├── .gitignore                   # Git 忽略文件配置
├── .env                         # .env文件配置
├── config.json                  # 项目配置文件
├── docker-compose.yml           # Docker Compose 配置文件
├── pyproject.toml               # Python 项目配置文件
└── README.md                    # 项目开发说明
```
### 用户配置
1. `.env`：该配置文件中用于`docker环境开发`需要设置`CUSTOM_MODEL_PATH=****`来指定宿主机中用户的OpenModelica模型所在目录
2. `example/example_config.json`:该配置文件中表示默认参数值，如paths为路径参数，logging为日志参数，simulation为仿真运行参数, simulation_parameter为模型扫描参数。

## 使用方法
1. 安装项目：`make dev-install` or `Makefile.bat dev-install`
2. 执行测试：`make test` or `Makefile.bat test`
3. 清理数据：`make clean` or `Makefile.bat clean`
4. 规范代码：`make check` or `Makefile.bat check`

## 开发与贡献

本项目处于活跃开发阶段，为确保代码的一致性、可读性和可维护性，所有 Python 代码应遵循以下详细规范：

#### 1. 代码风格与格式化

- **PEP 8**: 所有代码必须严格遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 风格指南。
- **自动化格式化**:
  - 使用 `black` 进行代码格式化，确保风格统一。
  - 使用 `ruff` 检查修复存在的错误和风格问题。
- **配置与执行**:
    ```bash
    black .
    ruff check . --fix
    ruff check .
    ```

#### 2. 命名规范 (Naming Conventions)

- **变量与函数**: 使用蛇形命名法（`snake_case`），例如 `my_variable`, `calculate_inventory()`。
- **类**: 使用帕斯卡命名法（`PascalCase`），例如 `SimulationManager`, `DataProcessor`。
- **常量**: 使用大写蛇形命名法（`UPPER_SNAKE_CASE`），例如 `DEFAULT_PRESSURE`, `MAX_ITERATIONS`。
- **内部成员**: 对于仅在模块或类内部使用的变量或方法，使用单个下划线前缀，例如 `_internal_method`。

#### 3. 文档字符串 (Docstrings)

- **必要性**: 每个公共模块、类和函数都必须包含文档字符串。
- **格式**: 文档字符串应遵循 **[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)** 格式。
- **内容**: 清晰地描述其功能、参数、返回值以及可能引发的异常。
- **示例**:
  ```python
      def get_unique_filename(base_path: str, filename: str) -> str:
          """
          如果文件已存在，则通过附加计数器来生成唯一的文件名。

          参数:
              base_path (str): 将保存文件的目录路径。
              filename (str): 所需的文件名（包括扩展名）。

          返回:
              str: 一个不存在的唯一文件路径。
          """
  ```

#### 4. 测试 (Testing)

- **框架**: 使用 `pytest` 作为主要的测试框架。
- **位置**: 所有测试代码应放在 `test/` 目录下。
- **命名**: 测试文件应以 `test_` 开头（例如 `test_simulation.py`），测试函数也应以 `test_` 开头。
- **覆盖率**: 鼓励为所有新功能和错误修复编写单元测试，以保持较高的测试覆盖率。
- **执行**:
  ```bash
  pytest -v test/
  ```

#### 5. Git 提交规范

- **格式**: 遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范，使提交历史清晰可读。
- **结构**: `<type>[optional scope]: <description>`
- **常用类型**:
  - `feat`: 引入新功能。
  - `fix`: 修复错误。
  - `docs`: 修改文档。
  - `style`: 代码格式化，不影响代码逻辑。
  - `refactor`: 代码重构。
  - `test`: 增加或修改测试。
  - `chore`: 构建过程或辅助工具的变动。
- **示例**:
  ```
  feat(simulation): add support for pulsed plasma scenarios
  fix(parser): correct handling of scientific notation in config
  ```

## 许可证

本项目尚未指定许可证。请根据需要选择合适的开源许可证（如 MIT）。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
`
