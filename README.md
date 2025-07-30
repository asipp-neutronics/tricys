
# tricys - TRitium Integrated CYcle Simulation

## 项目简介

tricys（TRitium Integrated CYcle Simulation）是一个用于分析聚变燃料循环的仿真系统，基于 OpenModelica 平台开发。该项目旨在通过建模和仿真，研究聚变反应堆燃料循环的动态行为，特别关注参数（如 `blanket.TBR`）对系统性能的影响。

当前版本提供了一个图形用户界面（GUI）和模块化的参数扫描功能，方便用户配置仿真参数、运行多次仿真并分析结果。

## 功能特点

- **图形用户界面**：
  - 通过 Tkinter 提供直观的交互界面，支持输入模型名称、参数范围（如 `blanket.TBR`）、仿真时间和步长。
  - 显示仿真成功或错误提示，适合初学者和专业用户。
- **参数扫描仿真**：
  - 支持单参数扫描，自动运行多组仿真，整合结果为单一 CSV 文件（`temp/combined_simulation_results.csv`）。
  - 输出限制为关键变量（如 `time` 和 `sds.I[1]`），提高效率。
- **OpenModelica 集成**：
  - 使用 `OMPython` 加载和运行 Modelica 模型（如 `example.Cycle`）。
- **文件管理**：
  - 动态创建和清理 `temp/` 目录中的临时文件，优化磁盘使用。
  - 通过 `.gitignore` 忽略 `temp/`、`__pycache__/` 和 `*.pyc`，保持仓库整洁。
- **跨平台支持**：
  - 基于 Python 和 OpenModelica，支持 Windows、Linux 和 macOS。

## 安装与依赖
为了简化开发环境的配置，本项目提供了  `docker-compose.yml` 文件，用于在容器化环境中运行和测试代码。

### 环境要求
- **Docker**: 最新版本（从 [Docker 官网](https://www.docker.com/) 下载）。
- **Docker Compose**: 最新版本（通常随 Docker Desktop 一起安装）。
- **Windows 11**: 最新版本（安装 WSL2 并默认启用 WSLg 功能）。
- **Ubuntu 24.04**: 运行 `xhost +local:` 命令。
- [ ] **CentOS**
- [ ] **Rocky**

### 安装步骤
1.  **克隆本仓库**:
    ```
    git clone https://github.com/asipp-neutronics/tricys.git
    cd tricys
    ```

2.  **构建 Docker 镜像**：
    在项目根目录下，运行以下命令构建镜像：
    ```bash
    docker compose build
    ```

3.  **启动开发容器**：此命令将在后台启动一个服务容器，默认加载并打开`OMEdit`
    ```bash
    docker compose up -d
    ```

4.  **在容器内执行命令**：
    通过以下命令进入正在运行的容器，以执行代码或运行测试：
    ```bash
    docker compose exec openmodelica-gui bash
    ```

5.  **停止和清理**：
    完成开发后，使用以下命令停止并移除容器：
    ```bash
    docker compose down -v
    ```
## 项目说明
### 文件结构
```
tricys/
├── .vscode/                # VS Code 编辑器配置
├── docs/                   # 项目文档
├── example/                # 示例模型
├── src/                    # Python 源代码
│   ├── analysis/           # 数据分析模块
│   ├── manager/            # 管理器模块（配置、日志等）
│   ├── simulation/         # 仿真运行模块
│   └── utils/              # 通用工具模块
├── test/                   # 测试代码
├── .gitignore              # Git 忽略文件配置
├── .env                    # .env文件配置
├── config.json             # 项目配置文件
├── docker-compose.yml      # Docker Compose 配置文件
├── Dockerfile              # Docker 镜像配置文件
├── pyproject.toml          # Python 项目配置文件
└── README.md               # 项目开发说明
```
### 用户配置
1. `.env`：该配置文件中需要设置`CUSTOM_MODEL_PATH=****`来指定宿主机中用户的OpenModelica模型所在目录
2. `config.json`:该配置文件中表示默认参数值，如paths为路径参数，logging为日志参数，simulation为仿真运行参数, sweep_parameter为扫描参数，overrides_parameter为模型覆盖参数。

## 使用方法
1. 安装项目：`make dev-install` or `make install`
2. 执行测试：
    - 单次运行仿真：`python3 -m simulation.single_simulation`
    - 扫描运行仿真：`python3 -m simulation.sweep_simulation`
    - 图行运行仿真：`python3 -m simualtion.visual_simualtion`
3. 清理数据：`make clear`
4. 规范代码：`make check`

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
  def calculate_tbr(breeding_ratio: float, neutron_multiplier: float) -> float:
      """计算氚增殖比 (Tritium Breeding Ratio)。

      Args:
          breeding_ratio: 基础增殖比。
          neutron_multiplier: 中子倍增因子。

      Returns:
          计算得出的总 TBR 值。

      Raises:
          ValueError: 如果任一参数为负数。
      """
      if breeding_ratio < 0 or neutron_multiplier < 0:
          raise ValueError("参数不能为负数")
      return breeding_ratio * neutron_multiplier
  ```

#### 4. 测试 (Testing)

- **框架**: 使用 `pytest` 作为主要的测试框架。
- **位置**: 所有测试代码应放在 `test/` 目录下。
- **命名**: 测试文件应以 `test_` 开头（例如 `test_simulation.py`），测试函数也应以 `test_` 开头。
- **覆盖率**: 鼓励为所有新功能和错误修复编写单元测试，以保持较高的测试覆盖率。
- **执行**:
  ```bash
  pytest test/
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
