# TRICYS 全栈多物理场 Docker 容器认知、基准测试与联合开发实战指南 (Docker Container & Co-Development Guide)

> **理论与架构基石**:  
> TRICYS v2.0 聚变堆氚燃料循环仿真系统不仅包含宏观微分代数方程（DAE）系统仿真，还深度集成了多个异构物理微观求解器、复杂现代 Web 全栈体系与数字化大屏。  
> 为了彻底消除多环境、多语言、多编译器、多依赖冲突的部署泥潭，TRICYS 全面推行 **Single-Container Unified All-in-One (`tricys_allinone`)** 容器架构。  
> 本指南系统梳理容器认知模型、宿主机极速落地与算例验证全流程，以及插件与已备求解器联合开发和调测闭环。

---

## 🧭 1. 统一 All-in-One 容器架构认知 (Container Architecture & Mental Model)

### 1.1 为什么采用 All-in-One 单容器架构？
传统聚变工程仿真通常涉及 Fortran/C++ 有限元库、Modelica 编译器、.NET 化工流程模拟器、Python 科学计算生态与现代前端可视化栈。若在不同宿主机系统（Windows / Ubuntu / macOS）直接原生安装，极易引发 GCC 版本冲突、PETSc/SLEPc 动态链接库断裂、Mono/.NET 运行时不兼容及 MSL 库路径混乱。

TRICYS 统一容器（基于 `dolfinx/dolfinx:v0.9.0` 深度定制）将以下全套工业与科学环境高度固化在一具容器镜像中：
1. **多保真度物理求解引擎 (Physics Solvers)**：
   - **OpenModelica 1.27.1 + OMSimulator + OMPython**：全厂 DAE 宏观动态求解、FMI 2.0 / SSP 导入导出；
   - **固化 Modelica 标准库 (MSL 4.1.0, 4.0.0, 3.2.3)**：内置自愈镜像，无需联网实时拉取库；
   - **FESTIM 2.0 + DOLFINx 0.9.0 + PETSc + SLEPc**：部件级有限元氢/氚多维扩散、滞留与热传导；
   - **DWSIM 8.8.4 (.NET / pythonnet)**：无头独立化工厂区流程模拟器，支持物性计算与化学平衡；
   - **CasADi >= 3.6 + h2iso**：深冷氢同位素多塔精馏 MESH 严密非线性撕裂流方程求解器；
   - **FMPy >= 0.3.26**：多速率联合仿真与 FMU 调度。
2. **现代全栈 Web 与数字孪生服务 (Fullstack Web & Digital Twin Services)**：
   - **FastAPI + Uvicorn + SQLite + WebSockets**：工业级 REST/WebSocket 后端 API；
   - **Vue 3 + Vite (`tricys_visual`)**：工业 Web 仿真管理控制台；
   - **GoView (`tricys_goview`)**：聚变堆氚燃料循环数字孪生动态大屏；
   - **Dash + Plotly + PyTables (`tricys_visualizer`)**：交互式 HDF5 科学曲线分析与工程报告生成器；
   - **Nginx 反向代理网关**：单端口聚合路由。

---

### 1.2 双端热挂载模型 (Host-Container Live Mount & Hot-Reload Paradigm)

TRICYS 容器开发遵循**“宿主机编码、容器内求解、全实时同步”**的核心范式：

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 宿主机 (Host Machine - Windows / macOS / Linux)                              │
│ - 开发者 IDE (VS Code / Cursor / PyCharm) 浏览与编辑代码                     │
│ - Git 版本控制与分支合并                                                     │
│ - 浏览器交互 (访问 http://localhost:8080 等前端页面)                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Docker Volume 实时双向热挂载
                                       ▼ (文件变更在两端无缝、即时反映，无需重构镜像)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 统一容器 tricys_allinone (/workspace/tricys)                                 │
│ - OpenModelica 1.27.1 / OMSimulator / MSL 4.1.0                             │
│ - FESTIM 2.0 / DOLFINx 0.9.0 (/dolfinx-env)                                 │
│ - DWSIM 8.8.4 (/opt/dwsim) / pythonnet 3.0.3                                │
│ - h2iso (/workspace/h2iso)                                                  │
│ - Nginx (80) + FastAPI (8000) + Dash (8050) + GoView (3020)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 关键卷挂载映射：
- `.:/workspace/tricys`：宿主机当前 TRICYS 仓库根目录直接挂载至容器内部工作区。在宿主机上修改任何 Python 源码、Modelica 模型、YAML 算例或插件代码，**容器内秒级实时生效**，无需 `docker build` 或 `docker cp`。
- `${TRICYS_BACKEND_PATH:-./tricys_backend}:/workspace/tricys_backend`：后端工程热挂载。
- `${H2ISO_PATH:-../h2iso}:/workspace/h2iso:ro`：h2iso 求解器只读挂载。
- `tricys_data:/data`：持久化 Docker Volume，保护 SQLite 数据库、计算工作区与 HDF5 上下文不随容器销毁而丢失。

---

### 1.3 端口映射与服务访问拓扑

| 宿主机端口 | 容器内端口 | 服务组件 | 访问 URL / 说明 |
| :--- | :--- | :--- | :--- |
| **`8080`** | `80` (Nginx) | **统一反向代理网关** | `http://localhost:8080/` (Vue3 控制台)<br>`http://localhost:8080/api/` (API 网关)<br>`http://localhost:8080/goview/` (数字孪生)<br>`http://localhost:8080/hdf5/` (HDF5 交互可视化) |
| **`8000`** | `8000` (Uvicorn) | **FastAPI 工业后端** | `http://localhost:8000/docs` (Swagger 交互式文档)<br>`http://localhost:8000/redoc` (ReDoc 文档) |
| **`3020`** | `3020` (Nginx/Node) | **GoView 数字孪生大屏** | `http://localhost:3020/` (直连大屏画布) |
| **`8050`** | `8050` (Dash) | **Dash/Plotly 科学可视化**| `http://localhost:8050/hdf5/` (直连 HDF5 可视化引擎) |

---

### 1.4 用户权限与安全契约
- 容器内运行用户：**`appuser`**（UID 1000, GID 1000），隶属于 `sudo` 组并配置免密 `sudo`。
- 遵循 OpenModelica 与 Linux 安全规范，避免使用 `root` 运行仿真，以防生成的缓存或结果文件在宿主机产生权限死锁（尤其在 Linux 宿主机上）。
- 宿主机执行 `docker exec` 时，**推荐带上 `-u appuser` 参数**。

---

### 1.5 跨平台 GUI 桌面透传 (Universal GUI Bridge)
容器配置了通用的 X11 与 WSLg 桥接机制：
- Windows 11 (WSL2 / Docker Desktop)：自动映射 `/run/desktop/mnt/host/wslg`，原生支持 Wayland/WSLg 图形加速；
- Linux / X11 Server：自动映射 `/tmp/.X11-unix`。
在容器内启动 `omedit` 时，图形窗口将直接透传至宿主机桌面，方便工程人员进行拓扑连线检查。

---

## 🚀 2. 宿主机全流程落地与工业基准算例测试 (Host Quickstart & Benchmark Validation)

当工程师或 Agent 将本仓库克隆到宿主机后，执行以下标准化流程进行环境验证与全算例测试：

### 2.1 步骤 1：克隆仓库并检出目标分支
在宿主机终端中执行：
```bash
git clone https://github.com/asipp-neutronics/tricys.git
cd tricys
git checkout V2.0
```

### 2.2 步骤 2：启动一站式 Docker 容器
通过 Docker Compose 启动统一容器：
```bash
# 启动后台守护进程 (若本地无预构建镜像则自动触发构建)
docker compose up -d

# 检查容器状态与健康探针 (应当显示 healthy 状态)
docker compose ps
```
若需查看后台初始化日志与各微服务启动情况：
```bash
docker logs -f tricys_allinone
```
> **预期日志指示**: Nginx 启动在 80，Uvicorn 启动在 8000，Dash Visualizer 启动在 8050，各组件就绪。

### 2.3 步骤 3：运行工业基准测试算例 (V2 Examples)

容器内置全功能 `tricys` CLI 工具（封装了 `python3 -m tricys.main`）。用户可采用两种方式进行算例验证：

#### 方式 A：自动化交互/批量算例运行器 (`tricys example`)
TRICYS 提供了覆盖 Schema 2.0 全部核心业务的工业级基准算例集（01 ~ 12）：
```bash
# 列出所有可用工业算例
docker exec -it -u appuser tricys_allinone tricys example --list

# 运行指定的单个算例 (例如算例 1: 启动自持亏空扫描)
docker exec -it -u appuser tricys_allinone tricys example --run 1 --no-vis

# 验证算例配置合法性
docker exec -it -u appuser tricys_allinone tricys example --validate 1

# 编译算例为底层显式 Pipeline
docker exec -it -u appuser tricys_allinone tricys example --compile 1
```

#### 方式 B：通过 TRICYS CLI 直接运行 Schema 2.0 工业 YAML 算例
配合 `--diagnose` 自愈屏障运行各个代表性算例：
```bash
# 1. 扫描与自持评估 (SC-1)
docker exec -it -u appuser tricys_allinone python3 -m tricys.main run \
  -c /workspace/tricys/tricys/example/example_data/v2_examples/01_sweep_startup_self_sufficiency.yaml --diagnose

# 2. 长脉冲工况二分寻优 (SC-2)
docker exec -it -u appuser tricys_allinone python3 -m tricys.main run \
  -c /workspace/tricys/tricys/example/example_data/v2_examples/02_simulate_pulsed_transient.yaml --diagnose

# 3. Sobol 全局敏感度分析 (SC-3)
docker exec -it -u appuser tricys_allinone python3 -m tricys.main run \
  -c /workspace/tricys/tricys/example/example_data/v2_examples/03_sensitivity_sobol_gsa.yaml --diagnose

# 6. 全系统质量守恒审计 (SC-4)
docker exec -it -u appuser tricys_allinone python3 -m tricys.main run \
  -c /workspace/tricys/tricys/example/example_data/v2_examples/06_simulate_mass_conservation_audit.yaml --diagnose

# 7. 自主开发插件流水线 (SC-5)
docker exec -it -u appuser tricys_allinone python3 -m tricys.main run \
  -c /workspace/tricys/tricys/example/example_data/v2_examples/07_custom_plugin_pipeline.yaml --diagnose
```

### 2.4 步骤 4：运行回归与单元测试套件
在容器内运行 `pytest` 执行系统级回归验证（指定工作区路径 `/workspace/tricys`）：
```bash
# 执行全部单元测试 (100+ 测试项，涵盖编译器、自愈诊断、可视化时基、物理总线等)
docker exec -it -u appuser -w /workspace/tricys tricys_allinone pytest test/ -v

# 仅测试核心自愈、配置校验与 CLI 模块
docker exec -it -u appuser -w /workspace/tricys tricys_allinone pytest test/test_v2_cli.py -v

# 测试交互可视化时基转换与渲染器
docker exec -it -u appuser -w /workspace/tricys tricys_allinone pytest test/visualizer/test_time_axis_unit.py -v
```

### 2.5 步骤 5：交互式可视化与结果验收 (Interactive Visualizer & Delivery)

TRICYS 的 Dash/Plotly 可视化引擎以常驻服务模式运行在容器内（监听容器内 8050 端口，并通过 Nginx 网关反向代理至 `8080/hdf5/`）。系统严格采用**基于安全加密 Token 的数据集上下文隔离机制**（而非无状态的通用文件浏览器）。

#### 访问与加载数据集的两种标准方式：
1. **自动生成带 Token 的访问链接（算例运行完成时）**：
   当运行任何配置了 `visualizer.launch_on_finish: true` 的 Schema 2.0 算例（如工业基准算例 01 ~ 12）后，仿真引擎在结果存储与报告导出完成后，会自动签发加密 Context Token，并在终端标准输出末尾打印访问链接：
   ```text
   INFO: 'launch_on_finish' is enabled. Launching visualizer...
   INFO: Reusing running shared HDF5 service on port 8050
   INFO: HDF5 visualizer available: http://localhost:8050/hdf5/?token=<JWT_TOKEN>
   ```
   宿主机浏览器直接访问该 URL（或将端口替换为反向代理网关 `http://localhost:8080/hdf5/?token=<JWT_TOKEN>`），即可无缝载入该次仿真的全部工况数据。

2. **主动调起/切换任意历史或指定 HDF5 仿真结果文件**：
   若需对宿主机或容器内已有的任意 HDF5 仿真结果文件（例如 `<timestamp>/results/sweep_results.h5`）启动可视化分析，直接在宿主机执行：
   ```bash
   docker exec -it -u appuser -w /workspace/tricys tricys_allinone \
     tricys hdf5 <path/to/results.h5>
   ```
   终端将秒级复用常驻服务并输出专属安全访问链接（含 `?token=...`）。

#### 交互式页面四大核心功能区结构：
在浏览器打开带 Token 的链接后，页面呈现实测的四大功能板块：
- **顶部运行概览卡片 (Run Overview)**：展示当前运行目录、工况总数 (Total Jobs)、时间跨度（自动按小时标准化为 `Time (h)`）、输出变量数、HDF5 文件大小与最后修改时间；
- **1. Run Details (Config & Logs)**：手风琴式折叠卡片，可直接展开查阅仿真运行的原始 YAML 配置文本与底层详细执行日志；
- **2. Select Simulation Jobs**：工况作业明细数据表，展示各工况参数（如 TBR、燃烧份额）与标定指标，支持分页筛选与勾选，点击 **“Apply Selection”** 按钮即可驱动下游图表刷新；
- **3. Plot Results & Key Metrics**：
  - **变量时序动态曲线 (Variable Time Series)**：多选变量下拉框、深色工程主题网格、线性/对数坐标切换、X 轴标准小时刻度 `Time (h)`；
  - **相平面与状态轨迹 (Phase Plane)**：任意两个状态变量之间的 2D 状态空间演化轨线；
  - **指标柱状图与统计摘要 (Metrics Bar Chart)**：储罐最低亏空点 $I_{min}$、自持转折点 $t_{turning}$ 等关键指标横向对比；
  - **参数敏感度等高线图 (Contour Plot) 与平行坐标图 (Parallel Coordinates)**：用于参数扫描与优化工况族的高维投影；
- **4. Export Academic Report**：支持实时 Markdown 预览与一键导出三格式出版级工程报告（Markdown、Word DOCX、PDF）及高分辨率矢量图表。

---

## 🔌 3. 插件自主开发与测试调测闭环 (Plugin Co-Development & Verification)

当用户提出 TRICYS 未内置的功能需求（如特定核素计算、专用格式导出、级联安全报警、动态前处理参数映射）时，Agent 或开发者可在宿主机 IDE 中利用双端热挂载无缝开发插件：

```mermaid
sequenceDiagram
    participant Dev as 宿主机 IDE / Agent
    participant Mount as 热挂载 (/workspace/tricys)
    participant Box as 容器内存沙箱 (tricys test)
    participant Pipe as 仿真流水线 (tricys run)

    Dev->>Mount: 1. 脚手架生成与宿主机编写插件代码
    Dev->>Box: 2. 毫秒级内存沙箱测试 (AST & 逻辑校验)
    Box-->>Dev: 零 OMC 耗时瞬时反馈 (PASS / Fail)
    Dev->>Mount: 3. 在 run_case.yaml 中声明式挂载插件
    Dev->>Pipe: 4. 驱动真实仿真 (带 --diagnose 屏障)
    Pipe-->>Dev: 仿真成功交付 / 报错读取 diagnostics.json 自愈
```

### 3.1 步骤 1：生成标准插件脚手架
在宿主机终端或容器内执行脚手架命令生成规范代码骨架：
```bash
docker exec -it -u appuser tricys_allinone python3 -m tricys.main plugin scaffold \
  custom_tritium_decay_auditor \
  --category postprocess \
  -o /workspace/tricys/plugins/custom_tritium_decay_auditor.py
```

### 3.2 步骤 2：在宿主机 IDE 中实现插件逻辑
由于热挂载，宿主机项目根目录已出现 `plugins/custom_tritium_decay_auditor.py`。在宿主机 IDE 打开并利用物理语义总线与强类型契约编写业务逻辑：

```python
"""自定义后处理插件示例：核素衰变与泄漏综合审计器."""
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from tricys.workflow.base import PostprocessPlugin
from tricys.workflow.context import PipelineContext

class CustomTritiumDecayAuditorPlugin(PostprocessPlugin):
    """自主开发的核素衰变与泄漏综合审计插件."""

    def __init__(self, alert_threshold_g: float = 100.0, **params: Any) -> None:
        super().__init__(**params)
        self.alert_threshold_g = alert_threshold_g

    def validate(self, context: PipelineContext) -> bool:
        """运行前校验上下文契约."""
        if not context.results:
            self.logger.warning("No simulation results found in context.")
            return False
        return True

    def run(self, context: PipelineContext) -> None:
        """利用物理语义总线执行审计核算."""
        for job in context.results:
            df = job.dataframe
            # 1. 通过物理语义总线解耦查询变量 (摆脱具体模型列名差异)
            inv_series = context.resolve_physics_variable(df, "sds_inventory")
            if inv_series is None or inv_series.empty:
                continue

            max_inv = float(inv_series.max())
            # 2. 评估自定义物理指标并写入指标池
            job.metrics["auditor_max_sds_g"] = max_inv
            job.extra["decay_audit_passed"] = (max_inv <= self.alert_threshold_g)

            self.logger.info(
                "Audit completed: max SDS inventory = %.2f g (Threshold: %.2f g)",
                max_inv,
                self.alert_threshold_g,
            )
```

> ⚠️ **AST 沙箱红线**：严禁调用 `subprocess`、`threading`、`socket`、`requests`、`eval`、`exec`、`os.system` 等危险操作。系统在加载插件时会自动进行静态 AST 安全扫描。

### 3.3 步骤 3：毫秒级内存沙箱自检 (Mock Testing)
在接入复杂仿真前，必须通过内置的内存沙箱进行 AST 安全与运行逻辑的毫秒级自检：
```bash
docker exec -it -u appuser tricys_allinone python3 -m tricys.main plugin test \
  /workspace/tricys/plugins/custom_tritium_decay_auditor.py --json
```
- **核心价值**：纯内存 Mock 注入时序 DataFrame，**零 OMC 编译时间，无需等待仿真**，秒级验证 `validate()` 与 `run()` 契约；
- 若存在语法错误、AST 沙箱违规或契约缺失，立刻捕获并精准输出行号。

### 3.4 步骤 4：声明式挂载与 YAML 校验
在仿真配置文件（如 `run_case.yaml`）中声明挂载：
```yaml
pipeline:
  name: "CustomAuditWorkflow"
  postprocessing:
    - name: "storage"
    - script: "plugins/custom_tritium_decay_auditor.py"
      class: "CustomTritiumDecayAuditorPlugin"
      params:
        alert_threshold_g: 250.0
    - name: "report"
```
校验配置合法性：
```bash
docker exec -it -u appuser tricys_allinone python3 -m tricys.main validate -c /workspace/tricys/run_case.yaml
```

### 3.5 步骤 5：带自愈屏障端到端执行
```bash
docker exec -it -u appuser tricys_allinone python3 -m tricys.main run \
  -c /workspace/tricys/run_case.yaml --diagnose
```
若插件抛出异常，读取导出的 `results/diagnostics.json` 进行自愈修复。

---

## 🔬 4. 容器内异构仿真程序联合开发与调试 (Multi-Physics Solver Co-Development)

TRICYS 容器已预置四大物理求解环境。下述工作流指导开发者如何利用容器资源进行单体调试与全厂协同仿真联调：

### 4.1 OpenModelica 1.27.1 (`omc` / OMPython / OMEdit)

- **定位**：全厂 0-D 宏观 DAE 动态微分代数方程组基石求解器。
- **环境确认**：
  ```bash
  docker exec -it -u appuser tricys_allinone omc --version
  ```
- **CLI 审查与自省**：
  在宿主机修改 `.mo` 模型后，直接通过容器执行四大维度静态健康审查与无幻觉自省（以仓库标准单文件基准模型为例）：
  ```bash
  # 静态健康审查 (时间量纲、流股守恒、初值稳健性)
  docker exec -it -u appuser tricys_allinone python3 -m tricys.main model review \
    /workspace/tricys/tricys/example/example_data/example_model_single/example_model.mo example_model.Cycle --json

  # 深度自省提取实测变量
  docker exec -it -u appuser tricys_allinone python3 -m tricys.main model inspect \
    /workspace/tricys/tricys/example/example_data/example_model_single/example_model.mo example_model.Cycle --json
  ```
- **桌面 GUI 交互检查 (OMEdit)**：
  在支持 WSLg 或 X11 Server 的宿主机上，可直接从容器弹出原生 OMEdit 图形界面：
  ```bash
  docker exec -it -u appuser -e DISPLAY=$DISPLAY tricys_allinone omedit \
    /workspace/tricys/tricys/example/example_data/example_model_single/example_model.mo
  ```

---

### 4.2 FESTIM 2.0 / DOLFINx (`/dolfinx-env`)

- **定位**：包层第一壁、偏滤器部件级高精度微观有限元（FEM）氢/氚多维热扩散、捕获与渗透计算。
- **环境隔离与确认**：
  FESTIM 2.0 运行在容器内预编译的高性能环境 `/dolfinx-env` 下：
  ```bash
  docker exec -it tricys_allinone /dolfinx-env/bin/python3 -c \
    "import festim, dolfinx; print(f'FESTIM: {festim.__version__}, DOLFINx: {dolfinx.__version__}')"
  ```
- **开发与联调范式（模式 2：`online_step` 在线步进）**：
  1. **编写处理器**：继承 `tricys.online_cosim.processor_base.AbstractTrackProcessor`，实现 `initialize` 与 `process`；
  2. **单体脚本验证**：在宿主机编写 FESTIM 仿真逻辑，在容器内独立执行单元测试验证网格收敛性：
     ```bash
     docker exec -it -u appuser tricys_allinone /dolfinx-env/bin/python3 \
       /workspace/tricys/tricys/example/example_data/v2_examples/09_cosim_online_step/festim_div_processor.py
     ```
  3. **联调运行**：在 YAML 算例中配置 `mode: online_step`，将偏滤器部件绑定为 FESTIM 外部求解器：
     ```bash
     docker exec -it -u appuser tricys_allinone python3 -m tricys.main run \
       -c /workspace/tricys/tricys/example/example_data/v2_examples/09_cosim_online_step.yaml --diagnose
     ```

---

### 4.3 DWSIM 8.8.4 (.NET / pythonnet)

- **定位**：化学工艺流股模拟、同位素分离稳态热力学物性与吸收解吸床流程模拟。
- **环境路径**：位于 `/opt/dwsim`。`dwsim_iss_handler.py` 采用双模自愈设计：优先通过 `clr` 加载 DWSIM Automation3 流程图；在无头 Linux/非 Mono 环境下，自动平滑切入内置的高精度三塔深冷精馏代理解析内核。
- **环境与 Handler 调测验证**：
  ```bash
  # 验证 DWSIM Handler 单体脚本执行与物料平衡
  docker exec -it -u appuser tricys_allinone python3 \
    /workspace/tricys/tricys/example/example_data/v2_examples/08_cosim_offline_table/dwsim_iss_handler.py

  # 运行 DWSIM 协同仿真单元回归测试
  docker exec -it -u appuser -w /workspace/tricys tricys_allinone pytest test/test_dwsim_handler.py -v
  ```
- **开发与联调范式（模式 1：`offline_table` 离线查表）**：
  1. **编写 Handler 函数**：`def run_dwsim_simulation(temp_input_csv: str, temp_output_csv: str, **params) -> dict`；
  2. **单体验证**：在宿主机修改 DWSIM 流程图文件（`.dwxmz`）或 Handler 代码，在容器内执行单体测试验证物性平衡；
  3. **联调运行**：执行离线查表协同仿真算例：
     ```bash
     docker exec -it -u appuser tricys_allinone python3 -m tricys.main run \
       -c /workspace/tricys/tricys/example/example_data/v2_examples/08_cosim_offline_table.yaml --diagnose
     ```

---

### 4.4 h2iso (深冷氢同位素分离精馏 MESH 严密求解器)

- **定位**：多塔深冷同位素精馏系统 MESH 方程（Material, Equilibrium, Summation, Heat）严密非线性撕裂流迭代求解。
- **环境挂载与路径**：位于 `/workspace/h2iso`（依托 CasADi 非线性求解优化库）。
- **环境确认**：
  ```bash
  docker exec -it tricys_allinone python3 -c \
    "import casadi; print(f'CasADi {casadi.__version__} ready for h2iso')"
  ```
- **开发与联调范式（模式 3：`parameter_injection` 参数注入）**：
  1. **编写参数注入 Handler 脚本**：从 `TRICYS_HANDLER_PARAMS` 获取入口流股条件，驱动 `h2iso` 计算精馏塔平衡，生成包含 Modelica 实例变量的 `override_cycle.txt`；
  2. **替换模型拓扑强校验**：提供 0-D 动态代理模型（如 `Generic_ISS_Adapters.ISS_I_Adapter`），确保与原模型端口命名、因果方向及向量维度 100% 对应，杜绝接口悬空；
  3. **单体脚本测试**：
     ```bash
     docker exec -it -u appuser tricys_allinone python3 \
       /workspace/tricys/tricys/example/example_data/v2_examples/10_cosim_parameter_injection/run_h2iso_wang2022.py
     ```
  4. **全厂极速长周期联调**：
     ```bash
     docker exec -it -u appuser tricys_allinone python3 -m tricys.main run \
       -c /workspace/tricys/tricys/example/example_data/v2_examples/10_cosim_parameter_injection.yaml --diagnose
     ```
     > **优势**: 既保持了微观同位素精馏严密机理物性，又让全厂千小时级演化在 20~40 秒内收敛完成。

---

## 🛠️ 5. 容器运维、热重启与故障自愈速查 (Troubleshooting & Maintenance)

### 5.1 修改前端或后端代码后的服务热重启
由于宿主机到容器的目录热挂载：
- 修改核心算法或 CLI 脚本：**瞬时生效，无需任何重启**；
- 修改了 Dash 可视化器 (`tricys/visualizer/`) 或 FastAPI 后端服务 (`tricys_backend/`)：
  ```bash
  # 秒级重启容器服务 (不重新构建镜像，保持挂载与持久化数据)
  docker restart tricys_allinone
  ```

### 5.2 宿主机权限异常自愈 (Linux 宿主机文件属主问题)
若在 Linux 宿主机上出现文件权限受限：
```bash
docker exec -it tricys_allinone chown -R 1000:1000 /workspace/tricys
```

### 5.3 容器健康状态与进程巡检
```bash
# 查看容器各服务资源消耗
docker stats tricys_allinone --no-stream

# 查看容器内运行的进程树
docker top tricys_allinone

# 实时追踪全栈服务标准输出日志
docker logs -f --tail 200 tricys_allinone
```

---

## 📋 6. 容器常用交互命令速查表 (Container CLI Cheat Sheet)

| 场景操作 | 宿主机执行命令 |
| :--- | :--- |
| **启动容器环境** | `docker compose up -d` |
| **查看服务状态** | `docker compose ps` |
| **停止容器环境** | `docker compose down` |
| **重启容器微服务** | `docker restart tricys_allinone` |
| **列出全部工业算例** | `docker exec -it -u appuser tricys_allinone tricys example --list` |
| **执行指定基准算例** | `docker exec -it -u appuser tricys_allinone tricys example --run <id> --no-vis` |
| **运行 Schema 2.0 算例** | `docker exec -it -u appuser -w /workspace/tricys tricys_allinone python3 -m tricys.main run -c <path/to/case.yaml> --diagnose` |
| **主动签发 HDF5 可视化链接** | `docker exec -it -u appuser -w /workspace/tricys tricys_allinone tricys hdf5 <path/to/results.h5>` |
| **模型静态健康审查** | `docker exec -it -u appuser -w /workspace/tricys tricys_allinone python3 -m tricys.main model review <path/to/model.mo> <ModelName> --json` |
| **模型变量深度自省** | `docker exec -it -u appuser -w /workspace/tricys tricys_allinone python3 -m tricys.main model inspect <path/to/model.mo> <ModelName> --json` |
| **插件脚手架生成** | `docker exec -it -u appuser -w /workspace/tricys tricys_allinone python3 -m tricys.main plugin scaffold <name> -o <path>` |
| **插件内存沙箱测试** | `docker exec -it -u appuser -w /workspace/tricys tricys_allinone python3 -m tricys.main plugin test <path> --json` |
| **运行全量回归测试** | `docker exec -it -u appuser -w /workspace/tricys tricys_allinone pytest test/ -v` |
| **打开 OMEdit 图形界面** | `docker exec -it -u appuser -e DISPLAY=$DISPLAY tricys_allinone omedit <path/to/model.mo>` |
| **进入容器交互 Bash** | `docker exec -it -u appuser tricys_allinone bash` |
