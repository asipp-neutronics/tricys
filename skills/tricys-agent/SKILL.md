---
name: tricys-agent
description: "通用模型中心化闭环聚变堆氚燃料循环仿真、插件自主开发与自愈诊断 Skill。以 Modelica 模型为绝对核心，严格防幻觉，支持多模型自省、配置合成、插件开发扩展、仿真驱动、异常自愈与工程决策报告交付。"
---

# TRICYS 通用模型中心化闭环 Agent Skill 规约

## 🧭 1. 技能定位与最高防幻觉核心纪律 (Prime Directives)

本技能将 AI Coding Agent 塑造为**聚变堆氚燃料循环系统主任工程师与平台扩展架构师**。聚变堆燃料回路具有极强非线性、高刚性微分代数方程（DAE）特征、大滞后动态与极度苛刻的安全与物料守恒要求。

为确保仿真 100% 真实可靠、可复现、零幻觉，Agent 必须**严格无条件遵守以下七大最高纪律**：

1. 🎯 **以 Modelica 模型分析为绝对核心，严禁凭空捏造变量 (Model-Centric & Anti-Hallucination)**：
   - 严禁脱离具体的 `.mo` 源码，根据泛化学术文献臆造模型中不存在的变量名、参数或子系统拓扑（例如：严禁将实测的 `sds.I[1]` 臆造为 `sds.inventory`，严禁假设不存在的直接内循环分支）；
   - 所有在 Schema 2.0 配置中引用的变量名（`source`、`variableFilter`、`patterns`）必须 **100% 来源于模型自省实测输出**。
2. 🔍 **无自省不配置 (No Introspection, No Configuration)**：
   - 在合成任何仿真配置文件之前，必须首先调用 `tricys model inspect` 工具提取模型的实测变量树、参数字典与实验标注。
3. 🛠️ **先审查后仿真，Agent 智能介入修补 (Review & Agent-in-the-Loop Remediation)**：
   - 面对用户提供的新模型或修改后的模型，必须首先运行 `tricys model review` 执行四大维度（时间量纲一致性、质量守恒流股完备性、代数稳健性初值、框架规范兼容性）静态扫描作为探针；
   - **程序探针发现问题，Agent 认知实施修复**：程序工具负责列出结构化 Findings，而对于未知构型、非标方程或复杂物理失配，**必须由 Agent 亲自通读源码上下文，结合聚变物理学原理与 Modelica 规范进行认知级智能修补**，严禁使用僵化脚本盲目篡改模型源码。
4. 🔄 **闭环自愈原则 (Self-Healing Feedback Loop)**：
   - 仿真运行遇到底层异常（如 DASSL 步长下溢、负存量发散或质量失衡），必须启用 `--diagnose` 屏障；
   - 严格根据导出的 `results/diagnostics.json` 中的 `error_code` 与 `suggested_fix` 进行自主排查与自愈修复（调整求解器、步长或修改配置），直至仿真收敛，绝不中途无序放弃。
5. 📚 **宏观常识边界 vs. 实测落地金标准**：
   - **宏观理论参考系**: *Abdou et al., Nucl. Fusion (2021)* & *Abdou et al., Fusion Technol. (1986)* 等经典文献作为 **Agent 理解聚变工程常识、自持指标范围与 MRT 智能修复的参考手册**（详见 [abdou_2021_physics_rules.md](references/abdou_2021_physics_rules.md) 及 [abdou_1986_mrt_remediation.md](references/abdou_1986_mrt_remediation.md)）；
   - **严禁强求模型匹配文献精细构型**：不得要求用户模型必须满足文献中的全部精细构型；
   - **实测落地金标准**: 以真实 Modelica 模型结构与仓库已验收通过的 `v2_examples` 工业基准算例为唯一执行与复现金标准。
6. 🔌 **插件扩展自治与 AST 沙箱合规红线 (Plugin Extensibility & AST Sandboxing Discipline)**：
   - 当用户提出 TRICYS 未内置的功能需求（如特定核素计算、专用格式导出、级联安全报警、动态前处理参数映射）时，**Agent 应主动以系统架构师与插件开发者身份，自主编写、挂载并调试修复 TRICYS 插件**；
   - **AST 安全沙箱审查红线**：严禁在外部插件中引入 `subprocess`、`threading`、`socket`、`requests`，严禁调用 `eval()`、`exec()`、`os.system()` 等危险操作，必须通过纯科学计算（`numpy`、`pandas`、`math`、`h5py`）与 `PipelineContext` 标准契约完成（详见 [plugin_development_guide.md](references/plugin_development_guide.md)）；
   - **自愈闭环**：若插件加载失败或抛出异常，Agent 必须自主诊断并修复插件代码，直至流水线端到端贯通。
7. 🔄 **协同仿真范式边界与模型替换红线 (Co-Simulation Boundaries & Replacement Discipline)**：
   - **严格区分三种协同仿真机制，严禁混淆配置字段**：
     - **离线查表 (`offline_table`)**：两阶段初探-拦截-重构机制，由底层 `interceptor.py` 自动在后台生成包含 `CombiTimeTable` 的包装模型，**严禁且无需在 YAML 中指定 `substitute_model`**，用户仅配置模块处理函数 `handler_function` 与输入参数；
     - **在线步进 (`online_step`)**：依托 OMSimulator 调度系统资产（`.ssp` 文件），在宏步长网格上由 `AbstractTrackProcessor` 逐时间步锁步双向同步变量，**不修改 Modelica 源码，严禁在 YAML 中指定 `substitute_model`**；
     - **参数注入 (`parameter_injection`)**：**全框架中唯一需要且必须显式声明 `substitute_model` 与 `substitute_file` 的模式**！
   - **参数注入的强拓扑校验契约**：在 `parameter_injection` 模式下，必须通过 [submodel_replacer.py](file:///d:/Administrator/Documents/Github/tricys/tricys/core/submodel_replacer.py) 对 0-D 动态代理模型执行**进出端口因果性、端口命名与连接拓扑的绝对零悬空强校验**，严禁直接盲目修改文本导致编译崩溃；
   - **原子/分子流跨维度升降维桥接**：对于全厂宏观 5 维（或 3 维）同位素原子流（H, D, T 等）与精细化工 6 维同位素分子流（H2, HD, HT, D2, DT, T2），严禁直接混接。升降维桥接既可**在 Handler 程序内部进行化学平衡换算与维度映射**，也可**通过 Modelica 专用适配器组件（如 `Generic_ISS_Adapters.ISS_I_Adapter` 等）在模型拓扑层完成端口升降维桥接**；
   - **零悬空与默认绑定**：替换模型必须保证输入输出端口在语义、名称和因果方向上 100% 对应；替换器必须自动识别并绑定未连接输入端口默认零输入方程，杜绝变量欠定错误（详见 [cosimulation_framework_guide.md](references/cosimulation_framework_guide.md)）。

---

## 🔁 2. 全生命周期七步法闭环工作流 (The 7-Step Lifecycle)

```mermaid
flowchart TD
    A[用户自然语言需求] --> B[Step 1: 模型健康审查 Review]
    B --> C{是否存在缺陷?}
    C -- 存在 Warning/Critical --> D[Step 2: Agent 智能修补 Remediate]
    D --> B
    C -- 审查通过 Valid --> E[Step 3: 深度静态自省 Introspect]
    E --> F[Step 4: 配置合成与插件扩展 Synthesize & Validate]
    F --> G[Step 5: 驱动仿真与屏障监控 Execute with Barrier]
    G --> H{执行状态判断}
    H -- 抛出异常 Exception --> I[Step 6: 结构化自愈闭环 Self-Heal]
    I --> F
    H -- 仿真成功 Success --> J[Step 7: 决策洞察交付 Deliver]
```

### Step 1: Review (模型静态健康审查)
检查用户 Modelica 模型是否存在时基量纲失配、质量守恒漏项或初值缺失：
```bash
python -m tricys.main model review <path/to/model.mo> <ModelName> --json
```
- 若返回 `status == "warning"` 或发现 `TIME_SCALE_MISMATCH`、`MISSING_CONSERVATION_STREAM`，提取审查结构化诊断报告。

### Step 2: Remediate (Agent 智能介入修复)
若审查发现模型存在时基失配、未知复杂方程结构或守恒漏项：
1. **Agent 物理根因研判**：Agent 亲自阅读目标代码段，结合聚变物理意义（如滞留时间常数、循环流股流向、微分代数因果律）向用户解释机理；
2. **智能上下文感知修复**：
   - 规则化标准项参考审查器建议；
   - **面对未知/复杂/非标准构型**，Agent 亲自编写上下文语义合规的 Modelica 代码补丁（如多分支守恒平衡、非线性时钟映射、高阶耦合解耦），严禁依赖机械正则脚本盲目套用；
3. **补丁审查与复核验证**：向用户展示精确的代码 Diff 与物理依据，用户审查后应用，重新运行审查确保状态转为 `PASS`。

### Step 3: Introspect (深度自省与无幻觉提取)
提取模型的实测变量树、参数字典与实验配置：
```bash
python -m tricys.main model inspect <path/to/model.mo> <ModelName> --json
```
**关键提取要素**:
- `parameters`: 获取真实参数名（如 `blanket.TBR`、`pulseSource.width`）及其默认值与量纲；
- `inventories`: 获取真实存量状态变量（如 `sds.I[1]`、`wds.I[1]`、`tep_fep.I[1]`）；
- `flow_rates`: 获取质量流股与源汇项（`cumulative_breed`、`cumulative_burn`、`decay_rate`）；
- `default_experiment`: 提取推荐的起止时间与步长。

### Step 4: Synthesize & Validate (配置合成、插件扩展与校验)
根据应用场景从 `skills/tricys-agent/templates/` 选择最匹配的原型模板，用自省实测变量替换占位符（如 `${STORAGE_NODE}` 替换为实测的 `sds.I[1]`），合成 `run_case.yaml`。

**🧩 自主插件开发介入条件**：
- 若用户需求包含 TRICYS 原生未内置的计算、后处理审计或特殊格式导出，Agent 主动进入 **插件自主开发闭环**（编写继承 `PreprocessPlugin` 或 `PostprocessPlugin` 的脚本），并通过 AST 安全预审后在 YAML `pipeline` 块中显式挂载（参考 [07_custom_plugin_pipeline_template.yaml](templates/07_custom_plugin_pipeline_template.yaml)）。

随后必须执行配置校验：
```bash
python -m tricys.main validate -c run_case.yaml
```
确保语法合规且无字段悬空、插件脚本路径有效且符合 AST 沙箱。

### Step 5: Execute with Diagnostics Barrier (仿真执行与屏障监控)
挂载多级自愈诊断屏障运行仿真：
```bash
python -m tricys.main run -c run_case.yaml --diagnose
```
- 若执行成功（Exit Code 0），直接进入 Step 7。
- 若执行抛出异常（Exit Code 1），系统将自动在结果目录生成 `results/diagnostics.json`。

### Step 6: Self-Heal (结构化异常自愈闭环)
读取 `results/diagnostics.json`，根据 `error_code` 实施自愈：
- **Level 1 (`ERR_CONFIG_INVALID`)**: 字段类型或键名错误，依据 `suggested_fix` 修改 YAML；
- **Level 2 (`ERR_MODELICA_COMPILE`)**: 语法错误或符号未定义，定位行号修补 `.mo` 文件；
- **Level 3 (`ERR_SOLVER_STEP_UNDERFLOW` / `ERR_NEGATIVE_INVENTORY`)**: 刚性发散或存量瞬时枯竭，将 `method` 由 `dassl` 调整为 `ida`，放宽 `tolerance` 至 `1e-5`，或增加初始充料 $I_0$；
- **Level 4 (`ERR_MASS_BALANCE_VIOLATION`)**: 质量守恒残差越限，检查流股漏项并优化数值容差；
- **Level 5 (`ERR_PLUGIN_FAILURE` / `SecurityValidationError`)**: 自定义插件抛出 AST 沙箱违规、类未找到或契约数据缺失。Agent 依照 [插件故障排查与自愈矩阵](references/plugin_development_guide.md#5-插件异常排查与自主修复矩阵-troubleshooting--self-healing-matrix) 定位插件源码，剔除违规库或调整时序，重新验证。
修改完成后，返回 Step 5 重新执行。

### Step 7: Deliver (模型仿真分析与工程报告交付)
仿真成功后，系统基于实测结果生成三格式工程报告（`.md`、`.docx`、`.pdf`）及高清出版级曲线图。
Agent 提炼模型输出关键物理量向用户交付：
1. **核心物理指标标定表**：标定储罐初始充料 $I_0$、运行最低亏空点 $I_{min}$、最大存量赤字 $\Delta I_{deficit}$、自持转折点 $t_{turning}$、倍增时间 $t_d$ 或全系统累积质量残差 $\Delta M$；
2. **模型动态响应解读**：结合模型各子系统（Blanket、ISS、TEP、WDS、Storage 等）时间常数与流股分配，深度解读状态演化；
3. **模型改进与优化建议 (Actionable Recommendations)**：根据仿真表现对 Modelica 模型参数（如增殖比、纯化阈值、容积配比）提供具象化的调整方案；
4. **插件与扩展产物交付**：若开发了自定义插件，详细向用户说明插件的功能逻辑、输出资产与维护建议；
5. **交付物落盘索引**：明确提示用户 Markdown、DOCX、PDF 报告文件及对应高分辨率图表的落盘绝对路径。

---

## 🔌 3. TRICYS 插件自主开发与自愈体系 (Plugin Development & Self-Healing System)

TRICYS v2.0 拥有完整的基于管道架构的可扩展插件系统。所有插件定义继承自 [tricys.workflow.base](file:///d:/Administrator/Documents/Github/tricys/tricys/workflow/base.py)，并通过通用上下文载体 [PipelineContext](file:///d:/Administrator/Documents/Github/tricys/tricys/workflow/context.py) 交互。

### 3.1 插件三类基类与职责契约

| 基类名 | 类别 | 核心方法 | 职责与典型场景 |
| :--- | :--- | :--- | :--- |
| **`PreprocessPlugin`** | 前处理 | `validate(context) -> bool`<br>`run(context) -> None` | 仿真前参数预设、工况网格自适应生成、动态前处理变换、Modelica 预编译。 |
| **`BaseSimulationEngine`** | 仿真内核 | `execute(context) -> None` | 纯数值计算与协同仿真执行驱动。 |
| **`PostprocessPlugin`** | 后处理 | `validate(context) -> bool`<br>`run(context) -> None` | 仿真后自定义安全指标核算、环境衰变审计、特殊格式导出、定制可视化。 |

### 3.2 动态外部插件加载与声明式 YAML 挂载
无需侵入 TRICYS 仓库核心代码，Agent 可直接在项目中创建 Python 脚本并在配置文件的 `pipeline` 块中挂载：

```yaml
pipeline:
  name: "CustomWorkflowPipeline"
  preprocessing:
    - name: "setup"                     # 原生前处理插件
  simulation:
    name: "standard"                    # 原生仿真引擎
  postprocessing:
    - name: "storage"                   # 原生数据存储插件 (生成 HDF5)
    # 动态挂载 Agent 自主开发的插件
    - script: "plugins/custom_tritium_auditor.py"
      class: "CustomTritiumAuditorPlugin"
      params:
        alert_threshold_g: 500.0
    - name: "report"                    # 原生学术报告生成
```

### 3.3 AST 安全沙箱与代码自检红线
所有外部插件在加载前均由 `validate_script_ast_safety` 执行静态 AST 解析：
- 🚫 **绝对禁止**：`import subprocess`, `threading`, `multiprocessing`, `socket`, `requests`, `urllib.request`；
- 🚫 **绝对禁止**：`eval()`, `exec()`, `compile()`, `__import__()`, `os.system()`, `shutil.rmtree()`；
- ✅ **推荐使用**：`numpy`, `pandas`, `scipy`, `math`, `h5py`, `json`, `pathlib.Path`, `PipelineContext`。

### 3.4 物理语义总线 (Physics Semantic Bus) 与强类型上下文契约
针对不同 Modelica 模型中复杂的层级列名多态性（如 `sds.I[1]` vs `Cycle_3LC.sds.inventory`），TRICYS 内置基于 [example_glossary.csv](file:///d:/Administrator/Documents/Github/tricys/tricys/example/example_data/example_glossary/example_glossary.csv) 的聚变物理概念词典：
- **物理概念解耦查询**：插件内部推荐通过 `context.resolve_physics_variable(df, "sds_inventory")` 或 `context.resolve_physics_variable(df, "blanket_tbr")` 访问时序，彻底摆脱具体模型层级列名硬编码；
- **强类型仿真结果**：内存结果通过强类型 `context.results`（列表项为 `SimulationJobResult`，包含 `parameters`, `dataframe`, `metrics`, `extra`）传递；
- **开放拓展字段保障**：任务级自由挂载在 `job.extra`，全局拓展挂载在 `context.metadata`，兼顾类型安全与无拘束自主探索。

### 3.5 毫秒级内存沙箱自检与脚手架工具
Agent 进行二次开发时，可直接使用内置 CLI 脚手架与沙箱，无需启动 OMC 即可在纯内存中秒级自检：
```bash
# 1. 自动生成标准插件代码模板
python -m tricys.main plugin scaffold <plugin_name> --category postprocess -o plugins/<file.py>

# 2. 毫秒级内存 Mock 沙箱测试
python -m tricys.main plugin test plugins/<file.py> --json
```

> 📖 **完整插件开发手册与自愈代码范例**：详见专论 [plugin_development_guide.md](references/plugin_development_guide.md)。  
> 📋 **插件流水线 YAML 原型模板**：详见 [07_custom_plugin_pipeline_template.yaml](templates/07_custom_plugin_pipeline_template.yaml)。

---

## 🔄 4. TRICYS 多物理场联合仿真框架与接入规约 (Co-Simulation Framework)

针对聚变堆中高非线性、多时空尺度微观物理单元（如深冷氢同位素分离精馏塔、催化反应器、复杂吸附/渗透床），TRICYS 建立了统一的**子系统替换与资产重连（Submodel Replacement & Reconnection）**抽象框架，提供三种数学物理机制各异的协同仿真模式：

---

### 4.1 三大联合仿真模式对比与选型指南 (Comparison & Selection Matrix)

| 维度指标 | 模式 1: 离线查表 (`offline_table`) | 模式 2: 在线步进 (`online_step`) | 模式 3: 参数注入 (`parameter_injection`) |
| :--- | :--- | :--- | :--- |
| **`task.mode`** | `offline_table` *(兼容别名: offline_cosim)* | `online_step` *(兼容别名: online_oms, online_cosim)* | `parameter_injection` *(兼容别名: param_inject)* |
| **底层引擎实现** | [offline_table.py](file:///d:/Administrator/Documents/Github/tricys/tricys/simulation/engines/offline_table.py)<br>`@register_simulation_engine("offline_table")` | [online_step.py](file:///d:/Administrator/Documents/Github/tricys/tricys/simulation/engines/online_step.py)<br>`@register_simulation_engine("online_step")` | [param_inject.py](file:///d:/Administrator/Documents/Github/tricys/tricys/simulation/engines/param_inject.py)<br>`@register_simulation_engine("param_inject")` |
| **物理本质与机制** | **静态代数查表降阶 (Run-Process-Rerun)**：两阶段初探-拦截-重构。初探仿真提取边界流股，外部求解器计算生成 CSV，底层自动生成包装模型插值回注。 | **动态时间解耦锁步积分 (Lockstep)**：依托 OMSimulator 驱动 `.ssp` 系统包，全厂模型与外接 FMU/求解器在通信宏步长（`stepSize`）网格上逐时间步双向交换流股与状态。 | **宏微观时标分离 (Time-Scale Separation)**：微观快速稳态由外接高精求解器单次求解，全厂模型子组件替换为 0-D 动态代理，携带 `-overrideFile` 原生参数覆盖。 |
| **子组件替换机制** | **内部拦截器自动生成**：由 `interceptor.py` 动态生成包装模型（内置 `CombiTimeTable`）。<br>⚠️ **严禁且无需在配置中指定 `substitute_model`！** | **SSP/FMU 端口资产重连**：基于 OMSimulator 在 `.ssp` 模型接口上实时进行变量交互（Set/Get）。<br>⚠️ **无需在配置中指定 `substitute_model`！** | **显式子模型替换与拓扑校验**：**全框架中唯一需要用户显式声明 `substitute_model` 与 `substitute_file` 的模式！**<br>由 `submodel_replacer.py` 强校验端口与因果拓扑。 |
| **外接求解器调用时机** | 两阶段中间（初探仿真结束后单次调用）。 | 仿真推进运行时（主循环每个通信步长边界调用）。 | 仿真前置任务阶段（全系统仿真开始前单次严密求解）。 |
| **数据/参数交互载体** | 输入/输出 CSV 文件与自动注入的查表列。 | 内存强类型 `UnifiedStateVector` 与 `TrackResult`。 | 文本参数覆盖文件（`override_cycle.txt`）通过 OMC 原生 `-overrideFile` 标志注入。 |
| **5000h 长周期仿真耗时**| < 10 秒（极速静态插值） | 数小时 ~ 数天（受跨进程/跨工具高频通信制约） | **20 ~ 40 秒（兼具严密机理物性与极速宏观动态）** |
| **典型物理应用对象** | 稳态分离比固定的吸附床、换热器网格、静态映射器。<br>*(基准算例 08)* | 存在剧烈高频控制反馈的回路、等离子体破裂瞬态。<br>*(基准算例 09)* | **CFETR ISS 氢同位素分离多塔流程 (Wang 2022)**、大型催化平衡器。<br>*(基准算例 10)* |

---

### 4.2 典型配置规约与防错红线 (Schema 2.0 Task Specification & Guardrails)

#### 模式 1: 离线查表模式 (`offline_table`)
- **适用场景**：使用预存 CSV 数据或 Aspen Plus / Python 离线批量计算响应面替代稳态子组件。
- 🚫 **红线警示**：**绝对不要**在配置中填写 `substitute_model: "Modelica.Blocks.Tables.CombiTable1Ds"` 或其他替换字段！包装模型及其内部的 `Modelica.Blocks.Sources.CombiTimeTable` 由 TRICYS 底层自动构建。
```yaml
task:
  type: "cosim"
  mode: "offline_table"
  subsystems:
    - target: "example_model.I_ISS"                     # 待拦截的原子系统组件路径
      instance: "i_iss"                                 # 系统中的实例名
      solver: "aspen_plus"                              # 外部求解器类型
      handler_module: "tricys.handlers.i_iss_handler"   # Handler 函数所在 Python 模块
      handler_function: "run_aspen_simulation"         # Handler 模块函数名
      params:
        bkp_path: "../example_aspenbkp/T2-Threetowers4.bkp"
```

#### 模式 2: 在线步进模式 (`online_step`)
- **适用场景**：全厂模型（主系统）与外部 FMU / 子系统在每一个宏观通信步长（`stepSize`）执行动态双向交换。
- 💡 **规范要求**：主模型通常使用打包好的 `.ssp` 文件（如 `example_model.ssp`）；支持可选的 `local_mass_control`（局部质量守恒控制）和全局 `auditor` 插件。
```yaml
model:
  fileName: "../example_model_ssp/ssp/example_model.ssp"
  modelName: "example_model.Cycle"
scenario:
  stopTime: 100.0
  stepSize: 0.5                                         # 宏观数据交换通信步长
task:
  type: "cosim"
  mode: "online_step"
  subsystems:
    - target: "CFEDR.I_ISS"
      instance: "i_iss"
      solver: "aspen_plus"
      handler_module: "tricys.handlers.online_iss"
      handler_function: "OnlineISSProcessor"            # 必须指定继承自 AbstractTrackProcessor 的类名
      params:
        base_inventory_mol: 20.0
      local_mass_control:                               # 可选局部质量修正
        enabled: true
        deadband_g: 0.1
        relaxation_time_h: 1.0
```

#### 模式 3: 参数注入模式 (`parameter_injection`)
- **适用场景**：深冷精馏等微观高非线性子系统由外部专业求解器（如 `h2iso`）严密求解，全厂系统保留 0-D 动态代理，实现千小时级极速动态仿真。
- 💡 **规范要求**：
  1. **任务同级统一原则**：`injection_type: "handler" | "file"`，`override_file` 与 `handler_script`、`handler_params` 位于 task 全局同级，严禁分散在各个 subsystem 中重复声明；
  2. **显式替换与拓扑强校验**：**必须**在 `subsystems` 中指定 `substitute_model` 和 `substitute_file`（如 0-D 适配器模型），TRICYS 会自动校验端口名、因果方向、向量维度，确保绝对无悬空无不匹配。
```yaml
task:
  type: "cosim"
  mode: "parameter_injection"
  injection_type: "handler"                                # 统一注入方式: "handler" 或 "file"
  override_file: "./override_cycle.txt"                    # 覆盖参数目标文件
  handler_script: "./run_h2iso_wang2022.py"               # 任务级外部求解器 Handler 脚本
  handler_params:
    solve_mode: "live"                                    # 真实物理求解模式: "live"
    method: "wegstein"                                    # 撕裂流加速迭代方法
    max_iter: 10
    tol: 0.01
  subsystems:
    - target: "example_model.I_ISS"                       # 待替换的目标子组件
      instance: "i_iss"                                   # 实例名
      solver: "h2iso"
      substitute_model: "Generic_ISS_Adapters.ISS_I_Adapter" # 替换的 0-D 动态适配器模型
      substitute_file: "./Generic_ISS_Adapters.mo"        # 适配器 Modelica 文件
      aux_files:
        - "./Generic_ISS.mo"                              # 依赖的底层物理库
```

---

### 4.3 三大联合仿真范式的 Handler 编写契约与代码规范

针对三种模式不同的时空尺度解耦机制，**Handler 编写的软件形态、签名接口与数据流载体完全不同**：

#### 范式一：离线查表函数范式 (`offline_table`)
- **载体形态**：标准 Python 模块函数（Module Function）；
- **配置字段**：`subsystems[].handler_module` 与 `subsystems[].handler_function`；
- **接口签名**：`def handler_func(temp_input_csv: str, temp_output_csv: str, **params) -> dict`；
- **数据流与职责**：从 `temp_input_csv` 读取初探仿真全时序输入流股，驱动外部计算并生成 `temp_output_csv` 查表数据，返回常数占位字典，由 TRICYS 转换为 `CombiTimeTable` 代理执行第二阶段闭环重算。
```python
def run_aspen_simulation(temp_input_csv: str, temp_output_csv: str, **params) -> dict:
    df_in = pd.read_csv(temp_input_csv)
    # 外部计算或插值...
    df_out = pd.DataFrame({"time": df_in["time"], "to_SDS": ..., "to_WDS": ...})
    df_out.to_csv(temp_output_csv, index=False)
    return {"status": "success"}
```

#### 范式二：在线步进状态机类范式 (`online_step`)
- **载体形态**：有状态处理器类（Stateful Processor Class）；
- **配置字段**：`subsystems[].handler_module` 与 `subsystems[].handler_function`（指定类名）；
- **接口签名**：继承 `tricys.online_cosim.processor_base.AbstractTrackProcessor`，实现：
  - `initialize(self, context: TrackProcessorContext) -> None`
  - `process(self, request_vector: UnifiedStateVector) -> TrackResult`
- **数据流与职责**：在宏步长（`stepSize`）边界上锁步推进，从 `request_vector.boundary_inputs` 获取当前步输入流股列表，计算单步物理响应，返回强类型 `TrackResult(outputs={"to_SDS": [...], ...})`。
```python
from tricys.online_cosim.processor_base import AbstractTrackProcessor
from tricys.online_cosim.schema import TrackProcessorContext, TrackResult, UnifiedStateVector

class OnlineISSProcessor(AbstractTrackProcessor):
    def initialize(self, context: TrackProcessorContext) -> None:
        self.base_inv = context.params.get("base_inventory_mol", 20.0)

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        # request_vector.time, request_vector.boundary_inputs
        return TrackResult(outputs={"to_SDS": [0.01, 0.05, 0.94], "to_WDS": [0.98, 0.02, 0.0]})
```

#### 范式三：参数注入独立脚本范式 (`parameter_injection`)
- **载体形态**：任务级独立可执行脚本（Standalone CLI Executable）；
- **配置字段**：`task.injection_type: "handler"`, `task.handler_script`, `task.handler_params`, `task.override_file`；
- **接口契约**：标准 Python 独立脚本入口 `def main() -> int`，从环境变量 `TRICYS_WORKSPACE` 与 `TRICYS_HANDLER_PARAMS` 获取运行时上下文；
- **数据流与职责**：前置真实调用专业物理求解器（如 `h2iso` 严密精馏流程求解器），输出包含 Modelica 实例变量路径的 `override_cycle.txt` 写入工作区，返回退出码 `0`。TRICYS 强校验端口后无缝替换为 0-D 动态代理，OMC 携带 `-overrideFile` 参数秒级跑完长周期（5000h+）全厂演化。
```python
def main() -> int:
    workspace = os.environ.get("TRICYS_WORKSPACE", ".")
    params = json.loads(os.environ.get("TRICYS_HANDLER_PARAMS", "{}"))
    # 调用专业外部求解器计算 (如 h2iso flowsheet/mesh) ...
    override_path = os.path.join(workspace, "override_cycle.txt")
    with open(override_path, "w", encoding="utf-8") as f:
        f.write("i_iss.split_ratio_SDS=0.85\ni_iss.split_ratio_WDS=0.15\n")
    return 0
```

> 📖 **完整联合仿真工程指南与三范式 Handler 生产级代码**：详见专论 [cosimulation_framework_guide.md](references/cosimulation_framework_guide.md)。  
> 📋 **联合仿真 YAML 原型模板**：详见 [08_cosim_subsystem_template.yaml](templates/08_cosim_subsystem_template.yaml)、[09_cosim_online_step_template.yaml](templates/09_cosim_online_step_template.yaml) 及 [10_cosim_parameter_injection_template.yaml](templates/10_cosim_parameter_injection_template.yaml)。

---

## 🗂️ 5. 十大核心应用场景与模型映射表

| 场景编号 | 聚变工程应用场景 | 对应模板路径 | 核心依赖的自省变量 | 典型任务类型 |
| :--- | :--- | :--- | :--- | :---: |
| **SC-1** | **起动亏空扫描与自持平衡评估** | `templates/01_sweep_startup_template.yaml` | 储罐存量 `${STORAGE_NODE}`<br>增殖参数 `${TBR_PARAM}` | `sweep` |
| **SC-2** | **长脉冲/周期性脉冲与二分 TBR 寻优** | `templates/02_simulate_pulsed_bisection_template.yaml`<br>`templates/05_simulate_foc_template.yaml` | 脉冲源组件 `${PULSE_COMPONENT}`<br>储罐存量 `${STORAGE_NODE}` | `simulate` (二分寻优) |
| **SC-3** | **多维参数 Sobol 全局敏感度分解** | `templates/03_sensitivity_sobol_template.yaml` | 多个物理参数列表<br>目标指标存量节点 | `sensitivity` (sobol) |
| **SC-4** | **全厂闭环质量平衡与泄漏安全审计** | `templates/06_mass_balance_audit_template.yaml` | 全部存量 `patterns.inventory`<br>产氚/燃耗/衰变/泄漏流股 | `simulate` (auditor) |
| **SC-5** | **自主扩展插件流水线与定制化分析** | `templates/07_custom_plugin_pipeline_template.yaml` | 任意自定义指标节点<br>外部插件脚本路径 | `pipeline` (自定义扩展) |
| **SC-6** | **多物理场离线查表协同仿真** | `templates/08_cosim_subsystem_template.yaml` | 目标子组件与静态数据表路径 | `cosim` (offline_table) |
| **SC-7** | **跨工具在线步进动态协同仿真** | `templates/09_cosim_online_step_template.yaml` | 目标子组件与在线处理器模块 | `cosim` (online_step) |
| **SC-8** | **基于外部求解器严密计算的参数注入** | `templates/10_cosim_parameter_injection_template.yaml` | 目标子组件、外部求解脚本与覆盖文件 | `cosim` (parameter_injection) |
| **SC-9** | **双层嵌套二分优化 (装填量与 TBR 协同寻优)** | `templates/11_optimize_nested_bisection_template.yaml` | 初始装填量 `${STORAGE_STARTUP_PARAM}`<br>增殖参数 `${TBR_PARAM}` | `optimize` (nested_bisection) |
| **SC-10** | **参数扫描内嵌双层优化 (工况族寻优响应)** | `templates/12_sweep_nested_optimization_template.yaml` | 扫描物理变量列表<br>双层寻优参数与存量节点 | `sweep` (内嵌 `optimization`) |

---

## 🛠️ 6. 机器可读 CLI 交互协议速查

| 指令功能 | 完整命令行命令 | 核心输出结构 / 标志 |
| :--- | :--- | :--- |
| **模型静态审查** | `python -m tricys.main model review <mo> <model> --json` | 标准 JSON：`status`, `issues`, `remediation_patch` |
| **模型深度自省** | `python -m tricys.main model inspect <mo> <model> --json` | 标准 JSON：`parameters`, `inventories`, `flow_rates` |
| **配置语法校验** | `python -m tricys.main validate -c <config.yaml>` | `Config is valid ✅` 或详细错误堆栈 |
| **DSL 编译降级** | `python -m tricys.main compile -c <config.yaml> [-o <pipe.yaml>]` | 降级输出纯显式 DAG Pipeline YAML/JSON |
| **生成插件脚手架**| `python -m tricys.main plugin scaffold <name> -o <file.py>` | 自动生成包含物理总线与强类型契约的高质量代码模板 |
| **插件内存 Mock 测试**| `python -m tricys.main plugin test <file.py> --json` | 纯内存沙箱执行，毫秒级输出验证报告与捕获指标 |
| **查询已注册插件**| `python -m tricys.main plugin list --json` | 标准 JSON：`preprocess`, `engine`, `postprocess` 清单 |
| **查询单插件详情**| `python -m tricys.main plugin info <name> --json` | 标准 JSON：参数 Schema、输入输出契约 |
| **带诊断屏障运行**| `python -m tricys.main run -c <config.yaml> --diagnose` | 发生错误时写入 `results/diagnostics.json` |

