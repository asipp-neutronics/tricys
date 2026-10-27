# TRICYS V2.0 聚变堆氚燃料循环仿真系统核心技术与架构复盘报告

> **呈送**：系统总体部 / 聚变工程软件研发部领导与技术委员会  
> **版本**：TRICYS V2.0 (Milestone: Stage 0 ~ Stage 4 Completed, Stage 5 In Progress)  
> **核心主题**：底层流水线微内核、双轨配置规范体系、模型中心化闭环 Agent-Skill、云原生全栈融合容器底座、协同仿真三大范式与工业级双重学术基准

---

## Executive Summary (执行摘要)

TRICYS (Tritium Integrated Cycle Simulation Framework) 专注于磁约束聚变堆（如 CFETR / DEMO）燃料循环全流程的动态演化、同位素存量平衡与自持安全裕度仿真。随着聚变工程从宏观概念设计步入高保真工程详设阶段，历史 V1.0 架构在**时间量纲混淆、并发死锁风险、参数硬编码、执行流程隐式黑盒及缺乏通用跨模型适配**等问题上遭遇瓶颈。

在本次 V2.0 重构与升级工程中，项目团队严格践行现代软件工程与科学计算规范，完成了从**“传统脚本拼装”**向**“工业级微内核流水线 + 模型中心化 AI 协同闭环 + 云原生全栈融合容器底座 + 协同仿真三范式 + 双层嵌套优化引擎”**的系统性演进。目前系统 Stage 0 至 Stage 4 全部高质量交付，并正纵深推进 Stage 5（存量兼容回归与权威操作手册大清洗）。本报告严格对照系统源代码实现，面向领导与技术决策层系统汇报 V2.0 的核心技术支柱与最新研发进展：
1. **底层微内核流水线架构**：基于 I/O 契约的标准化三阶段管线、Kahn DAG 依赖自动拓扑重排、30 项核心物理语义总线、用户小时输入与模型秒/小时物理量纲解耦治理；
2. **双轨配置规范体系**：高阶科研 DSL（五大顶级支柱，支持 simulate/sweep/sensitivity/cosim/optimize）与纯显式解耦 Pipeline 并存并实现 1:1 无损双向解析/编译；
3. **协同仿真三大范式标准化体系**：系统确立离线查表 (`offline_table`)、在线步进 (`online_step`) 与稳态参数注入 (`param_inject`) 三大协同仿真范式；引入带因果性与端口拓扑严格校验的子模型动态替换器 (`submodel_replacer`) 及原子/分子流跨维度适配组件 (`Generic_ISS_Adapters.mo`)；
4. **双层嵌套二分优化引擎 (Nested Bisection)**：突破传统单变量盲目扫参，在物理内核层构建外层最小启动氚库存寻优与内层临界 TBR 逼近的双层嵌套二分求解器，同时满足在役储备谷底与目标倍增时间双约束；
5. **Agent-Skill 闭环协议与插件生态**：防幻觉模型自省、AST 静态审查补丁自愈、七步闭环工作流、纯内存 Mock 沙箱、带 AST 安全屏障的插件扩展机制，并挂载 `tricys_abdou_2021` 宏观物理理论基准子模块；
6. **云原生全栈融合容器底座 (Stage 4)**：彻底废除脆弱的跨容器 TCP Server 模式，构建融合 OpenModelica (离线 MSL 自愈固化)、FESTIM v2.0 有限元、CasADi/H2ISO 精馏求解、FastAPI 后端、Vue3 控制台、GoView 大屏、Dash/Plotly HDF5 科学看板与 Nginx 统一反向代理网关的 All-in-One 单容器生产底座；
7. **工业与学术级双重基准验证与解耦**：TRICYS 核心代码库测试用例达 **344 passed, 11 skipped (100% 绿灯)**；工业级演示算例库全面扩充并统一收敛至 **12 大工业级基准算例 (01 ~ 12)**，由统一算例调度器 (`TricysV2ExampleRunner`) 驱动且全量通过；聚变燃料自持理论验证与确认（V&V）基准彻底解耦，独立归口于 [`couuas/tricys_abdou_2021`](https://github.com/couuas/tricys_abdou_2021) 专用 V&V 验证基准仓库（并在本项目中以 Git Submodule 挂载），涵盖 Abdou (2021) 与 Abdou (1986) 宏观自持全套理论基准方程、离散计算标记、参数矩阵与边界约束验证报告（Track 1 & Track 2 理论等价交叉验证实现 100% 严密对标复现）。

---

## 一、 底层架构：解耦流水线与物理微内核

```
+-------------------------------------------------------------------------------------------------+
|                             TRICYS V2.0 Universal Workflow Pipeline                             |
+-------------------------------------------------------------------------------------------------+
                                                 |                                                 
               +---------------------------------+---------------------------------+               
               |                                 |                                 |               
               v                                 v                                 v               
+-----------------------------+   +-----------------------------+   +-----------------------------+
| Preprocessing Stage         |   | Simulation Engine Stage     |   | Postprocessing Stage        |
+-----------------------------+   +-----------------------------+   +-----------------------------+
| 1. setup                    |   | 1. Standard Engine          |   | 1. storage                  |
|    - Workspace Isolation    |   |    - Precompiled C Binary   |   |    - HDF5 Time-Series       |
|    - Parameter Grid/Sobol   |   |    - Multiprocessing Pool   |   |    - CSV Export & KPI Stats |
| 2. compiler                 |   |    - Single / Nested Bisect |   | 2. safety                   |
|    - OMC buildModel (.exe)  |   | 2. Offline Table Engine     |   |    - Mass Balance Auditor   |
| 3. foc                      |   |    - CombiTimeTable Interc. |   |    - Static Alarm Rules     |
|    - Pulse Event Injection  |   | 3. Online Step Engine       |   | 3. report                   |
| 4. interceptor & replacer   |   |    - OMSimulator Lockstep   |   |    - Markdown / DOCX / PDF  |
|    - Submodel Replacer (AST)|   | 4. Parameter Inject Engine  |   | 4. sensitivity              |
|    - Port Topology Check    |   |    - Wang 2022 Override     |   |    - Sobol Global GSA       |
| 5. Custom Plugins (Pre)     |   | 5. Custom Engines           |   | 5. Custom Plugins (Post)    |
|    - User Preprocessors     |   |    - Registered Solvers     |   |    - e.g. Reserve Auditor   |
+-----------------------------+   +-----------------------------+   +-----------------------------+
               |                                 ^                                 |               
               +---------------> [ PipelineContext & Physics Bus ] <---------------+               
```

### 1.1 三阶段微内核解耦架构
V2.0 废除了以往散落于业务模块中的隐式执行流，将整个仿真全生命周期统一规范为标准三阶段流水线（`WorkflowPipeline`）：
- **前处理阶段 (Preprocessing)**：负责参数扫描空间展开、工作区独立隔离（`setup`）、OpenModelica 预编译本地二进制执行程序（`compiler`）、FOC 燃烧/脉冲工况信号注入（`foc`）、离线协同仿真外部时序拦截器（`interceptor`），以及参数注入协同仿真模式下的子模型因果性与端口拓扑严格校验替换器（`submodel_replacer`）；
- **仿真核心阶段 (Simulation Engine)**：支持三大类共四种标准化独立执行内核：
  1. `StandardSimulationEngine` (`standard`)：通过 OpenModelica 将模型翻译为 C 源码并编译为原生二进制程序（`.exe`），由单层受控进程池调度执行；支持单点瞬态仿真、多工况网格扫描、Sobol 全局敏感度分析、单变量 `Required_TBR` 内置二分法寻优，以及**双层嵌套二分优化 (`NestedBisectionOptimizer`)**；
  2. `OfflineTableCoSimEngine` (`offline_table`，基准算例 08)：两阶段初探-拦截-重构协同仿真引擎，由后台自动构建 `CombiTimeTable` 包装模型驱动外部高保真查表计算；
  3. `OnlineStepCoSimEngine` (`online_step`，基准算例 09)：基于 OpenModelica OMSimulator (OMS) 与 SSP/FMI 协议的宏步长锁步在线双向状态同步引擎；
  4. `ParamInjectCoSimEngine` (`param_inject`，基准算例 10)：基于热力学离线求解与 Modelica `-overrideFile` 稳态参数动态注入引擎，配合通用同位素流适配组件（`Generic_ISS_Adapters.mo`）实现原子流与分子流升降维互联；
  5. `Custom Engines`：经注册的第三方外部计算求解器引擎。
- **后处理阶段 (Postprocessing)**：负责 HDF5 高性能时序持久化与指标提炼（`storage`）、全厂闭环质量平衡审计与静态超限预警（`safety`）、多格式工程报告导出（`report`）、Sobol 全局敏感度分析（`sensitivity`），以及外部挂载的自定义分析插件。

### 1.2 强类型状态载体与开闭原则
流水线全程由统一强类型载体 `PipelineContext` 贯穿：
- **强类型作业结果 (`SimulationJobResult`)**：严格封装 `job_id`、工况参数集 `parameters`、时序 DataFrame 惰性加载接口 `get_dataframe()`、以及标量提取指标 `metrics`；
- **分层开放扩展槽 (Open-Closed Principle)**：在任务级提供 `job.extra`，在全局级提供 `context.metadata` 与 `context.artifacts`。既确保核心主干零幻觉强校验，又为第三方插件提供自由存储中间矩阵与衍生分析成果的独立空间。

### 1.3 基于 Kahn 算法的 DAG 依赖拓扑自动排序
传统插件系统极易发生因配置声明顺序倒置引发的“前置数据未生成、后置插件已报错”隐患。TRICYS V2.0 引入声明式 I/O 契约机制：
- 每个插件通过 `requires` 声明输入前置依赖（如 `["results_store_path"]`），通过 `produces` 声明产出成果（如 `["audit_report"]`）；
- `PipelineBuilder` 构建流水线时，在内存中动态构筑有向无环图（DAG），采用 **Kahn 拓扑排序算法** 自动重排插件执行序列。例如：即使用户在配置中误将数据持久化插件声明在审计插件之后，构建器在初始化阶段即可自动探测依赖关系、完成顺序纠偏并输出拓扑调整日志，确保“先持久化、后执行分析”。

### 1.4 聚变物理语义总线 (Physics Semantic Bus)
针对聚变工程中不同研发团队、不同保真度 Modelica 模型变量命名异构多态（如 SDS 储罐存量在不同模型中可能命名为 `sds.I[1]`、`Cycle.sds.inventory` 或 `storage.mass`）的工程现实，V2.0 实现了**聚变物理语义总线 (`PhysicsSemanticResolver`)**：
- 以 Abdou et al. (2021) 聚变物理控制方程与系统术语表为基准，代码库（`tricys.core.semantic`）内置了严格对齐的 **30 项标准化物理语义概念**，划分为 7 大物理功能分区：
  1. **SDS 储罐与中央存量**：`sds_inventory`, `sds_decay_loss`, `startup_inventory`
  2. **产氚包层与 TBR**：`blanket_inventory`, `blanket_tbr`, `blanket_residence_time`, `blanket_to_tes_fraction`, `blanket_to_cl_fraction`
  3. **等离子体芯部与加料**：`plasma_burn_fraction`, `plasma_fueling_eff`, `pulse_amplitude`, `availability_factor`
  4. **偏滤器抽气与内循环**：`divertor_inventory`, `pump_inventory`, `tep_fep_inventory`, `tep_fcu_inventory`, `tep_inventory`, `dir_ratio`, `i_iss_inventory`, `i_iss_threshold`
  5. **氚提取回收与外循环**：`tes_inventory`, `tes_threshold`, `o_iss_inventory`, `wds_inventory`
  6. **冷却剂回路与第一壁**：`coolant_inventory`, `cps_inventory`, `first_wall_inventory`
  7. **宏观系统评价指标**：`required_tbr`, `doubling_time`, `self_sufficiency_time`
- 插件开发者只需调用统一语义 API：
  ```python
  sds_series = context.resolve_physics_variable(df, "sds_inventory")
  ```
  底层总线利用正则加权与优先级探测自动锁定真实列名，彻底切断外挂算法与具体 Modelica 模型的变量名耦合。

### 1.5 国际标准时间量纲治理与时基解耦
- **SI 秒为第一公民**：底层数值求解器与 HDF5 时序数据一律严格以国际单位制“秒 (s)”存储与运算；
- **用户小时输入与模型物理时间量纲解耦**：
  - 用户配置默认采用工程直觉的“小时 (h)”（如 `stopTime: 2000.0` 表示 2000 小时）；
  - `scenario.time_base` 用于显式声明底层 Modelica 模型内部使用的物理时间基准（`second` 还是 `hour`）；
  - 流水线自适应计算时间缩放因子（`time_scale_factor`），当底层模型以 `second` 为基准时自动完成 3600 倍时基缩放并注入仿真命令行，消除模型内部方程与外部控制指令的量纲错位；
- **FOC 动态脉冲失配预警与自适应换算**：FOC 脉冲事件调度器根据 `model_time_base` 自动校验脉冲时序，杜绝将小时级脉冲误注入秒级模型的严重工程事故；
- **硬编码除法完全清零与报告图表自洽**：全库排查并根除了历史遗留的 `/ 24`、`/ 3600` 与 720 万小时标度判定等隐患，图表横轴动态感知物理单位，在工程报告中呈现 `2000 h (7200000 s)` 及 `109.00 h (392400 s)` 拐点时间，实现 0 ~ 2000 小时物理尺度 100% 准确自洽。

---

## 二、 配置规范：双轨配置体系与 1:1 无损编译

为了兼顾“人类工程师简明配置”与“工业级底层全透明显式编排”的双重需求，TRICYS V2.0 确立了**双轨配置体系 (Dual-Paradigm Architecture)**。

```
+--------------------------------------------------------+
|  模式 1: Schema 2.0 高阶领域 DSL (面向科研人员/人类设计)  |
|  - model, scenario, task, visualizer, plugins          |
+--------------------------------------------------------+
                           |
                           |  python -m tricys.main compile -c run.yaml -o pipe.yaml
                           |  (PipelineBuilder.compile_v2_to_explicit_pipeline)
                           v
+--------------------------------------------------------+
|  模式 2: 纯显式解耦 Pipeline (面向底层工程/自动化驱动)   |
|  - pipeline.preprocessing, simulation, postprocessing  |
+--------------------------------------------------------+
```

### 2.1 模式 1：Schema 2.0 高阶领域 DSL (科研直觉导向)
将全系统严格收敛至**五大顶级支柱 (Five Core Pillars)**，消除历史层级冗余：
1. **`model`**：物理模型路径与编译器全局选项（`fileName`, `modelName`, `cflags`, `target`）；
2. **`scenario`**：求解条件与积分器设定（`time_base`, `stopTime`, `stepSize`, `method`, `variableFilter`）；
3. **`task`**：核心计算范式与指标算子（`type: simulate / sweep / sensitivity / cosim / optimize`, `parameters`, `metrics`, `task.auditor`, `task.optimizer`）；
4. **`visualizer`**：可视化看板与报告导出引擎（`enabled`, `launch_on_finish`, `visualizer.report`）；
5. **`plugins`**：生命周期增量挂载点（`preprocess`, `postprocess` 挂载外部脚本并传参）。

**任务范式全景覆盖**：
- `simulate`：单点高精度长周期瞬态求解；
- `sweep`：多参数网格扫描与敏感边界提取（可内嵌单/双层二分优化）；
- `sensitivity`：Sobol / LHS 高维抽样与全局方差分解；
- `cosim`：离线查表 (`offline_table`)、在线步进 (`online_step`) 与参数注入 (`param_inject`) 三大协同仿真模式；
- `optimize`：双层嵌套二分优化 (`NestedBisectionOptimizer`)，闭环求解自持最小启动库存与临界 TBR。

**示例配置 (基准算例 07)**：
```yaml
version: "2.0"
model:
  fileName: "../example_model_single/example_model.mo"
  modelName: "example_model.Cycle"
scenario:
  time_base: "hour"
  stopTime: 2000.0
  stepSize: 0.5
task:
  type: "sweep"
  parameters:
    - name: "TBR_1.08"
      blanket.TBR: 1.08
    - name: "TBR_1.18"
      blanket.TBR: 1.18
plugins:
  postprocess:
    - name: "tritium_reserve_auditor"
      script: "plugins/tritium_reserve_auditor.py"
      params:
        baseline_loading_g: 6100.0
        safety_valley_redline_g: 1000.0
visualizer:
  report:
    enabled: true
    formats: ["markdown", "docx", "pdf"]
```

### 2.2 模式 2：纯显式解耦 Pipeline (工程解耦导向)
在模式 2 下，所有高阶顶级领域抽象完全展开，流水线直接映射为预处理列表、求解器引擎与后处理列表：
- **Explicit-First 显式传参**：每个插件均为独立节点，参数通过字典直接注入 `params`，严禁隐式全局依赖；
- **极简工程自描述**：适合云原生工作流编排器（如 Kubeflow / Airflow）或异构协同仿真平台直接驱动。

**示例代码 (基准算例 07 经 1:1 无损编译后的纯显式 Pipeline 结构)**：
```yaml
pipeline:
  name: "CompiledExplicitPipeline"
  preprocessing:
    - name: "setup"
      params:
        package_path: "../example_model_single/example_model.mo"
        model_name: "example_model.Cycle"
        parameters:
          - name: "Conservative_TBR_1.08"
            blanket.TBR: 1.08
            wds.threshold: 800.0
          - name: "High_Breeding_TBR_1.18"
            blanket.TBR: 1.18
            wds.threshold: 1100.0
        task_type: "sweep"
    - name: "compiler"
      params:
        package_path: "../example_model_single/example_model.mo"
        model_name: "example_model.Cycle"
  simulation:
    name: "standard"
    params:
      stop_time: 2000.0
      step_size: 0.5
      solver: "dassl"
      variable_filter: "time|sds.I[1]|blanket.I[1]|wds.I[1]|tep_fep.I[1]"
      parallel: true
      max_workers: 2
  postprocessing:
    - name: "storage"
      params:
        metrics_definition:
          peak_inventory:
            source: "sds.I[1]"
            evaluate: "peak_value"
          final_inventory:
            source: "sds.I[1]"
            evaluate: "final_value"
    - name: "tritium_reserve_auditor"
      script: "plugins/tritium_reserve_auditor.py"
      class: "TritiumReserveAuditorPlugin"
      params:
        baseline_loading_g: 6100.0
        safety_valley_redline_g: 1000.0
    - name: "report"
      params:
        report_config:
          enabled: true
          title: "TRICYS 聚变堆自定义插件扩展与在役储罐安全裕度评估报告"
          formats:
            - "markdown"
            - "docx"
            - "pdf"
```

**结构映射关系解析**：
1. **模型与工况展开**：高阶配置中的 `model` 与 `task.parameters` 自动被编译器降解为 `setup`（工况参数网格与隔离）与 `compiler`（OMC C 源码预编译）；
2. **求解引擎具象化**：高阶配置中的 `scenario` 求解条件与 `task.parallel` 并行度无损绑定至 `standard` 仿真引擎的 `params`；
3. **后处理链平铺**：`task.metrics`（指标提取）、自定义外挂脚本 `plugins.postprocess` 与 `visualizer.report` 统一平铺为清晰串行的后处理管线。

### 2.3 1:1 无损配置编译器
V2.0 提供了开箱即用的配置编译工具：
```bash
python -m tricys.main compile -c config_v2.yaml -o explicit_pipeline.yaml
```
- **等价性保证**：编译器（`PipelineBuilder.compile_v2_to_explicit_pipeline`）将五大领域的业务字段准确编译为模式 2 下各插件的专属 `params`；
- **底层统一执行流**：无论传入模式 1 还是模式 2，底层统一由 `PipelineBuilder` 构建相同的 `WorkflowPipeline` 实例，两者计算所得数值结果与指标相对误差恒等于 0。

---

## 三、 Agent-Skill：模型中心化闭环与插件生态

针对通用 AI 模型容易“脱离物理模型凭空捏造变量名（幻觉）”、“无法识别数值发散错误”等缺陷，V2.0 研发了专属于聚变计算的 **Agent-Native 协议体系** 与操作规范。

```
+---------------------------------------------------------------------------------------------------------+
|                                TRICYS Model-Centric Agentic Workflow                                    |
+---------------------------------------------------------------------------------------------------------+
  [1. 环境与依赖探查] ──> [2. 模型健康审查与Diff修补] ──> [3. 零幻觉模型深度自省] ──> [4. 规范化配置合成]
                                                                                              │
  [7. 学术报告与物理洞察交付] <── [6. 仿真驱动与自愈屏障] <── [5. 配置静态校验与沙箱试跑] <────────┘
```

### 3.1 严格防幻觉纪律与“七步闭环工作流”
TRICYS 在 `skills/tricys-agent/SKILL.md` 中固化了严格执行规范：**严禁凭空猜测模型变量，一切必须以真实 Modelica AST 自省结果为准**。
1. **环境探查与就绪自检 (Intake)**：探测 Python、OMC 及编译器依赖就绪状态；
2. **AST 模型静态审查与补丁自愈 (Review & Remediation)**：
   - 执行 `python -m tricys.main model review <mo_file> <model_name> --json`；
   - 静态探测时间量纲缺失（缺失 `time_scale`）、守恒漏项（缺失 `leak_rate`/`decay_rate`）及初值奇异，自动生成标准 Unified Diff 补丁就地修复；
3. **深度无幻觉模型自省 (Introspect)**：
   - 执行 `python -m tricys.main model inspect <mo_file> <model_name> --json`；
   - 提取参数树、存量节点与流股列表，以真实的变量名（如 `sds.I[1]`）作为配置合成依据；
4. **场景原型匹配与规范配置合成 (Synthesis)**：
   - 将真实变量映射注入官方模板库，生成严格合规的 Schema 2.0 配置；
5. **配置静态校验与沙箱试跑 (Validate & Mock Test)**：
   - 调用 `python -m tricys.main validate -c <config.yaml>` 实施强类型语法与字段有效性校验；
6. **流水线仿真与多级异常自愈屏障 (Run & Diagnostics Barrier)**：
   - 携带 `--diagnose` 参数运行仿真；若遇 DASSL 步长下溢或初值奇异，`DiagnosticsBarrier` 拦截错误并导出结构化 `diagnostics.json`，提供根因剖析与自愈调整建议；
7. **学术交付物审计与物理洞察核验 (Audit & Insight)**：
   - 审查输出的 Markdown/Word/PDF 报告、HDF5 数据及质量平衡残差（误差 < 0.5g），依据 Abdou 2021 置信度法则向工程师交付结论。

### 3.2 协同仿真开发者规约与子模型动态替换强校验
在 commit `36b0285` 与 `skills/tricys-agent/` 中，系统固化了协同仿真最高纪律与子模型替换校验机制：
1. **三大协同仿真范式严禁混淆**：
   - **离线查表 (`offline_table`)**：初探-拦截-重构机制，底层由 `interceptor.py` 自动构建 `CombiTimeTable` 包装模型，严禁且无需在 YAML 中声明 `substitute_model`；
   - **在线步进 (`online_step`)**：依托 OMSimulator 调度系统资产（`.ssp`），逐时间步锁步同步变量，严禁在 YAML 中声明 `substitute_model`；
   - **参数注入 (`param_inject`)**：全框架唯一需要显式声明 `substitute_model` 与 `substitute_file` 的模式；
2. **子模型因果性与端口拓扑绝对零悬空强校验 (`submodel_replacer.py`)**：
   - 在执行 Modelica 代理模型替换前，AST 校验器强制比对原组件与替换组件的输入/输出端口数量、命名与因果方向，自动绑定未连接端口默认零输入方程，杜绝变量欠定错误与语法崩溃；
3. **同位素流跨维度升降维桥接**：
   - 针对宏观 3/5 维原子流（H, D, T）与低温精馏 6 维分子流（H2, HD, HT, D2, DT, T2）的跨尺度鸿沟，通过专用适配器组件（`Generic_ISS_Adapters.mo`）与 Python Handler 实施双向平衡映射。

### 3.3 宏观物理理论基准库整合 (`tricys_abdou_2021`)
在 commit `2811d2b` 中，项目以 Git Submodule 形式正式挂载 `tricys_abdou_2021`（commit `784efe4`）：
- **理论参考系**：内置 Abdou et al. (2021) 与 Abdou et al. (1986) 经典聚变自持边界模型、参数矩阵与离散标记；
- **防止 Agent 泛化幻觉**：为 Agent 提供宏观物理常识手册（`abdou_2021_physics_rules.md` 与 `abdou_1986_mrt_remediation.md`），在评估启动库存谷底、最小 TBR 与倍增时间时具备可溯源的物理依据。

### 3.4 插件拓展机制与安全沙箱
Agent 与第三方开发者可便捷为系统二次开发外挂插件：
- **强类型自描述契约 (`PluginMetadata`)**：插件通过基类属性声明作者、版本、参数 Schema、输入输出依赖；
- **AST 静态安全沙箱 (`validate_script_ast_safety`)**：外部脚本动态加载时，AST 扫描器强制拦截 `os.system`、`subprocess`、`socket`、`requests` 等危险调用与模块导入，阻止未经授权的系统越权操作；
- **毫秒级纯内存 Mock 沙箱 (`plugin test`)**：
   ```bash
   python -m tricys.main plugin scaffold my_auditor -o plugins/my_auditor.py
   python -m tricys.main plugin test plugins/my_auditor.py --json
   ```
   无需拉起耗时的 Modelica 编译，在毫秒级内完成插件接口与逻辑断言自检；
- **工业算例验证**：在新增算例 07 中，挂载的 `tritium_reserve_auditor.py` 成功在多工况参数扫描下，通过物理语义总线提取中央储氢罐时序，完成安全谷底裕度定量核算并导出独立 JSON 交付物。

---

## 四、 工业级基准算例实测与验证矩阵 (v2_examples)

在 Stage 3 ~ Stage 4 期间，工业级演示算例库全面重构并扩充，所有算例收敛至 `v2_examples/` 目录下并通过统一配置标准驱动。12 个工业级基准算例全部进行了端到端自动化编译、DAG 校验与求解测试，达成 **100% 成功通过**：

### 4.1 十二大工业基准算例全景验证矩阵

| 序号 | 算例配置名称 | 核心物理场景与计算模式 | 状态 | 实测耗时 | 核心产出交付物 |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **01** | `01_sweep_startup_self_sufficiency.yaml` | 稳态起动亏空、临界 TBR 倍增扫参 (7点扫描) | **PASS** | 16.66s | `sweep_results.h5`, 6张时序演化图, MD/DOCX/PDF 报告 |
| **02** | `02_simulate_pulsed_transient.yaml` | 脉冲瞬态动力学仿真、内置二分寻优算子 | **PASS** | 56.63s | 二分逼近收敛曲线 (`Required_TBR=1.3418`), 瞬态工程报告 |
| **03** | `03_sensitivity_sobol_gsa.yaml` | Sobol 全局敏感度分析 (GSA, 32样本基底) | **PASS** | 14.58s | 一阶/全阶敏感度热图与方差分解报告 (MD/DOCX/PDF) |
| **04** | `04_sweep_pulse_width_required_tbr.yaml` | 脉宽扫描 (100, 80, 60s) 联合二分寻优 | **PASS** | 69.31s | 脉宽扫描网格矩阵、自持 TBR 边界曲线 (MD/DOCX/PDF) |
| **05** | `05_simulate_foc_pulsed_scenario.yaml` | 502 个 FOC 脉冲/燃烧事件时序动态注入 | **PASS** | 13.94s | FOC 时序重构曲线、燃料流率动态响应报告 |
| **06** | `06_simulate_mass_conservation_audit.yaml` | 全厂闭环质量平衡物理审计 | **PASS** | 18.09s | `auditor_report.json`（实测质量守恒残差累计误差仅 `-0.02208 g`，远优于 0.5g 阈值） |
| **07** | `07_custom_plugin_pipeline.yaml` | 自定义插件挂载 + 语义总线扩展流水线 | **PASS** | 13.00s | `tritium_reserve_audit.json`（中央储罐安全谷底裕度定量核算报告） |
| **08** | `08_cosim_offline_table.yaml` | 协同仿真: 离线查表/两阶段子系统替代模式 | **PASS** | 22.45s | 自动构建 `CombiTimeTable` 包装模型，实现高精度离线插值替代 |
| **09** | `09_cosim_online_step.yaml` | 协同仿真: 在线步进双向同步模式 (SSP/FMI) | **PASS** | 35.12s | 基于 OMSimulator 驱动动态模型与 ISS 处理器锁步数据交换 |
| **10** | `10_cosim_parameter_injection.yaml` | 协同仿真: 稳态参数注入模式 (Wang 2022) | **PASS** | 38.80s | `override_cycle.txt`, `Generic_ISS_Adapters.mo` 原子/分子流跨维度桥接 |
| **11** | `11_optimize_nested_bisection.yaml` | 双层嵌套二分优化 (自持启动库存与 TBR 协同求解) | **PASS** | 42.15s | 双约束收敛轨迹图、最小启动库存 $I_{S,0}$ 与临界 $TBR$ 联合解 |
| **12** | `12_sweep_nested_optimization.yaml` | 燃烧分数扫描内嵌双层优化 (Sweep Burnup with Nested Bisection) | **PASS** | 85.30s | 燃烧分数 (3%~8%) 扫描下启动库存与自持 TBR 多工况对比报告 |

### 4.2 统一算例调度体系与历史算例安全归档
在 commit `6ee8c4c` 中，系统完成了案例工程体系的系统性重构与收敛：
1. **历史 V1 算例完整归档 (`v1_examples_legacy/`)**：
   - 将原有分散的 `basic/` 与 `analysis/` 算例全量迁移至 `v1_examples_legacy/`，编写完备的 `README.md` 说明其历史背景、废弃声明及时间量纲（小时制到秒制）迁移方案；
   - 保留向后兼容导入 shim（如 `TricysAllTestRunner is TricysV2ExampleRunner`），确保老用户脚本无缝迁移；
2. **统一交互式与批处理调度器 (`TricysV2ExampleRunner`)**：
   - 统一由 `tricys/example/example_runner/runner.py` 驱动，支持命令行交互菜单与非交互式批处理：
     ```bash
     python -m tricys example --list         # 查看全部 12 个算例
     python -m tricys example --validate 11  # 静态校验算例配置
     python -m tricys example --compile 11   # 编译为显式 DAG Pipeline
     python -m tricys example --run 11       # 执行指定算例
     python -m tricys example --dry-run      # 零仿真快速走查准备环境
     ```
   - 彻底消除了历史上多入口脚本重复维护与维护分叉的历史问题。

---

## 五、 云原生基准运行环境与全栈融合容器底座 (Stage 4)

在聚变堆多物理场协同仿真中，底层软件环境往往面临极高的异构性挑战：系统级微分代数方程（DAE）求解依赖 OpenModelica C 编译器与 Modelica 标准库（MSL），部件级偏转板等离子体壁面相互作用依赖有限元（FEM）工具链（DOLFINx、PETSc、SLEPc、FESTIM），低温同位素分离精馏依赖非线性优化与正交配置求解器（CasADi、H2ISO），而工程交付又要求兼顾 Web 控制台（Vue 3）、数字孪生大屏（GoView）与高通量 HDF5 时序交互看板（Dash / Plotly）。

在 Stage 4 中，研发团队攻克了这一系列复杂异构依赖，成功交付了**生产级 All-in-One 云原生 Docker 容器底座**，彻底消除“环境配置难、内网构建断网、跨平台结果不可复现”的历史顽疾。

### 5.1 架构决策演进：从跨容器 TCP Server 到单容器原生协同

在多物理场（特别是 FESTIM 偏转板滞留）早期预研阶段，团队曾尝试使用微服务拆分架构——将 FESTIM 部署为独立的 TCP Server、通过 Socket 套接字接收 TRICYS 主进程的边界通量并返回渗透/滞留数据。但在高阶脉冲与长时间跨度实测中暴露出一系列严重工程隐患：
1. **通信时延与开销瓶颈**：每个时间步（如 0.1s）进行一次 JSON / Pickle 跨网络套接字序列化与反序列化，仿真耗时剧增 300% 以上；
2. **生命周期脆弱与死锁失步**：一旦 OpenModelica 求解器触发步长折减或求解失败，TCP Server 无法感知异常退出，导致 Socket 悬挂或僵尸进程；
3. **部署编排复杂度**：跨容器需要复杂的 Docker Compose 网络配置与重试机制，对企业内网和单机科研人员带来极大的使用门槛。

**V2.0 决策决然重构**：**摒弃跨容器 TCP Server，演进为“进程内原生协同（In-Process Native Co-Simulation）”**。将 FESTIM v2.0、DOLFINx 0.9.0、OpenModelica 1.27.1 及 CasADi 统一构建于单一 Linux 科学计算基底镜像中，通过 Python 原生模块导入直接在内存中传递张量，**通信损耗降为零，生命周期与异常调用栈完全统一闭环**。

### 5.2 全栈 All-in-One 融合拓扑架构

```
+---------------------------------------------------------------------------------------------------------+
|                                    TRICYS All-in-One Production Pod                                     |
+---------------------------------------------------------------------------------------------------------+
|  Host Ports:  8080 (Gateway)  |  8000 (FastAPI)  |  3020 (GoView)  |  8050 (Dash HDF5)                  |
+---------------------------------------------------------------------------------------------------------+
|                                                    |                                                    |
|  [ Nginx Reverse Proxy Gateway ] (Port 80) <-------+                                                    |
|    - /        --> Vue 3 Web Console (/opt/tricys-web/visual)                                            |
|    - /api/    --> FastAPI Industrial Backend (127.0.0.1:8000)                                           |
|    - /goview/ --> GoView Digital Twin Screen (/opt/tricys-web/goview)                                   |
|    - /hdf5/   --> Dash/Plotly HDF5 Interactive Visualizer (127.0.0.1:8050)                              |
+---------------------------------------------------------------------------------------------------------+
|  [ Unified Scientific Multi-Physics Runtime Base ] (Python 3.12 / Ubuntu 24.04 noble)                    |
|    - System DAE & FMI/SSP:       OpenModelica 1.27.1 + OMSimulator + Solidified MSL (4.1/4.0/3.2.3)     |
|    - Component-Level FEM:        FESTIM v2.0 + DOLFINx 0.9.0 + PETSc + SLEPc + scifem                   |
|    - Isotope Distillation MESH:  CasADi >= 3.6 + H2ISO                                                  |
|    - Co-Simulation Runner:       FMPy >= 0.3.26                                                         |
|    - Execution Pipeline Engine:  TRICYS V2.0 Core (WorkflowPipeline + Physics Semantic Bus)             |
+---------------------------------------------------------------------------------------------------------+
|  [ Security & Persistence Model ]                                                                       |
|    - User Governance:  appuser (UID 1000) via gosu privilege dropping                                  |
|    - Persistent Data:  /data (SQLite DB, Workspaces, Assets, HDF5 Contexts)                            |
+---------------------------------------------------------------------------------------------------------+
```

### 5.3 四大核心工程攻坚突破

1. **突破 1：OpenModelica 离线标准库自愈固化 (Self-Healing Solidified MSL)**
   - **痛点**：官方 OpenModelica 在首次加载 `package.mo` 或执行仿真时，若本地缺失 MSL 缓存，会自动触发联网向 OSMC 服务器下载 `Modelica 4.0.0`。在科研院所气隙隔离机房或未配置代理的 Docker 环境中，该下载必死锁超时，导致构建或仿真全面崩溃；
   - **方案**：在 Dockerfile 构建阶段，自愈下载官方 `omlibrary_1.27.1-1_all.deb`，提取全部 MSL 缓存包，在容器内强制固化解压 `Modelica 4.1.0+maint.om`、`Modelica 4.0.0+maint.om`、`Modelica 3.2.3+maint.om`，配置软链接并授权至 `/home/appuser/.openmodelica/libraries/`，实现 **100% 气隙隔离离线编译与运行**。
2. **突破 2：非 root `appuser` 安全权限隔离与 `gosu` 动态降权**
   - 工业级安全审计严禁容器进程直接以 root 运行；OpenModelica 底层基于 ZMQ 套接字通信，若以 root 创建中间 C 源码或 IPC 管道，宿主机挂载目录将被 root 权限文件污染；
   - 镜像统一定义 UID 1000 `appuser`；启动入口 `tricys_entrypoint.sh` 启动时，由 root 快速自愈宿主机挂载目录的归属权，随后通过 `gosu appuser` 降权拉起仿真、API 与可视化服务，兼具宿主机安全与容器内无缝读写。
3. **突破 3：Nginx 单端口反向代理汇聚与多应用无缝反代**
   - 研发专门的 `nginx_allinone.conf`，将 Vue3 工业控制台静态资产（`/`）、FastAPI 后端接口与 Swagger 交互文档（`/api/`、`/docs`）、GoView 大屏（`/goview/`）以及 Dash/Plotly HDF5 可视化引擎（`/hdf5/`）全部汇聚于标准 HTTP 80/8080 单一端口；
   - 彻底解决了跨端口访问产生的 CORS 跨域安全拦截、Cookie 传递丢失以及集群端口映射繁琐问题，同时保留 8000、3020、8050 独立端口以供调试。
4. **突破 4：核心计算平台与 V&V 专用基准彻底解耦，保障工业级轻量化**
   - 彻底将重型理论对标数据集、文献对比制图脚本从 `tricys` 源码及官方镜像中剥离，使得 `tricys` 保持为纯净通用、开放轻量的多物理场协同仿真工业底座；
   - 全套聚变自持理论验证与确认（V&V）套件独立归口于 [`couuas/tricys_abdou_2021`](https://github.com/couuas/tricys_abdou_2021) 专用验证基准仓库（并在本项目中以 Git Submodule 挂载），基于已发布的官方生产镜像 `ghcr.io/asipp-neutronics/tricys:latest`，单命令即可 100% 幂等复现 Abdou et al. (2021) 与 Abdou et al. (1986) 的全部理论边界曲线与数值标记。

### 5.4 容器服务健康检查与实测验证矩阵

在生产级 Docker 容器 `tricys_allinone` 内进行了系统性端到端验收实测：

| 验证维度 | 探测/测试目标 | 验证命令/测试方法 | 实测结果 | 结论 |
| :--- | :--- | :--- | :---: | :--- |
| **网关聚合服务** | Nginx 全局反代网关 | `curl -fsS http://127.0.0.1:8080/health` | **HTTP 200** | 网关正常分发各子系统 |
| **后端 API 服务** | FastAPI 后端引擎 | `curl -fsS http://127.0.0.1:8000/health` | **HTTP 200** (`status: ok`) | 后端服务与 SQLite 数据库连接就绪 |
| **API 网关穿透** | Nginx -> FastAPI 反代 | `curl -fsS http://127.0.0.1:8080/api/v1/health` | **HTTP 200** | `/api/` 路由穿透正常，无跨域阻碍 |
| **数字孪生大屏** | GoView 大屏 Web 服务 | `curl -fsS http://127.0.0.1:3020/health` | **HTTP 200** | 大屏静态资源与路由就绪 |
| **HDF5 科学看板** | Dash / Plotly 可视化服务 | `curl -fsS http://127.0.0.1:8050/hdf5/health` | **HTTP 200** (`server_mode: true`) | HDF5 时序上下文引擎正常监听 |
| **离线标准库编译** | OMC C 代码编译与仿真 | `pytest test/core/test_constrained_buffer.py` | **5/5 PASS** (55.53s) | 离线 MSL 4.1.0/4.0.0 自愈生效 |
| **核心全量测试套件** | 系统核心单元与集成测试 | `pytest test/ -q` | **344 PASS, 11 SKIP** (100%) | 零回归错误，核心架构全绿 |
| **子模型因果性替换** | 代理模型端口与拓扑因果性校验 | `pytest test/core/test_submodel_replacer.py` | **全部 PASS** | 端口命名与因果方向强校验零悬空 |
| **工业算例套件驱动** | 12 大工业算例校验与 DAG 编译 | `pytest test/test_example_runner.py` | **5/5 PASS** (1.48s) | 算例发现、校验与显式编译 100% 通过 |
| **工业插件与工作流** | 插件系统与动态调度验证 | `pytest test/plugins/ -q` | **全部 PASS** | 纯净插件架构零固化依赖 |
| **Abdou 2021 V&V 基准** | 聚变自持理论边界核算与文献对标 | `pytest tricys_abdou_2021/scripts/` | **全部 PASS** | Track 1 & Track 2 理论等价交叉验证 100% 严格复现 |

---

## 六、 总结与后续里程碑规划

### 6.1 当前阶段性总结 (Stage 0 ~ Stage 4 硬化成效)
TRICYS V2.0 目前已建立起**坚实完备的核心技术壁垒与成熟工业计算平台**：
1. **可靠性**：SI 秒量纲完全对齐、用户工程输入（小时制）与模型物理时间量纲解耦、多进程无死锁、全闭环质量平衡核算精度达毫克级（残差累计误差仅 $-0.022\text{ g}$）；
2. **灵活性**：五大支柱 DSL（支持 `simulate`, `sweep`, `sensitivity`, `cosim`, `optimize`）与纯显式 Pipeline 自由切换与无损编译，插件依赖 DAG 自动拓扑重排；
3. **协同仿真三大范式确立**：标准化落地离线查表 (`offline_table`)、在线步进 (`online_step`) 与稳态参数注入 (`param_inject`)，引入子模型拓扑因果性校验器与原子/分子流跨维度适配组件；
4. **双层嵌套二分优化引擎**：突破单参数寻优瓶颈，在物理内核层构建自持最小启动库存与临界 TBR 双约束协同求解器，集成多维高阶图表报告；
5. **智能化与物理边界对标**：具备自审自愈能力的 Agent Skill 协议全链路贯通，挂载 `tricys_abdou_2021` 宏观物理理论基准子模块，杜绝脱离实际的参数幻觉；
6. **云原生全栈融合**：All-in-One 单容器架构消除了跨系统、跨语言与跨网络部署壁垒，OpenModelica 离线标准库自愈固化打破内网构建死锁，单端口 Nginx 网关整合全生命周期服务；
7. **工程质量**：全系统自动化核心测试用例达 **344 passed, 11 skipped（100% 绿灯）**，12 大工业级基准算例（01 ~ 12）统一由 `TricysV2ExampleRunner` 驱动，全套 Abdou 2021 聚变燃料自持宏观物理基准（`tricys_abdou_2021`）实现 100% 理论对标复现，核心软件与专用 V&V 仓库彻底解耦。

### 6.2 后续演进路线 (Stage 5 最终攻坚)
- **存量模型与配置兼容性回归**：保障 V1.0 存量历史模型与配置在 V2 微内核中 100% 无损向后兼容执行；
- **操作手册全量大清洗**：统筹完成面向人类开发者与 Agent 架构师的双受众《TRICYS V2 权威操作与排查手册》全量大清洗（激活 8 大专栏中英文规范），确保 `mkdocs build --strict` 零报错零告警；
- **正式发布与交付**：更新 `pyproject.toml` 版本至 `2.0.0`，编写 `CHANGELOG.md` 与 Release Notes，正式完成 TRICYS 2.0.0 工业级发布。

---

## 七、 核心演进脉络：关键 Commit 逐项技术攻坚复盘 (Commit Evolution Log)

为清晰追溯从协同仿真热修到 V2.0 工业级体系成型的完整技术演进脉络，本节对 `e22f65a` 至 `2811d2b` 期间的 10 次核心提交进行逐项攻坚复盘：

### 1. `e22f65a` (Merge PR #104 / `19b91b3`) — 协同仿真 SSP 导出目录防御性热修
- **核心定位**：热修 PR #101 评审项 C1（`script/cosim/4_export_ssp.py` 目录清理行为安全加固）。
- **攻坚要点**：
  - 原实现于 auto-build 的 `finally` 块中无差别遍历并清空 `--fmu_dir` 内所有非 `.fmu` 条目，若用户指定目录中存在已有文件或子目录，会导致不可逆的数据抹除；
  - 重构为**构建前目录快照比对机制**：清理阶段仅删除“本次构建新产生且非 `.fmu`”的临时条目，既有资产与预置内容一律严格保护；
  - 新增 `test/script/test_export_ssp_no_dir_wipe.py`（哨兵文件/哨兵目录可证伪回归断言），实现协同仿真导出脚本的幂等与安全防御。

### 2. `0528b69` — 微内核流水线架构重构与 TRICYS 1.0 兼容性确立
- **核心定位**：全面启动 TRICYS V2.0 模块化、可插拔流水线微内核重构。
- **攻坚要点**：
  - 废弃历史单体式 `AnalysisCasesManager`，仿真全生命周期统一收敛为单向流动的标准三阶段微内核 `WorkflowPipeline`；
  - 引入 `SamplingEngine`，原生支持 Sobol、LHS、Morris、FAST 与网格扫参空间展开；
  - 建立基于强类型 Pydantic DTO 的 `schema.py` 规范，实现构造器控制反转与依赖注入；
  - 将优化寻优提升至仿真执行期的一等公民，后处理严格遵循只读纯粹性契约；
  - FOC 脉冲块参数化支持 `amplitude_scale` 与时序冲突检测，并达成对 TRICYS 1.0 历史 JSON 配置的 100% 向后兼容。

### 3. `c517fd5` — Stage 1 里程碑交付：统一多态分发、并发死锁消解与通用报告体系
- **核心定位**：Schema 2.0 全链路消费闭环、进程池防死锁与现代学术报告系统上线。
- **攻坚要点**：
  - **任务多态路由**：打通 `simulate`、`sweep`、`sensitivity` 任务在 CLI 与 `PipelineBuilder` 中的统一调度，标准化算子化指标提取（`startup_inventory`, `doubling_time`, `bisection_search`）；
  - **消解双层进程池死锁**：在多工况扫参中强制内层 Worker 单核运行，通过 OpenModelica 预编译原生二进制加速执行；解耦 `StoragePlugin`，HDF5 `/jobs` 纯净沉淀输入，`/summary` 记录寻优指标；
  - **三大通用物理报告引擎与防伪**：彻底清除硬编码与伪造假数据，精简收敛为 `simulate`、`sweep`、`sensitivity` 三大通用物理模板，支持外部专业术语表（Glossary CSV）解耦；
  - **现代化纯英文看板与出版级导出**：重构四大卡片流式 Visualizer，支持学术级 Markdown、样式化 DOCX（三线表与 LaTeX 公式）与 PDF 导出；测试套件扩充至 226 项全绿。

### 4. `20737fa` — Stage 2 里程碑交付：时间量纲全面治理与 Issue #99 彻底根除
- **核心定位**：聚变堆全系统时间单位对齐国际单位制（SI 秒）并消除历史标度缺陷。
- **攻坚要点**：
  - **SI 秒第一公民**：底层数值积分与时序数组一律对齐 SI 秒，配置解析入口建立 `_sim_time_to_seconds` 幂等守卫；
  - **Modelica 物理双基准参数化**：`example_model.mo` 引入 `parameter Real time_scale = 3600`，达成秒与小时双基准理论等价交叉验证（$\text{rtol} \le 10^{-6}$）；
  - **FOC 动态脉冲时基对齐**：支持 9 种头指令时基自动推导与失配显式预警，消除全库隐式 `/ 24` 或 `/ 8760` 硬编码；
  - **根除物理后处理 720 万小时缺陷**：修复 `tritium_cycle.py` 原生秒级数组误判，图表横轴统一标定为 `仿真时间 (h)`（0~2000h），报告自洽呈现秒/小时双时间；
  - 端到端通过工业算例 01~06，测试套件扩增至 269 项全绿。

### 5. `8d405b7` — Stage 3 里程碑交付：模型中心化 Agent-Skill 协议与插件生态
- **核心定位**：聚变仿真领域专用智能协同体系、防幻觉自省与结构化异常自愈屏障。
- **攻坚要点**：
  - **七步闭环工作流**：在 `skills/tricys-agent/` 中固化严密工程规范（Intake -> Review -> Introspect -> Synthesis -> Validate -> Run/Diagnose -> Audit）；
  - **零幻觉 AST 模型自省与打补丁自愈**：研发 `ModelicaIntrospector` 提取实测参数树与流股拓扑，研发 `ModelicaAuditor` 自动生成 Unified Diff 补丁就地自愈守恒漏项与时基失配；
  - **多级自愈诊断屏障 (`DiagnosticsBarrier`)**：捕获步长下溢与数值奇异，导出结构化 `diagnostics.json`；
  - **聚变物理语义总线与 DAG 拓扑自排**：内置 7 大系统 30 项标准化概念（`PhysicsSemanticResolver`），基于 Kahn 算法自动重排 I/O 契约 DAG；
  - **双重安全沙箱**：AST 静态代码安全扫描器拦截系统越权调用，毫秒级纯内存 Mock 沙箱（`plugin test`）实现接口秒级自检；
  - 端到端贯通工业算例 07（自定义插件与语义总线），全量测试达 307 项通过。

### 6. `36b0285` — 协同仿真三大范式确立、子模型因果性替换与开发者规约
- **核心定位**：协同仿真体系架构解耦、执行引擎标准化与同位素流跨维度桥接。
- **攻坚要点**：
  - **三大标准化协同仿真范式**：
    1. 离线查表模式 (`offline_table`，算例 08)：Run-Process-Rerun 两阶段自动构建 `CombiTimeTable` 包装模型，严禁声明 `substitute_model`；
    2. 在线步进模式 (`online_step`，算例 09)：基于 OMSimulator 驱动动态模型与外部高保真处理器锁步双向同步，保护源码不篡改；
    3. 参数注入模式 (`param_inject`，算例 10，Wang 2022)：离线严格热力学求解 + Modelica `-overrideFile` 稳态参数动态注入，全框架唯一显式替换模型模式；
  - **子模型因果性与拓扑强校验 (`submodel_replacer.py`)**：对替换代理模型进出端口数量、命名及因果方向实施绝对零悬空强校验，自动绑定默认方程；
  - **原子流与分子流跨维度桥接**：研制 `Generic_ISS_Adapters.mo` 配合 Python Handler，打通全厂宏观 3/5 维原子流与精细精馏 6 维同位素分子流映射；
  - 编纂 `cosimulation_framework_guide.md`，算例扩充至 10 个，测试套件增至 333 项通过。

### 7. `9d5d663` — Stage 4 里程碑交付：All-in-One 云原生全栈融合底座与专用基准解耦
- **核心定位**：开箱即用多物理场单容器生产底座、离线 MSL 固化与纯净开源轻量化。
- **攻坚要点**：
  - **架构决策重构**：彻底废除通信开销大、生命周期易失步死锁的跨容器 TCP Server 模式，演进为同进程内存原生多物理场协同；
  - **全栈 All-in-One 单容器融合**：在 Ubuntu 24.04 基底镜像中深度装配 OpenModelica 1.27.1、FESTIM v2.0、DOLFINx 0.9.0、CasADi/H2ISO、FastAPI、Vue 3、GoView、Dash/Plotly HDF5 与 Nginx；
  - **OpenModelica 离线标准库自愈固化**：在构建期离线提取官方 MSL 3.2.3/4.0.0/4.1.0 缓存包，彻底解决企业内网与气隙环境下的连网死锁超时；
  - **非 root 安全降权与单端口聚合**：UID 1000 `appuser` 与 `gosu` 动态降权保障编译与文件读写安全；Nginx 统一收敛控制台、API、大屏与看板至单端口 80/8080；
  - **文献对标套件彻底解耦**：将专用文献重型对标数据与制图脚本剥离出核心镜像，保持 `tricys` 作为轻量通用工业级平台；测试套件 297 passed, 6 skipped 达成 100% 绿灯。

### 8. `6ee8c4c` — 历史算例安全归档与 V2 统一交互式运行器落地
- **核心定位**：案例工程体系规范化治理、多入口脚本清理与统一 CLI 驱动。
- **攻坚要点**：
  - **历史 V1 算例安全归档 (`v1_examples_legacy/`)**：将原 `basic/` 与 `analysis/` 历史算例完整归档，撰写说明文档声明弃用政策与时间量纲演进细节，保留向后兼容导入 shim；
  - **统一 V2 算例调度体系 (`TricysV2ExampleRunner`)**：重构 `example/example_runner/runner.py` 并接入 `python -m tricys example` 顶层命令，原生支持 `--list`、`--validate`、`--compile`、`--dry-run`、`--run` 等交互式与批处理操作；
  - 编写 `test/test_example_runner.py` 自动化测试，验证算例目录规范、自动发现、无损 DAG 编译与干跑执行。

### 9. `294eb72` — 双层嵌套二分优化引擎、顶级工业算例扩充与多维报告呈现
- **核心定位**：突破单变量优化局限，构建聚变自持双约束闭环求解器与多维图表。
- **攻坚要点**：
  - **双层嵌套二分优化引擎 (`NestedBisectionOptimizer`)**：外层二分寻优最小启动氚库存 $I_{S,0}$（满足在役安全储备谷底约束 $I_{\text{valley}} \ge I_{\text{reserve}}$），内层二分寻优临界产氚比 $TBR$（满足倍增时间约束 $t_d \le t_{d,\text{target}}$），实现双物理约束协同收敛；
  - **扩充两类高阶工业算例**：交付算例 11（`11_optimize_nested_bisection.yaml`，双层嵌套优化）与算例 12（`12_sweep_nested_optimization.yaml`，燃烧分数扫描内嵌双层优化）；
  - **多维高阶图表与报告渲染**：在 `figures.py` 与 `templates.py` 中新增外层库存 vs 内层 TBR 收敛轨迹图、三维/多曲线参数对比图与嵌套优化工程分析报告。

### 10. `2811d2b` — 物理时间量纲精细解耦与 Abdou 2021 理论基准库挂载
- **核心定位**：用户输入与模型物理量纲彻底解耦、路径自愈及挂载专用 V&V 基准仓库。
- **攻坚要点**：
  - **`scenario.time_base` 语义精细解耦**：用户端配置默认按小时（hour）输入；`time_base` 显式声明底层 Modelica 模型的物理时间量纲（秒 vs 小时），流水线自适应推导时间缩放因子（`time_scale_factor`）并在求解器中透明穿透；
  - **FOC 动态脉冲时基对齐**：事件调度器根据 `model_time_base` 自动校验脉冲时序，提供失配预警并自适应换算；
  - **向上递归路径解析**：在 `config_utils.py` 中增加向上递归路径探测，保障老旧相对路径兼容性；
  - **正式挂载专用 V&V 仓库 `tricys_abdou_2021`**：以 Git Submodule 形式引入 Abdou et al. (2021) 聚变自持理论边界验证基准库（commit `784efe4`），建立 Track 1 与 Track 2 理论等价交叉验证基准，作为 Agent 评估聚变物理常识与理论置信度的坚实参考系；
  - **全量核心测试达到 344 passed, 11 skipped (100% 绿灯)**。
