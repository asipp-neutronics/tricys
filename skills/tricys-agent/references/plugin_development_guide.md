# TRICYS 插件开发、自主扩展与自愈深度指南 (Plugin Development & Self-Healing Guide)

> **理论与架构基石**:  
> TRICYS v2.0 采用高度模块化的三阶段流水线编排架构（`Preprocessing` → `Simulation` → `Postprocessing`），所有状态通过强类型的上下文载体 `PipelineContext` 贯通流转。  
> 系统实现了“**高阶领域 DSL 与低阶显式编排 IR 解耦、底层微内核插件统一、领域配置 1:1 无损编译、聚变物理语义映射总线以及 DAG 自动拓扑推导**”的 Agent-Native 架构。  
> 当用户的工程需求超出 TRICYS 原生内置能力时，**Agent 应主动作为系统架构师与插件开发者，自主构建、挂载、编译、测试并自愈 TRICYS 插件**。

---

## 📌 1. 双轨配置体系与微内核插件架构 (Dual-Track Architecture)

TRICYS 区分两套完全解耦但底层互通的配置模式，二者共享完全一致的微内核插件实现：

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 模式 1: 高阶领域声明 DSL (High-Level Domain DSL)                             │
│ 根节点: model | scenario | task | visualizer | plugins (增量钩子)             │
│ 面向: 聚变工程业务场景 (关注“物理模型是什么、扫什么参数、何时停机”)          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       │ 编译器降级 (PipelineBuilder.compile_v2_to_explicit_pipeline)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 模式 2: 低阶显式编排 IR (Low-Level Execution Pipeline IR)                   │
│ 根节点: 纯 pipeline: { preprocessing: [...], simulation: ..., postprocessing: [...] }│
│ 面向: Agent 拓扑组装 / 管道测试 (纯流水线结构，无根节点冗余，各插件显式传参)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼ 实例化驱动
┌─────────────────────────────────────────────────────────────────────────────┐
│ 统一微内核插件池 (Unified Micro-Kernel Plugins)                              │
│ setup | compiler | standard | storage | safety | report | custom_plugins    │
│ 契约: Explicit-First with Context-Fallback (显式传参优先，上下文自动兜底)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 模式对比与选型准则

| 对比维度 | 模式 1：高阶领域声明 (Schema 2.0 DSL) | 模式 2：纯显式编排 (Pipeline IR) |
| :--- | :--- | :--- |
| **顶级字段** | `model`, `scenario`, `task`, `visualizer`, `plugins` | 仅 `pipeline:` 块（无根节点业务字段冗余） |
| **插件挂载方式** | 在 `plugins.preprocess` 或 `plugins.postprocess` 中增量声明追加。 | 在 `pipeline.preprocessing` 或 `pipeline.postprocessing` 中显式排定全量执行拓扑。 |
| **参数传递机制** | 各插件从领域顶级字段推断输入，增量插件通过自身 `params:` 传参。 | **所有插件和引擎均通过各自的 `params:` 显式传参**。 |
| **拓扑排序行为** | 默认健全流水线 + 末尾安全追加。 | **自动基于 I/O 契约运行 DAG 拓扑排序**，无需担心时序倒置。 |
| **适用场景** | 聚变堆物理方案设计、标准参数扫描、日常仿真生产运行。 | Agent 复杂流水线组合、自定义引擎替换、插件独立单元测试 (Mock Testing)。 |
| **编译器支持** | 可调用 `PipelineBuilder.compile_v2_to_explicit_pipeline()` 1:1 无损下沉为模式 2。 | 原生直接被 `PipelineBuilder.build()` 实例化执行。 |

---

## 🧩 2. TRICYS 全量内置插件手册与五大顶级字段映射矩阵 (Built-in Plugins & DSL Mapping Matrix)

TRICYS 内部将仿真生命周期拆解为一系列独立的微内核插件。无论在 **模式 1 (Schema 2.0 DSL)** 还是 **模式 2 (Explicit Pipeline IR)** 下，底层均调用同一套插件实现。所有插件均遵循 **`Explicit-First with Context-Fallback`** 传参契约：若显式传入了 `params`，则优先采用显式参数；若未传入，则自动从 `context.config` 或高阶配置中智能解析兜底。

### 2.1 全量内置插件总览与注册别名表

| 阶段类别 | 插件标准名称 (`name`) | 注册别名 (`aliases`) | 核心实现类 | 核心功能与职责 |
| :--- | :--- | :--- | :--- | :--- |
| **Preprocessing** | **`setup`** | `simulation_setup`, `modelica_setup`, `isolation`, `job_generator` | `SimulationSetupPlugin` | 运行环境准备、工作空间隔离、多工况扫描任务网格展开 (`jobs`)。 |
| **Preprocessing** | **`compiler`** | `model_compiler`, `precompile` | `ModelCompilerPlugin` | Modelica 模型静态编译，生成独立 C 仿真执行体，支撑多进程高并发求解。 |
| **Preprocessing** | **`foc`** | `foc_injector`, `pulse_scenario` | `FOCPulseInjectorPlugin` | 解析聚变运行工况 (.foc) 脉冲脚本，动态注入脉冲加热与增殖驱动。 |
| **Preprocessing** | **`interceptor`** | `interceptor_builder`, `cosim_interceptor` | `CoSimInterceptorPlugin` | 离线协同仿真拦截器接口与流股置换元数据配置。 |
| **Simulation** | **`standard`** | `omc`, `standard_simulation` | `StandardSimulationEngine` | OpenModelica 标准数值积分内核，支持 DASSL/IDA 求解及多进程并发。 |
| **Simulation** | **`offline_cosim`**| `file_cosim` | `OfflineCoSimEngine` | 基于磁盘文件交换的离线多回路协同仿真引擎。 |
| **Simulation** | **`online_oms`** | `oms`, `omsimulator` | `OnlineOMSEngine` | 基于 OMSimulator / FMI / SSP 总线的在线高频协同仿真引擎。 |
| **Postprocessing**| **`storage`** | `simulation_storage`, `hdf5_storage`, `data_storage` | `SimulationStoragePlugin` | 关键指标计算评价、HDF5 高效时序持久化、内存结果强类型同步与 CSV 导出。 |
| **Postprocessing**| **`safety`** | `safety_auditor`, `audit`, `auditor`, `mass_conservation`, `offline_auditor`, `static_alarm` | `SafetyAuditorPlugin` | 全系统闭环质量守恒平衡定量审计与静态安全阈值越限监测告警。 |
| **Postprocessing**| **`report`** | `analysis_report`, `tasks`, `legacy_tasks`, `baseline_plot` | `AnalysisReportPlugin` | 系统运行工况深入分析、可视化图表绘制与三格式 (MD/DOCX/PDF) 学术工程报告交付。 |
| **Postprocessing**| **`sensitivity_analysis`** | `sensitivity`, `uq` | `SensitivityAnalysisPlugin` | Sobol / Morris 全局敏感度分析采样分解与不确定性量化评估。 |

---

### 2.2 内置插件显式参数手册 (Explicit Parameters Specification)

在 **模式 2 (显式编排 Pipeline)** 中，所有插件均通过 `params` 块显式传参：

#### 1. 前处理插件: `setup`
```yaml
- name: "setup"
  params:
    package_path: "path/to/package.mo"   # (str) Modelica 模型包绝对/相对路径
    model_name: "MyPackage.MyModel"      # (str) 仿真目标全限定类名
    parameters:                          # (dict 或 list[dict]) 单工况参数字典或多工况参数列表
      tbr: 1.15
      sds.V: 2.0
    task_type: "sweep"                   # (str) 任务类型: "simulate" | "sweep" | "sensitivity"
```

#### 2. 前处理插件: `compiler`
```yaml
- name: "compiler"
  params:
    package_path: "path/to/package.mo"   # (str, 可选) 待预编译的模型包路径
    model_name: "MyPackage.MyModel"      # (str, 可选) 目标模型名
    cflags: "-O3 -march=native"          # (str, 可选) 传给底层 C 编译器的编译优化标志
    target: "msvc"                       # (str, 默认 msvc/gcc) 目标平台环境
```

#### 3. 仿真引擎: `standard`
```yaml
simulation:
  name: "standard"
  params:
    stop_time: 10000.0                   # (float, 默认 1000.0) 积分终止时间 (秒制或小时制)
    step_size: 1.0                       # (float, 默认 10.0) 积分输出步长 / 通讯间隔
    start_time: 0.0                      # (float, 默认 0.0) 积分起始时间
    solver: "dassl"                      # (str, 默认 "dassl") DAE 求解器: "dassl" | "ida" | "cvode"
    tolerance: 1e-6                      # (float, 默认 1e-6) 积分相对容差
    output_format: "mat"                 # (str, 默认 "mat") 求解器原始输出文件格式 ("mat" | "csv")
    variable_filter: "time|sds.*|.*\.I.*" # (str, 可选) 变量正则过滤表达式 (减小结果体积)
    parallel: true                       # (bool, 默认 False) 是否启用多进程并行求解多工况
    max_workers: 4                       # (int, 可选) 最大并发 Worker 进程数
```

#### 4. 后处理插件: `storage`
```yaml
- name: "storage"
  params:
    metrics_definition:                  # (dict, 可选) 关键物理评价指标字典
      Min_Storage:
        source: "sds.inventory"
        evaluate: "min"
      Doubling_Time:
        source: "sds.inventory"
        evaluate: "doubling_time"
    results_dir: "./results"             # (str, 默认 ./results) 仿真成果落盘主目录
    export_csv: true                     # (bool, 默认 False) 是否将各工况抽取导出为独立 CSV
```

#### 5. 后处理插件: `safety`
```yaml
- name: "safety"
  params:
    auditor_config:                      # (dict, 可选) 全厂物料平衡审计规则
      tolerance: 1e-4
      patterns:
        inventory: ".*inventory.*|.*I\\[1\\].*"
        flow: ".*flow.*"
    alarm_rules:                         # (list[dict], 可选) 静态安全限值报警规则
      - variable: "sds.inventory"
        condition: "<"
        threshold: 500.0
        severity: "CRITICAL"
        action: "SCRAM"
```

#### 6. 后处理插件: `report`
```yaml
- name: "report"
  params:
    report_config:                       # (dict, 可选) 报告生成交付规范
      enabled: true
      title: "聚变堆在役储罐安全裕度与自持动力学评估报告"
      formats: ["markdown", "docx", "pdf"] # 输出格式清单
      sections: ["summary", "indicators", "timeseries", "recommendations"]
    run_baseline_plot: true              # (bool, 默认 False) 是否自动绘制基线演化图
```

#### 7. 后处理插件: `sensitivity_analysis`
```yaml
- name: "sensitivity_analysis"
  params:
    enabled: true                        # (bool, 默认 True) 激活敏感度分析
    sensitivity_config:                  # (dict, 可选) 采样与指标敏感度配置
      method: "sobol"
      num_samples: 64
      calc_second_order: false
```

---

### 2.3 五大顶级字段 ↔ 内置插件 1:1 双向映射矩阵

在 **模式 1 (Schema 2.0 DSL)** 中，用户只需声明面向物理业务的五大顶级字段。编译器 `compile_v2_to_explicit_pipeline` 会自动按照下表规则，将领域字段**无损单向编译（Lowering）**为底层插件的显式参数：

| Schema 2.0 五大顶级字段与属性 | 目标插件 (`Plugin Name`) | 目标插件显式参数 (`Plugin Param`) | 映射推断规则与数据转换说明 |
| :--- | :--- | :--- | :--- |
| **`model.fileName`** | `setup`, `compiler` | `package_path` | 直接透传为 Modelica 代码包路径。 |
| **`model.modelName`** | `setup`, `compiler` | `model_name` | 直接透传为仿真目标模型的完整类名。 |
| **`model.cflags`** | `compiler` | `cflags` | 传给编译器插件的底层编译优化标志。 |
| **`model.target`** | `compiler` | `target` | 传给编译器插件的目标架构平台。 |
| **`scenario.stopTime`** | `standard` | `stop_time` | 转换为浮点型仿真截止时间。 |
| **`scenario.stepSize`** | `standard` | `step_size` | 转换为浮点型仿真通讯步长。 |
| **`scenario.startTime`** | `standard` | `start_time` | 仿真起始时间（缺省默认 0.0）。 |
| **`scenario.method`** | `standard` | `solver` | 映射为求解器算法名称（如 `"dassl"` / `"ida"`）。 |
| **`scenario.tolerance`** | `standard` | `tolerance` | 转换为浮点型数值相对误差容限。 |
| **`scenario.outputFormat`** | `standard` | `output_format` | 原始轨迹导出格式（`"mat"` 或 `"csv"`）。 |
| **`scenario.variableFilter`** | `standard` | `variable_filter` | 正则过滤模式串，控制求解器变量记录范围。 |
| **`scenario.pulse`** / **`foc`** | `foc` | `foc_config` | 若配置了脉冲工况，自动激活 `foc` 插件注入脉冲。 |
| **`task.type`** | `setup` | `task_type` | 声明任务形态（`simulate`, `sweep`, `sensitivity`）。 |
| **`task.parameters`** | `setup` | `parameters` | 单工况参数字典或多工况参数网格字典列表。 |
| **`task.parallel`** | `standard`, `compiler` | `parallel` | 若为 `true`，触发 `compiler` 预编译并激活引擎多进程并行。 |
| **`task.max_workers`** | `standard` | `max_workers` | 并发最大进程数。 |
| **`task.metrics`** / **`task.metrics_definition`** | `storage` | `metrics_definition` | 传递给存储插件用于自动从时序核算关键评价标量。 |
| **`task.auditor`** / **`auditor`** | `safety` | `auditor_config` | 若存在物料平衡审计定义，自动激活 `safety` 审计插件。 |
| **`static_alarm.rules`** | `safety` | `alarm_rules` | 若声明了报警阈值，自动注入 `safety` 插件的报警监控规则。 |
| **`visualizer.report`** / **`report`** | `report` | `report_config` | 若配置了报告生成器，自动激活 `report` 交付插件。 |
| **`visualizer.baseline_plot`** | `report` | `run_baseline_plot` | 布尔开关，控制是否生成基线趋势对照大图。 |
| **`workspace.results_dir`** | `storage` | `results_dir` | 仿真数据、指标及 HDF5 库的落盘根目录。 |
| **`plugins.preprocess`** | `pipeline.preprocessing` | *(自身 params)* | **1:1 原样追加**：增量前处理外部脚本顺序并入前处理管线。 |
| **`plugins.postprocess`** | `pipeline.postprocessing` | *(自身 params)* | **1:1 原样追加**：增量后处理外部脚本顺序并入后处理管线。 |

---

### 2.4 编译降级实现原理与无损验证

TRICYS 提供了代码级和命令行级的 1:1 双向无损降级通道：

#### 1. 命令行一键编译查看
```bash
python -m tricys.main compile -c my_v2_config.yaml -o compiled_pipeline.yaml
```
该命令会将任何合法的 Schema 2.0 五大顶级字段配置瞬间转换为纯显式编排的 `pipeline:` 结构。

#### 2. Python 代码内无损调用
```python
from tricys.workflow.builder import compile_v2_to_explicit_pipeline

# 将五大顶级字段字典转换为显式 Pipeline 字典
explicit_pipeline_dict = compile_v2_to_explicit_pipeline(v2_config_dict)
```

通过这一编译映射机制，无论是人类物理学家使用 Schema 2.0 简洁书写物理业务，还是 AI Agent 针对微内核插件组装极其复杂的拓扑网络，底层的插件执行逻辑与物理计算完全 100% 统一复用！

---

## ⚛️ 3. 强类型数据契约与开放扩展字段 (Typed Context & Open Payloads)

为彻底根除 Agent 面对弱类型字典产生的“字段名幻觉”，TRICYS 在 [tricys.workflow.context](file:///d:/Administrator/Documents/Github/tricys/tricys/workflow/context.py) 中引入强类型数据契约，并遵循**开闭原则（Open-Closed Principle）**保留了分层动态扩展字段：

### 3.1 任务结果强类型契约 `SimulationJobResult`
```python
@dataclass
class SimulationJobResult:
    job_id: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    result_path: str = ""
    dataframe: Optional[pd.DataFrame] = None
    time_column: str = "time"
    time_unit: str = "second"
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 🌟 任务级动态扩展字段: 允许插件注入私有中间张量或工况标识
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """按需惰性加载 CSV 数据帧并缓存至内存。"""
```

### 3.2 全局上下文载体 `PipelineContext`
```python
@dataclass
class PipelineContext:
    # 强类型仿真结果列表 (100% 静态属性代码提示)
    results: List[SimulationJobResult]
    
    # 全局基础设施与资产
    results_store_path: Optional[str]
    results_dir: str
    package_path: str
    model_name: str
    
    # 物理语义总线解析器 (跨模型泛化)
    semantic_resolver: PhysicsSemanticResolver
    
    # 🌟 全局级动态扩展字段: 允许跨插件传递任意自定义上下文或成果路径
    metadata: Dict[str, Any]
    artifacts: Dict[str, Any]
```

---

## 🌌 4. 聚变物理语义总线与跨模型泛化词典 (Physics Semantic Bus)

不同聚变堆 Modelica 模型中变量命名千差万别（如 `sds.inventory` vs `Cycle_3LC.sds.inventory` vs `sds.I[1]`）。TRICYS 内置基于 [example_glossary.csv](file:///d:/Administrator/Documents/Github/tricys/tricys/example/example_data/example_glossary/example_glossary.csv) 的 **26 个跨模型泛化物理概念词典**，并在 [tricys.core.semantic](file:///d:/Administrator/Documents/Github/tricys/tricys/core/semantic.py) 中提供透明解析。

### 4.1 常用跨模型泛化物理变量标准表

| 物理概念 (`Physics Concept Key`) | 国际学术符号 | Modelica 候选列名自动匹配模式 | 工程物理含义 |
| :--- | :--- | :--- | :--- |
| **`sds_inventory`** | $I_{sds}(t)$ | `(?i).*(sds|storage).*(inventory|I\[1\]).*` | 中央储罐在役存量，全厂燃料安全基石 |
| **`sds_decay_loss`** | $\lambda I_{sds}$ | `(?i).*(sds|storage).*decay_loss.*` | 中央储罐放射性衰变损失总量 |
| **`blanket_inventory`** | $I_{bz}(t)$ | `(?i).*(blanket|bz).*(inventory|I\[1\]).*` | 增殖包层内部滞留量（主要库存持有者之一） |
| **`blanket_tbr`** | $TBR$ | `(?i).*(blanket|bz).*tbr.*` | 氚增殖比，燃料自持判定核心 (必须 > 1) |
| **`blanket_residence_time`** | $\tau_{bz}$ | `(?i).*(blanket|bz)\.(T|residence).*` | 包层吹扫气体平均滞留时间 |
| **`plasma_burn_fraction`** | $f_b$ | `(?i).*plasma\.(fb|f_b|burnup).*` | 等离子体燃烧份额（注入燃料实际聚变比率） |
| **`plasma_fueling_eff`** | $\eta_f$ | `(?i).*plasma\.(nf|fueling_eff).*` | 等离子体加料效率（实际进入芯部比率） |
| **`pulse_amplitude`** | $S_{burn}$ | `(?i).*pulse.*amplitude.*` | 氚消耗率/源项驱动强度 ($g/h$) |
| **`availability_factor`** | $DF$ | `(?i).*pulse.*width.*` | 运行占空比/可用率（运行时间占周期百分比） |
| **`divertor_inventory`** | $I_{div}(t)$ | `(?i).*div.*(inventory|I\[1\]).*` | 偏滤器材料滞留与渗透盘存量 |
| **`pump_inventory`** | $I_{pump}(t)$ | `(?i).*pump.*(inventory|I\[1\]).*` | 真空低温抽气泵盘存量 |
| **`tep_fep_inventory`** | $I_{fep}(t)$ | `(?i).*tep_fep.*(inventory|I\[1\]).*` | 氚工厂前端预处理除杂盘存量 |
| **`tep_fcu_inventory`** | $I_{fcu}(t)$ | `(?i).*tep_fcu.*(inventory|I\[1\]).*` | 氚工厂最终净化终端吸附盘存量 |
| **`dir_ratio`** | $DIR$ | `(?i).*tep_fep.*to_sds.*` | 直接内循环旁路比例（跳过精馏直接回送 SDS） |
| **`i_iss_inventory`** | $I_{i\_iss}(t)$ | `(?i).*i_iss.*(inventory|I\[1\]).*` | 内循环同位素分离低温精馏塔盘存量 |
| **`coolant_inventory`** | $I_{cool}(t)$ | `(?i).*coolant_pipe.*(inventory|I\[1\]).*` | 主冷却回路管道存量（巨大缓冲池） |
| **`cps_inventory`** | $I_{cps}(t)$ | `(?i).*cps.*(inventory|I\[1\]).*` | 冷却剂净化系统 (CPS) 吸附床存量 |
| **`tes_inventory`** | $I_{tes}(t)$ | `(?i).*tes.*(inventory|I\[1\]).*` | 包层吹扫气提取系统 (TES) 吸附床存量 |
| **`wds_inventory`** | $I_{wds}(t)$ | `(?i).*wds.*(inventory|I\[1\]).*` | 水去氚系统 (WDS) 精馏塔存量（高延迟部件） |
| **`first_wall_inventory`** | $I_{fw}(t)$ | `(?i).*fw.*(inventory|I\[1\]).*` | 第一壁材料滞留量 |

### 4.2 动态扩展机制 (Extensible Mapping)
在 YAML 中可随时为模型追加专属映射或覆盖默认规则：
```yaml
semantic_mapping:
  # 覆盖储罐映射规则
  sds_inventory: ["my_private_tank.mass_T"]
  # 追加全新的专属物理量
  cryo_leakage_rate: ["cryo_subsystem.leak_flow", "valve.leak"]
```

### 4.3 插件中极简调用范例
```python
class MyAuditorPlugin(PostprocessPlugin):
    def run(self, context: PipelineContext) -> None:
        df = context.results[0].dataframe
        
        # 1. 一行代码解析物理量时序（自动跨模型适配）
        sds_series = context.resolve_physics_variable(df, "sds_inventory")
        
        # 2. 一行代码提取全厂所有子系统的在役盘存量汇总字典
        all_inventories = context.get_all_inventories(df)
        # 返回: {"sds": Series, "blanket": Series, "coolant": Series, ...}
```

---

## 🔀 5. 声明式 I/O 契约与 DAG 自动拓扑排序 (Auto-DAG)

为了消除 Agent 人工排定插件执行顺序的心理负担，[PipelineBuilder](file:///d:/Administrator/Documents/Github/tricys/tricys/workflow/builder.py) 内置了**稳定拓扑排序算法（Kahn's Algorithm）**。

### 5.1 插件声明契约
插件只需声明输入依赖与产出标识：
```python
class TritiumAuditorPlugin(PostprocessPlugin):
    # 声明必须等待 HDF5 持久化完成才执行
    requires = ["results_store_path"]
    produces = ["custom_safety_summary"]
```

### 5.2 自动纠偏与依赖容错
如果用户或 Agent 在 YAML 中不慎写反了顺序（如将需要读 HDF5 的分析插件写在 `storage` 前面）：
```yaml
postprocessing:
  - script: "plugins/my_auditor.py"  # requires: results_store_path
  - name: "storage"                  # produces: results_store_path
```
**`PipelineBuilder` 会在构建期自动将 `storage` 调整到前面执行**，并打印拓扑重排日志，杜绝时序倒置崩溃。

---

## 🧪 6. 毫秒级内存 Mock 验证沙箱与脚手架 (Fast Feedback & CLI)

为了避免 Agent 每次编写完插件都要花费几十秒驱动真实 OpenModelica 编译仿真，TRICYS 提供了纯内存测试沙箱 [tricys.workflow.testing](file:///d:/Administrator/Documents/Github/tricys/tricys/workflow/testing.py) 与 CLI 工具：

### 6.1 CLI 快速脚手架生成
```bash
python -m tricys.main plugin scaffold my_evaluator --category postprocess -o plugins/my_evaluator.py
```

### 6.2 纯内存沙箱测试
```bash
python -m tricys.main plugin test plugins/my_evaluator.py --json
```

### 6.3 单元测试代码示例
```python
from tricys.workflow.testing import create_mock_context

# 1. 毫秒级生成包含真实聚变物理时序的 Mock 上下文
mock_ctx = create_mock_context(num_jobs=1, stop_time=1000.0, base_storage_g=6000.0)

# 2. 直接实例化插件并运行单测
plugin = CustomAuditorPlugin(alert_threshold_g=500.0)
plugin.run(mock_ctx)

# 3. 校验产出与指标 (总耗时 < 5ms)
assert mock_ctx.metadata["audit_status"] == "PASSED"
```

---

## ⚙️ 7. 配置双轨端到端范本对比

### 范本 A：模式 1 —— 领域声明式（Schema 2.0 DSL + 增量 Plugins）
```yaml
version: "2.0"

model:
  fileName: "skills/tricys-agent/CFEDR_2870_3lc_ssp.mo"
  modelName: "CFEDR_3LC.Cycle_3LC"

scenario:
  stopTime: 20000.0
  stepSize: 10.0
  method: "dassl"

task:
  type: "simulate"
  parameters:
    t_DD_end: 0.0
  metrics:
    Min_Storage: { source: "sds.inventory", evaluate: "min" }

plugins:
  postprocess:
    - script: "plugins/tritium_self_sufficiency_auditor.py"
      params: { alert_valley_inventory_g: 800.0 }
```

### 范本 B：模式 2 —— 纯显式编排式（Pure Pipeline IR）
**根节点无任何业务冗余字段，各插件显式传参，DAG 自动排序**：
```yaml
pipeline:
  name: "PureExplicitDemoPipeline"

  preprocessing:
    - name: "setup"
      params:
        package_path: "skills/tricys-agent/CFEDR_2870_3lc_ssp.mo"
        model_name: "CFEDR_3LC.Cycle_3LC"
        task_type: "simulate"
        parameters:
          t_DD_end: 0.0

  simulation:
    name: "standard"
    params:
      stop_time: 1000.0
      step_size: 10.0
      solver: "dassl"
      tolerance: 1e-6

  postprocessing:
    # 无论此处顺序如何，DAG 均能保证 storage 优先于依赖其结果的审计插件
    - script: "plugins/tritium_self_sufficiency_auditor.py"
      class: "TritiumSelfSufficiencyAuditorPlugin"
      params:
        alert_valley_inventory_g: 800.0

    - name: "storage"
      params:
        metrics_definition:
          Storage_Min: { source: "sds.inventory", evaluate: "min" }
        export_csv: true
```

---

## 🛡️ 8. AST 安全沙箱审查红线与自愈矩阵

1. **绝对禁用**：`subprocess`, `multiprocessing`, `threading`, `socket`, `requests`, `eval()`, `exec()`, `os.system`。
2. **推荐使用**：`numpy`, `pandas`, `scipy`, `math`, `h5py`, `json`, `pathlib.Path`。
3. **常见排查与修复**：
   - **`SecurityValidationError`**：移除子进程与网络请求，改用纯向量化科学计算；
   - **`KeyError in store`**：声明 `requires = ["results_store_path"]` 让系统 DAG 自动前置 `storage`；
   - **`Column Missing`**：使用 `context.resolve_physics_variable(df, "sds_inventory")` 彻底消除列名匹配失败。

