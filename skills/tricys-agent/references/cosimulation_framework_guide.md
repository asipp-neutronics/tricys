# TRICYS 多物理场联合仿真框架权威指南 (Co-Simulation Engineering Guide)

本指南专门面向使用 TRICYS 进行聚变堆多物理场联合仿真、子系统模型替换与外部专业求解器（如 `h2iso`、`Aspen Plus`、`DWSIM`、`FMU` 等）接入的 AI Coding Agent 与系统架构师。

---

## 🧭 1. 联合仿真体系全景与设计哲学 (Architecture & Philosophy)

聚变堆氘氚燃料循环是一个跨越宏观全厂动力学（千小时级演化）与微观单元高保真物理过程（秒级甚至毫秒级多组分传质/分离/相变）的极其复杂的超高刚性非线性系统。

如果将全厂全部设备（如 7 座深冷同位素分离精馏塔、数台催化反应器、多级冷凝吸附床）的几千个严密偏微分/微分代数方程（DAE）直接在单一 Modelica 模型中强行耦合积分，会导致：
1. **数值刚性发散**：微小时间尺度的液膜流体力学震荡导致 DASSL 步长下溢截断；
2. **计算开销爆炸**：5000 小时全厂运行仿真可能需要计算数周；
3. **跨领域工具割裂**：化工精馏（h2iso / Aspen）、等离子体物理、核反应传热模型分属不同成熟求解器，难以统一重写为 Modelica 纯方程。

为此，TRICYS 建立了**统一的子系统替换与资产重连（Submodel Replacement & Reconnection）**抽象框架，并将联合仿真凝练为三种互补的物理实现模式：

```
                                  [TRICYS 全系统级模型 (example_model.Cycle)]
                                                      |
                  +-----------------------------------+-----------------------------------+
                  |                                   |                                   |
                  v                                   v                                   v
        【模式 1: 离线查表】                 【模式 2: 在线步进】                 【模式 3: 参数注入】
         (offline_table)                      (online_step)                     (parameter_injection)
                  |                                   |                                   |
                  v                                   v                                   v
         [静态响应面 / CSV 网格]            [FMU / SSP / 双向在线步进]            [0-D 动态代理 + h2iso 严密求解]
                  |                                   |                                   |
       - 机制: CombiTable 插值             - 机制: Lockstep 宏步长同步         - 机制: -overrideFile 原生注入
       - 速度: 极高 (纳秒级)                - 速度: 低 (受通信与外接工具制约)    - 速度: 极高 (40s 跑完 5000h)
       - 精度: 稳态/拟稳态静态响应          - 精度: 完整捕获双向强动态反馈       - 精度: 严密 MESH 解 + 宏观质量延迟
       - 典型: 静态物性/稳态分离器          - 典型: 动态扰动/快速瞬态回路        - 典型: 深冷精馏多塔流程 (Wang 2022)
```

---

## 📊 2. 三种联合仿真模式本质对比 (The Three Modalities)

| 特征维度 | 模式 1: 离线查表 (`offline_table` / `offline_cosim`) | 模式 2: 在线步进 (`online_step` / `online_oms`) | 模式 3: 参数注入 (`parameter_injection` / `param_inject`) |
| :--- | :--- | :--- | :--- |
| **底层仿真引擎** | **`offline_table`**<br>(文件: `offline_table.py`，注册: `offline_table` / `offline_cosim`) | **`online_step`**<br>(文件: `online_step.py`，注册: `online_step` / `online_oms` / `online_cosim`) | **`param_inject`**<br>(文件: `param_inject.py`，注册: `param_inject` / `parameter_injection`) |
| **数学物理本质** | **代数插值降阶**：将子系统离线计算为静态多维时序响应数据，解耦微观动力学。 | **时间解耦锁步积分**：主系统与外部处理器在通信宏步长（`stepSize`）网格上同步推进并双向交换流股。 | **时标分离（Time-Scale Separation）**：微观快速稳态解与宏观储氚动态解算分离。 |
| **对原模型的替换机制** | **内部拦截器自动生成**：`interceptor.py` 内部自动生成包装模型（内置 `CombiTimeTable`），**严禁用户在配置中手动写 `substitute_model`！** | **SSP/FMU 资产重连**：基于 OMSimulator 在 `.ssp` 模型接口上动态双向变量赋值（Set/Get）。 | **显式子模型替换**：由用户在配置中显式指定 `substitute_model` 与 `substitute_file`，由 `submodel_replacer.py` 强校验后无缝替换。 |
| **外接求解器调用时机** | **两阶段中间调用**：第一阶段初探生成 `primary_inputs.csv`，调用 Handler 函数生成输出 CSV。 | **仿真运行时逐时间步动态交互**：主循环每个通信步长边界调用 Handler 类的 `process()`。 | **仿真前置任务阶段单次严密求解**：由独立 Handler 脚本真实求解生成 `override_cycle.txt`。 |
| **参数/数据注入机制** | `interceptor.py` 动态构建的 `CombiTimeTable` 读取生成的输出 CSV。 | 内存强类型 `UnifiedStateVector` 与 `TrackResult` 实时交互。 | OpenModelica 命令行原生 `-overrideFile override_cycle.txt` 指令启动。 |
| **全厂仿真耗时 (5000h)**| < 10 秒 | 数小时 ~ 数天 (受跨进程通信时延约束) | **20 ~ 40 秒 (兼具高精度机理与极速长周期)** |
| **适用物理对象** | 稳态分离比相对固定的吸附床、换热器、静态产率映射。 | 等离子体快速破裂瞬态、存在强烈高频控制反馈的回路。 | **CFETR ISS 氢同位素分离系统多塔串联流程（Wang 2022）、大型催化平衡器。** |

---

## 🔒 3. 子系统替换与拦截机制对比 (Replacement vs. Interception vs. Reconnection)

在 TRICYS 中，三种模式在模型层面的改造逻辑完全不同：
1. **模式 1 (`offline_cosim`)**：用户**无需且严禁**在 YAML 配置文件中指定 `substitute_model`。TRICYS 的两阶段机制由底层的 `interceptor.py`（`integrate_interceptor_model`）在内存和临时目录中自动为目标子组件生成 `_Interceptor` 包装模型，将 Handler 生成的 CSV 数据自动注入到内部 `CombiTimeTable` 中，用户只需声明 `target`, `instance`, `solver`, `handler_module`, `handler_function`, `params`。
2. **模式 2 (`online_oms`)**：模型通常为 `.ssp` 系统包。TRICYS 依托 OMSimulator 在宏步长边界通过 FMI 总线进行在线变量双向同步，用户声明 `handler_module` 与 `handler_function`（指定继承自 `AbstractTrackProcessor` 的类名）。
3. **模式 3 (`param_inject`)**：**全框架中唯一需要用户显式声明 `substitute_model` 和 `substitute_file` 的模式**！因为该模式需要将原微分方程组件替换为包含宏观质量守恒与水力学延迟 $\tau$ 的 0-D 动态代理组件（如 `Generic_ISS_Adapters.mo`），因此必须通过 [tricys.core.submodel_replacer](file:///d:/Administrator/Documents/Github/tricys/tricys/core/submodel_replacer.py) 执行**绝对零悬空的拓扑一致性强校验**（端口名、因果方向、向量维度、未连接端口默认零绑定方程）。对于全厂宏观同位素流（如 5 维/3 维原子流）与微观化工流（如 6 维分子流）之间的跨维度升降维，既可以在外部 Handler 求解程序中进行化学同位素平衡换算与映射，也可以通过 Modelica 专用适配器组件（如 `Generic_ISS_Adapters.ISS_I_Adapter`）在模型层完成拓扑升降维桥接。

---

## 📝 4. 模式详解与 Schema 2.0 配置规范

### 4.1 模式 1: 离线查表模式 (`offline_table` / `offline_cosim`)
适用于将外部子系统通过两阶段初探-重构机制进行离线查表替代的场景。底层对应 **`offline_cosim`** 引擎。
> ⚠️ **关键纪律**：严禁在 `subsystems` 中配置 `substitute_model` 或 `table_file`！拦截器模型由 TRICYS 内部自动生成。

```yaml
version: "2.0"
model:
  fileName: "../example_model/package.mo"
  modelName: "example_model.Cycle"

scenario:
  startTime: 0.0
  stopTime: 100.0
  stepSize: 0.5
  method: "dassl"
  variableFilter: "time|sds.I[1]"

task:
  type: "cosim"
  mode: "offline_table"                                 # 底层对应 offline_table 引擎 (兼容 offline_cosim)
  subsystems:
    - target: "example_model.I_ISS"                     # 待拦截替换的子系统组件名
      instance: "i_iss"                                 # 在主系统中的实例名
      solver: "dwsim"                                   # 外部求解器类型 (例如: dwsim, aspen_plus, python)
      handler_script_path: "./08_cosim_offline_table/dwsim_iss_handler.py" # 处理器脚本路径 (或 legacy: tricys.example.example_data.v1_examples_legacy.handlers.i_iss_handler)
      handler_function: "run_dwsim_simulation"         # 处理器函数名
      params:                                           # 传递给 handler_function 的自定义参数
        flowsheet_path: "../example_dwsim/T2_Threetowers4.dwxmz"
        base: 20.0
        retime: 60
        time_step: 3
```

---

### 4.2 模式 2: 在线步进模式 (`online_step` / `online_oms`)
适用于主系统与外部 FMU / 子系统在每个通信宏步长（`stepSize`）双向动态交互。底层对应 **`online_oms`** 引擎。
> 💡 **关键规范**：模型文件通常为 `.ssp` 包；`handler_function` 必须是继承自 `AbstractTrackProcessor` 的类名。

```yaml
version: "2.0"
model:
  fileName: "../example_model_ssp/ssp/example_model.ssp"
  modelName: "example_model.Cycle"

scenario:
  startTime: 0.0
  stopTime: 100.0
  stepSize: 0.5                                         # 宏观数据交换通信步长
  method: "dassl"
  variableFilter: "time|.*\\.I.*|.*\\.cumulative_.*"

task:
  type: "cosim"
  mode: "online_step"                                   # 或 "online_cosim" / "fmi_online"，底层对应 online_oms 引擎
  subsystems:
    - target: "CFEDR.I_ISS"
      instance: "i_iss"
      solver: "aspen_plus"                              # 或 "fmu"
      handler_module: "tricys.example.example_data.basic.9_online_cosim.i_iss_processor_base"
      handler_function: "I_ISSProcessorBase"            # 继承 AbstractTrackProcessor 的类名
      params:
        bkp_path: "../example_aspenbkp/T2-Threetowers4.bkp"
        base_inventory_mol: 20.0
        retime_h: 1.0
      local_mass_control:                               # 可选: 局部质量控制闭环
        enabled: true
        deadband_g: 0.1
        relaxation_time_h: 1.0
```

---

### 4.2 模式 2: 在线步进模式 (`online_step`)
适用于主系统与外部 FMU / 子系统在每个通信宏步长（`stepSize`）双向动态交互。

```yaml
version: "2.0"
model:
  fileName: "example_model.ssp"
  modelName: "example_model.Cycle"

scenario:
  startTime: 0.0
  stopTime: 500.0
  stepSize: 0.5                                 # 通信宏步长
  method: "dassl"

task:
  type: "cosim"
  mode: "online_step"
  subsystems:
    - target: "CFEDR.I_ISS"
      instance: "i_iss"
      solver: "aspen_plus"                      # 或 "fmu"
      handler_module: "handlers.i_iss_step_processor" # 在线步进处理类
      handler_function: "I_ISSStepProcessor"
      params:
        bkp_path: "aspen_models/ISS_column.bkp"
        base_inventory_mol: 20.0
      local_mass_control:                       # 局部质量控制闭环
        enabled: true
        deadband_g: 0.05
```

---

### 4.3 模式 3: 参数注入模式 (`parameter_injection`)
**聚变堆深冷精馏与长周期多物理场协同仿真推荐金标准**。
- **任务级统一原则**：`injection_type: "handler" | "file"`，`override_file` 与 `handler_script` 提升至全局任务级，避免在各 subsystem 中分散配置。
- **真实物理驱动**：前置调用 Handler 执行真实外部计算（如 `h2iso`），生成标准 Modelica 参数覆盖文本。

```yaml
version: "2.0"
model:
  fileName: "./10_cosim_parameter_injection/example_model.mo"
  modelName: "example_model.Cycle"

scenario:
  time_base: "hour"
  stopTime: 5000.0                              # 支持超长周期 (5000h+) 演化
  stepSize: 1.0
  method: "dassl"
  variableFilter: "time|sds.I[1]|blanket.I[1]|i_iss.to_SDS[1]|o_iss.to_SDS[1]"

task:
  type: "cosim"
  mode: "parameter_injection"
  injection_type: "handler"                     # 统一注入方式: "handler" (脚本生成) 或 "file" (直接文件)
  override_file: "./10_cosim_parameter_injection/override_cycle.txt" # 目标参数覆盖文件
  handler_script: "./10_cosim_parameter_injection/run_h2iso_wang2022.py" # 任务级外部求解器 Handler
  handler_params:
    wang2022_config_issi: "./10_cosim_parameter_injection/wang2022_issi.json"
    wang2022_config_isso: "./10_cosim_parameter_injection/wang2022_isso.json"
    solve_mode: "live"                          # "live" 真实求解 h2iso 严密 MESH
    method: "wegstein"                          # 撕裂流加速迭代方法
    max_iter: 10
    tol: 0.01
    continuation_substeps: 2
  subsystems:
    - target: "example_model.I_ISS"             # 待替换子组件 1
      instance: "i_iss"
      solver: "h2iso"
      substitute_model: "Generic_ISS_Adapters.ISS_I_Adapter"
      substitute_file: "./10_cosim_parameter_injection/Generic_ISS_Adapters.mo"
      aux_files:
        - "./10_cosim_parameter_injection/Generic_ISS.mo"
    - target: "example_model.O_ISS"             # 待替换子组件 2
      instance: "o_iss"
      solver: "h2iso"
      substitute_model: "Generic_ISS_Adapters.ISS_O_Adapter"
      substitute_file: "./10_cosim_parameter_injection/Generic_ISS_Adapters.mo"
      aux_files:
        - "./10_cosim_parameter_injection/Generic_ISS.mo"
```

---

---

## 🛠️ 5. 三大联合仿真范式的 Handler 编写规范与实现对比

在 TRICYS 中，针对三种不同的时空尺度解耦机制，**外部 Handler 的软件形态、接口签名、调用时序与数据流完全不同**。Agent 必须严格按照目标协同仿真模式选择对应的 Handler 编写范式：

| 范式维度 | 范式 1: 离线查表 (`offline_table`) | 范式 2: 在线步进 (`online_step`) | 范式 3: 参数注入 (`parameter_injection`) |
| :--- | :--- | :--- | :--- |
| **软件载体形态** | **模块函数 (Module Function)** | **有状态处理器类 (Stateful Class)** | **独立执行脚本 (CLI Script)** |
| **配置声明位置** | `subsystems[].handler_module`<br>`subsystems[].handler_function` | `subsystems[].handler_module`<br>`subsystems[].handler_function` | `task.handler_script`<br>`task.handler_params` (全局同级) |
| **核心接口契约** | `def func(temp_input_csv, temp_output_csv, **params) -> dict` | 继承 `AbstractTrackProcessor`<br>`def process(request_vector) -> TrackResult` | 命令行入口 `def main() -> int`<br>读环境变量 `TRICYS_WORKSPACE` |
| **数据通信载体** | **CSV 磁盘文件**（全时序离线数据） | **内存强类型向量**（`UnifiedStateVector`） | **参数覆盖文本**（`override_cycle.txt`） |
| **执行时序** | 两阶段初探-重构 (Run-Process-Rerun) | 仿真期按宏步长锁步推进 (Lockstep) | 仿真前置单次严密求解 + 原生注入 |

---

### 5.1 范式一：离线查表 (Offline Table) Handler 编写规范

#### 1. 软件架构与执行机制
离线查表采用 **Run-Process-Rerun (两阶段初探-拦截-重构)** 机制：
1. **阶段 1 (初探)**：TRICYS 驱动全厂模型进行基准试跑，提取目标子系统上下游的边界输入流股时序，落盘为 `temp_input_csv`；
2. **阶段 2 (外接求解)**：TRICYS 动态导入 `handler_module` 并调用 `handler_function`，将 `temp_input_csv` 传给求解器逐点计算，求解器将响应时序写入 `temp_output_csv`，并返回 Modelica 参数占位字典；
3. **阶段 3 (重构)**：TRICYS 内部通过 `interceptor.py`（`integrate_interceptor_model`）动态自动生成 `_Interceptor` 拦截包装模型（内部自动挂载 `CombiTimeTable`），无缝拦截重连原系统实例后启动第二阶段闭环重算。**用户在 YAML 配置文件中严禁且无需指定任何 `substitute_model`**。

#### 2. 函数签名与参数说明
```python
def handler_function(
    temp_input_csv: str,
    temp_output_csv: str,
    **params: Any,
) -> dict[str, str]:
    """离线查表标准处理函数接口.
    
    Args:
        temp_input_csv: 初探仿真生成的输入时序 CSV 文件路径 (包含 Time 及各输入流股)
        temp_output_csv: 处理器必须生成的输出时序 CSV 文件路径 (包含 Time 及各输出流股)
        **params: YAML 配置中 subsystems[].params 传入的自定义参数字典 (如 bkp_path, base 等)
        
    Returns:
        output_placeholder: 供 Modelica 替换组件使用的初始常数映射字典 (如 {"to_SDS": "{1,2,3}"})
    """
```

#### 3. 生产级参考实现范例 (参考 `08_cosim_offline_table/dwsim_iss_handler.py` 或 `v1_examples_legacy/handlers/i_iss_handler.py`)
```python
import logging
import pandas as pd

logger = logging.getLogger("offline_table_handler")

def run_iss_offline_table(
    temp_input_csv: str,
    temp_output_csv: str,
    time_step_h: float = 1.0,
    **kwargs,
) -> dict:
    """根据初探输入流股批量计算同位素分离响应并生成输出查表 CSV."""
    logger.info(f"Loading primary simulation input: {temp_input_csv}")
    input_df = pd.read_csv(temp_input_csv)
    
    # 1. 提取输入时序 (如来自 TEP 的三元同位素原子质量流)
    time_series = input_df["time"]
    # 假设输入列包含 i_iss.from_TEP[1] (T), i_iss.from_TEP[2] (D), i_iss.from_TEP[3] (H)
    t_flow = input_df.get("i_iss.from_TEP[1]", pd.Series([0.0]*len(time_series)))
    d_flow = input_df.get("i_iss.from_TEP[2]", pd.Series([0.0]*len(time_series)))
    
    # 2. 批量驱动外部专业工具 (如 Aspen Plus COM / DWSIM / 本地机理模型) 计算
    results = []
    for t_val, f_t, f_d in zip(time_series, t_flow, d_flow):
        # 稳态或拟稳态分离计算 (分离纯度: 99% T2 到 SDS, 杂质排入 WDS)
        sds_t = f_t * 0.99
        wds_t = f_t * 0.01
        results.append({
            "Time": t_val,
            "to_SDS[1]": sds_t,
            "to_SDS[2]": f_d * 0.95,
            "to_WDS[1]": wds_t,
            "to_WDS[2]": f_d * 0.05,
        })
        
    out_df = pd.DataFrame(results)
    
    # 3. 必须将输出结果写入 temp_output_csv 供 Modelica CombiTable 读取
    out_df.to_csv(temp_output_csv, index=False)
    logger.info(f"Saved lookup table results to: {temp_output_csv} ({len(out_df)} rows)")
    
    # 4. 返回初始常数字典
    output_placeholder = {
        "to_SDS": "{0,0,0,0,0}",
        "to_WDS": "{0,0,0,0,0}",
    }
    return output_placeholder
```

---

### 5.2 范式二：在线步进 (Online Step) Handler 编写规范

#### 1. 软件架构与执行机制
在线步进采用 **Lockstep Dynamic Exchange (双向锁步动态交换)** 机制：
1. **时钟同步**：系统仿真主进程与外部处理器在通信宏步长（`stepSize`）边界上推进；
2. **实例生命周期**：TRICYS 导入 `handler_module` 并实例化 `handler_function` 指定的类，在仿真启动时调用 `initialize(context)`；
3. **单步流股交换**：在每个宏步长 $t \to t + \Delta t$：
   - TRICYS 将当前步的输入端口流股打包为 `UnifiedStateVector` 传入 `process()`；
   - 处理器内部更新内部状态（如积存动态、衰变吸收），并可调用外接工具动态单步；
   - 处理器返回 `TrackResult(outputs=...)`，TRICYS 将其反馈给全厂 Modelica 模型推进下一步。

#### 2. 类接口契约与方法签名
必须继承自 `tricys.online_cosim.processor_base.AbstractTrackProcessor`：
```python
from tricys.online_cosim.processor_base import AbstractTrackProcessor
from tricys.online_cosim.schema import (
    TrackProcessorContext,
    TrackResult,
    UnifiedStateVector,
)

class CustomStepProcessor(AbstractTrackProcessor):
    def __init__(self, **params: Any) -> None:
        """接收 YAML 中 subsystems[].params 字典."""
        pass

    def initialize(self, context: TrackProcessorContext) -> None:
        """仿真启动前回调，用于重置内部状态、滞留量与队列."""
        pass

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        """核心单步推进接口.
        
        Args:
            request_vector:
                - current_time_h (float): 当前仿真时间 (小时)
                - dt_slow_h (float): 宏观通信步长 (小时)
                - boundary_inputs (dict): 输入端口数据字典 (如 {"from_TEP": [T, D, H, 0, 0]})
                - extra_state (dict): 上下文额外状态
                
        Returns:
            TrackResult:
                - outputs (dict): 输出端口流股字典 (如 {"to_SDS": [...], "to_WDS": [...]})
        """
        pass
```

#### 3. 生产级参考实现范例 (参考 `i_iss_processor_base.py`)
```python
import logging
from tricys.online_cosim.processor_base import AbstractTrackProcessor
from tricys.online_cosim.schema import (
    TrackProcessorContext,
    TrackResult,
    UnifiedStateVector,
)

logger = logging.getLogger("online_step_processor")

class ISSStepProcessor(AbstractTrackProcessor):
    def __init__(self, base_inventory_g: float = 50.0, tau_delay_h: float = 0.5, **kwargs):
        self.base_inventory_g = float(base_inventory_g)
        self.tau_delay_h = float(tau_delay_h)
        self.current_inventory_g = 0.0

    def initialize(self, context: TrackProcessorContext) -> None:
        """仿真开始前初始化内部状态."""
        self.current_inventory_g = 0.0
        logger.info("ISSStepProcessor initialized.")

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        current_t = request_vector.current_time_h
        dt = request_vector.dt_slow_h
        
        # 1. 获取输入边界流股 (5 维同位素原子质量流量 g/h)
        in_flow = request_vector.boundary_inputs.get("from_TEP_FCU", [0.0]*5)
        t_in_rate = float(in_flow[0]) if len(in_flow) > 0 else 0.0
        
        # 2. 内部动力学演化 (积分滞留量并计算放射性衰变)
        decay_constant = 6.4e-6 * 3600.0 # 1/h
        self.current_inventory_g += t_in_rate * dt
        self.current_inventory_g *= (1.0 - decay_constant * dt)
        
        # 3. 门限释放或分离动态计算
        if self.current_inventory_g > self.base_inventory_g:
            excess = self.current_inventory_g - self.base_inventory_g
            # 一阶惯性延迟流出
            out_rate = excess / max(self.tau_delay_h, 1e-4)
            self.current_inventory_g -= out_rate * dt
        else:
            out_rate = 0.0
            
        # 4. 组装输出端口流股并返回强类型 TrackResult
        out_sds = [out_rate * 0.99, 0.0, 0.0, 0.0, 0.0]
        out_wds = [out_rate * 0.01, 0.0, 0.0, 0.0, 0.0]
        
        return TrackResult(outputs={
            "to_SDS": out_sds,
            "to_WDS": out_wds,
        })
```

---

### 5.3 范式三：参数注入 (Parameter Injection) Handler 编写规范

#### 1. 软件架构与执行机制
参数注入采用 **Pre-Solve & Native Override (-overrideFile)** 机制：
1. **任务级独立子进程驱动**：TRICYS 前置通过独立子进程运行 `task.handler_script`；
2. **环境变量上下文注入**：TRICYS 自动注入 `TRICYS_WORKSPACE` 与 `TRICYS_HANDLER_PARAMS`；
3. **真实高保真稳态解算**：Handler 真实调用外部专业求解器（如 `h2iso` 严密 MESH 方程求解器），求解多塔串联流程与循环撕裂流；
4. **生成层次覆盖参数**：Handler 将各塔塔顶/塔底分离比、摩尔流率、温度剖面与滞留时间 $\tau$ 写入 `override_cycle.txt`；
5. **强拓扑校验与模型替换**：TRICYS 审查端口因果性并无缝替换为 0-D 动态代理；
6. **极速仿真执行**：OpenModelica 原生加载 `-overrideFile override_cycle.txt` 秒级启动并完成长周期积分。

#### 2. 通信契约与环境变量
| 环境变量名 | 类型 | 说明与规范 |
| :--- | :--- | :--- |
| **`TRICYS_WORKSPACE`** | 字符串 (路径) | 当前仿真输出隔离目录绝对路径。Handler 必须在此目录下输出 `override_cycle.txt`。 |
| **`TRICYS_HANDLER_PARAMS`**| 字符串 (JSON) | YAML 中 `task.handler_params` 的 JSON 序列化字符串。由 Handler 解码后作为求解配置。 |
| **退出码 (Exit Code)** | 整数 | 成功执行必须返回 `0`；发生不可恢复错误返回 `1` 并记录详细错误日志。 |

#### 3. 生产级参考实现范例 (参考 `run_h2iso_wang2022.py`)
```python
#!/usr/bin/env python3
"""Standard Parameter Injection External Solver Handler (h2iso Integration)."""

import argparse
import json
import logging
import os
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][h2iso_handler] %(levelname)s: %(message)s")
logger = logging.getLogger("h2iso_handler")

def main() -> int:
    # 1. 从环境变量获取隔离工作区与配置参数
    workspace = Path(os.environ.get("TRICYS_WORKSPACE", ".")).resolve()
    params_json = os.environ.get("TRICYS_HANDLER_PARAMS", "{}")
    params = json.loads(params_json) if params_json else {}

    logger.info(f"Target workspace: {workspace}")
    logger.info(f"Handler parameters: {params}")

    solve_mode = params.get("solve_mode", "live")
    max_iter = int(params.get("max_iter", 15))
    tol = float(params.get("tol", 1e-3))

    # 2. 真实调用专业物理求解器 (以 h2iso 严密 MESH 流程求解器为例)
    try:
        import h2iso
        from h2iso.flowsheet.schema import load_flowsheet
        from h2iso.flowsheet.solver import SequentialModularSolver
        from h2iso.codegen.modelica_0d import generate_cycle_override_file

        logger.info("Solving Wang 2022 cryogenic distillation flowsheets with h2iso...")
        
        # 示例: 加载 ISS-O 3塔+1平衡器+2撕裂流流程
        script_dir = Path(__file__).parent
        isso_file = script_dir / params.get("wang2022_config_isso", "wang2022_isso.json")
        isso_cfg = load_flowsheet(isso_file)
        isso_solver = SequentialModularSolver(isso_cfg, method="wegstein", continuation_substeps=2)
        isso_res = isso_solver.solve(max_iter=max_iter, tol=tol)
        logger.info(f"ISS-O solve finished: converged={isso_res.converged}, iters={isso_res.iterations}")

        # 3. 格式化生成 OpenModelica 层次参数覆盖内容 (必须包含完整实例名前缀)
        # 格式必须严格对应 Modelica 模型树: <instance>.<subsystem>.<parameter>=<value>
        override_lines = [
            "// Auto-generated full-cycle override parameters by h2iso",
            "// Target: example_model.Cycle with Generic_ISS_Adapters",
            "o_iss.core.CD1.T_top=20.0598",
            "o_iss.core.CD1.T_bottom=22.6937",
            "o_iss.core.CD1.tau=1.000000",
            "o_iss.core.CD1.m_top_ref[1]=8.84318996e+02",
            "o_iss.core.CD1.m_bottom_ref[1]=0.00000000e+00",
            # ... 其余各塔及各组分参数
        ]
        override_content = "\n".join(override_lines)

    except ImportError:
        logger.warning("h2iso package not available, using fallback parameters")
        override_content = "// Fallback parameters\n"

    # 4. 必须将 override_cycle.txt 写入 workspace 供 OpenModelica 启动时直接加载
    target_override_file = workspace / "override_cycle.txt"
    target_override_file.write_text(override_content, encoding="utf-8")
    logger.info(f"Successfully generated override file: {target_override_file} ({len(override_content.splitlines())} lines)")

    # 5. 返回状态码 0
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

---

### 5.4 三大 Handler 编写范式速查对比表

| 维度 | 范式 1: 离线查表 (`offline_table`) | 范式 2: 在线步进 (`online_step`) | 范式 3: 参数注入 (`parameter_injection`) |
| :--- | :--- | :--- | :--- |
| **编程范式** | **函数式** (Function-based) | **面向对象状态机** (OOP State-Machine) | **脚本/CLI 驱动** (Process-based) |
| **入口名称** | `run_aspen_simulation` 等自定义函数名 | `I_ISSProcessorBase` 等继承自基类的类名 | `main()` 函数与 `sys.exit(...)` |
| **上下文传递** | 函数入参：`temp_input_csv, temp_output_csv, **params` | 构造函数 `__init__(**params)` + `process(request_vector)` | 环境变量 `TRICYS_WORKSPACE` + `TRICYS_HANDLER_PARAMS` |
| **输入流获取** | 从 `temp_input_csv` 读取整段时间序列 | 从 `request_vector.boundary_inputs` 实时读取单步流股 | 从 JSON 配置文件或初始稳态工况输入 |
| **输出结果提交** | 写入 `temp_output_csv` 并返回占位字典 | 返回强类型 `TrackResult(outputs={...})` | 写入 `${WORKSPACE}/override_cycle.txt` |
| **子组件替换机制** | 框架 `interceptor.py` 内部自动生成包装模型（用户**无需配置**） | OMSimulator 在 `.ssp`/FMU 接口在线步进重连 | 用户**显式指定** `substitute_model` 替换为 0-D 动态代理 |
| **仿真调用方式** | 两阶段初探与重跑 | 主循环中逐时间步双向回调 | 一次编译后 `-overrideFile` 极速长周期执行 |


---

## 🔍 6. 联合仿真故障排查与自愈矩阵 (Troubleshooting & Self-Healing Matrix)

| 异常现象 / 错误日志 | 根本物理与软件机理 | 自愈诊断与修复方案 (Agent Action) |
| :--- | :--- | :--- |
| `PortCompatibilityError: Mismatched ports` | 替换组件与原组件的进出口端口名或因果方向不匹配。 | 运行 `tricys model inspect` 比对原模型与替换模型的端口列表，确保 `RealInput` 与 `RealOutput` 语义一致。 |
| `Dimension mismatch: Port 'xxx' dim 5 vs 6` | 尝试用 6 维分子模型直接替换 5 维原子模型，维度不兼容。 | 将 `substitute_model` 更改为包含维度映射的适配器组件（如 `Generic_ISS_Adapters.ISS_I_Adapter`）。 |
| `Under-determined system (equations < variables)` | 替换组件引入了未连接的多余输入端口，造成自由度悬空。 | 使用 TRICYS 内置的 `submodel_replacer`，其已支持自动零绑定未连接端口；或在 Modelica 中为端口显式赋值。 |
| `Parameter injection handler failed (Exit 1)` | 外部求解器脚本在执行中抛出异常（如收敛发散、依赖缺失）。 | 检查 `TRICYS_HANDLER_PARAMS` 中的容差 `tol` 与最大迭代步数 `max_iter`，放宽收敛容差或开启延拓步数 `continuation_substeps: 2`。 |
| `Warning: override variable 'xxx' not found in model` | `override_cycle.txt` 中的参数层级路径与替换后模型实例不匹配。 | 确认 Modelica 参数名前缀。例如实例名为 `o_iss`，核心模型为 `core`，塔名为 `CD1`，则前缀必须为 `o_iss.core.CD1.xxx`。 |
