# 自定义工况配置

TRICYS 现在支持使用 FOC（Fusion Operation Config）描述聚变装置的自定义工况，例如脉冲、停机、分段调度和循环重复。该能力适合把“运行工况”与“模型参数”分离：模型保持不变，运行工况由独立的 FOC 输入控制。

当前仓库中的示例入口位于：

- `tricys/example/example_data/basic/5_fusion_operation_config/fusion_operation_config.json`
- `tricys/example/example_data/example_foc/example_scenario_mix.foc`

---

## 1. 示例配置

当前推荐的文件驱动示例如下：

```json
{
    "paths": {
        "package_path": "../../example_model_single/example_model.mo"
    },
    "foc": {
        "foc_path": "../../example_foc/example_scenario_mix.foc",
        "foc_component": "pulseSource"
    },
    "simulation": {
        "model_name": "example_model.Cycle",
        "variableFilter": "time|sds.I[1]",
        "stop_time": 1200.0,
        "step_size": 0.5
    }
}
```

对应的 FOC 文件示例如下：

```text
# 测试混合工况
# 当前 FOC 以小时编写，并显式换算到模型默认秒单位
TIME_UNIT hour
TIME_CONVERSION hour_to_second

# 首先是一个短脉冲循环 (功率 1500 MW, 燃烧 0.9h, 停堆 0.1, 600次循环)
PULSE 1500 0.9 0.1 300

# 大停机 100h
DWELL 100

# 然后进行调度块 (长短混合)
BEGIN_SCHEDULE
  PULSE 2000 0.9 0.1 100
  POWER 1000
  BURN 200
  DWELL 100
END_SCHEDULE
REPEAT 2
```

### 1.1. FOC 时间头语法

如果 FOC 文件中的时间单位与模型内部时间基准不一致，或者你希望显式声明当前 FOC 使用的时间单位，可以在 `.foc` 文件顶部加入时间头指令：

```text
TIME_UNIT <second|hour|day|week|year>
TIME_CONVERSION <factor>
TIME_CONVERSION <source_unit>_to_<target_unit>
```

常见写法如下：

- 当 `TIME_UNIT` 和 `TIME_CONVERSION` 都未出现时，默认 FOC 时间单位为秒，且不做转换
- `TIME_UNIT hour`：仅声明当前 FOC 中的时间按小时书写，本身不触发转换
- FOC 的**有效**时间基准（经过 `TIME_CONVERSION` 之后）必须是**秒**——CFEDR 系列模型的时间轴 `time` 以秒为单位；若有效基准为小时/天/年（或裸数字因子），程序会发出告警，且调度会以错误速度推进
- `TIME_CONVERSION NONE`：不做额外时间换算，直接使用 FOC 中写入的时间值
- `TIME_CONVERSION 3600`：把后续时间统一乘以 `3600`
- `TIME_CONVERSION second_to_hour`：把以秒写入的 FOC 时间转换为模型所需的小时单位，`hour_to_second` 等反向命名转换同样支持
- 只有显式出现 `TIME_CONVERSION` 时程序才执行时间换算

限制如下：

- 支持的显式时间单位为 `second`、`hour`、`day`、`week`、`year`
- 这些头指令必须出现在 `POWER`、`BURN`、`DWELL`、`PULSE`、`BEGIN_SCHEDULE` 等业务指令之前
- `TIME_UNIT` 和 `TIME_CONVERSION` 各自只能声明一次
- 如果同时声明 `TIME_UNIT` 和命名式 `TIME_CONVERSION`，二者的起始单位必须一致

示例 1：FOC 按小时编写，并显式转换到模型内部默认秒单位

```text
TIME_UNIT hour
TIME_CONVERSION hour_to_second
PULSE 1500 0.9 0.1 300
```

示例 1b：FOC 按小时编写，但仅声明单位、不做转换 —— 对秒基准模型是**错误用法**

```text
TIME_UNIT hour
PULSE 1500 0.9 0.1 300
```

`TIME_UNIT` 只声明单位，**不触发转换**。CFEDR 系列模型的时间轴 `time` 以**秒**为单位，因此停留在小时的调度会快约 3600 倍。当 FOC 的有效时间基准不是秒时 TRICYS 会发出告警；按示例 1 加上 `TIME_CONVERSION hour_to_second` 即可修正。仅当你写入的时间本来就以模型时间基准（秒）为单位时，才使用这种只声明单位的写法。

示例 2：FOC 按秒编写，但模型内部使用小时

```text
TIME_CONVERSION second_to_hour
PULSE 1500 7200 3600 1
```

---

## 2. `foc` 顶层配置说明

FOC 相关字段不再放在 `simulation` 下，而是使用独立的顶层对象 `foc`。

### 2.1. `foc_path`

- 类型：字符串
- 作用：指向 `.foc` 文件
- 推荐：使用相对于配置文件的相对路径

### 2.2. `foc_component`

- 类型：字符串
- 作用：指定需要被 FOC 替换的目标子组件
- 要求：启用 FOC 时必须提供
- 当前支持：实例名、组件路径、或组件类型过滤

!!! warning "必须显式指定 foc_component"
    即使模型中只识别出一个 pulse-like 子组件，运行配置里仍应保存 `foc_component`，避免模型结构变化时产生歧义。

---

## 3. 输入方式

### 3.1. 文件方式：`foc_path`

这是当前文档示例采用的主方式。优点是：

- 运行工况可以单独复用和版本化
- 同一个模型可以快速切换多套 FOC 工况
- 命令行、示例库和基础配置都走这条路径

---

## 4. 适用场景

- 脉冲工况与长停机工况混合调度
- 不同运行策略下的库存响应对比
- 在不改 Modelica 主模型的前提下切换运行节奏
- 用示例库维护标准工况

---

## 5. 与基础配置的关系

自定义工况配置是在[基础配置](basic_configuration.md)上增加一个顶层 `foc` 块，而不是替代 `simulation`。

- `simulation` 决定模型、输出变量、总时长和步长
- `foc` 决定 pulse-like 组件如何被新的运行工况驱动

如果 `simulation.stop_time` 小于 FOC 调度总时长，TRICYS 会发出截断警告。

---

## 6. 下一步

- 如果你想了解这套机制为什么采用“顶层 `foc` + 工作区落盘”的结构，请继续阅读[自定义工况原理](../../explanation/tricys_basic/fusion_operation_principle.md)。
- 如果你想从 CLI 或 GUI 使用该示例，可以直接从 `tricys/example/example_data/basic/example_runner.json` 中选择对应示例。