# 参数扫描

参数扫描是 `tricys` 的核心功能之一，它允许您系统地研究一个或多个模型参数的变化对仿真结果的影响。您只需为每个感兴趣的参数提供一组值，`tricys` 会自动创建并运行所有可能的参数组合。

## 1. 配置文件示例

在基础配置之上，我们只需添加一个 `simulation_parameters` 字段，即可定义参数扫描。

```json
{
    "paths": {
        "package_path": "../../example_model_single/example_model.mo"
    },
    "simulation": {
        "model_name": "example_model.Cycle",
        "variableFilter": "time|sds.I[1]",
        "stop_time": 2000.0,
        "step_size": 0.5
    },
    "simulation_parameters": {
        "tep_fep.to_SDS_Fraction[1]": [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8],
        "blanket.TBR": "linspace:1.05:1.15:3"
    }
}
```

## 2. 配置项详解

- **描述**: `simulation_parameters`配置项 是一个字典（键值对集合），用于定义要扫描的参数及其对应的值。
- **键 (Key)**: 必须是 Modelica 模型中变量的**完整路径**。例如 `blanket.TBR` 或 `tep_fep.to_SDS_Fraction[1]`。
- **值 (Value)**: 可以是以下几种格式：
    1.  **离散值列表 (List)**:
        -   **格式**: `[v1, v2, v3, ...]`
        -   **示例**: `[0.1, 0.15, 0.2]`
        -   **说明**: 程序将依次使用列表中的每一个值进行仿真。

    2.  **高级扫描格式 (String)**:
        -   **描述**: `tricys` 支持多种紧凑的字符串格式来生成数值序列，非常适合定义线性、对数等序列。
        -   **示例**: `"linspace:1.05:1.15:3"` 表示在 1.05 和 1.15 之间生成 3 个等间距的数值。
        -   **支持的格式**:

| 格式 | 语法 | 描述 |
| :--- | :--- | :--- |
| 线性序列 (Range) | `"start:stop:step"` | 从 `start` 到 `stop` 生成一个步长为 `step` 的等差序列。例如: `"1:5:2"` 会生成 `[1, 3, 5]`。 |
| 线性间隔 (Linspace) | `"linspace:start:stop:num"` | 在 `start` 和 `stop` 之间生成 `num` 个等间距的数值。例如: `"linspace:0:10:3"` 会生成 `[0, 5, 10]`。 |
| 对数间隔 (Logspace) | `"log:start:stop:num"` | 在 `start` 和 `stop` 之间生成 `num` 个对数等距的数值，适合跨数量级的扫描。例如: `"log:1:100:3"` 会生成 `[1, 10, 100]`。 |
| 随机数 (Random) | `"rand:min:max:count"` | 在 `min` 和 `max` 之间生成 `count` 个均匀分布的随机数。例如: `"rand:0:1:2"` 可能会生成 `[0.23, 0.87]`。 |
| 从文件读取 (From File) | `"file:path/to/data.csv:column_name"` | 从指定的 CSV 文件中的 `column_name` 列读取数值作为扫描列表。 |
| 数组展开 (Array Expansion) | `"{val1, val2, ...}"` | 用于一次性设置 Modelica 数组中多个元素的特殊格式。例如，为参数 `my_array` 设置值 `"{10, 25, 50}"`，将会被自动展开为 `my_array[1]=10`, `my_array[2]=25`, `my_array[3]=50`。花括号内的值本身也可以是其他高级格式的字符串。 |

!!! tip "多参数扫描"
    - 您可以同时定义多个参数进行扫描，`tricys` 会计算所有参数值的**笛卡尔积**，生成一个包含所有可能组合的仿真任务列表。
    - 在上面的示例中，`tep_fep.to_SDS_Fraction[1]` 有 7 个值，`blanket.TBR` 有 3 个值，因此程序总共会运行 `7 * 3 = 21` 次仿真。

## 3. 结果输出

对于参数扫描任务，除了每次单独运行的 `simulation_result.csv` 文件外，`tricys` 还会生成一个汇总文件 `sweep_results.csv`，如下：

```
Working Directory/
└── {timestamp}/
    ├── log/        
        └── simulation_{timestamp}.log      # 运行日志
    ├── result/                 
        └── sweep_results.csv               # 仿真结果汇总数据
    └── temp/
        ├── job_1/                      
            └── job_1_simulation_result.csv # 仿真任务一模拟结果
        ├── job_2/                      
            └── job_2_simulation_result.csv # 仿真任务二模拟结果
        └── ......
                                   
```

- **`sweep_results.csv`**:
  - **第一列**: `time`，表示时间轴。
  - **其余列**: 每一列代表一次特定参数组合下的仿真结果。列标题清晰地标明了该次运行所使用的参数及其值，例如 `sds.I[1]&tep_fep.to_SDS_Fraction[1]=0.1&blanket.TBR=1.05`，方便您直接在 CSV 文件中比较不同工况下的结果。

---

## 4. 下一步

掌握了参数扫描后，您可以探索更高级的功能来提升效率和分析深度：

- **[并发运行](concurrent_operation.md)**：学习如何利用多核处理器并行执行大量扫描任务，大幅缩短仿真时间。
- **[后处理模块](post_processing_module.md)**：了解如何自动分析扫描结果，例如计算每个工况下的最大值、平均值或报警次数。
- **[敏感性分析](../tricys_analysis/index.md)**：进行更系统化的参数影响研究，例如 Sobol 全局敏感性分析。

