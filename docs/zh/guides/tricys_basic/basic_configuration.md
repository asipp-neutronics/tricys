# 基础配置

TRICYS 的所有仿真任务都由一个 JSON 配置文件驱动。这个文件详细定义了模型路径、仿真参数、扫描变量以及后处理等所有环节。

本节将介绍一个最基础的配置文件，用于运行单次仿真。掌握基础配置是使用 TRICYS 的第一步。

---

## 1. 配置文件示例

一个最小化的配置文件如下所示，它定义了要运行哪个模型以及如何运行。

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
    }
}
```

这个配置文件将运行一个氚燃料循环模型的仿真，时长为 2000 小时，每 0.5 小时输出一次结果，只保存时间和 SDS 系统氚库存两个变量。

---

## 2. 必须配置项

### 2.1. `paths` (路径配置)

这个部分用于定义所有与文件路径相关的设置。

#### `package_path` (字符串, 必填)

- **描述**: 指向 Modelica 模型的根 `package.mo` 文件。TRICYS 将从这里加载和解析您的模型。
- **路径类型**: 可以是绝对路径，也可以是相对于该 JSON 配置文件所在位置的相对路径。
- **示例**:
  ```json
  "package_path": "C:/Models/example_model/package.mo"  // 绝对路径 (Windows)
  "package_path": "/home/user/models/package.mo"         // 绝对路径 (Linux)
  "package_path": "../models/package.mo"                 // 相对路径
  ```

!!! tip "路径分隔符"
    在 JSON 文件中，Windows 路径可以使用正斜杠 `/` 或双反斜杠 `\\`：
    ```json
    "package_path": "C:/Models/package.mo"      // 推荐
    "package_path": "C:\\Models\\package.mo"    // 也可以
    ```

---

### 2.2. `simulation` (仿真配置)

这个部分包含了运行仿真所需的核心参数。

#### `model_name` (字符串, 必填)

- **描述**: 要仿真的完整 Modelica 模型名称。它遵循 `包名.模型名` 的格式。
- **格式**: `<PackageName>.<ModelName>`
- **示例**:
  ```json
  "model_name": "example_model.Cycle"
  ```
  在这个例子中，`example_model` 是包名（对应 `package.mo`），`Cycle` 是其中定义的具体模型。

!!! warning "模型名称必须完全匹配"
    模型名称区分大小写，必须与 Modelica 文件中定义的名称完全一致。

#### `variableFilter` (字符串, 必填)

- **描述**: 一个用于筛选输出结果的正则表达式。只有名称匹配该表达式的变量才会被保存到最终的 `.csv` 结果文件中。
- **格式**: 使用竖线 `|` 分隔多个变量名或模式。
- **示例**:
  ```json
  // 只保存时间和一个变量
  "variableFilter": "time|sds.I[1]"
  
  // 保存时间和一个数组变量
  "variableFilter": "time|sds.I[1-5]"

  // 保存多个特定变量
  "variableFilter": "time|sds.I[1]|blanket.I[1-5]|div.I[1-5]"
  
  ```

!!! tip "建议"
    为了减小输出文件大小和提高性能，建议只保存您真正需要分析的变量。

#### `stop_time` (浮点数, 必填)

- **描述**: 仿真的总时长（单位：秒）。仿真将从 `0` 时刻运行到 `stop_time`。
- **单位**: 小时

#### `step_size` (浮点数, 必填)

- **描述**: 仿真的时间步长（单位：秒）。这也是结果输出的时间间隔。
- **权衡**: 
  - 较小的步长：精度更高，但仿真时间更长，输出文件更大
  - 较大的步长：速度更快，但可能丢失快速变化的细节

---

## 3. 默认配置项

除了上述必填项，TRICYS 还提供了一系列可选配置，它们拥有合理的默认值。不设置这些选项时，系统将自动使用以下默认行为：

### 3.1. `paths` (路径配置)

| 参数 (Parameter) | 描述 (Description) | 默认值 (Default Value) |
| :--- | :--- | :--- |
| `results_dir` | 用于存放仿真结果的目录名。 | `"results"` |
| `temp_dir` | 用于存放临时文件的目录名。 | `"temp"` |
| `log_dir` | 用于存放日志文件的目录名。 | `"log"` |
| `db_path` | 用于存储和读取模型参数的 SQLite 数据库文件路径。 | 在每次运行时动态创建于临时目录中。 |

!!! info "输出目录结构"
    TRICYS 会在当前工作目录下创建一个以时间戳命名的主运行目录（例如 `20250116_103000/`）。上述所有输出目录（`results`, `temp`, `log`）都将默认创建在这个时间戳目录内部，以确保每次运行的输出结果都相互隔离。

### 3.2. `simulation` (仿真配置)

| 参数 (Parameter) | 描述 (Description) | 默认值 (Default Value) |
| :--- | :--- | :--- |
| `concurrent` | 是否开启并发（并行）仿真，用于加速参数扫描任务。 | `false` |
| `max_workers` | 如果开启并发，此选项用于指定最大并行工作进程数。 | 系统 CPU 核心数的一半 |
| `keep_temp_files` | 是否在仿真结束后保留临时文件（如模型编译文件）。对于调试非常有用。 | `true` |

### 3.3. `logging` (日志配置)

| 参数 (Parameter) | 描述 (Description) | 默认值 (Default Value) |
| :--- | :--- | :--- |
| `log_level` | 日志记录的最低级别。可选值包括 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"。 | `"INFO"` |
| `log_to_console` | 是否将日志实时输出到控制台。 | `true` |
| `log_count` | 在日志目录下保留的旧日志文件的最大数量。 | `5` |

---

## 4. 配置模板

以下是一些常用场景的配置模板，可以直接复制使用：

### 4.1 快速测试模板

此模板适用于快速验证模型是否能正常运行，它会保存所有变量，并以较短的时长运行。

```json
{
    "paths": {
        "package_path": "path/to/your/package.mo"
    },
    "simulation": {
        "model_name": "YourModel.Name",
        "variableFilter": "time|.*",
        "stop_time": 100.0,
        "step_size": 1.0
    }
}
```

### 4.2 生产环境模板

此模板适用于正式的、长时间的仿真。它只保存关键变量，将结果输出到指定的生产目录，并配置了更严格的日志记录和调试选项。

```json
{
    "paths": {
        "package_path": "path/to/your/package.mo",
        "results_dir": "production_results"
    },
    "simulation": {
        "model_name": "YourModel.Name",
        "variableFilter": "time|key_var1|key_var2",
        "stop_time": 86400.0,
        "step_size": 10.0,
        "keep_temp_files": false
    },
    "logging": {
        "log_level": "INFO",
        "log_to_console": false,
        "log_count": 10
    }
}
```

---

## 5. 如何运行

配置好文件后，有两种方式运行仿真：

### 5.1. 使用默认配置文件

将配置文件保存为 `config.json` 并放在项目根目录：

```bash
tricys
```

### 5.2. 指定配置文件

```bash
tricys -c my_config.json
```

---

## 6. 查看结果

仿真完成后，结果保存在时间戳子目录{timestamp}中：

```
Working Directory/
└── {timestamp}/
    ├── log/        
        └── simulation_{timestamp}.log  # 运行日志
    ├── result/                 
        └── simulation_result.csv       # 仿真结果数据
    └── temp/
        └── job_1/                      # 临时任务数据
                                   
```

### 6.1. 使用 Python 分析结果

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('results/20250116_103000/simulation_result.csv')

# 查看数据
print(df.head())
print(f"仿真时长: {df['time'].max()} 秒")
print(f"最终氚库存: {df['sds.I[1]'].iloc[-1]:.2f} g")

# 绘制氚库存变化曲线
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['sds.I[1]'], label='SDS 氚库存')
plt.xlabel('时间 (秒)')
plt.ylabel('氚库存 (g)')
plt.title('储存与输送系统氚库存随时间变化')
plt.legend()
plt.grid(True)
plt.savefig('inventory_plot.png', dpi=300)
plt.show()
```

### 6.2. 使用 Excel 查看

直接用 Microsoft Excel 或 LibreOffice Calc 打开 `simulation_result.csv` 文件。

---

## 7. 下一步

掌握了基础配置后，您可以继续学习：

- **[参数扫描](parameter_sweep.md)**：系统地研究参数对结果的影响
- **[并发运行](concurrent_operation.md)**：加速大规模仿真
- **[后处理模块](post_processing_module.md)**：自动分析和报告生成
- **[协同仿真](co_simulation_module.md)**：与外部软件集成

---
