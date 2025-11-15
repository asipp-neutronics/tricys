# 后处理模块

`tricys` 不仅能运行仿真，还提供了一个强大的后处理（Post-Processing）框架，允许您在仿真任务完成后自动对结果进行分析、生成报告或执行其他自定义操作。

后处理功能通过在配置文件中添加 `post_processing` 字段来实现。

## 1. 配置文件示例

以下示例展示了如何在参数扫描完成后，自动执行两个分析任务：一个使用内置的 `module`，另一个使用用户自定义的 `script_path`。

```json
{
    "paths": {
        ...
    },
    "simulation": {
        ...
    },
    "simulation_parameters": {
        "blanket.TBR": "linspace:1:1.5:10"
    },
    "post_processing": [
        {
            "module": "tricys.postprocess.static_alarm",
            "function": "check_thresholds",
            "params": {
                "rules": [{"columns": ["sds.I[1]"], "min": 0.0}]
            }
        },
        {
            "script_path": "scripts/my_custom_analyzer.py",
            "function": "analyze_peak_value",
            "params": {
                "target_column_pattern": "sds.I[1]*",
                "report_filename": "peak_values.json"
            }
        }
    ]
}
```

## 2. 配置项详解

### 2.1. `post_processing`

- **描述**: 这是一个列表（Array），其中每个元素代表一个独立的后处理步骤。这些步骤会按照它们在列表中的顺序依次执行。
- **执行时机**: 所有仿真任务（例如参数扫描中的每一次运行）全部完成后，`tricys` 会将汇总的结果（`sweep_results.csv`）加载到一个 Pandas DataFrame 中，并将其传递给您指定的后处理函数。

### 2.2. 后处理函数

列表中的每个对象都定义了一个要执行的 Python 函数。您可以通过以下两种方式之一来指定要调用的代码：

#### 方式一：`module` (模块加载)

- `module` (字符串, 必填):
  - **描述**: 要调用的 Python 函数所在的**模块**的完整路径。这要求您的代码是一个可以通过 `import` 语句加载的 Python 包或模块（例如，已安装或位于带有 `__init__.py` 的目录中）。
  - **示例**: `tricys.postprocess.static_alarm`

#### 方式二：`script_path` (脚本路径加载)

- `script_path` (字符串, 必填):
  - **描述**: 指向包含要调用函数的单个 Python **脚本文件**的路径。这更加灵活，不需要您的脚本成为一个正式的包。
  - **示例**: `scripts/my_custom_analyzer.py`

---

无论使用哪种方式，您都需要提供以下字段：

- `function` (字符串, 必填):
  - **描述**: 要在指定模块或脚本中调用的函数名。
  - **示例**: `check_thresholds`

- `params` (字典, 选填):
  - **描述**: 一个包含要传递给目标函数的关键字参数（keyword arguments）的字典。
  - **示例**: 在上面的例子中，`params` 为 `check_thresholds` 函数提供了 `rules` 参数。

## 3. 内置后处理模块

`tricys` 自带了一些常用的后处理模块，位于 `tricys/postprocess` 目录下：

- **`rise_analysis`**: 用于分析信号的上升时间、下降时间、峰值等动态特性。
- **`static_alarm`**: 用于检查结果是否超出了预设的静态阈值（上限或下限）。
- **`baseline_analysis`**: 用于执行基准工况分析。

## 4. 自定义后处理模块

后处理框架最大的优势在于其扩展性。您可以轻松编写自己的 Python 脚本来执行任何想要的分析。

### 4.1. 函数签名 

为了能被 `tricys` 框架正确调用，您的自定义后处理函数**必须**遵循特定的签名。框架会自动通过**关键字参数**传入两个核心数据：

1.  `results_df` (pd.DataFrame): 包含所有仿真运行汇总结果的 Pandas DataFrame。
2.  `output_dir` (str): 一个专属的输出目录路径，供您保存报告、图表等分析产物。

因此，您的函数签名必须能够接收这两个参数，以及您在 `params` 中定义的任何其他自定义参数。

一个标准的函数签名如下：

```python
import pandas as pd

def my_custom_function(results_df: pd.DataFrame, output_dir: str, **kwargs):
    """
    一个通用的后处理函数签名。
    
    - results_df: 由 tricys 传入的仿真结果。
    - output_dir: 由 tricys 提供的用于保存报告的目录。
    - **kwargs: 用于接收来自 JSON 配置中 "params" 的所有自定义参数。
    """
    # 从 kwargs 中获取自定义参数
    my_param = kwargs.get("my_param", "default_value")
    
    # 在这里编写您的分析代码...
    print(f"分析报告将保存在: {output_dir}")
    print(f"收到的自定义参数 my_param 的值为: {my_param}")
    print("结果数据预览:")
    print(results_df.head())
```

### 4.2. 完整示例

让我们创建一个完整的自定义后处理脚本，并展示如何通过 `script_path` 在配置中调用它。

**步骤 1: 创建分析脚本**

假设我们在项目下创建了一个名为 `scripts/my_custom_analyzer.py` 的文件：

```python
# scripts/my_custom_analyzer.py
import pandas as pd
import os
import json

def analyze_peak_value(results_df: pd.DataFrame, output_dir: str, target_column_pattern: str, report_filename: str = "peak_report.json"):
    """
    在所有匹配的列中查找峰值，并生成一份报告。
    """
    # 筛选出符合模式的列
    target_columns = [col for col in results_df.columns if target_column_pattern in col]
    
    if not target_columns:
        print(f"警告: 未找到匹配 '{target_column_pattern}' 的列。")
        return

    # 计算每一列的峰值
    peak_values = results_df[target_columns].max().to_dict()
    
    # 定义报告输出路径
    report_path = os.path.join(output_dir, report_filename)
    
    # 保存报告
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(peak_values, f, indent=4)
        
    print(f"峰值分析报告已保存至: {report_path}")

```

**步骤 2: 更新配置文件**

现在，在您的 `config.json` 中配置 `post_processing` 部分来调用这个脚本：

```json
{
    ...
    "post_processing": [
        {
            "script_path": "scripts/my_custom_analyzer.py",
            "function": "analyze_peak_value",
            "params": {
                "target_column_pattern": "sds.I[1]",
                "report_filename": "sds_peak_values.json"
            }
        }
    ]
}
```

当 `tricys` 完成所有仿真后，它会自动执行 `analyze_peak_value` 函数，并将仿真结果中所有包含 `sds.I[1]` 的列的峰值计算出来，最后将结果保存到 `post_processing/sds_peak_values.json` 文件中。

通过这种方式，您可以将任意复杂的数据分析流程无缝集成到 `tricys` 的自动化工作流中。

---

## 5. 下一步

掌握了如何创建和使用后处理模块后，您可以将它应用到更复杂的场景中：

- **[敏感性分析](../tricys_analysis/index.md)**：为复杂的敏感性分析结果编写专用的后处理脚本，以提取关键指标和生成可视化图表。
- **[协同仿真](co_simulation_module.md)**：对包含外部模块的协同仿真结果进行整合与分析。

