??? question "问：如何运行仿真？"
    TRICYS 提供多种运行方式：

    **1. 命令行（CLI）**：
    ```bash
    # 使用默认配置文件 config.json
    tricys

    # 指定配置文件
    tricys -c my_config.json

    # 运行分析任务
    tricys analysis -c analysis_config.json
    ```

    **2. 图形界面（GUI）**：
    ```bash
    tricys gui
    ```

    **3. 交互式示例**：
    ```bash
    # 运行所有示例的交互式菜单
    tricys example

    # 运行基础仿真示例
    tricys basic example

    # 运行分析示例
    tricys analysis example
    ```

??? question "问：如何理解输出文件？"
    仿真完成后，结果保存在 `results/` 目录下的时间戳子目录中：

    | 文件名 | 说明 |
    |--------|------|
    | `simulation_result.csv` | **单参数组合**的详细结果，包含所有变量随时间的变化。 |
    | `sweep_results.csv` | **多参数组合**（参数扫描）的汇总结果。 |
    | `sensitivity_analysis_summary.csv` | **【仅分析任务】** 敏感性分析的汇总指标，每行是一次运行的KPI。|
    | `requierd_tbr_summary.csv` | **【仅分析任务】** 当执行“TBR需求”等目标搜索任务时生成的优化结果。|
    | `simulation_*.log` | 详细的运行日志，包含调试信息。 |
    | `config.json` | 本次运行使用的完整配置备份。 |
    | `*_report.md` | **【仅分析任务】** 自动生成的AI分析报告。 |
    | `*.png` / `*.csv` | 各种图表和数据导出。 |

    **结果文件CSV结构**

    无论运行一个还是多个参数组合，最终的CSV文件都遵循相似的列命名规则。

    *   **基础情况 (无扫描参数)**:
        如果 `simulation_parameters` 为空，列名就是 `variableFilter` 中定义的变量名。
        ```csv
        time,sds.I[1],blanket.I[1],...
        0.0,10.5,2.3,...
        ```

    *   **参数扫描情况**:
        当 `simulation_parameters` 不为空时，列名会附加参数信息。
        ```csv
        time,sds.I[1]&blanket.TBR=1.05,sds.I[1]&blanket.TBR=1.1,...
        ```
        - **列名格式**: `<变量名>&<参数1>=<值1>&<参数2>=<值2>...`
        - `time` 列保持不变。
        - 每个参数组合下的每个变量都成为一个独立的列。列名由 **变量名** 和 **参数-值** 对拼接而成，并用 `&` 符号分隔。

??? question "问：如何定义复杂的参数扫描？"
    TRICYS 支持多种[参数扫描格式](../guides/tricys_basic/parameter_sweep.md)：

    | 功能 | 格式 | 示例 | 说明 |
    | :--- | :--- | :--- | :--- |
    | **离散列表** | `[v1, v2, ...]` | `[6, 12, 18]` | 一组离散值 |
    | **等差序列** | `"start:stop:step"` | `"1.05:1.15:0.05"` | 起始值、终止值、步长 |
    | **线性间隔** | `"linspace:start:stop:num"` | `"linspace:10:20:5"` | 生成 `num` 个等间距点 |
    | **对数间隔** | `"log:start:stop:num"` | `"log:1:1000:4"` | 生成 `num` 个对数尺度的点 |
    | **从文件读取** | `"file:path:column"` | `"file:data.csv:voltage"` | 从 CSV 文件的指定列读取 |

    **示例配置**：
    ```json
    {
        "simulation_parameters": {
            "blanket.TBR": [1.05, 1.1, 1.15, 1.2],
            "plasma.fb": "linspace:0.01:0.1:10",
            "tep_fep.to_SDS_Fraction[1]": "log:0.1:1.0:5"
        }
    }
    ```

??? question "问：如何过滤输出变量？"
    使用 `variableFilter` 参数来选择需要保存的变量。该参数支持正则表达式，但请注意其语法以匹配Modelica变量命名规则。

    **配置示例**：

    ```json
    {
        "simulation": {
            "variableFilter": "time|sds.I[1]|blanket.I[1-5]|div.I[1-5]"
        }
    }
    ```

    **常用模式**：
    *   `time`：时间变量（必须包含）
    *   `sds.I[1]`：精确匹配单个变量
    *   `sds.I[1-5]`：匹配数组变量 `sds.I[1]` 到 `sds.I[5]`
    *   `blanket.I[1-5]|div.I[1-5]`：匹配多个特定数组变量


??? question "问：仿真时间很长，如何加速？"
    可以采取以下优化措施：

    **1. [启用并发运行](../guides/tricys_basic/concurrent_operation.md)**：
    ```json
    {
        "simulation": {
            "concurrent": true,
            "max_workers": 4
        }
    }
    ```

    **2. 减少输出变量**：
    ```json
    {
        "simulation": {
            "variableFilter": "time|sds.I[1]"  # 只保存关键变量
        }
    }
    ```

    **3. 增大时间步长**（权衡精度）：
    ```json
    {
        "simulation": {
            "step_size": 1.0  # 从 0.5 增加到 1.0
        }
    }
    ```

    **4. 减少扫描点数**：
    ```json
    {
        "simulation_parameters": {
            "blanket.TBR": "linspace:1.05:1.15:5"  # 从 20 减少到 5
        }
    }
    ```

---