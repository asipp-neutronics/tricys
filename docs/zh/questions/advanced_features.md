??? question "问：什么是协同仿真？如何使用？"
    协同仿真允许 TRICYS 与外部软件（如 Aspen Plus）交互：

    **工作流程**：
    1. 运行初步仿真
    2. 调用外部处理器（Handler）
    3. 外部软件计算新数据
    4. 将数据注入回 Modelica 模型
    5. 运行最终完整仿真

    **配置示例**：
    ```json
    {
        "co_simulation": [
            {
                "mode": "interceptor",
                "submodel_name": "example_model.I_ISS",
                "instance_name": "i_iss",
                "handler_module": "tricys.handlers.i_iss_handler",
                "handler_function": "run_aspen_simulation",
                "params": {
                    "bkp_path": "path/to/aspen/file.bkp"
                }
            }
        ]
    }
    ```

    *   `mode`：`interceptor`（默认）或`replacement`。
    *   `handler_module`：处理器所在的模块。
    *   `handler_script_path`：或者，直接提供处理器脚本的路径。

    详见：[协同仿真模块](../guides/tricys_basic/co_simulation_module.md)

??? question "问：如何创建自定义后处理模块？"
    后处理模块是 Python 函数，接收仿真结果并执行分析：

    **1. 创建处理函数**：
    ```python
    # my_postprocess.py
    def analyze_results(df, output_filename="my_report.txt"):
        """
        自定义后处理函数
        
        Args:
            df: Pandas DataFrame，包含仿真结果
            output_filename: 输出文件名
        """
        # 执行分析
        max_inventory = df['sds.I[1]'].max()
        
        # 保存结果
        with open(output_filename, 'w') as f:
            f.write(f"最大氚库存: {max_inventory} g\n")
    ```
    **注意**：请将自定义模块（如`my_postprocess.py`）放置在 `tricys` 项目的根目录下，或确保它在 Python 的可可导入路径中。

    **2. 在配置文件中引用**：
    ```json
    {
        "post_processing": [
            {
                "module": "my_postprocess",
                "function": "analyze_results",
                "params": {
                    "output_filename": "custom_report.txt"
                }
            }
        ]
    }
    ```

??? question "问：如何进行敏感性分析？"
    TRICYS 提供多种敏感性分析方法：

    **1. 单参数敏感性分析**：
    ```bash
    tricys analysis -c single_param_analysis.json
    ```

    研究单个参数对 KPIs 的影响。

    **2. 多参数敏感性分析**：
    ```bash
    tricys analysis -c multi_param_analysis.json
    ```

    研究参数间的耦合效应。

    **3. SOBOL 全局敏感性分析**：
    ```bash
    tricys analysis -c sobol_analysis.json
    ```

    量化参数及其交互作用的贡献。

    **4. Latin 不确定性量化**：
    ```bash
    tricys analysis -c latin_analysis.json
    ```

    评估输入不确定性对输出的影响。

    详见：[敏感性分析教程](../guides/tricys_analysis/single_parameter_sensitivity_analysis.md)

??? question "问：如何定义自定义性能指标？"
    性能指标（Metrics）在 `sensitivity_analysis.metrics_definition` 中定义：

    **使用内置指标**：
    ```json
    {
        "metrics_definition": {
            "Max_Inventory": {
                "source_column": "sds.I[1]",
                "method": "max_value"
            }
        }
    }
    ```

    **内置指标方法**：
    * `get_final_value`
    * `max_value`, `min_value`, `mean_value`
    * `time_of_max`, `time_of_min`
    * `time_of_turning_point`
    * `calculate_startup_inventory`
    * `calculate_doubling_time`
    * `calculate_required_tbr`（二分法搜索）

    关于内置指标的详细物理意义，请参考 [核心性能指标详解](../explanation/tricys_analysis/performance_metrics.md)。

    **创建自定义指标**：
    ```python
    # my_metrics.py
    def calculate_peak_to_peak(series):
        """计算峰峰值"""
        return series.max() - series.min()
    ```

    在 `tricys/analysis/metric.py` 中注册您的函数，或直接在配置中引用。

---
??? question "如何为 Blanket 设置盘存与处理速率上限？"
    自 PR #82 起，`example_model.Blanket` 内置 **sigmoid 软约束**：当氚盘存 `I_total`
    接近 `capacity_max` 或瞬时出流接近 `rate_max` 时，会被平滑地限幅，并把溢出
    部分通过 `overflow_out[5]` 与 `rate_clip_out[5]` 端口暴露出来。

    **默认行为**：`capacity_max = rate_max = 1e9`，等价于"无约束"，
    与历史版本完全一致（无回归）。

    **通过 JSON 配置启用**：

    ```json
    {
      "paths": { "package_path": "../../example_model/package.mo" },
      "simulation": {
        "model_name": "example_model.Cycle",
        "variableFilter": "time|blanket.I[1]|blanket.I_total|blanket.overflow_out[1]|blanket.rate_clip_out[1]|sds.I[1]",
        "stop_time": 5000.0,
        "step_size": 1.0
      },
      "simulation_parameters": {
        "blanket.capacity_max": 500,
        "blanket.rate_max": 50,
        "blanket.softness": 0.02
      }
    }
    ```

    **参数说明**：

    | 参数 | 单位 | 含义 |
    |------|------|------|
    | `blanket.capacity_max` | g | Blanket 总盘存上限（sigmoid 软约束） |
    | `blanket.rate_max`     | g/h | 瞬时出流上限（sigmoid 软约束） |
    | `blanket.softness`     | – | 软过渡区相对宽度，默认 0.02；越小越接近硬约束 |

    **输出端口**：

    * `blanket.overflow_out[i]`：因容量约束被拒收的入流（i 对应同位素索引）
    * `blanket.rate_clip_out[i]`：因速率约束被截断的出流

    **完整示例**：见 `tricys/example/example_data/basic/7_blanket_constraints/`
    （单点配置 + 4×4 参数扫描）。

    **与 ConstrainedBuffer 的关系**：`ConstrainedBuffer`（PR #81 示例 6）是
    通用的"带约束储罐"组件，**没有** TBR 产氚源；而本特性把同样的约束机制
    **直接嵌入 Blanket**，保留其产氚行为，**不需要替换主模型 `Cycle.mo` 中的实例**。

??? question "问：如何为子系统设置最大盘存量和最大处理速率？"
    使用 `ConstrainedBuffer` 组件替换标准子系统模型（如 Blanket），即可获得容量上限和速率上限约束能力。

    **核心参数**：

    | 参数 | 含义 | 默认值 | 单位 |
    |------|------|--------|------|
    | `capacity_max` | 最大盘存量（总氚当量） | 1e9（无约束） | g |
    | `rate_max` | 最大出流速率 | 1e9（无约束） | g/h |
    | `softness` | Sigmoid 软约束系数 | 0.02 | — |
    | `to_Down_Fraction` | 送往下游的比例 | 1.0 | — |

    **JSON 配置示例**（设置容量 500g、速率 50 g/h）：
    ```json
    {
        "simulation_parameters": {
            "blanket_c.capacity_max": 500,
            "blanket_c.rate_max": 50
        }
    }
    ```

    **参数扫描示例**：
    ```json
    {
        "simulation_parameters": {
            "blanket_c.capacity_max": [200, 500, 1000, 2000],
            "blanket_c.rate_max": [20, 50, 100, 1000000000]
        }
    }
    ```

    **约束行为**：

    - 超容量时：入流被 sigmoid 函数软限制，超限部分导出至 `overflow_out` 端口
    - 超速率时：出流被限制在 `rate_max` 以内，超限部分导出至 `rate_clip_out` 端口
    - 质量守恒：所有被约束截流的物质均通过对应端口导出，不会凭空消失

    **完整示例配置**参见 `tricys/example/example_data/basic/6_constrained_buffer/`。


---
