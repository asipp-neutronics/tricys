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

    详见：[协同仿真模块](guides/tricys_basic/co_simulation_module.md)

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

    详见：[敏感性分析教程](guides/tricys_analysis/single_parameter_sensitivity_analysis.md)

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