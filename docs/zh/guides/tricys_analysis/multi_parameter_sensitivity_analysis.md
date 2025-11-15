# 多参数敏感性分析

在单参数敏感性分析的基础上，`tricys` 提供了功能更强大的**多参数敏感性分析**。该功能允许您在研究一个主独立参数（X轴）对性能指标（Y轴）的影响时，**同时扫描一个或多个背景参数**。

这使得您可以在一张图上生成一组“敏感性曲线族”，其中每一条曲线代表一个背景参数的特定取值。通过这种方式，您可以深入地理解参数之间的**交互和耦合效应**。此外，该功能还支持一种强大的“目标寻求”式分析。

关于性能指标的定义、术语表和单位映射等通用配置，请参考[通用介绍](./index.md)。

## 1. 核心概念：参数交互分析

这是多参数分析最常见的用法，其核心是在 `analysis_cases` 的某个案例内部，嵌入一个 `simulation_parameters` 字段。

### 1.1. 配置文件示例
```json
{
    // ...
    "sensitivity_analysis": {
        "enabled": true,
        "analysis_cases": [
            {
                "name": "DIR_PLASMA_Analysis",
                "independent_variable": "tep_fep.to_SDS_Fraction[1]", // 主独立参数 (X轴)
                "independent_variable_sampling": [0.1, 0.3, 0.6, 0.8],
                "dependent_variables": [ "Startup_Inventory", "Required_TBR" ], // 因变量 (Y轴)
                "simulation_parameters": {
                    "plasma.fb": [0.02, 0.04, 0.08, 0.09, 0.1], // 背景扫描参数 (生成多条曲线)
                    "plasma.nf": 0.5 // 固定的背景参数
                }
            }
        ],
        // ... (通用配置)
    }
}
```

### 1.2. 工作原理

- **主独立参数 (`independent_variable`)**: `tep_fep.to_SDS_Fraction[1]`，作为图表的 **X轴**。
- **背景扫描参数 (`simulation_parameters`)**:
    - `plasma.fb` 的值是一个**列表**，它将成为图表中的**图例 (Legend)**，每一条曲线对应 `plasma.fb` 的一个取值。
    - `plasma.nf` 的值是一个**标量**，它将在所有仿真中保持不变。
- **执行逻辑**: 程序会执行一个“嵌套循环”。对于 `independent_variable` 的每一个取值，程序会为 `plasma.fb` 的每一个值都运行一次仿真。总运行次数为 `len(independent_variable_sampling) * len(plasma.fb)`。

## 2. 高级用法：目标寻求分析

此功能支持一种“逆向”分析模式，即**目标寻求 (Goal-Seeking)**。您可以指定一个性能指标作为目标，反向求解为了达到这个目标，某个输入参数需要被设置成多少。

### 2.1. 配置文件示例

关键在于 `simulation_parameters` 中包含了一个与 `metrics_definition` 中优化指标同名的特殊对象（本例中为 `Required_TBR`）。

```json
{
    "name": "DoubleTime_PLASMA_Analysis",
    "independent_variable": "plasma.fb", // 主独立参数 (X轴)
    "dependent_variables": [ "Startup_Inventory", "Required_TBR" ], // 因变量 (Y轴)
    "simulation_parameters": {
        "plasma.nf": 0.5,
        "Required_TBR": { // 特殊的目标寻求配置
            "metric_name": "Doubling_Time", // 目标指标
            "metric_max_value": [4380, 8760, 13140, 17530] // 目标值的列表 (单位: 小时)
        }
    },
    // ...
    "metrics_definition": {
        // ...
        "Required_TBR": {
            "method": "bisection_search",
            "parameter_to_optimize": "blanket.TBR", // 需要求解的参数
            // ... (bisection_search 的其他配置)
        }
    }
}
```

### 2.2. 工作原理

- **问题描述**: 此配置旨在回答：“当 `plasma.fb` 变化时，为了分别实现不同的 `Doubling_Time` 目标（4380h, 8760h, ...），我们需要的 `blanket.TBR` 是多少？”
- **执行逻辑**: 对于X轴上的每一个 `plasma.fb` 值，程序会针对 `metric_max_value` 列表中的**每一个目标值**启动一次 `bisection_search` 优化循环，以求解出对应的 `blanket.TBR` 值。
- **结果解读**: 最终的图表将展示，在不同的倍增时间目标约束下，所需的TBR是如何随着`plasma.fb`变化的。

## 3. 分析报告输出

与单参数分析的主要区别在于**性能指标分析图**：

-   对于**参数交互分析**，图表将包含一组曲线，图例对应背景参数（如 `plasma.fb`）的不同取值。
-   对于**目标寻求分析**，图表同样包含多条曲线，但图例对应的是不同的性能目标约束（如 `Doubling_Time = 4380h`）。

这使得多维度的数据关系能够被清晰地呈现在一张二维图表中。报告的其他部分与通用结构一致。

## 4. 完整示例配置
<details>
<summary>example/analysis/3_multi_parameter_sensitivity_analysis/multi_parameter_sensitivity_analysis.json</summary>

{
    "paths": {
        "package_path": "../../example_model_single/example_model.mo"
    },
    "simulation": {
        "model_name": "example_model.Cycle",
        "variableFilter": "time|sds.I[1]",
        "stop_time": 12000.0,
        "step_size": 0.5
    },
    "sensitivity_analysis": {
        "enabled": true,
        "analysis_cases": [
            {
                "name": "DIR_PLASMA_Analysis",
                "independent_variable": "tep_fep.to_SDS_Fraction[1]",
                "independent_variable_sampling": [0.1,0.3,0.6,0.8],
                "dependent_variables": [
                    "Startup_Inventory",
                    "Required_TBR"
                ],
                "simulation_parameters": {
                    "plasma.fb": [0.02,0.04,0.08,0.09,0.1],
                    "plasma.nf":0.5
                },
                "plot_type":"line",
                "combine_plots":true,
                "sweep_time":["sds.I[1]"]
            },
            {
                "name": "Pulse_PLASMA_Analysis",
                "independent_variable": "pulseSource.width",
                "independent_variable_sampling": [50,60,70,80,90,99],
                "dependent_variables": [
                    "Startup_Inventory",
                    "Required_TBR"
                ],
                "simulation_parameters": {
                    "plasma.fb": [0.02,0.04,0.08,0.09,0.1],
                    "plasma.nf":0.5
                },
                "plot_type":"line",
                "combine_plots":true,
                "sweep_time":["sds.I[1]"]
            },
            {
                "name": "DoubleTime_PLASMA_Analysis",
                "independent_variable": "plasma.fb",
                "independent_variable_sampling": [0.02,0.05,0.08,0.1],
                "dependent_variables": [
                    "Startup_Inventory",
                    "Required_TBR"
                ],
                "simulation_parameters": {
                    "plasma.nf":0.5,
                    "Required_TBR": {
                        "metric_name":"Doubling_Time",
                        "metric_max_value": [4380,8760,13140,17530]
                    }
                },
                "plot_type":"line",
                "combine_plots":true,
                "sweep_time":["sds.I[1]"]
            }
        ],
        "metrics_definition": {
            "Startup_Inventory": {
                "source_column": "sds.I[1]",
                "method": "calculate_startup_inventory"
            },
            "Self_Sufficiency_Time": {
                "source_column": "sds.I[1]",
                "method": "time_of_turning_point"
            },
            "Doubling_Time": {
                "source_column": "sds.I[1]",
                "method": "calculate_doubling_time"
            },
            "Required_TBR": {
                "source_column": "sds.I[1]",
                "method": "bisection_search",
                "parameter_to_optimize": "blanket.TBR",
                "search_range": [1,1.5],
                "tolerance": 0.005,
                "max_iterations": 10
            }
        },
        "glossary_path": "../../example_glossary/example_glossary.csv",
		"unit_map": {
			"Doubling_Time": {
				"unit": "days",
				"conversion_factor": 24
			},
			"Startup_Inventory": {
				"unit": "kg",
				"conversion_factor": 1000
			},
            "Self_Sufficiency_Time": {
				"unit": "days",
				"conversion_factor": 24
            },
			"width":{
				"unit": "%"
			}
		}
    }
}

</details>

## 5. AI 增强分析

`tricys` 的所有分析模块都深度集成了大型语言模型（LLM），能够将原始的图表和数据自动转化为结构化的学术风格报告。

### 5.1. 启用方式

在您的分析案例配置中（即 `analysis_cases` 列表中的任意一个对象，或 `post_processing` 的 `params` 中），添加 `"ai": true` 即可为该案例激活 AI 分析功能。

```json
"analysis_cases": [
    {
        "name": "TBR_Analysis_with_AI_Report",
        "independent_variable": "blanket.TBR",
        "independent_variable_sampling": [1.05, 1.1, 1.15, 1.2],
        "dependent_variables": [ "Doubling_Time" ],
        "ai": true
    }
]
```

### 5.2. 环境配置

在使用此功能前，您必须在项目的**根目录**下创建一个名为 `.env` 的文件，并填入您的大语言模型 API 凭据。这确保了您的密钥安全，不会被提交到版本控制中。

```env
# .env file in project root
API_KEY="sk-your_api_key_here"
BASE_URL="https://your_api_base_url/v1"
AI_MODEL="your_model_name_here"
```

### 5.3. 输出报告

启用后，除了标准的分析报告 (`analysis_report_...md`)，`tricys` 还会在该案例的 `report` 文件夹内生成两份额外的报告：

- **`analysis_report_{case_name}_{model_name}.md`**: 在核心报告的基础上，末尾追加了由 AI 生成的对数据和图表的深度文字解读。
- **`academic_report_{case_name}_{model_name}.md`**: 一份完全由 AI 撰写的、结构严谨的学术风格报告。这份报告通常包含摘要、引言、方法、结果与讨论、结论等部分，可以直接作为汇报材料或论文初稿使用。
