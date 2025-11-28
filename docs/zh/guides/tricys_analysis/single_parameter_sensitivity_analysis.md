# 单参数敏感性分析

单参数敏感性分析是 `tricys` 的核心功能之一，旨在研究**单个独立参数**的变化如何影响一系列用户定义的**关键性能指标 (KPIs)**。

该功能会自动运行一系列仿真（每次仿真对应独立参数的一个取值），计算每个仿真结果的性能指标，并生成图表来直观地展示它们之间的关系。关于性能指标的定义、术语表和单位映射等通用配置，请参考[通用介绍](./index.md)。

## 1. 配置文件示例

单参数敏感性分析的核心配置位于 `analysis_cases` 列表中。每个对象代表一个独立的分析案例。

```json
{
    // ... (paths, simulation)
    "sensitivity_analysis": {
        "enabled": true,
        "analysis_cases": [
            {
                "name": "TBR_Analysis",
                "independent_variable": "blanket.TBR",
                "independent_variable_sampling": [1.05, 1.1, 1.15, 1.2],
                "dependent_variables": [
                    "Startup_Inventory",
                    "Self_Sufficiency_Time",
                    "Doubling_Time"
                ],
                "plot_type":"line",
                "combine_plots":true,
                "sweep_time":["sds.I[1]"]
            }
        ],
        // ... (通用配置: metrics_definition, glossary_path, unit_map)
    }
}
```

## 2. 关键配置项详解

- `independent_variable` (字符串): 要进行扫描的**独立参数**的完整模型路径。这将是分析图表的X轴。
- `independent_variable_sampling` (列表): 为独立参数提供的一组离散的扫描值。程序会为列表中的每个值运行一次仿真。
- `dependent_variables` (列表): 要分析的因变量列表（即在 `metrics_definition` 中定义的指标名称）。这将是分析图表的Y轴。
- `plot_type` (字符串): 生成的敏感性图表的类型，通常为 `"line"` (线图)。
- `combine_plots` (布尔值): 是否将多个因变量的分析结果绘制在同一张图表中。`true` 会生成一张包含多个子图的组合图，`false` 则为每个因变量生成一张独立的图。
- `sweep_time` (列表): 一个包含原始变量名的列表。对于此列表中的每个变量，程序会生成一张“族谱图”，即将**每次**参数扫描得到的时间演化曲线绘制在同一张图上，便于比较动态行为的差异。

## 3. 分析报告输出

分析报告的结构与[通用说明](./index.md)中描述的类似，但其核心**性能指标分析图**具有以下特点：

- 图表的X轴是您定义的 `independent_variable`。
- Y轴是 `dependent_variables` 中定义的性能指标。
- 如果 `combine_plots` 为 `true`，报告将包含一张组合图，其中每个子图展示一个性能指标随独立参数变化的趋势。
- 如果 `sweep_time` 被定义，报告还会包含一张“族谱图”，展示原始变量（如 `sds.I[1]`）在不同独立参数取值下的时间演化曲线。

## 4. 完整示例配置
<details>
<summary>example/analysis/2_single_parameter_sensitivity_analysis/single_parameter_sensitivity_analysis.json</summary>

```json
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
                "name": "Width_Analysis",
                "independent_variable": "pulseSource.width",
                "independent_variable_sampling": [50,60,70,80,90,99],
                "dependent_variables": [
                    "Startup_Inventory",
                    "Self_Sufficiency_Time",
                    "Doubling_Time"
                ],
                "plot_type":"line",
                "combine_plots":true,
                "sweep_time":["sds.I[1]"]
            },
            {
                "name": "Fueling_Efficiency_Analysis",
                "independent_variable": "plasma.nf",
                "independent_variable_sampling": [0.01,0.05,0.1,0.2,0.4,0.5,0.6,0.7,0.8,0.9],
                "dependent_variables": [
                    "Startup_Inventory",
                    "Self_Sufficiency_Time",
                    "Doubling_Time"
                ],
                "plot_type":"line",
                "combine_plots":true,
                "sweep_time":["sds.I[1]"]
            },
            {
                "name": "Burn_Fraction_Analysis",
                "independent_variable": "plasma.fb",
                "independent_variable_sampling": [0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
                "dependent_variables": [
                    "Startup_Inventory",
                    "Self_Sufficiency_Time",
                    "Doubling_Time"
                ],
                "plot_type":"line",
                "combine_plots":true,
                "sweep_time":["sds.I[1]"]
            },
            {
                "name": "DIR_Analysis",
                "independent_variable": "tep_fep.to_SDS_Fraction[1]",
                "independent_variable_sampling": [0.1,0.15,0.2,0.3,0.4,0.6,0.8],
                "dependent_variables": [
                    "Startup_Inventory",
                    "Self_Sufficiency_Time",
                    "Doubling_Time"
                ],
                "plot_type":"line",
                "combine_plots":true,
                "sweep_time":["sds.I[1]"]
            },
            {
                "name": "TBR_Analysis",
                "independent_variable": "blanket.TBR",
                "independent_variable_sampling": [1.05,1.1,1.15,1.2],
                "dependent_variables": [
                    "Startup_Inventory",
                    "Self_Sufficiency_Time",
                    "Doubling_Time"
                ],
                "plot_type":"line",
                "combine_plots":true,
                "sweep_time":["sds.I[1]"]
            },
            {
                "name": "I_ISS_Analysis",
                "independent_variable": "i_iss.T",
                "independent_variable_sampling": [4,6,8,10],
                "dependent_variables": [
                    "Startup_Inventory",
                    "Self_Sufficiency_Time",
                    "Doubling_Time"
                ],
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
			"power":{
				"unit": "MW"
			},
			"period":{
				"unit": "hours"
			},
			"width":{
				"unit": "%"
			}
		}
    }
}

```

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