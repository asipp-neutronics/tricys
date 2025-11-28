# SOBOL全局敏感性分析

全局敏感性分析 (Global Sensitivity Analysis, GSA) 是一种强大的工具，用于量化模型输入参数（以及它们之间的交互作用）对模型输出方差的贡献。与一次只改变一个参数的局部敏感性分析不同，GSA 会同时在整个参数空间内进行探索。

`tricys` 集成了业界标准的 `SALib` 库，提供了对 **Sobol** 方法的直接支持。Sobol 是一种基于方差的 GSA 方法，它能够高效地计算出每个参数对模型输出不确定性的贡献度，包括参数的独立影响和参数间的交互影响。

关于性能指标的定义、术语表和单位映射等通用配置，请参考[通用介绍](./index.md)。

## 1. 配置文件示例

SOBOL 分析的配置与之前的敏感性分析有显著不同，主要体现在 `independent_variable` 变为列表，`independent_variable_sampling` 变为对象，并新增了 `analyzer` 字段。

```json
{
    // ... (paths, simulation)
    "sensitivity_analysis": {
        "enabled": true,
        "analysis_cases": [
            {
                "name": "SALIB_SOBOL_Analysis",
                "independent_variable": ["pulseSource.width", "plasma.nf", "plasma.fb", "tep_fep.to_SDS_Fraction[1]", "blanket.TBR"],
                "independent_variable_sampling": {
                      "pulseSource.width": { "bounds": [50, 90], "distribution": "unif" },
                      "plasma.nf": { "bounds": [0.1, 0.9], "distribution": "unif" },
                      "plasma.fb": { "bounds": [0.03, 0.07], "distribution": "unif" },
                      "tep_fep.to_SDS_Fraction[1]": { "bounds": [0.1, 0.8], "distribution": "unif" },
                      "blanket.TBR": { "bounds": [1.05, 1.25], "distribution": "unif" }
                },
                "dependent_variables": [
                    "Startup_Inventory",
                    "Self_Sufficiency_Time",
                    "Doubling_Time"
                ],
                "analyzer": {
                    "method": "sobol",
                    "sample_N": 256
                }
            }
        ],
        // ... (通用配置)
    }
}
```

## 2. 关键配置项详解

- `independent_variable` (列表): 一个包含**所有**要进行全局敏感性分析的输入参数的列表。
- `independent_variable_sampling` (对象): 一个字典，为 `independent_variable` 列表中的每一个参数定义其采样范围和概率分布。
  - **键**: 参数的完整路径。
  - **值**: 包含 `bounds` (列表 `[min, max]`) 和 `distribution` (字符串, 如 `"unif"`) 的对象。
- `analyzer` (对象): 定义要使用的 GSA 方法及其参数。
  - `method`: 对于 Sobol 分析，固定为 `"sobol"`。
  - `sample_N`: Sobol 采样所需的基数 `N`。
    - ⚠️ **重要**: 实际运行的仿真总次数将是 **`N * (2D + 2)`**，其中 `D` 是 `independent_variable` 的数量。例如，本例中 `D=5`, `N=256`，则总仿真次数为 `256 * (2*5 + 2) = 3072` 次。

## 3. 分析报告输出

报告的核心内容是针对**每一个**因变量（性能指标）的独立分析结果。例如，对于 `Startup_Inventory` 指标，报告会包含：

1.  **Sobol敏感性指数表**: 一个 Markdown 表格，精确列出每个输入参数的一阶（S1）、总阶（ST）敏感性指数及其置信区间。
2.  **敏感性指数图**: 嵌入的条形图，直观地对比各个参数的 S1 和 ST 指数，便于快速识别关键影响因素。

### 如何解读Sobol指数

-   **S1 (一阶指数)**: 参数的**独立贡献**。S1 值越高，表示该参数对模型输出的**直接影响**越大。
-   **ST (总阶指数)**: 参数的**总体贡献**，包括其独立影响以及它与所有其他参数的**交互作用**。
-   **交互作用 (Interaction)**: `ST - S1` 的差值可以近似衡量该参数与其他参数的**交互效应强度**。如果一个参数的 `ST` 远大于其 `S1`，说明该参数的很多影响是通过与其他参数的耦合、协同作用实现的。
-   
## 4. 完整示例配置
<details>
<summary>example/analysis/4_sobol_global_sensitivity_analysis/sobol_global_sensitivity_analysis.json</summary>

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
                "name": "SALIB_SOBOL_Analysis",
                "independent_variable":["pulseSource.width","plasma.nf","plasma.fb","tep_fep.to_SDS_Fraction[1]","blanket.TBR"],
                "independent_variable_sampling":{
                      "pulseSource.width": {
                          "bounds": [50,90],
                          "distribution": "unif"
                      },
                      "plasma.nf": {
                          "bounds": [0.1,0.9],
                          "distribution": "unif"
                      },
                      "plasma.fb": {
                          "bounds": [0.03,0.07],
                          "distribution": "unif"
                      },
                      "tep_fep.to_SDS_Fraction[1]": {
                          "bounds": [0.1,0.8],
                          "distribution": "unif"
                      },
                      "blanket.TBR": {
                          "bounds": [1.05, 1.25],
						  "distribution": "unif"
					  }
                },
                "dependent_variables": [
                    "Startup_Inventory",
                    "Self_Sufficiency_Time",
                    "Doubling_Time"
                ],
                "analyzer": {
                    "method": "sobol",
                    "sample_N": 256
                }
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
