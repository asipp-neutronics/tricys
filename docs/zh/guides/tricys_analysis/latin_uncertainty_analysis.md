# Latin不确定性量化分析

不确定性量化 (Uncertainty Quantification, UQ) 是评估模型或系统可靠性的关键步骤。它的目的不是找出哪个参数最敏感，而是回答一个更重要的问题：“当我的输入参数在一定范围内不确定时，我的输出结果（性能指标）的可能范围是多大？其概率分布是怎样的？”

`tricys` 利用高效的 **Latin 超立方采样 (Latin Hypercube Sampling, LHS)** 技术来进行 UQ 分析。LHS 是一种分层采样方法，它能用相对较少的样本点均匀地探索整个多维参数空间，从而高效地评估输入不确定性如何传播到模型输出。

关于性能指标的定义、术语表和单位映射等通用配置，请参考[通用介绍](./index.md)。

## 1. 配置文件示例

LHS 分析的配置与 SOBOL 全局敏感性分析非常相似，主要区别在于 `analyzer.method` 被设置为 `"latin"`。

```json
{
    // ... (paths, simulation)
    "sensitivity_analysis": {
        "enabled": true,
        "analysis_cases": [
            {
                "name": "SALIB_LATIN_Analysis",
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
                    "method": "latin",
                    "sample_N": 256
                }
            }
        ],
        // ... (通用配置)
    }
}
```

## 2. 关键配置项详解

- `independent_variable` (列表): 一个包含所有被视为不确定性来源的输入参数的列表。
- `independent_variable_sampling` (对象): 一个字典，为每个不确定参数定义其采样范围和概率分布。
- `analyzer` (对象): 定义要使用的分析方法及其参数。
  - `method`: 固定为 `"latin"`。
  - `sample_N`: Latin 超立方采样生成的样本数量，直接对应于将要运行的仿真总次数。

## 3. 工作流详解

当 `tricys` 接收到一个包含 `analyzer` 字段的分析任务时（如本例中的LHS分析），它会启动一个基于 `SALib` 库的特殊工作流。

1.  **识别分析类型**: 在 `tricys.simulation.simulation_analysis.py` 的主流程中，程序首先检测到这是一个 SALib 分析任务。
2.  **定义问题和采样**: 程序将任务委托给 `tricys.analysis.salib.py` 模块，该模块根据配置定义一个 `SALib` “问题空间”，并调用 LHS 采样函数生成 `N` 个参数样本点。
3.  **执行批量仿真**: 生成的 `N` 个参数样本被写入一个临时的 CSV 文件中，`tricys` 随后为每个样本点执行一次仿真。
4.  **收集和分析结果**: 所有仿真运行完毕后，程序会计算每个性能指标的 `N` 个输出结果，并对其进行统计分析（计算均值、标准差、百分位数等）。
5.  **生成报告和图表**: 最后，程序根据统计分析结果生成最终的 Markdown 报告，其中包含详细的统计数据表格和输出分布图（直方图和累积分布函数图）。

## 4. 分析报告输出

报告的核心内容是针对**每一个**因变量（性能指标）的独立统计分析。例如，对于 `Startup_Inventory` 指标，报告会包含：

1.  **统计摘要 (Statistical Summary)**: 提供一组核心的统计数据，包括均值、标准差、最小值和最大值。
2.  **分布关键点 (Key Distribution Points / CDF)**: 提供一系列百分位数（如5%, 25%, 50% (中位数), 75%, 95%），用于精确描述输出指标的累积分布情况。
3.  **输出分布 (直方图数据)**: 一个表格，展示了输出结果在不同数值区间的频数分布。
4.  **输出分布图 (Output Distribution Plot)**: 嵌入的 `.png` 图表，包含两个子图：
    -   **直方图**: 直观展示输出指标的概率密度分布形状。
    -   **累积分布函数 (CDF)**: 展示了输出指标值小于或等于某个特定值的概率。

### 如何解读UQ结果

不确定性量化分析的重点是理解**输出的分布**，而不是参数的敏感度排序。

-   **看中心趋势**: **均值 (Mean)** 和 **中位数 (Median)** 告诉您性能指标最可能落在哪个值附近。
-   **看离散程度**: **标准差 (Standard Deviation)** 和 **5%-95%百分位数的范围** 揭示了输出的不确定性有多大。范围越宽，说明输入参数的不确定性对系统性能造成的影响越大，系统的鲁棒性可能越差。
-   **看分布形状**: **直方图**的形状很重要。一个对称的、类似正态分布的钟形曲线是比较理想的。如果分布出现**长尾**或**偏斜**，可能意味着系统在某些参数组合下容易出现极端的好或坏的结果，这对于风险评估至关重要。


## 5. 完整示例配置
<details>
<summary>example/analysis/5_latin_uncertainty_analysis/latin_uncertainty_analysis.json</summary>

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
                "name": "SALIB_LATIN_Analysis",
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
                    "method": "latin",
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

## 6. AI 增强分析

`tricys` 的所有分析模块都深度集成了大型语言模型（LLM），能够将原始的图表和数据自动转化为结构化的学术风格报告。

### 6.1. 启用方式

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

### 6.2. 环境配置

在使用此功能前，您必须在项目的**根目录**下创建一个名为 `.env` 的文件，并填入您的大语言模型 API 凭据。这确保了您的密钥安全，不会被提交到版本控制中。

```env
# .env file in project root
API_KEY="sk-your_api_key_here"
BASE_URL="https://your_api_base_url/v1"
AI_MODEL="your_model_name_here"
```

### 6.3. 输出报告

启用后，除了标准的分析报告 (`analysis_report_...md`)，`tricys` 还会在该案例的 `report` 文件夹内生成两份额外的报告：

- **`analysis_report_{case_name}_{model_name}.md`**: 在核心报告的基础上，末尾追加了由 AI 生成的对数据和图表的深度文字解读。
- **`academic_report_{case_name}_{model_name}.md`**: 一份完全由 AI 撰写的、结构严谨的学术风格报告。这份报告通常包含摘要、引言、方法、结果与讨论、结论等部分，可以直接作为汇报材料或论文初稿使用。
