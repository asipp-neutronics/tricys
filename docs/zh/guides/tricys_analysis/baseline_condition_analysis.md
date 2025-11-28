# 基准工况分析

基准工况分析（Baseline Condition Analysis）是 `tricys` 分析工具集中的一项核心功能。它用于对单一、确定的参数配置（即“基准工况”）下的系统行为进行全面评估，并自动生成一份标准化的 Markdown 分析报告。

该功能本质上是 `tricys` 的后处理模块，关于后处理模块的通用配置，请参考[后处理模块](../../tricys_basic/post_processing_module.md)。关于性能指标的定义、术语表和单位映射等通用配置，请参考[通用介绍](./index.md)。

## 1. 配置文件示例

该分析的配置文件是**一次单独的仿真**，紧跟着一个特定的**后处理步骤**。因此，配置文件中不包含 `sensitivity_analysis` 扫描部分。

```json
{
    "paths": {
        "package_path": "../../example_model_single/example_model.mo"
    },
    "simulation": {
        "model_name": "example_model.Cycle",
        "variableFilter": "time|sds.I[1]|div.I[1]|cps.I[1]|tes.I[1]|blanket.I[1]|i_iss.I[1]|wds.I[1]|o_iss.I[1]",
        "stop_time": 2000.0,
        "step_size": 0.5
    },
    "post_processing": [
        {
           "module": "tricys.postprocess.baseline_analysis",
           "function": "baseline_analysis",
           "params": {
                "detailed_var": "sds.I[1]",
                "glossary_path": "../../example_glossary/example_glossary.csv"
            }
        }
    ]
}
```

## 2. 关键配置项详解

核心在于 `post_processing` 部分的配置：

- `module`: 固定为 `tricys.postprocess.baseline_analysis`。
- `function`: 固定为 `baseline_analysis`。
- `params`:
    - `detailed_var` (字符串, 选填):
        - **描述**: 指定一个模型中的变量，报告将为该变量生成一张包含“细节视图”的时间演化曲线图，放大其“自持点”附近区域。
    - `glossary_path` (字符串, 选填):
        - **描述**: 指向一个“术语表” CSV 文件的路径。

## 3. 分析报告输出

报告的结构与[通用说明](./index.md)中描述的类似，但其核心内容是针对单次运行的快照分析，包含：

1.  **关键性能指标 (Key Performance Indicators)**: 基于 `detailed_var` 计算出的 `Startup Inventory`, `Self-Sufficiency Time` 和 `Doubling Time`。
2.  **模拟结果时序图 (Time-series Plot)**: 包含全局视图和基于 `detailed_var` 的细节视图。
3.  **模拟结束时各变量最终值 (Final Values Bar Chart)**: 各子系统在仿真结束时的氚库存分布条形图。
4.  **数据表格**: 包括最终值数据表和关键阶段（初始、转折点、结束）的数据切片。


## 4. AI 增强分析

`tricys` 的所有分析模块都深度集成了大型语言模型（LLM），能够将原始的图表和数据自动转化为结构化的学术风格报告。

### 4.1. 启用方式

对于基准工况分析，AI 功能是通过在 `post_processing` 任务的 `params` 对象中添加 `"ai": true` 来激活的。

```json
"post_processing": [
    {
       "module": "tricys.postprocess.baseline_analysis",
       "function": "baseline_analysis",
       "params": {
            "detailed_var": "sds.I[1]",
            "glossary_path": "../../example_glossary/example_glossary.csv",
            "ai": true
        }
    }
]
```

### 4.2. 环境配置

在使用此功能前，您必须在项目的**根目录**下创建一个名为 `.env` 的文件，并填入您的大语言模型 API 凭据。这确保了您的密钥安全，不会被提交到版本控制中。

```env
# .env file in project root
API_KEY="sk-your_api_key_here"
BASE_URL="https://your_api_base_url/v1"
AI_MODEL="your_model_name_here"
```

### 4.3. 输出报告

启用后，除了标准的分析报告 (`analysis_report_...md`)，`tricys` 还会在该案例的 `report` 文件夹内生成两份额外的报告：

- **`analysis_report_{case_name}_{model_name}.md`**: 在核心报告的基础上，末尾追加了由 AI 生成的对数据和图表的深度文字解读。
- **`academic_report_{case_name}_{model_name}.md`**: 一份完全由 AI 撰写的、结构严谨的学术风格报告。这份报告通常包含摘要、引言、方法、结果与讨论、结论等部分，可以直接作为汇报材料或论文初稿使用。