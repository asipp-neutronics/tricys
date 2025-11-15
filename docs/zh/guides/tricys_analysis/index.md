# 自动分析 (TRICYS ANALYSIS)

`TRICYS ANALYSIS` 是 `tricys` 的高级分析模块，旨在将一系列复杂的仿真、后处理和报告生成任务自动化。与执行单次仿真的 `TRICYS BASIC` 不同，`ANALYSIS` 模块通过一个统一的配置文件，可以执行从参数扫描、敏感性分析到不确定性量化等多种高级分析任务。

本章节将首先介绍各类分析任务中通用的核心配置项，然后分节详细介绍每一种具体的分析模式。

## 1. 通用配置项

在 `tricys` 的分析配置文件中，`sensitivity_analysis` 对象是所有分析任务的核心。以下是其中跨多种分析类型通用的关键字段。

### 1.1. 多案例并发执行

- **`analysis_cases` (分析案例列表)**
    - **描述**: 这是一个数组，也是 `TRICYS ANALYSIS` 模式的核心。它允许您在一个配置文件中定义**多个独立的分析任务**。每个数组元素都是一个完整的分析案例对象，可以有各自独立的 `name`、`independent_variable`、`dependent_variables` 等。`tricys` 会依次执行这个列表中的每一个案例。
    - **应用**: 当您需要对比不同模型版本、不同初始条件或不同分析方法（例如，在同一批次中运行一个单参数敏感性分析和一个SOBOL分析）时，此功能非常有用。

- **`concurrent_cases` (并发执行案例)**
    - **描述**: 一个布尔值 (`true` 或 `false`)，默认为 `false`。当设置为 `true` 时，`tricys` 会启用**多进程并行计算**，同时执行 `analysis_cases` 列表中的多个案例。
    - **应用**: 对于包含大量独立分析案例的配置文件，开启此选项可以利用多核CPU的优势，**显著缩短总分析时间**。

- **`max_case_workers` (最大并发进程数)**
    - **描述**: 一个整数，仅在 `concurrent_cases` 为 `true` 时生效。用于指定并行执行的最大进程数。
    - **默认值**: 如果不设置，`tricys` 会默认使用您机器上的CPU核心数。
    - **建议**: 建议设置为不超过您计算机的物理CPU核心数，以获得最佳性能。

### 1.2. `metrics_definition` (指标定义)

这是最关键的部分，用于定义您关心的性能指标 (KPIs)，也就是分析结果中的**因变量**。

- **结构**: 一个字典，其中每个**键**都是您为指标赋予的唯一名称（如 `Startup_Inventory`）。
- **值**: 一个描述如何计算该指标的对象。
    - `source_column`: 用于计算的原始数据来源，即仿真结果 (`.csv`) 中的列名。
    - `method`: `tricys.analysis.metric` 模块中用于计算的函数名。
- **详解**: `tricys` 内置了多种常用指标计算函数。关于内置核心性能指标（如 `Startup_Inventory`, `Doubling_Time` 等）的详细物理意义和计算方法，请参阅 [核心性能指标详解](../../explanation/performance_metrics.md)。

### 1.3. `glossary_path` (术语表)

- **描述**: 指向一个“术语表” CSV 文件的路径。提供此文件可以极大地增强报告的可读性，因为它会将代码中简写的变量名（如 `sds.I[1]`）映射为易于理解的中文名称和描述。
- **格式**: 这是一个标准的 CSV 文件，其列头应包含 `模型参数 (Model Parameter)` (必填), `英文术语 (English Term)`, `中文翻译 (Chinese Translation)` 等。

### 1.4. `unit_map` (单位映射)

- **描述**: 一个字典，用于自定义报告图表中的单位，使结果更直观。
- **键**: 变量名或指标名。
- **值**: 包含 `unit` (单位字符串) 和 `conversion_factor` (从原始仿真单位到目标单位的换算系数) 的对象。例如，如果仿真时间单位是小时，通过 `"conversion_factor": 24` 可以将 `Doubling_Time` 的单位换算为天。

### 1.5. AI 增强分析 (`"ai": true`)

`tricys` 的所有分析模块都内置了强大的 **AI 分析功能**。

- **启用方式**: 在具体分析案例的配置中（例如 `analysis_cases` 的某个元素内，或 `post_processing` 的 `params` 中）加入 `"ai": true` 即可激活。
- **环境准备**: 使用此功能前，必须在项目的**根目录**下创建一个名为 `.env` 的文件，并填入您的大语言模型（LLM）API 凭据。
    ```env
    # .env file
    API_KEY="sk-your_api_key_here"
    BASE_URL="https://your_api_base_url/v1"
    AI_MODEL="your_model_name_here"
    ```
- **功能**: 启用后，`tricys` 会在生成标准图表和数据报告的基础上，额外调用 LLM：
    1.  对分析结果进行深度解读，并将结果追加到核心的 Markdown 报告中。
    2.  生成一份完全由 AI 撰写的、结构严谨的**学术风格报告** (`academic_report.md`)，可直接用于汇报或作为论文初稿。

### 1.6. 通用配置示例

以下 JSON 片段展示了上述通用配置项在 `sensitivity_analysis` 对象中的实际应用。

```json
"sensitivity_analysis": {
    "enabled": true,
    "concurrent_cases": true,
    "max_case_workers": 4,
    "analysis_cases": [
        // ... 此处为具体分析案例的定义，详见各分析类型文档 ...
    ],
    "metrics_definition": {
        "Startup_Inventory": {
            "source_column": "sds.I[1]",
            "method": "calculate_startup_inventory"
        },
        "Doubling_Time": {
            "source_column": "sds.I[1]",
            "method": "calculate_doubling_time"
        },
        "Required_TBR": {
            "method": "bisection_search",
            "parameter_to_optimize": "blanket.TBR",
            "search_range": [1, 1.5]
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
        }
    }
}
```

## 2. 通用分析报告输出

无论执行何种类型的分析，`tricys` 都会在当前工作目录下创建一个以时间戳命名的文件夹，用于存放所有结果。对于在配置文件 `analysis_cases` 中定义的每一个分析案例，都会在时间戳文件夹内创建一个以案例 `name` 命名的子文件夹。

一个典型的输出目录结构如下所示：

```
<run_timestamp>/
├── Case_A_Name/
│   ├── report/
│   │   ├── analysis_report_Case_A_Name.md              # 核心分析报告
│   │   ├── academic_report_Case_A_Name_gpt-4.md      # (可选) AI 学术报告
│   │   ├── analysis_plot_1.svg                         # 分析图表1
│   │   └── analysis_plot_2.svg                         # 分析图表2
│   └── results/
│       └── ... (中间数据文件)
│
├── Case_B_Name/
│   ├── report/
│   │   └── ...
│   └── results/
│       └── ...
│
└── execution_report.md (全局执行报告)
```

每个案例子文件夹的核心产出都位于其内部的 `report` 文件夹中，通常包含以下内容：

### 2.1. 核心分析报告

- **文件**: `analysis_report_{case_name}.md`
- **格式**: Markdown
- **内容**: 这是分析结果的汇总，以结构化的方式整合了配置、图表和数据。其通用内容包括：
    - **分析案例配置详情**: 详细列出用于本次分析的所有配置参数，确保分析过程的透明和可复现。
    - **性能指标总表**: 一个清晰的 Markdown 表格，列出了独立变量的每一次取值，以及对应计算出的所有因变量（性能指标）的精确数值。这是所有分析图表的原始数据来源。
    - **分析图表**: 嵌入的 **SVG 格式**矢量图。图表的具体类型和内容取决于分析模式（例如，单参数分析的趋势线图、SOBOL分析的敏感性指数条形图等），具体请参阅各分析类型的文档。

### 2.2. (可选) AI 增强报告

如果配置了 `"ai": true`，`report` 文件夹中将额外出现：

- **`analysis_report_{case_name}_{model_name}.md`**: 在核心报告的基础上，末尾追加了由 AI 生成的对数据和图表的深度解读。
- **`academic_report_{case_name}_{model_name}.md`**: 一份完全由 AI 撰写的、结构严谨的学术风格报告，可直接用于汇报或作为论文初稿。

## 3. 分析类型导航

根据您的研究目的，可以选择以下不同的分析模式。请点击链接查看每种模式的详细配置方法和应用场景。

- **[基准工况分析](./baseline_condition_analysis.md)**: 对单一、确定的参数配置进行全面评估。
- **[单参数敏感性分析](./single_parameter_sensitivity_analysis.md)**: 研究单个独立参数的变化如何影响系统性能。
- **[多参数敏感性分析](./multi_parameter_sensitivity_analysis.md)**: 分析多个参数之间的交互和耦合效应，或进行“目标寻求”式分析。
- **[SOBOL全局敏感性分析](./sobol_global_sensitivity_analysis.md)**: 量化多个输入参数及其交互作用对模型输出方差的贡献。
- **[Latin不确定性量化分析](./latin_uncertainty_analysis.md)**: 评估输入参数的不确定性如何传播到模型输出中，并分析输出的概率分布。
