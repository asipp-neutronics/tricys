# Automated Analysis (TRICYS ANALYSIS)

`TRICYS ANALYSIS` is the advanced analysis module of `tricys`, designed to automate a series of complex simulation, post-processing, and report generation tasks. Unlike `TRICYS BASIC`, which executes a single simulation, the `ANALYSIS` module uses a unified configuration file to perform various advanced analysis tasks, from parameter sweeps and sensitivity analysis to uncertainty quantification.

This chapter will first introduce the core configuration items common to all analysis tasks, and then provide a detailed introduction to each specific analysis mode in separate sections.

## 1. Common Configuration Items

In the `tricys` analysis configuration file, the `sensitivity_analysis` object is at the core of all analysis tasks. The following are the key fields that are common across multiple analysis types.

### 1.1. Multi-Case Concurrent Execution

- **`analysis_cases` (List of Analysis Cases)**
  - **Description**: This is an array and the core of the `TRICYS ANALYSIS` mode. It allows you to define **multiple independent analysis tasks** in a single configuration file. Each element in the array is a complete analysis case object, which can have its own independent `name`, `independent_variable`, `dependent_variables`, etc. `tricys` will execute each case in this list sequentially.
  - **Application**: This feature is very useful when you need to compare different model versions, different initial conditions, or different analysis methods (for example, running a single-parameter sensitivity analysis and a SOBOL analysis in the same batch).

- **`concurrent_cases` (Concurrent Case Execution)**
  - **Description**: A boolean value (`true` or `false`), defaulting to `false`. When set to `true`, `tricys` will enable **multi-process parallel computing** to execute multiple cases from the `analysis_cases` list simultaneously.
  - **Application**: For configuration files containing a large number of independent analysis cases, enabling this option can leverage the power of multi-core CPUs to **significantly reduce the total analysis time**.

- **`max_case_workers` (Maximum Number of Concurrent Workers)**
  - **Description**: An integer that is effective only when `concurrent_cases` is `true`. It is used to specify the maximum number of processes for parallel execution.
  - **Default Value**: If not set, `tricys` will default to the number of CPU cores on your machine.
  - **Recommendation**: It is recommended to set this to a value no greater than the number of physical CPU cores on your computer for optimal performance.

### 1.2. `metrics_definition` (Metric Definitions)

This is the most critical part, used to define the Key Performance Indicators (KPIs) you care about, which are the **dependent variables** in the analysis results.

- **Structure**: A dictionary where each **key** is a unique name you assign to the metric (e.g., `Startup_Inventory`).
- **Value**: An object describing how to calculate that metric.
    - `source_column`: The source of the raw data for the calculation, i.e., the column name in the simulation results (`.csv`).
    - `method`: The name of the function in the `tricys.analysis.metric` module used for the calculation.
- **Details**: `tricys` has several built-in functions for common metric calculations. For a detailed explanation of the physical meaning and calculation methods of core performance metrics (like `Startup_Inventory`, `Doubling_Time`, etc.), please refer to [Core Performance Metrics Explained](../../explanation/performance_metrics.md).

### 1.3. `glossary_path` (Glossary)

- **Description**: A path to a "glossary" CSV file. Providing this file can greatly enhance the readability of the report, as it maps abbreviated variable names from the code (e.g., `sds.I[1]`) to easy-to-understand names and descriptions.
- **Format**: This is a standard CSV file, and its headers should include `Model Parameter` (required), `English Term`, `Chinese Translation`, etc.

### 1.4. `unit_map` (Unit Map)

- **Description**: A dictionary used to customize the units in the report charts, making the results more intuitive.
- **Key**: The variable or metric name.
- **Value**: An object containing `unit` (the unit string) and `conversion_factor` (the conversion factor from the original simulation unit to the target unit). For example, if the simulation time unit is hours, a `"conversion_factor": 24` can convert the unit of `Doubling_Time` to days.

### 1.5. AI-Enhanced Analysis (`"ai": true`)

All analysis modules in `tricys` have powerful built-in **AI analysis capabilities**.

- **How to Enable**: Add `"ai": true` to the configuration of a specific analysis case (e.g., within an element of `analysis_cases`, or in the `params` of a `post_processing` task) to activate it.
- **Environment Setup**: Before using this feature, you must create a file named `.env` in the project's **root directory** and fill in your Large Language Model (LLM) API credentials.
    ```env
    # .env file
    API_KEY="sk-your_api_key_here"
    BASE_URL="https://your_api_base_url/v1"
    AI_MODEL="your_model_name_here"
    ```
- **Functionality**: When enabled, `tricys` will, in addition to generating standard charts and data reports, make an extra call to the LLM to:
    1.  Provide an in-depth interpretation of the analysis results and append it to the core Markdown report.
    2.  Generate a well-structured **academic-style report** (`academic_report.md`) written entirely by the AI, which can be used directly for presentations or as a draft for a paper.

### 1.6. Common Configuration Example

The following JSON snippet shows how the above common configuration items are used in the `sensitivity_analysis` object.

```json
"sensitivity_analysis": {
    "enabled": true,
    "concurrent_cases": true,
    "max_case_workers": 4,
    "analysis_cases": [
        // ... Definition of specific analysis cases here, see docs for each analysis type ...
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

## 2. Common Analysis Report Output

Regardless of the type of analysis performed, `tricys` will create a folder named with a timestamp in the current working directory to store all results. For each analysis case defined in the `analysis_cases` configuration, a subfolder named after the case's `name` will be created inside the timestamp folder.

A typical output directory structure is as follows:

```
<run_timestamp>/
├── Case_A_Name/
│   ├── report/
│   │   ├── analysis_report_Case_A_Name.md              # Core analysis report
│   │   ├── academic_report_Case_A_Name_gpt-4.md      # (Optional) AI academic report
│   │   ├── analysis_plot_1.svg                         # Analysis plot 1
│   │   └── analysis_plot_2.svg                         # Analysis plot 2
│   └── results/
│       └── ... (Intermediate data files)
│
├── Case_B_Name/
│   ├── report/
│   │   └── ...
│   └── results/
│       └── ...
│
└── execution_report.md (Global execution report)
```

The core output of each case subfolder is located in its internal `report` folder, which usually contains the following:

### 2.1. Core Analysis Report

- **File**: `analysis_report_{case_name}.md`
- **Format**: Markdown
- **Content**: This is a summary of the analysis results, integrating configuration, charts, and data in a structured way. Its common content includes:
    - **Analysis Case Configuration Details**: A detailed list of all configuration parameters used for this analysis, ensuring the transparency and reproducibility of the analysis process.
    - **Summary Table of Performance Metrics**: A clear Markdown table listing each value of the independent variable and the corresponding calculated values of all dependent variables (performance metrics). This is the raw data source for all analysis charts.
    - **Analysis Charts**: Embedded vector graphics in **SVG format**. The specific type and content of the charts depend on the analysis mode (e.g., trend line charts for single-parameter analysis, bar charts of sensitivity indices for SOBOL analysis, etc.). Please refer to the documentation for each analysis type for details.

### 2.2. (Optional) AI-Enhanced Report

If `"ai": true` is configured, the `report` folder will additionally contain:

- **`analysis_report_{case_name}_{model_name}.md`**: An in-depth interpretation of the data and charts generated by the AI, appended to the end of the core report.
- **`academic_report_{case_name}_{model_name}.md`**: A well-structured, academic-style report written entirely by the AI, which can be used directly for presentations or as a draft for a paper.

## 3. Analysis Type Navigation

Depending on your research objectives, you can choose from the following different analysis modes. Please click the links to view detailed configuration methods and application scenarios for each mode.

- **[Baseline Condition Analysis](./baseline_condition_analysis.md)**: For a comprehensive evaluation of a single, fixed parameter configuration.
- **[Single-Parameter Sensitivity Analysis](./single_parameter_sensitivity_analysis.md)**: To study how changes in a single independent parameter affect system performance.
- **[Multi-Parameter Sensitivity Analysis](./multi_parameter_sensitivity_analysis.md)**: To analyze the interaction and coupling effects between multiple parameters, or to perform "goal-seeking" analysis.
- **[SOBOL Global Sensitivity Analysis](./sobol_global_sensitivity_analysis.md)**: To quantify the contribution of multiple input parameters and their interactions to the variance of the model output.
- **[Latin Uncertainty Quantification Analysis](./latin_uncertainty_analysis.md)**: To assess how uncertainty in input parameters propagates to the model output and to analyze the probability distribution of the output.