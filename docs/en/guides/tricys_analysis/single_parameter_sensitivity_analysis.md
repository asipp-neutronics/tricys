# Single-Parameter Sensitivity Analysis

Single-parameter sensitivity analysis is one of the core features of `tricys`, designed to study how changes in a **single independent parameter** affect a series of user-defined **Key Performance Indicators (KPIs)**.

This feature automatically runs a series of simulations (one for each value of the independent parameter), calculates the performance metrics for each simulation result, and generates charts to visually represent the relationships between them. For common configurations such as metric definitions, glossaries, and unit maps, please refer to the [Automated Analysis Main Page](./index.md).

## 1. Configuration File Example

The core configuration for a single-parameter sensitivity analysis is located in the `analysis_cases` list. Each object represents an independent analysis case.

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
        // ... (Common configurations: metrics_definition, glossary_path, unit_map)
    }
}
```

## 2. Key Configuration Items Explained

- `independent_variable` (string): The full model path of the **independent parameter** to be scanned. This will be the X-axis of the analysis chart.
- `independent_variable_sampling` (list): A set of discrete values for the independent parameter. The program will run a simulation for each value in this list.
- `dependent_variables` (list): A list of dependent variables to be analyzed (i.e., the metric names defined in `metrics_definition`). This will be the Y-axis of the analysis chart.
- `plot_type` (string): The type of sensitivity chart to generate, typically `"line"`.
- `combine_plots` (boolean): Whether to plot the analysis results of multiple dependent variables on the same chart. `true` generates a composite chart with multiple subplots, while `false` generates a separate chart for each dependent variable.
- `sweep_time` (list): A list of raw variable names. For each variable in this list, the program will generate a "family plot," which draws the time evolution curve from **each** parameter sweep on the same graph, making it easy to compare differences in dynamic behavior.

## 3. Analysis Report Output

The structure of the analysis report is similar to that described in the [General Introduction](./index.md), but its core **performance metric analysis chart** has the following characteristics:

- The X-axis of the chart is the `independent_variable` you defined.
- The Y-axis is the performance metrics defined in `dependent_variables`.
- If `combine_plots` is `true`, the report will include a composite chart where each subplot shows the trend of one performance metric as the independent parameter changes.
- If `sweep_time` is defined, the report will also include a "family plot" showing the time evolution curves of the raw variable (e.g., `sds.I[1]`) for different values of the independent parameter.

## 4. Full Example Configuration
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

## 5. AI-Enhanced Analysis

All analysis modules in `tricys` are deeply integrated with Large Language Models (LLMs), capable of automatically converting raw charts and data into structured, academic-style reports.

### 5.1. How to Enable

In your analysis case configuration (i.e., within any object in the `analysis_cases` list or in the `params` of a `post_processing` task), add `"ai": true` to activate the AI analysis feature for that case.

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

### 5.2. Environment Setup

Before using this feature, you must create a file named `.env` in the project's **root directory** and fill in your Large Language Model API credentials. This ensures that your keys are kept secure and are not committed to version control.

```env
# .env file in project root
API_KEY="sk-your_api_key_here"
BASE_URL="https://your_api_base_url/v1"
AI_MODEL="your_model_name_here"
```

### 5.3. Output Reports

When enabled, in addition to the standard analysis report (`analysis_report_...md`), `tricys` will generate two additional reports in the case's `report` folder:

- **`analysis_report_{case_name}_{model_name}.md`**: Appends an in-depth textual interpretation of the data and charts, generated by the AI, to the end of the core report.
- **`academic_report_{case_name}_{model_name}.md`**: A well-structured, academic-style report written entirely by the AI. This report typically includes sections such as Abstract, Introduction, Methods, Results and Discussion, and Conclusion, and can be used directly for presentations or as a draft for a paper.
