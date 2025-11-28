# Multi-Parameter Sensitivity Analysis

Building on single-parameter sensitivity analysis, `tricys` offers a more powerful **multi-parameter sensitivity analysis**. This feature allows you to study the impact of a primary independent parameter (X-axis) on performance metrics (Y-axis) while **simultaneously sweeping one or more background parameters**.

This enables you to generate a "family of sensitivity curves" on a single chart, where each curve represents a specific value of a background parameter. In this way, you can gain a deep understanding of the **interaction and coupling effects** between parameters. Additionally, this feature supports a powerful "goal-seeking" analysis mode.

For common configurations such as metric definitions, glossaries, and unit maps, please refer to the [Automated Analysis Main Page](./index.md).

## 1. Core Concept: Parameter Interaction Analysis

This is the most common use of multi-parameter analysis, and its core is to embed a `simulation_parameters` field inside a case within `analysis_cases`.

### 1.1. Configuration File Example
```json
{
    // ...
    "sensitivity_analysis": {
        "enabled": true,
        "analysis_cases": [
            {
                "name": "DIR_PLASMA_Analysis",
                "independent_variable": "tep_fep.to_SDS_Fraction[1]", // Primary independent parameter (X-axis)
                "independent_variable_sampling": [0.1, 0.3, 0.6, 0.8],
                "dependent_variables": [ "Startup_Inventory", "Required_TBR" ], // Dependent variables (Y-axis)
                "simulation_parameters": {
                    "plasma.fb": [0.02, 0.04, 0.08, 0.09, 0.1], // Background sweep parameter (generates multiple curves)
                    "plasma.nf": 0.5 // Fixed background parameter
                }
            }
        ],
        // ... (Common configurations)
    }
}
```

### 1.2. How It Works

- **Primary Independent Parameter (`independent_variable`)**: `tep_fep.to_SDS_Fraction[1]`, serves as the **X-axis** of the chart.
- **Background Sweep Parameters (`simulation_parameters`)**:
    - The value of `plasma.fb` is a **list**, which will become the **Legend** in the chart. Each curve corresponds to one value of `plasma.fb`.
    - The value of `plasma.nf` is a **scalar**, which will remain constant across all simulations.
- **Execution Logic**: The program executes a "nested loop". For each value of the `independent_variable`, the program runs a simulation for each value of `plasma.fb`. The total number of runs is `len(independent_variable_sampling) * len(plasma.fb)`.

## 2. Advanced Usage: Goal-Seeking Analysis

This feature supports a "reverse" analysis mode, known as **Goal-Seeking**. You can specify a performance metric as a target and solve for the required value of an input parameter to achieve that target.

### 2.1. Configuration File Example

The key is that `simulation_parameters` contains a special object with the same name as an optimization metric in `metrics_definition` (in this case, `Required_TBR`).

```json
{
    "name": "DoubleTime_PLASMA_Analysis",
    "independent_variable": "plasma.fb", // Primary independent parameter (X-axis)
    "dependent_variables": [ "Startup_Inventory", "Required_TBR" ], // Dependent variables (Y-axis)
    "simulation_parameters": {
        "plasma.nf": 0.5,
        "Required_TBR": { // Special goal-seeking configuration
            "metric_name": "Doubling_Time", // Target metric
            "metric_max_value": [4380, 8760, 13140, 17530] // List of target values (unit: hours)
        }
    },
    // ...
    "metrics_definition": {
        // ...
        "Required_TBR": {
            "method": "bisection_search",
            "parameter_to_optimize": "blanket.TBR", // Parameter to be solved for
            // ... (other configurations for bisection_search)
        }
    }
}
```

### 2.2. How It Works

- **Problem Description**: This configuration aims to answer the question: "As `plasma.fb` changes, what is the required `blanket.TBR` to achieve different `Doubling_Time` targets (4380h, 8760h, ...)?"
- **Execution Logic**: For each value of `plasma.fb` on the X-axis, the program initiates a `bisection_search` optimization loop for **each target value** in the `metric_max_value` list to solve for the corresponding `blanket.TBR` value.
- **Result Interpretation**: The final chart will show how the required TBR changes with `plasma.fb` under different doubling time target constraints.

## 3. Analysis Report Output

The main difference from single-parameter analysis lies in the **performance metric analysis chart**:

-   For **parameter interaction analysis**, the chart will contain a set of curves, with the legend corresponding to different values of the background parameter (e.g., `plasma.fb`).
-   For **goal-seeking analysis**, the chart will also contain multiple curves, but the legend will correspond to different performance target constraints (e.g., `Doubling_Time = 4380h`).

This allows multi-dimensional data relationships to be clearly presented in a single two-dimensional chart. The rest of the report follows the common structure.

## 4. Full Example Configuration
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