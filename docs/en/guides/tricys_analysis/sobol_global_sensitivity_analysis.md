# SOBOL Global Sensitivity Analysis

Global Sensitivity Analysis (GSA) is a powerful tool for quantifying the contribution of model input parameters (and their interactions) to the variance of the model output. Unlike local sensitivity analysis, which changes one parameter at a time, GSA explores the entire parameter space simultaneously.

`tricys` integrates the industry-standard `SALib` library, providing direct support for the **Sobol** method. Sobol is a variance-based GSA method that can efficiently calculate the contribution of each parameter to the model output's uncertainty, including both the independent influence of parameters and their interaction effects.

For common configurations such as metric definitions, glossaries, and unit maps, please refer to the [Automated Analysis Main Page](./index.md).

## 1. Configuration File Example

The configuration for a SOBOL analysis is significantly different from previous sensitivity analyses, mainly in that `independent_variable` becomes a list, `independent_variable_sampling` becomes an object, and a new `analyzer` field is added.

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
        // ... (Common configurations)
    }
}
```

## 2. Key Configuration Items Explained

- `independent_variable` (list): A list containing **all** input parameters to be included in the global sensitivity analysis.
- `independent_variable_sampling` (object): A dictionary that defines the sampling range and probability distribution for each parameter in the `independent_variable` list.
  - **Key**: The full path of the parameter.
  - **Value**: An object containing `bounds` (a list `[min, max]`) and `distribution` (a string, e.g., `"unif"`).
- `analyzer` (object): Defines the GSA method to be used and its parameters.
  - `method`: For Sobol analysis, this is fixed as `"sobol"`.
  - `sample_N`: The base number of samples `N` required for Sobol sampling.
    - ⚠️ **Important**: The total number of simulations to be run will be **`N * (2D + 2)`**, where `D` is the number of `independent_variable`s. For example, in this case, `D=5` and `N=256`, so the total number of simulations will be `256 * (2*5 + 2) = 3072`.

## 3. Analysis Report Output

The core content of the report is an independent analysis result for **each** dependent variable (performance metric). For example, for the `Startup_Inventory` metric, the report will include:

1.  **Sobol Sensitivity Indices Table**: A Markdown table that precisely lists the first-order (S1) and total-order (ST) sensitivity indices and their confidence intervals for each input parameter.
2.  **Sensitivity Indices Chart**: An embedded bar chart that visually compares the S1 and ST indices of each parameter, making it easy to quickly identify key influencing factors.

### How to Interpret Sobol Indices

-   **S1 (First-order index)**: The **independent contribution** of the parameter. A higher S1 value indicates that the parameter has a greater **direct impact** on the model output.
-   **ST (Total-order index)**: The **total contribution** of the parameter, including its independent effect and its **interaction** with all other parameters.
-   **Interaction**: The difference `ST - S1` can be used to approximate the **strength of the interaction effect** of that parameter with others. If a parameter's `ST` is much larger than its `S1`, it means that much of its influence is realized through coupling and synergistic effects with other parameters.

## 4. Full Example Configuration
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