# Post-Processing Module

`tricys` not only runs simulations but also provides a powerful Post-Processing framework that allows you to automatically analyze results, generate reports, or perform other custom operations after the simulation tasks are complete.

The post-processing feature is enabled by adding a `post_processing` field to the configuration file.

## 1. Configuration File Example

The following example shows how to automatically execute two analysis tasks after a parameter sweep is completed: one using a built-in `module` and the other using a user-defined `script_path`.

```json
{
    "paths": {
        ...
    },
    "simulation": {
        ...
    },
    "simulation_parameters": {
        "blanket.TBR": "linspace:1:1.5:10"
    },
    "post_processing": [
        {
            "module": "tricys.postprocess.static_alarm",
            "function": "check_thresholds",
            "params": {
                "rules": [{"columns": ["sds.I[1]"], "min": 0.0}]
            }
        },
        {
            "script_path": "scripts/my_custom_analyzer.py",
            "function": "analyze_peak_value",
            "params": {
                "target_column_pattern": "sds.I[1]*",
                "report_filename": "peak_values.json"
            }
        }
    ]
}
```

## 2. Configuration Details

### 2.1. `post_processing`

- **Description**: This is a list (Array) where each element represents an independent post-processing step. These steps will be executed sequentially in the order they appear in the list.
- **Execution Timing**: After all simulation tasks (e.g., every run in a parameter sweep) are completed, `tricys` will load the aggregated results (`sweep_results.csv`) into a Pandas DataFrame and pass it to your specified post-processing functions.

### 2.2. Post-Processing Function

Each object in the list defines a Python function to be executed. You can specify the code to be called in one of two ways:

#### Method 1: `module` (Module Loading)

- `module` (string, required):
  - **Description**: The full path of the **module** where the Python function to be called is located. This requires your code to be a Python package or module that can be loaded with an `import` statement (e.g., installed or located in a directory with an `__init__.py` file).
  - **Example**: `tricys.postprocess.static_alarm`

#### Method 2: `script_path` (Script Path Loading)

- `script_path` (string, required):
  - **Description**: The path to a single Python **script file** containing the function to be called. This is more flexible and does not require your script to be part of a formal package.
  - **Example**: `scripts/my_custom_analyzer.py`

---

Regardless of which method you use, you need to provide the following fields:

- `function` (string, required):
  - **Description**: The name of the function to be called in the specified module or script.
  - **Example**: `check_thresholds`

- `params` (dictionary, optional):
  - **Description**: A dictionary containing keyword arguments to be passed to the target function.
  - **Example**: In the example above, `params` provides the `rules` argument to the `check_thresholds` function.

## 3. Built-in Post-Processing Modules

`tricys` comes with several common post-processing modules, located in the `tricys/postprocess` directory:

- **`rise_analysis`**: For analyzing dynamic characteristics of signals, such as rise time, fall time, and peak values.
- **`static_alarm`**: For checking if results exceed preset static thresholds (upper or lower limits).
- **`baseline_analysis`**: For performing baseline condition analysis.

## 4. Custom Post-Processing Modules

The greatest advantage of the post-processing framework is its extensibility. You can easily write your own Python scripts to perform any analysis you want.

### 4.1. Function Signature 

To be called correctly by the `tricys` framework, your custom post-processing function **must** follow a specific signature. The framework automatically passes two core pieces of data via **keyword arguments**:

1.  `results_df` (pd.DataFrame): A Pandas DataFrame containing the aggregated results of all simulation runs.
2.  `output_dir` (str): A dedicated output directory path for you to save reports, charts, and other analysis artifacts.

Therefore, your function signature must be able to accept these two arguments, as well as any other custom arguments you define in `params`.

A standard function signature is as follows:

```python
import pandas as pd

def my_custom_function(results_df: pd.DataFrame, output_dir: str, **kwargs):
    """
    A generic post-processing function signature.
    
    - results_df: The simulation results passed in by tricys.
    - output_dir: The directory provided by tricys for saving reports.
    - **kwargs: Used to receive all custom parameters from "params" in the JSON configuration.
    """
    # Get custom parameters from kwargs
    my_param = kwargs.get("my_param", "default_value")
    
    # Write your analysis code here...
    print(f"Analysis report will be saved in: {output_dir}")
    print(f"Received custom parameter my_param with value: {my_param}")
    print("Preview of result data:")
    print(results_df.head())
```

### 4.2. Complete Example

Let's create a complete custom post-processing script and show how to call it from the configuration using `script_path`.

**Step 1: Create the analysis script**

Suppose we create a file named `scripts/my_custom_analyzer.py` in our project:

```python
# scripts/my_custom_analyzer.py
import pandas as pd
import os
import json

def analyze_peak_value(results_df: pd.DataFrame, output_dir: str, target_column_pattern: str, report_filename: str = "peak_report.json"):
    """
    Finds the peak value in all matching columns and generates a report.
    """
    # Filter for columns that match the pattern
    target_columns = [col for col in results_df.columns if target_column_pattern in col]
    
    if not target_columns:
        print(f"Warning: No columns found matching '{target_column_pattern}'.")
        return

    # Calculate the peak value for each column
    peak_values = results_df[target_columns].max().to_dict()
    
    # Define the report output path
    report_path = os.path.join(output_dir, report_filename)
    
    # Save the report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(peak_values, f, indent=4)
        
    print(f"Peak value analysis report saved to: {report_path}")

```

**Step 2: Update the configuration file**

Now, configure the `post_processing` section in your `config.json` to call this script:

```json
{
    ...
    "post_processing": [
        {
            "script_path": "scripts/my_custom_analyzer.py",
            "function": "analyze_peak_value",
            "params": {
                "target_column_pattern": "sds.I[1]",
                "report_filename": "sds_peak_values.json"
            }
        }
    ]
}
```

When `tricys` finishes all simulations, it will automatically execute the `analyze_peak_value` function, calculate the peak values for all columns containing `sds.I[1]` in the simulation results, and finally save the results to the `post_processing/sds_peak_values.json` file.

In this way, you can seamlessly integrate any complex data analysis process into the automated workflow of `tricys`.

---

## 5. Next Steps

After mastering how to create and use post-processing modules, you can apply them to more complex scenarios:

- **[Sensitivity Analysis](../tricys_analysis/index.md)**: Write dedicated post-processing scripts for complex sensitivity analysis results to extract key metrics and generate visualizations.
- **[Co-Simulation](co_simulation_module.md)**: Integrate and analyze the results of co-simulations that include external modules.
---