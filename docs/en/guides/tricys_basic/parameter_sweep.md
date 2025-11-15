# Parameter Sweep

Parameter sweep is one of the core features of `tricys`, allowing you to systematically study the effect of changes in one or more model parameters on the simulation results. You simply provide a set of values for each parameter of interest, and `tricys` will automatically create and run all possible combinations.

## 1. Configuration File Example

On top of the basic configuration, we just need to add a `simulation_parameters` field to define the parameter sweep.

```json
{
    "paths": {
        "package_path": "../../example_model_single/example_model.mo"
    },
    "simulation": {
        "model_name": "example_model.Cycle",
        "variableFilter": "time|sds.I[1]",
        "stop_time": 2000.0,
        "step_size": 0.5
    },
    "simulation_parameters": {
        "tep_fep.to_SDS_Fraction[1]": [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8],
        "blanket.TBR": "linspace:1.05:1.15:3"
    }
}
```

## 2. Configuration Details

- **Description**: The `simulation_parameters` item is a dictionary (a collection of key-value pairs) used to define the parameters to be swept and their corresponding values.
- **Key**: Must be the **full path** of a variable in the Modelica model. For example, `blanket.TBR` or `tep_fep.to_SDS_Fraction[1]`.
- **Value**: Can be in one of the following formats:
    1.  **List of Discrete Values**:
        -   **Format**: `[v1, v2, v3, ...]`
        -   **Example**: `[0.1, 0.15, 0.2]`
        -   **Description**: The program will run a simulation for each value in the list.

    2.  **Advanced Sweep Format (String)**:
        -   **Description**: `tricys` supports several compact string formats for generating numerical sequences, which are very suitable for defining linear, logarithmic, and other series.
        -   **Example**: `"linspace:1.05:1.15:3"` means generate 3 equally spaced numbers between 1.05 and 1.15.
        -   **Supported Formats**:

| Format | Syntax | Description |
| :--- | :--- | :--- |
| Range | `"start:stop:step"` | Generates an arithmetic sequence from `start` to `stop` with a step of `step`. E.g., `"1:5:2"` generates `[1, 3, 5]`. |
| Linspace | `"linspace:start:stop:num"` | Generates `num` equally spaced values between `start` and `stop`. E.g., `"linspace:0:10:3"` generates `[0, 5, 10]`. |
| Logspace | `"log:start:stop:num"` | Generates `num` logarithmically spaced values between `start` and `stop`, suitable for sweeps across orders of magnitude. E.g., `"log:1:100:3"` generates `[1, 10, 100]`. |
| Random | `"rand:min:max:count"` | Generates `count` uniformly distributed random numbers between `min` and `max`. E.g., `"rand:0:1:2"` might generate `[0.23, 0.87]`. |
| From File | `"file:path/to/data.csv:column_name"` | Reads numerical values from the `column_name` column of the specified CSV file to use as the sweep list. |
| Array Expansion | `"{val1, val2, ...}"` | A special format for setting multiple elements of a Modelica array at once. For example, setting the value `"{10, 25, 50}"` for a parameter `my_array` will be automatically expanded to `my_array[1]=10`, `my_array[2]=25`, `my_array[3]=50`. The values inside the curly braces can themselves be strings in other advanced formats. |

!!! tip "Multi-Parameter Sweep"
    - You can define multiple parameters to sweep simultaneously. `tricys` will calculate the **Cartesian product** of all parameter values to generate a list of simulation tasks covering all possible combinations.
    - In the example above, `tep_fep.to_SDS_Fraction[1]` has 7 values, and `blanket.TBR` has 3 values, so the program will run a total of `7 * 3 = 21` simulations.

## 3. Result Output

For a parameter sweep task, in addition to the `simulation_result.csv` file for each individual run, `tricys` will also generate a summary file `sweep_results.csv`, as shown below:

```
Working Directory/
└── {timestamp}/
    ├── log/        
        └── simulation_{timestamp}.log      # Run log
    ├── result/                 
        └── sweep_results.csv               # Aggregated simulation result data
    └── temp/
        ├── job_1/                      
            └── job_1_simulation_result.csv # Simulation result for task 1
        ├── job_2/                      
            └── job_2_simulation_result.csv # Simulation result for task 2
        └── ......
                                   
```

- **`sweep_results.csv`**:
  - **First column**: `time`, representing the time axis.
  - **Other columns**: Each column represents the simulation result for a specific parameter combination. The column headers clearly indicate the parameters and their values used for that run, for example, `sds.I[1]&tep_fep.to_SDS_Fraction[1]=0.1&blanket.TBR=1.05`, making it easy to compare results from different conditions directly in the CSV file.

---

## 4. Next Steps

After mastering parameter sweeps, you can explore more advanced features to improve efficiency and analysis depth:

- **[Concurrent Operation](concurrent_operation.md)**: Learn how to use multi-core processors to execute a large number of sweep tasks in parallel, significantly reducing simulation time.
- **[Post-Processing Module](post_processing_module.md)**: Learn how to automatically analyze sweep results, such as calculating the maximum value, average value, or number of alarms for each condition.
- **[Sensitivity Analysis](../tricys_analysis/index.md)**: Conduct more systematic studies of parameter impacts, such as Sobol global sensitivity analysis.