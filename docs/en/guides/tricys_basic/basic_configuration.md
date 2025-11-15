# Basic Configuration

All simulation tasks in TRICYS are driven by a JSON configuration file. This file details every aspect of the process, including model paths, simulation parameters, variable sweeps, and post-processing.

This section introduces the most basic configuration file for running a single simulation. Mastering the basic configuration is the first step to using TRICYS.

---

## 1. Configuration File Example

A minimal configuration file is shown below. It defines which model to run and how to run it.

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
    }
}
```

This configuration will run a simulation of a tritium fuel cycle model for a duration of 2000 hours, outputting results every 0.5 hours, and saving only the time and the SDS system's tritium inventory variables.

---

## 2. Required Configuration Items

### 2.1. `paths` (Path Configuration)

This section is used to define all settings related to file paths.

#### `package_path` (string, required)

- **Description**: Points to the root `package.mo` file of the Modelica model. TRICYS will load and parse your model from here.
- **Path Type**: Can be an absolute path or a relative path from the location of the JSON configuration file.
- **Examples**:
  ```json
  "package_path": "C:/Models/example_model/package.mo"  // Absolute path (Windows)
  "package_path": "/home/user/models/package.mo"         // Absolute path (Linux)
  "package_path": "../models/package.mo"                 // Relative path
  ```

!!! tip "Path Separators"
    In JSON files, Windows paths can use either forward slashes `/` or double backslashes `\`:
    ```json
    "package_path": "C:/Models/package.mo"      // Recommended
    "package_path": "C:\\Models\\package.mo"    // Also works
    ```

---

### 2.2. `simulation` (Simulation Configuration)

This section contains the core parameters required to run the simulation.

#### `model_name` (string, required)

- **Description**: The full name of the Modelica model to be simulated. It follows the `PackageName.ModelName` format.
- **Format**: `<PackageName>.<ModelName>`
- **Example**:
  ```json
  "model_name": "example_model.Cycle"
  ```
  In this example, `example_model` is the package name (corresponding to `package.mo`), and `Cycle` is the specific model defined within it.

!!! warning "Model Name Must Match Exactly"
    The model name is case-sensitive and must exactly match the name defined in the Modelica file.

#### `variableFilter` (string, required)

- **Description**: A regular expression used to filter the output results. Only variables whose names match this expression will be saved to the final `.csv` result file.
- **Format**: Use a vertical bar `|` to separate multiple variable names or patterns.
- **Examples**:
  ```json
  // Save only time and one variable
  "variableFilter": "time|sds.I[1]"
  
  // Save time and an array variable
  "variableFilter": "time|sds.I[1-5]"

  // Save multiple specific variables
  "variableFilter": "time|sds.I[1]|blanket.I[1-5]|div.I[1-5]"
  
  ```

!!! tip "Recommendation"
    To reduce output file size and improve performance, it is recommended to save only the variables you actually need to analyze.

#### `stop_time` (float, required)

- **Description**: The total duration of the simulation. The simulation will run from time `0` to `stop_time`.
- **Unit**: hours

#### `step_size` (float, required)

- **Description**: The time step of the simulation. This is also the interval at which results are output.
- **Trade-off**: 
  - Smaller step size: higher accuracy, but longer simulation time and larger output files.
  - Larger step size: faster, but may miss fast-changing details.

---

## 3. Default Configuration Items

In addition to the required items above, TRICYS provides a series of optional configurations with reasonable default values. If these options are not set, the system will automatically use the following default behaviors:

### 3.1. `paths` (Path Configuration)

| Parameter | Description | Default Value |
| :--- | :--- | :--- |
| `results_dir` | Directory name for storing simulation results. | `"results"` |
| `temp_dir` | Directory name for storing temporary files. | `"temp"` |
| `log_dir` | Directory name for storing log files. | `"log"` |
| `db_path` | Path to the SQLite database file for storing and reading model parameters. | Dynamically created in the temporary directory at each run. |

!!! info "Output Directory Structure"
    TRICYS will create a main run directory named with a timestamp in the current working directory (e.g., `20250116_103000/`). All the above output directories (`results`, `temp`, `log`) will be created inside this timestamped directory by default, ensuring that the outputs of each run are isolated from one another.

### 3.2. `simulation` (Simulation Configuration)

| Parameter | Description | Default Value |
| :--- | :--- | :--- |
| `concurrent` | Whether to enable concurrent (parallel) simulation, used to speed up parameter sweep tasks. | `false` |
| `max_workers` | If concurrency is enabled, this option specifies the maximum number of parallel worker processes. | Half the number of system CPU cores |
| `keep_temp_files` | Whether to keep temporary files (like model compilation files) after the simulation is finished. Very useful for debugging. | `true` |

### 3.3. `logging` (Logging Configuration)

| Parameter | Description | Default Value |
| :--- | :--- | :--- |
| `log_level` | The minimum level for logging. Options include "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL". | `"INFO"` |
| `log_to_console` | Whether to output logs to the console in real-time. | `true` |
| `log_count` | The maximum number of old log files to keep in the log directory. | `5` |

---

## 4. Configuration Templates

Here are some configuration templates for common scenarios that you can copy and use directly:

### 4.1 Quick Test Template

This template is suitable for quickly verifying if a model can run correctly. It saves all variables and runs for a short duration.

```json
{
    "paths": {
        "package_path": "path/to/your/package.mo"
    },
    "simulation": {
        "model_name": "YourModel.Name",
        "variableFilter": "time|.*",
        "stop_time": 100.0,
        "step_size": 1.0
    }
}
```

### 4.2 Production Environment Template

This template is suitable for formal, long-duration simulations. It saves only key variables, outputs results to a specified production directory, and configures stricter logging and debugging options.

```json
{
    "paths": {
        "package_path": "path/to/your/package.mo",
        "results_dir": "production_results"
    },
    "simulation": {
        "model_name": "YourModel.Name",
        "variableFilter": "time|key_var1|key_var2",
        "stop_time": 86400.0,
        "step_size": 10.0,
        "keep_temp_files": false
    },
    "logging": {
        "log_level": "INFO",
        "log_to_console": false,
        "log_count": 10
    }
}
```

---

## 5. How to Run

After configuring the file, there are two ways to run the simulation:

### 5.1. Using the Default Configuration File

Save the configuration file as `config.json` and place it in the project root directory:

```bash
tricys
```

### 5.2. Specifying a Configuration File

```bash
tricys -c my_config.json
```

---

## 6. Viewing Results

After the simulation is complete, the results are saved in a timestamped subdirectory:

```
Working Directory/
└── {timestamp}/
    ├── log/        
        └── simulation_{timestamp}.log  // Run log
    ├── result/                 
        └── simulation_result.csv       // Simulation result data
    └── temp/
        └── job_1/                      // Temporary job data
                                   
```

### 6.1. Analyzing Results with Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Read the results
df = pd.read_csv('results/20250116_103000/simulation_result.csv')

# View the data
print(df.head())
print(f"Simulation duration: {df['time'].max()} seconds")
print(f"Final tritium inventory: {df['sds.I[1]'].iloc[-1]:.2f} g")

# Plot the tritium inventory change curve
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['sds.I[1]'], label='SDS Tritium Inventory')
plt.xlabel('Time (seconds)')
plt.ylabel('Tritium Inventory (g)')
plt.title('SDS Tritium Inventory Over Time')
plt.legend()
plt.grid(True)
plt.savefig('inventory_plot.png', dpi=300)
plt.show()
```

### 6.2. Viewing with Excel

Directly open the `simulation_result.csv` file with Microsoft Excel or LibreOffice Calc.

---

## 7. Next Steps

After mastering the basic configuration, you can proceed to learn about:

- **[Parameter Sweep](parameter_sweep.md)**: Systematically study the impact of parameters on the results.
- **[Concurrent Operation](concurrent_operation.md)**: Speed up large-scale simulations.
- **[Post-Processing Module](post_processing_module.md)**: Automate analysis and report generation.
- **[Co-Simulation](co_simulation_module.md)**: Integrate with external software.

---