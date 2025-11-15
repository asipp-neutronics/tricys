??? question "Q: How to run a simulation?"
    TRICYS offers several ways to run simulations:

    **1. Command-Line Interface (CLI)**:
    ```bash
    # Use the default configuration file config.json
    tricys

    # Specify a configuration file
    tricys -c my_config.json

    # Run an analysis task
    tricys analysis -c analysis_config.json
    ```

    **2. Graphical User Interface (GUI)**:
    ```bash
    tricys gui
    ```

    **3. Interactive Examples**:
    ```bash
    # Run an interactive menu for all examples
    tricys example

    # Run basic simulation examples
    tricys basic example

    # Run analysis examples
    tricys analysis example
    ```

??? question "Q: How to understand the output files?"
    After the simulation is complete, the results are saved in a timestamped subdirectory under `results/`:

    | Filename | Description |
    |---|---|
    | `simulation_result.csv` | Detailed results for a **single parameter combination**, containing all variables over time. |
    | `sweep_results.csv` | Aggregated results for **multiple parameter combinations** (parameter sweep). |
    | `sensitivity_analysis_summary.csv` | **[Analysis Tasks Only]** Summary metrics for sensitivity analysis, where each row is the KPI for one run. |
    | `requierd_tbr_summary.csv` | **[Analysis Tasks Only]** Optimization results generated when performing goal-seeking tasks like "Required TBR". |
    | `simulation_*.log` | Detailed runtime log, including debugging information. |
    | `config.json` | A backup of the full configuration used for this run. |
    | `*_report.md` | **[Analysis Tasks Only]** Auto-generated AI analysis report. |
    | `*.png` / `*.csv` | Various charts and data exports. |

    **Result CSV File Structure**

    Whether running one or multiple parameter combinations, the final CSV files follow similar column naming conventions.

    *   **Base Case (No Scanned Parameters)**:
        If `simulation_parameters` is empty, the column names are simply the variable names defined in `variableFilter`.
        ```csv
        time,sds.I[1],blanket.I[1],...
        0.0,10.5,2.3,...
        ```

    *   **Parameter Sweep Case**:
        When `simulation_parameters` is not empty, the column names will have parameter information appended.
        ```csv
        time,sds.I[1]&blanket.TBR=1.05,sds.I[1]&blanket.TBR=1.1,...
        ```
        - **Column Name Format**: `<variable_name>&<param1>=<value1>&<param2>=<value2>...`
        - The `time` column remains unchanged.
        - Each variable under each parameter combination becomes a separate column. The column name is formed by joining the **variable name** and **parameter-value** pairs, separated by the `&` symbol.

??? question "Q: How to define complex parameter scans?"
    TRICYS supports several [parameter scan formats](../guides/tricys_basic/parameter_sweep.md):

    | Feature | Format | Example | Description |
    | :--- | :--- | :--- | :--- |
    | **Discrete List** | `[v1, v2, ...]` | `[6, 12, 18]` | A set of discrete values |
    | **Arithmetic Series** | `"start:stop:step"` | `"1.05:1.15:0.05"` | Start, stop, and step size |
    | **Linear Spacing** | `"linspace:start:stop:num"` | `"linspace:10:20:5"` | Generate `num` evenly spaced points |
    | **Logarithmic Spacing** | `"log:start:stop:num"` | `"log:1:1000:4"` | Generate `num` logarithmically scaled points |
    | **Read from File** | `"file:path:column"` | `"file:data.csv:voltage"` | Read from a specified column in a CSV file |

    **Example Configuration**:
    ```json
    {
        "simulation_parameters": {
            "blanket.TBR": [1.05, 1.1, 1.15, 1.2],
            "plasma.fb": "linspace:0.01:0.1:10",
            "tep_fep.to_SDS_Fraction[1]": "log:0.1:1.0:5"
        }
    }
    ```

??? question "Q: How to filter output variables?"
    Use the `variableFilter` parameter to select the variables to be saved. This parameter supports regular expressions, but be mindful of the syntax to match Modelica's variable naming rules.

    **Configuration Example**:
    ```json
    {
        "simulation": {
            "variableFilter": "time|sds.I[1]|blanket.I[1-5]|div.I[1-5]"
        }
    }
    ```

    **Common Patterns**:
    *   `time`: The time variable (must be included).
    *   `sds.I[1]`: Exact match for a single variable.
    *   `sds.I[1-5]`: Matches array variables from `sds.I[1]` to `sds.I[5]`.
    *   `blanket.I[1-5]|div.I[1-5]`: Matches multiple specific array variables.

??? question "Q: The simulation is very slow, how to speed it up?"
    You can take the following optimization measures:

    **1. [Enable concurrent execution](../guides/tricys_basic/concurrent_operation.md)**:
    ```json
    {
        "simulation": {
            "concurrent": true,
            "max_workers": 4
        }
    }
    ```

    **2. Reduce the number of output variables**:
    ```json
    {
        "simulation": {
            "variableFilter": "time|sds.I[1]"  # Only save key variables
        }
    }
    ```

    **3. Increase the time step** (trade-off with accuracy):
    ```json
    {
        "simulation": {
            "step_size": 1.0  # Increase from 0.5 to 1.0
        }
    }
    ```

    **4. Reduce the number of scan points**:
    ```json
    {
        "simulation_parameters": {
            "blanket.TBR": "linspace:1.05:1.15:5"  # Reduce from 20 to 5
        }
    }
    ```
