# TRICYS Analysis Workflow

The analysis workflow of `tricys` (`simulation_analysis.py`) is a powerful, analysis-oriented automated process. Built upon the core simulation functionality, it incorporates complex multi-mode scheduling, goal-seeking optimization, and report generation capabilities. This document provides a detailed breakdown of its internal workflow.

## 1. Core Workflow Diagram

Below is the complete workflow diagram for the `tricys` analysis process. It starts by reading the configuration, intelligently selects one of three primary operating modes, and executes the corresponding tasks until completion.

```mermaid
graph TD
    %% === 1. Startup Phase ===
    subgraph Sub_A ["A. Startup & Mode Selection"]
        A1[Start: main config.json] --> A2["Prepare Config & Logs<br>analysis_prepare_config()"]
        A2 --> A3{"<font size=4><b>Execution Mode Dispatch</b></font><br>Inside run_simulation()"}
    end

    %% Mode Dispatch Logic
    A3 --> B_GROUP{analysis_cases defined?}
    B_GROUP -- Yes --> C1
    B_GROUP -- No --> D_GROUP{Is it a SALib analysis?}
    
    D_GROUP -- Yes --> E1
    D_GROUP -- No --> F1

    %% === 2. Mode One ===
    subgraph Sub_C ["C. Mode 1: Multi-Case Analysis"]
        direction TB
        C1["<b>Mode 1: Multi-Case Analysis</b>"]
        C1 --> C2[Create isolated workspace<br>& config for each case]
        C2 --> C3{Run cases in parallel?<br>concurrent_cases}
        
        C3 -- Yes --> C4[Use <b>ProcessPoolExecutor</b><br>to call _execute_analysis_case in parallel]
        C3 -- No --> C5[Loop sequentially<br>and call _execute_analysis_case]
        
        %% Subgraph for single case execution
        subgraph Sub_C6 ["C6. Single Case Execution (_execute_analysis_case)"]
            direction TB
            C6_1[Change to isolated workspace] --> C6_2["<b>Recursively call run_simulation<br>(with inner concurrency disabled)</b>"] 
            C6_2 --> C6_3[Execute the full flow of Mode 3]
        end

        C4 --> C6_1
        C5 --> C6_1
        
        C6_3 --> C7[All cases completed]
        C7 --> C8["Consolidate reports<br>from all cases<br>consolidate_reports()"]
        C8 --> Z1[End]
    end

    %% === 3. Mode Two ===
    subgraph Sub_E ["E. Mode 2: SALib Analysis"]
        direction TB
        E1["<b>Mode 2: SALib Analysis</b>"] --> E2[Call run_salib_analysis]
        
        %% Subgraph for SALib flow
        subgraph Sub_E3 ["E3. SALib Internal Workflow"]
            direction TB
            E3_1[1. Generate parameter samples using SALib] --> E3_2[2. Run simulation for each sample] 
            E3_2 --> E3_3[3. Collect results and compute sensitivity indices]
        end
        
        E2 --> E3_1
        E3_3 --> Z2[End]
    end

    %% === 4. Mode Three ===
    subgraph Sub_F ["F. Mode 3: Standard Sweep & Analysis"]
        direction TB
        F1["<b>Mode 3: Standard Sweep & Analysis</b>"] --> F2[Generate list of simulation jobs]
        F2 --> F3{Run jobs in parallel?<br>concurrent}
        
        F3 -- Yes --> F4["Use <b>ThreadPoolExecutor (Standard Sim)</b><br>or <b>ProcessPoolExecutor (Co-Sim)</b><br>to run each job in parallel"]
        F3 -- No --> F5["Execute sequentially<br>_run_sequential_sweep()"]
        
        %% Subgraph for single job execution
        subgraph Sub_F6 ["F6. Single Job Execution (_run_..._job)"]
            direction TB
            F6_1[1. Run simulation in isolated workspace] --> F6_2{2. Optimization goal configured?<br>(e.g., Required_...)}
            
            F6_2 -- Yes --> F6_3["<b>Optimization Sub-flow</b><br>Call _run_bisection_search_for_job"]
            
            %% Nested subgraph for bisection search
            subgraph Sub_F6_4 ["F6_4. Bisection Search Loop"]
                F6_4_1[a. Iteratively run simulations<br>within search range] --> F6_4_2[b. Check if metric meets target] 
                F6_4_2 --> F6_4_3[c. Narrow search range]
                F6_4_3 --> F6_4_1
            end
            
            F6_3 --> F6_4_1
            F6_4_2 -- Met or Finished --> F6_5
            
            F6_5["3. Return simulation result path<br>& <b>optimization results</b>"]
            F6_2 -- No --> F6_5
        end

        F4 --> F6_1
        F5 --> F6_1

        F6_5 --> F7[All jobs completed]
        F7 --> F8["<b>Result Aggregation & Post-processing</b>"]
        
        %% Subgraph for subsequent steps
        subgraph Sub_F9 ["F9. Subsequent Steps"]
            direction TB
            F9_1[a. Merge simulation results to sweep_results.csv] --> F9_2[b. Merge optimization results to requierd_tbr_summary.csv]
            F9_2 --> F9_3["c. Run sensitivity analysis<br>_run_sensitivity_analysis()<br>(Extract metrics, generate plots)"]
            F9_3 --> F9_4["d. Execute custom post-processing<br>_run_post_processing()"]
        end
        
        F8 --> F9_1
        F9_4 --> Z3[End]
    end

    %% Style Definitions
    style C1 fill:#e3f2fd,stroke:#333,stroke-width:2px
    style E1 fill:#e8f5e9,stroke:#333,stroke-width:2px
    style F1 fill:#fbe9e7,stroke:#333,stroke-width:2px
```

## 2. Detailed Workflow Steps

### 2.1. Startup and Mode Selection

The entire process begins in the `main` function, which is responsible for loading and preprocessing the configuration file (`analysis_prepare_config`) and setting up the logging system. The core logic resides in the `run_simulation` function, which first performs **mode dispatch** to decide which core workflow to execute next.

---

### 2.2. Mode 1: Multi-Case Analysis

This mode is activated when `analysis_cases` is defined in the configuration file. It is used to execute a series of independent, comparable analysis studies.

1.  **Environment Setup**: The framework creates a completely separate sub-workspace for each case defined in `analysis_cases` and generates a customized configuration file for each. This ensures that the execution environment for each case (including model modifications, temporary files, and results) is isolated.
2.  **Concurrent Execution**: If `"concurrent_cases": true` is configured, `tricys` starts a **process pool** (`ProcessPoolExecutor`) to execute all cases in parallel. Using processes is crucial here, as each case is a full `tricys` run instance that requires independent memory space and file system permissions to avoid conflicts.
3.  **Recursive Call**: The execution of each case is wrapped by the `_execute_analysis_case` function, which **recursively calls `run_simulation`** while forcing internal concurrency to be disabled (to prevent nested process pools). This means each case internally executes the complete workflow of "Mode 3".
4.  **Report Consolidation**: After all cases are completed, the framework calls functions like `consolidate_reports` to collect the analysis results and reports from all sub-workspaces and generate a top-level summary report, facilitating cross-case comparisons for the user.

---

### 2.3. Mode 2: SALib Global Sensitivity Analysis

If the configuration points to a Global Sensitivity Analysis (GSA) task (identified by keywords like `independent_variable_sampling` and `analyzer`), `tricys` hands over control to a dedicated SALib workflow.

1.  **Call `run_salib_analysis`**: This is the entry point for the SALib process.
2.  **Parameter Sampling**: It uses the SALib library (e.g., `saltelli.sample`) to generate a large set of parameter samples based on the configured parameter distributions.
3.  **Batch Simulation**: It runs a Modelica simulation for each parameter sample.
4.  **Result Analysis**: It collects all simulation results and uses a SALib analyzer (e.g., `sobol.analyze`) to calculate the first-order, second-order, and total-order sensitivity indices (S1, S2, ST) for each parameter.
5.  **Output Report**: The sensitivity indices and related plots are outputted as the final analysis report.

---

### 2.4. Mode 3: Standard Sweep and Analysis

This is the most fundamental and core workflow, executed when the conditions for the other two modes are not met.

1.  **Job Generation**: A list of simulation jobs is generated based on the `simulation_parameters`.
2.  **Job Execution**:
    *   Based on the `concurrent` configuration, all `jobs` are executed either in parallel or sequentially.
    *   The execution of each job is handled by functions like `_run_single_job` or `_run_co_simulation`.
3.  **Optimization Sub-flow (Core Feature)**: After a single simulation job is completed, the system checks if an optimization goal is configured (metrics prefixed with `Required_`).
    *   If yes, the system immediately calls `_run_bisection_search_for_job` to start a **bisection search loop**.
    *   This loop **iteratively runs multiple simulations** to find the optimal parameter value that satisfies a predefined metric (e.g., a TBR > 1.05), adjusting the parameter based on the results of each iteration until a solution is found or the maximum number of iterations is reached.
4.  **Result Aggregation**:
    *   After all tasks (including all simulations within the optimization sub-flow) are complete, the framework aggregates two types of data:
        *   The raw time-series data from all simulations is merged into `sweep_results.csv`.
        *   The final results of all optimization tasks (e.g., the optimal `enrichment` for `TBR>1.05` is `0.85`) are merged into `requierd_tbr_summary.csv`.
5.  **Final Analysis and Post-processing**:
    *   `_run_sensitivity_analysis` is called to load the aggregated data, calculate final Key Performance Indicators (KPIs), and generate analysis plots and summaries.
    *   `_run_post_processing` is called to execute any user-defined custom scripts (e.g., to generate reports in a specific format).

