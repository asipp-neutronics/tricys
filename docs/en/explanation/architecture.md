## 1. Overall Architecture

TRICYS adopts a layered architectural design, primarily consisting of the following layers:

```
┌─────────────────────────────────────────────┐
│               User Interface Layer            │
│  (tricys basic, tricys analysis, tricys gui)  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│             Simulation Execution Layer        │
│    (simulation, simulation_analysis)        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│                 Core Function Layer           │
│        (Jobs, Modelica, Interceptor)          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│           Analysis & Post-processing Layer    │
│         (Metric, Plot, Report, SALib)         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│                  Utility Layer                │
│     (Config, File, Log, SQLite Utils)       │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│              External Dependency Layer        │
│    (OpenModelica, Pandas, NumPy, SALib)     │
└─────────────────────────────────────────────┘
```

## 2. Program Layers

### 2.1. User Interface Layer

**Location**: `tricys/main.py`

**Responsibilities**:

- Provides two interaction methods: [Command-Line Interface (CLI) and Graphical User Interface (GUI)](../guides/quickstart.md#5-tricys-related-commands).
- Parses user-input command-line arguments and subcommands (`basic`, `analysis`, `gui`, `example`, `archive`, `unarchive`).
- Routes requests to the appropriate modules in the Simulation Execution Layer.
- Manages user sessions and configuration file loading.

**Key Functions**:

- **CLI Command Dispatch**: Automatically identifies the run mode based on subcommands or configuration file content.
- **GUI Interaction**: Provides a visual interface for parameter setting, simulation startup, and result viewing.
- **Example Runner**: Integrates interactive selection and execution of examples.

---

### 2.2. Simulation Execution Layer

**Location**: `tricys/simulation/`

**Responsibilities**:

- **Basic Simulation Mode** (`simulation.py`): Executes single or parameter-sweep simulation tasks.
- **Sensitivity Analysis Mode** (`simulation_analysis.py`): Executes various sensitivity analysis workflows.
- Manages the complete lifecycle of simulation tasks (initialization, execution, post-processing).
- Coordinates calls to the Core Function, Analysis, and Post-processing layers.

See also: [Simulation Flow](tricys_basic/simulation_flow.md) and [Analysis Flow](tricys_analysis/analysis_flow.md)

---

### 2.3. Core Function Layer

**Location**: `tricys/core/`

**Responsibilities**:

- **Modelica Interaction** (`modelica.py`): Communicates with the OpenModelica engine via OMPython.
- **Job Generation** (`jobs.py`): Generates parameter sweep tasks and simulation jobs based on the configuration.
- **Interceptor Mechanism** (`interceptor.py`): Generates and integrates interceptor models to enable [co-simulation](tricys_basic/co_simulation.md).

See also: [API Reference - Core Module](../api/tricys_core.md)

---

### 2.4. Analysis & Post-processing Layer

**Location**: `tricys/analysis/`, `tricys/postprocess/`

**Responsibilities**:

- **Performance Metric Calculation** (`analysis/metric.py`): Calculates [key metrics](tricys_analysis/performance_metrics.md) such as startup inventory, doubling time, and turning points.
- **Data Visualization** (`analysis/plot.py`): Generates time-series plots, parameter sweep plots, comparison plots, etc.
- **Sensitivity Analysis** (`analysis/salib.py`): Integrates the SALib library to perform various [sensitivity analysis methods](tricys_analysis/salib_integration.md).
- **Analysis Report Generation** (`analysis/report.py`): Automatically generates analysis [reports in Markdown format](tricys_analysis/analysis_report.md), with optional AI enhancement.
- **Post-processing Modules** (`postprocess/`): Provides extensible [data post-processing capabilities](../guides/tricys_basic/post_processing_module.md).


See also: [API Reference - Analysis Module](../api/tricys_analysis.md)

---

### 2.5. Utility Layer

**Location**: `tricys/utils/`

**Responsibilities**:

- **Configuration Management** (`config_utils.py`): Handles configuration file loading, validation, and preprocessing.
- **File Operations** (`file_utils.py`): Manages file paths, unique filename generation, and archiving.
- **Logging System** (`log_utils.py`): Provides structured logging and configuration recovery.
- **Database Operations** (`sqlite_utils.py`): Manages SQLite data storage and querying.


See also: [API Reference - Utilities](../api/tricys_utils.md)

---

### 2.6. External Dependency Layer

**Key Dependencies**:

- **OpenModelica**: The engine for compiling and executing Modelica models.
- **OMPython**: The interface library between Python and OpenModelica.
- **SALib**: A library for [sensitivity analysis and uncertainty quantification](tricys_analysis/salib_integration.md).
- **Pandas/NumPy**: Used for data processing and numerical computation.
- **Matplotlib/Seaborn**: Used for data visualization.
- **OpenAI** (Optional): For generating [AI-enhanced analysis reports](tricys_analysis/analysis_report.md).

**Responsibilities**:

- Provides the underlying simulation engine, numerical computation, and scientific computing support.
- Ensures cross-platform compatibility and high-performance computation capabilities.

---

## 3. Design Principles

1. **Modularity**: Each functional module has a single responsibility and is independent.
2. **Extensibility**: Easy to add new [post-processing modules](../guides/tricys_basic/post_processing_module.md), [performance metrics](tricys_analysis/performance_metrics.md), and [co-simulation handlers](../guides/tricys_basic/co_simulation_module.md#3-writing-your-own-handler).
3. **Configuration-Driven**: All simulation tasks are defined through [JSON configuration files](../guides/tricys_basic/basic_configuration.md).
4. **Automation**: Fully automated workflow from simulation to [analysis report generation](tricys_analysis/analysis_report.md).
5. **Openness**: Open-source design that encourages community contributions.