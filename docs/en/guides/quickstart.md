# Quick Start

This guide will walk you through the installation process for `tricys` in a **Windows** environment and run a basic command-line simulation.

## 1. Prerequisites

Before you begin, please ensure your system meets the following requirements:

- **Python**: Version 3.8 or higher.
- **Git**: For cloning the project repository.
- **OpenModelica**: OpenModelica needs to be installed, and its command-line tool (`omc.exe`) must be added to the system's `PATH` environment variable.

!!! tip "Tip"
    When installing Python on Windows, be sure to check the **"Add Python to PATH"** option to ensure that the `python` and `pip` commands are available in the terminal.

## 2. Installation Steps

### a. Clone the Project Repository

Open your terminal (e.g., PowerShell or Cmd) and use `git` to clone the `tricys` source code.

```shell
git clone https://github.com/asipp-neutronics/tricys.git
cd tricys
```

### b. Create and Activate a Virtual Environment

In the project root directory, create an independent Python virtual environment to isolate project dependencies.

```shell
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

After activation, you will see `(venv)` appear before your terminal prompt.

### c. Install Project Dependencies

Use `pip` to install `tricys` and all its development dependencies. The `-e` flag installs it in "editable" mode, which means any changes you make to the source code will take effect immediately.

```shell
pip install -e ".[win]"

or # or use the Makefile.bat script to install dependencies

Makefile.bat win-install
```


## 3. Run an Example

`tricys` provides an interactive example runner to help you quickly explore and run all available examples, including basic simulations and advanced analysis tasks. This is the easiest way to verify your installation and understand the capabilities of `tricys`.

### a. Start the Example Runner

In a terminal with the virtual environment activated, execute the following command to start the example runner:

```shell
tricys example
```

### b. Select and Run an Example

This command starts a unified example runner that scans and lists all available examples from the `example/basic` and `example/analysis` directories.

You will see a menu similar to the one below:

```text
============================================================
         TRICYS Unified Example Runner
============================================================

  1. [BASIC] Basic Configuration
     Description: A basic simulation with a single run
     Config: basic_configuration.json

  2. [BASIC] Parameter Sweep
     Description: A multi-run simulation with parameter sweeps
     Config: parameter_sweep.json

  ...

  6. [ANALYSIS] Baseline Condition Analysis
     Description: Baseline condition analysis for TBR search
     Config: baseline_condition_analysis.json

  ...

  0. Exit
  h. Show help
  s. Rescan example directories

============================================================
```

- **Enter a number** (e.g., `1`) and press Enter to run the corresponding example.
- The program will automatically copy the example files to the `test_example` folder in the project root directory and execute the task there.
- This design ensures that the original example files are not modified and keeps the workspace clean.

### c. View the Results

After the task is completed, the results will be saved in the `test_example` directory, inside the subfolder for the respective example.

For example, if you ran the `Basic Configuration` example, the results will be located in the `test_example/basic/1_basic_configuration/` directory. There you will find:

- `simulation_result.csv`: Contains data for all output variables over time.
- `simualtion_{timestamp}.log`: A detailed log file for this run.
- `basic_configuation.json`: A backup of the full configuration used for this run.

## 4. Run the Graphical User Interface (GUI)

If you prefer a graphical interface, you can start the `tricys` GUI.

```shell
tricys gui
```

The GUI provides an interactive interface for loading models, setting parameters, defining sweep ranges, and starting simulations.

## 5. TRICYS Related Commands

Besides using the example runner, you can also use `tricys`'s various subcommands to perform specific tasks directly. Here is a detailed list of all available commands:

| Command / Usage | Description |
| :--- | :--- |
| **Main Commands / Usage** | |
| `tricys` | Requires specifying a configuration file via the `-c` argument. The program automatically detects the file content and executes the corresponding workflow (standard simulation or analysis). |
| `tricys basic` | Runs a standard simulation. Requires specifying a configuration file via the `-c` argument, or the existence of a default `config.json` in the current directory. |
| `tricys analysis` | Runs a simulation analysis. Requires specifying a configuration file via the `-c` argument, or the existence of a default `config.json` in the current directory. |
| `tricys gui` | Launches the interactive Graphical User Interface (GUI). |
| `tricys example` | Runs an interactive example runner capable of starting all types of examples. |
| `tricys archive <timestamp>` | Archives the simulation or analysis results of the specified `timestamp` into a zip file. |
| `tricys unarchive <zip_file>` | Extracts a previously archived run file (`zip_file`). |
| **`analysis` Subcommands** | |
| `tricys analysis retry <timestamp>` | Retries the analysis for an existing report where the AI analysis failed, based on its `timestamp`. |
| `tricys analysis example` | Runs an interactive analysis example. |
| **`basic` Subcommands** | |
| `tricys basic example` | Runs an interactive basic function example. |
| **General Arguments** | |
| `-h, --help` | Shows the help message and exits. |
| **Implicit Behavior** | |
| `(No command)` | If `-c` or other subcommands are not provided, the program searches for and uses the `config.json` file in the current directory. |

![tricys_command.svg](../../assets/tricys_command.svg)

---

Congratulations! You have successfully installed and run `tricys`. Next, you can explore more advanced features like [Parameter Sweep](tricys_basic/parameter_sweep.md) or [Co-Simulation](tricys_basic/co_simulation_module.md).