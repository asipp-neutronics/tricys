# TRICYS - Tritium Integrated Cycle Simulation Platform

[![license](https://img.shields.io/badge/license-Apache--2.0-green)](./LICENSE)
[![python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![docs](https://img.shields.io/badge/docs-English-blue.svg)](https://asipp-neutronics.github.io/tricys/)

**TRICYS** (**TR**itium **I**ntegrated **CY**cle **S**imulation) is an open-source, modular, multi-scale fusion reactor tritium fuel cycle simulator, designed to provide physics-based dynamic closed-loop analysis, strictly adhering to plant-wide mass conservation principles.

Our goal is to provide researchers and engineers with a flexible and robust platform to explore various tritium management strategies, optimize system designs, and gain deep insights into tritium flow and inventory dynamics in fusion reactor environments.

![Tritium Fuel Cycle System](./docs/en/assets/cycle_system.png)

## Features

- **Parameter Scanning & Concurrency**: Systematically investigate the impact of multiple parameters on system performance, supporting concurrent execution and large-scale batch simulations.
- **Sub-module Co-simulation**: Supports data exchange with external tools (such as Aspen Plus) to achieve sub-module system integration.
- **Automated Report Generation**: Automatically generates standardized Markdown analysis reports, including charts, statistics, and visualization results.
- **Advanced Sensitivity Analysis**: Supports custom sensitivity analysis of system parameters, integrating the SALib library to quantify the impact of parameters on outputs.
- **AI-Enhanced Analysis**: Integrates Large Language Models (LLM) to automatically transform raw charts and data into structured academic-style reports.

## Quick Start: Windows Local Installation

To ensure full compatibility with co-simulation features involving external Windows software like Aspen Plus, we prioritize and recommend Windows local installation.

### 1. Requirements
1.  **Python**: 3.8 or higher await (Recommend checking "Add Python to PATH" during installation).
2.  **Git**: For cloning the code repository.
3.  **OpenModelica**: Ensure its command-line tool (`omc.exe`) is added to the system's `PATH` environment variable.

### 2. Installation Steps

a. **Clone Project Repository**
   Open a terminal (e.g., PowerShell) and use `git` to clone the source code.
   ```shell
   git clone https://github.com/asipp-neutronics/tricys.git
   cd tricys
   ```

b. **Create and Activate Virtual Environment**
   To isolate project dependencies, it is recommended to create a separate Python virtual environment.
   ```shell
   # Create virtual environment
   py -m venv venv
   # Activate virtual environment
   .\venv\Scripts\activate
   ```

c. **Install Project Dependencies**
   Use `pip` to install `tricys` and all its dependencies in "editable" mode.
   ```shell
   pip install -e ".[win]"
   ```
   Alternatively, you can use the convenient script provided by the project:
   ```shell
   Makefile.bat win-install
   ```

### 3. Run an Example

After installation, you can launch the interactive example runner to quickly experience the core features of `tricys`.

```shell
tricys example
```
This command will scan and list all available basic and advanced analysis examples. You only need to enter the number as prompted to automatically run the corresponding example task.


## Alternative: Docker (Standard 0D Simulation)

If you do not require co-simulation with external Windows software, to simplify the development environment configuration, this project maintains two container images, supporting **VSCode & Dev Containers** to run and test code in a containerized environment:
1. [ghcr.io/asipp-neutronics/tricys_openmodelica_gui:docker_dev](https://github.com/orgs/asipp-neutronics/packages/container/tricys_openmodelica_ompython/476218036?tag=docker_dev): With OMEdit visualization application
2. [ghcr.io/asipp-neutronics/tricys_openmodelica_ompython:docker_dev](https://github.com/orgs/asipp-neutronics/packages/container/tricys_openmodelica_gui/476218102?tag=docker_dev): Without OMEdit visualization application

**To switch dev containers, please remove the original container and modify docker-compose.yml**
```
image: ghcr.io/asipp-neutronics/tricys_openmodelica_gui:docker_dev
```

### 1. Requirements
- **Docker**: Latest version.
- **VSCode**: Latest version, with [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension installed.

### 2. One-click Development Environment Setup

1.  **Clone Repository**:
    ```bash
    git clone https://github.com/asipp-neutronics/tricys.git
    cd tricys
    ```

2.  **Open in VSCode**:
    ```bash
    code .
    ```

3.  **Reopen in Container**: VSCode will detect the `.devcontainer` directory and prompt "Reopen in Container". Click that button.
    > When building the container for the first time, the specified Docker image needs to be downloaded, which may take some time.

4.  **Install Project Dependencies**: After the container starts successfully, execute the following command in the VSCode terminal to install the Python libraries required by the project.
    ```bash
    make dev-install
    ```


## Documentation

For more detailed feature introductions, configuration guides, and advanced tutorials, please visit our [Online Documentation](https://asipp-neutronics.github.io/tricys/en/).

## Contribution

We welcome any contributions from the community! If you wish to participate in the development of `tricys`, please follow these guidelines:

- **Code Style**: Use `black` for code formatting, `ruff` for style checking and fixing.
- **Naming Conventions**: Follow `snake_case` (variables/functions) and `PascalCase` (classes) conventions.
- **Docstrings**: All public modules, classes, and functions must include Google-style docstrings.
- **Testing**: Use `pytest` to write unit tests and ensure high coverage.
- **Git Commits**: Follow [Conventional Commits](https://www.conventionalcommits.org/) specification to keep commit history clear and readable.

## License

This project is licensed under the [Apache-2.0](./LICENSE) License.
