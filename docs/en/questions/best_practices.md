??? question "Q: How to organize large simulation projects?"
    Recommended directory structure:

    ```
    my_fusion_project/
    ├── models/                 # Modelica models
    │   ├── package.mo
    │   └── ...
    ├── configs/                # Configuration files
    │   ├── baseline.json
    │   ├── sensitivity.json
    │   └── ...
    ├── scripts/                # Helper scripts
    │   ├── prepare_data.py
    │   └── post_analysis.py
    ├── data/                   # Input data
    │   └── external_data.csv
    ├── results/                # Simulation results (auto-generated)
    └── reports/                # Final reports
    ```

??? question "Q: How to version control configuration files?"
    Use Git for version control:

    ```bash
    # Initialize Git repository
    git init

    # Create .gitignore
    echo "results/" >> .gitignore
    echo "*.log" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo "__pycache__/" >> .gitignore

    # Commit configuration files
    git add configs/*.json
    git commit -m "feat: add baseline configuration"
    ```

??? question "Q: How to run distributed simulations on multiple machines?"
    Currently, TRICYS does not directly support distributed computing, but you can:

    1. **Manually split tasks**:
       * Divide the parameter scan into multiple subsets
       * Run different subsets on different machines
       * Manually merge the results

    2. **Use a cluster scheduler** (like SLURM):
       * Submit each parameter combination as a separate job
       * Use a post-processing script to aggregate the results

??? question "Q: How to optimize model performance?"
    **At the Modelica model level**:
    1. Simplify the model structure, avoid overly complex equations
    2. Use appropriate numerical solver settings
    3. Avoid algebraic loops and excessive events

    **At the TRICYS level**:
    1. [Enable concurrent execution](../guides/tricys_basic/concurrent_operation.md)
    2. Reduce the number of output variables
    3. [Use co-simulation to replace complex subsystems](../guides/tricys_basic/co_simulation_module.md)
