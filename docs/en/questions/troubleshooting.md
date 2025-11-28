??? question "Q: Simulation failed, how to debug?"
    **1. Check the log file**:
    ```bash
    # View the log of the latest run
    tail -f results/<timestamp>/simulation_*.log
    ```

    **2. Common errors and solutions**:

    | Error Message | Possible Cause | Solution |
    |---|---|---|
    | `Model not found` | Incorrect model path | Check if `package_path` is correct |
    | `Failed to compile` | Modelica syntax error | Open the model in OMEdit to check for errors |
    | `Variable not found` | Variable in `variableFilter` does not exist | Check the variable name spelling and use the correct path |
    | `Out of memory` | Too many concurrent processes or model is too large | Reduce `max_workers` or increase system memory |
    | `Permission denied` | File permission issue | Check the read/write permissions of the working directory |

    **3. Enable detailed logging**:
    ```json
    {
        "logging": {
            "level": "DEBUG"
        }
    }
    ```

??? question "Q: GUI fails to start or display incorrectly?"
    **Windows/Linux Local**:
    * Make sure Tkinter is installed: `pip install tk`
    * Check the display environment variable: `echo $DISPLAY`

    **Docker Container**:
    * Windows 11: Make sure the WSLg feature of WSL2 is enabled
    * Linux: Run `xhost +local:` to allow the container to access X11
    * Use an image with GUI support: `tricys_openmodelica_gui`

??? question "Q: Parameter scan results are incomplete?"
    Possible reasons:

    1. **Some simulations failed**:
       * Check the error messages in the log file
       * Check if the parameter values are reasonable (e.g., avoid division by zero, negative values, etc.)

    2. **The output variable filter is too strict**:
       * Check if `variableFilter` matches the required variables

    3. **Concurrency issues**:
       * Try disabling concurrency: `"concurrent": false`
       * Check if any processes have crashed

??? question "Q: How to report a bug?"
    Please create a new issue in [GitHub Issues](https://github.com/asipp-neutronics/tricys/issues) and provide:

    1. **Environment information**:
       * Operating system and version
       * Python version
       * OpenModelica version
       * TRICYS version

    2. **Steps to reproduce**:
       * The complete configuration file
       * The command you ran
       * The model you used (if possible)

    3. **Error information**:
       * The full error stack trace
       * Relevant log snippets

    4. **Expected behavior**:
       * What you expected to happen
       * What actually happened
