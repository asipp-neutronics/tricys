??? question "Q: Which operating systems are supported?"
    TRICYS supports the following operating systems:

    * **Windows 10/11** (WSL2 + Docker recommended)
    * **Ubuntu 20.04+**
    * **CentOS/Rocky Linux 8+**
    * **macOS** (via Docker)

??? question "Q: Is Docker mandatory?"
    No, it's not mandatory. Docker provides the easiest installation method, but you can also:

    * Install locally on Windows (requires OpenModelica and Python)
    * Install locally on Linux
    * Use WSL2 (Windows Subsystem for Linux)

    The advantage of Docker is environment isolation and being ready to use out-of-the-box, which is suitable for beginners.

??? question "Q: What version of Python is required?"
    TRICYS requires **Python 3.8 or higher**. Python 3.10 or 3.11 is recommended for optimal performance and compatibility.

??? question "Q: Is OpenModelica required?"
    Yes. TRICYS uses OpenModelica as its modeling and simulation backend. Please ensure that:

    1. OpenModelica is installed
    2. The `omc` command is accessible from the command line (added to PATH)

    **Verification method**:
    ```bash
    omc --version
    ```

??? question "Q: How to solve the \"omc command not found\" error?"
    **Windows**:
    1. Confirm the OpenModelica installation path (usually `C:\OpenModelica\bin`)
    2. Add it to the system environment variable PATH
    3. Restart the terminal or VSCode

    **Linux**:
    ```bash
    # Check the location of omc
    which omc

    # If not found, add it to PATH
    export PATH="/opt/openmodelica/bin:$PATH"

    # Or add it permanently to ~/.bashrc
    echo 'export PATH="/opt/openmodelica/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```

??? question "Q: What to do if Docker image download is slow?"
    You can use a local Docker image mirror:

    ```bash
    # Edit Docker configuration (Linux)
    sudo nano /etc/docker/daemon.json

    # Add mirror addresses
    {
      "registry-mirrors": [
        "https://docker.mirrors.ustc.edu.cn",
        "https://hub-mirror.c.163.com"
      ]
    }

    # Restart Docker
    sudo systemctl restart docker
    ```
