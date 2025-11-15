??? question "问：支持哪些操作系统？"
    TRICYS 支持以下操作系统：

    * **Windows 10/11**（推荐使用 WSL2 + Docker）
    * **Ubuntu 20.04+**
    * **CentOS/Rocky Linux 8+**
    * **macOS**（通过 Docker）

??? question "问：必须使用 Docker 吗？"
    不是必须的。Docker 提供了最简便的安装方式，但您也可以：

    * 在 Windows 上本地安装（需要安装 OpenModelica 和 Python）
    * 在 Linux 上本地安装
    * 使用 WSL2（Windows Subsystem for Linux）

    Docker 的优势是环境隔离和开箱即用，适合初学者。

??? question "问：需要什么版本的 Python？"
    TRICYS 要求 **Python 3.8 或更高版本**。推荐使用 Python 3.10 或 3.11 以获得最佳性能和兼容性。

??? question "问：OpenModelica 是必需的吗？"
    是的。TRICYS 使用 OpenModelica 作为建模和仿真后端。请确保：

    1. 已安装 OpenModelica
    2. `omc` 命令可在命令行中访问（已添加到 PATH）

    **验证方法**：
    ```bash
    omc --version
    ```

??? question "问：如何解决\"找不到 omc 命令\"的错误？"
    **Windows**：
    1. 确认 OpenModelica 安装路径（通常是 `C:\OpenModelica\bin`）
    2. 添加到系统环境变量 PATH
    3. 重启终端或 VSCode

    **Linux**：
    ```bash
    # 检查 omc 位置
    which omc

    # 如果未找到，添加到 PATH
    export PATH="/opt/openmodelica/bin:$PATH"

    # 或者永久添加到 ~/.bashrc
    echo 'export PATH="/opt/openmodelica/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```

??? question "问：Docker 镜像下载很慢怎么办？"
    可以使用国内的 Docker 镜像加速器：

    ```bash
    # 编辑 Docker 配置（Linux）
    sudo nano /etc/docker/daemon.json

    # 添加镜像加速地址
    {
      "registry-mirrors": [
        "https://docker.mirrors.ustc.edu.cn",
        "https://hub-mirror.c.163.com"
      ]
    }

    # 重启 Docker
    sudo systemctl restart docker
    ```

??? question "问：项目中的 `Makefile` 和 `Makefile.bat` 是做什么用的？"
    `Makefile` (适用于 Linux/macOS) 和 `Makefile.bat` (适用于 Windows) 提供了一系列快捷命令来简化常见的开发任务，例如安装、清理和测试。

    **常用命令**:

    *   **安装依赖**:
        ```bash
        # (Linux/macOS)
        make dev-install

        # (Windows)
        Makefile.bat dev-install
        ```
        此命令会安装项目的所有开发依赖。

    *   **代码格式化与检查**:
        ```bash
        # (Linux/macOS)
        make check

        # (Windows)
        Makefile.bat check
        ```
        此命令会自动格式化代码并运行静态检查。

    *   **运行测试**:
        ```bash
        # (Linux/macOS)
        make test

        # (Windows)
        Makefile.bat test
        ```

    *   **清理项目**:
        ```bash
        # (Linux/macOS)
        make clean

        # (Windows)
        Makefile.bat clean
        ```
        此命令会删除所有构建缓存和临时文件。

    使用这些命令可以帮助您保持开发环境的一致性。更多命令请查看文件内容或使用 `make help` / `Makefile.bat help`。

??? question "问：如何才能在本地查看和构建文档？"
    项目使用 [MkDocs](https://www.mkdocs.org/) 和 [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 主题来生成文档。我们已经通过 `Makefile` 和 `Makefile.bat` 提供了快捷命令。

    **步骤如下**:

    1.  **安装文档依赖**:
        ```bash
        # (Linux/macOS)
        make docs-install

        # (Windows)
        Makefile.bat docs-install
        ```

    2.  **启动本地预览服务器**:
        ```bash
        # (Linux/macOS)
        make docs-serve

        # (Windows)
        Makefile.bat docs-serve
        ```
        此命令会启动一个本地服务器（通常在 `http://127.0.0.1:18000`），并且当您修改文档源文件时，网页会自动刷新。

    3.  **构建静态网站** (可选):
        ```bash
        # (Linux/macOS)
        make docs-build

        # (Windows)
        Makefile.bat docs-build
        ```
        如果您想生成完整的静态 HTML 文件（例如用于部署），可以运行此命令。生成的文件位于 `site/` 目录下。


---