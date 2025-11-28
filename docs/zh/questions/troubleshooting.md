??? question "问：仿真失败，如何调试？"
    **1. 查看日志文件**：
    ```bash
    # 查看最新运行的日志
    tail -f results/<timestamp>/simulation_*.log
    ```

    **2. 常见错误及解决方法**：

    | 错误信息 | 可能原因 | 解决方法 |
    |---------|---------|---------|
    | `Model not found` | 模型路径错误 | 检查 `package_path` 是否正确 |
    | `Failed to compile` | Modelica 语法错误 | 在 OMEdit 中打开模型检查错误 |
    | `Variable not found` | `variableFilter` 中的变量不存在 | 检查变量名拼写，使用正确的路径 |
    | `Out of memory` | 并发进程过多或模型太大 | 减少 `max_workers` 或增加系统内存 |
    | `Permission denied` | 文件权限问题 | 检查工作目录的读写权限 |

    **3. 启用详细日志**：
    ```json
    {
        "logging": {
            "level": "DEBUG"
        }
    }
    ```

??? question "问：GUI 无法启动或显示不正常？"
    **Windows/Linux 本地**：
    * 确保安装了 Tkinter：`pip install tk`
    * 检查显示环境变量：`echo $DISPLAY`

    **Docker 容器**：
    * Windows 11：确保 WSL2 的 WSLg 功能已启用
    * Linux：运行 `xhost +local:` 允许容器访问 X11
    * 使用包含 GUI 支持的镜像：`tricys_openmodelica_gui`

??? question "问：参数扫描结果不完整？"
    可能的原因：

    1. **某些仿真失败**：
       * 查看日志文件中的错误信息
       * 检查参数值是否合理（如避免除零、负值等）

    2. **输出变量过滤器太严格**：
       * 检查 `variableFilter` 是否匹配了所需变量

    3. **并发问题**：
       * 尝试禁用并发：`"concurrent": false`
       * 查看是否有进程崩溃

??? question "问：如何报告 Bug？"
    请在 [GitHub Issues](https://github.com/asipp-neutronics/tricys/issues) 中创建新 Issue，并提供：

    1. **环境信息**：
       * 操作系统和版本
       * Python 版本
       * OpenModelica 版本
       * TRICYS 版本

    2. **重现步骤**：
       * 完整的配置文件
       * 运行的命令
       * 使用的模型（如果可能）

    3. **错误信息**：
       * 完整的错误堆栈
       * 相关的日志片段

    4. **预期行为**：
       * 您期望发生什么
       * 实际发生了什么

---