??? question "问：如何组织大型仿真项目？"
    推荐的目录结构：

    ```
    my_fusion_project/
    ├── models/                 # Modelica 模型
    │   ├── package.mo
    │   └── ...
    ├── configs/                # 配置文件
    │   ├── baseline.json
    │   ├── sensitivity.json
    │   └── ...
    ├── scripts/                # 辅助脚本
    │   ├── prepare_data.py
    │   └── post_analysis.py
    ├── data/                   # 输入数据
    │   └── external_data.csv
    ├── results/                # 仿真结果（自动生成）
    └── reports/                # 最终报告
    ```

??? question "问：如何版本管理配置文件？"
    使用 Git 进行版本控制：

    ```bash
    # 初始化 Git 仓库
    git init

    # 创建 .gitignore
    echo "results/" >> .gitignore
    echo "*.log" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo "__pycache__/" >> .gitignore

    # 提交配置文件
    git add configs/*.json
    git commit -m "feat: add baseline configuration"
    ```

??? question "问：如何在多台机器上分布式运行？"
    目前 TRICYS 不直接支持分布式计算，但您可以：

    1. **手动分割任务**：
       * 将参数扫描分成多个子集
       * 在不同机器上运行不同的子集
       * 手动合并结果

    2. **使用集群调度器**（如 SLURM）：
       * 将每个参数组合作为独立的作业提交
       * 使用后处理脚本汇总结果

??? question "问：如何优化模型性能？"
    **Modelica 模型层面**：
    1. 简化模型结构，避免过度复杂的方程
    2. 使用适当的数值求解器设置
    3. 避免代数环和事件过多

    **TRICYS 层面**：
    1. [启用并发运行](../guides/tricys_basic/concurrent_operation.md)
    2. 减少输出变量
    3. [使用协同仿真替代复杂子系统](../guides/tricys_basic/co_simulation_module.md)