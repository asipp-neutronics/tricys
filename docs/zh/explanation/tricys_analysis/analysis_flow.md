`tricys` 的分析工作流 (`simulation_analysis.py`) 是一个强大的、以分析为导向的自动化流程。它在核心仿真功能之上，构建了复杂的多模式调度、目标寻优和报告生成能力。本篇文档将详细解析其内部的完整工作流程。

## 1. 核心流程图

下面是 `tricys` 分析工作流的完整流程图。它从读取配置开始，智能地选择三种主要操作模式之一，并执行相应的任务直到结束。

```mermaid
graph TD
    %% === 1. 启动阶段 ===
    subgraph Sub_A ["A. 启动与模式选择"]
        A1[开始: main config.json] --> A2["准备配置与日志<br>analysis_prepare_config()"]
        A2 --> A3{"<font size=4><b>运行模式判断</b></font><br>在 run_simulation() 内部"}
    end

    %% 模式判定逻辑
    A3 --> B_GROUP{analysis_cases 是否存在?}
    B_GROUP -- 是 --> C1
    B_GROUP -- 否 --> D_GROUP{是否为 SALib 分析?}
    
    D_GROUP -- 是 --> E1
    D_GROUP -- 否 --> F1

    %% === 2. 模式一 ===
    subgraph Sub_C ["C. 模式一: 多案例分析"]
        direction TB
        C1["<b>模式一: 多案例分析</b>"]
        C1 --> C2[为每个案例创建<br>独立工作目录和配置]
        C2 --> C3{并行执行案例?<br>concurrent_cases}
        
        C3 -- 是 --> C4[使用 <b>ProcessPoolExecutor</b><br>并行调用 _execute_analysis_case]
        C3 -- 否 --> C5[串行循环<br>逐个调用 _execute_analysis_case]
        
        %% 子图：单个案例执行
        subgraph Sub_C6 ["C6. 单个案例的执行 _execute_analysis_case"]
            direction TB
            C6_1[进入独立工作目录] --> C6_2["<b>递归调用 run_simulation<br>(禁用内部并发)</b>"] 
            C6_2 --> C6_3[执行模式三的完整流程]
        end

        %% 修复点：连接到子图内部的第一个节点 C6_1，而不是子图本身
        C4 --> C6_1
        C5 --> C6_1
        
        C6_3 --> C7[所有案例执行完毕]
        C7 --> C8["汇总所有案例的报告<br>consolidate_reports()"]
        C8 --> Z1[结束]
    end

    %% === 3. 模式二 ===
    subgraph Sub_E ["E. 模式二: SALib 分析"]
        direction TB
        E1["<b>模式二: SALib 分析</b>"] --> E2[调用 run_salib_analysis]
        
        %% 子图：SALib 流程
        subgraph Sub_E3 ["E3. SALib 内部流程"]
            direction TB
            E3_1[1. 使用 SALib 生成参数样本] --> E3_2[2. 为每个样本运行仿真] 
            E3_2 --> E3_3[3. 收集结果并计算敏感度指数]
        end
        
        %% 修复点：连接到子图内部节点 E3_1
        E2 --> E3_1
        E3_3 --> Z2[结束]
    end

    %% === 4. 模式三 ===
    subgraph Sub_F ["F. 模式三: 标准扫描与分析"]
        direction TB
        F1["<b>模式三: 标准扫描与分析</b>"] --> F2[生成仿真任务列表 jobs]
        F2 --> F3{并行执行任务?<br>concurrent}
        
        F3 -- 是 --> F4["使用 <b>ThreadPoolExecutor (标准)</b><br>或 <b>ProcessPoolExecutor (协同)</b><br>并行执行每个任务"]
        F3 -- 否 --> F5["串行执行<br>_run_sequential_sweep()"]
        
        %% 子图：单个任务执行
        subgraph Sub_F6 ["F6. 单个任务的执行 _run_..._job"]
            direction TB
            F6_1[1. 在隔离目录中运行仿真] --> F6_2{2. 是否配置了优化目标?<br>Required_...}
            
            F6_2 -- 是 --> F6_3["<b>优化子流程</b><br>调用 _run_bisection_search_for_job"]
            
            %% 嵌套子图：二分搜索
            subgraph Sub_F6_4 ["F6_4. 二分搜索循环"]
                F6_4_1[a. 在搜索范围内<br>迭代执行仿真] --> F6_4_2[b. 检查指标是否满足条件] 
                F6_4_2 --> F6_4_3[c. 缩小搜索范围]
                F6_4_3 --> F6_4_1
            end
            
            F6_3 --> F6_4_1
            F6_4_2 -- 满足或结束 --> F6_5
            
            F6_5["3. 返回仿真结果路径<br>和<b>优化结果</b>"]
            F6_2 -- 否 --> F6_5
        end

        %% 修复点：连接到子图内部节点 F6_1
        F4 --> F6_1
        F5 --> F6_1

        F6_5 --> F7[所有任务执行完毕]
        F7 --> F8["<b>结果聚合与后处理</b>"]
        
        %% 子图：后续步骤
        subgraph Sub_F9 ["F9. 后续步骤"]
            direction TB
            F9_1[a. 合并仿真结果到 sweep_results.csv] --> F9_2[b. 合并优化结果到 requierd_tbr_summary.csv]
            F9_2 --> F9_3["c. 执行敏感性分析<br>_run_sensitivity_analysis()<br>(提取指标、生成图表)"]
            F9_3 --> F9_4["d. 执行自定义后处理<br>_run_post_processing()"]
        end
        
        %% 修复点：连接到子图内部节点 F9_1
        F8 --> F9_1
        F9_4 --> Z3[结束]
    end

    %% 样式定义
    style C1 fill:#e3f2fd,stroke:#333,stroke-width:2px
    style E1 fill:#e8f5e9,stroke:#333,stroke-width:2px
    style F1 fill:#fbe9e7,stroke:#333,stroke-width:2px
```

## 2. 流程步骤详解

### 2.1. 启动与模式选择

整个流程始于 `main` 函数，它负责加载和预处理配置文件 (`analysis_prepare_config`) 并设置日志系统。核心逻辑位于 `run_simulation` 函数中，它首先会进行**模式判断**，以决定接下来执行哪个核心工作流。

---

### 2.2. 模式一：多案例分析 

当配置文件中定义了 `analysis_cases` 时，此模式被激活。它用于执行一系列独立的、可对比的分析研究。

1.  **环境设置**：框架会为 `analysis_cases` 中的每一个案例创建一个完全独立的子工作目录，并为其生成一份定制化的配置文件。这确保了每个案例的运行环境（包括模型修改、临时文件和结果）都是隔离的。
2.  **并发执行**：如果配置了 `"concurrent_cases": true`，`tricys` 会启动一个**进程池** (`ProcessPoolExecutor`) 来并行执行所有案例。使用进程是至关重要的，因为每个案例都是一个完整的 `tricys` 运行实例，需要独立的内存空间和文件系统权限以避免冲突。
3.  **递归调用**：每个案例的执行由 `_execute_analysis_case` 函数包裹，该函数会**递归地调用 `run_simulation`**，并强制禁用内部的并发执行（防止进程池嵌套）。这意味着每个案例内部会完整地执行一遍“模式三”的流程。
4.  **报告汇总**：所有案例执行完毕后，框架会调用 `consolidate_reports` 等函数，从所有子工作目录中收集分析结果和报告，并生成一份顶层的汇总报告，方便用户进行跨案例的比较。

---

### 2.3. 模式二：SALib 全局敏感性分析

如果配置指向一个全局敏感性分析（GSA）任务（通过 `independent_variable_sampling` 和 `analyzer` 等关键字识别），`tricys` 会将控制权移交给专门的 SALib 工作流。

1.  **调用 `run_salib_analysis`**：这是 SALib 流程的入口。
2.  **参数采样**：使用 SALib 库（如 `saltelli.sample`）根据配置的参数分布生成大量的参数样本集。
3.  **批量仿真**：为每一个参数样本运行一次 Modelica 仿真。
4.  **结果分析**：收集所有仿真结果，并使用 SALib 的分析器（如 `sobol.analyze`）计算每个参数的一阶、二阶和总阶敏感性指数（S1, S2, ST）。
5.  **输出报告**：将敏感性指数和相关图表输出为最终的分析报告。

---

### 2.4. 模式三：标准扫描与分析

这是最基础也是最核心的工作流，当以上两种模式的条件都不满足时，就会执行此流程。

1.  **生成任务**：根据 `simulation_parameters` 生成仿真任务列表 `jobs`。
2.  **执行仿真任务**：
    *   根据 `concurrent` 配置，选择并行或串行执行所有 `jobs`。
    *   每个任务的执行由 `_run_single_job` 或 `_run_co_simulation` 等函数处理。
3.  **优化子流程（核心特色）**：在单次仿真任务执行完毕后，系统会检查是否配置了优化目标（以 `Required_` 为前缀的指标）。
    *   如果存在，系统会立即调用 `_run_bisection_search_for_job`，启动一个**二分搜索循环**。
    *   该循环会为了寻找一个能满足预设指标（例如，氚增殖比TBR > 1.05）的最优参数值，而**迭代地运行多次仿真**，并根据每次的结果动态调整参数，直到找到解或达到最大迭代次数。
4.  **结果聚合**：
    *   所有任务（包括优化子流程中的所有仿真）完成后，框架会聚合两类数据：
        *   所有仿真的原始时间序列数据，合并到 `sweep_results.csv`。
        *   所有优化任务的最终结果（例如，`TBR>1.05` 对应的最优 `enrichment` 是 `0.85`），合并到 `requierd_tbr_summary.csv`。
5.  **最终分析与后处理**：
    *   调用 `_run_sensitivity_analysis`，它会加载聚合后的数据，计算最终的关键性能指标（KPI），并生成分析图表和摘要。
    *   调用 `_run_post_processing`，执行用户自定义的任何最终处理脚本（例如，生成特定的报告格式）。
