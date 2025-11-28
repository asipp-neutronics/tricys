## 1. 整体架构

TRICYS 采用分层架构设计，主要包含以下几层：

```
┌─────────────────────────────────────────────┐
│               用户界面层             
│ tricys basic, tricys analysis, tricys gui       
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│               仿真执行层                     
│     simulation, simulation_analysis    
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│               核心功能层                     
│      Jobs, Modelica, Interceptor            
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│             分析与后处理层                   
│       Metric, Plot, Report, SALib           
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│               工具函数层                     
│      Config, File, Log, SQLite Utils        
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│                外部依赖层                    
│     OpenModelica, Pandas, NumPy, SALib      
└─────────────────────────────────────────────┘
```

## 2. 程序分层

### 2.1. 用户界面层

**位置**：`tricys/main.py`

**职责**：

- 提供[命令行接口（CLI）和图形用户界面（GUI）](../guides/quickstart.md#5-tricys-相关命令)两种交互方式
- 解析用户输入的命令行参数和子命令（`basic`, `analysis`, `gui`, `example`, `archive`, `unarchive`）
- 路由请求到相应的仿真执行层模块
- 管理用户会话和配置文件加载

**关键功能**：

- CLI 命令分发：根据子命令或配置文件内容自动识别运行模式
- GUI 交互界面：提供可视化的参数设置、仿真启动和结果查看功能
- 示例运行器：集成交互式示例选择和运行功能

---

### 2.2. 仿真执行层

**位置**：`tricys/simulation/`

**职责**：

- **基础仿真模式** (`simulation.py`)：执行单次或参数扫描仿真任务
- **灵敏度分析模式** (`simulation_analysis.py`)：执行多种敏感性分析工作流
- 管理仿真任务的完整生命周期（初始化、执行、后处理）
- 协调核心功能层、分析层和后处理层的调用

详见：[仿真执行流程](tricys_basic/simulation_flow.md) 和 [自动分析流程](tricys_analysis/analysis_flow.md)

---

### 2.3. 核心功能层

**位置**：`tricys/core/`

**职责**：

- **Modelica 交互** (`modelica.py`)：通过 OMPython 与 OpenModelica 引擎通信
- **任务生成** (`jobs.py`)：根据配置生成参数扫描任务和仿真作业
- **拦截器机制** (`interceptor.py`)：生成和集成拦截器模型，实现[协同仿真](tricys_basic/co_simulation.md)

详见：[API 参考 - 核心模块 (Core)](../api/tricys_core.md)

---

### 2.4. 分析与后处理层

**位置**：`tricys/analysis/`, `tricys/postprocess/`

**职责**：

- **性能指标计算** (`analysis/metric.py`)：计算启动盘存、倍增时间、转折点等[关键指标](tricys_analysis/performance_metrics.md)
- **数据可视化** (`analysis/plot.py`)：生成时间序列图、参数扫描图、对比图等
- **灵敏度分析** (`analysis/salib.py`)：集成 SALib 库执行多种[敏感性分析方法](tricys_analysis/salib_integration.md)
- **分析报告生成** (`analysis/report.py`)：自动生成 Markdown 格式的[分析报告](tricys_analysis/analysis_report.md)，支持 AI 增强
- **后处理模块** (`postprocess/`)：提供可扩展的[数据后处理功能](../guides/tricys_basic/post_processing_module.md)


详见：[API 参考 - 分析模块 (Analysis)](../api/tricys_analysis.md)

---

### 2.5. 工具函数层

**位置**：`tricys/utils/`

**职责**：

- **配置管理** (`config_utils.py`)：配置文件加载、验证和预处理
- **文件操作** (`file_utils.py`)：文件路径处理、唯一文件名生成、归档管理
- **日志系统** (`log_utils.py`)：结构化日志记录和配置恢复
- **数据库操作** (`sqlite_utils.py`)：SQLite 数据存储和查询


详见：[API 参考 - 工具函数 (Utilities)](../api/tricys_utils.md)

---

### 2.6. 外部依赖层

**主要依赖**：

- **OpenModelica**：Modelica 模型编译和仿真执行引擎
- **OMPython**：Python 与 OpenModelica 的接口库
- **SALib**：[敏感性分析和不确定性量化库](tricys_analysis/salib_integration.md)
- **Pandas/NumPy**：数据处理和数值计算
- **Matplotlib/Seaborn**：数据可视化
- **OpenAI**（可选）：[AI 增强的分析报告生成](tricys_analysis/analysis_report.md)

**职责**：

- 提供底层的仿真引擎、数值计算和科学计算支持
- 确保跨平台兼容性和高性能计算能力

---

## 3. 设计原则

1. **模块化**：每个功能模块职责单一，相互独立
2. **可扩展**：易于添加新的[后处理模块](../guides/tricys_basic/post_processing_module.md)、[性能指标](tricys_analysis/performance_metrics.md)和[协同仿真处理器](../guides/tricys_basic/co_simulation_module.md#3-编写您自己的处理器)
3. **配置驱动**：所有仿真任务通过 [JSON 配置文件](../guides/tricys_basic/basic_configuration.md)定义
4. **自动化**：从仿真到[分析报告生成](tricys_analysis/analysis_report.md)的全流程自动化
5. **开放性**：开源设计，支持社区贡献