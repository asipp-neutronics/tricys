---
hide:
  - navigation
  - toc
---

# TRICYS: 氚燃料循环集成仿真平台

欢迎使用 **TRICYS** (**TR**itium **I**ntegrated **CY**cle **S**imulation) 氚燃料循环集成仿真平台。TRICYS 是一个开源、模块化、多尺度的聚变堆氚燃料循环仿真器，旨在提供基于物理的动态闭环分析，并严格遵守全厂范围的质量守恒原则。

我们的目标是为研究人员和工程师提供一个灵活且强大的平台，用于探索各种氚管理策略、优化系统设计，并深入理解聚变反应堆环境中氚的流动与库存动态。

我们欢迎社区贡献！无论您是聚变科学家、软件工程师，还是对开源项目充满热情的爱好者，都有多种方式参与 TRICYS 的发展。

[快速开始](guides/quickstart.md){ .md-button .md-button--primary }
[报告问题](https://github.com/asipp-neutronics/tricys/issues){ .md-button  }
[氚燃料循环0维系统示例模型](guides/models/cycle.md){ .md-button }


---

<div class="grid cards" markdown>

<div class="card" markdown>
<h3 style="text-align: center;"> 📚 TRICYS 能做什么？</h3>
* **[参数扫描与并发](guides/tricys_basic/parameter_sweep.md)**：系统地研究多个参数对系统性能的影响，支持并发运行和大规模批量仿真。
* **[子模块协同仿真](guides/tricys_basic/co_simulation_module.md)**：支持与外部工具（如 Aspen Plus）进行数据交换完成子模块系统集成。
* **[自动化报告生成](explanation/tricys_analysis/analysis_report.md)**：自动生成标准化的 Markdown 分析报告，包含图表、统计数据和可视化结果。
* **[参数敏感性分析](guides/tricys_analysis/index.md)**：支持系统参数的自定义敏感性分析，并集成SALib（Sensitivity Analysis Library in Python）库量化参数对输出的影响。

</div>

<div class="card" markdown>
<h3 style="text-align: center;"> 🔬 为什么选择 TRICYS？</h3>
* **准确性与灵活性**：结合详细的物理模型和高度可配置的系统架构。
* **模块化设计**：易于集成到现有的工作流程和自动化系统中。
* **工业应用**：适用于聚变堆设计优化、运行策略评估和安全分析。
* **社区驱动**：受益于协作开发和透明的决策过程。
* **教育工具**：为学生和新研究人员理解聚变燃料循环动力学提供了极好的资源。
</div>

</div>