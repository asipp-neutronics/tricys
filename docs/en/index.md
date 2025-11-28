---
hide:
  - navigation
  - toc
---

# TRICYS: Tritium Integrated Cycle Simulation Platform

Welcome to **TRICYS** (**TR**itium **I**ntegrated **CY**cle **S**imulation), the Tritium Integrated Cycle Simulation platform. TRICYS is an open-source, modular, and multi-scale fusion reactor tritium fuel cycle simulator, designed to provide physics-based dynamic closed-loop analysis while strictly adhering to plant-wide mass balance principles.

Our goal is to provide researchers and engineers with a flexible and powerful platform for exploring various tritium management strategies, optimizing system design, and deeply understanding the dynamics of tritium flow and inventory within a fusion reactor environment.

We welcome community contributions! Whether you are a fusion scientist, a software engineer, or an enthusiast passionate about open-source projects, there are many ways to get involved in the development of TRICYS.

[Quick Start](guides/quickstart.md){ .md-button .md-button--primary }
[Report an Issue](https://github.com/asipp-neutronics/tricys/issues){ .md-button  }
[Example model of a 0-dimensional tritium fuel cycle system](guides/models/cycle.md){ .md-button }

---

<div class="grid cards" markdown>

<div class="card" markdown>
<h3 style="text-align: center;"> 📚 What can TRICYS do?</h3>
* **[Parameter Scanning & Concurrency](guides/tricys_basic/parameter_sweep.md)**: Systematically study the impact of multiple parameters on system performance, supporting [concurrent execution](guides/tricys_basic/concurrent_operation.md) and large-scale batch simulations.
* **[Sub-module Co-simulation](guides/tricys_basic/co_simulation_module.md)**: Supports data exchange with external tools (like Aspen Plus) to complete sub-module system integration.
* **[Automated Report Generation](explanation/tricys_analysis/analysis_report.md)**: Automatically generate standardized Markdown analysis reports, including charts, statistical data, and visualized results.
* **[Parameter Sensitivity Analysis](guides/tricys_analysis/index.md)**: Supports custom sensitivity analysis of system parameters and integrates the SALib (Sensitivity Analysis Library in Python) library to quantify the impact of parameters on outputs.

</div>

<div class="card" markdown>
<h3 style="text-align: center;"> 🔬 Why choose TRICYS?</h3>
* **Accuracy & Flexibility**: Combines detailed physical models with a highly configurable system architecture.
* **Modular Design**: Easy to integrate into existing workflows and automation systems.
* **Industrial Applications**: Suitable for fusion reactor design optimization, operational strategy evaluation, and safety analysis.
* **Community-Driven**: Benefits from collaborative development and transparent decision-making processes.
* **Educational Tool**: Provides an excellent resource for students and new researchers to understand fusion fuel cycle dynamics.
</div>

</div>