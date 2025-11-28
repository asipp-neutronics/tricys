---
hide:
  - navigation
---

TRICYS follows [Semantic Versioning](https://semver.org/):

- **Major**: Incremented for incompatible API changes
- **Minor**: Incremented for adding new functionality in a backward-compatible manner
- **Patch**: Incremented for backward-compatible bug fixes

---

## 1.0.0 ( 2025-11-15 )

TRICYS 1.0.0 is the first official stable release of the project, marking the completion of core functionalities and readiness for production environments.

* **Parameter Scanning and Concurrency**: Systematically study the impact of multiple parameters on system performance, supporting concurrent execution and large-scale batch simulations.
* **Sub-module Co-simulation**: Supports data exchange with external tools (e.g., Aspen Plus) to achieve sub-module system integration.
* **Automated Report Generation**: Automatically generates standardized Markdown analysis reports, including charts, statistical data, and visualization results.
* **Parameter Sensitivity Analysis**: Supports custom sensitivity analysis for system parameters and integrates the SALib (Sensitivity Analysis Library in Python) library to quantify the impact of parameters on the output.