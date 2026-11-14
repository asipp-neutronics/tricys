---
hide:
  - navigation
---

TRICYS follows [Semantic Versioning](https://semver.org/):

- **Major**: Incremented for incompatible API changes
- **Minor**: Incremented for adding new functionality in a backward-compatible manner
- **Patch**: Incremented for backward-compatible bug fixes

---

## 2.0.1 ( Unreleased )

### Changed (behaviour)

* **Any failed job now yields a non-zero exit code**: previously a failed job in a sweep or a co-simulation only wrote to the log, the process still exited `0`, and the results file merely had fewer rows — indistinguishable from a successful run. `StandardSimulationEngine` (both sequential and concurrent paths) and `OfflineTableEngine` now raise `JobExecutionError` (carrying the failed job IDs and reasons) when any job fails, and the CLI exits `1`. **Scripts that relied on partial failures returning 0 must be adapted.**

---

## 2.0.0 ( 2026-11-14 )

TRICYS 2.0.0 is an architectural rebuild: from procedure-oriented script assembly to a pluggable pipeline with a model-centric engineering base.

### Added

* **Pluggable workflow engine**: a three-stage pipeline (preprocess → simulation engine → postprocess) where stages exchange data only through a strongly typed `PipelineContext`; plugins can be registered and extended.
* **Schema 2.0 with two configuration tracks**: a research DSL (`simulate` / `sweep` / `sensitivity` / `cosim` / `optimize`) alongside explicit pipelines; 1.0 JSON configurations are still accepted.
* **Time-base governance**: new `time_base` (`second` / `hour`). Configuration time fields are always hours and the framework converts them to the model's base unit, eradicating the 3600× dimensional mismatch of Issue #99.
* **Three co-simulation paradigms**: offline table (`offline_table`), online step (`online_step`), and steady-state parameter injection (`param_inject`); DWSIM (CAPE-OPEN) and FESTIM 2.0 divertor FEM online-step integration.
* **Nested bisection optimisation engine**: outer search for the minimum start-up tritium inventory and inner approach to the critical TBR, constrained simultaneously by the in-service reserve trough and the target doubling time.
* **Model introspection and review**: `tricys model inspect` / `tricys model review` audit time scale, conservation streams and initial values, and can emit a remediation patch; `tricys run --diagnose` provides structured failure interception.
* **Visualisation and reporting**: HDF5 result dashboard; reports export to Markdown / DOCX / PDF.
* **Cloud-native container base**: a single All-in-One image bundling OpenModelica, FESTIM, CasADi/H2ISO, the FastAPI backend, the Vue3 console, the GoView wall display, the Dash dashboard and an Nginx gateway.
* **Examples**: 12 v2 industrial benchmark cases (01–12); 1.0 examples are archived under `v1_examples_legacy/`.

### Changed (breaking)

* New `time_base`, defaulting to `second`. **Models authored in hours during the 1.0 era must set `time_base: hour` explicitly**, otherwise results are off by 3600×. See "Upgrading from TRICYS 1.0 to 2.0".
* `tricys.handlers` archived to `tricys.example.example_data.v1_examples_legacy.handlers`, with only a `sys.modules` compatibility shim retained.
* Prebuilt FMU / SSP assets removed (`example_model_ssp/` is no longer shipped).
* `tricys.handlers` dropped from the package list.

### Fixed

* **Inconsistent time-base conversion between the sequential and concurrent paths**: 1.0-style configurations (time fields under `simulation`) bypassed the hour → second conversion in sequential mode, so the same configuration reached OpenModelica with stopTime values 3600× apart while both reported success. Both paths now share a single conversion entry point. **Results produced in sequential mode from 1.0-style configurations before this fix are invalid and must be re-run.**

### Known issues

* Job failures do not reach the exit code: partial failures in a sweep or co-simulation still return `0`. Scheduled for 2.0.1.
* The online mass auditor takes its initial reading as the baseline without sanity checks; abnormal readings (0 / NaN / negative) are accepted. Scheduled for 2.0.1.

---

## 1.0.0 ( 2025-11-15 )

TRICYS 1.0.0 is the first official stable release of the project, marking the completion of core functionalities and readiness for production environments.

* **Parameter Scanning and Concurrency**: Systematically study the impact of multiple parameters on system performance, supporting concurrent execution and large-scale batch simulations.
* **Sub-module Co-simulation**: Supports data exchange with external tools (e.g., Aspen Plus) to achieve sub-module system integration.
* **Automated Report Generation**: Automatically generates standardized Markdown analysis reports, including charts, statistical data, and visualization results.
* **Parameter Sensitivity Analysis**: Supports custom sensitivity analysis for system parameters and integrates the SALib (Sensitivity Analysis Library in Python) library to quantify the impact of parameters on the output.