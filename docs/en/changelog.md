---
hide:
  - navigation
---

TRICYS follows [Semantic Versioning](https://semver.org/):

- **Major**: Incremented for incompatible API changes
- **Minor**: Incremented for adding new functionality in a backward-compatible manner
- **Patch**: Incremented for backward-compatible bug fixes

---

## 2.0.2 ( 2026-11-18 )

### Fixed

* **Example 9 (`09_cosim_online_step`) could not run on 2.0.1**: the example enabled `auditor` without configuring `patterns`, so the initialization check introduced in 2.0.1 aborted the run. `patterns.inventory: ["*.I?1?"]` was added (`[` is a character class in fnmatch, hence `I[1]` is written `I?1?`), matching 14 inventory variables. A full 100 h run now completes locally: initial inventory 3000 g, final 2999.69 g, **mass balance error −0.31 g** (warn 50 g / kill 1000 g).

### Changed (behaviour)

* **"Auditor enabled without `patterns`" is now a warning instead of an error**: that is an unconfigured auditor, not a corrupt one, so it no longer aborts the run (restores 2.0.0 behaviour). **Patterns configured but matching no variable** still raises — that is the dangerous "asked for an audit, silently got none" case.
* Fixed the false "0 of 0 variables unreadable" edge case when no inventory bindings exist.

---

## 2.0.1 ( 2026-11-16 )

### Changed (behaviour)

* **A 1.0-shaped configuration now warns about its time base**: when neither `time_base` nor a `scenario` block is present, the engine logs a WARNING naming the 3600x risk and the fix. The default stays `second` (SI); behaviour is unchanged.

* **Any failed job now yields a non-zero exit code**: previously a failed job in a sweep or a co-simulation only wrote to the log, the process still exited `0`, and the results file merely had fewer rows — indistinguishable from a successful run. `StandardSimulationEngine` (both sequential and concurrent paths) and `OfflineTableEngine` now raise `JobExecutionError` (carrying the failed job IDs and reasons) when any job fails, and the CLI exits `1`. **Scripts that relied on partial failures returning 0 must be adapted.**

* **Fixed: multi-file Modelica packages (`package.mo`) could not run.** OMPython 4 copies the `.mo` into its own work directory before loading it, which separates `package.mo` from its sibling `.mo` files and makes the package name disagree with the temporary directory name (6 legacy configurations failed because of it). The whole package directory is now staged under a directory named after the package before loading.

* **Fixed: the online mass auditor did not validate its initial baseline.** A NaN / infinite / negative initial inventory, a completely unreadable inventory set, or inventory patterns that matched no variable now raise `AuditorInitializationError` instead of being accepted; an exactly-zero initial total raises a warning. Previously such values silently became the baseline and every later mass balance was computed against a wrong origin.

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