from __future__ import annotations

# Functional capability plugins
from tricys.postprocess.plugins.report_plugin import (
    AnalysisReportPlugin,
    BaselinePlotPlugin,
    LegacyTaskPlugin,
)
from tricys.postprocess.plugins.safety_plugin import (
    OfflineAuditorPlugin,
    SafetyAuditorPlugin,
    StaticAlarmPlugin,
)
from tricys.postprocess.plugins.sensitivity_plugin import (
    SensitivityAnalysisPlugin,
    execute_sensitivity_analysis,
)
from tricys.postprocess.plugins.storage_plugin import (
    CSVExportPlugin,
    HDF5StoragePlugin,
    MetricCalculatorPlugin,
    SimulationStoragePlugin,
    export_hdf5_to_csv,
    process_and_store_job_result,
)

__all__ = [
    # Primary functional plugins
    "SimulationStoragePlugin",
    "SafetyAuditorPlugin",
    "AnalysisReportPlugin",
    "SensitivityAnalysisPlugin",
    # Helper utilities
    "execute_sensitivity_analysis",
    "export_hdf5_to_csv",
    "process_and_store_job_result",
    # Legacy compatibility aliases
    "HDF5StoragePlugin",
    "CSVExportPlugin",
    "MetricCalculatorPlugin",
    "OfflineAuditorPlugin",
    "StaticAlarmPlugin",
    "BaselinePlotPlugin",
    "LegacyTaskPlugin",
]
