from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class BaseDTO(BaseModel):
    """Base DTO allowing arbitrary fields for forward/backward schema compatibility."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )


# ==============================================================================
# TRICYS 2.0 Core Pillars: model, scenario, task, visualizer, plugins
# ==============================================================================


class LibraryDep(BaseDTO):
    """External or standard Modelica library dependency (e.g. Modelica 4.0.0)."""

    name: str = Field(description="Library name, e.g. Modelica")
    version: Optional[str] = Field(
        default=None, description="Library version, e.g. 4.0.0"
    )

    @model_validator(mode="before")
    @classmethod
    def from_string(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"name": v}
        return v


class ModelConfig(BaseDTO):
    """Level 1 Pillar: Physical model specification and compiler options."""

    fileName: str = Field(
        description="Path to Modelica model source file (.mo or package.mo)"
    )
    modelName: str = Field(
        description="Full Modelica class name, e.g. example_model.Cycle"
    )
    loadModel: List[LibraryDep] = Field(
        default_factory=list, description="Modelica library dependencies to load"
    )
    cflags: str = Field(
        default="", description="C compiler optimization flags, e.g. -O3"
    )
    target: str = Field(
        default="gcc", description="Target compiler toolchain (gcc, msvc)"
    )


class PulseConfig(BaseDTO):
    """Fusion operation cycle / dynamic pulse injection settings."""

    foc_file: Optional[str] = Field(
        default=None, description="Path to .foc schedule DSL file"
    )
    target_component: Optional[str] = Field(
        default=None, description="Target Modelica instance name to replace"
    )
    foc_component: Optional[str] = Field(
        default=None, description="Alternative alias for target_component"
    )
    variable_mapping: Dict[str, str] = Field(
        default_factory=dict, description="Mapping from foc signal to modelica variable"
    )
    power_scale: float = Field(
        default=1.0, description="Pulse power scaling multiplier"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Dynamic pulse source parameters"
    )


class ScenarioConfig(BaseDTO):
    """Level 1 Pillar: Simulation condition and numerical solver settings."""

    stopTime: float = Field(
        description="Simulation stop time in SI seconds (float > 0)"
    )
    startTime: float = Field(
        default=0.0, description="Simulation start time in SI seconds"
    )
    stepSize: float = Field(
        default=1.0, description="Communication step size in SI seconds"
    )
    numberOfIntervals: Optional[int] = Field(
        default=None,
        description="Total communication intervals (alternative to stepSize)",
    )
    tolerance: float = Field(default=1e-6, description="Solver relative tolerance")
    method: str = Field(
        default="dassl",
        description="OpenModelica numerical solver (dassl, ida, cvode, rungekutta)",
    )
    outputFormat: str = Field(
        default="csv", description="Output file format (csv, mat)"
    )
    variableFilter: str = Field(
        default=".*", description="Regex filter for logged output timeseries"
    )
    simflags: str = Field(
        default="", description="Runtime flags passed to Modelica simulation executable"
    )
    pulse: Optional[PulseConfig] = Field(
        default=None, description="Dynamic pulse scenario injection"
    )

    @field_validator("stopTime")
    @classmethod
    def validate_stop_time(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"stopTime must be a positive float in seconds, got {v}")
        return float(v)


class TargetMetricConfig(BaseDTO):
    """Evaluation criteria for bisection search target metric."""

    source: Optional[str] = Field(
        default=None, description="Variable name to observe, e.g. sds.I[1]"
    )
    evaluate: str = Field(
        default="turning_point",
        description="Operator to evaluate: doubling_time, turning_point, peak_value, min, etc.",
    )
    threshold: Optional[float] = Field(
        default=None, description="Target numerical threshold for turning point / limit"
    )
    direction: Optional[str] = Field(
        default="min_to_max",
        description="Direction of slope reversal: min_to_max or max_to_min",
    )
    operator: Optional[str] = Field(default=None, description="Alias for evaluate")
    target_value: Optional[float] = Field(
        default=None, description="Alias for threshold"
    )
    max_value: Optional[float] = Field(default=None, description="Alias for threshold")

    @model_validator(mode="before")
    @classmethod
    def normalize_target_metric(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "operator" in data and "evaluate" not in data:
                data["evaluate"] = data["operator"]
            if "target_value" in data and "threshold" not in data:
                data["threshold"] = data["target_value"]
            if "max_value" in data and "threshold" not in data:
                data["threshold"] = data["max_value"]
        return data


class MetricConfig(BaseDTO):
    """Feature extraction operator and bisection optimization configuration."""

    source: Optional[str] = Field(
        default=None, description="Modelica variable name to extract from timeseries"
    )
    evaluate: str = Field(
        default="peak_value",
        description="Feature operator: peak_value, min, final_value, turning_point, doubling_time, integral, mean, settling_time, bisection_search",
    )
    unit: Optional[str] = Field(
        default=None, description="Physical unit label for plotting"
    )
    threshold: Optional[float] = Field(
        default=None, description="Upper/lower safety threshold"
    )

    # Bisection search specific fields
    parameter: Optional[str] = Field(
        default=None, description="Alias for parameter_to_optimize (e.g. blanket.TBR)"
    )
    parameter_to_optimize: Optional[str] = Field(
        default=None,
        description="Parameter name to tune during bisection (e.g. blanket.TBR)",
    )
    search_bounds: Optional[List[float]] = Field(
        default=None, description="Search interval [low, high] for bisection"
    )
    search_range: Optional[List[float]] = Field(
        default=None, description="Alias for search_bounds [low, high]"
    )
    tolerance: Optional[float] = Field(
        default=None, description="Convergence tolerance for bisection search"
    )
    max_iterations: Optional[int] = Field(
        default=15, description="Maximum iterations for bisection search"
    )
    target_metric: Optional[TargetMetricConfig] = Field(
        default=None, description="Inner target metric criteria for bisection search"
    )
    metric_name: Optional[str] = Field(
        default=None,
        description="Flat alias for target metric name (Doubling_Time, etc.)",
    )
    metric_max_value: Optional[float] = Field(
        default=None,
        description="Flat alias for target metric threshold (e.g. 2 years in seconds)",
    )
    source_column: Optional[str] = Field(
        default=None,
        description="Flat alias for target variable column (e.g. sds.I[1])",
    )

    # General operator parameters
    direction: Optional[str] = Field(
        default=None, description="Direction parameter for turning_point"
    )
    factor: Optional[float] = Field(
        default=None, description="Factor for doubling_time (default 2.0)"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_metric_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "parameter" in data and "parameter_to_optimize" not in data:
                data["parameter_to_optimize"] = data["parameter"]
            if "parameter_to_optimize" in data and "parameter" not in data:
                data["parameter"] = data["parameter_to_optimize"]
            if "search_bounds" in data and "search_range" not in data:
                data["search_range"] = data["search_bounds"]
            if "search_range" in data and "search_bounds" not in data:
                data["search_bounds"] = data["search_range"]
        return data


class SimulateTaskConfig(BaseDTO):
    """Level 2 task configuration for single-run simulation / single-point bisection."""

    type: Literal["simulate"] = "simulate"
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Single-point initial value / parameter override",
    )
    metrics: Dict[str, MetricConfig] = Field(
        default_factory=dict, description="Feature extraction or bisection metrics"
    )


class SweepTaskConfig(BaseDTO):
    """Level 2 task configuration for multi-case / grid parameter sweeps."""

    type: Literal["sweep"] = "sweep"
    parallel: bool = Field(
        default=False, description="Enable parallel multi-worker execution"
    )
    max_workers: Optional[int] = Field(
        default=None, description="Max parallel worker processes (default: auto)"
    )
    parameters: Union[List[Dict[str, Any]], Dict[str, List[Any]], Dict[str, Any]] = (
        Field(
            description="Sweep space: List of dicts (case list), Dict of lists (grid), or external file ref"
        )
    )
    metrics: Dict[str, MetricConfig] = Field(
        default_factory=dict, description="Summary metrics to compute across all cases"
    )

    @field_validator("parameters")
    @classmethod
    def validate_sweep_parameters(cls, v: Any) -> Any:
        if isinstance(v, list):
            if not v:
                raise ValueError(
                    "parameters list must contain at least one operating case dictionary"
                )
            for idx, item in enumerate(v):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"parameters[{idx}] must be a dictionary representing an operating case, got {type(item).__name__}"
                    )
        elif isinstance(v, dict):
            if "file" not in v:
                # Grid or scalar map
                pass
        else:
            raise ValueError(
                f"Sweep parameters must be a list of dicts, a dict of lists, or a dict with 'file', got {type(v).__name__}"
            )
        return v


class SensitivityTaskConfig(BaseDTO):
    """Level 2 task configuration for global sensitivity and uncertainty analysis."""

    type: Literal["sensitivity"] = "sensitivity"
    method: Literal["sobol", "latin", "morris"] = Field(
        default="sobol", description="Sampling algorithm: sobol, latin, or morris"
    )
    samples: int = Field(default=128, description="Base sample count N")
    seed: Optional[int] = Field(
        default=None, description="Random number generator seed"
    )
    parallel: bool = Field(default=False, description="Enable parallel batch execution")
    max_workers: Optional[int] = Field(
        default=None, description="Max parallel worker processes (default: auto)"
    )
    parameters: Dict[str, Any] = Field(
        description="Uncertainty parameter bounds [min, max] or distribution specs"
    )
    metrics: Dict[str, MetricConfig] = Field(
        description="Sensitivity target responses Y (supports bisection_search)"
    )


class SubsystemSpec(BaseDTO):
    """External subsystem replacement specification in co-simulation."""

    target: str = Field(
        description="Target Modelica class to replace, e.g. CFEDR.I_ISS"
    )
    instance: Optional[str] = Field(
        default=None, description="Instance name in parent model"
    )
    solver: str = Field(
        description="Subsystem solver adapter: aspen_plus, h2iso, fmu, python_script"
    )
    model_file: Optional[str] = Field(
        default=None, description="External subsystem model file"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Subsystem runtime parameters"
    )


class CosimTaskConfig(BaseDTO):
    """Level 2 task configuration for multi-physics co-simulation."""

    type: Literal["cosim"] = "cosim"
    mode: str = Field(
        default="subsystem_replacement",
        description="Coupling mode: subsystem_replacement or fmi_online",
    )
    exchange_step: float = Field(
        default=1.0, description="Macro exchange time step in SI seconds"
    )
    subsystems: List[SubsystemSpec] = Field(
        default_factory=list, description="Subsystems participating in co-simulation"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Co-simulation parameter overrides"
    )
    metrics: Dict[str, MetricConfig] = Field(
        default_factory=dict, description="Co-simulation extraction metrics"
    )


# TaskTypeUnion enforces that task.type must be one of the 4 valid computation types.
# Invalid types like parse, archive, visualize are strictly rejected!
TaskTypeUnion = Annotated[
    Union[SimulateTaskConfig, SweepTaskConfig, SensitivityTaskConfig, CosimTaskConfig],
    Field(discriminator="type"),
]


class AIInsightConfig(BaseDTO):
    """LLM engineering insight generation configuration."""

    enabled: bool = Field(
        default=False, description="Enable AI automated analysis insights"
    )
    focus_areas: List[str] = Field(
        default_factory=list, description="Specific analysis focus topics for LLM"
    )
    provider: Optional[str] = Field(
        default=None, description="AI provider / model configuration"
    )


class ReportConfig(BaseDTO):
    """Level 1 Pillar / Root: Automated report delivery engine."""

    enabled: bool = Field(default=True, description="Enable report generation")
    template: str = Field(
        default="executive_summary",
        description="Built-in report template name (e.g. executive_summary, comprehensive_dossier, kpi_metrics)",
    )
    title: str = Field(default="Tricys Simulation Report", description="Report title")
    formats: List[str] = Field(
        default_factory=lambda: ["markdown"],
        description="Export formats: markdown, docx, pdf, html",
    )
    output_dir: str = Field(default="reports", description="Report storage directory")
    sections: List[str] = Field(
        default_factory=lambda: [
            "executive_summary",
            "kpi_metrics_table",
            "sensitivity_ranking",
            "safety_audit_summary",
            "custom_plots",
        ],
        description="Report sections to render",
    )
    figures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Specific figures and variable trajectories to plot in report",
    )
    ai_insight: Optional[AIInsightConfig] = Field(
        default=None, description="AI insight generation settings"
    )


class VisualizerConfig(BaseDTO):
    """Level 1 Pillar: Interactive Web dashboard and report delivery hub."""

    enabled: bool = Field(default=True, description="Enable visualizer module")
    port: int = Field(default=8080, description="Web dashboard listening port")
    host: str = Field(default="127.0.0.1", description="Web dashboard host address")
    launch_on_finish: bool = Field(
        default=False,
        description="Automatically open browser upon simulation completion",
    )
    report: Optional[ReportConfig] = Field(
        default=None, description="Internalized report delivery engine"
    )


class PluginSpec(BaseDTO):
    """Lifecycle plugin specification."""

    name: str = Field(description="Unique plugin identifier name")
    plugin: Optional[str] = Field(
        default=None,
        description="Built-in plugin name (e.g. mass_conservation, auto_compress)",
    )
    script: Optional[str] = Field(
        default=None, description="Path to external custom Python plugin script"
    )
    class_name: Optional[str] = Field(
        default=None,
        alias="class",
        description="Python class name inside plugin script",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Plugin parameters"
    )


class EngineSpec(BaseDTO):
    """Custom simulation execution engine specification."""

    type: str = Field(
        default="standard", description="Engine type: standard, slurm_cluster"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Engine parameters"
    )


class PluginsConfig(BaseDTO):
    """Level 1 Pillar: Lifecycle extension pipeline hooks."""

    preprocess: List[PluginSpec] = Field(
        default_factory=list, description="Plugins executed before simulation"
    )
    simulation_engine: Optional[EngineSpec] = Field(
        default=None, description="Custom simulation engine override"
    )
    postprocess: List[PluginSpec] = Field(
        default_factory=list, description="Plugins executed after simulation"
    )


class TricysV2Config(BaseDTO):
    """Root strongly-typed configuration DTO for TRICYS 2.0.

    Structured strictly around the 5 Core Pillars:
    1. model: Physical model & compiler settings
    2. scenario: Simulation conditions & solver settings
    3. task: Scientific calculation task (simulate, sweep, sensitivity, cosim)
    4. visualizer: Interactive Web dashboard & report generation
    5. plugins: Lifecycle extension hooks (auditor, archive, etc.)
    """

    version: str = Field(default="2.0", description="TRICYS specification version")
    model: ModelConfig = Field(
        description="Level 1 Pillar: Physical model specification"
    )
    scenario: ScenarioConfig = Field(
        description="Level 1 Pillar: Simulation condition and solver"
    )
    task: TaskTypeUnion = Field(
        description="Level 1 Pillar: Calculation task and parameters/metrics"
    )
    visualizer: Optional[VisualizerConfig] = Field(
        default=None, description="Level 1 Pillar: Dashboard and report delivery"
    )
    report: Optional[ReportConfig] = Field(
        default=None,
        description="Level 1 Pillar / Extension: Automated engineering report delivery",
    )
    plugins: Optional[PluginsConfig] = Field(
        default=None, description="Level 1 Pillar: Lifecycle plugins"
    )

    # Backwards-compatibility synthetic accessors for legacy plugins
    @property
    def paths(self) -> PathsConfig:
        return PathsConfig(
            package_path=self.model.fileName,
            results_dir="results",
            temp_dir="temp",
        )

    @property
    def simulation(self) -> SimulationConfig:
        return SimulationConfig(
            model_name=self.model.modelName,
            stop_time=self.scenario.stopTime,
            step_size=self.scenario.stepSize,
            variableFilter=self.scenario.variableFilter,
            execute_mode="standard",
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TricysV2Config:
        """Constructs and validates TricysV2Config from a dictionary."""
        return cls.model_validate(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to a plain dictionary."""
        return self.model_dump(exclude_none=True, by_alias=True)


# ==============================================================================
# TRICYS 1.0 Legacy Models (for backward compatibility)
# ==============================================================================


class PathsConfig(BaseDTO):
    """Legacy 1.0 file paths configuration."""

    package_path: str = Field(
        default="", description="Path to Modelica package.mo or single model"
    )
    results_dir: str = Field(
        default="results", description="Directory to store outputs and HDF5"
    )
    temp_dir: str = Field(
        default="temp", description="Temporary working directory for isolation"
    )


class SimulationConfig(BaseDTO):
    """Legacy 1.0 simulation execution settings."""

    model_name: str = Field(default="", description="Modelica model class name")
    stop_time: float = Field(default=0.0, description="Simulation stop time in seconds")
    step_size: float = Field(default=0.0, description="Solver communication step size")
    variableFilter: Optional[str] = Field(
        default=None, description="Regex filter for saved variables"
    )
    concurrent: bool = Field(
        default=False, description="Enable parallel batch execution"
    )
    max_workers: Optional[int] = Field(
        default=None, description="Max concurrent worker processes"
    )
    keep_temp_files: bool = Field(
        default=True, description="Whether to preserve temporary job folders"
    )
    execute_mode: str = Field(
        default="standard", description="Execution mode: standard, fast, enhanced"
    )
    maximize_workers: bool = Field(
        default=False, description="Whether to maximize worker allocation"
    )


class FOCConfig(BaseDTO):
    """Legacy 1.0 FOC schedule configuration."""

    foc_path: str = Field(default="", description="Path to .foc DSL schedule file")
    foc_component: Optional[str] = Field(
        default=None, description="Target component name to replace"
    )
    strategy: str = Field(
        default="table", description="Injection strategy: table, block, parameter"
    )


class SafetyAuditorConfig(BaseDTO):
    """Legacy 1.0 Safety Auditor settings."""

    enabled: bool = Field(
        default=False, description="Whether mass balance audit is enabled"
    )
    tolerance: float = Field(
        default=1e-4, description="Relative tolerance for mass conservation"
    )
    subsystems: List[str] = Field(
        default_factory=list, description="Subsystems to include in audit"
    )


class StaticAlarmRule(BaseDTO):
    """Legacy 1.0 alarm rule definition."""

    variable: str = Field(description="Name of the variable to monitor")
    threshold: float = Field(description="Numeric limit threshold")
    condition: str = Field(default="gt", description="Condition: gt, lt, gte, lte, eq")
    level: str = Field(
        default="warning", description="Alarm severity: info, warning, critical"
    )


class StaticAlarmConfig(BaseDTO):
    """Legacy 1.0 alarm configuration."""

    rules: List[Dict[str, Any]] = Field(
        default_factory=list, description="Threshold monitoring rules"
    )


class SensitivityAnalysisConfig(BaseDTO):
    """Legacy 1.0 sensitivity analysis configuration."""

    enabled: bool = Field(default=False, description="Enable sensitivity analysis")
    analysis_cases: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of analysis cases"
    )
    analysis_case: Optional[Dict[str, Any]] = Field(
        default=None, description="Single analysis case"
    )
    metrics_definition: Dict[str, Any] = Field(
        default_factory=dict, description="Metric calculation formulas"
    )
    unit_map: Dict[str, str] = Field(
        default_factory=dict, description="Variable unit labels for plotting"
    )
    glossary_path: str = Field(default="", description="Path to terminology glossary")


class PreprocessConfig(BaseDTO):
    """Legacy 1.0 preprocessing directives."""

    sampling: Optional[Dict[str, Any]] = Field(
        default=None, description="General sampling specifications"
    )
    isolation: bool = Field(
        default=True, description="Isolate workspace before running"
    )


def is_v2_config(data: Dict[str, Any]) -> bool:
    """Helper function to detect whether a raw dictionary conforms to TRICYS 2.0 schema."""
    if not isinstance(data, dict):
        return False
    if "version" in data and str(data["version"]).startswith("2"):
        return True
    return "model" in data and "scenario" in data


class TricysConfig(BaseDTO):
    """Legacy 1.0 root strongly-typed configuration DTO.

    Preserved for backwards compatibility with TRICYS 1.0 pipelines and plugins.
    Automatically delegates to TricysV2Config if 2.0 structure is detected.
    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    foc: Optional[FOCConfig] = None
    co_simulation: Optional[Dict[str, Any]] = None
    auditor: Optional[SafetyAuditorConfig] = None
    static_alarm: Optional[StaticAlarmConfig] = None
    sensitivity_analysis: Optional[SensitivityAnalysisConfig] = None
    simulation_parameters: Optional[Dict[str, Any]] = None
    pipeline: Optional[Dict[str, Any]] = None
    run_timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Union[TricysV2Config, TricysConfig]:
        """Constructs and validates TricysConfig from a raw dictionary.

        If TRICYS 2.0 configuration structure is detected, returns TricysV2Config.
        """
        if is_v2_config(data):
            return TricysV2Config.model_validate(data)
        return cls.model_validate(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes back to a plain dictionary."""
        return self.model_dump(exclude_none=True)
