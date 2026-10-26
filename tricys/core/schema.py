from __future__ import annotations

import logging
from pathlib import Path
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
    foc_path: Optional[str] = Field(
        default=None, description="Alternative alias for foc_file"
    )
    target_component: Optional[str] = Field(
        default=None, description="Target Modelica instance name to replace"
    )
    foc_component: Optional[str] = Field(
        default=None, description="Alternative alias for target_component"
    )
    strategy: str = Field(
        default="table", description="FOC pulse injection strategy ('table' or 'array')"
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

    @model_validator(mode="before")
    @classmethod
    def unify_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "foc_path" in data and "foc_file" not in data:
                data["foc_file"] = data["foc_path"]
            elif "foc_file" in data and "foc_path" not in data:
                data["foc_path"] = data["foc_file"]
            if "target_component" in data and "foc_component" not in data:
                data["foc_component"] = data["target_component"]
            elif "foc_component" in data and "target_component" not in data:
                data["target_component"] = data["foc_component"]
        return data


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
    time_base: Literal["second", "hour"] = Field(
        default="second",
        description="Base time unit for simulation conditions: 'second' (default native SI) or 'hour' (auto-scaled).",
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
    threshold: Optional[Union[float, List[float]]] = Field(
        default=None, description="Target numerical threshold for turning point / limit"
    )
    threshold_source: Optional[str] = Field(
        default=None,
        description="Modelica variable column name providing dynamic/model-computed threshold (e.g. comp11_Storage.I_reserve_req or I_reserve_req)",
    )
    reserve_source: Optional[str] = Field(
        default=None,
        description="Alias for threshold_source specifically for reserve inventory requirement",
    )
    direction: Optional[str] = Field(
        default="min_to_max",
        description="Direction of slope reversal: min_to_max or max_to_min",
    )
    operator: Optional[str] = Field(default=None, description="Alias for evaluate")
    target_value: Optional[Union[float, List[float]]] = Field(
        default=None, description="Alias for threshold"
    )
    max_value: Optional[Union[float, List[float]]] = Field(
        default=None, description="Alias for threshold"
    )
    smooth_window: Optional[Union[float, str]] = Field(
        default=None,
        description="Temporal smoothing window in time units (e.g. 96.0 h) or parameter name. None means no smoothing.",
    )
    cycle_period: Optional[Union[float, str]] = Field(
        default=None, description="Alias for smooth_window"
    )
    smooth_method: Optional[str] = Field(
        default="mean",
        description="Smoothing method: mean (rolling average) or median",
    )
    interpolate: Optional[bool] = Field(
        default=False,
        description="Whether to use sub-step linear interpolation for threshold crossing",
    )

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
            if "cycle_period" in data and "smooth_window" not in data:
                data["smooth_window"] = data["cycle_period"]
            if "smooth_window" in data and "cycle_period" not in data:
                data["cycle_period"] = data["smooth_window"]
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
    conversion_factor: Optional[float] = Field(
        default=None, description="Scaling factor to convert raw seconds to target unit"
    )
    threshold: Optional[Union[float, List[float]]] = Field(
        default=None, description="Upper/lower safety threshold"
    )

    # Smoothing and interpolation
    smooth_window: Optional[Union[float, str]] = Field(
        default=None,
        description="Temporal smoothing window in time units (e.g. 96.0 h) or parameter name. None means no smoothing.",
    )
    cycle_period: Optional[Union[float, str]] = Field(
        default=None, description="Alias for smooth_window"
    )
    smooth_method: Optional[str] = Field(
        default="mean",
        description="Smoothing method: mean (rolling average) or median",
    )
    interpolate: Optional[bool] = Field(
        default=False,
        description="Whether to use sub-step linear interpolation for threshold crossing",
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
            if "cycle_period" in data and "smooth_window" not in data:
                data["smooth_window"] = data["cycle_period"]
            if "smooth_window" in data and "cycle_period" not in data:
                data["cycle_period"] = data["smooth_window"]
        return data


class SafetyAuditorConfig(BaseDTO):
    """Mass Balance Auditor configuration."""

    enabled: bool = Field(
        default=False, description="Whether mass balance audit is enabled"
    )
    warn_threshold_g: float = Field(
        default=0.5, description="Warning threshold for mass balance error in grams"
    )
    kill_threshold_g: float = Field(
        default=50.0, description="Kill threshold for mass balance error in grams"
    )
    tolerance: float = Field(
        default=1e-4, description="Relative tolerance for mass conservation"
    )
    patterns: Dict[str, List[str]] = Field(
        default_factory=dict, description="Pattern definitions for mass balance audit"
    )
    subsystems: List[str] = Field(
        default_factory=list, description="Subsystems to include in audit"
    )


class OuterBisectionConfig(BaseDTO):
    """Configuration for the outer loop of nested bisection optimization."""

    parameter: str = Field(
        default="I_S_0", description="Parameter to optimize in outer loop (e.g. I_S_0)"
    )
    search_bounds: Optional[List[float]] = Field(
        default=None, description="Search bounds [low, high] (mandatory for Track 2)"
    )
    tolerance: float = Field(
        default=1.0, description="Outer loop convergence tolerance"
    )
    max_iterations: int = Field(default=15, description="Max outer iterations")
    threshold_source: Optional[str] = Field(
        default=None,
        description="Modelica variable column name providing dynamic/model-computed threshold",
    )
    reserve_column: Optional[str] = Field(
        default=None,
        description="Modelica variable column name for reserve inventory requirement",
    )
    target_metric: TargetMetricConfig = Field(
        description="Target metric condition for outer loop"
    )

    @field_validator("search_bounds")
    @classmethod
    def validate_bounds(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if not v or len(v) != 2 or float(v[0]) >= float(v[1]):
            raise ValueError(
                "search_bounds must be a list of two numbers [low, high] with low < high"
            )
        return [float(x) for x in v]


class InnerBisectionConfig(BaseDTO):
    """Configuration for the inner loop of nested bisection optimization."""

    parameter: str = Field(
        default="Lambda",
        description="Parameter to optimize in inner loop (e.g. Lambda)",
    )
    search_bounds: Optional[List[float]] = Field(
        default=None, description="Search bounds [low, high] (mandatory for Track 2)"
    )
    tolerance: float = Field(
        default=0.001, description="Inner loop convergence tolerance"
    )
    max_iterations: int = Field(default=15, description="Max inner iterations")
    metric_name: str = Field(
        default="Required_TBR", description="Name of the inner result metric"
    )
    target_metric: TargetMetricConfig = Field(
        description="Target metric condition for inner loop"
    )

    @field_validator("search_bounds")
    @classmethod
    def validate_bounds(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if not v or len(v) != 2 or float(v[0]) >= float(v[1]):
            raise ValueError(
                "search_bounds must be a list of two numbers [low, high] with low < high"
            )
        return [float(x) for x in v]


class Track1BasisConfig(BaseDTO):
    """Configuration for Track 1: Dual-Basis Closed-Form Direct Optimization."""

    mode: Literal["direct", "bisection"] = Field(
        default="direct",
        description="Solving mode: 'direct' (analytical closed-form 0-iteration) or 'bisection' (vector algebraic search)",
    )
    cache_basis: bool = Field(
        default=True,
        description="Enable cross-task basis reuse when dynamic time constants are identical",
    )
    sds_column: Optional[str] = Field(
        default=None,
        description="Explicit storage inventory column. If None, auto-detected via detect_sds_column",
    )
    reserve_column: Optional[str] = Field(
        default=None,
        description="Modelica variable column name for reserve inventory threshold (e.g. comp11_Storage.I_reserve_req)",
    )
    interpolation_method: Literal["pchip", "cubic", "linear"] = Field(
        default="pchip",
        description="Interpolation method for basis functions at doubling time: 'pchip' (shape-preserving cubic), 'cubic', or 'linear'",
    )
    subgrid_nadir_refinement: bool = Field(
        default=True,
        description="Enable continuous quadratic vertex refinement for lowest storage nadir",
    )
    relative_tolerance: float = Field(
        default=1e-5,
        description="Relative convergence tolerance for algebraic bisection mode",
    )


class NestedBisectionConfig(BaseDTO):
    """Nested/bi-level bisection optimization specification supporting Dual-Track."""

    track: Literal["auto", "track1_basis", "track2_bisection"] = Field(
        default="auto",
        description=(
            "Solving track: 'auto' (auto detect), 'track1_basis' (dual-basis closed-form), 'track2_bisection' (nested bisection)"
        ),
    )
    algorithm: Optional[str] = Field(
        default="nested_bisection",
        description="Optimization algorithm identifier",
    )
    track1: Optional[Track1BasisConfig] = Field(
        default_factory=Track1BasisConfig,
        description="Track 1 dual-basis configuration",
    )
    outer: Optional[OuterBisectionConfig] = Field(
        default=None,
        description="Outer loop configuration (mandatory for Track 2)",
    )
    inner: Optional[InnerBisectionConfig] = Field(
        default=None,
        description="Inner loop configuration (mandatory for Track 2)",
    )


class OptimizeTaskConfig(BaseDTO):
    """Level 2 task configuration for standalone closed-loop optimization."""

    type: Literal["optimize"] = "optimize"
    track: Literal["auto", "track1_basis", "track2_bisection"] = Field(
        default="auto",
        description="Solving track selector",
    )
    algorithm: Optional[str] = Field(
        default="nested_bisection",
        description="Optimization algorithm identifier",
    )
    track1: Optional[Track1BasisConfig] = Field(
        default_factory=Track1BasisConfig,
        description="Track 1 dual-basis configuration",
    )
    outer: Optional[OuterBisectionConfig] = None
    inner: Optional[InnerBisectionConfig] = None
    parallel: bool = Field(default=False, description="Enable parallel execution")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Base parameters for simulation point"
    )
    metrics: Dict[str, MetricConfig] = Field(
        default_factory=dict, description="Additional feature extraction metrics"
    )
    auditor: Optional[SafetyAuditorConfig] = Field(
        default=None, description="Task-level mass balance auditor configuration"
    )


class SimulateTaskConfig(BaseDTO):
    """Level 2 task configuration for single-run simulation / single-point bisection."""

    type: Literal["simulate"] = "simulate"
    parallel: bool = Field(default=False, description="Enable parallel execution")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Single-point initial value / parameter override",
    )
    optimization: Optional[NestedBisectionConfig] = Field(
        default=None, description="Optional nested bisection optimization block"
    )
    metrics: Dict[str, MetricConfig] = Field(
        default_factory=dict, description="Feature extraction or bisection metrics"
    )
    auditor: Optional[SafetyAuditorConfig] = Field(
        default=None, description="Task-level mass balance auditor configuration"
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
    parameters: Optional[
        Union[List[Dict[str, Any]], Dict[str, List[Any]], Dict[str, Any]]
    ] = Field(
        default=None,
        description="Sweep space: List of dicts (case list), Dict of lists (grid), or external file ref",
    )
    grid: Optional[Dict[str, Any]] = Field(
        default=None, description="Multi-parameter Cartesian grid sweep specification"
    )
    optimization: Optional[NestedBisectionConfig] = Field(
        default=None,
        description="Optional nested bisection optimization block for each case",
    )
    metrics: Dict[str, MetricConfig] = Field(
        default_factory=dict, description="Summary metrics to compute across all cases"
    )
    auditor: Optional[SafetyAuditorConfig] = Field(
        default=None, description="Task-level mass balance auditor configuration"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_sweep_definition(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("parameters") and not data.get("grid"):
                raise ValueError(
                    "Sweep task requires either 'parameters' or 'grid' to be defined."
                )
            if (
                "parameters" in data
                and isinstance(data["parameters"], list)
                and not data["parameters"]
            ):
                raise ValueError(
                    "parameters list must contain at least one operating case dictionary"
                )
        return data


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
    auditor: Optional[SafetyAuditorConfig] = Field(
        default=None, description="Task-level mass balance auditor configuration"
    )


class SubsystemSpec(BaseDTO):
    """External subsystem replacement specification in co-simulation."""

    target: str = Field(
        description="Target Modelica class to replace, e.g. CFEDR.I_ISS"
    )
    instance: Optional[str] = Field(
        default=None, description="Instance name in parent model"
    )
    solver: Optional[str] = Field(
        default="python_script",
        description="Subsystem solver adapter: aspen_plus, h2iso, fmu, python_script",
    )
    model_file: Optional[str] = Field(
        default=None, description="External subsystem model file"
    )
    handler_module: Optional[str] = Field(
        default=None, description="Python module path for subsystem handler"
    )
    handler_function: Optional[str] = Field(
        default=None, description="Python function/class name in handler module"
    )
    handler_script_path: Optional[str] = Field(
        default=None, description="External script path for subsystem handler"
    )
    local_mass_control: Optional[Dict[str, Any]] = Field(
        default=None, description="Online local mass conservation controls"
    )
    shadow_fmu_fallback: Optional[Dict[str, Any]] = Field(
        default=None, description="Online shadow FMU fallback settings"
    )
    substitute_model: Optional[str] = Field(
        default=None,
        description="Replacement Modelica model name, e.g. Generic_ISS_Adapters.ISS_I_Adapter",
    )
    substitute_file: Optional[str] = Field(
        default=None, description="Replacement Modelica .mo file path"
    )
    aux_files: List[str] = Field(
        default_factory=list,
        description="Auxiliary/dependent Modelica files required by substitute model",
    )
    override_file: Optional[str] = Field(
        default=None, description="Subsystem-specific parameter override file path"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Subsystem runtime parameters"
    )


class ParameterInjectionConfig(BaseDTO):
    """Specification for parameter injection data source and generation method."""

    type: str = Field(
        default="file",
        description="Injection method: 'file' (static override file) or 'handler' (dynamic solver script)",
    )
    override_file: Optional[str] = Field(
        default=None,
        description="Path to parameter override file (.txt) used by -overrideFile",
    )
    handler_script: Optional[str] = Field(
        default=None,
        description="Path to external solver script (e.g. h2iso distillation solver)",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime parameters passed to the external handler script",
    )


class CosimTaskConfig(BaseDTO):
    """Level 2 task configuration for multi-physics co-simulation."""

    type: Literal["cosim"] = "cosim"
    mode: str = Field(
        default="offline_table",
        description="Coupling mode: offline_table (subsystem_replacement), online_step (fmi_online), or parameter_injection",
    )
    exchange_step: Optional[float] = Field(
        default=None, description="Optional macro exchange time step in SI seconds"
    )
    subsystems: List[SubsystemSpec] = Field(
        default_factory=list, description="Subsystems participating in co-simulation"
    )
    injection_type: Optional[str] = Field(
        default="file",
        description="Parameter injection method: 'file' (direct override_file) or 'handler' (solver script)",
    )
    injection: Optional[ParameterInjectionConfig] = Field(
        default=None,
        description="Unified configuration for parameter injection data source and solver handler",
    )
    override_file: Optional[str] = Field(
        default=None,
        description="Path to parameter override file for parameter_injection mode",
    )
    handler_script: Optional[str] = Field(
        default=None,
        description="Task-level external solver script for parameter injection (e.g. h2iso solver)",
    )
    handler_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters passed to the task-level solver handler script",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Co-simulation parameter overrides"
    )
    metrics: Dict[str, MetricConfig] = Field(
        default_factory=dict, description="Co-simulation extraction metrics"
    )
    auditor: Optional[SafetyAuditorConfig] = Field(
        default=None, description="Task-level mass balance auditor configuration"
    )


# TaskTypeUnion enforces that task.type must be one of the valid computation types.
# Invalid types like parse, archive, visualize are strictly rejected!
TaskTypeUnion = Annotated[
    Union[
        SimulateTaskConfig,
        SweepTaskConfig,
        SensitivityTaskConfig,
        CosimTaskConfig,
        OptimizeTaskConfig,
    ],
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

    name: Optional[str] = Field(
        default=None, description="Unique plugin identifier name"
    )
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

    @model_validator(mode="before")
    @classmethod
    def populate_default_name(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("name"):
                data["name"] = (
                    data.get("plugin")
                    or data.get("class")
                    or data.get("class_name")
                    or (
                        Path(data["script"]).stem
                        if data.get("script")
                        else "custom_plugin"
                    )
                )
        return data


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
    3. task: Scientific calculation task (simulate, sweep, sensitivity, cosim) + task.auditor
    4. visualizer: Interactive Web dashboard & report generation (including visualizer.report)
    5. plugins: Lifecycle extension hooks
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
    plugins: Optional[PluginsConfig] = Field(
        default=None, description="Level 1 Pillar: Lifecycle plugins"
    )
    report: Optional[ReportConfig] = Field(
        default=None,
        description="Legacy root-level report delivery engine (canonical location: visualizer.report)",
    )
    auditor: Optional[SafetyAuditorConfig] = Field(
        default=None,
        description="Legacy root-level mass balance auditor configuration (canonical location: task.auditor)",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_v2_canonical_structure(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. Normalize auditor: canonical is task.auditor
            task_dict = data.get("task")
            if isinstance(task_dict, dict):
                if "auditor" in task_dict and "auditor" not in data:
                    data["auditor"] = task_dict["auditor"]
                elif "auditor" in data and "auditor" not in task_dict:
                    task_dict["auditor"] = data["auditor"]

            # 2. Normalize report: canonical is visualizer.report
            vis_dict = data.get("visualizer")
            if isinstance(vis_dict, dict):
                if "report" in vis_dict and "report" not in data:
                    data["report"] = vis_dict["report"]
                elif "report" in data and "report" not in vis_dict:
                    vis_dict["report"] = data["report"]
            elif "report" in data and vis_dict is None:
                data["visualizer"] = {"report": data["report"]}
        return data

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
            time_base=self.scenario.time_base,
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
    time_base: str = Field(default="second", description="Time base: second or hour")
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
