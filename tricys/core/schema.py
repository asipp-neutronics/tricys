from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class BaseDTO(BaseModel):
    """Base DTO allowing arbitrary fields for forward/backward schema compatibility."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )


class PathsConfig(BaseDTO):
    """File paths configuration."""

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
    """Numerical solver and simulation execution settings."""

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
    """Fusion Operation Cycle scenario injection configuration."""

    foc_path: str = Field(default="", description="Path to .foc DSL schedule file")
    foc_component: Optional[str] = Field(
        default=None, description="Target component name to replace"
    )
    strategy: str = Field(
        default="table", description="Injection strategy: table, block, parameter"
    )


class SafetyAuditorConfig(BaseDTO):
    """Plant-wide tritium mass conservation audit settings."""

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
    """Safety threshold alarm rule definition."""

    variable: str = Field(description="Name of the variable to monitor")
    threshold: float = Field(description="Numeric limit threshold")
    condition: str = Field(default="gt", description="Condition: gt, lt, gte, lte, eq")
    level: str = Field(
        default="warning", description="Alarm severity: info, warning, critical"
    )


class StaticAlarmConfig(BaseDTO):
    """Safety alarm rules collection."""

    rules: List[Dict[str, Any]] = Field(
        default_factory=list, description="Threshold monitoring rules"
    )


class SensitivityAnalysisConfig(BaseDTO):
    """Sensitivity and uncertainty analysis configuration."""

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
    """Preprocessing directives."""

    sampling: Optional[Dict[str, Any]] = Field(
        default=None, description="General sampling specifications"
    )
    isolation: bool = Field(
        default=True, description="Isolate workspace before running"
    )


class TricysConfig(BaseDTO):
    """Root strongly-typed configuration DTO for Tricys simulations."""

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
    def from_dict(cls, data: Dict[str, Any]) -> TricysConfig:
        """Constructs and validates TricysConfig from a raw dictionary."""
        return cls.model_validate(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes back to a plain dictionary."""
        return self.model_dump(exclude_none=True)
