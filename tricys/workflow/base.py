from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from tricys.workflow.context import PipelineContext

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Strongly-typed self-describing metadata contract for workflow plugins and engines."""

    name: str
    category: str  # "preprocess", "engine", "postprocess"
    description: str = ""
    version: str = "2.0.0"
    author: str = "TRICYS Core Team"
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    input_requirements: list[str] = field(default_factory=list)
    output_artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineError(Exception):
    """Base exception for errors during workflow pipeline execution."""


class SecurityValidationError(PipelineError):
    """Raised when an external plugin script violates AST security sandboxing rules."""


class JobExecutionError(PipelineError):
    """Raised when one or more simulation jobs fail during a run, sweep or co-simulation.

    Policy: **any** failed job fails the whole invocation, so the CLI exits non-zero
    instead of reporting a green run with silently missing results. Per-job reasons are
    carried on the exception so callers and logs can show exactly which jobs died.
    """

    def __init__(
        self, engine: str, failures: list[tuple[int, str]], total: int
    ) -> None:
        self.engine = engine
        self.failures = list(failures)
        self.total = total
        detail = "; ".join(f"job {job_id}: {reason}" for job_id, reason in self.failures[:5])
        if len(self.failures) > 5:
            detail += f"; ... (+{len(self.failures) - 5} more)"
        super().__init__(
            f"{engine}: {len(self.failures)}/{total} job(s) failed ({detail})"
        )

    @property
    def failed(self) -> int:
        """Number of failed jobs."""
        return len(self.failures)



class BasePlugin(ABC):
    """Abstract root contract for all TRICYS workflow plugins and engines."""

    name: str = "base_plugin"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """Returns self-describing metadata for this plugin."""
        doc = (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
        return PluginMetadata(
            name=cls.name,
            category="plugin",
            description=doc or f"Plugin: {cls.name}",
        )

    def validate(self, context: PipelineContext) -> bool:
        """Validate if this plugin can/should execute on the given context."""
        return True


class PreprocessPlugin(BasePlugin):
    """Abstract contract for preprocessing plugins.

    Preprocessing plugins inspect, prepare, validate, or transform the model,
    parameters, scenario inputs, and workspaces before simulation begins.
    """

    name: str = "base_preprocess"

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """Returns self-describing metadata for this preprocessing plugin."""
        doc = (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
        return PluginMetadata(
            name=cls.name,
            category="preprocess",
            description=doc or f"Preprocessing plugin: {cls.name}",
        )

    def validate(self, context: PipelineContext) -> bool:
        """Validate if this plugin can/should execute on the given context."""
        return True

    @abstractmethod
    def run(self, context: PipelineContext) -> None:
        """Execute preprocessing step, mutating or augmenting the context."""
        pass


class PostprocessPlugin(BasePlugin):
    """Abstract contract for postprocessing plugins.

    Postprocessing plugins inspect, extract, persist, visualize, audit, or report
    on the simulation results stored in the PipelineContext.
    """

    name: str = "base_postprocess"

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """Returns self-describing metadata for this postprocessing plugin."""
        doc = (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
        return PluginMetadata(
            name=cls.name,
            category="postprocess",
            description=doc or f"Postprocessing plugin: {cls.name}",
        )

    def validate(self, context: PipelineContext) -> bool:
        """Validate if this plugin can/should execute on the given context."""
        return True

    @abstractmethod
    def run(self, context: PipelineContext) -> None:
        """Execute postprocessing step on simulation outputs."""
        pass


class BaseSimulationEngine(BasePlugin):
    """Abstract contract for core simulation engines.

    Engines represent the pure numerical and co-simulation execution kernel.
    """

    name: str = "base_engine"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """Returns self-describing metadata for this simulation engine."""
        doc = (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
        return PluginMetadata(
            name=cls.name,
            category="engine",
            description=doc or f"Simulation engine: {cls.name}",
        )

    @abstractmethod
    def execute(self, context: PipelineContext) -> None:
        """Execute simulation using prepared context and write results to context."""
        pass
