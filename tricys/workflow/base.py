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
