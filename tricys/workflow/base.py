from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from tricys.workflow.context import PipelineContext

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base exception for errors during workflow pipeline execution."""


class PreprocessPlugin(ABC):
    """Abstract contract for preprocessing plugins.

    Preprocessing plugins inspect, prepare, validate, or transform the model,
    parameters, scenario inputs, and workspaces before simulation begins.
    """

    name: str = "base_preprocess"

    def __init__(self, **params: Any) -> None:
        self.params = params

    def validate(self, context: PipelineContext) -> bool:
        """Validate if this plugin can/should execute on the given context."""
        return True

    @abstractmethod
    def run(self, context: PipelineContext) -> None:
        """Execute preprocessing step, mutating or augmenting the context."""
        pass


class PostprocessPlugin(ABC):
    """Abstract contract for postprocessing plugins.

    Postprocessing plugins inspect, extract, persist, visualize, audit, or report
    on the simulation results stored in the PipelineContext.
    """

    name: str = "base_postprocess"

    def __init__(self, **params: Any) -> None:
        self.params = params

    def validate(self, context: PipelineContext) -> bool:
        """Validate if this plugin can/should execute on the given context."""
        return True

    @abstractmethod
    def run(self, context: PipelineContext) -> None:
        """Execute postprocessing step on simulation outputs."""
        pass


class BaseSimulationEngine(ABC):
    """Abstract contract for core simulation engines.

    Engines represent the pure numerical and co-simulation execution kernel.
    """

    name: str = "base_engine"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def execute(self, context: PipelineContext) -> None:
        """Execute simulation using prepared context and write results to context."""
        pass
