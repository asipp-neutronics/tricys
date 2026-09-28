from __future__ import annotations

from tricys.workflow.base import (
    BaseSimulationEngine,
    PipelineError,
    PostprocessPlugin,
    PreprocessPlugin,
)
from tricys.workflow.builder import PipelineBuilder
from tricys.workflow.context import PipelineContext
from tricys.workflow.pipeline import WorkflowPipeline
from tricys.workflow.registry import (
    PluginRegistry,
    register_engine,
    register_postprocess_plugin,
    register_preprocess_plugin,
    register_simulation_engine,
)

__all__ = [
    "PipelineContext",
    "PreprocessPlugin",
    "PostprocessPlugin",
    "BaseSimulationEngine",
    "PipelineError",
    "PluginRegistry",
    "WorkflowPipeline",
    "PipelineBuilder",
    "register_preprocess_plugin",
    "register_postprocess_plugin",
    "register_simulation_engine",
    "register_engine",
]

# Auto-register all built-in plugins and engines
import tricys.postprocess.plugins  # noqa: F401
import tricys.preprocessing  # noqa: F401
import tricys.simulation.engines  # noqa: F401
