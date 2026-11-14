from __future__ import annotations

import logging
import time
from typing import List, Optional

from tricys.workflow.base import (
    BaseSimulationEngine,
    PipelineError,
    PostprocessPlugin,
    PreprocessPlugin,
)
from tricys.workflow.context import PipelineContext

logger = logging.getLogger(__name__)


class WorkflowPipeline:
    """Orchestrator for the pluggable lifecycle: Preprocessing -> Simulation -> Postprocessing."""

    def __init__(
        self,
        engine: BaseSimulationEngine,
        preprocessors: Optional[List[PreprocessPlugin]] = None,
        postprocessors: Optional[List[PostprocessPlugin]] = None,
        name: str = "TricysPipeline",
    ) -> None:
        self.name = name
        self.engine = engine
        self.preprocessors: List[PreprocessPlugin] = preprocessors or []
        self.postprocessors: List[PostprocessPlugin] = postprocessors or []

    def add_preprocessor(self, plugin: PreprocessPlugin) -> WorkflowPipeline:
        self.preprocessors.append(plugin)
        return self

    def add_postprocessor(self, plugin: PostprocessPlugin) -> WorkflowPipeline:
        self.postprocessors.append(plugin)
        return self

    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the entire pipeline in strict sequential order."""
        logger.info(f"[{self.name}] Initiating workflow execution")
        start_total = time.time()

        # ----------------------------------------------------
        # 1. Preprocessing Stage
        # ----------------------------------------------------
        context.execution_status = "preprocessing"
        logger.info(
            f"[{self.name}] Running Preprocessing Stage ({len(self.preprocessors)} plugins)"
        )
        for i, plugin in enumerate(self.preprocessors, 1):
            plugin_name = getattr(plugin, "name", plugin.__class__.__name__)
            try:
                if not plugin.validate(context):
                    logger.info(
                        f"[{self.name}] Skipping preprocess plugin '{plugin_name}' (validation skipped)"
                    )
                    continue

                logger.info(
                    f"[{self.name}] Executing preprocess plugin {i}/{len(self.preprocessors)}: '{plugin_name}'"
                )
                t0 = time.time()
                plugin.run(context)
                elapsed = time.time() - t0
                logger.debug(
                    f"[{self.name}] Preprocess plugin '{plugin_name}' finished in {elapsed:.3f}s"
                )
            except Exception as exc:
                context.execution_status = "failed"
                context.execution_error = (
                    f"Preprocessing failed in plugin '{plugin_name}': {exc}"
                )
                logger.error(context.execution_error, exc_info=True)
                raise PipelineError(context.execution_error) from exc

        # ----------------------------------------------------
        # 2. Simulation Stage (Core Engine)
        # ----------------------------------------------------
        context.execution_status = "simulating"
        engine_name = getattr(self.engine, "name", self.engine.__class__.__name__)
        logger.info(f"[{self.name}] Running Simulation Stage (Engine: '{engine_name}')")
        try:
            t0 = time.time()
            self.engine.execute(context)
            elapsed = time.time() - t0
            logger.info(
                f"[{self.name}] Simulation Engine '{engine_name}' finished in {elapsed:.3f}s"
            )
        except Exception as exc:
            context.execution_status = "failed"
            context.execution_error = f"Simulation engine '{engine_name}' failed: {exc}"
            logger.error(context.execution_error, exc_info=True)
            raise PipelineError(context.execution_error) from exc

        # ----------------------------------------------------
        # 3. Postprocessing Stage
        # ----------------------------------------------------
        context.execution_status = "postprocessing"
        logger.info(
            f"[{self.name}] Running Postprocessing Stage ({len(self.postprocessors)} plugins)"
        )
        for i, plugin in enumerate(self.postprocessors, 1):
            plugin_name = getattr(plugin, "name", plugin.__class__.__name__)
            try:
                if not plugin.validate(context):
                    logger.info(
                        f"[{self.name}] Skipping postprocess plugin '{plugin_name}' (validation skipped)"
                    )
                    continue

                logger.info(
                    f"[{self.name}] Executing postprocess plugin {i}/{len(self.postprocessors)}: '{plugin_name}'"
                )
                t0 = time.time()
                plugin.run(context)
                elapsed = time.time() - t0
                logger.debug(
                    f"[{self.name}] Postprocess plugin '{plugin_name}' finished in {elapsed:.3f}s"
                )
            except Exception as exc:
                # Log postprocessing error but allow remaining pipeline postprocessors or mark warning
                logger.error(
                    f"[{self.name}] Postprocessing plugin '{plugin_name}' failed: {exc}",
                    exc_info=True,
                )
                context.metadata[f"postprocess_error_{plugin_name}"] = str(exc)

        context.execution_status = "completed"
        total_elapsed = time.time() - start_total
        logger.info(
            f"[{self.name}] Workflow finished successfully in {total_elapsed:.3f}s"
        )
        return context
