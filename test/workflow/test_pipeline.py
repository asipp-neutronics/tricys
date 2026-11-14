from __future__ import annotations

import pytest

from tricys.workflow.base import (
    BaseSimulationEngine,
    PipelineError,
    PostprocessPlugin,
    PreprocessPlugin,
)
from tricys.workflow.context import PipelineContext
from tricys.workflow.pipeline import WorkflowPipeline


class MockPreprocess(PreprocessPlugin):
    name = "mock_preprocess"

    def run(self, context: PipelineContext) -> None:
        context.set_metadata("pre_val", 42)
        context.jobs = [{"param1": 100}]


class MockFailingPreprocess(PreprocessPlugin):
    name = "mock_failing_preprocess"

    def run(self, context: PipelineContext) -> None:
        raise ValueError("Simulated preprocess failure")


class MockSimulationEngine(BaseSimulationEngine):
    name = "mock_engine"

    def execute(self, context: PipelineContext) -> None:
        assert context.get_metadata("pre_val") == 42
        assert len(context.jobs) == 1
        context.raw_results = [{"job_id": 1, "output": 200}]
        context.set_metadata("engine_ran", True)


class MockPostprocess(PostprocessPlugin):
    name = "mock_postprocess"

    def run(self, context: PipelineContext) -> None:
        assert context.get_metadata("engine_ran") is True
        assert len(context.raw_results) == 1
        context.set_metadata("post_val", 999)


def test_workflow_pipeline_end_to_end():
    config = {
        "paths": {"results_dir": "results", "temp_dir": "temp"},
        "simulation": {"model_name": "TestModel", "stop_time": 10.0, "step_size": 0.1},
    }
    context = PipelineContext.from_config(config)

    pipeline = WorkflowPipeline(
        engine=MockSimulationEngine(),
        preprocessors=[MockPreprocess()],
        postprocessors=[MockPostprocess()],
        name="UnitTestPipeline",
    )

    result_ctx = pipeline.run(context)

    assert result_ctx.execution_status == "completed"
    assert result_ctx.get_metadata("pre_val") == 42
    assert result_ctx.get_metadata("engine_ran") is True
    assert result_ctx.get_metadata("post_val") == 999
    assert result_ctx.raw_results == [{"job_id": 1, "output": 200}]


def test_workflow_pipeline_preprocess_failure():
    config = {"paths": {}, "simulation": {}}
    context = PipelineContext.from_config(config)

    pipeline = WorkflowPipeline(
        engine=MockSimulationEngine(),
        preprocessors=[MockFailingPreprocess()],
        name="FailurePipeline",
    )

    with pytest.raises(PipelineError) as exc_info:
        pipeline.run(context)

    assert "Simulated preprocess failure" in str(exc_info.value)
    assert context.execution_status == "failed"
