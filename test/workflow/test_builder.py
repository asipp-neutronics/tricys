from __future__ import annotations

from tricys.workflow.base import (
    PostprocessPlugin,
    PreprocessPlugin,
)
from tricys.workflow.builder import PipelineBuilder
from tricys.workflow.registry import (
    register_postprocess_plugin,
    register_preprocess_plugin,
)


def test_builder_auto_inference_standard():
    config = {
        "paths": {"package_path": "dummy.mo", "results_dir": "results"},
        "simulation": {
            "model_name": "TestModel",
            "stop_time": 10.0,
            "step_size": 0.1,
            "concurrent": False,
        },
        "simulation_parameters": {"param1": [1, 2, 3]},
    }
    pipeline = PipelineBuilder.build(config, export_csv=True)

    pre_names = [p.name for p in pipeline.preprocessors]
    post_names = [p.name for p in pipeline.postprocessors]

    assert "setup" in pre_names
    assert pipeline.engine.name == "standard"
    assert "storage" in post_names
    assert getattr(pipeline.postprocessors[0], "export_csv", None) is True


def test_builder_auto_inference_foc_and_auditor():
    config = {
        "paths": {"package_path": "dummy.mo", "results_dir": "results"},
        "simulation": {
            "model_name": "TestModel",
            "stop_time": 10.0,
            "step_size": 0.1,
            "concurrent": True,
        },
        "foc": {"foc_path": "scenario.foc"},
        "auditor": {"enabled": True},
    }
    pipeline = PipelineBuilder.build(config, export_csv=False)

    pre_names = [p.name for p in pipeline.preprocessors]
    post_names = [p.name for p in pipeline.postprocessors]

    assert "setup" in pre_names
    assert "foc" in pre_names
    assert "compiler" in pre_names  # Concurrent mode infers compiler
    assert "storage" in post_names
    assert "safety" in post_names


def test_builder_explicit_pipeline_legacy_aliases():
    config = {
        "pipeline": {
            "name": "LegacyAliasPipeline",
            "preprocessing": [{"name": "isolation"}, {"name": "job_generator"}],
            "simulation": {"name": "standard"},
            "postprocessing": [{"name": "hdf5_storage"}, {"name": "offline_auditor"}],
        }
    }
    pipeline = PipelineBuilder.build(config)
    assert len(pipeline.preprocessors) == 2
    assert len(pipeline.postprocessors) == 2
    assert pipeline.engine.name == "standard"


def test_builder_explicit_pipeline():
    @register_preprocess_plugin("my_explicit_pre")
    class ExplicitPre(PreprocessPlugin):
        name = "my_explicit_pre"

        def run(self, context):
            pass

    @register_postprocess_plugin("my_explicit_post")
    class ExplicitPost(PostprocessPlugin):
        name = "my_explicit_post"

        def run(self, context):
            pass

    config = {
        "pipeline": {
            "name": "CustomUserPipeline",
            "preprocessing": [{"name": "my_explicit_pre"}],
            "simulation": {"name": "standard"},
            "postprocessing": [{"name": "my_explicit_post"}],
        }
    }

    pipeline = PipelineBuilder.build(config)
    assert pipeline.name == "CustomUserPipeline"
    assert len(pipeline.preprocessors) == 1
    assert pipeline.preprocessors[0].name == "my_explicit_pre"
    assert pipeline.engine.name == "standard"
    assert len(pipeline.postprocessors) == 1
    assert pipeline.postprocessors[0].name == "my_explicit_post"
