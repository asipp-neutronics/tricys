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


def test_builder_auto_inference_v2_plugins():
    v2_config = {
        "version": "2.0",
        "model": {"fileName": "models/Cycle.mo", "modelName": "Cycle"},
        "scenario": {"stopTime": 864000.0, "stepSize": 1.0},
        "task": {
            "type": "sweep",
            "parallel": True,
            "parameters": [
                {"name": "Case1", "TBR": 1.15},
                {"name": "Case2", "TBR": 1.10},
            ],
            "metrics": {
                "peak_inv": {"source": "sds.I[1]", "evaluate": "peak_value"},
            },
        },
        "plugins": {
            "postprocess": [
                {
                    "name": "auditor",
                    "plugin": "mass_conservation",
                    "params": {"tolerance": 1e-4},
                },
                {
                    "name": "archive",
                    "plugin": "auto_compress",
                    "params": {"format": "zip"},
                },
            ],
        },
    }
    from tricys.workflow.base import PostprocessPlugin
    from tricys.workflow.registry import PluginRegistry

    class DummyArchivePlugin(PostprocessPlugin):
        name = "archive"

        def run(self, context):
            pass

    PluginRegistry.register_postprocess("archive", DummyArchivePlugin)

    pipeline = PipelineBuilder.build(v2_config)
    pre_names = [p.name for p in pipeline.preprocessors]
    post_names = [p.name for p in pipeline.postprocessors]

    assert "setup" in pre_names
    assert "compiler" in pre_names  # Inferred because task.parallel is True
    assert pipeline.engine.name == "standard"
    assert "storage" in post_names
    assert "safety" in post_names  # Inferred from plugins.postprocess.auditor
    assert "archive" in post_names  # Inferred from plugins.postprocess.archive
