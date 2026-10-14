from __future__ import annotations

import tempfile
from pathlib import Path

from tricys.workflow.base import (
    BaseSimulationEngine,
    PostprocessPlugin,
    PreprocessPlugin,
)
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import (
    PluginRegistry,
    register_engine,
    register_postprocess_plugin,
    register_preprocess_plugin,
)


def test_builtin_plugins_registered():
    # Functional preprocess plugins
    assert PluginRegistry.get_preprocess_class("setup") is not None
    assert PluginRegistry.get_preprocess_class("modelica_setup") is not None
    assert PluginRegistry.get_preprocess_class("compiler") is not None
    assert PluginRegistry.get_preprocess_class("foc") is not None
    assert PluginRegistry.get_preprocess_class("interceptor") is not None

    # Legacy preprocess aliases
    assert PluginRegistry.get_preprocess_class("isolation") is not None
    assert PluginRegistry.get_preprocess_class("job_generator") is not None
    assert PluginRegistry.get_preprocess_class("foc_injector") is not None
    assert PluginRegistry.get_preprocess_class("interceptor_builder") is not None

    # Functional postprocess plugins
    assert PluginRegistry.get_postprocess_class("storage") is not None
    assert PluginRegistry.get_postprocess_class("data_storage") is not None
    assert PluginRegistry.get_postprocess_class("safety") is not None
    assert PluginRegistry.get_postprocess_class("report") is not None
    assert PluginRegistry.get_postprocess_class("sensitivity_analysis") is not None
    assert PluginRegistry.get_postprocess_class("sensitivity") is not None

    # Legacy postprocess aliases
    assert PluginRegistry.get_postprocess_class("hdf5_storage") is not None
    assert PluginRegistry.get_postprocess_class("csv_export") is not None
    assert PluginRegistry.get_postprocess_class("metric") is not None
    assert PluginRegistry.get_postprocess_class("offline_auditor") is not None
    assert PluginRegistry.get_postprocess_class("baseline_plot") is not None
    assert PluginRegistry.get_postprocess_class("static_alarm") is not None
    assert PluginRegistry.get_postprocess_class("legacy_tasks") is not None

    # Simulation Engines
    assert PluginRegistry.get_engine_class("standard") is not None
    assert PluginRegistry.get_engine_class("offline_table") is not None
    assert PluginRegistry.get_engine_class("offline_cosim") is not None
    assert PluginRegistry.get_engine_class("online_step") is not None
    assert PluginRegistry.get_engine_class("online_oms") is not None
    assert PluginRegistry.get_engine_class("online_cosim") is not None
    assert PluginRegistry.get_engine_class("param_inject") is not None
    assert PluginRegistry.get_engine_class("parameter_injection") is not None


def test_decorator_registration():
    @register_preprocess_plugin("dummy_pre_test")
    class DummyPre(PreprocessPlugin):
        name = "dummy_pre_test"

        def run(self, context: PipelineContext) -> None:
            context.set_metadata("pre_executed", True)

    @register_postprocess_plugin("dummy_post_test")
    class DummyPost(PostprocessPlugin):
        name = "dummy_post_test"

        def run(self, context: PipelineContext) -> None:
            context.set_metadata("post_executed", True)

    @register_engine("dummy_engine_test")
    class DummyEngine(BaseSimulationEngine):
        name = "dummy_engine_test"

        def execute(self, context: PipelineContext) -> None:
            context.set_metadata("engine_executed", True)

    assert PluginRegistry.get_preprocess_class("dummy_pre_test") is DummyPre
    assert PluginRegistry.get_postprocess_class("dummy_post_test") is DummyPost
    assert PluginRegistry.get_engine_class("dummy_engine_test") is DummyEngine

    pre_inst = PluginRegistry.resolve_preprocess_plugin("dummy_pre_test")
    assert isinstance(pre_inst, DummyPre)

    post_inst = PluginRegistry.resolve_postprocess_plugin("dummy_post_test")
    assert isinstance(post_inst, DummyPost)


def test_dynamic_script_loading():
    script_content = """
from tricys.workflow.base import PreprocessPlugin
from tricys.workflow.context import PipelineContext

class CustomExternalPre(PreprocessPlugin):
    name = "custom_ext"
    def run(self, context: PipelineContext) -> None:
        context.set_metadata("external_loaded", True)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script_content)
        temp_script_path = f.name

    try:
        plugin = PluginRegistry.resolve_preprocess_plugin(
            {"script_path": temp_script_path, "class": "CustomExternalPre"}
        )
        assert isinstance(plugin, PreprocessPlugin)
        ctx = PipelineContext.from_config({})
        plugin.run(ctx)
        assert ctx.get_metadata("external_loaded") is True
    finally:
        Path(temp_script_path).unlink(missing_ok=True)
