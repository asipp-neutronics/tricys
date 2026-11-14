from __future__ import annotations

from tricys.core.schema import (
    FOCConfig,
    PathsConfig,
    SimulationConfig,
    TricysConfig,
)
from tricys.workflow.context import PipelineContext


def test_schema_models_basic():
    paths = PathsConfig(
        package_path="path/to/pkg.mo", results_dir="out", temp_dir="tmp"
    )
    assert paths.package_path == "path/to/pkg.mo"
    assert paths.results_dir == "out"

    sim = SimulationConfig(model_name="Cycle", stop_time=100.0, step_size=0.1)
    assert sim.model_name == "Cycle"
    assert sim.stop_time == 100.0
    assert sim.step_size == 0.1

    foc = FOCConfig(foc_path="scenario.foc", strategy="table")
    assert foc.foc_path == "scenario.foc"
    assert foc.strategy == "table"


def test_tricys_config_from_dict_and_extra_fields():
    raw = {
        "paths": {"package_path": "model.mo"},
        "simulation": {"model_name": "TestModel", "stop_time": 500.0},
        "custom_plugin_config": {"custom_key": 42},
    }
    cfg = TricysConfig.from_dict(raw)
    assert cfg.paths.package_path == "model.mo"
    assert cfg.simulation.model_name == "TestModel"
    assert cfg.simulation.stop_time == 500.0
    # Extra field is preserved
    assert getattr(cfg, "custom_plugin_config") == {"custom_key": 42}


def test_pipeline_context_has_typed_config():
    config = {
        "paths": {"package_path": "pkg.mo", "results_dir": "results_dir"},
        "simulation": {"model_name": "ModelX", "stop_time": 200.0},
    }
    context = PipelineContext.from_config(config)
    assert context.typed_config is not None
    assert isinstance(context.typed_config, TricysConfig)
    assert context.typed_config.simulation.model_name == "ModelX"
    assert context.typed_config.simulation.stop_time == 200.0
