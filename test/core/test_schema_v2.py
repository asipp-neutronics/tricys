from __future__ import annotations

import pytest
from pydantic import ValidationError

from tricys.core.schema import (
    AIInsightConfig,
    CosimTaskConfig,
    EngineSpec,
    LibraryDep,
    MetricConfig,
    ModelConfig,
    PluginsConfig,
    PluginSpec,
    PulseConfig,
    ReportConfig,
    ScenarioConfig,
    SensitivityTaskConfig,
    SimulateTaskConfig,
    SubsystemSpec,
    SweepTaskConfig,
    TargetMetricConfig,
    TricysConfig,
    TricysV2Config,
    VisualizerConfig,
    is_v2_config,
)


@pytest.fixture
def minimal_v2_simulate_dict():
    return {
        "version": "2.0",
        "model": {
            "fileName": "models/Cycle.mo",
            "modelName": "example_model.Cycle",
        },
        "scenario": {
            "stopTime": 1000.0,
            "stepSize": 1.0,
        },
        "task": {
            "type": "simulate",
            "parameters": {"WDS.threshold": 1200.0},
            "metrics": {
                "max_inv": {
                    "source": "sds.I[1]",
                    "evaluate": "peak_value",
                }
            },
        },
    }


# ==============================================================================
# Model & Scenario Pillars
# ==============================================================================


def test_model_config_valid():
    model = ModelConfig(
        fileName="path/to/package.mo",
        modelName="CFEDR.System",
        loadModel=[LibraryDep(name="Modelica", version="4.0.0")],
        cflags="-O3",
        target="gcc",
    )
    assert model.fileName == "path/to/package.mo"
    assert model.modelName == "CFEDR.System"
    assert len(model.loadModel) == 1
    assert model.loadModel[0].name == "Modelica"
    assert model.loadModel[0].version == "4.0.0"
    assert model.cflags == "-O3"


def test_scenario_config_valid_and_validator():
    scenario = ScenarioConfig(
        stopTime=86400.0,
        startTime=0.0,
        stepSize=0.5,
        method="dassl",
        tolerance=1e-6,
        variableFilter="time|sds.*",
    )
    assert scenario.stopTime == 86400.0
    assert scenario.stepSize == 0.5
    assert scenario.method == "dassl"

    # stopTime <= 0 should fail
    with pytest.raises(ValidationError):
        ScenarioConfig(stopTime=0.0)

    with pytest.raises(ValidationError):
        ScenarioConfig(stopTime=-100.0)


def test_pulse_config():
    pulse = PulseConfig(
        foc_file="schedule.foc",
        target_component="source_component",
        power_scale=1.5,
        params={"duty_cycle": 0.8},
    )
    assert pulse.foc_file == "schedule.foc"
    assert pulse.power_scale == 1.5


# ==============================================================================
# Task Pillar & Discriminator Union
# ==============================================================================


def test_simulate_task_config():
    task = SimulateTaskConfig(
        type="simulate",
        parameters={"TBR": 1.15},
        metrics={
            "final_val": MetricConfig(source="sds.I[1]", evaluate="final_value"),
        },
    )
    assert task.type == "simulate"
    assert task.parameters["TBR"] == 1.15
    assert task.metrics["final_val"].evaluate == "final_value"


def test_sweep_task_config_case_list():
    task = SweepTaskConfig(
        type="sweep",
        parallel=True,
        max_workers=4,
        parameters=[
            {"name": "Case1", "TBR": 1.15},
            {"name": "Case2", "TBR": 1.25},
        ],
        metrics={
            "peak_inv": MetricConfig(source="sds.I[1]", evaluate="peak_value"),
        },
    )
    assert task.type == "sweep"
    assert task.parallel is True
    assert task.max_workers == 4
    assert len(task.parameters) == 2


def test_sweep_task_config_validation_failures():
    # Empty list
    with pytest.raises(ValidationError):
        SweepTaskConfig(type="sweep", parameters=[])

    # Non-dict in list
    with pytest.raises(ValidationError):
        SweepTaskConfig(type="sweep", parameters=["invalid_item"])

    # Invalid type for parameters (e.g. integer)
    with pytest.raises(ValidationError):
        SweepTaskConfig(type="sweep", parameters=12345)


def test_sweep_task_config_grid_and_file():
    # Grid of lists
    task_grid = SweepTaskConfig(
        type="sweep",
        parameters={"blanket.TBR": [1.05, 1.15, 1.25], "tep.eta": [0.8, 0.9]},
    )
    assert isinstance(task_grid.parameters, dict)

    # File reference
    task_file = SweepTaskConfig(
        type="sweep",
        parameters={"file": "sweep_space.csv"},
    )
    assert task_file.parameters["file"] == "sweep_space.csv"


def test_sensitivity_task_config():
    task = SensitivityTaskConfig(
        type="sensitivity",
        method="sobol",
        samples=64,
        seed=42,
        parallel=True,
        max_workers=8,
        parameters={"TBR": [1.05, 1.35], "WDS.threshold": [800.0, 1600.0]},
        metrics={
            "final_storage": MetricConfig(
                source="sds.I[1]",
                evaluate="final_value",
            )
        },
    )
    assert task.type == "sensitivity"
    assert task.method == "sobol"
    assert task.samples == 64
    assert task.seed == 42


def test_sensitivity_task_config_invalid_method():
    with pytest.raises(ValidationError):
        SensitivityTaskConfig(
            type="sensitivity",
            method="unsupported_method",  # Must be sobol, latin, or morris
            parameters={"TBR": [1.05, 1.35]},
            metrics={},
        )


def test_cosim_task_config():
    task = CosimTaskConfig(
        type="cosim",
        mode="subsystem_replacement",
        exchange_step=0.5,
        subsystems=[
            SubsystemSpec(
                target="CFEDR.I_ISS",
                instance="iss",
                solver="h2iso",
                model_file="models/iss.json",
                params={"reflux_ratio": 12.0},
            )
        ],
        parameters={"global_scale": 1.0},
        metrics={
            "t_inv": MetricConfig(source="iss.total_inventory", evaluate="final_value")
        },
    )
    assert task.type == "cosim"
    assert task.mode == "subsystem_replacement"
    assert task.exchange_step == 0.5
    assert len(task.subsystems) == 1
    assert task.subsystems[0].solver == "h2iso"


# ==============================================================================
# Metric & TargetMetric Configs
# ==============================================================================


def test_metric_config_operators_and_bisection():
    metric = MetricConfig(
        source="sds.I[1]",
        evaluate="bisection_search",
        unit="day",
        parameter_to_optimize="TBR",
        search_bounds=[1.0, 1.4],
        tolerance=0.001,
        max_iterations=20,
        target_metric=TargetMetricConfig(
            source="sds.I[1]",
            evaluate="turning_point",
            threshold=1000.0,
            direction="min_to_max",
        ),
        factor=2.0,
    )
    assert metric.evaluate == "bisection_search"
    assert metric.parameter_to_optimize == "TBR"
    assert metric.search_bounds == [1.0, 1.4]
    assert metric.target_metric.direction == "min_to_max"


# ==============================================================================
# Visualizer & Plugins Pillars
# ==============================================================================


def test_visualizer_and_report_config():
    vis = VisualizerConfig(
        enabled=True,
        port=8088,
        host="0.0.0.0",
        launch_on_finish=False,
        report=ReportConfig(
            enabled=True,
            title="Fusion System Report",
            formats=["markdown", "pdf"],
            ai_insight=AIInsightConfig(
                enabled=True,
                focus_areas=["tritium_inventory", "safety_margin"],
                provider="gpt-4o",
            ),
        ),
    )
    assert vis.port == 8088
    assert vis.report.ai_insight.enabled is True
    assert "tritium_inventory" in vis.report.ai_insight.focus_areas


def test_plugins_config():
    plugins = PluginsConfig(
        preprocess=[
            PluginSpec(name="foc", plugin="foc_injector", params={"scale": 1.0}),
            PluginSpec(
                name="custom_prep",
                script="./scripts/my_prep.py",
                class_name="MyPrepPlugin",
                params={"foo": "bar"},
            ),
        ],
        simulation_engine=EngineSpec(type="standard", params={"fast": True}),
        postprocess=[
            PluginSpec(name="safety", plugin="mass_conservation"),
        ],
    )
    assert len(plugins.preprocess) == 2
    assert plugins.preprocess[1].class_name == "MyPrepPlugin"
    assert plugins.simulation_engine.type == "standard"
    assert plugins.postprocess[0].plugin == "mass_conservation"


# ==============================================================================
# Root TricysV2Config & is_v2_config Detection
# ==============================================================================


def test_tricys_v2_config_full(minimal_v2_simulate_dict):
    v2_data = minimal_v2_simulate_dict.copy()
    v2_data["visualizer"] = {"launch_on_finish": False}
    v2_data["plugins"] = {
        "postprocess": [{"name": "auditor", "plugin": "mass_conservation"}]
    }

    config = TricysV2Config.from_dict(v2_data)
    assert config.version == "2.0"
    assert config.model.modelName == "example_model.Cycle"
    assert config.scenario.stopTime == 1000.0
    assert config.task.type == "simulate"
    assert config.visualizer.launch_on_finish is False

    # Test synthetic backwards compatibility properties
    assert config.paths.package_path == "models/Cycle.mo"
    assert config.simulation.model_name == "example_model.Cycle"
    assert config.simulation.stop_time == 1000.0
    assert config.simulation.step_size == 1.0

    # Serialization
    dumped = config.to_dict()
    assert dumped["version"] == "2.0"
    assert dumped["model"]["modelName"] == "example_model.Cycle"


def test_tricys_v2_config_discriminator_rejection(minimal_v2_simulate_dict):
    # An invalid task type like "parse" or "unsupported_task" must be rejected
    v2_data = minimal_v2_simulate_dict.copy()
    v2_data["task"] = {"type": "non_existent_task", "parameters": {}}

    with pytest.raises(ValidationError):
        TricysV2Config.from_dict(v2_data)


def test_is_v2_config_detection(minimal_v2_simulate_dict):
    assert is_v2_config(minimal_v2_simulate_dict) is True

    # Structural detection without explicit version field
    struct_dict = {
        "model": minimal_v2_simulate_dict["model"],
        "scenario": minimal_v2_simulate_dict["scenario"],
        "task": minimal_v2_simulate_dict["task"],
    }
    assert is_v2_config(struct_dict) is True

    # Missing model
    no_model = struct_dict.copy()
    del no_model["model"]
    assert is_v2_config(no_model) is False

    # Missing scenario
    no_scenario = struct_dict.copy()
    del no_scenario["scenario"]
    assert is_v2_config(no_scenario) is False

    # Non-dictionary
    assert is_v2_config("invalid_string") is False
    assert is_v2_config([]) is False
    assert is_v2_config(None) is False

    # Legacy 1.0 dict
    legacy_1 = {
        "simulation": {"model_name": "MyModel", "stop_time": 100},
        "paths": {"package_path": "MyModel.mo"},
    }
    assert is_v2_config(legacy_1) is False


def test_tricys_config_from_dict_polymorphic(minimal_v2_simulate_dict):
    # TricysConfig.from_dict should automatically return TricysV2Config when v2 data is provided
    result = TricysConfig.from_dict(minimal_v2_simulate_dict)
    assert isinstance(result, TricysV2Config)
    assert result.model.modelName == "example_model.Cycle"

    # When legacy 1.0 data is provided, it should return TricysConfig
    legacy_data = {
        "simulation": {"model_name": "OldModel", "stop_time": 50.0},
        "paths": {"package_path": "OldModel.mo"},
    }
    legacy_result = TricysConfig.from_dict(legacy_data)
    assert isinstance(legacy_result, TricysConfig)
    assert legacy_result.simulation.model_name == "OldModel"


def test_v2_canonical_task_auditor_and_visualizer_report(minimal_v2_simulate_dict):
    """Verify that task.auditor and visualizer.report parse cleanly and synchronize with top-level accessors."""
    v2_data = minimal_v2_simulate_dict.copy()
    v2_data["task"] = {
        "type": "simulate",
        "parameters": {"WDS.threshold": 1200.0},
        "auditor": {
            "enabled": True,
            "warn_threshold_g": 0.5,
            "kill_threshold_g": 20.0,
            "patterns": {
                "inventory": ["*.I?1?"],
                "cumulative_sources": ["*blanket.cumulative_breed?1?"],
            },
        },
    }
    v2_data["visualizer"] = {
        "enabled": True,
        "report": {
            "enabled": True,
            "title": "Custom Mass Balance Report",
            "formats": ["markdown", "pdf"],
        },
    }

    config = TricysV2Config.from_dict(v2_data)
    # Check canonical locations
    assert config.task.auditor is not None
    assert config.task.auditor.enabled is True
    assert config.task.auditor.warn_threshold_g == 0.5
    assert config.visualizer.report is not None
    assert config.visualizer.report.title == "Custom Mass Balance Report"

    # Check synchronized accessors
    assert config.auditor is not None
    assert config.auditor.warn_threshold_g == 0.5
    assert config.report is not None
    assert config.report.title == "Custom Mass Balance Report"
