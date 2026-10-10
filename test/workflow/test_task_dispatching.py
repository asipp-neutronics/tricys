from tricys.core.schema import TricysV2Config
from tricys.plugins.sensitivity_plugin import SensitivityAnalysisPlugin
from tricys.preprocessing.compiler import ModelCompilerPlugin
from tricys.utils.concurrency_utils import get_safe_max_workers
from tricys.utils.config_utils import basic_prepare_config
from tricys.workflow.builder import PipelineBuilder
from tricys.workflow.context import PipelineContext


def test_schema_v2_field_effectiveness_mapping():
    v2_raw = {
        "version": "2.0",
        "model": {
            "modelName": "CFEDR.TritiumCycle",
            "fileName": "package.mo",
            "cflags": "-O3 -march=native",
            "loadModel": ["Modelica", "Complex"],
            "target": "openmodelica",
        },
        "scenario": {
            "stopTime": 3600.0,
            "startTime": 100.0,
            "numberOfIntervals": 1800,
            "tolerance": 1e-7,
            "method": "ida",
            "outputFormat": "csv",
            "variableFilter": "sds.*|blanket.*",
            "simflags": "-lv=LOG_STATS",
            "pulse": {
                "foc_file": "pulse_profile.csv",
                "foc_component": "pulseSource",
                "variable_mapping": {"pulseSource.width": "width"},
            },
        },
        "task": {
            "type": "sweep",
            "parallel": True,
            "max_workers": 4,
            "parameters": [
                {"blanket.TBR": 1.1},
                {"blanket.TBR": 1.2},
            ],
            "metrics": {
                "peak_inv": {"source": "sds.I[1]", "evaluate": "peak_value"},
                "steady_inv": {"source": "sds.I[1]", "evaluate": "final_value"},
            },
        },
    }

    # Validate against Schema 2.0 Pydantic model
    validated = TricysV2Config.model_validate(v2_raw)
    assert validated.version == "2.0"

    # Test basic_prepare_config
    prepared_config, _ = basic_prepare_config(v2_raw)

    sim = prepared_config["simulation"]
    assert sim["model_name"] == "CFEDR.TritiumCycle"
    assert sim["cflags"] == "-O3 -march=native"
    assert sim["loadModel"] == ["Modelica", "Complex"]
    assert sim["target"] == "openmodelica"
    assert sim["stop_time"] == 3600.0
    assert sim["start_time"] == 100.0
    assert sim["step_size"] == 2.0  # 3600.0 / 1800 intervals
    assert sim["tolerance"] == 1e-7
    assert sim["method"] == "ida"
    assert sim["outputFormat"] == "csv"
    assert sim["variableFilter"] == "sds.*|blanket.*"
    assert sim["simflags"] == "-lv=LOG_STATS"
    assert sim["concurrent"] is True
    assert sim["max_workers"] == 4

    # Pulse mapping
    assert "foc" in prepared_config
    assert prepared_config["foc"]["component_name"] == "pulseSource"

    # Metrics mapping
    assert "metrics_definition" in prepared_config
    assert "peak_inv" in prepared_config["metrics_definition"]

    # PipelineContext construction
    ctx = PipelineContext.from_config(prepared_config)
    assert ctx.model_name == "CFEDR.TritiumCycle"
    assert ctx.stop_time == 3600.0
    assert ctx.step_size == 2.0
    assert "peak_inv" in ctx.metrics_definition
    assert ctx.typed_config is not None


def test_schema_v2_task_dispatching_simulate():
    simulate_cfg = {
        "version": "2.0",
        "model": {"modelName": "DemoModel", "fileName": "package.mo"},
        "scenario": {"stopTime": 1000.0, "stepSize": 1.0},
        "task": {
            "type": "simulate",
            "parameters": {"blanket.TBR": 1.15},
            "metrics": {
                "max_t": {"source": "temp", "evaluate": "max"},
            },
        },
    }
    prep, _ = basic_prepare_config(simulate_cfg)
    pipeline = PipelineBuilder.build(prep)
    ctx = PipelineContext.from_config(prep)

    # Preprocessors should include setup
    pre_names = [p.name for p in pipeline.preprocessors]
    assert "setup" in pre_names

    # Run setup to expand jobs
    for p in pipeline.preprocessors:
        if p.name == "setup":
            p.run(ctx)
            break

    assert len(ctx.jobs) == 1
    assert ctx.jobs[0]["blanket.TBR"] == 1.15
    assert ctx.get_metadata("sampling_method") == "simulate"


def test_schema_v2_task_dispatching_sensitivity():
    sensitivity_cfg = {
        "version": "2.0",
        "model": {"modelName": "DemoModel", "fileName": "package.mo"},
        "scenario": {"stopTime": 1000.0, "stepSize": 1.0},
        "task": {
            "type": "sensitivity",
            "method": "latin",
            "samples": 16,
            "parallel": True,
            "max_workers": 2,
            "parameters": {
                "blanket.TBR": [1.05, 1.25],
                "plasma.fraction": [0.01, 0.05],
            },
            "metrics": {
                "final_i": {"source": "inventory", "evaluate": "final_value"},
            },
        },
    }
    prep, _ = basic_prepare_config(sensitivity_cfg)
    pipeline = PipelineBuilder.build(prep)
    ctx = PipelineContext.from_config(prep)

    # Should infer compiler because task.parallel=True
    pre_classes = [p.__class__ for p in pipeline.preprocessors]
    assert ModelCompilerPlugin in pre_classes

    # Postprocessors should include SensitivityAnalysisPlugin
    post_classes = [p.__class__ for p in pipeline.postprocessors]
    assert SensitivityAnalysisPlugin in post_classes

    # Check setup job expansion for sampling
    for p in pipeline.preprocessors:
        if p.name == "setup":
            p.run(ctx)
            break

    assert len(ctx.jobs) == 16
    assert ctx.get_metadata("sampling_method") == "latin"


def test_deadlock_free_concurrency_bounds():
    # Outer level safe worker bound
    workers = get_safe_max_workers(user_requested_limit=32, task_count=4)
    assert workers <= 4

    # Nested worker guard: when inside a job worker, max_workers must be 1
    inner_workers = get_safe_max_workers(user_requested_limit=1, task_count=10)
    assert inner_workers == 1
