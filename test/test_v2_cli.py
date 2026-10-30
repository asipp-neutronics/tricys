import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tricys.main import main as cli_main
from tricys.utils.config_utils import basic_prepare_config

TEST_DIR = Path("temp_v2_cli_test").resolve()


@pytest.fixture(autouse=True)
def setup_teardown():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)


def test_basic_prepare_config_v2_yaml():
    """Verify that basic_prepare_config loads and validates TRICYS 2.0 YAML configuration."""
    model_file = TEST_DIR / "MyModel.mo"
    model_file.touch()

    config_yaml_path = TEST_DIR / "config_v2.yaml"
    v2_data = {
        "model": {
            "fileName": str(model_file),
            "modelName": "MyPackage.MyModel",
        },
        "scenario": {
            "stopTime": 500.0,
            "stepSize": 0.5,
            "method": "dassl",
            "variableFilter": "time|val.*",
        },
        "task": {
            "type": "sweep",
            "parameters": [
                {"name": "CaseA", "TBR": 1.15},
                {"name": "CaseB", "TBR": 1.25},
            ],
            "parallel": True,
            "max_workers": 2,
        },
        "visualizer": {
            "launch_on_finish": False,
        },
    }

    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(v2_data, f)

    runtime_cfg, orig_cfg = basic_prepare_config(str(config_yaml_path))

    assert runtime_cfg["model"]["modelName"] == "MyPackage.MyModel"
    assert runtime_cfg["scenario"]["stopTime"] == 500.0
    assert runtime_cfg["task"]["type"] == "sweep"
    assert len(runtime_cfg["task"]["parameters"]) == 2
    # Check backwards-compatible synthesized paths
    assert "paths" in runtime_cfg
    assert runtime_cfg["paths"]["package_path"] == str(model_file)
    assert os.path.isdir(runtime_cfg["paths"]["results_dir"])


def test_cli_run_v2_yaml_dispatch():
    """Verify that 'tricys run -c config.yaml' dispatches to simulation_main with TRICYS 2.0 config."""
    model_file = TEST_DIR / "TestModel.mo"
    model_file.touch()

    config_yaml_path = TEST_DIR / "config_run.yaml"
    v2_data = {
        "model": {
            "fileName": str(model_file),
            "modelName": "TestModel",
        },
        "scenario": {
            "stopTime": 100.0,
            "stepSize": 1.0,
        },
        "task": {
            "type": "simulate",
        },
    }

    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(v2_data, f)

    test_argv = ["tricys", "run", "-c", str(config_yaml_path), "--csv"]

    with (
        patch("sys.argv", test_argv),
        patch("tricys.main.simulation_main") as mock_sim_main,
    ):
        cli_main()
        assert mock_sim_main.called
        called_config = mock_sim_main.call_args[0][0]
        assert called_config["model"]["modelName"] == "TestModel"
        assert called_config["task"]["type"] == "simulate"
        assert mock_sim_main.call_args[1]["export_csv"] is True


def test_cli_flags_enhanced_and_turbo_v2():
    """Verify that --enhanced and --turbo correctly update TRICYS 2.0 task config."""
    model_file = TEST_DIR / "TestModel.mo"
    model_file.touch()

    config_yaml_path = TEST_DIR / "config_turbo.yaml"
    v2_data = {
        "model": {
            "fileName": str(model_file),
            "modelName": "TestModel",
        },
        "scenario": {
            "stopTime": 100.0,
            "stepSize": 1.0,
        },
        "task": {
            "type": "sweep",
            "parameters": [{"TBR": 1.1}, {"TBR": 1.2}],
            "parallel": False,
        },
    }

    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(v2_data, f)

    test_argv = ["tricys", "run", "-c", str(config_yaml_path), "--enhanced", "--turbo"]

    with (
        patch("sys.argv", test_argv),
        patch("tricys.main.simulation_main") as mock_sim_main,
    ):
        cli_main()
        assert mock_sim_main.called
        called_config = mock_sim_main.call_args[0][0]
        assert called_config["task"]["parallel"] is True
        assert called_config["task"]["max_workers"] >= 1


def test_cli_visualizer_launch_on_finish():
    """Verify that visualizer.launch_on_finish triggers visualizer_main."""
    model_file = TEST_DIR / "TestModel.mo"
    model_file.touch()

    config_yaml_path = TEST_DIR / "config_vis.yaml"
    v2_data = {
        "model": {
            "fileName": str(model_file),
            "modelName": "TestModel",
        },
        "scenario": {
            "stopTime": 10.0,
            "stepSize": 1.0,
        },
        "task": {
            "type": "simulate",
        },
        "visualizer": {
            "port": 8888,
            "launch_on_finish": True,
        },
    }

    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(v2_data, f)

    test_argv = ["tricys", "-c", str(config_yaml_path)]

    with (
        patch("sys.argv", test_argv),
        patch("tricys.main.simulation_main"),
        patch("tricys.visualizer.main.start") as mock_vis_start,
    ):
        cli_main()
        assert mock_vis_start.called


def test_v2_example_04_pulse_width_sweep_config():
    """Verify that 04_sweep_pulse_width_required_tbr.yaml parses cleanly as valid V2 config."""
    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "04_sweep_pulse_width_required_tbr.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "sweep"
    assert len(runtime_cfg["task"]["parameters"]) == 3
    widths = [p["pulseSource.width"] for p in runtime_cfg["task"]["parameters"]]
    assert widths == [100.0, 80.0, 60.0]

    # Verify metrics
    metrics = runtime_cfg["task"]["metrics"]
    assert "startup_inventory" in metrics
    assert "Required_TBR" in metrics
    assert metrics["startup_inventory"]["evaluate"] == "startup_inventory"
    assert metrics["Required_TBR"]["evaluate"] == "bisection_search"
    assert metrics["Required_TBR"]["parameter"] == "blanket.TBR"


def test_v2_example_05_foc_pulsed_scenario_config():
    """Verify that 05_simulate_foc_pulsed_scenario.yaml parses cleanly as valid V2 config with FOC injection."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "05_simulate_foc_pulsed_scenario.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "simulate"
    assert runtime_cfg["simulation"]["stop_time"] == 1200.0 * 3600.0
    assert runtime_cfg["simulation"]["step_size"] == 0.5 * 3600.0

    # Verify FOC configuration
    assert "foc" in runtime_cfg
    assert runtime_cfg["foc"]["foc_component"] == "pulseSource"
    assert runtime_cfg["foc"]["strategy"] == "table"
    assert os.path.exists(runtime_cfg["foc"]["foc_path"])

    # Verify pipeline auto-inference
    pipeline = PipelineBuilder.build(runtime_cfg)
    assert any(p.name == "foc" for p in pipeline.preprocessors)


def test_v2_example_06_mass_conservation_audit_config():
    """Verify that 06_simulate_mass_conservation_audit.yaml parses cleanly as valid V2 config with mass auditor."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "06_simulate_mass_conservation_audit.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "simulate"
    assert runtime_cfg["simulation"]["stop_time"] == 2000.0 * 3600.0
    assert runtime_cfg["simulation"]["step_size"] == 0.5 * 3600.0

    # Verify auditor configuration under task (and propagated runtime alias)
    assert "auditor" in runtime_cfg["task"]
    auditor_cfg = runtime_cfg["task"]["auditor"]
    assert auditor_cfg["enabled"] is True
    assert auditor_cfg["warn_threshold_g"] == 0.5
    assert auditor_cfg["kill_threshold_g"] == 50.0
    patterns = auditor_cfg["patterns"]
    assert "*.I?1?" in patterns["inventory"]
    assert "*blanket.cumulative_breed?1?" in patterns["cumulative_sources"]
    assert "*.leak_rate?1?" in patterns["leak"]
    assert "*.decay_rate?1?" in patterns["decay"]
    assert "*plasma.cumulative_burn?1?" in patterns["cumulative_burn"]

    # Verify report configuration under visualizer
    assert "visualizer" in runtime_cfg
    assert "report" in runtime_cfg["visualizer"]
    assert runtime_cfg["visualizer"]["report"]["enabled"] is True

    # Verify pipeline auto-inference attaches safety auditor and report
    pipeline = PipelineBuilder.build(runtime_cfg)
    assert any(p.name in ["safety", "auditor"] for p in pipeline.postprocessors)
    assert any(p.name == "report" for p in pipeline.postprocessors)


def test_v2_example_08_cosim_offline_table_config():
    """Verify that 08_cosim_offline_table.yaml parses cleanly and configures offline table cosim (migrated from basic 4)."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "08_cosim_offline_table.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "cosim"
    assert runtime_cfg["task"]["mode"] == "offline_table"
    assert len(runtime_cfg["task"]["subsystems"]) == 1

    subsystem = runtime_cfg["task"]["subsystems"][0]
    assert subsystem["target"] == "example_model.I_ISS"
    assert subsystem["instance"] == "i_iss"
    assert subsystem["solver"] == "dwsim"
    assert "dwsim_iss_handler.py" in subsystem["handler_script_path"]
    assert subsystem["handler_function"] == "run_dwsim_simulation"

    # Verify co_simulation runtime synthesis
    assert "co_simulation" in runtime_cfg
    assert runtime_cfg["co_simulation"]["engine"] in ("offline_table", "offline_cosim")
    assert runtime_cfg["co_simulation"]["mode"] == "offline_table"
    assert len(runtime_cfg["co_simulation"]["handlers"]) == 1
    handler = runtime_cfg["co_simulation"]["handlers"][0]
    assert handler["solver"] == "dwsim"
    assert "dwsim_iss_handler.py" in handler["handler_script_path"]
    assert handler["handler_function"] == "run_dwsim_simulation"
    assert handler["params"]["base"] == 20.0

    # Verify pipeline auto-inference
    pipeline = PipelineBuilder.build(runtime_cfg)
    assert pipeline.engine.name in ("offline_table", "offline_cosim")
    pre_names = [p.name for p in pipeline.preprocessors]
    assert "setup" in pre_names
    assert "interceptor" in pre_names

    # Verify 1:1 compilation to explicit pipeline
    compiled = PipelineBuilder.compile_v2_to_explicit_pipeline(runtime_cfg)
    assert compiled["pipeline"]["simulation"]["name"] in (
        "offline_table",
        "offline_cosim",
    )
    assert any(
        p["name"] == "interceptor" for p in compiled["pipeline"]["preprocessing"]
    )


def test_v2_example_09_cosim_online_step_config():
    """Verify that 09_cosim_online_step.yaml parses cleanly and configures online step cosim (migrated from basic 9)."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "09_cosim_online_step.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "cosim"
    assert runtime_cfg["task"]["mode"] == "online_step"
    assert len(runtime_cfg["task"]["subsystems"]) == 1

    subsystem = runtime_cfg["task"]["subsystems"][0]
    assert subsystem["target"] == "example_model.DIV"
    assert subsystem["instance"] == "div"
    assert "festim_div_processor" in subsystem["handler_script_path"]
    assert subsystem["handler_function"] == "FESTIMDivertorProcessor"

    # Verify co_simulation runtime synthesis
    assert "co_simulation" in runtime_cfg
    assert runtime_cfg["co_simulation"]["engine"] in ("online_step", "online_oms")
    assert runtime_cfg["co_simulation"]["mode"] == "online_step"
    handler = runtime_cfg["co_simulation"]["handlers"][0]
    assert handler["mode"] == "online"

    # Verify auditor synthesized from plugins
    assert "auditor" in runtime_cfg
    assert runtime_cfg["auditor"]["enabled"] is True

    # Verify pipeline auto-inference
    pipeline = PipelineBuilder.build(runtime_cfg)
    assert pipeline.engine.name in ("online_step", "online_oms")

    # Verify 1:1 compilation to explicit pipeline
    compiled = PipelineBuilder.compile_v2_to_explicit_pipeline(runtime_cfg)
    assert compiled["pipeline"]["simulation"]["name"] in ("online_step", "online_oms")


def test_v2_example_10_cosim_parameter_injection_config():
    """Verify that 10_cosim_parameter_injection.yaml parses cleanly and configures parameter injection."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "10_cosim_parameter_injection.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "cosim"
    assert runtime_cfg["task"]["mode"] == "parameter_injection"
    assert len(runtime_cfg["task"]["subsystems"]) == 2
    sub0 = runtime_cfg["task"]["subsystems"][0]
    assert sub0["substitute_model"] == "Generic_ISS_Adapters.ISS_I_Adapter"
    assert "aux_files" in sub0
    assert "override_file" in runtime_cfg["co_simulation"]
    assert "override_file" in runtime_cfg["simulation"]

    # Verify pipeline auto-inference
    pipeline = PipelineBuilder.build(runtime_cfg)
    assert pipeline.engine.name == "param_inject"

    # Verify 1:1 compilation to explicit pipeline
    compiled = PipelineBuilder.compile_v2_to_explicit_pipeline(runtime_cfg)
    assert compiled["pipeline"]["simulation"]["name"] == "param_inject"
    assert "override_file" in compiled["pipeline"]["simulation"]["params"]


def test_v2_example_10_standalone_directory_config():
    """Verify that 10_cosim_parameter_injection/10_cosim_parameter_injection.yaml in standalone directory parses cleanly."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "10_cosim_parameter_injection"
        / "10_cosim_parameter_injection.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "cosim"
    assert runtime_cfg["task"]["mode"] == "parameter_injection"
    assert runtime_cfg["task"]["injection_type"] == "handler"
    assert len(runtime_cfg["task"]["subsystems"]) == 2

    # Verify pipeline auto-inference
    pipeline = PipelineBuilder.build(runtime_cfg)
    assert pipeline.engine.name == "param_inject"

    compiled = PipelineBuilder.compile_v2_to_explicit_pipeline(runtime_cfg)
    assert compiled["pipeline"]["simulation"]["name"] == "param_inject"


def test_v2_example_01_sweep_startup_self_sufficiency_config():
    """Verify that 01_sweep_startup_self_sufficiency.yaml parses cleanly as valid V2 config."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "01_sweep_startup_self_sufficiency.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "sweep"
    assert len(runtime_cfg["task"]["parameters"]) == 3
    assert runtime_cfg["task"]["parameters"][0]["blanket.TBR"] == 1.10
    assert "startup_inventory" in runtime_cfg["task"]["metrics"]

    pipeline = PipelineBuilder.build(runtime_cfg)
    assert pipeline.engine.name == "standard"


def test_v2_example_02_simulate_pulsed_transient_config():
    """Verify that 02_simulate_pulsed_transient.yaml parses cleanly as valid V2 config."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "02_simulate_pulsed_transient.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "simulate"
    assert "Required_TBR" in runtime_cfg["task"]["metrics"]
    assert (
        runtime_cfg["task"]["metrics"]["Required_TBR"]["evaluate"] == "bisection_search"
    )

    pipeline = PipelineBuilder.build(runtime_cfg)
    assert pipeline.engine.name == "standard"


def test_v2_example_03_sensitivity_sobol_gsa_config():
    """Verify that 03_sensitivity_sobol_gsa.yaml parses cleanly as valid V2 config."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "03_sensitivity_sobol_gsa.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "sensitivity"
    assert runtime_cfg["task"]["method"] == "sobol"
    assert runtime_cfg["task"]["samples"] == 8

    pipeline = PipelineBuilder.build(runtime_cfg)
    assert pipeline.engine.name == "standard"


def test_v2_example_07_custom_plugin_pipeline_config():
    """Verify that 07_custom_plugin_pipeline.yaml parses cleanly and mounts dynamic plugin."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "07_custom_plugin_pipeline.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "sweep"
    assert "plugins" in runtime_cfg
    assert len(runtime_cfg["plugins"]["postprocess"]) == 1

    pipeline = PipelineBuilder.build(runtime_cfg)
    assert pipeline.engine.name == "standard"
    post_names = [p.name for p in pipeline.postprocessors]
    assert "tritium_reserve_auditor" in post_names


def test_v2_example_11_optimize_nested_bisection_config():
    """Verify that 11_optimize_nested_bisection.yaml parses cleanly as an optimize task with nested_bisection."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "11_optimize_nested_bisection.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "optimize"
    assert runtime_cfg["task"]["algorithm"] == "nested_bisection"
    task_opt = runtime_cfg["task"].get("optimization", runtime_cfg["task"])
    assert "outer" in task_opt
    assert "inner" in task_opt

    pipeline = PipelineBuilder.build(runtime_cfg)
    assert pipeline.engine.name == "standard"


def test_v2_example_12_sweep_nested_optimization_config():
    """Verify that 12_sweep_nested_optimization.yaml parses cleanly as sweep task with embedded nested bisection."""
    from tricys.workflow.builder import PipelineBuilder

    yaml_path = (
        Path(__file__).parent.parent
        / "tricys"
        / "example"
        / "example_data"
        / "v2_examples"
        / "12_sweep_nested_optimization.yaml"
    )
    assert yaml_path.exists(), f"File {yaml_path} should exist"

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_path))
    assert runtime_cfg["task"]["type"] == "sweep"
    assert "optimization" in runtime_cfg["task"]
    assert runtime_cfg["task"]["optimization"]["algorithm"] == "nested_bisection"
    assert "outer" in runtime_cfg["task"]["optimization"]
    assert "inner" in runtime_cfg["task"]["optimization"]

    pipeline = PipelineBuilder.build(runtime_cfg)
    assert pipeline.engine.name == "standard"


V2_EXAMPLES_DIR = (
    Path(__file__).parent.parent / "tricys" / "example" / "example_data" / "v2_examples"
)
ALL_V2_YAMLS = sorted([p for p in V2_EXAMPLES_DIR.glob("**/*.yaml") if p.is_file()])


@pytest.mark.parametrize("yaml_file", ALL_V2_YAMLS, ids=lambda p: p.name)
def test_all_v2_examples_load_and_compile(yaml_file):
    """Dynamically discover and rigorously test all YAML configuration files in v2_examples."""
    from tricys.workflow.builder import PipelineBuilder

    runtime_cfg, orig_cfg = basic_prepare_config(str(yaml_file))
    assert "model" in runtime_cfg
    assert "task" in runtime_cfg

    # Ensure pipeline builder can auto-infer or build workflow pipeline
    pipeline = PipelineBuilder.build(runtime_cfg)
    assert pipeline.engine is not None
    assert pipeline.engine.name in (
        "standard",
        "offline_table",
        "offline_cosim",
        "online_step",
        "online_oms",
        "param_inject",
    )

    # Ensure explicit pipeline compilation succeeds without error
    compiled = PipelineBuilder.compile_v2_to_explicit_pipeline(runtime_cfg)
    assert "pipeline" in compiled
    assert "simulation" in compiled["pipeline"]
