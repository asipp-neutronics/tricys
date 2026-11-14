from __future__ import annotations

from tricys.preprocessing.setup import SimulationSetupPlugin
from tricys.workflow.context import PipelineContext


def test_simulation_setup_plugin_with_sobol_sampling(tmp_path):
    config = {
        "paths": {
            "temp_dir": str(tmp_path / "temp"),
            "results_dir": str(tmp_path / "results"),
        },
        "simulation": {
            "model_name": "TestModel",
            "stop_time": 100.0,
            "step_size": 1.0,
        },
        "preprocess": {
            "sampling": {
                "mode": "sobol",
                "sobol": {
                    "parameters": {
                        "alpha": {"bounds": [0.0, 1.0]},
                        "beta": {"bounds": [10.0, 20.0]},
                    },
                    "sample_n": 4,
                    "seed": 1,
                },
            }
        },
    }

    context = PipelineContext.from_config(config)
    plugin = SimulationSetupPlugin()
    plugin.run(context)

    # 4 * (2 + 2) = 16 jobs
    assert len(context.jobs) == 16
    assert context.get_metadata("sampling_method") == "sobol"
    assert context.get_metadata("total_jobs") == 16
    assert "sampling_problem" in context.metadata
    problem = context.get_metadata("sampling_problem")
    assert problem["names"] == ["alpha", "beta"]

    for job in context.jobs:
        assert "alpha" in job
        assert "beta" in job
        assert 0.0 <= job["alpha"] <= 1.0
        assert 10.0 <= job["beta"] <= 20.0


def test_simulation_setup_plugin_with_legacy_grid_cases(tmp_path):
    config = {
        "paths": {
            "temp_dir": str(tmp_path / "temp"),
            "results_dir": str(tmp_path / "results"),
        },
        "simulation": {
            "model_name": "TestModel",
        },
        "sensitivity_analysis": {
            "analysis_cases": [
                {
                    "independent_variable": "var1",
                    "independent_variable_sampling": [1, 2, 3],
                    "simulation_parameters": {
                        "var2": [10, 20],
                    },
                }
            ]
        },
    }

    context = PipelineContext.from_config(config)
    plugin = SimulationSetupPlugin()
    plugin.run(context)

    # 3 * 2 = 6 jobs
    assert len(context.jobs) == 6
    assert context.get_metadata("sampling_method") == "grid"
    assert context.get_metadata("total_jobs") == 6
