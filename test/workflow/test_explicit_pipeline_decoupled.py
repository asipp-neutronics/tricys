from __future__ import annotations

from tricys.plugins.storage_plugin import SimulationStoragePlugin
from tricys.preprocessing.setup import SimulationSetupPlugin
from tricys.simulation.engines.standard import StandardSimulationEngine
from tricys.workflow.builder import PipelineBuilder


def test_pure_explicit_pipeline_builder() -> None:
    """Tests that a pure explicit pipeline without top-level model/scenario/task blocks
    correctly instantiates all plugins and engines with their explicit parameters.
    """
    pure_pipeline_config = {
        "pipeline": {
            "name": "PureDecoupledPipeline",
            "preprocessing": [
                {
                    "name": "setup",
                    "params": {
                        "package_path": "skills/tricys-agent/CFEDR_2870_3lc_ssp.mo",
                        "model_name": "CFEDR_3LC.Cycle_3LC",
                        "task_type": "simulate",
                        "parameters": {"Cycle_3LC.sds.V": 1.5},
                    },
                }
            ],
            "simulation": {
                "name": "standard",
                "params": {
                    "stop_time": 5000.0,
                    "step_size": 5.0,
                    "solver": "dassl",
                    "tolerance": 1e-5,
                    "parallel": False,
                },
            },
            "postprocessing": [
                {
                    "name": "storage",
                    "params": {
                        "metrics_definition": {
                            "Min_Storage": {
                                "source": "sds.inventory",
                                "evaluate": "min",
                            }
                        },
                        "export_csv": True,
                    },
                }
            ],
        }
    }

    pipeline = PipelineBuilder.build(pure_pipeline_config)
    assert pipeline.name == "PureDecoupledPipeline"
    assert len(pipeline.preprocessors) == 1
    assert len(pipeline.postprocessors) == 1

    # Verify Preprocessor
    setup_p = pipeline.preprocessors[0]
    assert isinstance(setup_p, SimulationSetupPlugin)
    assert setup_p.package_path == "skills/tricys-agent/CFEDR_2870_3lc_ssp.mo"
    assert setup_p.model_name == "CFEDR_3LC.Cycle_3LC"
    assert setup_p.parameters == {"Cycle_3LC.sds.V": 1.5}
    assert setup_p.task_type == "simulate"

    # Verify Engine
    engine = pipeline.engine
    assert isinstance(engine, StandardSimulationEngine)
    assert engine.stop_time == 5000.0
    assert engine.step_size == 5.0
    assert engine.solver == "dassl"
    assert engine.tolerance == 1e-5
    assert engine.parallel is False

    # Verify Postprocessor
    storage_p = pipeline.postprocessors[0]
    assert isinstance(storage_p, SimulationStoragePlugin)
    assert storage_p.export_csv is True
    assert storage_p.metrics_definition == {
        "Min_Storage": {"source": "sds.inventory", "evaluate": "min"}
    }


def test_compile_v2_to_explicit_pipeline_lossless(tmp_path) -> None:
    """Tests lossless compilation from Schema 2.0 5 Core Pillars into pure explicit pipeline."""
    mock_plugin_file = tmp_path / "tritium_self_sufficiency_auditor.py"
    mock_plugin_file.write_text(
        "from tricys.workflow.base import PostprocessPlugin\n"
        "from tricys.workflow.context import PipelineContext\n\n"
        "class TritiumSelfSufficiencyAuditorPlugin(PostprocessPlugin):\n"
        "    name = 'tritium_self_sufficiency_auditor'\n"
        "    def __init__(self, alert_valley_inventory_g: float = 500.0):\n"
        "        super().__init__()\n"
        "        self.alert_valley_inventory_g = alert_valley_inventory_g\n"
        "    def run(self, context: PipelineContext) -> None:\n"
        "        pass\n",
        encoding="utf-8",
    )

    v2_config = {
        "version": "2.0",
        "model": {
            "fileName": "skills/tricys-agent/CFEDR_2870_3lc_ssp.mo",
            "modelName": "CFEDR_3LC.Cycle_3LC",
            "cflags": "-O3",
            "target": "gcc",
        },
        "scenario": {
            "startTime": 0.0,
            "stopTime": 12000.0,
            "stepSize": 10.0,
            "method": "dassl",
            "tolerance": 1e-6,
            "outputFormat": "csv",
        },
        "task": {
            "type": "sweep",
            "parallel": True,
            "max_workers": 4,
            "parameters": [
                {"Cycle_3LC.sds.V": 1.2},
                {"Cycle_3LC.sds.V": 1.5},
            ],
            "metrics": {"Storage_Min": {"source": "sds.inventory", "evaluate": "min"}},
        },
        "visualizer": {
            "report": {
                "title": "CFEDR Verification Report",
                "formats": ["markdown", "pdf"],
            }
        },
        "plugins": {
            "postprocess": [
                {
                    "script": str(mock_plugin_file),
                    "class": "TritiumSelfSufficiencyAuditorPlugin",
                    "params": {"alert_valley_inventory_g": 600.0},
                }
            ]
        },
    }

    compiled = PipelineBuilder.compile_v2_to_explicit_pipeline(
        v2_config, pipeline_name="CompiledCFEDRPipeline"
    )

    assert "pipeline" in compiled
    p_dict = compiled["pipeline"]
    assert p_dict["name"] == "CompiledCFEDRPipeline"

    # Preprocessing checks: setup + compiler (due to parallel: True)
    pre = p_dict["preprocessing"]
    assert len(pre) == 2
    assert pre[0]["name"] == "setup"
    assert (
        pre[0]["params"]["package_path"] == "skills/tricys-agent/CFEDR_2870_3lc_ssp.mo"
    )
    assert pre[0]["params"]["task_type"] == "sweep"
    assert len(pre[0]["params"]["parameters"]) == 2

    assert pre[1]["name"] == "compiler"
    assert pre[1]["params"]["cflags"] == "-O3"
    assert pre[1]["params"]["target"] == "gcc"

    # Simulation check
    sim = p_dict["simulation"]
    assert sim["name"] == "standard"
    assert sim["params"]["stop_time"] == 12000.0
    assert sim["params"]["step_size"] == 10.0
    assert sim["params"]["solver"] == "dassl"
    assert sim["params"]["parallel"] is True
    assert sim["params"]["max_workers"] == 4

    # Postprocessing checks: storage + custom plugin + report
    post = p_dict["postprocessing"]
    assert len(post) == 3
    assert post[0]["name"] == "storage"
    assert "Storage_Min" in post[0]["params"]["metrics_definition"]

    assert post[1]["script"] == str(mock_plugin_file)
    assert post[1]["params"]["alert_valley_inventory_g"] == 600.0

    assert post[2]["name"] == "report"
    assert post[2]["params"]["report_config"]["title"] == "CFEDR Verification Report"

    # Build directly from compiled dictionary
    built_pipeline = PipelineBuilder.build(compiled)
    assert built_pipeline.name == "CompiledCFEDRPipeline"
    assert len(built_pipeline.preprocessors) == 2
    assert len(built_pipeline.postprocessors) == 3
