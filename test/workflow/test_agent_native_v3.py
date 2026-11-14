from __future__ import annotations

from typing import Any

import pandas as pd

from tricys.core.semantic import PhysicsSemanticResolver
from tricys.workflow.base import PostprocessPlugin
from tricys.workflow.builder import PipelineBuilder
from tricys.workflow.context import PipelineContext, SimulationJobResult
from tricys.workflow.testing import create_mock_context


def test_physics_semantic_resolver_default_and_custom() -> None:
    """Tests that PhysicsSemanticResolver accurately discovers polymorphic Modelica columns
    and supports dynamic custom mapping overrides.
    """
    df = pd.DataFrame(
        {
            "time": [0.0, 10.0, 20.0],
            "Cycle_3LC.sds.inventory": [6000.0, 5800.0, 5950.0],
            "blanket.I[1]": [100.0, 110.0, 120.0],
            "coolant_pipe.inventory": [50.0, 52.0, 54.0],
            "plasma.fb": [0.03, 0.03, 0.03],
            "my_special_valve_leak": [0.1, 0.2, 0.3],
        }
    )

    resolver = PhysicsSemanticResolver(
        custom_mapping={
            "valve_leak": ["my_special_valve_leak"],
            "sds_inventory": ["Cycle_3LC.sds.inventory"],
        }
    )

    # 1. Standard concept resolution
    sds_s = resolver.resolve_variable(df, "sds_inventory")
    assert sds_s is not None
    assert float(sds_s.iloc[0]) == 6000.0

    # 2. Polymorphic I[1] resolution
    bz_s = resolver.resolve_variable(df, "blanket_inventory")
    assert bz_s is not None
    assert float(bz_s.iloc[-1]) == 120.0

    # 3. Custom concept resolution
    leak_s = resolver.resolve_variable(df, "valve_leak")
    assert leak_s is not None
    assert float(leak_s.iloc[1]) == 0.2

    # 4. Multi-component discovery
    all_inv = resolver.get_all_inventories(df)
    assert "sds" in all_inv
    assert "blanket" in all_inv
    assert "coolant" in all_inv


def test_strongly_typed_contract_and_extensible_fields() -> None:
    """Tests SimulationJobResult typed contract and open-closed extension payloads."""
    df = pd.DataFrame({"time": [0, 1], "sds.inventory": [100, 200]})
    res = SimulationJobResult(
        job_id=42,
        parameters={"Cycle.sds.V": 1.8},
        dataframe=df,
        metrics={"min_sds": 100.0},
        extra={"custom_indicator": "STABLE_BREEDING", "tensor_dim": [10, 20]},
    )

    assert res.job_id == 42
    assert res.dataframe is not None
    assert res.metrics["min_sds"] == 100.0
    # Test extension payload
    assert res.extra["custom_indicator"] == "STABLE_BREEDING"

    # Context-level extension payload
    ctx = PipelineContext.from_config({"version": "2.0"})
    ctx.metadata["agent_audit_hash"] = "9f8a7c"
    ctx.artifacts["custom_summary_path"] = "/tmp/summary.json"

    assert ctx.metadata["agent_audit_hash"] == "9f8a7c"
    assert ctx.artifacts["custom_summary_path"] == "/tmp/summary.json"


class MockDependentPlugin(PostprocessPlugin):
    name = "mock_dependent_plugin"
    requires = ["results_store_path"]

    def run(self, context: PipelineContext) -> None:
        context.metadata["dependent_executed"] = True


def test_topological_dag_sorting_in_pipeline_builder() -> None:
    """Tests that PipelineBuilder automatically reorders plugins when dependent plugin
    is declared before its required producer (e.g. mock_dependent before storage).
    """
    config = {
        "pipeline": {
            "name": "DAGTestPipeline",
            "preprocessing": [{"name": "setup"}],
            "simulation": {"name": "standard"},
            # Notice inverted order: dependent plugin is written BEFORE storage!
            "postprocessing": [
                MockDependentPlugin(),
                {"name": "storage"},
            ],
        }
    }

    pipeline = PipelineBuilder.build(config)
    post_names = [getattr(p, "name", "") for p in pipeline.postprocessors]

    # Verify DAG automatically pushed 'storage' before 'mock_dependent_plugin'
    assert post_names == ["storage", "mock_dependent_plugin"]


def test_mock_testing_sandbox_fast_feedback() -> None:
    """Tests that Agent can test a plugin against an in-memory mock context in milliseconds."""
    mock_ctx = create_mock_context(num_jobs=2, stop_time=500.0, base_storage_g=5500.0)

    assert len(mock_ctx.results) == 2
    assert mock_ctx.results[0].dataframe is not None
    assert "Cycle_3LC.sds.inventory" in mock_ctx.results[0].dataframe.columns

    # Test physics variable resolution on mock context
    sds_ts = mock_ctx.resolve_physics_variable(
        mock_ctx.results[0].dataframe, "sds_inventory"
    )
    assert sds_ts is not None
    assert len(sds_ts) > 0
    assert sds_ts.min() < 5500.0


def test_cli_helper_scaffold_test_and_compile(tmp_path: Any) -> None:
    """Tests Agent-Native scaffolding, mock test runner, and v2 lowering compiler."""
    from tricys.workflow.cli_helper import (
        compile_v2_cli,
        generate_plugin_scaffold,
        test_plugin_in_mock_sandbox,
    )

    # 1. Test scaffold generation
    plugin_file = tmp_path / "test_agent_plugin.py"
    code = generate_plugin_scaffold(
        "auto_agent_evaluator", category="postprocess", output_path=str(plugin_file)
    )
    assert "class AutoAgentEvaluatorPlugin" in code
    assert plugin_file.exists()

    # 2. Test sandbox execution of scaffolded plugin file
    res = test_plugin_in_mock_sandbox(str(plugin_file))
    assert res["status"] == "SUCCESS"
    assert res["plugin"] == "AutoAgentEvaluatorPlugin"
    assert "auto_agent_evaluator_job_count" in res["metrics_emitted"]

    # 3. Test compilation of Schema 2.0 configuration
    sample_v2_cfg = tmp_path / "sample_v2.yaml"
    sample_v2_cfg.write_text(
        """
version: "2.0"
model:
  fileName: "test.mo"
  modelName: "Test.Model"
scenario:
  stopTime: 1200
  stepSize: 1.0
task:
  type: "simulate"
visualizer:
  report:
    enabled: true
""",
        encoding="utf-8",
    )
    out_pipe = tmp_path / "compiled_pipeline.yaml"
    compiled_yaml = compile_v2_cli(str(sample_v2_cfg), str(out_pipe))

    assert "pipeline:" in compiled_yaml
    assert "preprocessing:" in compiled_yaml
    assert "simulation:" in compiled_yaml
    assert "postprocessing:" in compiled_yaml
    assert out_pipe.exists()
