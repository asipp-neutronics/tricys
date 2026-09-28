from __future__ import annotations

from unittest.mock import patch

from tricys.analysis.metric import get_optimization_tasks
from tricys.analysis.optimizer import (
    BisectionOptimizer,
    OptimizationEngine,
    OptimizationResult,
)


def test_optimization_result_dataclass():
    res = OptimizationResult(
        metric_name="Required_TBR",
        target_parameter="blanket.TBR",
        optimal_parameter_value=1.12,
        optimal_metric_value=10500.0,
        converged=True,
        iterations=5,
        history=[{"iter": 0, "val": 1.15}, {"iter": 1, "val": 1.12}],
    )
    assert res.metric_name == "Required_TBR"
    assert res.target_parameter == "blanket.TBR"
    assert res.optimal_parameter_value == 1.12
    assert res.optimal_metric_value == 10500.0
    assert res.converged is True
    assert res.iterations == 5
    assert len(res.history) == 2


def test_optimization_engine_delegation():
    config = {
        "sensitivity_analysis": {
            "metrics_definition": {
                "Required_TBR": {
                    "method": "bisection_search",
                    "parameter_to_optimize": "blanket.TBR",
                    "search_range": [1.0, 1.3],
                }
            }
        }
    }
    job_params = {"plasma.nf": 0.5}
    fast_ctx = {
        "exe": "model.exe",
        "xml": "init.xml",
        "om_bin": "/bin",
        "temp_dir": "tmp",
    }

    with patch.object(BisectionOptimizer, "run_optimization_tasks_fast") as mock_run:
        mock_run.return_value = ({"blanket.TBR": 1.15}, {"Required_TBR": 9000.0})
        p, v = OptimizationEngine.execute_tasks(
            config, job_params, job_id=1, fast_context=fast_ctx
        )
        assert p == {"blanket.TBR": 1.15}
        assert v == {"Required_TBR": 9000.0}
        mock_run.assert_called_once_with(
            config=config, job_params=job_params, job_id=1, fast_context=fast_ctx
        )


def test_get_optimization_tasks():
    config = {
        "sensitivity_analysis": {
            "metrics_definition": {
                "Startup_Inventory": {"method": "calculate_startup_inventory"},
                "Required_TBR": {"method": "bisection_search"},
            }
        }
    }
    tasks = get_optimization_tasks(config)
    assert tasks == ["Required_TBR"]
