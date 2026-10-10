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


def test_get_optimization_tasks_v2_task_metrics():
    """Verify get_optimization_tasks correctly discovers bisection tasks from V2 task.metrics."""
    v2_config = {
        "task": {
            "type": "simulate",
            "metrics": {
                "peak_inv": {"source": "sds.I[1]", "evaluate": "peak_value"},
                "Required_TBR": {
                    "evaluate": "bisection_search",
                    "parameter": "blanket.TBR",
                    "search_bounds": [1.0, 1.4],
                    "target_metric": {
                        "source": "sds.I[1]",
                        "evaluate": "doubling_time",
                        "threshold": 63072000.0,
                    },
                },
            },
        }
    }
    tasks = get_optimization_tasks(v2_config)
    assert tasks == ["Required_TBR"]


def test_bisection_optimizer_v2_doubling_time_mock_run():
    """Test BisectionOptimizer executes bisection search for doubling_time within 2 years."""
    v2_config = {
        "simulation": {"stop_time": 100000.0},
        "task": {
            "type": "simulate",
            "metrics": {
                "Required_TBR": {
                    "evaluate": "bisection_search",
                    "parameter": "blanket.TBR",
                    "search_bounds": [1.05, 1.25],
                    "tolerance": 0.01,
                    "max_iterations": 4,
                    "target_metric": {
                        "source": "sds.I[1]",
                        "evaluate": "doubling_time",
                        "threshold": 63072000.0,
                    },
                },
            },
        },
    }
    fast_ctx = {
        "exe": "dummy.exe",
        "xml": "dummy.xml",
        "om_bin": "/bin",
        "temp_dir": "temp",
    }

    # Mock run_fast_subprocess_job and calculate_doubling_time
    with patch("tricys.analysis.optimizer.run_fast_subprocess_job") as mock_sub:
        with patch("tricys.analysis.optimizer.calculate_doubling_time") as mock_td:
            with patch("os.path.exists", return_value=True):
                with patch("pandas.read_csv") as mock_csv:
                    import pandas as pd

                    mock_csv.return_value = pd.DataFrame(
                        {
                            "time": [0, 50000, 100000],
                            "sds.I[1]": [1000, 1500, 2200],
                        }
                    )

                    # If TBR >= 1.15, doubling time is 50000000s (< 63072000s), else NaN (fails)
                    def side_effect(series, time_series):
                        return 50000000.0

                    mock_td.side_effect = side_effect
                    mock_sub.return_value = "dummy_result.csv"

                    params, vals = BisectionOptimizer.run_bisection_search_fast(
                        v2_config, "job_1", "Required_TBR", fast_ctx
                    )
                    assert "Required_TBR" in params
                    assert 1.05 <= params["Required_TBR"] <= 1.25
