from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from tricys.analysis.metric import get_optimization_tasks
from tricys.analysis.optimizer import (
    BisectionOptimizer,
    NestedBisectionOptimizer,
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


def test_bisection_optimizer_reserve_residual_root_finding():
    """Test BisectionOptimizer executes bisection search for reserve_condition_residual >= 0."""
    v2_config = {
        "simulation": {"stop_time": 500.0},
        "task": {
            "type": "optimize",
            "metrics": {
                "Optimal_IS0": {
                    "evaluate": "bisection_search",
                    "parameter": "I_S_0",
                    "search_bounds": [1000.0, 5000.0],
                    "tolerance": 1.0,
                    "max_iterations": 15,
                    "target_metric": {
                        "source": "reserve_condition_residual",
                        "evaluate": "final_value",
                        "threshold": 0.0,
                        "comparator": ">=",
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

    with patch("tricys.analysis.optimizer.run_fast_subprocess_job") as mock_sub:
        with patch("os.path.exists", return_value=True):
            with patch("pandas.read_csv") as mock_csv:

                def mock_run_job(job_params, *args, **kwargs):
                    val = job_params.get("I_S_0", 0.0)
                    # Critical startup inventory is 2500.0; residual = val - 2500.0
                    residual = val - 2500.0
                    mock_csv.return_value = pd.DataFrame(
                        {
                            "time": [0, 250, 500],
                            "reserve_condition_residual": [
                                residual,
                                residual,
                                residual,
                            ],
                        }
                    )
                    return "dummy_result.csv"

                mock_sub.side_effect = mock_run_job

                params, vals = BisectionOptimizer.run_bisection_search_fast(
                    v2_config, "job_opt", "Optimal_IS0", fast_ctx
                )
                assert "Optimal_IS0" in params
                assert pytest.approx(params["Optimal_IS0"], abs=2.0) == 2500.0


def test_nested_bisection_schema_validation():
    from pydantic import ValidationError

    from tricys.core.schema import OptimizeTaskConfig

    # Valid config
    valid_cfg = {
        "type": "optimize",
        "algorithm": "nested_bisection",
        "outer": {
            "parameter": "I_S_0",
            "search_bounds": [1000.0, 50000.0],
            "tolerance": 1.0,
            "target_metric": {
                "source": "sds.inventory",
                "evaluate": "min_value",
                "threshold": 2500.0,
            },
        },
        "inner": {
            "parameter": "Lambda",
            "search_bounds": [1.0, 1.5],
            "metric_name": "Required_TBR",
            "target_metric": {
                "source": "sds.inventory",
                "evaluate": "doubling_time",
                "threshold": [8766.0, 43830.0],
            },
        },
    }
    opt_task = OptimizeTaskConfig(**valid_cfg)
    assert opt_task.algorithm == "nested_bisection"
    assert opt_task.outer.parameter == "I_S_0"
    assert opt_task.inner.metric_name == "Required_TBR"

    # Missing search_bounds in outer raises ValidationError
    invalid_outer = {**valid_cfg, "outer": {"parameter": "I_S_0"}}
    with pytest.raises(ValidationError):
        OptimizeTaskConfig(**invalid_outer)

    # Invalid bounds (low >= high) raises ValidationError
    invalid_bounds = {
        **valid_cfg,
        "outer": {
            "parameter": "I_S_0",
            "search_bounds": [5000.0, 1000.0],
            "target_metric": {"source": "sds.inventory", "evaluate": "min_value"},
        },
    }
    with pytest.raises(ValidationError):
        OptimizeTaskConfig(**invalid_bounds)


def test_nested_bisection_single_threshold():
    """Test NestedBisectionOptimizer with single doubling target."""
    config = {
        "simulation": {"time_base": "hour", "stop_time": 50000.0},
        "task": {
            "type": "optimize",
            "algorithm": "nested_bisection",
            "outer": {
                "parameter": "I_S_0",
                "search_bounds": [1000.0, 20000.0],
                "tolerance": 2.0,
                "max_iterations": 15,
                "target_metric": {
                    "source": "sds.inventory",
                    "evaluate": "min_value",
                    "threshold": 2500.0,
                    "comparator": ">=",
                },
            },
            "inner": {
                "parameter": "Lambda",
                "search_bounds": [1.05, 1.35],
                "tolerance": 0.005,
                "max_iterations": 15,
                "metric_name": "Required_TBR",
                "target_metric": {
                    "source": "sds.inventory",
                    "evaluate": "doubling_time",
                    "threshold": 43830.0,
                    "comparator": "<=",
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

    with patch("tricys.analysis.optimizer.run_fast_subprocess_job") as mock_sub:
        with patch("os.path.exists", return_value=True):
            with patch("pandas.read_csv") as mock_csv:
                # Mock physics:
                # - Nadir = I_S_0 - 8000.0 (so nadir >= 2500 when I_S_0 >= 10500.0)
                # - Doubling time = 40000.0 h if Lambda >= 1.18 else 60000.0 h
                def mock_run_job(job_params, *args, **kwargs):
                    return "dummy_result.csv"

                mock_sub.side_effect = mock_run_job

                with patch(
                    "tricys.analysis.optimizer.calculate_doubling_time"
                ) as mock_dt:
                    with patch("tricys.analysis.optimizer.get_min_value") as mock_min:

                        def dt_side_effect(series, time_series):
                            # From call args, retrieve lambda if possible or mock
                            return (
                                40000.0 if mock_run_job.last_lambda >= 1.18 else 60000.0
                            )

                        def run_with_tracking(job_params, *args, **kwargs):
                            mock_run_job.last_lambda = job_params.get("Lambda", 1.10)
                            mock_run_job.last_is0 = job_params.get("I_S_0", 10000.0)
                            mock_csv.return_value = pd.DataFrame(
                                {
                                    "time": [0, 40000],
                                    "sds.inventory": [
                                        mock_run_job.last_is0,
                                        mock_run_job.last_is0 * 2,
                                    ],
                                }
                            )
                            return "dummy_result.csv"

                        mock_sub.side_effect = run_with_tracking
                        mock_dt.side_effect = lambda s, t: (
                            40000.0 if mock_run_job.last_lambda >= 1.18 else 60000.0
                        )
                        mock_min.side_effect = lambda s: mock_run_job.last_is0 - 8000.0

                        opt_params, opt_metrics = (
                            NestedBisectionOptimizer.run_nested_bisection_fast(
                                config=config,
                                job_params={},
                                job_id_prefix="test_nested",
                                nested_cfg=config["task"],
                                fast_context=fast_ctx,
                            )
                        )

                        assert "I_S_0" in opt_params
                        assert "Required_TBR" in opt_params
                        # I_S_0 should converge to ~10500.0 (where nadir = 2500)
                        assert pytest.approx(opt_params["I_S_0"], abs=5.0) == 10500.0
                        # Required_TBR should converge to ~1.18
                        assert (
                            pytest.approx(opt_params["Required_TBR"], abs=0.01) == 1.18
                        )


def test_nested_bisection_multiple_doubling_times():
    """Test NestedBisectionOptimizer with a list of doubling time thresholds."""
    config = {
        "simulation": {"time_base": "hour", "stop_time": 70000.0},
        "task": {
            "type": "optimize",
            "algorithm": "nested_bisection",
            "outer": {
                "parameter": "I_S_0",
                "search_bounds": [5000.0, 20000.0],
                "tolerance": 5.0,
                "target_metric": {
                    "source": "sds.inventory",
                    "evaluate": "min_value",
                    "threshold": 2000.0,
                },
            },
            "inner": {
                "parameter": "Lambda",
                "search_bounds": [1.05, 1.40],
                "tolerance": 0.01,
                "metric_name": "Required_TBR",
                "target_metric": {
                    "source": "sds.inventory",
                    "evaluate": "doubling_time",
                    "threshold": [8766.0, 43830.0],
                    "comparator": "<=",
                },
            },
        },
    }
    fast_ctx = {"exe": "d.exe", "xml": "d.xml", "om_bin": "/bin", "temp_dir": "temp"}

    with patch("tricys.analysis.optimizer.run_fast_subprocess_job") as mock_sub:
        with patch("os.path.exists", return_value=True):
            with patch("pandas.read_csv") as mock_csv:

                def tracking_run(job_params, *args, **kwargs):
                    tracking_run.last_lambda = job_params.get("Lambda", 1.10)
                    tracking_run.last_is0 = job_params.get("I_S_0", 10000.0)
                    mock_csv.return_value = pd.DataFrame(
                        {
                            "time": [0, 40000],
                            "sds.inventory": [
                                tracking_run.last_is0,
                                tracking_run.last_is0 * 2,
                            ],
                        }
                    )
                    return "dummy.csv"

                mock_sub.side_effect = tracking_run

                with patch(
                    "tricys.analysis.optimizer.calculate_doubling_time"
                ) as mock_dt:
                    with patch("tricys.analysis.optimizer.get_min_value") as mock_min:
                        # If td target is 8766.0, need Lambda >= 1.30
                        # If td target is 43830.0, need Lambda >= 1.15
                        mock_dt.side_effect = lambda s, t: (
                            5000.0
                            if tracking_run.last_lambda >= 1.30
                            else (
                                30000.0 if tracking_run.last_lambda >= 1.15 else 80000.0
                            )
                        )
                        mock_min.side_effect = lambda s: tracking_run.last_is0 - 8000.0

                        opt_params, opt_metrics = (
                            NestedBisectionOptimizer.run_nested_bisection_fast(
                                config=config,
                                job_params={},
                                job_id_prefix="multi_td",
                                nested_cfg=config["task"],
                                fast_context=fast_ctx,
                            )
                        )

                        # Check that outputs contain suffixed keys for both thresholds
                        assert "I_S_0(8766.0 units)" in opt_params
                        assert "Required_TBR(8766.0 units)" in opt_params
                        assert "I_S_0(43830.0 units)" in opt_params
                        assert "Required_TBR(43830.0 units)" in opt_params

                        # Verify TBR for 1-yr doubling (8766h) is higher (~1.30) than 5-yr (~1.15)
                        assert (
                            opt_params["Required_TBR(8766.0 units)"]
                            > opt_params["Required_TBR(43830.0 units)"]
                        )
                        assert (
                            pytest.approx(
                                opt_params["Required_TBR(8766.0 units)"], abs=0.02
                            )
                            == 1.30
                        )
                        assert (
                            pytest.approx(
                                opt_params["Required_TBR(43830.0 units)"], abs=0.02
                            )
                            == 1.15
                        )


def test_optimization_engine_routing():
    """Verify OptimizationEngine routes to NestedBisectionOptimizer when optimization block is configured."""
    sweep_cfg = {
        "simulation": {"time_base": "hour", "stop_time": 50000.0},
        "task": {
            "type": "sweep",
            "parameters": [{"AF": 0.5}],
            "optimization": {
                "algorithm": "nested_bisection",
                "outer": {
                    "parameter": "I_S_0",
                    "search_bounds": [1000.0, 5000.0],
                    "target_metric": {
                        "source": "sds.inventory",
                        "evaluate": "min_value",
                        "threshold": 500.0,
                    },
                },
                "inner": {
                    "parameter": "Lambda",
                    "search_bounds": [1.1, 1.3],
                    "metric_name": "Required_TBR",
                    "target_metric": {
                        "source": "sds.inventory",
                        "evaluate": "doubling_time",
                        "threshold": 40000.0,
                    },
                },
            },
        },
    }
    assert OptimizationEngine.has_optimization(sweep_cfg) is True

    fast_ctx = {"exe": "d.exe", "xml": "d.xml", "om_bin": "/bin", "temp_dir": "temp"}
    with patch.object(
        NestedBisectionOptimizer,
        "run_nested_bisection_fast",
        return_value=({"I_S_0": 2000.0, "Required_TBR": 1.15}, {}),
    ) as mock_nested:
        p, v = OptimizationEngine.execute_tasks(sweep_cfg, {"AF": 0.5}, "001", fast_ctx)
        assert mock_nested.called
        assert p["I_S_0"] == 2000.0
        assert p["Required_TBR"] == 1.15


def test_nested_bisection_example_model_parameters():
    """Verify NestedBisectionOptimizer converges with example_model parameters (I_S_0, blanket.TBR, sds.I[1])."""
    config = {
        "simulation": {"time_base": "hour", "stop_time": 2000.0},
        "task": {
            "type": "optimize",
            "track": "track2_bisection",
            "algorithm": "nested_bisection",
            "outer": {
                "parameter": "I_S_0",
                "search_bounds": [1000.0, 5000.0],
                "tolerance": 10.0,
                "target_metric": {
                    "source": "sds.I[1]",
                    "evaluate": "min_value",
                    "threshold": 1200.0,
                    "comparator": ">=",
                },
            },
            "inner": {
                "parameter": "blanket.TBR",
                "metric_name": "Required_TBR",
                "search_bounds": [1.05, 1.25],
                "tolerance": 0.005,
                "target_metric": {
                    "source": "sds.I[1]",
                    "evaluate": "doubling_time",
                    "threshold": 1500.0,
                    "comparator": "<=",
                },
            },
        },
    }
    fast_ctx = {
        "exe": "model.exe",
        "xml": "model.xml",
        "om_bin": "/bin",
        "temp_dir": "temp",
    }

    with patch("tricys.analysis.optimizer.run_fast_subprocess_job") as mock_sub:
        with patch("os.path.exists", return_value=True):
            with patch("pandas.read_csv") as mock_csv:

                def tracking_run(job_params, *args, **kwargs):
                    tracking_run.last_tbr = job_params.get("blanket.TBR", 1.10)
                    tracking_run.last_is0 = job_params.get("I_S_0", 3000.0)
                    mock_csv.return_value = pd.DataFrame(
                        {
                            "time": [0, 1500],
                            "sds.I[1]": [
                                tracking_run.last_is0,
                                tracking_run.last_is0 * 2,
                            ],
                        }
                    )
                    return "dummy.csv"

                mock_sub.side_effect = tracking_run

                with patch(
                    "tricys.analysis.optimizer.calculate_doubling_time"
                ) as mock_dt:
                    with patch("tricys.analysis.optimizer.get_min_value") as mock_min:
                        # Doubles in 1200h if TBR >= 1.15 else 3000h
                        mock_dt.side_effect = lambda s, t, **kw: (
                            1200.0 if tracking_run.last_tbr >= 1.15 else 3000.0
                        )
                        # Nadir = I_S_0 - 1500.0 (>= 1200 when I_S_0 >= 2700)
                        mock_min.side_effect = lambda s: tracking_run.last_is0 - 1500.0

                        opt_params, opt_values = (
                            NestedBisectionOptimizer.run_nested_bisection_fast(
                                config=config,
                                job_params={"wds.threshold": 1000.0},
                                job_id_prefix="example_model_nested",
                                nested_cfg=config["task"],
                                fast_context=fast_ctx,
                            )
                        )

                        assert "I_S_0" in opt_params
                        assert "Required_TBR" in opt_params
                        assert pytest.approx(opt_params["I_S_0"], abs=10.0) == 2700.0
                        assert (
                            pytest.approx(opt_params["Required_TBR"], abs=0.01) == 1.15
                        )
