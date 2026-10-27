from __future__ import annotations

import numpy as np
import pytest

from tricys.core.sampling import (
    SamplingEngine,
    build_salib_problem,
    generate_salib_samples,
)


def test_build_salib_problem_dict_and_shorthand():
    specs = {
        "param1": {"bounds": [10.0, 50.0], "distribution": "unif"},
        "param2": [0.1, 0.9],
        "invalid_param": "ignored_non_spec",
    }
    problem = build_salib_problem(specs)

    assert problem["num_vars"] == 2
    assert problem["names"] == ["param1", "param2"]
    assert problem["bounds"] == [[10.0, 50.0], [0.1, 0.9]]
    assert problem["dists"] == ["unif", "unif"]


def test_generate_salib_samples_sobol():
    problem = {
        "num_vars": 2,
        "names": ["a", "b"],
        "bounds": [[0.0, 1.0], [10.0, 20.0]],
        "dists": ["unif", "unif"],
    }
    # Sobol with N=8, calc_second_order=False -> N * (D + 2) = 8 * 4 = 32
    samples = generate_salib_samples(
        problem, method="sobol", N=8, calc_second_order=False, seed=42
    )
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (32, 2)
    assert np.all(samples[:, 0] >= 0.0) and np.all(samples[:, 0] <= 1.0)
    assert np.all(samples[:, 1] >= 10.0) and np.all(samples[:, 1] <= 20.0)


def test_generate_salib_samples_latin():
    problem = {
        "num_vars": 3,
        "names": ["x", "y", "z"],
        "bounds": [[0.0, 5.0], [10.0, 15.0], [-1.0, 1.0]],
        "dists": ["unif", "unif", "unif"],
    }
    samples = generate_salib_samples(problem, method="latin", N=15, seed=123)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (15, 3)


def test_generate_salib_samples_unsupported_method():
    problem = {"num_vars": 1, "names": ["x"], "bounds": [[0, 1]], "dists": ["unif"]}
    with pytest.raises(ValueError, match="Unsupported statistical sampling method"):
        generate_salib_samples(problem, method="non_existent_method")


def test_sampling_engine_statistical_generate_jobs():
    specs = {
        "temp": {"bounds": [300.0, 400.0], "distribution": "unif"},
        "flow": [1.0, 5.0],
    }
    engine = SamplingEngine(
        specs=specs,
        method="latin",
        sample_n=10,
        base_params={"fixed_param": 99.9},
        seed=42,
    )
    jobs = engine.generate_jobs()

    assert len(jobs) == 10
    for job in jobs:
        assert "temp" in job
        assert 300.0 <= job["temp"] <= 400.0
        assert "flow" in job
        assert 1.0 <= job["flow"] <= 5.0
        assert job["fixed_param"] == 99.9


def test_sampling_engine_grid_generate_jobs():
    specs = {
        "param_a": [1, 2],
        "param_b": [10, 20, 30],
    }
    engine = SamplingEngine(
        specs=specs,
        method="grid",
        base_params={"baseline": "active"},
    )
    jobs = engine.generate_jobs()

    assert len(jobs) == 6  # 2 * 3 = 6
    assert all(job.get("baseline") == "active" for job in jobs)
    a_values = {job["param_a"] for job in jobs}
    b_values = {job["param_b"] for job in jobs}
    assert a_values == {1, 2}
    assert b_values == {10, 20, 30}


def test_sampling_engine_from_config_schema2():
    config = {
        "preprocess": {
            "sampling": {
                "mode": "sobol",
                "sobol": {
                    "parameters": {
                        "alpha": {"bounds": [0.1, 0.5]},
                        "beta": [1.0, 2.0],
                    },
                    "sample_n": 8,
                    "seed": 99,
                },
            }
        }
    }
    engine = SamplingEngine.from_config(config)
    assert engine is not None
    assert engine.method == "sobol"
    assert engine.sample_n == 8
    jobs = engine.generate_jobs()
    # N=8, D=2, calc_second_order=False -> 32
    assert len(jobs) == 32


def test_sampling_engine_from_config_legacy_sobol():
    config = {
        "sensitivity_analysis": {
            "analysis_cases": [
                {
                    "name": "Sobol_Case",
                    "independent_variable_sampling": {
                        "p1": {"bounds": [1.0, 10.0]},
                        "p2": {"bounds": [20.0, 30.0]},
                    },
                    "analyzer": {
                        "method": "latin",
                        "sample_N": 25,
                    },
                }
            ]
        }
    }
    engine = SamplingEngine.from_config(config)
    assert engine is not None
    assert engine.method == "latin"
    assert engine.sample_n == 25
    jobs = engine.generate_jobs()
    assert len(jobs) == 25


def test_sampling_engine_from_config_legacy_grid_cases():
    config = {
        "sensitivity_analysis": {
            "analysis_cases": [
                {
                    "independent_variable": "pressure",
                    "independent_variable_sampling": [100, 200, 300],
                    "simulation_parameters": {
                        "ratio": [0.2, 0.4],
                    },
                }
            ]
        }
    }
    engine = SamplingEngine.from_config(config)
    assert engine is not None
    assert engine.method == "grid"
    jobs = engine.generate_jobs()
    assert len(jobs) == 6  # 3 * 2 = 6
