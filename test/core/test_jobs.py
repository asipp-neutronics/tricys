import numpy as np
import pandas as pd
import pytest

from tricys.core.jobs import (
    _expand_array_parameters,
    generate_simulation_jobs,
    parse_parameter_value,
)


# Tests for _expand_array_parameters
@pytest.mark.build_test
def test_expand_array_parameters_no_special_format():
    params = {"a": 1, "b": "hello"}
    assert _expand_array_parameters(params) == params


@pytest.mark.build_test
def test_expand_array_parameters_with_special_format():
    params = {"param": "{1, [1,2,3], '1:2:1'}"}
    expected = {"param[1]": 1, "param[2]": [1, 2, 3], "param[3]": "1:2:1"}
    assert _expand_array_parameters(params) == expected


@pytest.mark.build_test
def test_expand_array_parameters_mixed():
    params = {"a": 10, "param": "{'x', 2.5}", "b": "test"}
    expected = {"a": 10, "param[1]": "x", "param[2]": 2.5, "b": "test"}
    assert _expand_array_parameters(params) == expected


@pytest.mark.build_test
def test_expand_array_parameters_invalid_format():
    params = {"param": "{1, [2,3"}
    assert _expand_array_parameters(params) == params


@pytest.mark.build_test
def test_expand_array_parameters_empty_dict():
    assert _expand_array_parameters({}) == {}


# Tests for parse_parameter_value
@pytest.mark.build_test
def test_parse_single_values():
    assert parse_parameter_value(10) == [10]
    assert parse_parameter_value("hello") == ["hello"]
    assert parse_parameter_value(3.14) == [3.14]


@pytest.mark.build_test
def test_parse_list_value():
    assert parse_parameter_value([1, 2, 3]) == [1, 2, 3]


@pytest.mark.build_test
def test_parse_range_string():
    assert parse_parameter_value("1:5:2") == [1.0, 3.0, 5.0]


@pytest.mark.build_test
def test_parse_linspace_string():
    result = parse_parameter_value("linspace:0:10:5")
    assert np.allclose(result, [0.0, 2.5, 5.0, 7.5, 10.0])


@pytest.mark.build_test
def test_parse_logspace_string():
    result = parse_parameter_value("log:1:1000:4")
    assert np.allclose(result, [1.0, 10.0, 100.0, 1000.0])


@pytest.mark.build_test
def test_parse_logspace_invalid():
    assert parse_parameter_value("log:0:100:3") == ["log:0:100:3"]


@pytest.mark.build_test
def test_parse_rand_string():
    result = parse_parameter_value("rand:0:1:5")
    assert len(result) == 5
    assert all(0 <= x <= 1 for x in result)


@pytest.mark.build_test
def test_parse_file_string_column(tmpdir):
    p = tmpdir.mkdir("sub").join("data.csv")
    df = pd.DataFrame({"voltage": [1.1, 2.2, 3.3], "current": [10, 20, 30]})
    df.to_csv(p, index=False)
    result = parse_parameter_value(f"file:{p}:voltage")
    assert result == [1.1, 2.2, 3.3]


@pytest.mark.build_test
def test_parse_file_string_no_column(tmpdir):
    p = tmpdir.mkdir("sub").join("sampling.csv")
    df = pd.DataFrame({"p1": [1, 2], "p2": [3, 4]})
    df.to_csv(p, index=False)
    result = parse_parameter_value(f"file:{p}")
    assert result == [str(p)]


@pytest.mark.build_test
def test_parse_invalid_string():
    assert parse_parameter_value("invalid:format") == ["invalid:format"]
    assert parse_parameter_value("1:a:2") == ["1:a:2"]


# Tests for generate_simulation_jobs
@pytest.mark.build_test
def test_generate_jobs_no_sweeps():
    params = {"a": 1, "b": "x"}
    expected = [{"a": 1, "b": "x"}]
    assert generate_simulation_jobs(params) == expected


@pytest.mark.build_test
def test_generate_jobs_with_one_sweep():
    params = {"a": "1:3:1", "b": 10}
    expected = [{"a": 1.0, "b": 10}, {"a": 2.0, "b": 10}, {"a": 3.0, "b": 10}]
    jobs = generate_simulation_jobs(params)
    assert jobs == expected


@pytest.mark.build_test
def test_generate_jobs_with_multiple_sweeps():
    params = {"a": [1, 2], "b": "10:20:10"}
    expected = [
        {"a": 1, "b": 10.0},
        {"a": 1, "b": 20.0},
        {"a": 2, "b": 10.0},
        {"a": 2, "b": 20.0},
    ]
    jobs = generate_simulation_jobs(params)
    # Sort jobs and expected results to ensure comparison is order-independent
    sorted_jobs = sorted(jobs, key=lambda x: (x["a"], x["b"]))
    sorted_expected = sorted(expected, key=lambda x: (x["a"], x["b"]))
    assert sorted_jobs == sorted_expected


@pytest.mark.build_test
def test_generate_jobs_with_array_expansion():
    params = {"p": "{10, 20}", "q": 5}
    expected = [{"p[1]": 10, "p[2]": 20, "q": 5}]
    assert generate_simulation_jobs(params) == expected


@pytest.mark.build_test
def test_generate_jobs_with_array_and_sweeps():
    params = {"p": "{10, 20}", "s": "1:2:1"}
    expected = [
        {"p[1]": 10, "p[2]": 20, "s": 1.0},
        {"p[1]": 10, "p[2]": 20, "s": 2.0},
    ]
    jobs = generate_simulation_jobs(params)
    sorted_jobs = sorted(jobs, key=lambda x: x["s"])
    assert sorted_jobs == expected


@pytest.mark.build_test
def test_generate_jobs_empty():
    assert generate_simulation_jobs({}) == [{}]


@pytest.mark.build_test
def test_generate_jobs_from_file(tmpdir):
    p = tmpdir.mkdir("sub").join("data.csv")
    df = pd.DataFrame({"p1": [1, 2], "p2": ["a", "b"]})
    df.to_csv(p, index=False)
    params = {"file": str(p), "p3": 10}
    expected = [{"p1": 1, "p2": "a", "p3": 10}, {"p1": 2, "p2": "b", "p3": 10}]
    jobs = generate_simulation_jobs(params)
    assert jobs == expected
