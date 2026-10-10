from __future__ import annotations

import json
import os
import shutil

import pandas as pd
import pytest

from tricys.plugins.safety_plugin import SafetyAuditorPlugin
from tricys.plugins.static_alarm import check_thresholds
from tricys.workflow.context import PipelineContext

# --- Fixtures and Setup ---
TEST_DIR = "temp_static_alarm_test"


def create_hdf5_fixture(
    file_path: str, jobs_df: pd.DataFrame, results_df: pd.DataFrame
) -> None:
    with pd.HDFStore(file_path, mode="w") as store:
        store.put("jobs", jobs_df, format="table", data_columns=True)
        store.append("results", results_df, index=False, data_columns=True)


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Fixture to create and cleanup the test directory for each test."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)


# --- Tests for static_alarm.py & SafetyAuditorPlugin ---


@pytest.mark.build_test
def test_check_thresholds_no_alarm():
    """Tests that no alarm is triggered when data is within thresholds."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1]})
    df = pd.DataFrame(
        {
            "time": [0, 1, 2, 3],
            "var1": [1, 2, 3, 4],
            "var2": [10, 20, 30, 40],
            "job_id": [1, 1, 1, 1],
        }
    )
    create_hdf5_fixture(h5_path, jobs_df, df)
    rules = [{"columns": ["var1"], "min": 0, "max": 5}]
    check_thresholds(h5_path, TEST_DIR, rules)

    report_path = os.path.join(TEST_DIR, "alarm_report.json")
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert not any(item["has_alarm"] for item in report if item["variable"] == "var1")


@pytest.mark.build_test
def test_check_thresholds_max_exceeded():
    """Tests that an alarm is triggered when max threshold is exceeded."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1]})
    df = pd.DataFrame(
        {"time": [0, 1, 2, 3], "var1": [1, 6, 3, 4], "job_id": [1, 1, 1, 1]}
    )
    create_hdf5_fixture(h5_path, jobs_df, df)
    rules = [{"columns": ["var1"], "min": 0, "max": 5}]
    check_thresholds(h5_path, TEST_DIR, rules)

    report_path = os.path.join(TEST_DIR, "alarm_report.json")
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert any(item["has_alarm"] for item in report if item["variable"] == "var1")


@pytest.mark.build_test
def test_check_thresholds_min_exceeded():
    """Tests that an alarm is triggered when min threshold is exceeded."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1]})
    df = pd.DataFrame(
        {"time": [0, 1, 2, 3], "var1": [1, -1, 3, 4], "job_id": [1, 1, 1, 1]}
    )
    create_hdf5_fixture(h5_path, jobs_df, df)
    rules = [{"columns": ["var1"], "min": 0, "max": 5}]
    check_thresholds(h5_path, TEST_DIR, rules)

    report_path = os.path.join(TEST_DIR, "alarm_report.json")
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert any(item["has_alarm"] for item in report if item["variable"] == "var1")


@pytest.mark.build_test
def test_check_thresholds_sweep_task():
    """Tests threshold checking for parameter sweep results."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1, 2], "p": ["1", "2"]})
    df = pd.DataFrame(
        {
            "time": [0, 1, 2, 0, 1, 2],
            "var": [1, 2, 3, 4, 8, 6],
            "job_id": [1, 1, 1, 2, 2, 2],
        }
    )
    create_hdf5_fixture(h5_path, jobs_df, df)
    rules = [{"columns": ["var"], "min": 0, "max": 5}]
    check_thresholds(h5_path, TEST_DIR, rules)

    report_path = os.path.join(TEST_DIR, "alarm_report.json")
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    alarm_p1 = next(item["has_alarm"] for item in report if item["p"] == "1")
    alarm_p2 = next(item["has_alarm"] for item in report if item["p"] == "2")

    assert not alarm_p1
    assert alarm_p2


@pytest.mark.build_test
def test_safety_auditor_plugin_execution():
    """Tests the functional SafetyAuditorPlugin execution on PipelineContext."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1]})
    df = pd.DataFrame({"time": [0, 1], "safety_var": [10.0, 99.0], "job_id": [1, 1]})
    create_hdf5_fixture(h5_path, jobs_df, df)

    rules = [{"columns": ["safety_var"], "min": 0.0, "max": 50.0}]
    plugin = SafetyAuditorPlugin(alarm_rules=rules)

    context = PipelineContext(
        config={"static_alarm": {"rules": rules}},
        results_dir=TEST_DIR,
        results_store_path=h5_path,
    )

    assert plugin.validate(context) is True
    plugin.run(context)

    # Post_processing report directory
    report_path = os.path.join(TEST_DIR, "post_processing", "alarm_report.json")
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert any(item["has_alarm"] for item in report if item["variable"] == "safety_var")
