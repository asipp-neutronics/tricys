import json
import os
import shutil

import pandas as pd
import pytest

from tricys.postprocess.rise_analysis import analyze_rise_dip

# --- Fixtures and Setup ---
TEST_DIR = "temp_rise_analysis_test"


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Fixture to create and cleanup the test directory for each test."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


# --- Tests for rise_analysis.py ---


@pytest.mark.build_test
def test_analyze_rise_dip_feature_present():
    """Tests that a 'dip and rise' feature is correctly detected."""
    df = pd.DataFrame(
        {
            "time": [0, 10, 20, 30, 40],
            "var&p=A": [100, 80, 75, 85, 110],  # clear dip and rise
        }
    )
    analyze_rise_dip(df, TEST_DIR)

    report_path = os.path.join(TEST_DIR, "rise_report.json")
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        report = json.load(f)

    assert len(report) == 1
    assert report[0]["rises"] is True
    assert report[0]["p"] == "A"


@pytest.mark.build_test
def test_analyze_rise_dip_feature_absent_monotonic():
    """Tests that a monotonic decrease is not flagged as 'dip and rise'."""
    df = pd.DataFrame(
        {
            "time": [0, 10, 20, 30, 40],
            "var&p=B": [100, 90, 80, 70, 60],  # monotonic decrease
        }
    )
    analyze_rise_dip(df, TEST_DIR)

    report_path = os.path.join(TEST_DIR, "rise_report.json")
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        report = json.load(f)

    assert len(report) == 1
    assert report[0]["rises"] is False
    assert report[0]["p"] == "B"


@pytest.mark.build_test
def test_analyze_rise_dip_multiple_curves():
    """Tests analysis with multiple curves, one with and one without the feature."""
    df = pd.DataFrame(
        {
            "time": [0, 10, 20, 30, 40],
            "var&p=A": [100, 80, 75, 85, 110],  # with rise
            "var&p=B": [100, 90, 80, 70, 60],  # without rise
        }
    )
    analyze_rise_dip(df, TEST_DIR)

    report_path = os.path.join(TEST_DIR, "rise_report.json")
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        report = json.load(f)

    assert len(report) == 2
    report_dict = {item["p"]: item["rises"] for item in report}
    assert report_dict["A"] is True
    assert report_dict["B"] is False
