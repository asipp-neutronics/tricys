from __future__ import annotations

import os
import shutil

import pandas as pd
import pytest

from tricys.analysis.baseline_analysis import baseline_analysis
from tricys.plugins.report_plugin import BaselinePlotPlugin
from tricys.workflow.context import PipelineContext

TEST_DIR = "temp_baseline_analysis_test"


def create_hdf5_fixture(
    file_path: str, jobs_df: pd.DataFrame, results_df: pd.DataFrame
) -> None:
    with pd.HDFStore(file_path, mode="w") as store:
        store.put("jobs", jobs_df, format="table", data_columns=True)
        store.append("results", results_df, index=False, data_columns=True)


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Set up and tear down the test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)


@pytest.mark.build_test
def test_baseline_analysis():
    """Test the direct baseline_analysis function."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1]})
    results_df = pd.DataFrame(
        {
            "time": range(10),
            "sds.I[1]": range(10, 20),
            "sds.I[2]": range(20, 30),
            "job_id": [1] * 10,
        }
    )
    create_hdf5_fixture(h5_path, jobs_df, results_df)

    output_dir = os.path.join(TEST_DIR, "output")

    baseline_analysis(h5_path, output_dir)

    report_dir = os.path.join(TEST_DIR, "report")
    assert os.path.exists(report_dir)

    plot1_path = os.path.join(report_dir, "simulation_all_curves_detailed.svg")
    plot2_path = os.path.join(report_dir, "final_values_bar_chart.svg")
    report_path = os.path.join(report_dir, "baseline_condition_analysis_report.md")

    assert os.path.exists(plot1_path)
    assert os.path.exists(plot2_path)
    assert os.path.exists(report_path)


@pytest.mark.build_test
def test_baseline_plot_plugin_lifecycle():
    """Test the BaselinePlotPlugin executing through PipelineContext."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1]})
    results_df = pd.DataFrame(
        {
            "time": range(10),
            "sds.I[1]": range(10, 20),
            "sds.I[2]": range(20, 30),
            "job_id": [1] * 10,
        }
    )
    create_hdf5_fixture(h5_path, jobs_df, results_df)

    plugin = BaselinePlotPlugin(run_baseline_plot=True)
    context = PipelineContext(
        config={"postprocess": {"baseline": {"enabled": True}}},
        results_dir=TEST_DIR,
        results_store_path=h5_path,
    )

    plugin.run(context)

    report_dir = os.path.join(TEST_DIR, "report")
    assert os.path.exists(report_dir)
    assert os.path.exists(
        os.path.join(report_dir, "baseline_condition_analysis_report.md")
    )
