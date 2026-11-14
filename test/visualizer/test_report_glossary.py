"""Tests for terminology glossary support in TRICYS report generation and figures."""

import os

import pandas as pd
import pytest

from tricys.visualizer.report.figures import (
    generate_all_report_figures,
)
from tricys.visualizer.report.templates import render_report


@pytest.fixture
def sample_simulation_data():
    time = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    results_df = pd.DataFrame(
        {
            "time": time,
            "sds.I[1]": [8000.0, 7500.0, 7000.0, 7200.0, 7600.0, 8100.0],
            "blanket.I[1]": [0.0, 50.0, 150.0, 250.0, 260.0, 260.0],
            "wds.threshold": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "job_id": [1] * len(time),
        }
    )
    jobs = [
        {"job_id": 1, "name": "Case 1", "blanket.TBR": 1.15, "wds.threshold": 1000.0}
    ]
    return results_df, jobs


def test_report_without_glossary(sample_simulation_data):
    """When NO glossary is provided, parameter names must NOT be mapped or hardcoded."""
    results_df, jobs = sample_simulation_data
    cfg = {
        "model": {
            "parameters": {
                "blanket.TBR": 1.15,
                "wds.threshold": 1000.0,
            }
        },
        "scenario": {
            "variableFilter": "time|sds.I[1]|blanket.I[1]|wds.threshold",
        },
        "report": {
            "enabled": True,
            "title": "Unmapped Report",
        },
    }

    # 1. Figure generation: check labels
    figs = generate_all_report_figures(
        results_df=results_df,
        jobs_df=pd.DataFrame(jobs),
        metrics_df=None,
        task_type="simulate",
        task_config=cfg,
    )
    assert len(figs) >= 1

    # 2. Render simulate report
    report_md = render_report(
        template_id="simulate",
        config=cfg,
        jobs=jobs,
        results_df=results_df,
        figures=figs,
    )

    # Must contain original parameter names
    assert "blanket.I[1]" in report_md
    assert "wds.threshold" in report_md

    # Must NOT contain hardcoded subsystem registry names
    assert "增殖包层产氚区" not in report_md
    assert "工艺子系统 (" not in report_md


def test_report_with_glossary(sample_simulation_data):
    """When a glossary CSV is provided, mapped terms must appear in tables and charts."""
    results_df, jobs = sample_simulation_data
    glossary_csv = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../tricys/example/example_data/example_glossary/example_glossary.csv",
        )
    )
    assert os.path.exists(glossary_csv)

    cfg = {
        "model": {
            "parameters": {
                "blanket.TBR": 1.15,
                "wds.threshold": 1000.0,
            }
        },
        "scenario": {
            "variableFilter": "time|sds.I[1]|blanket.I[1]|wds.threshold",
        },
        "report": {
            "enabled": True,
            "glossary_path": glossary_csv,
            "title": "Mapped Professional Report",
        },
    }

    # 1. Figure generation
    figs = generate_all_report_figures(
        results_df=results_df,
        jobs_df=pd.DataFrame(jobs),
        metrics_df=None,
        task_type="simulate",
        task_config=cfg,
    )
    assert len(figs) >= 1

    # 2. Render simulate report
    report_md = render_report(
        template_id="simulate",
        config=cfg,
        jobs=jobs,
        results_df=results_df,
        figures=figs,
    )

    # Must contain mapped professional terminology
    assert "增殖包层" in report_md
    assert "水去氚系统" in report_md
