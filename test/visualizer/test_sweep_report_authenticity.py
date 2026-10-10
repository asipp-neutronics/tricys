import pandas as pd

from tricys.visualizer.report.figures import (
    generate_metrics_comparison_plot,
    generate_valley_comparison_plot,
)
from tricys.visualizer.report.templates import render_sweep


def test_sweep_report_kpi_table_and_baseline_authenticity():
    config = {
        "model": {"modelName": "CFEDR_Cycle"},
        "task": {"type": "sweep"},
        "report": {"title": "多工况扫描报告"},
    }
    # Case 3 is first row, Case 1 has 'Baseline' in name
    jobs = [
        {
            "job_id": 3,
            "name": "Case_3_High_TBR",
            "blanket.TBR": 1.25,
            "wds.thresh": 1200.0,
        },
        {
            "job_id": 2,
            "name": "Case_2_Low_TBR",
            "blanket.TBR": 1.05,
            "wds.thresh": 800.0,
        },
        {
            "job_id": 1,
            "name": "Case_1_Nominal_Baseline",
            "blanket.TBR": 1.15,
            "wds.thresh": 1000.0,
        },
    ]
    metrics_df = pd.DataFrame(
        [
            {"job_id": 3, "final_inventory": 4500.0, "peak_inventory": 5200.0},
            {"job_id": 2, "final_inventory": 2100.0, "peak_inventory": 3100.0},
            {"job_id": 1, "final_inventory": 2800.0, "peak_inventory": 3400.0},
        ]
    )

    report_md = render_sweep(config=config, jobs=jobs, metrics_df=metrics_df)

    # 1. Section 1.1 Baseline check: must identify Case 1 as baseline (1.15), NOT row 0 (1.25)
    assert "基准工况取值 (Baseline)" in report_md
    assert "1.15" in report_md
    assert "blanket.TBR" in report_md
    assert "wds.thresh" in report_md

    # 2. Section 2 KPI table check: must contain case names and parameter values along with metrics
    assert "Case_3_High_TBR" in report_md
    assert "Case_2_Low_TBR" in report_md
    assert "Case_1_Nominal_Baseline" in report_md
    assert "4500" in report_md
    assert "5200" in report_md

    # 3. Facts check: no hardcoded deficit or fake hours
    assert "必需首炉外供氚装量" not in report_md
    assert "达到氚自持转折时滞" not in report_md
    assert "final_inventory" in report_md
    assert "peak_inventory" in report_md


def test_sweep_report_mean_when_no_baseline():
    config = {
        "model": {"modelName": "CFEDR_Cycle"},
        "task": {"type": "sweep"},
    }
    jobs = [
        {"job_id": 1, "name": "Scan_A", "temp": 300.0},
        {"job_id": 2, "name": "Scan_B", "temp": 400.0},
    ]
    metrics_df = pd.DataFrame(
        [
            {"job_id": 1, "efficiency": 0.85},
            {"job_id": 2, "efficiency": 0.92},
        ]
    )

    report_md = render_sweep(config=config, jobs=jobs, metrics_df=metrics_df)

    # No case has baseline in name -> should show 样本均值 (Mean), not 基准工况取值
    assert "样本均值 (Mean)" in report_md
    assert "350" in report_md


def test_metrics_comparison_plot_no_fabricated_titles():
    metrics_df = pd.DataFrame(
        [
            {
                "job_id": 1,
                "name": "Case_1",
                "final_inventory": 4643.88,
                "peak_inventory": 5000.0,
            },
            {
                "job_id": 2,
                "name": "Case_2",
                "final_inventory": 2058.30,
                "peak_inventory": 3000.0,
            },
        ]
    )
    jobs_df = pd.DataFrame(
        [
            {"job_id": 1, "name": "Case_1", "tbr": 1.2},
            {"job_id": 2, "name": "Case_2", "tbr": 1.1},
        ]
    )

    fig_res = generate_metrics_comparison_plot(metrics_df, jobs_df)
    assert fig_res is not None
    assert fig_res["title"] == "多工况核心性能指标横向对比柱状图"
    # Description should be truthful
    assert "必需首炉外供装填量" not in fig_res["description"]
    assert "达到自持转折时滞" not in fig_res["description"]


def test_valley_comparison_plot_skipped_when_no_valley():
    # results where inventory is strictly increasing or constant (no drawdown valley)
    results_df = pd.DataFrame(
        {
            "time": [0.0, 100.0, 200.0, 300.0],
            "job_id": [1, 1, 1, 1],
            "sds.I[1]": [1000.0, 1200.0, 1400.0, 1600.0],
        }
    )
    jobs_df = pd.DataFrame([{"job_id": 1, "name": "Case_1"}])

    fig_res = generate_valley_comparison_plot(results_df, jobs_df)
    # Must be None because no drawdown valley physically occurred!
    assert fig_res is None


def test_sweep_report_required_tbr_as_metric_not_parameter():
    config = {
        "model": {"modelName": "example_model.Cycle"},
        "task": {
            "type": "sweep",
            "metrics": {
                "startup_inventory": {
                    "source": "sds.I[1]",
                    "evaluate": "startup_inventory",
                },
                "Required_TBR": {
                    "evaluate": "bisection_search",
                    "parameter": "blanket.TBR",
                },
            },
        },
    }
    jobs = [
        {"job_id": 1, "name": "Width_100", "pulseSource.width": 100.0},
        {"job_id": 2, "name": "Width_80", "pulseSource.width": 80.0},
    ]
    metrics_df = pd.DataFrame(
        [
            {"job_id": 1, "startup_inventory": 1907.0, "Required_TBR": 1.294},
            {"job_id": 2, "startup_inventory": 1816.0, "Required_TBR": 1.322},
        ]
    )

    report_md = render_sweep(config=config, jobs=jobs, metrics_df=metrics_df)

    # Parameter exploration table 1.1 must only list pulseSource.width
    assert "pulseSource.width" in report_md
    sec1 = report_md.split("## 2. 多工况核心性能指标")[0]
    assert "pulseSource.width" in sec1
    # Required_TBR must NOT be in Section 1 parameter tables
    assert "Required_TBR" not in sec1

    # In section 2 and 3, Required_TBR must be present as a metric
    sec2 = report_md.split("## 2. 多工况核心性能指标")[1]
    assert "Required_TBR" in sec2

    # Verify generate_metrics_comparison_plot includes Required_TBR and generates valid plot
    fig_res = generate_metrics_comparison_plot(
        metrics_df, pd.DataFrame(jobs), config=config
    )
    assert fig_res is not None
    assert fig_res["title"] == "多工况核心性能指标横向对比柱状图"
    assert fig_res["image_data"] is not None
