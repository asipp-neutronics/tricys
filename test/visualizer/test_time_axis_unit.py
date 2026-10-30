import numpy as np
import pandas as pd
import pytest

from tricys.analysis.tritium_cycle import extract_startup_inventory_metrics
from tricys.visualizer.callbacks import _format_time_range
from tricys.visualizer.report.figures import (
    generate_inventory_trajectories_plot,
    generate_subsystem_dynamics_plot,
    generate_valley_comparison_plot,
)


def test_format_time_range_with_different_time_bases():
    # 1. Modelica time_base == "second": raw seconds scaled to hours
    val_sec, detail_sec = _format_time_range(
        0.0, 7200000.0, config_data={"scenario": {"time_base": "second"}}
    )
    assert "0 h -> 2000 h" in val_sec
    assert "time_base=second" in detail_sec
    assert "7.2e+06 s" in detail_sec or "7200000" in detail_sec

    # 2. Modelica time_base == "hour": raw hours passed as hours
    val_hr, detail_hr = _format_time_range(
        0.0, 2000.0, config_data={"scenario": {"time_base": "hour"}}
    )
    assert "0 h -> 2000 h" in val_hr
    assert "time_base=hour" in detail_hr


def test_update_results_graph_hour_scaling():
    import dash

    from tricys.visualizer.callbacks import register_callbacks

    app = dash.Dash(__name__)
    register_callbacks(app)

    entry = next(v for k, v in app.callback_map.items() if "results-graph.figure" in k)
    callback_fn = getattr(entry["callback"], "__wrapped__", entry["callback"])

    # Raw results data in seconds (e.g. 0 to 7.2e6 s, which is 2000 hours)
    data = [
        {"job_id": 1, "time": 0.0, "sds.I[1]": 8000.0},
        {"job_id": 1, "time": 3600000.0, "sds.I[1]": 6500.0},
        {"job_id": 1, "time": 7200000.0, "sds.I[1]": 9000.0},
    ]

    # 1. When time_base == "second", x data must be scaled to hours (0, 1000, 2000)
    config_sec = {"scenario": {"time_base": "second"}}
    fig_sec = callback_fn(data, None, "absolute", ["sds.I[1]"], config_sec)
    x_vals_sec = list(fig_sec.data[0].x)
    assert x_vals_sec == pytest.approx([0.0, 1000.0, 2000.0])
    assert fig_sec.layout.xaxis.title.text == "Time (h)"

    # 2. When time_base == "hour", raw data is already hours (0, 100, 200)
    data_hour = [
        {"job_id": 1, "time": 0.0, "sds.I[1]": 8000.0},
        {"job_id": 1, "time": 100.0, "sds.I[1]": 6500.0},
        {"job_id": 1, "time": 200.0, "sds.I[1]": 9000.0},
    ]
    config_hour = {"scenario": {"time_base": "hour"}}
    fig_hour = callback_fn(data_hour, None, "absolute", ["sds.I[1]"], config_hour)
    x_vals_hour = list(fig_hour.data[0].x)
    assert x_vals_hour == pytest.approx([0.0, 100.0, 200.0])

    # 3. Legacy file without time_base in config (e.g. stop_time = 500.0 h): MUST NOT divide by 3600!
    data_legacy_500 = [
        {"job_id": 1, "time": 0.0, "sds.I[1]": 8000.0},
        {"job_id": 1, "time": 250.0, "sds.I[1]": 6500.0},
        {"job_id": 1, "time": 500.0, "sds.I[1]": 9000.0},
    ]
    config_legacy_500 = {"simulation": {"stop_time": 500.0}}
    fig_legacy = callback_fn(
        data_legacy_500, None, "absolute", ["sds.I[1]"], config_legacy_500
    )
    x_vals_legacy = list(fig_legacy.data[0].x)
    assert x_vals_legacy == pytest.approx([0.0, 250.0, 500.0])
    assert fig_legacy.layout.xaxis.title.text == "Time (h)"

    # 4. Long legacy run (e.g. 12000 hours): MUST NOT divide by 3600!
    data_12000 = [
        {"job_id": 1, "time": 0.0, "sds.I[1]": 8000.0},
        {"job_id": 1, "time": 6000.0, "sds.I[1]": 6500.0},
        {"job_id": 1, "time": 12000.0, "sds.I[1]": 9000.0},
    ]
    config_12000 = {"simulation": {"stop_time": 12000.0}}
    fig_12000 = callback_fn(data_12000, None, "absolute", ["sds.I[1]"], config_12000)
    x_vals_12000 = list(fig_12000.data[0].x)
    assert x_vals_12000 == pytest.approx([0.0, 6000.0, 12000.0])

    # 5. Missing / None config with hour-scale data: MUST NOT divide by 3600!
    fig_no_cfg = callback_fn(data_legacy_500, None, "absolute", ["sds.I[1]"], None)
    x_vals_no_cfg = list(fig_no_cfg.data[0].x)
    assert x_vals_no_cfg == pytest.approx([0.0, 250.0, 500.0])


def test_format_time_range_legacy_and_unspecified():
    # Legacy file with stop_time = 500.0: should be 0 h -> 500 h, not 0 h -> 0.138889 h
    v1, d1 = _format_time_range(
        0.0, 500.0, config_data={"simulation": {"stop_time": 500.0}}
    )
    assert "0 h -> 500 h" in v1

    # 12000 h run without time_base: should be 0 h -> 12000 h, not 0 h -> 3.33 h
    v2, d2 = _format_time_range(
        0.0, 12000.0, config_data={"simulation": {"stop_time": 12000.0}}
    )
    assert "0 h -> 12000 h" in v2

    # None config with t_max < 50000: stays hours
    v3, d3 = _format_time_range(0.0, 1000.0, config_data=None)
    assert "0 h -> 1000 h" in v3

    # None config with t_max >= 50000: recognized as seconds and scaled to hours
    v4, d4 = _format_time_range(0.0, 7200000.0, config_data=None)
    assert "0 h -> 2000 h" in v4


def test_figures_dd_phase_shading_matches_hour_scale():
    # 2000 hours in seconds: 7.2e6 s
    times_sec = np.linspace(0, 7200000.0, 101)
    sds_inv = 8000.0 - 2000.0 * np.sin(np.pi * times_sec / 7200000.0)
    results_df = pd.DataFrame(
        {"time": times_sec, "sds.inventory": sds_inv, "job_id": 1}
    )
    # t_dd = 100.0 hours (Schema unit is always hours)
    jobs_df = pd.DataFrame([{"job_id": 1, "name": "Case_1", "t_dd": 100.0}])
    config = {"scenario": {"time_base": "second", "stopTime": 2000.0}}

    fig_res = generate_inventory_trajectories_plot(
        results_df=results_df, jobs_df=jobs_df, config=config
    )
    assert fig_res is not None
    assert "data:image/png;base64," in fig_res["image_data"]


def test_figures_valley_comparison_filters_dd_in_hours():
    # 2000 hours in seconds: 7.2e6 s
    times_sec = np.linspace(0, 7200000.0, 201)
    # Dip happens around 1500 hours (5.4e6 s)
    sds_inv = 8000.0 - 3000.0 * np.sin(np.pi * times_sec / 7200000.0)
    results_df = pd.DataFrame(
        {"time": times_sec, "sds.inventory": sds_inv, "job_id": 1}
    )
    # Ignition delay t_dd = 100.0 hours
    jobs_df = pd.DataFrame([{"job_id": 1, "name": "Case_1", "t_dd": 100.0}])
    config = {"scenario": {"time_base": "second", "stopTime": 2000.0}}

    fig_res = generate_valley_comparison_plot(
        results_df=results_df, jobs_df=jobs_df, config=config
    )
    assert fig_res is not None
    assert "谷底" in fig_res["title"]


def test_figures_subsystem_dynamics_plot_with_second_model():
    times_sec = np.linspace(0, 7200000.0, 101)
    sds_inv = 8000.0 - 2000.0 * np.sin(np.pi * times_sec / 7200000.0)
    results_df = pd.DataFrame(
        {
            "time": times_sec,
            "sds.inventory": sds_inv,
            "tep.inventory": np.linspace(0, 500, 101),
            "job_id": 1,
        }
    )
    jobs_df = pd.DataFrame([{"job_id": 1, "name": "Case_1", "t_dd": 50.0}])
    config = {
        "scenario": {
            "time_base": "second",
            "stopTime": 2000.0,
            "variableFilter": ["tep.inventory"],
        }
    }

    fig_res = generate_subsystem_dynamics_plot(
        results_df=results_df, jobs_df=jobs_df, config=config
    )
    assert fig_res is not None
    assert (
        "上下联动展示" in fig_res["description"]
        or "物理时间对齐关系" in fig_res["description"]
    )


def test_extract_startup_inventory_metrics_with_non_zero_ignition_delay():
    # 2000 hours in seconds: 7.2e6 s
    times_sec = np.linspace(0, 7200000.0, 201)
    # Valley at t = 3.6e6 s (1000 h)
    sds_inv = 8000.0 - 2000.0 * np.sin(np.pi * times_sec / 7200000.0)
    df = pd.DataFrame({"time": times_sec, "sds.I[1]": sds_inv, "job_id": 1})
    # t_DD_end = 100.0 hours
    jobs_df = pd.DataFrame([{"job_id": 1, "name": "Case 2000h", "t_DD_end": 100.0}])

    cfg = {"scenario": {"time_base": "second", "stopTime": 2000.0}}
    metrics = extract_startup_inventory_metrics(df, jobs_df, config=cfg)

    assert len(metrics) == 1
    row = metrics.iloc[0]
    # Ignition delay must be exactly 100.0 h, NOT 100 / 3600 = 0.027 h!
    assert row["点火起始时间_h"] == pytest.approx(100.0)
    # Total duration must be 2000.0 h
    assert row["总仿真时长_h"] == pytest.approx(2000.0)
    # Absolute turning point at 1000.0 h:
    assert row["自持转折绝对时间_h"] == pytest.approx(1000.0)
    # Relative delay from ignition (1000.0 - 100.0 = 900.0 h):
    assert row["自持转折相对时滞_h"] == pytest.approx(900.0)


def test_resolve_plot_time_scale_always_hours():
    from tricys.visualizer.report.figures import resolve_plot_time_scale

    # 500 h run without time_base
    u, f = resolve_plot_time_scale(500.0, config={"simulation": {"stop_time": 500.0}})
    assert u == "h"
    assert f == 1.0

    # 12000 h run without time_base
    u, f = resolve_plot_time_scale(
        12000.0, config={"simulation": {"stop_time": 12000.0}}
    )
    assert u == "h"
    assert f == 1.0

    # 2000 h run in seconds (7.2e6 s)
    u, f = resolve_plot_time_scale(
        7200000.0, config={"scenario": {"time_base": "second", "stopTime": 2000.0}}
    )
    assert u == "h"
    assert f == 3600.0

    # Explicit hour model
    u, f = resolve_plot_time_scale(
        2000.0, config={"scenario": {"time_base": "hour", "stopTime": 2000.0}}
    )
    assert u == "h"
    assert f == 1.0
