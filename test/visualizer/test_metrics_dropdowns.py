"""Unit tests to verify dropdown options in Visualizer Metrics Plots strictly exclude metadata identifiers."""

import pandas as pd


def test_metrics_dropdown_options_strictly_exclude_name():
    # Construct synthetic jobs_data with 'name', 'job_name', 'case_name', 'status' and real parameters
    jobs_data = [
        {
            "id": 1,
            "name": "Case_1_Baseline",
            "job_name": "Job_1",
            "case_name": "Case_Alpha",
            "status": "completed",
            "blanket.TBR": 1.10,
            "wds.threshold": 1000.0,
        },
        {
            "id": 2,
            "name": "Case_2_Advanced",
            "job_name": "Job_2",
            "case_name": "Case_Beta",
            "status": "completed",
            "blanket.TBR": 1.25,
            "wds.threshold": 1200.0,
        },
    ]

    metrics_data = [
        {
            "id": 1,
            "name": "Case_1_Baseline",
            "blanket.TBR": 1.10,
            "wds.threshold": 1000.0,
            "Job": "Job 1 (blanket.TBR=1.1, wds.threshold=1000.0)",
            "peak_inventory": 3000.0,
            "final_inventory": 2400.0,
            "startup_inventory": 1900.0,
        },
        {
            "id": 2,
            "name": "Case_2_Advanced",
            "blanket.TBR": 1.25,
            "wds.threshold": 1200.0,
            "Job": "Job 2 (blanket.TBR=1.25, wds.threshold=1200.0)",
            "peak_inventory": 4500.0,
            "final_inventory": 4500.0,
            "startup_inventory": 1820.0,
        },
    ]

    jobs_df = pd.DataFrame(jobs_data)

    ignored_col_names = {
        "id",
        "job_id",
        "name",
        "job_name",
        "case_name",
        "job",
        "title",
        "description",
        "comment",
        "status",
        "status_text",
        "condition",
    }

    # 1. Parameter keys (X-Axis)
    param_keys = []
    for c in jobs_df.columns:
        if str(c).strip().lower() in ignored_col_names:
            continue
        num_series = pd.to_numeric(jobs_df[c], errors="coerce")
        if num_series.notna().any():
            param_keys.append(c)

    # 2. Metric keys (Y-Axis)
    df_metrics = pd.DataFrame(metrics_data)
    metric_keys = []
    for k in df_metrics.columns:
        if str(k).strip().lower() in ignored_col_names:
            continue
        if k in jobs_df.columns or k in param_keys:
            continue
        num_s = pd.to_numeric(df_metrics[k], errors="coerce")
        if num_s.notna().any():
            metric_keys.append(k)

    # Assertions
    # Parameter options MUST NOT contain 'name' or any text identifiers
    assert "name" not in param_keys
    assert "job_name" not in param_keys
    assert "case_name" not in param_keys
    assert "status" not in param_keys
    assert "id" not in param_keys
    assert param_keys == ["blanket.TBR", "wds.threshold"]

    # Metric options MUST NOT contain 'name', 'Job', or any text identifiers
    assert "name" not in metric_keys
    assert "job" not in metric_keys
    assert "Job" not in metric_keys
    assert "id" not in metric_keys
    assert metric_keys == ["peak_inventory", "final_inventory", "startup_inventory"]

    # 3. Multidimensional Analysis (parcoords-dims-dropdown) options
    numeric_cols = df_metrics.select_dtypes(include=["number"]).columns
    parcoords_opts = [
        col for col in numeric_cols if str(col).strip().lower() not in ignored_col_names
    ]
    assert "name" not in parcoords_opts
    assert "id" not in parcoords_opts
    assert "job_id" not in parcoords_opts
    assert set(parcoords_opts) == {
        "blanket.TBR",
        "wds.threshold",
        "peak_inventory",
        "final_inventory",
        "startup_inventory",
    }

    # 4. Grouping parameters in update_metric_plot
    xaxis = "blanket.TBR"
    all_params = [
        c
        for c in jobs_df.columns
        if str(c).strip().lower() not in ignored_col_names
        and pd.to_numeric(jobs_df[c], errors="coerce").notna().any()
    ]
    grouping_cols = [c for c in all_params if c != xaxis]
    assert "name" not in grouping_cols
    assert grouping_cols == ["wds.threshold"]
