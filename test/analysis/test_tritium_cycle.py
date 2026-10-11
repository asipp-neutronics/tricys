"""Unit tests for tritium_cycle physics post-processing engine."""

import numpy as np
import pandas as pd
import pytest

from tricys.analysis.tritium_cycle import (
    analyze_simulation_mass_conservation,
    analyze_subsystem_holdup_dynamics,
    derive_fusion_engineering_parameters,
    detect_sds_column,
    extract_startup_inventory_metrics,
    generate_tritium_mass_flows_proof_table,
)


@pytest.fixture
def synthetic_tfc_data():
    """Generates synthetic multi-job tritium fuel cycle dataset mimicking CFEDR 2000MW."""
    time = np.linspace(0, 1000, 201)

    # Case 1: Direct D-T (ignition at t=0)
    # SDS dips to valley around t=200, then recovers
    sds_c1 = 8000.0 - 6600.0 * (time / 200.0) * np.exp(1.0 - time / 200.0)
    # Subsystems paving
    ps_c1 = 735.0 * (1.0 - np.exp(-time / 0.17))
    tep_c1 = 1330.0 * (1.0 - np.exp(-time / 2.0))
    iiss_c1 = 1880.0 * (1.0 - np.exp(-time / 6.0))
    bz_c1 = 270.0 * (1.0 - np.exp(-time / 24.0))
    tes_c1 = 335.0 * (1.0 - np.exp(-time / 12.0))

    df1 = pd.DataFrame(
        {
            "time": time,
            "sds.inventory": sds_c1,
            "ps.inventory": ps_c1,
            "tep.inventory": tep_c1,
            "i_iss.inventory": iiss_c1,
            "bz.inventory": bz_c1,
            "tes.inventory": tes_c1,
            "job_id": 1,
        }
    )

    # Case 2: D-D phase up to t=200, ignition at t=200
    sds_c2 = np.where(
        time < 200.0,
        8000.0 + 3.0 * time,
        8600.0
        - 5900.0 * ((time - 200.0) / 200.0) * np.exp(1.0 - (time - 200.0) / 200.0),
    )
    df2 = pd.DataFrame(
        {
            "time": time,
            "sds.inventory": sds_c2,
            "ps.inventory": ps_c1,
            "tep.inventory": tep_c1,
            "i_iss.inventory": iiss_c1,
            "bz.inventory": bz_c1,
            "tes.inventory": tes_c1,
            "job_id": 2,
        }
    )

    results_df = pd.concat([df1, df2], ignore_index=True)
    jobs_df = pd.DataFrame(
        [
            {"job_id": 1, "name": "Case 1: Direct D-T", "t_DD_end": 0.0},
            {"job_id": 2, "name": "Case 2: 0.5 FPY D-D + D-T", "t_DD_end": 200.0},
        ]
    )

    return results_df, jobs_df


def test_derive_fusion_engineering_parameters():
    cfg = {
        "model": {
            "parameters": {
                "P_fus": 2000.0,
                "tep.DIR": 0.60,
                "plasma.Fueling_Efficiency": 0.50,
                "plasma.Burn_Fraction": 0.03,
                "pulse.burn_T": 2.0,
                "pulse.width": 80.0,
                "TBR": 1.10,
                "TBR_Required": 1.0482,
            }
        }
    }
    p = derive_fusion_engineering_parameters(cfg)
    assert p["P_fus"] == 2000.0
    assert p["duty"] == 0.8
    assert p["DIR"] == 0.6
    assert p["TBR_net"] == 1.10
    assert p["TBR_Required"] == 1.0482
    assert pytest.approx(p["Delta_TBR"], rel=1e-3) == 0.0518
    # Burn rate: 2000 * 6.3935e-3 = 12.787 g/h
    assert pytest.approx(p["m_burn_peak"], rel=1e-3) == 12.787
    # Fuel rate: 12.787 / 0.015 = 852.47 g/h
    assert pytest.approx(p["m_fuel_peak"], rel=1e-3) == 852.47
    # Period avg
    assert pytest.approx(p["m_burn_avg"], rel=1e-3) == 12.787 * 0.8


def test_detect_sds_column(synthetic_tfc_data):
    results_df, _ = synthetic_tfc_data
    col = detect_sds_column(results_df)
    assert col == "sds.inventory"


def test_extract_startup_inventory_metrics(synthetic_tfc_data):
    results_df, jobs_df = synthetic_tfc_data
    jobs_df_with_req = jobs_df.copy()
    jobs_df_with_req["TBR_Required"] = 1.0482
    metrics_df = extract_startup_inventory_metrics(results_df, jobs_df_with_req)

    assert len(metrics_df) == 2
    assert "必需首炉外供氚量_g" in metrics_df.columns
    assert "自持转折相对时滞_h" in metrics_df.columns
    assert "最低谷底存量I_min_g" in metrics_df.columns
    assert "必需产氚比_TBR_Required" in metrics_df.columns
    assert "自持增殖裕度_Delta_TBR" in metrics_df.columns

    # Case 1 valley around 200
    r1 = metrics_df[metrics_df["job_id"] == 1].iloc[0]
    assert r1["最低谷底存量I_min_g"] < 8000.0
    assert r1["必需首炉外供氚量_g"] > 5000.0
    assert r1["必需产氚比_TBR_Required"] == 1.0482
    assert pytest.approx(r1["自持转折相对时滞_h"], abs=10.0) == 200.0


def test_analyze_subsystem_holdup_dynamics(synthetic_tfc_data):
    results_df, _ = synthetic_tfc_data
    sub_df = analyze_subsystem_holdup_dynamics(results_df, target_job_id=1)

    assert not sub_df.empty
    assert "子系统" in sub_df.columns
    assert "t95稳态铺底时间_h" in sub_df.columns
    assert "稳态滞留均值_g" in sub_df.columns
    # Ensure fake constant 特征时滞_h was removed
    assert "特征时滞_h" not in sub_df.columns

    # When NO glossary is configured, columns strictly use original variable names
    sub_keys = sub_df["key"].tolist()
    assert "tep.inventory" in sub_keys
    assert "i_iss.inventory" in sub_keys
    # Subsystem display name also strictly matches original variable name
    sub_names = sub_df["子系统"].tolist()
    assert "tep.inventory" in sub_names


def test_glossary_mapping_mechanism(tmp_path):
    from tricys.analysis.tritium_cycle import (
        load_glossary_mapping,
        resolve_subsystem_metadata,
    )

    # 1. Without glossary: strictly return original name
    meta_none = resolve_subsystem_metadata("blanket.I[1]")
    assert meta_none["name"] == "blanket.I[1]"
    assert meta_none["short_name"] == "blanket.I[1]"

    # 2. With glossary CSV
    csv_file = tmp_path / "test_glossary.csv"
    csv_file.write_text(
        "模型参数 (Model Parameter),英文术语 (English Term),中文翻译 (Chinese Translation),别名 (Alias),说明 (Explanation)\n"
        "blanket.I[1],Blanket Tritium Inventory,增殖包层氚盘存量,Breeding Zone,增殖包层测试说明\n"
        "wds.threshold,WDS Threshold,水去氚系统铺底量,WDS Threshold,WDS测试说明\n",
        encoding="utf-8",
    )

    cfg = {"report": {"glossary_path": str(csv_file)}}
    g_map = load_glossary_mapping(config=cfg)
    assert "blanket.I[1]" in g_map
    assert g_map["blanket.I[1]"]["zh"] == "增殖包层氚盘存量"

    # Resolve with config
    meta_with_g = resolve_subsystem_metadata("blanket.I[1]", config=cfg)
    assert meta_with_g["name"] == "增殖包层氚盘存量 (blanket.I[1])"
    assert meta_with_g["short_name"] == "增殖包层氚盘存量"
    assert meta_with_g["desc"] == "增殖包层测试说明"

    # Parameter not in glossary falls back to original name
    meta_other = resolve_subsystem_metadata("unknown_var.x", config=cfg)
    assert meta_other["name"] == "unknown_var.x"


def test_generate_tritium_mass_flows_proof_table():
    proof_df = generate_tritium_mass_flows_proof_table(
        P_fus=2000.0, duty=0.80, DIR=0.60, eta_f=0.50, f_b=0.03, TBR_net=1.10
    )
    assert len(proof_df) == 13
    assert proof_df.iloc[0]["流路编号"] == "F01"
    assert proof_df.iloc[-1]["流路编号"] == "NET"

    # F01 fueling rate
    f01 = proof_df[proof_df["流路编号"] == "F01"].iloc[0]
    assert pytest.approx(f01["平顶瞬时流率_gh"], rel=1e-2) == 852.47
    assert pytest.approx(f01["放电周期平均流率_gh"], rel=1e-2) == 681.97

    # Net accum rate > 0
    net_row = proof_df[proof_df["流路编号"] == "NET"].iloc[0]
    assert "+" in str(net_row["平顶瞬时流率_gh"])


def test_analyze_simulation_mass_conservation(synthetic_tfc_data):
    results_df, _ = synthetic_tfc_data
    res = analyze_simulation_mass_conservation(results_df[results_df["job_id"] == 1])
    assert "max_relative_residual" in res
    assert "is_conserved" in res


def test_calculate_real_sensitivity_indices():
    from SALib.sample import sobol as sobol_sample

    from tricys.analysis.tritium_cycle import calculate_real_sensitivity_indices

    # Test 1: Authentic Sobol Saltelli sampling
    task_config = {
        "task": {
            "type": "sensitivity",
            "method": "sobol",
            "parameters": {
                "blanket.TBR": [1.05, 1.30],
                "wds.threshold": [800.0, 1400.0],
            },
            "metrics": {
                "final_storage": {"source": "sds.I[1]"},
            },
        }
    }
    problem = {
        "num_vars": 2,
        "names": ["blanket.TBR", "wds.threshold"],
        "bounds": [[1.05, 1.30], [800.0, 1400.0]],
    }
    # 16 samples * (2 + 2) = 64 runs for calc_second_order=False
    X = sobol_sample.sample(problem, 16, calc_second_order=False)
    jobs_df = pd.DataFrame(X, columns=["blanket.TBR", "wds.threshold"])
    jobs_df["job_id"] = np.arange(1, len(jobs_df) + 1)

    # Physical relation: final_storage strongly driven by TBR, unaffected by wds.threshold
    y = 5000.0 * (jobs_df["blanket.TBR"] - 1.05) ** 2
    metrics_df = pd.DataFrame({"job_id": jobs_df["job_id"], "final_storage": y})

    sens_df = calculate_real_sensitivity_indices(
        jobs_df=jobs_df,
        metrics_df=metrics_df,
        task_config=task_config,
    )
    assert not sens_df.empty
    assert "一阶敏感度 (S1)" in sens_df.columns
    assert "全阶敏感度 (ST)" in sens_df.columns
    assert "Sobol 全局方差分解" in sens_df.iloc[0]["分析方法"]
    # blanket.TBR must be ranked top
    assert sens_df.iloc[0]["敏感参数"] == "blanket.TBR"
    assert sens_df.iloc[0]["全阶敏感度 (ST)"] > 0.5
    # wds.threshold must have ST close to 0
    wds_row = sens_df[sens_df["敏感参数"] == "wds.threshold"].iloc[0]
    assert pytest.approx(wds_row["全阶敏感度 (ST)"], abs=0.01) == 0.0

    # Test 2: Fallback correlation / regression when not Sobol structure
    jobs_arb = pd.DataFrame(
        {
            "job_id": [1, 2, 3, 4],
            "blanket.TBR": [1.05, 1.10, 1.20, 1.30],
            "wds.threshold": [800.0, 900.0, 1000.0, 1100.0],
        }
    )
    metrics_arb = pd.DataFrame(
        {
            "job_id": [1, 2, 3, 4],
            "final_storage": [10.0, 20.0, 40.0, 60.0],
        }
    )
    sens_arb = calculate_real_sensitivity_indices(
        jobs_df=jobs_arb,
        metrics_df=metrics_arb,
        task_config={"task": {"type": "sweep"}},
    )
    assert not sens_arb.empty
    assert "Pearson 线性相关度 (r)" in sens_arb.columns
    assert "方差贡献率 (R²)" in sens_arb.columns
    assert "单参数回归与相关性分析" in sens_arb.iloc[0]["分析方法"]
    assert sens_arb.iloc[0]["方差贡献率 (R²)"] > 0.9


def test_extract_startup_inventory_metrics_hour_scaling():
    """Verify that OMC simulation output (seconds, up to 7.2e6 s) is accurately converted to hours."""
    # 2000 hours in seconds: 7,200,000 s
    times = np.linspace(0, 7200000.0, 101)
    sds_inv = 8000.0 - 1000.0 * np.sin(np.pi * times / 7200000.0)
    df = pd.DataFrame({"time": times, "sds.I[1]": sds_inv, "job_id": 1})
    jobs_df = pd.DataFrame([{"job_id": 1, "name": "Case 2000h", "t_DD_end": 0.0}])

    cfg = {"scenario": {"time_base": "hour", "stopTime": 2000.0}}
    metrics = extract_startup_inventory_metrics(df, jobs_df, config=cfg)

    assert len(metrics) == 1
    row = metrics.iloc[0]
    # Total simulation duration must be exactly 2000.0 hours, NOT 7,200,000 hours!
    assert row["总仿真时长_h"] == pytest.approx(2000.0, rel=1e-3)
    # Turning point at half-sine (1000.0 hours):
    assert row["自持转折相对时滞_h"] == pytest.approx(1000.0, rel=1e-3)


def test_plot_time_scale_resolution():
    """Verify resolve_plot_time_scale respects scenario.time_base."""
    from tricys.visualizer.report.figures import resolve_plot_time_scale

    # Hour config with 7.2e6 seconds
    unit_h, factor_h = resolve_plot_time_scale(
        7200000.0, config={"scenario": {"time_base": "hour"}}
    )
    assert unit_h == "h"
    assert factor_h == 3600.0

    # Second config
    unit_s, factor_s = resolve_plot_time_scale(
        1800.0, config={"scenario": {"time_base": "second"}}
    )
    assert unit_s == "s"
    assert factor_s == 1.0
