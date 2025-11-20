import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tricys.analysis.report import (
    consolidate_reports,
    generate_analysis_cases_summary,
    generate_prompt_templates,
)

TEST_DIR = "temp_report_test"


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def create_dummy_results(
    case_dir: Path, case_name: str, independent_variable: str = "plasma.fb"
):
    results_dir = case_dir / "results"
    os.makedirs(results_dir, exist_ok=True)

    # Create dummy summary csv
    summary_df = pd.DataFrame(
        {
            independent_variable: [0.08, 0.09, 0.10],
            "Startup_Inventory": [20, 18, 16],
            "Self_Sufficiency_Time": [300, 250, 200],
        }
    )
    summary_df.to_csv(results_dir / "sensitivity_analysis_summary.csv", index=False)

    # Create dummy plot svg
    (results_dir / f"line_Startup_Inventory_vs_{independent_variable}_zh.svg").touch()

    # Create dummy sweep results csv
    sweep_df = pd.DataFrame(
        {
            "time": [0, 10, 20],
            f"sds.I[1]&{independent_variable}=0.08": [100, 90, 80],
            f"sds.I[1]&{independent_variable}=0.09": [110, 100, 90],
            f"sds.I[1]&{independent_variable}=0.10": [120, 110, 100],
        }
    )
    sweep_df.to_csv(results_dir / "sweep_results.csv", index=False)


def test_generate_analysis_cases_summary():
    """Tests the generate_analysis_cases_summary function."""
    run_workspace = Path(TEST_DIR).resolve() / "20250101_120000"
    os.makedirs(run_workspace, exist_ok=True)

    # Setup for case 1
    case1_dir = run_workspace / "Case_A"
    os.makedirs(case1_dir, exist_ok=True)
    create_dummy_results(case1_dir, "Case_A", independent_variable="plasma.fb")
    config1_path = case1_dir / "config.json"

    original_config = {
        "run_timestamp": "20250101_120000",
        "sensitivity_analysis": {
            "metrics_definition": {
                "Startup_Inventory": {
                    "source_column": "sds.I[1]",
                    "method": "calculate_startup_inventory",
                },
                "Self_Sufficiency_Time": {
                    "source_column": "sds.I[1]",
                    "method": "time_of_turning_point",
                },
            }
        },
    }

    case1_data = {
        "name": "Case_A",
        "independent_variable": "plasma.fb",
        "independent_variable_sampling": [0.08, 0.09, 0.10],
        "dependent_variables": ["Startup_Inventory", "Self_Sufficiency_Time"],
        "sweep_time": ["sds.I[1]"],
    }

    with open(config1_path, "w") as f:
        json.dump(
            {
                "simulation_parameters": {},
                "sensitivity_analysis": {"analysis_case": case1_data},
            },
            f,
        )

    case_configs = [
        {
            "index": 0,
            "workspace": str(case1_dir),
            "config_path": str(config1_path),
            "case_data": case1_data,
            "config": json.loads(config1_path.read_text()),
        },
    ]

    # Change CWD to the test dir so the report is generated there
    original_cwd = os.getcwd()
    # The function under test creates the timestamp dir inside the CWD.
    # So we chdir to TEST_DIR to have the 2025... dir created there.
    os.chdir(TEST_DIR)
    try:
        generate_analysis_cases_summary(case_configs, original_config)

        report_path = (
            Path(original_config["run_timestamp"])
            / f"execution_report_{original_config['run_timestamp']}.md"
        )
        assert report_path.exists()

        content = report_path.read_text(encoding="utf-8")
        assert "Analysis Cases Execution Report" in content
        assert "Case_A" in content
        assert "✓ Success" in content
    finally:
        os.chdir(original_cwd)


@pytest.mark.build_test
def test_generate_prompt_templates_and_consolidate_reports():
    """Tests the report generation and consolidation process."""
    run_workspace = Path(TEST_DIR) / "20250101_120000"
    os.makedirs(run_workspace, exist_ok=True)

    # Setup for case 1
    case1_dir = run_workspace / "Case_A"
    os.makedirs(case1_dir, exist_ok=True)
    create_dummy_results(case1_dir, "Case_A", independent_variable="plasma.fb")
    config1_path = case1_dir / "config.json"

    # Setup for case 2
    case2_dir = run_workspace / "Case_B"
    os.makedirs(case2_dir, exist_ok=True)
    create_dummy_results(case2_dir, "Case_B", independent_variable="i_iss.T")
    config2_path = case2_dir / "config.json"

    original_config = {
        "run_timestamp": "20250101_120000",
        "sensitivity_analysis": {
            "metrics_definition": {
                "Startup_Inventory": {
                    "source_column": "sds.I[1]",
                    "method": "calculate_startup_inventory",
                },
                "Self_Sufficiency_Time": {
                    "source_column": "sds.I[1]",
                    "method": "time_of_turning_point",
                },
            }
        },
    }

    case1_data = {
        "name": "Case_A",
        "independent_variable": "plasma.fb",
        "independent_variable_sampling": [0.08, 0.09, 0.10],
        "dependent_variables": ["Startup_Inventory", "Self_Sufficiency_Time"],
        "sweep_time": ["sds.I[1]"],
    }

    case2_data = {
        "name": "Case_B",
        "independent_variable": "i_iss.T",
        "independent_variable_sampling": [18.0, 19.0],
        "dependent_variables": ["Startup_Inventory"],
    }

    with open(config1_path, "w") as f:
        json.dump(
            {
                "simulation_parameters": {},
                "sensitivity_analysis": {"analysis_case": case1_data},
            },
            f,
        )
    with open(config2_path, "w") as f:
        json.dump(
            {
                "simulation_parameters": {},
                "sensitivity_analysis": {"analysis_case": case2_data},
            },
            f,
        )

    case_configs = [
        {
            "index": 0,
            "workspace": str(case1_dir),
            "config_path": str(config1_path),
            "case_data": case1_data,
            "config": json.loads(config1_path.read_text()),
        },
        {
            "index": 1,
            "workspace": str(case2_dir),
            "config_path": str(config2_path),
            "case_data": case2_data,
            "config": json.loads(config2_path.read_text()),
        },
    ]

    # --- Test generate_prompt_templates ---
    generate_prompt_templates(case_configs, original_config)

    report1_path = case1_dir / "results" / "analysis_report_Case_A.md"
    report2_path = case2_dir / "results" / "analysis_report_Case_B.md"

    assert report1_path.exists()
    assert report2_path.exists()

    report1_content = report1_path.read_text(
        encoding="utf-8"
    ).lower()  # Use lower for case-insensitive check
    assert "# plasma fb 敏感性分析报告" in report1_content
    assert "## 关键动态数据切片：过程数据" in report1_content

    # --- Test consolidate_reports ---
    consolidate_reports(case_configs, original_config)

    # Check that files were moved
    assert not report1_path.exists()
    assert (case1_dir / "report" / "analysis_report_Case_A.md").exists()
    assert not (
        case1_dir / "results" / "line_Startup_Inventory_vs_plasma.fb_zh.svg"
    ).exists()
    assert (
        case1_dir / "report" / "line_Startup_Inventory_vs_plasma.fb_zh.svg"
    ).exists()

    assert not report2_path.exists()
    assert (case2_dir / "report" / "analysis_report_Case_B.md").exists()
