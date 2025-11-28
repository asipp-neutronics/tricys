import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tricys.analysis.plot import (
    generate_analysis_plots,
    load_glossary,
    plot_sweep_time_series,
    set_plot_language,
)

TEST_DIR = "temp_plot_test"


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Set up and tear down the test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def create_dummy_glossary(path: Path) -> None:
    """Create a dummy glossary file."""
    glossary_data = {
        "模型参数 (Model Parameter)": [
            "sds.I[1]",
            "plasma.fb",
            "i_iss.T",
            "Startup_Inventory",
            "Self_Sufficiency_Time",
        ],
        "英文术语 (English Term)": [
            "SDS Inventory",
            "Plasma FB",
            "I_ISS Temperature",
            "Startup Inventory",
            "Self-Sufficiency Time",
        ],
        "中文翻译 (Chinese Translation)": [
            "SDS库存",
            "等离子体FB",
            "I_ISS温度",
            "启动库存",
            "自持时间",
        ],
    }
    df = pd.DataFrame(glossary_data)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def test_set_plot_language():
    """Test the set_plot_language function."""
    # Test setting to Chinese
    set_plot_language("cn")
    import matplotlib.pyplot as plt

    assert "SimHei" in plt.rcParams["font.sans-serif"]
    assert plt.rcParams["axes.unicode_minus"] is False

    # Test setting back to English
    set_plot_language("en")
    assert plt.rcParams["font.sans-serif"] == plt.rcParamsDefault["font.sans-serif"]
    assert (
        plt.rcParams["axes.unicode_minus"] is plt.rcParamsDefault["axes.unicode_minus"]
    )


def test_load_glossary():
    """Test the load_glossary function."""
    glossary_path = Path(TEST_DIR) / "glossary.csv"
    create_dummy_glossary(glossary_path)

    english_map, chinese_map = load_glossary(str(glossary_path))

    assert "sds.I[1]" in english_map
    assert english_map["sds.I[1]"] == "SDS Inventory"
    assert "sds.I[1]" in chinese_map
    assert chinese_map["sds.I[1]"] == "SDS库存"

    # Test with non-existent file
    english_map, chinese_map = load_glossary("non_existent_file.csv")
    assert not english_map
    assert not chinese_map


def test_generate_analysis_plots():
    """Test the generate_analysis_plots function."""
    summary_df = pd.DataFrame(
        {
            "plasma.fb": [0.08, 0.09, 0.10],
            "i_iss.T": [18.0, 19.0, 20.0],
            "Startup_Inventory": [20, 18, 16],
            "Self_Sufficiency_Time": [300, 250, 200],
            "Required_TBR": [1.05, 1.06, 1.07],
        }
    )
    analysis_case = {
        "name": "Test Case",
        "independent_variable": "plasma.fb",
        "dependent_variables": [
            "Startup_Inventory",
            "Self_Sufficiency_Time",
            "Required_TBR",
        ],
        "default_simulation_values": {"i_iss.T": 19.0},
    }
    glossary_path = Path(TEST_DIR) / "glossary.csv"
    create_dummy_glossary(glossary_path)

    plot_paths = generate_analysis_plots(
        summary_df, analysis_case, TEST_DIR, glossary_path=str(glossary_path)
    )

    assert len(plot_paths) > 0
    for path in plot_paths:
        assert os.path.exists(path)


def test_plot_sweep_time_series():
    """Test the plot_sweep_time_series function."""
    sweep_df = pd.DataFrame(
        {
            "time": range(10),
            "sds.I[1]&plasma.fb=0.08": range(10, 20),
            "sds.I[1]&plasma.fb=0.09": range(20, 30),
        }
    )
    csv_path = Path(TEST_DIR) / "sweep_results.csv"
    sweep_df.to_csv(csv_path, index=False)

    plot_paths = plot_sweep_time_series(
        csv_path=str(csv_path),
        save_dir=TEST_DIR,
        y_var_name="sds.I[1]",
        independent_var_name="plasma.fb",
    )

    assert len(plot_paths) > 0
    for path in plot_paths:
        assert os.path.exists(path)
