import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = "temp_div_handler_test"
TEMP_HANDLER_DIR = os.path.join(TEST_DIR, "temp_handler")


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Set up and tear down the test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEMP_HANDLER_DIR)

    # Add temp dir to path to allow importing from it
    sys.path.insert(0, TEST_DIR)

    yield

    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    sys.path.remove(TEST_DIR)


def test_run_div_simulation():
    """Test the run_div_simulation function."""

    # Create a dummy handler and csv in a temporary directory
    dummy_handler_content = """
import os
import pandas as pd

def run_div_simulation(temp_input_csv: str, temp_output_csv: str, **kwargs) -> dict:
    handler_dir = os.path.dirname(__file__)
    source_csv_path = os.path.join(handler_dir, "div_handler.csv")

    try:
        source_df = pd.read_csv(source_csv_path)
    except FileNotFoundError:
        pd.DataFrame({"time": []}).to_csv(temp_output_csv, index=False)
        raise

    columns_to_select = [
        "time",
        "div.to_CL[1]",
        "div.to_CL[2]",
        "div.to_CL[3]",
        "div.to_CL[4]",
        "div.to_CL[5]",
    ]

    if not all(col in source_df.columns for col in columns_to_select):
        missing_cols = [
            col for col in columns_to_select if col not in source_df.columns
        ]
        raise ValueError(
            f"The source file {source_csv_path} is missing required columns: "
            f"{missing_cols}"
        )

    output_df = source_df[columns_to_select].copy()
    output_df.to_csv(temp_output_csv, index=False)
    output_placeholder = {"to_CL": "{1,2,3,4,5,6}"}
    return output_placeholder
"""

    dummy_csv_content = """time,div.to_CL[1],div.to_CL[2],div.to_CL[3],div.to_CL[4],div.to_CL[5],other_col
0,0.1,0.4,0.7,1.0,1.3,9
1,0.2,0.5,0.8,1.1,1.4,8
2,0.3,0.6,0.9,1.2,1.5,7
"""

    (Path(TEMP_HANDLER_DIR) / "__init__.py").touch()
    (Path(TEMP_HANDLER_DIR) / "div_handler.py").write_text(
        dummy_handler_content, encoding="utf-8"
    )
    (Path(TEMP_HANDLER_DIR) / "div_handler.csv").write_text(
        dummy_csv_content, encoding="utf-8"
    )

    from temp_handler.div_handler import run_div_simulation as dummy_run_div_simulation

    temp_input_csv = os.path.join(TEST_DIR, "input.csv")
    temp_output_csv = os.path.join(TEST_DIR, "output.csv")

    result = dummy_run_div_simulation(temp_input_csv, temp_output_csv)

    assert os.path.exists(temp_output_csv)
    output_df = pd.read_csv(temp_output_csv)
    assert len(output_df) == 3
    assert "other_col" not in output_df.columns
    assert result == {"to_CL": "{1,2,3,4,5,6}"}
