import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = "temp_i_iss_handler_test"
TEMP_HANDLER_DIR = os.path.join(TEST_DIR, "temp_handler")


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Set up and tear down the test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEMP_HANDLER_DIR)

    # Add temp dir to path to allow importing from it
    abs_test_dir = os.path.abspath(TEST_DIR)
    sys.path.insert(0, abs_test_dir)

    # Create an __init__.py in TEST_DIR to make it a package
    (Path(TEST_DIR) / "__init__.py").touch()

    yield

    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    sys.path.remove(abs_test_dir)


def test_run_dummy_simulation():
    """Test the run_dummy_simulation function."""
    import importlib.util

    # Create a dummy handler and csv in a temporary directory
    dummy_handler_content = """
import os
import pandas as pd

def run_dummy_simulation(temp_input_csv: str, temp_output_csv: str, **kwargs) -> dict:
    handler_dir = os.path.dirname(__file__)
    source_csv_path = os.path.join(handler_dir, "i_iss_handler.csv")

    try:
        source_df = pd.read_csv(source_csv_path)
    except FileNotFoundError:
        pd.DataFrame({"time": []}).to_csv(temp_output_csv, index=False)
        raise

    columns_to_select = [
        "time",
        "i_iss.to_SDS[1]",
        "i_iss.to_SDS[2]",
        "i_iss.to_SDS[3]",
        "i_iss.to_SDS[4]",
        "i_iss.to_SDS[5]",
        "i_iss.to_WDS[1]",
        "i_iss.to_WDS[2]",
        "i_iss.to_WDS[3]",
        "i_iss.to_WDS[4]",
        "i_iss.to_WDS[5]",
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
    output_placeholder = {
        "to_SDS": "{1,2,3,4,5,6}",
        "to_WDS": "{1,7,8,9,10,11}",
    }
    return output_placeholder
"""

    dummy_csv_content = """time,i_iss.to_SDS[1],i_iss.to_SDS[2],i_iss.to_SDS[3],i_iss.to_SDS[4],i_iss.to_SDS[5],i_iss.to_WDS[1],i_iss.to_WDS[2],i_iss.to_WDS[3],i_iss.to_WDS[4],i_iss.to_WDS[5],other_col
0,1,2,3,4,5,6,7,8,9,10,11
1,1,2,3,4,5,6,7,8,9,10,11
"""

    handler_path = Path(TEMP_HANDLER_DIR) / "i_iss_handler.py"
    csv_path = Path(TEMP_HANDLER_DIR) / "i_iss_handler.csv"

    handler_path.write_text(dummy_handler_content, encoding="utf-8")
    csv_path.write_text(dummy_csv_content, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("i_iss_handler", handler_path)
    dummy_handler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dummy_handler)

    temp_input_csv = os.path.join(TEST_DIR, "input.csv")
    temp_output_csv = os.path.join(TEST_DIR, "output.csv")

    result = dummy_handler.run_dummy_simulation(temp_input_csv, temp_output_csv)

    assert os.path.exists(temp_output_csv)
    output_df = pd.read_csv(temp_output_csv)
    assert len(output_df) == 2
    assert "other_col" not in output_df.columns
    assert result == {
        "to_SDS": "{1,2,3,4,5,6}",
        "to_WDS": "{1,7,8,9,10,11}",
    }
