import json
import os
import shutil
import sqlite3
import gc

import pytest

from tricys.utils.db_utils import (
    create_parameters_table,
    get_parameters_from_db,
    store_parameters_in_db,
    update_sweep_values_in_db,
)

# Define an output directory and DB path based on the test file's name
OUTPUT_DIR = os.path.splitext(__file__)[0]
DB_PATH = os.path.join(OUTPUT_DIR, "test.db")


# Sample data for testing
SAMPLE_PARAMS = [
    {
        "name": "coolant_pipe.to_CPS_Fraction",
        "type": "Real",
        "defaultValue": "1e-2",
        "comment": "",
        "dimensions": "()",
    },
    {
        "name": "coolant_pipe.to_FW_Fraction",
        "type": "Real",
        "defaultValue": "0.6",
        "comment": "",
        "dimensions": "()",
    },
]


def setup_and_teardown(request):
    """Fixture to create and cleanup the test output directory for each test."""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    def cleanup():
        if getattr(request.node, "test_passed", False):
            print(f"\nTest passed. Cleaning up test directory: {OUTPUT_DIR}")
            gc.collect()  # Ensure all file handles are released
            if os.path.exists(OUTPUT_DIR):
                shutil.rmtree(OUTPUT_DIR)
        else:
            print(f"\nTest failed. Intermediate files kept at: {OUTPUT_DIR}")

    request.addfinalizer(cleanup)

@pytest.mark.build_test
def test_create_parameters_table(request):
    """Tests the creation of the parameters table."""
    setup_and_teardown(request)
    try:
        create_parameters_table(DB_PATH)
        assert os.path.exists(DB_PATH)
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='parameters'"
            )
            assert (
                cursor.fetchone() is not None
            ), "The 'parameters' table was not created."
    except sqlite3.OperationalError as e:
        pytest.fail(f"Database error occurred: {e}")
    request.node.test_passed = True

@pytest.mark.build_test
def test_store_and_get_parameters(request):
    """Tests storing and retrieving parameters."""
    setup_and_teardown(request)
    try:
        create_parameters_table(DB_PATH)
        store_parameters_in_db(DB_PATH, SAMPLE_PARAMS)

        params = get_parameters_from_db(DB_PATH)
        assert len(params) == 2

        params_dict = {p["name"]: p for p in params}

        assert params_dict["coolant_pipe.to_CPS_Fraction"]["default_value"] == "1e-2"
        assert params_dict["coolant_pipe.to_FW_Fraction"]["default_value"] == "0.6"
    except sqlite3.OperationalError as e:
        pytest.fail(f"Database error occurred: {e}")
    request.node.test_passed = True

@pytest.mark.build_test
def test_update_sweep_values(request):
    """Tests updating sweep values for a parameter."""
    setup_and_teardown(request)
    try:
        create_parameters_table(DB_PATH)
        store_parameters_in_db(DB_PATH, SAMPLE_PARAMS)

        sweep_values = [1.0, 2.0, 3.0]
        update_sweep_values_in_db(
            DB_PATH, {"coolant_pipe.to_CPS_Fraction": sweep_values}
        )

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sweep_values FROM parameters WHERE name=?",
                ("coolant_pipe.to_CPS_Fraction",),
            )
            result = cursor.fetchone()
            assert result is not None
            assert json.loads(result[0]) == sweep_values
    except sqlite3.OperationalError as e:
        pytest.fail(f"Database error occurred: {e}")
    request.node.test_passed = True