import json
import os
import sqlite3

import pytest

from tricys.utils.db_utils import (
    create_parameters_table,
    get_parameters_from_db,
    store_parameters_in_db,
    update_sweep_values_in_db,
)

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


@pytest.fixture
def db_path(tmp_path):
    """Provides a temporary path for the test database."""
    return tmp_path / "test.db"


def test_create_parameters_table(db_path):
    """Tests the creation of the parameters table."""
    try:
        create_parameters_table(db_path)
        assert os.path.exists(db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='parameters'"
            )
            assert (
                cursor.fetchone() is not None
            ), "The 'parameters' table was not created."
    except sqlite3.OperationalError as e:
        pytest.fail(f"Database error occurred: {e}")


def test_store_and_get_parameters(db_path):
    """Tests storing and retrieving parameters."""
    try:
        create_parameters_table(db_path)
        store_parameters_in_db(db_path, SAMPLE_PARAMS)

        # Assuming get_parameters_from_db returns a list of dicts
        params = get_parameters_from_db(db_path)
        assert len(params) == 2

        # Convert list to dict for easier assertion, matching old test logic
        params_dict = {p["name"]: p for p in params}

        assert params_dict["coolant_pipe.to_CPS_Fraction"]["default_value"] == "1e-2"
        assert params_dict["coolant_pipe.to_FW_Fraction"]["default_value"] == "0.6"
    except sqlite3.OperationalError as e:
        pytest.fail(f"Database error occurred: {e}")


def test_update_sweep_values(db_path):
    """Tests updating sweep values for a parameter."""
    try:
        create_parameters_table(db_path)
        store_parameters_in_db(db_path, SAMPLE_PARAMS)

        sweep_values = [1.0, 2.0, 3.0]
        update_sweep_values_in_db(
            db_path, {"coolant_pipe.to_CPS_Fraction": sweep_values}
        )

        with sqlite3.connect(db_path) as conn:
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
