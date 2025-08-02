import json
import os
import sqlite3

import pytest

from tricys.manager.config_manager import config_manager
from tricys.utils.db_utils import (
    create_parameters_table,
    get_db_path,
    get_parameters_from_db,
    store_parameters_in_db,
    update_sweep_values_in_db,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
log_dir = os.path.join(PROJECT_ROOT, config_manager.get("paths.log_dir"))
db_path = os.path.join(PROJECT_ROOT, config_manager.get("paths.db_path"))

# Sample data for testing
SAMPLE_PARAMS = [
    {
        "name": "coolant_pipe.to_CPS_Fraction",
        "type": "Real",
        "defaultValue": "1e-2",
        "sweep_values": "",
        "comment": "",
        "dimensions": "()",
    },
    {
        "name": "coolant_pipe.to_FW_Fraction",
        "type": "Real",
        "defaultValue": "0.6",
        "sweep_values": "",
        "comment": "",
        "dimensions": "()",
    },
]


def test_get_db_path():
    assert get_db_path() == "/tricys/data/parameters.db"


def test_create_parameters_table():
    """Tests the creation of the parameters table in a real database file."""
    try:
        (
            os.remove("/tricys/data/parameters.db")
            if os.path.exists("/tricys/data/parameters.db")
            else None
        )
        create_parameters_table()
        assert os.path.exists("/tricys/data/parameters.db")
        with sqlite3.connect("/tricys/data/parameters.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='parameters'"
            )
            assert (
                cursor.fetchone() is not None
            ), "The 'parameters' table was not created."
            (
                os.remove("/tricys/data/parameters.db")
                if os.path.exists("/tricys/data/parameters.db")
                else None
            )
            os.system(f"rm -rf {log_dir}")
    except sqlite3.OperationalError as e:
        pytest.fail(f"Database error occurred: {e}")


def test_store_and_get_parameters():
    """Tests storing and retrieving parameters from a real database file."""
    try:
        (
            os.remove("/tricys/data/parameters.db")
            if os.path.exists("/tricys/data/parameters.db")
            else None
        )
        create_parameters_table()
        store_parameters_in_db(SAMPLE_PARAMS)
        params = get_parameters_from_db()
        assert len(params) == 2
        assert params["coolant_pipe.to_CPS_Fraction"]["default_value"] == "1e-2"
        assert params["coolant_pipe.to_FW_Fraction"]["default_value"] == "0.6"
        (
            os.remove("/tricys/data/parameters.db")
            if os.path.exists("/tricys/data/parameters.db")
            else None
        )
        os.system(f"rm -rf {log_dir}")
    except sqlite3.OperationalError as e:
        pytest.fail(f"Database error occurred: {e}")


def test_update_sweep_values():
    """Tests updating sweep values for a parameter in a real database file."""
    try:
        (
            os.remove("/tricys/data/parameters.db")
            if os.path.exists("/tricys/data/parameters.db")
            else None
        )
        create_parameters_table()
        store_parameters_in_db(SAMPLE_PARAMS)

        sweep_values = [1.0, 2.0, 3.0]
        update_sweep_values_in_db({"coolant_pipe.to_CPS_Fraction": sweep_values})

        with sqlite3.connect("/tricys/data/parameters.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sweep_values FROM parameters WHERE name=?",
                ("coolant_pipe.to_CPS_Fraction",),
            )
            result = cursor.fetchone()
            assert result is not None
            assert json.loads(result[0]) == sweep_values
        (
            os.remove("/tricys/data/parameters.db")
            if os.path.exists("/tricys/data/parameters.db")
            else None
        )
        os.system(f"rm -rf {log_dir}")
    except sqlite3.OperationalError as e:
        pytest.fail(f"Database error occurred: {e}")
