"""Utilities for interacting with the simulation parameter SQLite database.

This module provides functions to create, store, update, and retrieve simulation
parameter data from a SQLite database file.
"""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def create_parameters_table(db_path: str) -> None:
    """Creates the parameters table in the database if it does not exist.

    Args:
        db_path: The path to the SQLite database file.

    Raises:
        sqlite3.Error: If a database error occurs during table creation.

    Note:
        Creates parent directories if they don't exist. Table schema includes:
        name (TEXT PRIMARY KEY), type, default_value, sweep_values, description, dimensions.
        Uses CREATE TABLE IF NOT EXISTS for safe repeated calls.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    logger.debug(f"Ensuring 'parameters' table exists in {db_path}")
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS parameters (
                    name TEXT PRIMARY KEY,
                    type TEXT,
                    default_value TEXT,
                    sweep_values TEXT,
                    description TEXT,
                    dimensions TEXT
                )
            """
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error while creating table: {e}", exc_info=True)
        raise


def store_parameters_in_db(db_path: str, params_data: List[Dict[str, Any]]) -> None:
    """Stores or replaces a list of parameter details in the database.

    Args:
        db_path: The path to the SQLite database file.
        params_data: A list of dictionaries, where each dictionary contains
            details for a single parameter.

    Raises:
        sqlite3.Error: If a database error occurs during insertion.

    Note:
        Uses INSERT OR REPLACE for upsert behavior. JSON-encodes defaultValue
        and stores dimensions with '()' default. Skips parameters without names.
        Expected param dict keys: name, type, defaultValue, comment, dimensions.
    """
    logger.info(f"Storing {len(params_data)} parameters into '{db_path}'")
    if not params_data:
        logger.warning("Parameter data is empty, nothing to store.")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for param in params_data:
                name = param.get("name")
                if not name:
                    continue

                value_json = json.dumps(param.get("defaultValue"))
                dimensions = param.get(
                    "dimensions", "()"
                )  # Default to '()' if not present

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO parameters (name, type, default_value, sweep_values, description, dimensions)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        name,
                        param.get("type", "Real"),
                        value_json,
                        None,
                        param.get("comment", ""),
                        dimensions,
                    ),
                )
            conn.commit()
        logger.info("Successfully stored/updated parameters in the database.")
    except sqlite3.Error as e:
        logger.error(f"Database error while storing parameters: {e}", exc_info=True)
        raise


def update_sweep_values_in_db(db_path: str, param_sweep: Dict[str, Any]) -> None:
    """Updates the 'sweep_values' for specified parameters in the database.

    Args:
        db_path: The path to the SQLite database file.
        param_sweep: A dictionary where keys are parameter names and values are
            the corresponding sweep values (e.g., a list).

    Raises:
        sqlite3.Error: If a database error occurs during the update.

    Note:
        Converts numpy arrays to lists before JSON encoding. Warns if parameter
        not found in database. Uses UPDATE statement so parameters must exist
        before calling this function.
    """
    logger.info(f"Updating sweep values in '{db_path}'")
    if not param_sweep:
        logger.warning("param_sweep dictionary is empty. No values to update.")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for param_name, sweep_values in param_sweep.items():
                if isinstance(sweep_values, np.ndarray):
                    sweep_values = sweep_values.tolist()

                sweep_values_json = json.dumps(sweep_values)

                cursor.execute(
                    """
                    UPDATE parameters SET sweep_values = ? WHERE name = ?
                """,
                    (sweep_values_json, param_name),
                )

                if cursor.rowcount == 0:
                    logger.warning(
                        f"Parameter '{param_name}' not found in database. No sweep value updated."
                    )
            conn.commit()
        logger.info("Sweep values updated successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database error while updating sweep values: {e}", exc_info=True)
        raise


def get_parameters_from_db(db_path: str) -> List[Dict[str, Any]]:
    """Retrieves parameter details from the database.

    Args:
        db_path: The path to the SQLite database file.

    Returns:
        A list of parameter dictionaries, each containing the name, default_value,
        description, and sweep_values.

    Note:
        JSON-decodes stored values. Returns empty string for sweep_values if None.
        Result dict keys: name, default_value, description, sweep_values.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, default_value, description, sweep_values FROM parameters"
        )
        params = []
        for name, default_value, description, sweep_values in cursor.fetchall():
            params.append(
                {
                    "name": name,
                    "default_value": json.loads(default_value),
                    "description": description,
                    "sweep_values": json.loads(sweep_values) if sweep_values else "",
                }
            )
    return params
