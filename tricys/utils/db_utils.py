"""本模块提供与SQLite数据库交互的实用功能。"""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List

import numpy as np

from tricys.manager.config_manager import config_manager

logger = logging.getLogger(__name__)


def get_db_path() -> str:
    """从配置中构建数据库的绝对路径。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    db_relative_path = config_manager.get("paths.db_path")
    if not db_relative_path:
        raise ValueError("Database path is not defined in the configuration.")
    return os.path.join(project_root, db_relative_path)


def create_parameters_table() -> None:
    """如果数据库中不存在参数表，则创建它。"""
    db_path = get_db_path()
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


def store_parameters_in_db(params_data: List[Dict[str, Any]]) -> None:
    """
    在数据库中存储或替换参数详细信息列表。

    参数:
        params_data (List[Dict[str, Any]]): 参数详细信息字典的列表（由om_utils.get_all_parameters_details返回）。
    """
    db_path = get_db_path()
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


def update_sweep_values_in_db(param_sweep: Dict[str, Any]) -> None:
    """
    更新数据库中指定参数的“sweep_values”。

    参数:
        param_sweep (Dict[str, Any]): 一个字典，其中键是参数名称，值是扫描值列表。
    """
    db_path = get_db_path()
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


def get_parameters_from_db() -> dict:
    """从数据库中读取参数。"""
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, default_value FROM parameters")
        params = {}
        for name, default_value in cursor.fetchall():
            params[name] = {"default_value": json.loads(default_value)}
    return params
