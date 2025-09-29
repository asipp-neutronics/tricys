import ast
import itertools
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _expand_array_parameters(simulation_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expands special array-like string values into indexed parameters.
    It safely parses the string content, recognizing numbers, lists, and strings.

    For example, a key-value pair:
    "param": "{1, [1,2,3], '1:2:1'}"
    will be expanded into:
    "param[1]": 1,
    "param[2]": [1, 2, 3],
    "param[3]": "1:2:1"
    """
    expanded_params = {}
    for name, value in simulation_params.items():
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            try:
                # To make it a valid Python literal, get the content inside the braces
                # and wrap it in list brackets `[]`.
                content_str = value[1:-1]
                list_like_str = f"[{content_str}]"

                # Safely evaluate the string into a Python list object
                parsed_values = ast.literal_eval(list_like_str)

                if not isinstance(parsed_values, list):
                    raise TypeError("Parsed value is not a list as expected.")

                # Create the new indexed keys from the parsed values
                for i, parsed_val in enumerate(parsed_values):
                    new_key = f"{name}[{i + 1}]"  # Use 1-based indexing
                    expanded_params[new_key] = parsed_val

            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                # If parsing fails for any reason, fall back to treating it as a single literal string.
                logger.warning(
                    f"Could not parse complex array-like string '{value}' for parameter '{name}'. "
                    "Treating it as a single literal value."
                )
                expanded_params[name] = value
        else:
            # If it's not the special format, just copy it over.
            expanded_params[name] = value
    return expanded_params


def parse_parameter_value(value: Any) -> List[Any]:
    """
    Parses a parameter value which can be a single value, a list, or a string
    with special formats:
    - "start:stop:step" -> e.g., "1:10:2" for a linear range.
    - "linspace:start:stop:num" -> e.g., "linspace:0:10:5" for 5 points from 0 to 10.
    - "log:start:stop:num" -> e.g., "log:1:1000:4" for 4 points on a log scale.
    - "rand:min:max:count" -> e.g., "rand:0:1:10" for 10 random numbers.
    - "file:path:column" -> e.g., "file:data.csv:voltage" to read a CSV column.
    - "file:path" -> e.g., "file:sampling.csv" to read all parameters from CSV (first row as parameter names).
    """
    if not isinstance(value, str):
        return value if isinstance(value, list) else [value]

    if ":" not in value:
        return [value]  # Just a plain string

    try:
        prefix, args_str = value.split(":", 1)
        prefix = prefix.lower()

        if prefix == "linspace":
            start, stop, num = map(float, args_str.split(":"))
            return np.linspace(start, stop, int(num)).round(8).tolist()

        if prefix == "log":
            start, stop, num = map(float, args_str.split(":"))
            if start <= 0 or stop <= 0:
                raise ValueError("Log scale start and stop values must be positive.")
            return (
                np.logspace(np.log10(start), np.log10(stop), int(num)).round(8).tolist()
            )

        if prefix == "rand":
            low, high, count = map(float, args_str.split(":"))
            return np.random.uniform(low, high, int(count)).round(8).tolist()

        if prefix == "file":
            # Handle file paths that may contain colons (e.g., Windows C:\...)
            try:
                # Check if there's a column name specified
                if ":" in args_str:
                    file_path, column_name = args_str.rsplit(":", 1)
                    if not os.path.isabs(file_path.strip()):
                        abs_file_path = os.path.abspath(
                            os.path.join(os.getcwd(), file_path.strip())
                        )
                    else:
                        abs_file_path = file_path.strip()
                    df = pd.read_csv(abs_file_path)
                    return df[column_name.strip()].tolist()
                else:
                    # Return the file path for later processing
                    return [args_str.strip()]
            except (ValueError, FileNotFoundError, KeyError):
                # Re-raise to be caught by the outer try-except block
                raise

        # Fallback to original start:stop:step logic if no prefix matches
        start, stop, step = map(float, value.split(":"))
        return np.arange(start, stop + step / 2, step).round(8).tolist()

    except (ValueError, FileNotFoundError, KeyError, IndexError) as e:
        logger.error(
            f"Invalid format or error processing parameter value '{value}'. Error: {e}"
        )
        return [value]  # On any error, treat as a single literal value


def _load_jobs_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Load job parameters from a CSV file.
    The first line serves as parameter names, and subsequent lines as parameter values, with each line generating a job.

    Args:
        file_path: CSV file path (can be a relative or absolute path)

    Returns:
        A list of jobs, where each job is a dictionary of parameters
    """
    try:
        # Process relative paths and convert them to absolute paths based on the current working directory
        if not os.path.isabs(file_path.strip()):
            abs_file_path = os.path.abspath(
                os.path.join(os.getcwd(), file_path.strip())
            )
            logger.debug(f"Convert relative path: '{file_path}' -> '{abs_file_path}'")
        else:
            abs_file_path = file_path.strip()

        if not os.path.exists(abs_file_path):
            raise FileNotFoundError(f"The CSV file does not exist: {abs_file_path}")

        df = pd.read_csv(abs_file_path)
        jobs = []

        for _, row in df.iterrows():
            job = {}
            for column in df.columns:
                job[column] = row[column]
            jobs.append(job)

        logger.info(f"Loaded {len(jobs)} jobs from CSV file '{abs_file_path}'")
        return jobs

    except Exception as e:
        logger.error(f"Error loading CSV file '{file_path}': {e}")
        raise


def generate_simulation_jobs(
    simulation_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generates a list of simulation jobs from parameters, handling sweeps and array expansion."""

    logger.info(f"Generating simulation jobs.{simulation_params}")
    if "file" in simulation_params:
        file_value = simulation_params["file"]
        if isinstance(file_value, str):
            csv_jobs = _load_jobs_from_csv(file_value)

            other_params = {k: v for k, v in simulation_params.items() if k != "file"}
            for job in csv_jobs:
                job.update(other_params)

            return csv_jobs

    # First, expand any array-like parameters before processing.
    processed_params = _expand_array_parameters(simulation_params)

    sweep_params = {}
    single_value_params = {}

    for name, value in processed_params.items():
        parsed_values = parse_parameter_value(value)
        if len(parsed_values) > 1:
            sweep_params[name] = parsed_values
        else:
            single_value_params[name] = parsed_values[0] if parsed_values else None

    if not sweep_params:
        return [single_value_params] if single_value_params else [{}]

    sweep_names = list(sweep_params.keys())
    sweep_values = list(sweep_params.values())
    jobs = []
    for combo in itertools.product(*sweep_values):
        job = single_value_params.copy()
        job.update(dict(zip(sweep_names, combo)))
        jobs.append(job)
    return jobs
