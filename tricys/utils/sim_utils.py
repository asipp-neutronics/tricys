import itertools
import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def parse_parameter_value(value: Any) -> List[Any]:
    """Parses a parameter value which can be a single value, a list, or a range string."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and ":" in value:
        try:
            start, stop, step = map(float, value.split(":"))
            return np.arange(start, stop + step / 2, step).tolist()
        except ValueError:
            logger.error(f"Invalid range format for parameter value: {value}")
            return [value]
    return [value]


def generate_simulation_jobs(
    simulation_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generates a list of simulation jobs from parameters, handling sweeps."""
    sweep_params = {}
    single_value_params = {}
    for name, value in simulation_params.items():
        parsed_values = parse_parameter_value(value)
        if len(parsed_values) > 1:
            sweep_params[name] = parsed_values
        else:
            single_value_params[name] = parsed_values[0]
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
