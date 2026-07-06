"""Dedicated entry point for co-simulation workflows.

This module centralizes online and offline co-simulation entry dispatch so the
standard simulation entry does not own top-level co-simulation routing.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Union

from tricys.online_cosim.runtime import run_online_cosimulation
from tricys.simulation.simulation import run_simulation
from tricys.utils.config_utils import basic_prepare_config
from tricys.utils.log_utils import setup_logging


def run_cosimulation(config: Dict[str, Any], export_csv: bool = False) -> None:
    co_simulation_config = config.get("co_simulation")
    if not isinstance(co_simulation_config, dict):
        raise ValueError("Co-simulation entry requires a 'co_simulation' config block")

    if co_simulation_config.get("engine") == "online_oms":
        run_online_cosimulation(config, export_csv=export_csv)
        return

    run_simulation(config, export_csv=export_csv)


def main(
    config_or_path: Union[str, Dict[str, Any]],
    base_dir: str | None = None,
    export_csv: bool = False,
) -> None:
    config, original_config = basic_prepare_config(config_or_path, base_dir=base_dir)
    setup_logging(config, original_config)

    try:
        run_cosimulation(config, export_csv=export_csv)
    except Exception as exc:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(
            "Co-simulation execution failed",
            exc_info=True,
            extra={"exception": str(exc)},
        )
        sys.exit(1)
