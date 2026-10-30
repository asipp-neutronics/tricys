"""TRICYS legacy co-simulation handlers (v1 examples / Aspen Plus)."""

from __future__ import annotations

from tricys.example.example_data.v1_examples_legacy.handlers import div_handler
from tricys.example.example_data.v1_examples_legacy.handlers.div_handler import (
    run_div_simulation,
)

try:
    from tricys.example.example_data.v1_examples_legacy.handlers import i_iss_handler
    from tricys.example.example_data.v1_examples_legacy.handlers.i_iss_handler import (
        AspenEnhanced,
        run_aspen_simulation,
        run_dummy_simulation,
    )
except Exception:
    i_iss_handler = None
    run_aspen_simulation = None
    AspenEnhanced = None
    run_dummy_simulation = None

__all__ = [
    "div_handler",
    "i_iss_handler",
    "run_div_simulation",
]
if run_aspen_simulation is not None:
    __all__.extend(["run_aspen_simulation", "AspenEnhanced", "run_dummy_simulation"])
