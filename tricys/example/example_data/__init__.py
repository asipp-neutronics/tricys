"""TRICYS example data package."""

import importlib
import sys

from . import v1_examples_legacy as v1_examples_legacy
from .v1_examples_legacy import basic

# Backward compatibility alias
sys.modules["tricys.example.example_data.basic"] = basic
try:
    m_cosim = importlib.import_module(
        "tricys.example.example_data.v1_examples_legacy.basic.9_online_cosim"
    )
    sys.modules["tricys.example.example_data.basic.9_online_cosim"] = m_cosim
    m_proc = importlib.import_module(
        "tricys.example.example_data.v1_examples_legacy.basic.9_online_cosim.i_iss_processor_base"
    )
    sys.modules[
        "tricys.example.example_data.basic.9_online_cosim.i_iss_processor_base"
    ] = m_proc
except Exception:
    pass
