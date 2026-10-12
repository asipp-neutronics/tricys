"""Fixtures for DWSIM parity tests.

Provides a shared DWSIM flowsheet instance (built once per session)
and utility functions for running test cases.
"""

import os
import sys

import pytest

# Paths
DWSIM_DIR = os.environ.get("DWSIM_DIR", "/usr/local/lib/dwsim")
DOTNET_ROOT = os.environ.get("DOTNET_ROOT", "/usr/lib/dotnet")
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "script", "dwsim")
BASELINE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "example", "example_dwsim"
)


def _setup_dwsim_runtime():
    """Initialize .NET runtime and DWSIM references (once per process)."""
    os.environ["DOTNET_ROOT"] = DOTNET_ROOT

    try:
        from clr_loader import get_coreclr
        from pythonnet import set_runtime
    except ImportError as exc:
        pytest.skip(f"DWSIM tests require pythonnet/clr_loader: {exc}")

    try:
        rt = get_coreclr(dotnet_root=DOTNET_ROOT)
        set_runtime(rt)
    except Exception as exc:
        pytest.skip(f"DWSIM tests require a working .NET runtime: {exc}")

    try:
        import clr
    except Exception as exc:
        pytest.skip(f"DWSIM tests could not import clr: {exc}")

    sys.path.append(DWSIM_DIR)
    try:
        clr.AddReference("DWSIM.Automation")
        clr.AddReference("DWSIM.Interfaces")
    except Exception as exc:
        pytest.skip(f"DWSIM assemblies are not available: {exc}")


@pytest.fixture(scope="session")
def dwsim_interf():
    """Session-scoped DWSIM Automation3 interface."""
    _setup_dwsim_runtime()
    try:
        from DWSIM.Automation import Automation3
    except Exception as exc:
        pytest.skip(f"DWSIM.Automation is not available: {exc}")
    return Automation3()


@pytest.fixture(scope="session")
def dwsim_flowsheet(dwsim_interf):
    """Session-scoped DWSIM flowsheet built from scratch.

    Returns (sim, built_objects) tuple.
    """
    script_dir = os.path.abspath(SCRIPT_DIR)
    sys.path.insert(0, script_dir)
    from build_dwsim_flowsheet import build_three_towers, configure_srk_bip
    from register_compounds import register_compounds

    sim = dwsim_interf.CreateFlowsheet()
    register_compounds(sim, {})
    sim.CreateAndAddPropertyPackage("Soave-Redlich-Kwong (SRK)")
    configure_srk_bip(sim, {})
    objects = build_three_towers(sim, {})
    return sim, objects


@pytest.fixture(scope="session")
def aspen_baseline():
    """Load Aspen baseline CSV if available.

    Returns dict {case_name: {stream_name: {"H": val, "D": val, "T": val}}}
    or None if baseline file not found.
    """
    import csv

    csv_path = os.path.join(BASELINE_DIR, "aspen_baseline.csv")
    if not os.path.exists(csv_path):
        return None

    baseline = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row["case"]
            stream = row["stream"]
            if case not in baseline:
                baseline[case] = {}
            baseline[case][stream] = {
                "H": float(row["H_g_h"]),
                "D": float(row["D_g_h"]),
                "T": float(row["T_g_h"]),
            }
    return baseline
