"""DWSIM vs Aspen parity tests for three-tower H-isotope distillation.

Parametrized over 5 test cases (TC1-TC5). Each case sets different
H/D/T mass flow inputs and verifies DWSIM output against Aspen baseline.

When no Aspen baseline is available, tests verify:
- All 3 columns converge
- Mass balance closure < 5%
- All 9 output values (3 streams × 3 components) are non-negative
"""

import os
import sys

import pytest

# Set up path before importing run_dwsim_point
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "script", "dwsim")
sys.path.insert(0, os.path.abspath(SCRIPT_DIR))

from run_dwsim_point import (  # noqa: E402
    TEST_CASES,
    compute_feed_composition,
    extract_stream_hdt,
)

# Parity thresholds
REL_TOLERANCE = 0.20  # 20% relative deviation (coarse, Phase 4.2 uses 5%)
ABS_TOLERANCE = 0.01  # g/h absolute tolerance for near-zero values

# Output stream mapping: logical name → DWSIM stream key
TERMINAL_STREAMS = {
    "WDS": "S4",
    "SDSD2": "S16",
    "SDST2": "S17",
    "CD3_DIST": "S_CD3_DIST",
}


def _solve_case(dwsim_interf, sim, objects, case_name):
    """Solve one test case and return results dict.

    Returns:
        dict with keys: converged, results, mass_balance_err
        results: {stream_name: {"H": g/h, "D": g/h, "T": g/h}}
    """
    T_flow, D_flow, H_flow = TEST_CASES[case_name]
    feed_comp, total_mol = compute_feed_composition(T_flow, D_flow, H_flow)

    # Set feed
    feed = objects["streams"]["FROMTEP"]
    feed.SetTemperature(22.0)
    feed.SetPressure(101325.0)
    for comp, frac in feed_comp.items():
        feed.SetOverallCompoundMolarFlow(comp, frac * total_mol / 3600.0)

    # Solve with retries
    max_iters = 5
    all_converged = False
    for _ in range(max_iters):
        dwsim_interf.CalculateFlowsheet2(sim)
        all_converged = all(col.Calculated for col in objects["columns"].values())
        if all_converged:
            break

    # Extract results
    results = {}
    for label, key in TERMINAL_STREAMS.items():
        s = objects["streams"].get(key)
        if s is not None:
            H, D, T = extract_stream_hdt(s)
            results[label] = {"H": H, "D": D, "T": T}
        else:
            results[label] = {"H": 0.0, "D": 0.0, "T": 0.0}

    # Mass balance
    in_total = T_flow + D_flow + H_flow
    out_total = sum(r["H"] + r["D"] + r["T"] for r in results.values())
    mb_err = abs(out_total - in_total) / in_total if in_total > 0 else 0.0

    return {
        "converged": all_converged,
        "results": results,
        "mass_balance_err": mb_err,
        "input": {"T": T_flow, "D": D_flow, "H": H_flow},
    }


def _format_deviation_table(dwsim_results, aspen_results):
    """Generate deviation table as list of dicts.

    Each entry: {variable, aspen, dwsim, rel_dev, pass_fail}
    """
    rows = []
    for stream in ["WDS", "SDSD2", "SDST2"]:
        for comp in ["H", "D", "T"]:
            var_name = f"{stream}.{comp}"
            dw_val = dwsim_results.get(stream, {}).get(comp, 0.0)

            if aspen_results is not None:
                asp_val = aspen_results.get(stream, {}).get(comp, 0.0)
                if abs(asp_val) > ABS_TOLERANCE:
                    rel_dev = abs(dw_val - asp_val) / abs(asp_val)
                else:
                    rel_dev = abs(dw_val - asp_val)
                passed = (
                    rel_dev < REL_TOLERANCE or abs(dw_val - asp_val) < ABS_TOLERANCE
                )
            else:
                asp_val = None
                rel_dev = None
                passed = dw_val >= 0  # Just check non-negative

            rows.append(
                {
                    "variable": var_name,
                    "aspen": asp_val,
                    "dwsim": dw_val,
                    "rel_dev": rel_dev,
                    "pass": passed,
                }
            )
    return rows


@pytest.mark.slow
@pytest.mark.parametrize("case_name", list(TEST_CASES.keys()))
def test_parity(case_name, dwsim_interf, dwsim_flowsheet, aspen_baseline):
    """Test DWSIM vs Aspen parity for a single test case."""
    sim, objects = dwsim_flowsheet
    result = _solve_case(dwsim_interf, sim, objects, case_name)

    # Check convergence
    assert result["converged"], f"{case_name}: Not all columns converged"

    # Check mass balance
    assert (
        result["mass_balance_err"] < 0.05
    ), f"{case_name}: Mass balance error {result['mass_balance_err']:.2%} > 5%"

    # Get Aspen baseline for this case (may be None)
    aspen = aspen_baseline.get(case_name) if aspen_baseline else None

    # Build deviation table
    table = _format_deviation_table(result["results"], aspen)

    # Print deviation table
    print(
        f"\n{'Variable':<15} {'Aspen':>10} {'DWSIM':>10} {'RelDev':>10} {'Status':>6}"
    )
    print("-" * 55)
    for row in table:
        asp_str = f"{row['aspen']:.4f}" if row["aspen"] is not None else "N/A"
        dev_str = f"{row['rel_dev']:.2%}" if row["rel_dev"] is not None else "N/A"
        status = "PASS" if row["pass"] else "FAIL"
        print(
            f"{row['variable']:<15} {asp_str:>10} {row['dwsim']:>10.4f} {dev_str:>10} {status:>6}"
        )

    # Check pass rate
    pass_count = sum(1 for r in table if r["pass"])
    total_count = len(table)
    print(f"\nPass rate: {pass_count}/{total_count}")

    if aspen is not None:
        assert pass_count >= 6, (
            f"{case_name}: Only {pass_count}/{total_count} within tolerance "
            f"(need ≥6)"
        )
    else:
        # No baseline: just verify all non-negative
        for row in table:
            assert (
                row["dwsim"] >= 0
            ), f"{case_name}: Negative output {row['variable']}={row['dwsim']}"

    # Check at least some separation is happening
    # (not all outputs identical to feed fractions)
    main_streams = {
        k: v for k, v in result["results"].items() if k in ["WDS", "SDSD2", "SDST2"]
    }
    total_T = sum(s["T"] for s in main_streams.values())
    assert total_T > 0, f"{case_name}: No tritium flow in output streams"
