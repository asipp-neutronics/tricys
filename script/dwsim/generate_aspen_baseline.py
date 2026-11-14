#!/usr/bin/env python3
"""Generate Aspen Plus baseline data for DWSIM parity testing.

Runs 5 representative feed conditions through the T2-Threetowers4 Aspen model
and records the steady-state output streams (S4/WDS, S17/SDST2, S16/SDSD2).

Usage (Windows only):
    python script/dwsim/generate_aspen_baseline.py \\
        --bkp example/example_aspenbkp/T2-Threetowers4.bkp \\
        --output test/dwsim/fixtures/aspen_baseline.csv

Output CSV columns:
    case_id, T_in, D_in, H_in,
    EH2, EHD, ED2, EHT, EDT, ET2, total_mol,
    WDS_H, WDS_D, WDS_T,
    SDST2_H, SDST2_D, SDST2_T,
    SDSD2_H, SDSD2_D, SDSD2_T

Requires: pywin32, Aspen Plus installed and licensed.
"""

import argparse
import csv
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Molar masses (g/mol)
M_T = 3.016
M_D = 2.014
M_H = 1.008

# 5 test cases: (T_flow_g_h, D_flow_g_h, H_flow_g_h, description)
TEST_CASES = [
    ("TC1", 100.0, 50.0, 10.0, "Nominal (high T enrichment)"),
    ("TC2", 50.0, 50.0, 50.0, "Equimolar feed"),
    ("TC3", 150.0, 10.0, 5.0, "Near-pure T feed"),
    ("TC4", 10.0, 100.0, 50.0, "Low T, high D feed"),
    ("TC5", 80.0, 30.0, 20.0, "Intermediate case"),
]

CSV_HEADER = [
    "case_id",
    "description",
    "T_in_g_h",
    "D_in_g_h",
    "H_in_g_h",
    "EH2",
    "EHD",
    "ED2",
    "EHT",
    "EDT",
    "ET2",
    "total_mol_h",
    "WDS_H_g_h",
    "WDS_D_g_h",
    "WDS_T_g_h",
    "SDST2_H_g_h",
    "SDST2_D_g_h",
    "SDST2_T_g_h",
    "SDSD2_H_g_h",
    "SDSD2_D_g_h",
    "SDSD2_T_g_h",
    "mass_balance_rel_err",
]


def compute_equilibrium_composition(T_flow, D_flow, H_flow):
    """Compute equilibrium isotopologue composition from elemental mass flows.

    Follows the ideal isotope scrambling model used in i_iss_handler.py.
    Returns (EH2, EHD, ED2, EHT, EDT, ET2, total_mol_flow).
    """
    T_mol = T_flow / M_T
    D_mol = D_flow / M_D
    H_mol = H_flow / M_H
    total = T_mol + D_mol + H_mol

    if total < 1e-12:
        return (0, 0, 0, 0, 0, 0, 0)

    ET = T_mol / total
    ED = D_mol / total
    EH = H_mol / total

    return (
        EH**2,  # EH2
        2 * EH * ED,  # EHD
        ED**2,  # ED2
        2 * EH * ET,  # EHT
        2 * ED * ET,  # EDT
        ET**2,  # ET2
        total,
    )


def run_single_case(aspen_enhanced, T_flow, D_flow, H_flow):
    """Run one steady-state case and return results.

    Args:
        aspen_enhanced: AspenEnhanced instance
        T_flow, D_flow, H_flow: Mass flows in g/h

    Returns:
        tuple: (composition, stream_results)
    """
    comp = compute_equilibrium_composition(T_flow, D_flow, H_flow)
    EH2, EHD, ED2, EHT, EDT, ET2, total = comp

    ratios = [EH2, EHD, ED2, EHT, EDT, ET2, total]
    aspen_enhanced.set_composition(ratios)
    aspen_enhanced.run_step()
    results = aspen_enhanced.get_stream_results()

    return comp, results


def main():
    parser = argparse.ArgumentParser(
        description="Generate Aspen baseline for DWSIM parity test"
    )
    parser.add_argument("--bkp", required=True, help="Path to Aspen .bkp file")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()

    bkp_path = os.path.abspath(args.bkp)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(bkp_path):
        logger.error(f"BKP file not found: {bkp_path}")
        sys.exit(1)

    # Import Windows-only modules
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from tricys.example.example_data.v1_examples_legacy.handlers.i_iss_handler import (
            AspenEnhanced,
        )
    except ImportError as e:
        logger.error(f"Cannot import AspenEnhanced: {e}")
        logger.error("This script must run on Windows with Aspen Plus and pywin32.")
        sys.exit(1)

    aspen = None
    try:
        aspen = AspenEnhanced(bkp_path)
        logger.info("Aspen initialized.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        rows = []
        for case_id, T_flow, D_flow, H_flow, desc in TEST_CASES:
            logger.info(
                f"Running {case_id}: T={T_flow}, D={D_flow}, H={H_flow} ({desc})"
            )

            comp, results = run_single_case(aspen, T_flow, D_flow, H_flow)
            EH2, EHD, ED2, EHT, EDT, ET2, total = comp

            wds = results["WDS"]  # [H, D, T]
            sdst2 = results["SDST2"]  # [H, D, T]
            sdsd2 = results["SDSD2"]  # [H, D, T]

            # Mass balance check
            total_out_H = wds[0] + sdst2[0] + sdsd2[0]
            total_out_D = wds[1] + sdst2[1] + sdsd2[1]
            total_out_T = wds[2] + sdst2[2] + sdsd2[2]
            total_in = T_flow + D_flow + H_flow
            total_out = total_out_H + total_out_D + total_out_T
            rel_err = abs(total_in - total_out) / max(total_in, 1e-12)

            row = [
                case_id,
                desc,
                T_flow,
                D_flow,
                H_flow,
                EH2,
                EHD,
                ED2,
                EHT,
                EDT,
                ET2,
                total,
                wds[0],
                wds[1],
                wds[2],
                sdst2[0],
                sdst2[1],
                sdst2[2],
                sdsd2[0],
                sdsd2[1],
                sdsd2[2],
                rel_err,
            ]
            rows.append(row)

            logger.info(f"  WDS:   H={wds[0]:.4f}, D={wds[1]:.4f}, T={wds[2]:.4f}")
            logger.info(
                f"  SDST2: H={sdst2[0]:.4f}, D={sdst2[1]:.4f}, T={sdst2[2]:.4f}"
            )
            logger.info(
                f"  SDSD2: H={sdsd2[0]:.4f}, D={sdsd2[1]:.4f}, T={sdsd2[2]:.4f}"
            )
            logger.info(f"  Mass balance rel err: {rel_err:.6f}")

        # Write CSV
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)

        logger.info(f"Baseline written to {output_path}")
        logger.info(f"Total cases: {len(rows)}")

        # Validate mass balance
        max_err = max(r[-1] for r in rows)
        if max_err > 0.01:
            logger.warning(
                f"Max mass balance error {max_err:.4f} exceeds 1% threshold!"
            )
        else:
            logger.info(f"All mass balances within 1% (max={max_err:.6f})")

    except Exception as e:
        logger.error(f"Baseline generation failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if aspen:
            aspen.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
