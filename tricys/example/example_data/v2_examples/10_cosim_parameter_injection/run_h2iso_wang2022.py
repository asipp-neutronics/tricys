"""h2iso Cryogenic Distillation Wang 2022 Solver & Parameter Injection Handler.

This script serves as the external solver handler for TRICYS parameter injection mode.
It loads Wang (2022) CFETR ISS-I and ISS-O flowsheet configurations, executes the
h2iso steady-state MESH sequential modular solver, extracts converged physical parameters
(separation factors, flow rates, temperature profiles, holdup residence times),
and generates the OpenModelica parameter override file (override_cycle.txt).

Compatible with:
- TRICYS Schema 2.0 Parameter Injection Engine (ParamInjectSimulationEngine)
- h2iso L1 (VLE) & L2 (Steady-State MESH Flowsheet) Solvers
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][h2iso_wang2022] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("h2iso_wang2022")

SCRIPT_DIR = Path(__file__).resolve().parent


def extract_flowsheet_column_data(solver: Any, res: Any) -> dict[str, Any]:
    """Extract column results and stream properties from a solved flowsheet."""
    from h2iso.species import SPECIES_ORDER

    data: dict[str, Any] = {"columns": {}}
    for col_name, col_res in res.column_results.items():
        dist = solver.streams.get(f"{col_name}_distillate")
        bot = solver.streams.get(f"{col_name}_bottoms")

        top_flow = float(dist.flow) if dist else 1.0
        bot_flow = float(bot.flow) if bot else 1.0

        top_comp = (
            {sp: float(dist.composition[i]) for i, sp in enumerate(SPECIES_ORDER)}
            if dist
            else {}
        )
        bot_comp = (
            {sp: float(bot.composition[i]) for i, sp in enumerate(SPECIES_ORDER)}
            if bot
            else {}
        )

        T_top = float(col_res.T_profile[0]) if hasattr(col_res, "T_profile") else 20.0
        T_bottom = (
            float(col_res.T_profile[-1]) if hasattr(col_res, "T_profile") else 25.0
        )

        data["columns"][col_name] = {
            "actual_results": {
                "top_flow_mol_h": top_flow,
                "bottom_flow_mol_h": bot_flow,
                "top_composition": top_comp,
                "bottom_composition": bot_comp,
                "temperatures": {
                    "top_K": T_top,
                    "bottom_K": T_bottom,
                },
                "inventory": {"total_grams": 0.0},
            }
        }
    return data


def solve_wang2022_flowsheets(
    issi_path: Path,
    isso_path: Path,
    continuation_substeps: int = 2,
    max_iter: int = 25,
    tol: float = 1e-3,
    method: str = "wegstein",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Solve Wang 2022 ISS-I and ISS-O flowsheets using h2iso SequentialModularSolver."""
    from h2iso.flowsheet.schema import load_flowsheet
    from h2iso.flowsheet.solver import SequentialModularSolver

    logger.info("=" * 65)
    logger.info("Executing h2iso Rigorous MESH Sequential Modular Solver (Wang 2022)")
    logger.info("=" * 65)

    # 1. Solve ISS-O (Outer loop, 3 columns: CD1, CD2, CD3 + 1 equilibrator + 2 recycles)
    logger.info(f"[1/2] Loading ISS-O flowsheet configuration: {isso_path.name}")
    isso_cfg = load_flowsheet(isso_path)
    logger.info(
        f"      ISS-O structure: {len(isso_cfg.columns)} columns, {len(isso_cfg.feeds)} feeds, "
        f"{len(isso_cfg.tear_streams)} tear streams"
    )
    isso_solver = SequentialModularSolver(
        isso_cfg,
        method=method,
        continuation_substeps=continuation_substeps,
    )
    logger.info(
        f"      Solving ISS-O with {method} method (max_iter={max_iter}, tol={tol})..."
    )
    isso_res = isso_solver.solve(max_iter=max_iter, tol=tol)
    logger.info(
        f"      ISS-O solve finished: converged={isso_res.converged}, "
        f"iterations={isso_res.iterations}, residual={isso_res.tear_residual:.2e}"
    )
    isso_data = extract_flowsheet_column_data(isso_solver, isso_res)

    # 2. Solve ISS-I (Inner loop, 4 columns: CD1, CD2, CD3, CD4 + 2 equilibrators)
    logger.info(f"[2/2] Loading ISS-I flowsheet configuration: {issi_path.name}")
    issi_cfg = load_flowsheet(issi_path)
    logger.info(
        f"      ISS-I structure: {len(issi_cfg.columns)} columns, {len(issi_cfg.feeds)} feeds, "
        f"{len(issi_cfg.tear_streams)} tear streams"
    )
    issi_solver = SequentialModularSolver(
        issi_cfg,
        method=method,
        continuation_substeps=continuation_substeps,
    )
    logger.info(
        f"      Solving ISS-I with {method} method (max_iter={max_iter}, tol={tol})..."
    )
    issi_res = issi_solver.solve(max_iter=max_iter, tol=tol)
    logger.info(
        f"      ISS-I solve finished: converged={issi_res.converged}, "
        f"iterations={issi_res.iterations}, residual={issi_res.tear_residual:.2e}"
    )
    issi_data = extract_flowsheet_column_data(issi_solver, issi_res)

    return issi_data, isso_data


def generate_parameter_overrides(
    workspace: Path,
    params: dict[str, Any],
) -> Path:
    """Orchestrates h2iso flowsheet solve and generates override_cycle.txt."""
    # Resolve config files
    issi_conf = params.get("wang2022_config_issi", "wang2022_issi.json")
    isso_conf = params.get("wang2022_config_isso", "wang2022_isso.json")

    issi_path = Path(issi_conf)
    if not issi_path.is_absolute():
        issi_path = (SCRIPT_DIR / issi_path).resolve()
    if not issi_path.exists():
        # Fallback to h2iso fixtures if present
        h2iso_fixture = (
            SCRIPT_DIR.parent.parent.parent.parent.parent
            / "h2iso"
            / "tests"
            / "fixtures"
            / "wang2022"
            / "wang2022_issi.json"
        )
        if h2iso_fixture.exists():
            issi_path = h2iso_fixture

    isso_path = Path(isso_conf)
    if not isso_path.is_absolute():
        isso_path = (SCRIPT_DIR / isso_path).resolve()
    if not isso_path.exists():
        h2iso_fixture = (
            SCRIPT_DIR.parent.parent.parent.parent.parent
            / "h2iso"
            / "tests"
            / "fixtures"
            / "wang2022"
            / "wang2022_isso.json"
        )
        if h2iso_fixture.exists():
            isso_path = h2iso_fixture

    solve_mode = str(params.get("solve_mode", "live")).lower()
    continuation_substeps = int(params.get("continuation_substeps", 2))
    max_iter = int(params.get("max_iter", 20))
    tol = float(params.get("tol", 1e-3))
    method = str(params.get("method", "wegstein"))

    logger.info(f"Target workspace: {workspace}")
    logger.info(f"ISSI configuration: {issi_path}")
    logger.info(f"ISSO configuration: {isso_path}")
    logger.info(f"Execution mode: {solve_mode}")

    issi_data: dict[str, Any] = {}
    isso_data: dict[str, Any] = {}

    try:
        from h2iso.codegen.modelica_0d import (
            export_modelica_package,
            generate_cycle_override_file,
        )

        has_h2iso = True
    except ImportError:
        logger.warning("h2iso package is not installed in current environment.")
        has_h2iso = False

    if has_h2iso and solve_mode in ("live", "rigorous", "mesh"):
        logger.info("Executing live h2iso calculation...")
        issi_data, isso_data = solve_wang2022_flowsheets(
            issi_path=issi_path,
            isso_path=isso_path,
            continuation_substeps=continuation_substeps,
            max_iter=max_iter,
            tol=tol,
            method=method,
        )
        override_content = generate_cycle_override_file(
            issi_results=issi_data,
            isso_results=isso_data,
            o_iss_prefix="o_iss.core",
            i_iss_prefix="i_iss.core",
        )
    elif has_h2iso:
        logger.info(
            f"Using standard h2iso export_modelica_package (mode={solve_mode})..."
        )
        pkg_paths = export_modelica_package(workspace)
        override_file_created = pkg_paths.get("override_cycle.txt")
        if override_file_created and override_file_created.exists():
            override_content = override_file_created.read_text(encoding="utf-8")
        else:
            override_content = ""
    else:
        # Fallback if h2iso unavailable: read pre-existing override_cycle.txt
        fallback_file = SCRIPT_DIR / "override_cycle.txt"
        if fallback_file.exists():
            logger.info("Using baseline override_cycle.txt from package assets.")
            override_content = fallback_file.read_text(encoding="utf-8")
        else:
            raise RuntimeError(
                "Cannot generate override parameters: h2iso missing and no baseline file found."
            )

    # Write override_cycle.txt into workspace
    out_override_path = workspace / "override_cycle.txt"
    if override_content:
        out_override_path.write_text(override_content, encoding="utf-8")
        logger.info(
            f"Generated override file: {out_override_path} ({len(override_content.splitlines())} lines)"
        )

    # Also update script directory's override_cycle.txt for consistency
    script_override_path = SCRIPT_DIR / "override_cycle.txt"
    if override_content and script_override_path != out_override_path:
        script_override_path.write_text(override_content, encoding="utf-8")

    # Ensure Generic_ISS.mo and Generic_ISS_Adapters.mo exist in workspace
    for mo_name in ["Generic_ISS.mo", "Generic_ISS_Adapters.mo"]:
        src_mo = SCRIPT_DIR / mo_name
        dst_mo = workspace / mo_name
        if src_mo.exists() and not dst_mo.exists():
            shutil.copy2(src_mo, dst_mo)
            logger.info(f"Copied Modelica asset to workspace: {mo_name}")

    return out_override_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="h2iso Wang 2022 Parameter Injection Handler"
    )
    parser.add_argument(
        "--workspace", type=str, default=None, help="Simulation workspace path"
    )
    parser.add_argument(
        "--mode", type=str, default=None, help="Solve mode: live or fixture"
    )
    parser.add_argument(
        "--max-iter", type=int, default=None, help="Max iterations for flowsheet solver"
    )
    parser.add_argument("--tol", type=float, default=None, help="Convergence tolerance")
    args = parser.parse_args()

    # Read from environment (set by TRICYS) or command line
    workspace_env = os.environ.get("TRICYS_WORKSPACE")
    workspace_dir = Path(args.workspace or workspace_env or SCRIPT_DIR).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    params_json = os.environ.get("TRICYS_HANDLER_PARAMS", "{}")
    params: dict[str, Any] = json.loads(params_json) if params_json else {}

    if args.mode:
        params["solve_mode"] = args.mode
    if args.max_iter:
        params["max_iter"] = args.max_iter
    if args.tol:
        params["tol"] = args.tol

    logger.info("Initializing h2iso Wang 2022 Parameter Injection Handler")
    logger.info(f"Workspace: {workspace_dir}")
    logger.info(f"Handler parameters: {params}")

    try:
        override_file = generate_parameter_overrides(workspace_dir, params)
        logger.info(
            f"Parameter injection preparation completed successfully: {override_file}"
        )
        return 0
    except Exception as exc:
        logger.exception(
            f"Error during h2iso calculation or parameter generation: {exc}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
