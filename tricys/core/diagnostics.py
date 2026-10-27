"""Multi-Level Diagnostics Barrier and Self-Healing Engine for TRICYS v2.0.

This module intercepts four levels of simulation and workflow failures:
- Level 1: Configuration / Schema validation errors (ERR_CONFIG_INVALID)
- Level 2: Modelica compilation and syntax errors (ERR_MODELICA_COMPILE)
- Level 3: Numerical solver step underflow and divergence (ERR_SOLVER_STEP_UNDERFLOW / ERR_NEGATIVE_INVENTORY)
- Level 4: Closed-loop mass balance conservation violations (ERR_MASS_BALANCE_VIOLATION)

It emits structured diagnostics.json containing root causes and automated suggested fixes for Agent self-healing.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ErrorLevel(str, Enum):
    """Categorized failure level hierarchy."""

    LEVEL_1_CONFIG = "LEVEL_1_CONFIG"
    LEVEL_2_MODELICA = "LEVEL_2_MODELICA"
    LEVEL_3_SOLVER = "LEVEL_3_SOLVER"
    LEVEL_4_PHYSICS = "LEVEL_4_PHYSICS"
    UNKNOWN = "UNKNOWN"


@dataclass
class SuggestedFix:
    """Actionable remediation recommendation for AI Agents."""

    action: str  # "modify_yaml", "patch_modelica", "adjust_solver", "check_streams"
    modifications: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    patch_diff: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticRecord:
    """Complete diagnostic artifact output on failure."""

    status: str = "FAILED"
    error_level: ErrorLevel = ErrorLevel.UNKNOWN
    error_code: str = "ERR_UNKNOWN"
    message: str = ""
    root_cause: str = ""
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[SuggestedFix] = None
    traceback_summary: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["error_level"] = self.error_level.value
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class DiagnosticsClassifier:
    """Classifies exceptions into four levels and generates actionable remediation."""

    @staticmethod
    def classify(
        exc: Exception,
        context_snapshot: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticRecord:
        ctx = context_snapshot or {}
        exc_str = str(exc)
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        short_tb = [line.strip() for line in tb_lines[-4:] if line.strip()]

        # ---------------------------------------------------------
        # Level 1: Configuration / Schema validation error
        # ---------------------------------------------------------
        from pydantic import ValidationError

        if isinstance(exc, (ValidationError, ValueError, KeyError)) or (
            "Schema" in type(exc).__name__
            or "Config" in type(exc).__name__
            or "validation" in exc_str.lower()
        ):
            # Extract invalid fields from ValidationError if possible
            invalid_fields = []
            if isinstance(exc, ValidationError):
                for err in exc.errors():
                    loc = ".".join(str(p) for p in err.get("loc", []))
                    msg = err.get("msg", "")
                    invalid_fields.append(f"{loc}: {msg}")

            field_hint = (
                f"Validation errors: {'; '.join(invalid_fields)}"
                if invalid_fields
                else exc_str
            )
            return DiagnosticRecord(
                status="FAILED",
                error_level=ErrorLevel.LEVEL_1_CONFIG,
                error_code="ERR_CONFIG_INVALID",
                message=f"Configuration validation failed: {exc_str}",
                root_cause=field_hint,
                context_snapshot=ctx,
                suggested_fix=SuggestedFix(
                    action="modify_yaml",
                    explanation="Fix invalid or unexpected fields in configuration YAML according to Schema 2.0.",
                ),
                traceback_summary=short_tb,
            )

        # ---------------------------------------------------------
        # Level 2: Modelica compilation / syntax error
        # ---------------------------------------------------------
        if (
            "Modelica" in exc_str
            or "CompilationFailed" in type(exc).__name__
            or "CompilerError" in type(exc).__name__
            or "loadFile" in exc_str
            or "buildModel" in exc_str
            or "syntax error" in exc_str.lower()
        ):
            # Find line number if available
            line_match = re.search(r"line\s+(\d+)", exc_str, re.IGNORECASE)
            line_no = line_match.group(1) if line_match else "unknown"

            return DiagnosticRecord(
                status="FAILED",
                error_level=ErrorLevel.LEVEL_2_MODELICA,
                error_code="ERR_MODELICA_COMPILE",
                message=f"Modelica model compilation or parsing failed: {exc_str}",
                root_cause=f"Modelica compiler encountered syntax or binding error near line {line_no}.",
                context_snapshot=ctx,
                suggested_fix=SuggestedFix(
                    action="patch_modelica",
                    explanation="Run 'python -m tricys.main model review <mo> <model> --json' to diagnose and patch model issues.",
                ),
                traceback_summary=short_tb,
            )

        # ---------------------------------------------------------
        # Level 3: Numerical solver divergence / step underflow / negative inventory
        # ---------------------------------------------------------
        if (
            "dassl" in exc_str.lower()
            or "step size" in exc_str.lower()
            or "underflow" in exc_str.lower()
            or "convergence" in exc_str.lower()
            or "singularity" in exc_str.lower()
            or "negative inventory" in exc_str.lower()
        ):
            is_negative = "negative inventory" in exc_str.lower()
            code = (
                "ERR_NEGATIVE_INVENTORY" if is_negative else "ERR_SOLVER_STEP_UNDERFLOW"
            )
            cause = (
                "Physical inventory dropped below zero due to excessive depletion or solver overshoot."
                if is_negative
                else "Solver integration failed to converge; step size dropped below minimum threshold."
            )

            current_solver = ctx.get("solver", "dassl")
            alt_solver = "ida" if current_solver == "dassl" else "dassl"
            relaxed_tol = round(float(ctx.get("tolerance", 1e-6)) * 10, 9)

            return DiagnosticRecord(
                status="FAILED",
                error_level=ErrorLevel.LEVEL_3_SOLVER,
                error_code=code,
                message=f"Numerical simulation kernel halted: {exc_str}",
                root_cause=cause,
                context_snapshot=ctx,
                suggested_fix=SuggestedFix(
                    action="adjust_solver",
                    modifications={
                        "scenario.solver": alt_solver,
                        "scenario.tolerance": relaxed_tol,
                    },
                    explanation=(
                        f"Switch solver from '{current_solver}' to '{alt_solver}' and relax tolerance "
                        f"to {relaxed_tol} to overcome stiff numerical boundaries."
                    ),
                ),
                traceback_summary=short_tb,
            )

        # ---------------------------------------------------------
        # Level 4: Closed-loop mass conservation violation
        # ---------------------------------------------------------
        if (
            "conservation" in exc_str.lower()
            or "mass balance" in exc_str.lower()
            or "residual" in exc_str.lower()
        ):
            return DiagnosticRecord(
                status="FAILED",
                error_level=ErrorLevel.LEVEL_4_PHYSICS,
                error_code="ERR_MASS_BALANCE_VIOLATION",
                message=f"Mass balance conservation threshold exceeded: {exc_str}",
                root_cause="Cumulative plant-wide tritium balance residual |Delta M| exceeded safety margin.",
                context_snapshot=ctx,
                suggested_fix=SuggestedFix(
                    action="check_streams",
                    explanation="Verify decay_rate and leak_rate stream completeness, or audit transfer efficiencies.",
                ),
                traceback_summary=short_tb,
            )

        # Fallback: Unknown execution error
        return DiagnosticRecord(
            status="FAILED",
            error_level=ErrorLevel.UNKNOWN,
            error_code="ERR_RUNTIME_EXECUTION",
            message=exc_str or type(exc).__name__,
            root_cause="Unhandled exception encountered during workflow execution.",
            context_snapshot=ctx,
            suggested_fix=SuggestedFix(
                action="inspect_logs",
                explanation="Examine traceback summary to identify source of failure.",
            ),
            traceback_summary=short_tb,
        )


class DiagnosticsBarrier:
    """Context manager wrapping workflow execution to intercept and emit structured diagnostics."""

    def __init__(
        self,
        output_path: Optional[Union[str, Path]] = "diagnostics.json",
        context_snapshot: Optional[Dict[str, Any]] = None,
        exit_on_error: bool = False,
    ):
        self.output_path = Path(output_path) if output_path else None
        self.context_snapshot = context_snapshot or {}
        self.exit_on_error = exit_on_error
        self.record: Optional[DiagnosticRecord] = None

    def __enter__(self) -> DiagnosticsBarrier:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val is None:
            return False  # No exception, proceed normally

        # Intercept and classify exception
        self.record = DiagnosticsClassifier.classify(
            exc_val, context_snapshot=self.context_snapshot
        )

        # Persist diagnostics.json
        if self.output_path:
            try:
                self.output_path.write_text(
                    self.record.to_json(indent=2), encoding="utf-8"
                )
                logger.info(f"Diagnostics report generated at: {self.output_path}")
            except Exception as e:
                logger.warning(f"Failed to write diagnostics report: {e}")

        # Emit to stderr
        print("\n" + "=" * 60, file=sys.stderr)
        print(
            f"TRICYS DIAGNOSTIC BARRIER INTERCEPTED [{self.record.error_level.value}]",
            file=sys.stderr,
        )
        print(f"Error Code : {self.record.error_code}", file=sys.stderr)
        print(f"Root Cause : {self.record.root_cause}", file=sys.stderr)
        if self.record.suggested_fix:
            print(
                f"Suggested  : {self.record.suggested_fix.explanation}", file=sys.stderr
            )
        print("=" * 60 + "\n", file=sys.stderr)

        if self.exit_on_error:
            sys.exit(1)

        # Do not suppress exception so callers can handle or test it
        return False
