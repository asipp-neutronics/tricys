"""Modelica Model Health Auditor and Remediation Engine.

This module inspects Modelica models for compliance with TRICYS v2.0 standards:
1. Time scale consistency (SI seconds native and time_scale parameterization).
2. Closed-loop mass conservation streams (decay_rate, leak_rate, breed, burn).
3. Continuous state variable initial values (start = ...) and algebraic safety.
4. TRICYS framework compatibility and observability.

It generates structured audit reports and unified diff patches for automated remediation.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tricys.core.introspect import ModelIntrospection, introspect_model

logger = logging.getLogger(__name__)


class AuditSeverity(str, Enum):
    """Severity levels for model audit findings."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Finding:
    """An individual issue identified during model auditing."""

    code: str
    severity: AuditSeverity
    component: str
    message: str
    suggestion: str
    remediation_patch: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass
class ModelAuditReport:
    """Overall audit report for a Modelica model."""

    package_path: str
    model_name: str
    status: str  # "PASS", "WARNING", "FAIL"
    findings: List[Finding] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_path": self.package_path,
            "model_name": self.model_name,
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ModelAuditor:
    """Audits Modelica code against TRICYS 2.0 physics and simulation conventions."""

    def __init__(self, package_path: Union[str, Path], model_name: str):
        self.package_path = str(Path(package_path).resolve())
        self.model_name = model_name
        self.raw_content = Path(self.package_path).read_text(encoding="utf-8")
        self.introspection: ModelIntrospection = introspect_model(
            self.package_path, self.model_name
        )

    def audit(self) -> ModelAuditReport:
        """Executes full multi-dimensional model audit."""
        findings: List[Finding] = []

        # 1. Audit time scale
        findings.extend(self._audit_time_scale())

        # 2. Audit conservation streams
        findings.extend(self._audit_conservation_streams())

        # 3. Audit initial values
        findings.extend(self._audit_initial_values())

        # 4. Audit TRICYS framework compatibility
        findings.extend(self._audit_framework_compatibility())

        # Calculate summary and status
        crit_count = sum(1 for f in findings if f.severity == AuditSeverity.CRITICAL)
        warn_count = sum(1 for f in findings if f.severity == AuditSeverity.WARNING)
        info_count = sum(1 for f in findings if f.severity == AuditSeverity.INFO)

        status = "PASS"
        if crit_count > 0:
            status = "FAIL"
        elif warn_count > 0:
            status = "WARNING"

        summary = {
            "total_findings": len(findings),
            "critical": crit_count,
            "warning": warn_count,
            "info": info_count,
        }

        return ModelAuditReport(
            package_path=self.package_path,
            model_name=self.model_name,
            status=status,
            findings=findings,
            summary=summary,
        )

    def _audit_time_scale(self) -> List[Finding]:
        """Audits time scale parameterization (preventing 3600x mismatch / Issue #99)."""
        findings: List[Finding] = []

        # Check top-level parameter
        has_time_scale = any(
            p.name == "time_scale" or p.name.endswith(".time_scale")
            for p in self.introspection.parameters.values()
        )

        if not has_time_scale:
            # Check if raw code defines time_scale
            if not re.search(r"\bparameter\s+Real\s+time_scale\b", self.raw_content):
                findings.append(
                    Finding(
                        code="TIME_SCALE_MISSING",
                        severity=AuditSeverity.CRITICAL,
                        component=self.model_name,
                        message=(
                            f"Top-level model '{self.model_name}' lacks 'time_scale' parameter declaration. "
                            "This violates TRICYS 2.0 SI seconds standards and leads to 3600x time dilation."
                        ),
                        suggestion=(
                            "Add 'inner parameter Real time_scale = 3600 \"Time-base factor: 1=hour, 3600=second\";' "
                            "to top-level model, and declare 'outer parameter Real time_scale;' in subcomponents."
                        ),
                    )
                )

        # Check subcomponents with derivatives der(...) or time scaling
        pattern_model = re.compile(
            r"\b(model|block)\s+([A-Za-z0-9_]+)\b(.*?)\bend\s+\2\s*;",
            re.DOTALL,
        )
        for match in pattern_model.finditer(self.raw_content):
            sub_name = match.group(2)
            sub_body = match.group(3)

            # Skip top-level model itself if matched
            if sub_name == self.model_name.split(".")[-1]:
                continue

            has_der = bool(re.search(r"\bder\s*\(", sub_body))
            has_hardcoded_ts = bool(
                re.search(r"\bConstants\.Seconds_per_hour\b", sub_body)
            )

            if has_der or has_hardcoded_ts:
                has_outer_ts = bool(
                    re.search(r"\bouter\s+parameter\s+Real\s+time_scale\b", sub_body)
                )

                # Inspect equation block specifically
                eq_match = re.search(r"\bequation\b(.*)", sub_body, re.DOTALL)
                eq_body = eq_match.group(1) if eq_match else ""

                ts_in_eq = bool(re.search(r"\btime_scale\b", eq_body))
                uses_hardcoded_in_eq = bool(
                    re.search(r"\bConstants\.Seconds_per_hour\b", eq_body)
                )

                if not has_outer_ts:
                    findings.append(
                        Finding(
                            code="TIME_SCALE_OUTER_MISSING",
                            severity=AuditSeverity.WARNING,
                            component=sub_name,
                            message=(
                                f"Component '{sub_name}' contains dynamic equations or time-scaling but lacks "
                                "'outer parameter Real time_scale' declaration."
                            ),
                            suggestion=f"Add 'outer parameter Real time_scale;' inside model/block '{sub_name}'.",
                        )
                    )
                if uses_hardcoded_in_eq:
                    findings.append(
                        Finding(
                            code="HARDCODED_TIME_CONSTANT",
                            severity=AuditSeverity.WARNING,
                            component=sub_name,
                            message=(
                                f"Component '{sub_name}' equations use hardcoded 'Constants.Seconds_per_hour' "
                                "instead of parameterized 'time_scale'."
                            ),
                            suggestion=(
                                f"Replace 'Constants.Seconds_per_hour' with 'time_scale' in '{sub_name}' equations "
                                "to ensure simulation responds to time_scale."
                            ),
                        )
                    )
                elif has_der and not ts_in_eq:
                    findings.append(
                        Finding(
                            code="TIME_SCALE_UNUSED_IN_DERIVATIVE",
                            severity=AuditSeverity.WARNING,
                            component=sub_name,
                            message=(
                                f"Component '{sub_name}' declares time_scale but does not divide der(...) by time_scale."
                            ),
                            suggestion="Ensure der(...) equations are normalized: 'der(var) = (...) / time_scale;'.",
                        )
                    )

        return findings

    def _audit_conservation_streams(self) -> List[Finding]:
        """Audits mass conservation streams (decay_rate, leak_rate, breed, burn)."""
        findings: List[Finding] = []

        # Inspect subcomponents in raw content
        pattern_model = re.compile(
            r"\b(model|block)\s+([A-Za-z0-9_]+)\b(.*?)\bend\s+\2\s*;",
            re.DOTALL,
        )

        for match in pattern_model.finditer(self.raw_content):
            sub_name = match.group(2)
            sub_body = match.group(3)

            # Check decay loss
            has_decay_loss = bool(
                re.search(r"\b(decay_loss|decay_constant|lambda)\b", sub_body)
            )
            has_decay_rate = bool(re.search(r"\bdecay_rate\b", sub_body))
            if has_decay_loss and not has_decay_rate:
                findings.append(
                    Finding(
                        code="MISSING_DECAY_RATE_STREAM",
                        severity=AuditSeverity.WARNING,
                        component=sub_name,
                        message=(
                            f"Component '{sub_name}' declares decay losses but lacks an explicit 'decay_rate' stream. "
                            "This prevents full-plant closed-loop tritium mass balance audits."
                        ),
                        suggestion=(
                            f"Declare 'Real decay_rate[5];' and assign 'decay_rate[i] = decay_loss[i]*I[i] / time_scale;' "
                            f"in model '{sub_name}'."
                        ),
                    )
                )

            # Check non-radioactive / leak loss
            has_leak_loss = bool(
                re.search(r"\b(nonradio_loss|leak_loss|loss_fraction)\b", sub_body)
            )
            has_leak_rate = bool(re.search(r"\bleak_rate\b", sub_body))
            if has_leak_loss and not has_leak_rate:
                findings.append(
                    Finding(
                        code="MISSING_LEAK_RATE_STREAM",
                        severity=AuditSeverity.WARNING,
                        component=sub_name,
                        message=(
                            f"Component '{sub_name}' declares non-radioactive/leak losses but lacks an explicit 'leak_rate' stream."
                        ),
                        suggestion=(
                            f"Declare 'Real leak_rate[5];' and calculate leak rates in model '{sub_name}'."
                        ),
                    )
                )

        return findings

    def _audit_initial_values(self) -> List[Finding]:
        """Audits whether continuous state variables have explicit start values."""
        findings: List[Finding] = []

        # Find variables that appear in der(...)
        for inv in self.introspection.inventories:
            if inv.is_state and inv.start is None:
                findings.append(
                    Finding(
                        code="MISSING_STATE_START_VALUE",
                        severity=AuditSeverity.WARNING,
                        component=inv.component or self.model_name,
                        message=(
                            f"Continuous state inventory '{inv.name}' lacks an explicit '(start = ...)' initialization attribute. "
                            "This can lead to solver divergence, initial equation singularities, or non-deterministic trajectories."
                        ),
                        suggestion=f"Add '(start = 0.0)' or appropriate initial inventory to declaration of '{inv.name}'.",
                    )
                )

        return findings

    def _audit_framework_compatibility(self) -> List[Finding]:
        """Audits compatibility with TRICYS v2.0 simulation & analysis workflows."""
        findings: List[Finding] = []

        # Verify existence of key components
        subcomp_names = {
            name.lower() for name in self.introspection.subcomponents.keys()
        }
        has_storage = any(
            name in subcomp_names for name in ("sds", "storage", "bed", "tank")
        )
        if not has_storage:
            findings.append(
                Finding(
                    code="NO_STORAGE_COMPONENT_FOUND",
                    severity=AuditSeverity.INFO,
                    component=self.model_name,
                    message=(
                        f"No recognizable tritium storage subsystem (e.g. 'sds') was identified in '{self.model_name}'. "
                        "Startup inventory and doubling time metrics require a storage node."
                    ),
                    suggestion="Ensure storage subsystem is instantiated and accessible as a state variable.",
                )
            )

        return findings

    def generate_remediation_patch(
        self, findings: Optional[List[Finding]] = None
    ) -> str:
        """Generates a standard Unified Diff patch to fix identified model issues."""
        if findings is None:
            findings = self.audit().findings

        modified_text = self._apply_findings_directly(self.raw_content, findings)

        # Generate unified diff
        orig_lines = self.raw_content.splitlines(keepends=True)
        mod_lines = modified_text.splitlines(keepends=True)

        diff = difflib.unified_diff(
            orig_lines,
            mod_lines,
            fromfile=f"a/{Path(self.package_path).name}",
            tofile=f"b/{Path(self.package_path).name}",
        )
        return "".join(diff)

    def apply_remediation(
        self,
        patch_diff: str,
        target_path: Optional[Union[str, Path]] = None,
        backup: bool = True,
    ) -> bool:
        """Applies a generated diff patch to the target file.

        Args:
            patch_diff: Unified diff string.
            target_path: Optional target destination (defaults to original package_path).
            backup: Whether to create a .bak backup file.

        Returns:
            True if applied successfully, False otherwise.
        """
        if not patch_diff.strip():
            logger.info("Empty patch; no modifications needed.")
            return True

        dest_path = Path(target_path) if target_path else Path(self.package_path)

        if backup and dest_path.exists():
            backup_path = dest_path.with_suffix(dest_path.suffix + ".bak")
            backup_path.write_text(
                dest_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            logger.info(f"Created backup at {backup_path}")

        # Apply simple line-based patch reconstructed from modified text
        report = self.audit()
        patched_text = self._apply_findings_directly(self.raw_content, report.findings)
        dest_path.write_text(patched_text, encoding="utf-8")
        logger.info(f"Successfully applied remediation patch to {dest_path}")
        return True

    def _apply_findings_directly(self, text: str, findings: List[Finding]) -> str:
        """Direct transformation applying remediation without external patch tool."""
        res = text
        if any(f.code == "TIME_SCALE_MISSING" for f in findings):
            short_name = self.model_name.split(".")[-1]
            match = re.search(rf"\bmodel\s+{re.escape(short_name)}\b.*?\n", res)
            if match:
                pos = match.end()
                code = (
                    "    inner parameter Real time_scale = 3600 "
                    '"Time-base factor: 1 = hour-based axis, 3600 = second-based axis";\n'
                )
                res = res[:pos] + code + res[pos:]

        for f in findings:
            if f.code == "TIME_SCALE_OUTER_MISSING":
                match = re.search(
                    rf"\b(model|block)\s+{re.escape(f.component)}\b.*?\n", res
                )
                if match:
                    pos = match.end()
                    code = (
                        "    outer parameter Real time_scale "
                        '"Time-base factor: 1 = hour-based axis, 3600 = second-based axis";\n'
                    )
                    res = res[:pos] + code + res[pos:]

        # Substantive remediation: Replace hardcoded Constants.Seconds_per_hour with time_scale in equations
        if re.search(r"\bConstants\.Seconds_per_hour\b", res):
            res = re.sub(r"\bConstants\.Seconds_per_hour\b", "time_scale", res)

        # Normalize derivatives lacking time_scale
        if any(
            f.code in ("TIME_SCALE_UNUSED_IN_DERIVATIVE", "HARDCODED_TIME_CONSTANT")
            for f in findings
        ):
            res = ModelAuditor._normalize_derivatives(res)

        return res

    @staticmethod
    def _normalize_derivatives(text: str) -> str:
        """Normalizes differential equations to divide by time_scale if unscaled."""
        model_pattern = re.compile(
            r"(\b(?:model|block)\s+([A-Za-z0-9_]+)\b.*?\bend\s+\2\s*;)", re.DOTALL
        )

        def fix_model(m):
            full_model = m.group(1)
            eq_match = re.search(r"(\bequation\b)(.*)", full_model, re.DOTALL)
            if not eq_match:
                return full_model

            eq_body = eq_match.group(2)

            def fix_der(der_m):
                lhs = der_m.group(1)
                rhs = der_m.group(2).strip()

                if re.search(r"\btime_scale\b", rhs):
                    return der_m.group(0)
                if rhs in ("0", "0.0", "0.00"):
                    return der_m.group(0)
                if re.search(r"/\s*(?:3600(?:\.0*)?)\b", rhs):
                    new_rhs = re.sub(r"/\s*(?:3600(?:\.0*)?)\b", "/ time_scale", rhs)
                    return f"{lhs} = {new_rhs};"
                if "Constants.Seconds_per_hour" in rhs:
                    new_rhs = rhs.replace("Constants.Seconds_per_hour", "time_scale")
                    return f"{lhs} = {new_rhs};"

                # Check if outer parenthesis already encloses the entire RHS
                is_enclosed = False
                if rhs.startswith("(") and rhs.endswith(")"):
                    count = 0
                    is_enclosed = True
                    for i, ch in enumerate(rhs):
                        if ch == "(":
                            count += 1
                        elif ch == ")":
                            count -= 1
                        if count == 0 and i < len(rhs) - 1:
                            is_enclosed = False
                            break

                if is_enclosed:
                    new_rhs = f"{rhs} / time_scale"
                else:
                    new_rhs = f"({rhs}) / time_scale"
                return f"{lhs} = {new_rhs};"

            der_pattern = re.compile(r"(\bder\s*\([^=]+\))\s*=\s*([^;]+);", re.DOTALL)
            new_eq_body = der_pattern.sub(fix_der, eq_body)

            return full_model[: eq_match.start(2)] + new_eq_body

        return model_pattern.sub(fix_model, text)


def review_model(
    package_path: Union[str, Path],
    model_name: str,
) -> ModelAuditReport:
    """Convenience function to review a Modelica model and return an audit report."""
    auditor = ModelAuditor(package_path, model_name)
    return auditor.audit()
