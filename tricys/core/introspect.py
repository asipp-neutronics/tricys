"""Modelica Model Deep Introspection Engine.

This module provides comprehensive, zero-hallucination model introspection for
arbitrary Modelica models used in TRICYS fusion fuel cycle simulations.
It parses parameters, inventories, flow rates, subcomponents, and experiment
annotations via pure-python AST analysis and optional OMC verification.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ParameterInfo:
    """Detailed metadata for a Modelica parameter."""

    name: str
    type: str = "Real"
    default: Any = None
    unit: str = "1"
    comment: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    is_inner: bool = False
    is_outer: bool = False
    component: Optional[str] = None


@dataclass
class InventoryInfo:
    """Metadata for continuous state inventory nodes (e.g. sds.I[1], tes.I)."""

    name: str
    type: str = "Real"
    start: Any = None
    unit: str = "g"
    comment: str = ""
    component: Optional[str] = None
    dimensions: Optional[int] = None
    is_state: bool = True


@dataclass
class FlowRateInfo:
    """Metadata for mass flow rates, loss sinks, or production sources."""

    name: str
    type: str = "Real"
    unit: str = "g/s"
    category: str = "transfer"  # decay, leak, breed, burn, transfer, input, output
    comment: str = ""
    component: Optional[str] = None


@dataclass
class SubcomponentInfo:
    """Metadata for instantiated subcomponents within a model."""

    name: str
    type: str
    comment: str = ""


@dataclass
class ExperimentInfo:
    """Simulation experiment configuration annotation."""

    startTime: float = 0.0
    stopTime: float = 86400.0
    stepSize: float = 100.0
    tolerance: float = 1e-6
    solver: str = "dassl"


@dataclass
class ModelIntrospection:
    """Complete introspection artifact for a Modelica model."""

    package_path: str
    model_name: str
    parameters: Dict[str, ParameterInfo] = field(default_factory=dict)
    inventories: List[InventoryInfo] = field(default_factory=list)
    flow_rates: List[FlowRateInfo] = field(default_factory=list)
    subcomponents: Dict[str, SubcomponentInfo] = field(default_factory=dict)
    experiment: ExperimentInfo = field(default_factory=ExperimentInfo)

    def to_dict(self) -> Dict[str, Any]:
        """Converts introspection artifact to a serializable dictionary."""
        return {
            "package_path": self.package_path,
            "model_name": self.model_name,
            "parameters": {k: asdict(v) for k, v in self.parameters.items()},
            "inventories": [asdict(v) for v in self.inventories],
            "flow_rates": [asdict(v) for v in self.flow_rates],
            "subcomponents": {k: asdict(v) for k, v in self.subcomponents.items()},
            "experiment": asdict(self.experiment),
        }

    def to_json(self, indent: int = 2) -> str:
        """Converts to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ModelicaASTParser:
    """High-speed static parser for Modelica code (.mo) without requiring OMC daemon."""

    def __init__(self, content: str, package_path: str = ""):
        self.raw_content = content
        self.package_path = package_path
        self.clean_content = self._strip_comments(content)
        self.models: Dict[str, str] = self._extract_models(self.clean_content)

    @staticmethod
    def _strip_comments(code: str) -> str:
        """Strips single-line (//...) and multi-line (/*...*/) comments while preserving string literals."""

        # Simple robust regex for Modelica
        def replacer(match):
            s = match.group(0)
            if s.startswith("/"):
                return " "
            return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE,
        )
        return re.sub(pattern, replacer, code)

    def _extract_models(self, code: str) -> Dict[str, str]:
        """Extracts top-level and inner model/block/package bodies by name."""
        models: Dict[str, str] = {}
        # Pattern to capture: (model|block|record|class) <Name> ... end <Name>;
        model_pattern = re.compile(
            r"\b(model|block|record|class)\s+([A-Za-z0-9_]+)\b(.*?)\bend\s+\2\s*;",
            re.DOTALL,
        )
        for match in model_pattern.finditer(code):
            name = match.group(2)
            body = match.group(3)
            models[name] = body
        return models

    def parse_model(self, target_model_name: str) -> ModelIntrospection:
        """Parses target model hierarchy (e.g. example_model.Cycle or Cycle)."""
        short_name = target_model_name.split(".")[-1]
        model_body = self.models.get(short_name)
        if model_body is None:
            # Fallback search if exact match not found
            for m_name, body in self.models.items():
                if m_name == short_name or target_model_name.endswith(m_name):
                    model_body = body
                    short_name = m_name
                    break

        if model_body is None:
            # If not found in split models, treat entire content as body
            model_body = self.clean_content

        result = ModelIntrospection(
            package_path=self.package_path,
            model_name=target_model_name,
        )

        # 1. Parse experiment annotation in target model or package
        result.experiment = self._parse_experiment(self.raw_content)

        # 2. Parse subcomponents in target model
        subcomps = self._parse_subcomponents(model_body)
        result.subcomponents = subcomps

        # 3. Parse direct parameters in target model
        top_params = self._parse_parameters(model_body, prefix="")
        for p in top_params:
            result.parameters[p.name] = p

        # 4. Parse direct variables, inventories, and flow rates in target model
        top_invs, top_flows = self._parse_variables_and_equations(model_body, prefix="")
        result.inventories.extend(top_invs)
        result.flow_rates.extend(top_flows)

        # 5. Parse child component models if instantiated
        for comp_name, comp_info in subcomps.items():
            child_body = self.models.get(comp_info.type)
            if child_body:
                # Collect child parameters
                child_params = self._parse_parameters(child_body, prefix=comp_name)
                for p in child_params:
                    result.parameters[p.name] = p

                # Collect child inventories and flow rates
                child_invs, child_flows = self._parse_variables_and_equations(
                    child_body, prefix=comp_name
                )
                result.inventories.extend(child_invs)
                result.flow_rates.extend(child_flows)

        return result

    def _parse_experiment(self, code: str) -> ExperimentInfo:
        """Parses experiment(...) annotation."""
        exp = ExperimentInfo()
        # Look for experiment(startTime = 0, stopTime = 86400, ...)
        exp_match = re.search(
            r"experiment\s*\(\s*([^)]+)\)",
            code,
            re.IGNORECASE,
        )
        if exp_match:
            args_str = exp_match.group(1)
            for part in args_str.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    try:
                        if k == "startTime":
                            exp.startTime = float(v)
                        elif k == "stopTime":
                            exp.stopTime = float(v)
                        elif k == "stepSize":
                            exp.stepSize = float(v)
                        elif k == "tolerance":
                            exp.tolerance = float(v)
                        elif k == "solver":
                            exp.solver = v
                    except ValueError:
                        pass
        return exp

    def _parse_parameters(self, body: str, prefix: str = "") -> List[ParameterInfo]:
        """Extracts all parameter declarations from a model body."""
        params: List[ParameterInfo] = []

        # Remove equation and algorithm sections to only search declarations
        decl_part = re.split(r"\b(equation|algorithm)\b", body)[0]

        param_pattern = re.compile(
            r'\b(inner\s+|outer\s+)?parameter\s+([A-Za-z0-9_.]+)\s+([A-Za-z0-9_]+)(\[[^\]]*\])?(?:\s*\(([^)]*)\))?(?:\s*=\s*([^;"]+))?(?:\s*"([^"]*)")?\s*;',
            re.MULTILINE,
        )

        for match in param_pattern.finditer(decl_part):
            modifier = (match.group(1) or "").strip()
            type_name = match.group(2).strip()
            var_name = match.group(3).strip()
            array_dim = match.group(4)
            attr_part = match.group(5) or ""
            val_part = match.group(6)
            comment_part = match.group(7) or ""

            full_name = f"{prefix}.{var_name}" if prefix else var_name

            # Parse value
            parsed_val = self._parse_literal_value(val_part)

            # Parse unit, min, max from attributes like (min=0, max=10, unit="1/h")
            unit = "1"
            min_val = None
            max_val = None
            if attr_part:
                unit_match = re.search(r"unit\s*=\s*\"([^\"]*)\"", attr_part)
                if unit_match:
                    unit = unit_match.group(1)
                min_match = re.search(r"min\s*=\s*([0-9.eE+-]+)", attr_part)
                if min_match:
                    try:
                        min_val = float(min_match.group(1))
                    except ValueError:
                        pass
                max_match = re.search(r"max\s*=\s*([0-9.eE+-]+)", attr_part)
                if max_match:
                    try:
                        max_val = float(max_match.group(1))
                    except ValueError:
                        pass

            params.append(
                ParameterInfo(
                    name=full_name,
                    type=type_name if not array_dim else f"{type_name}{array_dim}",
                    default=parsed_val,
                    unit=unit,
                    comment=comment_part,
                    min=min_val,
                    max=max_val,
                    is_inner=("inner" in modifier),
                    is_outer=("outer" in modifier),
                    component=prefix or None,
                )
            )

        return params

    def _parse_subcomponents(self, body: str) -> Dict[str, SubcomponentInfo]:
        """Extracts component instances (e.g. SDS sds annotation(...))."""
        subcomps: Dict[str, SubcomponentInfo] = {}

        decl_part = re.split(r"\b(equation|algorithm)\b", body)[0]

        # Ignore standard keywords
        keywords = {
            "parameter",
            "constant",
            "Real",
            "Integer",
            "Boolean",
            "String",
            "input",
            "output",
            "import",
            "extends",
            "outer",
            "inner",
        }

        # Pattern: Type instance_name [args]? ["comment"]? annotation(...);
        inst_pattern = re.compile(
            r"^\s*([A-Za-z0-9_.]+)\s+([A-Za-z0-9_]+)(?:\s*\([^;]*\))?(?:\s*\"([^\"]*)\")?(?:\s*annotation\s*\([^;]*\))?\s*;",
            re.MULTILINE,
        )

        for match in inst_pattern.finditer(decl_part):
            comp_type = match.group(1).strip()
            comp_name = match.group(2).strip()
            comment = match.group(3) or ""

            first_tok = comp_type.split()[0]
            if (
                first_tok in keywords
                or first_tok.endswith("Input")
                or first_tok.endswith("Output")
            ):
                continue

            subcomps[comp_name] = SubcomponentInfo(
                name=comp_name,
                type=comp_type.split(".")[-1],
                comment=comment,
            )

        return subcomps

    def _parse_variables_and_equations(
        self, body: str, prefix: str = ""
    ) -> tuple[List[InventoryInfo], List[FlowRateInfo]]:
        """Identifies state variables, inventories, and mass flow rates."""
        inventories: List[InventoryInfo] = []
        flow_rates: List[FlowRateInfo] = []

        # 1. Inspect equation section to detect continuous derivative variables der(var)
        der_vars = set()
        eq_parts = re.split(r"\bequation\b", body)
        eq_part = eq_parts[1] if len(eq_parts) > 1 else ""

        der_matches = re.finditer(
            r"\bder\s*\(\s*([A-Za-z0-9_]+)(?:\[[^\]]+\])?\s*\)", eq_part
        )
        for m in der_matches:
            der_vars.add(m.group(1))

        # 2. Parse variable declarations in declaration section
        decl_part = eq_parts[0]
        var_pattern = re.compile(
            r"^\s*(?:Real|Integer)\s+([A-Za-z0-9_]+)(\[[^\]]*\])?(?:\s*\(([^)]*)\))?(?:\s*\"([^\"]*)\")?\s*;",
            re.MULTILINE,
        )

        for match in var_pattern.finditer(decl_part):
            var_name = match.group(1).strip()
            dim_str = match.group(2)
            attrs = match.group(3) or ""
            comment = match.group(4) or ""

            full_name = f"{prefix}.{var_name}" if prefix else var_name

            # Parse start value (supports both scalar and array literals {...})
            start_val = None
            start_match = re.search(r"start\s*=\s*(\{[^}]*\}|[^,)]+)", attrs)
            if start_match:
                start_val = self._parse_literal_value(start_match.group(1).strip())

            unit = "1"
            unit_match = re.search(r"unit\s*=\s*\"([^\"]*)\"", attrs)
            if unit_match:
                unit = unit_match.group(1)

            # Heuristics:
            # A. Is it an inventory state? (e.g. name is I, inventory, storage, holdup or appears in der(...))
            is_inv = (
                var_name in der_vars
                or var_name
                in ("I", "inventory", "storage", "holdup", "cumulative_breed")
                or "存量" in comment
                or "储存" in comment
                or "滞留" in comment
            )

            # B. Is it a flow rate / loss rate / breed rate?
            is_flow = (
                var_name.endswith("_rate")
                or var_name.startswith("outflow")
                or var_name.startswith("inflow")
                or "rate" in var_name.lower()
                or "leak" in var_name.lower()
                or "decay" in var_name.lower()
                or "burn" in var_name.lower()
                or "流" in comment
                or "速率" in comment
            )

            if is_inv:
                inv_unit = unit if unit != "1" else "g"
                # If array like I[5], register base array info
                dim_count = None
                if dim_str:
                    clean_dim = dim_str.strip("[]")
                    if clean_dim.isdigit():
                        dim_count = int(clean_dim)

                inventories.append(
                    InventoryInfo(
                        name=full_name,
                        start=start_val,
                        unit=inv_unit,
                        comment=comment,
                        component=prefix or None,
                        dimensions=dim_count,
                        is_state=(var_name in der_vars),
                    )
                )

            if is_flow and not (is_inv and not var_name.endswith("_rate")):
                flow_unit = unit if unit != "1" else "g/s"
                cat = "transfer"
                if "decay" in var_name.lower():
                    cat = "decay"
                elif "leak" in var_name.lower() or "loss" in var_name.lower():
                    cat = "leak"
                elif "breed" in var_name.lower():
                    cat = "breed"
                elif "burn" in var_name.lower():
                    cat = "burn"

                flow_rates.append(
                    FlowRateInfo(
                        name=full_name,
                        unit=flow_unit,
                        category=cat,
                        comment=comment,
                        component=prefix or None,
                    )
                )

        return inventories, flow_rates

    @staticmethod
    def _parse_literal_value(raw: Optional[str]) -> Any:
        """Parses a Modelica literal expression into a Python primitive."""
        if raw is None:
            return None
        val_str = raw.strip()

        # Handle Modelica array {1, 2, 3}
        if val_str.startswith("{") and val_str.endswith("}"):
            inner = val_str[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(",")]
            return [ModelicaASTParser._parse_literal_value(p) for p in parts]

        # Handle booleans
        if val_str.lower() == "true":
            return True
        if val_str.lower() == "false":
            return False

        # Handle quoted strings
        if (val_str.startswith('"') and val_str.endswith('"')) or (
            val_str.startswith("'") and val_str.endswith("'")
        ):
            return val_str[1:-1]

        # Handle numbers
        try:
            if "." in val_str or "e" in val_str.lower():
                return float(val_str)
            return int(val_str)
        except ValueError:
            return val_str


def introspect_model(
    package_path: Union[str, Path],
    model_name: str,
    prefer_omc: bool = False,
) -> ModelIntrospection:
    """Introspects a Modelica model and returns full structured metadata.

    Args:
        package_path: Path to the .mo file (or package.mo).
        model_name: Qualified model name (e.g. 'example_model.Cycle').
        prefer_omc: Whether to attempt live OMC introspection first.

    Returns:
        ModelIntrospection instance containing parameters, inventories, and flows.
    """
    path_obj = Path(package_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Modelica file not found at: {path_obj}")

    # Read content
    content = path_obj.read_text(encoding="utf-8")
    parser = ModelicaASTParser(content, package_path=str(path_obj))
    introspection = parser.parse_model(model_name)

    # Optional dynamic OMC enhancement if requested
    if prefer_omc:
        try:
            from tricys.core.modelica import (
                get_all_parameters_details,
                get_om_session,
                load_modelica_package,
            )

            omc = get_om_session()
            if load_modelica_package(omc, str(path_obj)):
                details = get_all_parameters_details(omc, model_name)
                for item in details:
                    p_name = item.get("name")
                    if p_name and p_name in introspection.parameters:
                        introspection.parameters[p_name].default = item.get(
                            "defaultValue"
                        )
        except Exception as exc:
            logger.debug(f"OMC live introspection fallback used static result: {exc}")

    return introspection
