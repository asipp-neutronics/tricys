from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PortDescriptor:
    """Descriptor for a Modelica port component (RealInput / RealOutput)."""

    name: str
    direction: str  # "input" or "output"
    dim: int = 1
    type_name: str = ""
    comment: str = ""

    def __repr__(self) -> str:
        dim_str = f"[{self.dim}]" if self.dim > 1 else ""
        return f"PortDescriptor({self.direction} {self.name}{dim_str}, type={self.type_name})"


@dataclass
class ConnectionDescriptor:
    """Descriptor for a Modelica connection involving a specific submodel instance."""

    source_path: str
    target_path: str
    instance_name: str
    port_name: str
    role_in_connection: str  # "source" or "target"
    raw_statement: str


@dataclass
class CompatibilityReport:
    """Detailed validation report comparing target and substitute submodel ports."""

    is_compatible: bool
    target_model: str
    substitute_model: str
    instance_name: str
    parent_system: str
    matched_ports: List[Dict[str, Any]] = field(default_factory=list)
    missing_ports: List[Dict[str, Any]] = field(default_factory=list)
    direction_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    dim_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    unconnected_substitute_inputs: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: str = ""


class PortCompatibilityError(ValueError):
    """Raised when a substitute submodel fails port or connection compatibility checks."""

    def __init__(self, report: CompatibilityReport):
        super().__init__(report.diagnostics)
        self.report = report


def extract_ports_from_text(
    modelica_code: str, model_name: Optional[str] = None
) -> Dict[str, PortDescriptor]:
    """Statically extracts RealInput and RealOutput port declarations from Modelica code.

    If model_name is provided, attempts to isolate the specific model block first.
    """
    code_to_parse = modelica_code
    if model_name:
        short_name = model_name.split(".")[-1]
        pattern = re.compile(
            r"\bmodel\s+"
            + re.escape(short_name)
            + r"\b(.*?)\bend\s+"
            + re.escape(short_name)
            + r"\s*;",
            re.DOTALL,
        )
        match = pattern.search(modelica_code)
        if match:
            code_to_parse = match.group(1)

    ports: Dict[str, PortDescriptor] = {}

    # Regex for RealInput and RealOutput declarations
    # Matches declarations like:
    #   Modelica.Blocks.Interfaces.RealInput from_TEP_FCU[5] "comment" annotation(...);
    #   Modelica.Blocks.Interfaces.RealOutput to_SDS[5] "comment";
    #   RealInput u;
    port_pattern = re.compile(
        r"(?:Modelica\.Blocks\.Interfaces\.)?(RealInput|RealOutput)\s+"
        r"([a-zA-Z0-9_]+)(?:\[(\d+)\])?\s*"
        r"(?:\"([^\"]*)\")?",
        re.MULTILINE,
    )

    for match in port_pattern.finditer(code_to_parse):
        port_type_kw = match.group(1)
        port_name = match.group(2)
        dim_val = int(match.group(3)) if match.group(3) else 1
        comment_val = match.group(4) or ""
        direction = "input" if port_type_kw == "RealInput" else "output"

        ports[port_name] = PortDescriptor(
            name=port_name,
            direction=direction,
            dim=dim_val,
            type_name=f"Modelica.Blocks.Interfaces.{port_type_kw}",
            comment=comment_val,
        )

    return ports


def extract_ports_from_omc(omc: Any, model_name: str) -> Dict[str, PortDescriptor]:
    """Extracts ports using active OMC session via getComponents query."""
    ports: Dict[str, PortDescriptor] = {}
    try:
        components = omc.sendExpression(f"getComponents({model_name})")
        if not components:
            return ports

        for comp in components:
            comp_type = str(comp[0])
            port_name = str(comp[1])
            comment = str(comp[2]) if len(comp) > 2 else ""
            dims = comp[11] if len(comp) > 11 else []
            dim = (
                int(dims[0])
                if (dims and len(dims) > 0 and str(dims[0]).isdigit())
                else 1
            )

            if "RealInput" in comp_type:
                ports[port_name] = PortDescriptor(
                    name=port_name,
                    direction="input",
                    dim=dim,
                    type_name=comp_type,
                    comment=comment,
                )
            elif "RealOutput" in comp_type:
                ports[port_name] = PortDescriptor(
                    name=port_name,
                    direction="output",
                    dim=dim,
                    type_name=comp_type,
                    comment=comment,
                )
    except Exception as exc:
        logger.warning(f"Failed to query OMC getComponents for {model_name}: {exc}")

    return ports


def inspect_model_ports(
    model_name: str,
    package_path: Optional[str] = None,
    code_text: Optional[str] = None,
    omc: Optional[Any] = None,
) -> Dict[str, PortDescriptor]:
    """Inspects ports of a Modelica model using OMC if available, falling back to static parsing."""
    ports: Dict[str, PortDescriptor] = {}

    # Try OMC first if session is provided
    if omc is not None:
        try:
            ports = extract_ports_from_omc(omc, model_name)
        except Exception:
            ports = {}

    if ports:
        return ports

    # Fallback to text parsing
    if code_text is None and package_path is not None and os.path.exists(package_path):
        try:
            with open(package_path, "r", encoding="utf-8", errors="replace") as f:
                code_text = f.read()
        except Exception as e:
            logger.warning(f"Could not read package file {package_path}: {e}")

    if code_text:
        ports = extract_ports_from_text(code_text, model_name=model_name)

    return ports


def extract_system_connections(
    system_code: str,
    system_model_name: Optional[str] = None,
    instance_name: Optional[str] = None,
) -> List[ConnectionDescriptor]:
    """Extracts connect(...) statements from the system model involving instance_name."""
    code_to_parse = system_code
    if system_model_name:
        short_name = system_model_name.split(".")[-1]
        pattern = re.compile(
            r"\bmodel\s+"
            + re.escape(short_name)
            + r"\b(.*?)\bend\s+"
            + re.escape(short_name)
            + r"\s*;",
            re.DOTALL,
        )
        match = pattern.search(system_code)
        if match:
            code_to_parse = match.group(1)

    connections: List[ConnectionDescriptor] = []

    # Regex for connect(endpoint1, endpoint2)
    connect_pattern = re.compile(
        r"connect\s*\(\s*([a-zA-Z0-9_\.\[\]]+)\s*,\s*([a-zA-Z0-9_\.\[\]]+)\s*\)",
        re.MULTILINE,
    )

    for match in connect_pattern.finditer(code_to_parse):
        src_raw = match.group(1).strip()
        dst_raw = match.group(2).strip()
        full_stmt = match.group(0).strip()

        # Check if instance_name is involved in either endpoint
        if instance_name:
            inst_prefix = f"{instance_name}."
            if src_raw.startswith(inst_prefix):
                port_part = src_raw[len(inst_prefix) :].split("[")[0].strip()
                connections.append(
                    ConnectionDescriptor(
                        source_path=src_raw,
                        target_path=dst_raw,
                        instance_name=instance_name,
                        port_name=port_part,
                        role_in_connection="source",
                        raw_statement=full_stmt,
                    )
                )
            elif dst_raw.startswith(inst_prefix):
                port_part = dst_raw[len(inst_prefix) :].split("[")[0].strip()
                connections.append(
                    ConnectionDescriptor(
                        source_path=src_raw,
                        target_path=dst_raw,
                        instance_name=instance_name,
                        port_name=port_part,
                        role_in_connection="target",
                        raw_statement=full_stmt,
                    )
                )
        else:
            # Add general connection
            connections.append(
                ConnectionDescriptor(
                    source_path=src_raw,
                    target_path=dst_raw,
                    instance_name="",
                    port_name="",
                    role_in_connection="general",
                    raw_statement=full_stmt,
                )
            )

    return connections


def validate_port_and_connection_compatibility(
    target_model_name: str,
    substitute_model_name: str,
    instance_name: str,
    parent_system_name: str,
    system_code_or_path: str,
    substitute_code_or_path: str,
    omc: Optional[Any] = None,
) -> CompatibilityReport:
    """Strictly validates that substitute submodel has matching upstream/downstream ports and connections.

    Rules enforced:
    1. All ports of the instance actually referenced by parent system connect(...) statements
       MUST exist in the substitute submodel.
    2. Port directions (RealInput vs RealOutput) must match system topology and target model causality.
    3. Port array dimensions (e.g. [5]) must strictly match the target model and connection requirements.
    4. Unconnected new inputs in the substitute model are identified so they can be assigned default values.

    Raises:
        PortCompatibilityError: If any port is missing, has mismatched causality, or has incompatible dimensions.
    """
    # 1. Load source texts
    system_text = (
        Path(system_code_or_path).read_text(encoding="utf-8", errors="replace")
        if os.path.isfile(system_code_or_path)
        else system_code_or_path
    )
    substitute_text = (
        Path(substitute_code_or_path).read_text(encoding="utf-8", errors="replace")
        if os.path.isfile(substitute_code_or_path)
        else substitute_code_or_path
    )

    # 2. Extract ports for target model and substitute model
    target_ports = inspect_model_ports(
        model_name=target_model_name,
        package_path=(
            system_code_or_path if os.path.isfile(system_code_or_path) else None
        ),
        code_text=system_text,
        omc=omc,
    )

    substitute_ports = inspect_model_ports(
        model_name=substitute_model_name,
        package_path=(
            substitute_code_or_path if os.path.isfile(substitute_code_or_path) else None
        ),
        code_text=substitute_text,
        omc=omc,
    )

    # 3. Extract all parent system connections for this instance
    system_conns = extract_system_connections(
        system_code=system_text,
        system_model_name=parent_system_name,
        instance_name=instance_name,
    )

    matched_ports: List[Dict[str, Any]] = []
    missing_ports: List[Dict[str, Any]] = []
    direction_mismatches: List[Dict[str, Any]] = []
    dim_mismatches: List[Dict[str, Any]] = []
    unconnected_inputs: List[Dict[str, Any]] = []

    # Map of ports referenced in parent connections
    referenced_ports: set[str] = set()

    for conn in system_conns:
        port_name = conn.port_name
        referenced_ports.add(port_name)

        # Target port baseline
        target_port = target_ports.get(port_name)
        subst_port = substitute_ports.get(port_name)

        if not subst_port:
            missing_ports.append(
                {
                    "port_name": port_name,
                    "instance": instance_name,
                    "expected_direction": (
                        target_port.direction if target_port else "unknown"
                    ),
                    "expected_dim": target_port.dim if target_port else 1,
                    "connection": conn.raw_statement,
                    "reason": f"Port '{port_name}' is wired in parent system connection but missing in substitute model '{substitute_model_name}'",
                }
            )
            continue

        # Check direction
        if target_port:
            if subst_port.direction != target_port.direction:
                direction_mismatches.append(
                    {
                        "port_name": port_name,
                        "expected": target_port.direction,
                        "actual": subst_port.direction,
                        "connection": conn.raw_statement,
                        "reason": f"Port '{port_name}' direction mismatch: expected '{target_port.direction}', found '{subst_port.direction}'",
                    }
                )
                continue

            # Check dimension
            if subst_port.dim != target_port.dim:
                dim_mismatches.append(
                    {
                        "port_name": port_name,
                        "expected_dim": target_port.dim,
                        "actual_dim": subst_port.dim,
                        "connection": conn.raw_statement,
                        "reason": f"Port '{port_name}' dimension mismatch: expected dim={target_port.dim}, found dim={subst_port.dim}",
                    }
                )
                continue

        matched_ports.append(
            {
                "port_name": port_name,
                "direction": subst_port.direction,
                "dim": subst_port.dim,
                "connection": conn.raw_statement,
            }
        )

    # Also check any target model ports that might not have direct connections but define public interface
    for target_pname, target_p in target_ports.items():
        if (
            target_pname not in referenced_ports
            and target_pname not in substitute_ports
        ):
            # Warning only for unreferenced ports, or flag if needed
            logger.debug(
                f"Notice: Target port '{target_pname}' is defined in {target_model_name} but not in {substitute_model_name} (unconnected in system)"
            )

    # Detect any new input ports in substitute that are NOT connected in parent system
    for subst_pname, subst_p in substitute_ports.items():
        if subst_p.direction == "input" and subst_pname not in referenced_ports:
            unconnected_inputs.append(
                {
                    "port_name": subst_pname,
                    "dim": subst_p.dim,
                    "comment": subst_p.comment,
                }
            )

    is_compatible = (
        len(missing_ports) == 0
        and len(direction_mismatches) == 0
        and len(dim_mismatches) == 0
    )

    # Build detailed diagnostic report
    diagnostics_lines = [
        "=" * 80,
        f"TRICYS SUBMODEL COMPATIBILITY AUDIT REPORT: {'[PASSED]' if is_compatible else '[FAILED]'}",
        "=" * 80,
        f"Parent System Model : {parent_system_name}",
        f"Target Submodel      : {target_model_name} (instance: '{instance_name}')",
        f"Substitute Submodel  : {substitute_model_name}",
        f"Referenced Ports     : {len(referenced_ports)} port(s) audited in connections",
        f"Matched Connections  : {len(matched_ports)} port(s) fully compatible",
    ]

    if not is_compatible:
        diagnostics_lines.append("-" * 80)
        diagnostics_lines.append("CRITICAL INCOMPATIBILITIES DETECTED:")
        if missing_ports:
            diagnostics_lines.append("  [Missing Required Ports]:")
            for item in missing_ports:
                diagnostics_lines.append(
                    f"    - {item['port_name']} (expected: {item['expected_direction']}[{item['expected_dim']}])"
                )
                diagnostics_lines.append(f"      Connection: {item['connection']}")
        if direction_mismatches:
            diagnostics_lines.append("  [Direction Mismatches]:")
            for item in direction_mismatches:
                diagnostics_lines.append(
                    f"    - {item['port_name']}: expected '{item['expected']}', got '{item['actual']}'"
                )
                diagnostics_lines.append(f"      Connection: {item['connection']}")
        if dim_mismatches:
            diagnostics_lines.append("  [Dimension Mismatches]:")
            for item in dim_mismatches:
                diagnostics_lines.append(
                    f"    - {item['port_name']}: expected dim={item['expected_dim']}, got dim={item['actual_dim']}"
                )
                diagnostics_lines.append(f"      Connection: {item['connection']}")
        diagnostics_lines.append("-" * 80)
        diagnostics_lines.append(
            "Remediation: Modify the substitute Modelica model to match all interface port names, "
            "causality directions, and vector dimensions before attempting parameter injection."
        )

    if unconnected_inputs:
        diagnostics_lines.append("-" * 80)
        diagnostics_lines.append(
            f"Note: Found {len(unconnected_inputs)} unconnected input port(s) in substitute submodel: "
            + ", ".join(f"{p['port_name']}[{p['dim']}]" for p in unconnected_inputs)
        )
        diagnostics_lines.append(
            "These will automatically be assigned default zero bindings in adapter equations."
        )

    diagnostics_lines.append("=" * 80)
    diagnostics_text = "\n".join(diagnostics_lines)

    report = CompatibilityReport(
        is_compatible=is_compatible,
        target_model=target_model_name,
        substitute_model=substitute_model_name,
        instance_name=instance_name,
        parent_system=parent_system_name,
        matched_ports=matched_ports,
        missing_ports=missing_ports,
        direction_mismatches=direction_mismatches,
        dim_mismatches=dim_mismatches,
        unconnected_substitute_inputs=unconnected_inputs,
        diagnostics=diagnostics_text,
    )

    if not is_compatible:
        logger.error(diagnostics_text)
        raise PortCompatibilityError(report)

    logger.info(
        f"Submodel replacement port compatibility check PASSED for instance '{instance_name}' ({target_model_name} -> {substitute_model_name})"
    )
    return report


def replace_submodel_in_package(
    package_path: str,
    target_model_name: str,
    substitute_model_name: str,
    substitute_code_or_file: Optional[str] = None,
    instance_name: Optional[str] = None,
    unconnected_inputs: Optional[List[Dict[str, Any]]] = None,
    output_package_path: Optional[str] = None,
) -> str:
    """Seamlessly replaces target submodel in Modelica package code.

    Supports:
    1. Single-file package: Modifies in-place or writes to output_package_path,
       replacing `model Target ... end Target;` with adapter inheritance:
       `model Target extends Substitute; ... end Target;`
    2. Multi-file package: Replaces Target.mo with replacement code.
    """
    output_path = output_package_path or package_path
    target_short_name = target_model_name.split(".")[-1]

    with open(package_path, "r", encoding="utf-8", errors="replace") as f:
        package_code = f.read()

    # Build default equation bindings for any unconnected substitute inputs
    equations_code = ""
    if unconnected_inputs:
        eq_lines = []
        for inp in unconnected_inputs:
            pname = inp["port_name"]
            dim = inp.get("dim", 1)
            if dim > 1:
                zeros = ", ".join(["0"] * dim)
                eq_lines.append(f"    {pname} = {{{zeros}}};")
            else:
                eq_lines.append(f"    {pname} = 0.0;")
        if eq_lines:
            equations_code = "\n  equation\n" + "\n".join(eq_lines)

    # Replacement submodel declaration
    new_submodel_code = (
        f'  model {target_short_name} "Replaced by TRICYS parameter injection"\n'
        f"    extends {substitute_model_name};\n"
        f"{equations_code}\n"
        f"  end {target_short_name};"
    )

    # Match and replace existing submodel in package
    submodel_pattern = re.compile(
        r"(\bmodel\s+"
        + re.escape(target_short_name)
        + r"\b.*?end\s+"
        + re.escape(target_short_name)
        + r"\s*;)",
        re.DOTALL,
    )

    if submodel_pattern.search(package_code):
        modified_code = submodel_pattern.sub(new_submodel_code, package_code, count=1)
        logger.info(
            f"Replaced submodel '{target_short_name}' using extends {substitute_model_name}"
        )
    else:
        # Fallback: append before package end
        package_name = target_model_name.split(".")[0]
        end_pkg_pattern = re.compile(
            r"(\bend\s+" + re.escape(package_name) + r"\s*;)", re.IGNORECASE
        )
        if end_pkg_pattern.search(package_code):
            modified_code = end_pkg_pattern.sub(
                f"{new_submodel_code}\n\n\\1", package_code, count=1
            )
            logger.info(
                f"Appended replacement submodel '{target_short_name}' before package end"
            )
        else:
            raise ValueError(
                f"Could not locate model '{target_short_name}' or end of package in {package_path}"
            )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(modified_code)

    return output_path


def execute_subsystem_handler(
    handler_script_path: str,
    params: Dict[str, Any],
    workspace_dir: str,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """Executes external handler script to perform high-fidelity submodel solve (e.g. h2iso)."""
    script_path = Path(handler_script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Subsystem handler script not found: {script_path}")

    logger.info(f"Executing external subsystem handler: {script_path.name}")
    cmd = [sys.executable, str(script_path)]

    # Pass parameters as environment or args
    env = os.environ.copy()
    env["TRICYS_WORKSPACE"] = workspace_dir
    import json

    env["TRICYS_HANDLER_PARAMS"] = json.dumps(params)

    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        logger.info(f"Subsystem handler finished successfully: {proc.stdout.strip()}")
        return {"stdout": proc.stdout, "status": "success"}
    except subprocess.CalledProcessError as exc:
        logger.error(f"Subsystem handler script failed: {exc.stderr}")
        raise RuntimeError(
            f"Subsystem handler {script_path.name} failed: {exc.stderr}"
        ) from exc
