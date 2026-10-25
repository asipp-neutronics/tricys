from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tricys.online_cosim.oms_runtime import OmsSystemRuntime


@dataclass(frozen=True, slots=True)
class ModelComponent:
    class_name: str
    instance_name: str
    full_class_name: str = ""


@dataclass(frozen=True, slots=True)
class ModelConnection:
    source: str
    target: str


def extract_model_topology(
    omc, model_name: str
) -> tuple[list[ModelComponent], list[ModelConnection]]:
    components_raw = omc.sendExpression(f"getComponents({model_name})") or []
    components = []

    # Filter only models and blocks
    for component in components_raw:
        type_name = str(component[0])
        # Fast path for known primitives
        if (
            type_name in {"Real", "Integer", "Boolean", "String"}
            or "Units.SI" in type_name
            or "SIunits" in type_name
        ):
            continue

        restriction = omc.sendExpression(f"getClassRestriction({type_name})")
        if restriction in ("model", "block"):
            components.append(
                ModelComponent(
                    class_name=type_name.split(".")[-1],
                    instance_name=str(component[1]),
                    full_class_name=type_name,
                )
            )

    connection_count = omc.sendExpression(f"getConnectionCount({model_name})")
    connections: list[ModelConnection] = []
    if isinstance(connection_count, int):
        for index in range(1, connection_count + 1):
            connection_info = omc.sendExpression(
                f"getNthConnection({model_name}, {index})"
            )
            source, target = connection_info[0], connection_info[1]
            # Only keep connections between block ports (must contain at least one dot)
            if "." in source and "." in target:
                connections.append(ModelConnection(source=source, target=target))

    return components, connections


def build_component_type_map(components: Iterable[ModelComponent]) -> dict[str, str]:
    return {component.instance_name: component.class_name for component in components}


def add_topology_connection(
    runtime: OmsSystemRuntime,
    component_types: dict[str, str],
    source: str,
    target: str,
    system_name: str = "default",
    vector_width: int = 5,
    scalar_component_types: set[str] | None = None,
) -> None:
    source_component, source_signal = source.split(".", 1)
    target_component, target_signal = target.split(".", 1)

    try:
        # 1. Try connecting as scalar or record
        runtime.add_connection(
            runtime.cref(system_name, source_component, source_signal),
            runtime.cref(system_name, target_component, target_signal),
        )
        return
    except ValueError:
        pass

    # 2. Try connecting as an array (vector) dynamically
    connected_any = False
    index = 1
    while True:
        try:
            runtime.add_connection(
                runtime.cref(
                    system_name, source_component, f"{source_signal}[{index}]"
                ),
                runtime.cref(
                    system_name, target_component, f"{target_signal}[{index}]"
                ),
            )
            connected_any = True
            index += 1
        except ValueError:
            break

    if not connected_any:
        import logging

        logging.getLogger(__name__).warning(
            f"Failed to connect {source} and {target}: connectors not found in FMU."
        )
