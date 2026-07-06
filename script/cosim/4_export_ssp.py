import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable

from OMPython import OMCSessionZMQ


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
    model: Any,
    component_types: dict[str, str],
    source: str,
    target: str,
    system_name: str = "default",
    vector_width: int = 5,
    scalar_component_types: set[str] | None = None,
) -> None:
    from OMSimulator import CRef

    source_component, source_signal = source.split(".", 1)
    target_component, target_signal = target.split(".", 1)

    try:
        # 1. Try connecting as scalar or record
        model.addConnection(
            CRef(system_name, source_component, source_signal),
            CRef(system_name, target_component, target_signal),
        )
        return
    except ValueError:
        pass

    # 2. Try connecting as an array (vector) dynamically
    connected_any = False
    index = 1
    while True:
        try:
            model.addConnection(
                CRef(system_name, source_component, f"{source_signal}[{index}]"),
                CRef(system_name, target_component, f"{target_signal}[{index}]"),
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


def export_ssp(mo_file, model_name, fmu_dir, out_ssp):
    mo_file = os.path.abspath(mo_file).replace("\\", "/")
    fmu_dir = os.path.abspath(fmu_dir)
    out_ssp = os.path.abspath(out_ssp)

    print("Starting OMC Session...")
    omc = OMCSessionZMQ()

    print(f"Loading file: {mo_file}")
    res = omc.sendExpression(f'loadFile("{mo_file}")')
    if str(res).strip().lower() != "true":
        print(f"Warning: loadFile returned {res}")

    print(f"Extracting topology for model: {model_name}")
    try:
        components, connections = extract_model_topology(omc, model_name)
    except Exception as e:
        print(f"Error extracting topology: {e}")
        sys.exit(1)

    component_types = build_component_type_map(components)

    try:
        from OMSimulator import SSP, CRef
    except ImportError:
        print("Error: OMSimulator Python bindings are required to export SSP")
        sys.exit(1)

    model = SSP()

    system_name = "default"
    added_resources = set()
    for component in components:
        fmu_path = os.path.join(fmu_dir, f"{component.class_name}.fmu")
        if not os.path.exists(fmu_path):
            print(
                f"Warning: Required FMU not found for component {component.full_class_name} at {fmu_path}. Attempting to build..."
            )
            omc_cwd = omc.sendExpression("cd()")
            omc.sendExpression(f'cd("{fmu_dir.replace(chr(92), "/")}")')
            build_cmd = f'buildModelFMU({component.full_class_name}, version="2.0", fmuType="cs")'
            try:
                gen_path = omc.sendExpression(build_cmd, parsed=True)
                if not gen_path:
                    error = omc.sendExpression("getErrorString()")
                    print(
                        f"Error: Failed to auto-build FMU for {component.full_class_name}. OMC Error: {error}"
                    )
                    sys.exit(1)
                elif not os.path.exists(fmu_path):
                    # sometimes OMC returns a path like Modelica.Blocks.Sources.Pulse.fmu which isn't the class_name?
                    # class_name is 'Pulse', full_class_name is 'Modelica.Blocks.Sources.Pulse'
                    if isinstance(gen_path, str) and os.path.exists(
                        os.path.join(fmu_dir, gen_path)
                    ):
                        import shutil

                        shutil.move(os.path.join(fmu_dir, gen_path), fmu_path)
            except Exception as e:
                print(
                    f"Error: Exception during auto-build of {component.full_class_name}: {e}"
                )
                sys.exit(1)
            finally:
                if omc_cwd:
                    omc.sendExpression(f'cd("{omc_cwd.replace(chr(92), "/")}")')

                # Cleanup intermediate files generated by auto-build
                import shutil

                for item in os.listdir(fmu_dir):
                    if not item.endswith(".fmu"):
                        item_path = os.path.join(fmu_dir, item)
                        try:
                            if os.path.isfile(item_path) or os.path.islink(item_path):
                                os.remove(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                        except Exception:
                            pass

        if not os.path.exists(fmu_path):
            print(
                f"Error: Required FMU still not found for component {component.full_class_name} at {fmu_path}"
            )
            sys.exit(1)

        resource_name = f"resources/{component.class_name}.fmu"
        if resource_name not in added_resources:
            model.addResource(fmu_path.replace("\\", "/"), new_name=resource_name)
            added_resources.add(resource_name)
        model.addComponent(CRef(system_name, component.instance_name), resource_name)

    for connection in connections:
        add_topology_connection(
            model,
            component_types,
            connection.source,
            connection.target,
            system_name=system_name,
        )

    os.makedirs(os.path.dirname(out_ssp), exist_ok=True)
    model.export(out_ssp)

    # Patch the SSP to be compatible with OMEdit (which rejects version="2.0" and fails on some namespaces)
    try:
        import re
        import shutil
        import tempfile
        import zipfile

        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(out_ssp, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        ssd_path = os.path.join(temp_dir, "SystemStructure.ssd")
        if os.path.exists(ssd_path):
            with open(ssd_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Replace version="2.0" with version="1.0"
            content = re.sub(r'version="2\.0"', 'version="1.0"', content)

            # Strip out unused namespaces that crash OMEdit's libxml2 due to Windows path space bugs
            content = re.sub(r'xmlns:ssv="[^"]+"', "", content)
            content = re.sub(r'xmlns:ssm="[^"]+"', "", content)
            content = re.sub(r'xmlns:ssb="[^"]+"', "", content)
            content = re.sub(r'xmlns:oms="[^"]+"', "", content)

            with open(ssd_path, "w", encoding="utf-8") as f:
                f.write(content)

        # Re-zip
        with zipfile.ZipFile(out_ssp, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Failed to patch SSP for OMEdit compatibility: {e}")

    print(f"SSP successfully exported and patched for OMEdit at {out_ssp}")

    try:
        omc.sendExpression("quit()")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SSP from Modelica model.")
    parser.add_argument("--mo_file", required=True, help="Path to the .mo file")
    parser.add_argument(
        "--model_name", required=True, help="Top-level model name (e.g. CFEDR.Cycle)"
    )
    parser.add_argument(
        "--fmu_dir", required=True, help="Directory containing the built FMUs"
    )
    parser.add_argument("--out_ssp", required=True, help="Path to output .ssp file")
    args = parser.parse_args()

    export_ssp(args.mo_file, args.model_name, args.fmu_dir, args.out_ssp)
