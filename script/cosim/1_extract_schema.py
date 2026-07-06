import argparse
import json
import logging
import os
from typing import Any, Dict, List

from OMPython import OMCSessionZMQ

logger = logging.getLogger(__name__)


def extract_schema_from_omc(
    omc: Any, package_name: str, target_classes: List[str] = None
) -> Dict[str, Any]:
    """
    Extracts the schema from an active OMC session for the specified classes in a package.

    Args:
        omc: An active OMPython.OMCSessionZMQ instance.
        package_name: The top-level package name (e.g., "CFEDR").
        target_classes: A list of specific class names to extract (e.g., ["DIV", "I_ISS"]).
                        If None, extracts all classes in the package.

    Returns:
        A dictionary mapping class names to their schema representations.
    """
    schema = {}

    classes = omc.sendExpression(f"getClassNames({package_name})", parsed=True)
    if not classes:
        logger.warning(f"No classes found in package '{package_name}'.")
        return schema

    for cls in classes:
        if target_classes is not None and cls not in target_classes:
            continue

        full_cls_name = f"{package_name}.{cls}"
        elements_raw = omc.sendExpression(f"getElements({full_cls_name})", parsed=False)

        cls_schema = {"parameters": {}, "variables": {}, "connectors": {}}

        if not elements_raw:
            continue

        elements_strs = elements_raw.split("}, {")
        for el_str in elements_strs:
            el_str = el_str.strip("{").strip("}")
            parts = []
            in_string = False
            in_brace = 0
            current_part = ""
            for char in el_str:
                if char == '"':
                    in_string = not in_string
                    current_part += char
                elif char == "{" and not in_string:
                    in_brace += 1
                    current_part += char
                elif char == "}" and not in_string:
                    in_brace -= 1
                    current_part += char
                elif char == "," and not in_string and in_brace == 0:
                    parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
            parts.append(current_part.strip())

            if len(parts) >= 11:
                el_type = parts[2]
                name = parts[3]
                description = parts[4].strip('"')
                variability = parts[10].strip('"')  # e.g. "parameter"

                # Dimension is the 15th element (index 14) typically, or the last part
                dimension_part = parts[-1] if parts[-1].startswith("{") else "{}"
                dimension = dimension_part.strip("{}").strip()
                if not dimension:
                    dimension = None
                else:
                    dimension = [d.strip() for d in dimension.split(",")]

                # Use raw omc command for value
                val_raw = omc.sendExpression(
                    f"getComponentModifierValue({full_cls_name}, {name})", parsed=False
                )
                val = val_raw.strip() if val_raw else None
                if val == '""':
                    val = None

                el_info = {
                    "type": el_type,
                    "dimension": dimension,
                    "description": description,
                    "value": val,
                }

                if "Modelica.Blocks.Interfaces" in el_type:
                    cls_schema["connectors"][name] = el_info
                elif variability == "parameter" or variability == "constant":
                    cls_schema["parameters"][name] = el_info
                else:
                    cls_schema["variables"][name] = el_info

        schema[cls] = cls_schema

    return schema


def extract_schema(mo_file, package_name, out_json):
    mo_file = os.path.abspath(mo_file).replace("\\", "/")

    print("Starting OMC Session...")
    omc = OMCSessionZMQ()

    print(f"Loading file: {mo_file}")
    res = omc.sendExpression(f'loadFile("{mo_file}")')
    if str(res).strip().lower() != "true":
        print(f"Warning: loadFile returned {res}")

    # Get all classes in the specified package
    print(f"Extracting classes from package: {package_name}")
    schema = extract_schema_from_omc(omc, package_name)

    out_json = os.path.abspath(out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=4, ensure_ascii=False)

    print(f"Schema successfully generated at {out_json}")
    try:
        omc.sendExpression("quit()")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Modelica schema to JSON.")
    parser.add_argument("--mo_file", required=True, help="Path to the .mo file")
    parser.add_argument(
        "--package", required=True, help="Top-level package name (e.g. CFEDR)"
    )
    parser.add_argument("--out_json", required=True, help="Path to output schema.json")
    args = parser.parse_args()

    extract_schema(args.mo_file, args.package, args.out_json)
