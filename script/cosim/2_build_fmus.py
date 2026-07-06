import argparse
import json
import os
import shutil

from OMPython import OMCSessionZMQ


def build_fmus(mo_file, package_name, schema_json, out_dir):
    mo_file = os.path.abspath(mo_file).replace("\\", "/")
    schema_json = os.path.abspath(schema_json)
    out_dir = os.path.abspath(out_dir).replace("\\", "/")

    with open(schema_json, "r", encoding="utf-8") as f:
        schema = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    print("Starting OMC Session...")
    omc = OMCSessionZMQ()

    # CD into output directory so FMUs are generated there
    res = omc.sendExpression(f'cd("{out_dir}")')
    print(f"OMC Working Directory: {res}")

    print(f"Loading file: {mo_file}")
    res = omc.sendExpression(f'loadFile("{mo_file}")')
    if str(res).strip().lower() != "true":
        print(f"Warning: loadFile returned {res}")

    for cls in schema.keys():
        full_cls_name = f"{package_name}.{cls}"
        print(f"Building FMU for: {full_cls_name}")

        # We build CS (Co-Simulation) FMUs for FMI 2.0
        build_cmd = f'buildModelFMU({full_cls_name}, version="2.0", fmuType="cs")'
        try:
            fmu_path = omc.sendExpression(build_cmd, parsed=True)
            if fmu_path:
                print(f"  -> Successfully generated: {fmu_path}")
            else:
                # Check errors
                error = omc.sendExpression("getErrorString()")
                print(f"  -> Failed to generate FMU. Error: {error}")
        except Exception as e:
            print(
                f"  -> Skipped or Failed to generate FMU for {full_cls_name}. Reason: {e}"
            )
            continue

    try:
        omc.sendExpression("quit()")
    except Exception:
        pass

    print(f"\nCleaning up intermediate files in {out_dir}...")
    for item in os.listdir(out_dir):
        if not item.endswith(".fmu"):
            item_path = os.path.join(out_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"  -> Warning: Failed to remove {item}: {e}")

    print(f"\nAll FMUs have been built in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FMUs for components in schema.")
    parser.add_argument("--mo_file", required=True, help="Path to the .mo file")
    parser.add_argument(
        "--package", required=True, help="Top-level package name (e.g. CFEDR)"
    )
    parser.add_argument("--schema", required=True, help="Path to the schema.json")
    parser.add_argument(
        "--out_dir", required=True, help="Directory to save generated FMUs"
    )
    args = parser.parse_args()

    build_fmus(args.mo_file, args.package, args.schema, args.out_dir)
