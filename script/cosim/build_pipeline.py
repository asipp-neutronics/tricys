import argparse
import os
import subprocess
import sys


def run_step(name, cmd):
    print(f"\n{'='*60}")
    print(f"Running Step: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: Step '{name}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"SUCCESS: Step '{name}' completed successfully.\n")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for extracting schema, building FMUs, and generating processors."
    )
    parser.add_argument("--mo_file", required=True, help="Path to the source .mo file")
    parser.add_argument(
        "--package",
        required=True,
        help="Top-level package name in the .mo file (e.g. CFEDR)",
    )
    parser.add_argument(
        "--out_fmu_dir", required=True, help="Directory to save the built .fmu files"
    )
    parser.add_argument(
        "--out_proc_dir",
        required=True,
        help="Directory to save the generated processor base files",
    )
    parser.add_argument(
        "--out_schema",
        default="schema.json",
        help="Path to save the intermediate schema.json (default: schema.json in cwd)",
    )
    parser.add_argument(
        "--model_name",
        help="Top-level model name (e.g. CFEDR.Cycle). If provided, step 4 will export an SSP file.",
    )
    parser.add_argument(
        "--out_ssp_dir",
        help="Directory to save the exported .ssp file (defaults to the same directory as --mo_file)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable

    script1 = os.path.join(script_dir, "1_extract_schema.py")
    script2 = os.path.join(script_dir, "2_build_fmus.py")
    script3 = os.path.join(script_dir, "3_generate_processors.py")

    # Step 1: Extract Schema
    run_step(
        "1. Extract Schema",
        [
            python_exe,
            script1,
            "--mo_file",
            args.mo_file,
            "--package",
            args.package,
            "--out_json",
            args.out_schema,
        ],
    )

    # Step 2: Build FMUs
    run_step(
        "2. Build FMUs",
        [
            python_exe,
            script2,
            "--mo_file",
            args.mo_file,
            "--package",
            args.package,
            "--schema",
            args.out_schema,
            "--out_dir",
            args.out_fmu_dir,
        ],
    )

    # Step 3: Generate Processors
    run_step(
        "3. Generate Typed Processors",
        [
            python_exe,
            script3,
            "--schema",
            args.out_schema,
            "--out_dir",
            args.out_proc_dir,
        ],
    )

    # Step 4: Export SSP (Optional)
    if args.model_name:
        script4 = os.path.join(script_dir, "4_export_ssp.py")
        out_ssp_dir = (
            args.out_ssp_dir
            if args.out_ssp_dir
            else os.path.dirname(os.path.abspath(args.mo_file))
        )
        ssp_filename = os.path.splitext(os.path.basename(args.mo_file))[0] + ".ssp"
        out_ssp = os.path.join(out_ssp_dir, ssp_filename)
        run_step(
            "4. Export SSP",
            [
                python_exe,
                script4,
                "--mo_file",
                args.mo_file,
                "--model_name",
                args.model_name,
                "--fmu_dir",
                args.out_fmu_dir,
                "--out_ssp",
                out_ssp,
            ],
        )

    print("Pipeline finished successfully!")


if __name__ == "__main__":
    main()
