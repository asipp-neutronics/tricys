import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tricys.simulation.simulation import main as simulation_main
from tricys.simulation.simulation_analysis import main as analysis_main
from tricys.simulation.simulation_gui import main as gui_main
from tricys.utils.file_utils import archive_run, unarchive_run


def run_example_runner():
    """Finds and executes the tricys_all_runner.py script."""
    try:
        python_executable = sys.executable
        main_py_path = Path(__file__).resolve()
        project_root = main_py_path.parent.parent
        runner_script = (
            project_root / "script" / "example_runner" / "tricys_all_runner.py"
        )

        if not runner_script.exists():
            print(
                f"Error: Example runner script not found at {runner_script}",
                file=sys.stderr,
            )
            sys.exit(1)

        print("INFO: Launching the interactive example runner...")
        # The runner is interactive, so it will take over the console.
        # We don't need to manage argv for it.
        subprocess.run([python_executable, str(runner_script)])

    except Exception as e:
        print(
            f"An unexpected error occurred while trying to run the example runner: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    """Main entry point for the tricys command-line interface."""
    # Main parser
    parser = argparse.ArgumentParser(
        description="Tricys - TRitium Integrated CYcle Simulation Framework",
        add_help=False,
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the JSON configuration file."
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and exit."
    )

    # Subparsers for explicit commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("basic", help="Run a standard simulation.", add_help=False)

    subparsers.add_parser("analysis", help="Run a simulation analysis.", add_help=False)

    subparsers.add_parser("gui", help="Launch the interactive GUI.", add_help=False)

    subparsers.add_parser(
        "example", help="Run the interactive example runner.", add_help=False
    )

    archive_parser = subparsers.add_parser(
        "archive", help="Archive a simulation or analysis run."
    )
    archive_parser.add_argument(
        "timestamp", type=str, help="Timestamp of the run to archive."
    )

    unarchive_parser = subparsers.add_parser("unarchive", help="Unarchive a run.")
    unarchive_parser.add_argument(
        "zip_file", type=str, help="Path to the archive file to unarchive."
    )

    # --- Argument Parsing Logic ---

    main_args, remaining_argv = parser.parse_known_args()
    original_argv = sys.argv

    # Handle help request
    if main_args.help:
        parser.print_help(sys.stderr)
        sys.exit(0)

    # 1. Handle explicit subcommands
    if main_args.command:
        if main_args.command == "basic":
            sys.argv = [f"{original_argv[0]} {main_args.command}"] + remaining_argv
            simulation_main()
        elif main_args.command == "analysis":
            sys.argv = [f"{original_argv[0]} {main_args.command}"] + remaining_argv
            analysis_main()
        elif main_args.command == "gui":
            sys.argv = [f"{original_argv[0]} {main_args.command}"] + remaining_argv
            gui_main()
        elif main_args.command == "example":
            run_example_runner()
        elif main_args.command == "archive":
            archive_run(main_args.timestamp)
        elif main_args.command == "unarchive":
            unarchive_run(main_args.zip_file)
        return

    # 2. Determine config path (explicit -c or default)
    config_path = main_args.config
    if not config_path:
        if os.path.exists("config.json"):
            print("INFO: No command or config specified, using default: config.json")
            config_path = "config.json"
            # Reconstruct argv for downstream parsers that expect '-c'
            sys.argv = [original_argv[0], "-c", config_path]
        else:
            # No command, no -c, no default config.json -> show help and exit
            parser.print_help(sys.stderr)
            sys.exit(1)

    # 3. Handle config-based dispatch using the determined config_path
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(
            f"Error reading or parsing config file '{config_path}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Decide which main to call based on config content
    is_analysis = "sensitivity_analysis" in config_data and config_data.get(
        "sensitivity_analysis", {}
    ).get("enabled", False)

    # If -c was explicitly passed, the original argv is correct for the downstream parser.
    # If we are using the default, we have already modified sys.argv.
    if main_args.config:
        sys.argv = original_argv

    if is_analysis:
        print(
            "INFO: Detected 'sensitivity_analysis' in config. Running analysis workflow."
        )
        analysis_main()
    else:
        print(
            "INFO: No 'sensitivity_analysis' detected in config. Running standard simulation workflow."
        )
        simulation_main()
    return


if __name__ == "__main__":
    main()
