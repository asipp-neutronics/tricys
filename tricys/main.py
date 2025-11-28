import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tricys.simulation.simulation import main as simulation_main
from tricys.simulation.simulation_analysis import main as analysis_main
from tricys.simulation.simulation_analysis import retry_analysis
from tricys.simulation.simulation_gui import main as gui_main
from tricys.utils.file_utils import archive_run, unarchive_run


def run_example_runner() -> None:
    """Finds and executes the tricys_all_runner.py script.

    This function locates the main example runner script within the project
    and executes it in a separate subprocess. It handles errors if the
    script is not found.
    """
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


def main() -> None:
    """Main entry point for the tricys command-line interface.

    Parses command-line arguments to dispatch tasks. It can run a standard
    simulation, a simulation analysis, launch the GUI, or handle utility
    commands like archiving runs. The behavior is determined by subcommands
    or the content of a specified configuration file.
    """
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
            # This block replaces the logic that was in simulation.py's _parse_command_line_args
            basic_parser = argparse.ArgumentParser()
            basic_subparsers = basic_parser.add_subparsers(dest="subcommand")
            basic_subparsers.add_parser("example")
            basic_parser.add_argument("-c", "--config", type=str, default=None)

            basic_args, _ = basic_parser.parse_known_args(remaining_argv)

            if basic_args.subcommand == "example":
                # This is the 'basic example' case.
                import importlib.util

                script_path = (
                    Path(__file__).parent.parent
                    / "script"
                    / "example_runner"
                    / "tricys_runner.py"
                )
                spec = importlib.util.spec_from_file_location(
                    "tricys_runner", script_path
                )
                tricys_runner = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tricys_runner)
                tricys_runner.main()
                sys.exit(0)

            # This is the standard 'basic' simulation case.
            config_path = basic_args.config
            if not config_path:
                default_config_path = "config.json"
                if os.path.exists(default_config_path):
                    config_path = default_config_path
                    print(
                        "INFO: No config file specified for 'basic' command, using default: config.json"
                    )
                else:
                    print(
                        "Error: 'basic' command requires a config file via -c or a default 'config.json' must exist.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            simulation_main(config_path)
        elif main_args.command == "analysis":
            # This block replaces the logic in simulation_analysis.py's _parse_command_line_args
            analysis_parser = argparse.ArgumentParser(
                description="Run a simulation analysis."
            )
            analysis_subparsers = analysis_parser.add_subparsers(
                dest="subcommand", help="Analysis commands"
            )

            analysis_subparsers.add_parser(
                "example", help="Run analysis examples interactively"
            )

            retry_parser = analysis_subparsers.add_parser(
                "retry", help="Retry failed AI analysis for existing reports."
            )
            retry_parser.add_argument(
                "timestamp", type=str, help="Timestamp of the run to retry."
            )

            analysis_parser.add_argument(
                "-c",
                "--config",
                type=str,
                default=None,
                help="Path to the JSON configuration file.",
            )

            analysis_args, _ = analysis_parser.parse_known_args(remaining_argv)

            if analysis_args.subcommand == "retry":
                retry_analysis(analysis_args.timestamp)
                sys.exit(0)

            if analysis_args.subcommand == "example":
                import importlib.util

                script_path = (
                    Path(__file__).parent.parent
                    / "script"
                    / "example_runner"
                    / "tricys_ana_runner.py"
                )
                spec = importlib.util.spec_from_file_location(
                    "tricys_ana_runner", script_path
                )
                tricys_ana_runner = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tricys_ana_runner)
                tricys_ana_runner.main()
                sys.exit(0)

            config_path = analysis_args.config
            if not config_path:
                default_config_path = "config.json"
                if os.path.exists(default_config_path):
                    config_path = default_config_path
                    print(
                        "INFO: No config file specified for 'analysis' command, using default: config.json"
                    )
                else:
                    print(
                        "Error: 'analysis' command requires a config file via -c or a default 'config.json' must exist.",
                        file=sys.stderr,
                    )
                    analysis_parser.print_help(sys.stderr)
                    sys.exit(1)

            analysis_main(config_path)
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

    if is_analysis:
        print(
            "INFO: Detected 'sensitivity_analysis' in config. Running analysis workflow."
        )
        analysis_main(config_path)
    else:
        print(
            "INFO: No 'sensitivity_analysis' detected in config. Running standard simulation workflow."
        )
        simulation_main(config_path)
    return


if __name__ == "__main__":
    main()
