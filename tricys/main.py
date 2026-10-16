import os
import sys

# Ensure UTF-8 mode globally on Windows before launching any subcommands or OpenModelica sessions
if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional

from tricys.analysis.runner import retry_analysis
from tricys.core.diagnostics import DiagnosticsBarrier
from tricys.simulation.simulation import main as cosim_main
from tricys.simulation.simulation import main as simulation_main
from tricys.utils.file_utils import archive_run, unarchive_run
from tricys.utils.model_parser_cli import parse_model_cli


def run_example_runner(extra_args: Optional[list] = None) -> None:
    """Finds and executes the tricys_all_runner.py script.

    This function locates the main example runner script within the project
    and executes it in a separate subprocess. It handles errors if the
    script is not found.
    """
    try:
        python_executable = sys.executable
        main_py_path = Path(__file__).resolve()
        # project_root = main_py_path.parent.parent
        runner_script = main_py_path.parent / "example" / "example_runner" / "runner.py"

        if not runner_script.exists():
            print(
                f"Error: Example runner script not found at {runner_script}",
                file=sys.stderr,
            )
            sys.exit(1)

        print("INFO: Launching the interactive example runner...")
        cmd = [python_executable, str(runner_script)]
        if extra_args:
            cmd.extend(extra_args)
        subprocess.run(cmd)

    except Exception as e:
        print(
            f"An unexpected error occurred while trying to run the example runner: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    """Main entry point for the tricys command-line interface.

    Parses command-line arguments to dispatch tasks. It can run a standard
    simulation, a simulation analysis, or handle utility commands like
    archiving runs. The behavior is determined by subcommands or the content
    of a specified configuration file.
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
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Run in enhanced mode (sets execute_mode='enhanced' in config).",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Maximize process usage (ignores default safety limits).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV format after simulation.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Enable self-healing structured failure interception and emit diagnostics.json.",
    )

    # Subparsers for explicit commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("basic", help="Run a standard simulation.", add_help=False)

    subparsers.add_parser("analysis", help="Run a simulation analysis.", add_help=False)

    subparsers.add_parser("cosim", help="Run a co-simulation workflow.", add_help=False)

    subparsers.add_parser(
        "example", help="Run the interactive example runner.", add_help=False
    )

    subparsers.add_parser(
        "run",
        help="Run a simulation or task from a configuration file.",
        add_help=False,
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a configuration file syntax and schema.",
    )
    validate_parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to configuration file."
    )

    subparsers.add_parser(
        "visualize", help="Launch the interactive results visualizer.", add_help=False
    )

    subparsers.add_parser(
        "hdf5",
        help="Launch the HDF5 results visualizer (legacy alias for visualize).",
        add_help=False,
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

    parse_parser = subparsers.add_parser(
        "parse", help="Parse a Modelica model and output parameters as JSON."
    )
    parse_parser.add_argument(
        "package_path", type=str, help="Path to the package.mo file."
    )
    parse_parser.add_argument(
        "model_name",
        type=str,
        help="Fully qualified model name (e.g. package.Model).",
    )

    model_parser = subparsers.add_parser(
        "model", help="Modelica model inspection, review, and remediation tools."
    )
    model_sub = model_parser.add_subparsers(
        dest="model_action", help="Model operations"
    )

    inspect_sub = model_sub.add_parser(
        "inspect", help="Deeply inspect model parameters, inventories, and flow rates."
    )
    inspect_sub.add_argument("package_path", type=str, help="Path to package.mo file.")
    inspect_sub.add_argument("model_name", type=str, help="Qualified model name.")
    inspect_sub.add_argument(
        "--json", action="store_true", help="Output results in JSON format."
    )
    inspect_sub.add_argument(
        "-o", "--output", type=str, default=None, help="Save JSON output to file."
    )

    review_sub = model_sub.add_parser(
        "review",
        help="Audit model for time scale, conservation streams, and initial values.",
    )
    review_sub.add_argument("package_path", type=str, help="Path to package.mo file.")
    review_sub.add_argument("model_name", type=str, help="Qualified model name.")
    review_sub.add_argument(
        "--json", action="store_true", help="Output audit report in JSON format."
    )
    review_sub.add_argument(
        "--patch-output", type=str, default=None, help="Save generated patch to file."
    )
    review_sub.add_argument(
        "--apply-patch", action="store_true", help="Directly apply remediation patch."
    )

    plugin_parser = subparsers.add_parser(
        "plugin", help="Inspect and query workflow plugins and simulation engines."
    )
    plugin_sub = plugin_parser.add_subparsers(
        dest="plugin_action", help="Plugin operations"
    )

    plugin_list_sub = plugin_sub.add_parser(
        "list", help="List all registered workflow plugins."
    )
    plugin_list_sub.add_argument(
        "--json", action="store_true", help="Output plugin list formatted as JSON."
    )

    plugin_info_sub = plugin_sub.add_parser(
        "info", help="Show detailed metadata for a specific plugin."
    )
    plugin_info_sub.add_argument("name", type=str, help="Name of the plugin or engine.")
    plugin_info_sub.add_argument(
        "--json", action="store_true", help="Output plugin details formatted as JSON."
    )

    plugin_scaffold_sub = plugin_sub.add_parser(
        "scaffold",
        help="Generate production-ready plugin template code for Agent secondary development.",
    )
    plugin_scaffold_sub.add_argument("name", type=str, help="Name of the plugin.")
    plugin_scaffold_sub.add_argument(
        "--category",
        type=str,
        choices=["postprocess", "preprocess"],
        default="postprocess",
        help="Plugin lifecycle category.",
    )
    plugin_scaffold_sub.add_argument(
        "-o", "--output", type=str, default=None, help="Save template to file."
    )

    plugin_test_sub = plugin_sub.add_parser(
        "test",
        help="Test a plugin in an in-memory mock sandbox in milliseconds without OMC.",
    )
    plugin_test_sub.add_argument(
        "target", type=str, help="Path to plugin python file or registered plugin name."
    )
    plugin_test_sub.add_argument(
        "--class-name", type=str, default=None, help="Specific plugin class name."
    )
    plugin_test_sub.add_argument(
        "--json", action="store_true", help="Output test report in JSON format."
    )

    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile Schema 2.0 configuration into an explicit DAG pipeline format.",
    )
    compile_parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to config file."
    )
    compile_parser.add_argument(
        "-o", "--output", type=str, default=None, help="Save compiled pipeline to file."
    )
    compile_parser.add_argument(
        "--json",
        action="store_true",
        help="Output compiled pipeline as JSON instead of YAML.",
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
            basic_parser.add_argument(
                "--csv", action="store_true", help="Export results to CSV."
            )

            basic_args, _ = basic_parser.parse_known_args(remaining_argv)

            if basic_args.subcommand == "example":
                print("\n💡 提示: TRICYS 架构已全面升级至 V2.0 (Schema 2.0)。")
                print("   系统已自动切换至 V2.0 现代算例体系。\n")
                run_example_runner(remaining_argv[1:] if remaining_argv else None)
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

            export_csv = main_args.csv or basic_args.csv
            simulation_main(config_path, export_csv=export_csv)
        elif main_args.command == "cosim":
            cosim_parser = argparse.ArgumentParser()
            cosim_parser.add_argument("-c", "--config", type=str, default=None)
            cosim_parser.add_argument(
                "--csv", action="store_true", help="Export results to CSV."
            )
            cosim_args, _ = cosim_parser.parse_known_args(remaining_argv)

            config_path = cosim_args.config
            if not config_path:
                default_config_path = "config.json"
                if os.path.exists(default_config_path):
                    config_path = default_config_path
                    print(
                        "INFO: No config file specified for 'cosim' command, using default: config.json"
                    )
                else:
                    print(
                        "Error: 'cosim' command requires a config file via -c or a default 'config.json' must exist.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            export_csv = main_args.csv or cosim_args.csv
            cosim_main(config_path, export_csv=export_csv)
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
                print("\n💡 提示: TRICYS 架构已全面升级至 V2.0 (Schema 2.0)。")
                print(
                    "   敏感性分析等任务已统一作为 V2.0 任务原语执行，系统自动进入 V2.0 算例体系。\n"
                )
                run_example_runner(remaining_argv[1:] if remaining_argv else None)
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
            export_csv = main_args.csv
            simulation_main(config_path, export_csv=export_csv)
        elif main_args.command == "example":
            run_example_runner(remaining_argv)
            return
        elif main_args.command in ("visualize", "hdf5"):
            from tricys.visualizer.main import start as visualizer_main

            # Reconstruct argv for the visualizer's argument parser
            sys.argv = [f"{original_argv[0]} {main_args.command}"] + remaining_argv
            visualizer_main()
            return
        elif main_args.command == "run":
            run_parser = argparse.ArgumentParser(
                description="Run a simulation or task from a configuration file."
            )
            run_parser.add_argument(
                "-c",
                "--config",
                type=str,
                default=None,
                help="Path to the configuration file (.json, .yaml, or .yml).",
            )
            run_parser.add_argument(
                "--csv", action="store_true", help="Export results to CSV format."
            )
            run_parser.add_argument(
                "--enhanced",
                action="store_true",
                help="Run in parallel/enhanced execution mode.",
            )
            run_parser.add_argument(
                "--turbo",
                action="store_true",
                help="Maximize worker processes.",
            )
            run_parser.add_argument(
                "--diagnose",
                action="store_true",
                help="Enable self-healing structured failure interception.",
            )
            run_args, _ = run_parser.parse_known_args(remaining_argv)
            if run_args.config:
                main_args.config = run_args.config
            if run_args.csv:
                main_args.csv = True
            if run_args.enhanced:
                main_args.enhanced = True
            if run_args.turbo:
                main_args.turbo = True
            if getattr(run_args, "diagnose", False):
                main_args.diagnose = True
            # Proceed to config determination and execution
        elif main_args.command == "archive":
            archive_run(main_args.timestamp)
            return
        elif main_args.command == "unarchive":
            unarchive_run(main_args.zip_file)
            return
        elif main_args.command == "parse":
            parse_model_cli(main_args.package_path, main_args.model_name)
            return
        elif main_args.command == "model":
            action = getattr(main_args, "model_action", None)
            if action == "inspect":
                from tricys.core.introspect import introspect_model

                res = introspect_model(main_args.package_path, main_args.model_name)
                json_str = res.to_json(indent=2)
                if getattr(main_args, "output", None):
                    Path(main_args.output).write_text(json_str, encoding="utf-8")
                    print(f"INFO: Model introspection saved to {main_args.output}")
                else:
                    print(json_str)
                return

            elif action == "review":
                from tricys.core.model_auditor import ModelAuditor

                auditor = ModelAuditor(main_args.package_path, main_args.model_name)
                report = auditor.audit()
                patch_str = auditor.generate_remediation_patch(report.findings)

                if getattr(main_args, "apply_patch", False) and patch_str:
                    auditor.apply_remediation(patch_str)
                    print(
                        f"INFO: Successfully applied remediation patch to {main_args.package_path}",
                        file=sys.stderr,
                    )

                if getattr(main_args, "patch_output", None) and patch_str:
                    Path(main_args.patch_output).write_text(patch_str, encoding="utf-8")
                    print(
                        f"INFO: Remediation patch saved to {main_args.patch_output}",
                        file=sys.stderr,
                    )

                if getattr(main_args, "json", False):
                    print(report.to_json(indent=2))
                else:
                    print(
                        f"Model Review Report: {report.model_name} -> [{report.status}]"
                    )
                    print(f"Summary: {report.summary}")
                    if report.findings:
                        print("\nFindings:")
                        for f in report.findings:
                            sev = (
                                f.severity.value
                                if hasattr(f.severity, "value")
                                else str(f.severity)
                            )
                            print(f"  - [{sev}] {f.code} in {f.component}: {f.message}")
                            print(f"    Suggestion: {f.suggestion}")
                    else:
                        print(
                            "No issues found. Model complies with TRICYS v2.0 standards."
                        )
                return
            else:
                model_parser.print_help(sys.stderr)
                sys.exit(1)
        elif main_args.command == "plugin":
            import tricys.workflow  # noqa: F401
            from tricys.workflow.registry import PluginRegistry

            action = getattr(main_args, "plugin_action", None)
            if action == "list":
                data = PluginRegistry.introspect()
                if getattr(main_args, "json", False):
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                else:
                    print("TRICYS Registered Plugins & Simulation Engines:")
                    for cat, items in data.items():
                        print(f"\n[{cat.upper()}] ({len(items)} components):")
                        for item in items:
                            aliases_str = (
                                f" (aliases: {', '.join(item.get('aliases', []))})"
                                if len(item.get("aliases", [])) > 1
                                else ""
                            )
                            print(
                                f"  * {item['name']}{aliases_str}: {item.get('description', '')}"
                            )
                return

            elif action == "info":
                name = getattr(main_args, "name", "")
                info = PluginRegistry.get_plugin_info(name)
                if not info:
                    print(
                        f"Error: Plugin '{name}' not found in registry.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                if getattr(main_args, "json", False):
                    print(json.dumps(info, indent=2, ensure_ascii=False))
                else:
                    print(f"Plugin Info: {info.get('name')}")
                    print(f"Category: {info.get('category', 'unknown')}")
                    print(f"Description: {info.get('description', '')}")
                    print(f"Author: {info.get('author', 'TRICYS Core Team')}")
                    print(f"Version: {info.get('version', '2.0.0')}")
                    if info.get("input_requirements"):
                        print(
                            f"Input Requirements: {', '.join(info['input_requirements'])}"
                        )
                    if info.get("output_artifacts"):
                        print(
                            f"Output Artifacts: {', '.join(info['output_artifacts'])}"
                        )
                    if info.get("parameters_schema"):
                        print(
                            f"Parameters Schema: {json.dumps(info['parameters_schema'], indent=2, ensure_ascii=False)}"
                        )
                return

            elif action == "scaffold":
                from tricys.workflow.cli_helper import generate_plugin_scaffold

                name = getattr(main_args, "name", "my_custom_plugin")
                category = getattr(main_args, "category", "postprocess")
                out_path = getattr(main_args, "output", None)
                code = generate_plugin_scaffold(name, category, out_path)
                if not out_path:
                    print(code)
                else:
                    print(f"INFO: Successfully generated plugin scaffold at {out_path}")
                return

            elif action == "test":
                from tricys.workflow.cli_helper import test_plugin_in_mock_sandbox

                target = getattr(main_args, "target", "")
                cls_name = getattr(main_args, "class_name", None)
                try:
                    res = test_plugin_in_mock_sandbox(target, cls_name)
                    if getattr(main_args, "json", False):
                        print(json.dumps(res, indent=2, ensure_ascii=False))
                    else:
                        print(
                            f"Plugin Mock Test: {res.get('plugin')} -> [{res.get('status')}]"
                        )
                        print(
                            f"Metrics Emitted: {json.dumps(res.get('metrics_emitted', {}), indent=2)}"
                        )
                        print(
                            f"Artifacts Emitted: {json.dumps(res.get('artifacts_emitted', {}), indent=2)}"
                        )
                except Exception as e:
                    print(f"Error testing plugin: {e}", file=sys.stderr)
                    sys.exit(1)
                return
            else:
                plugin_parser.print_help(sys.stderr)
                sys.exit(1)
        elif main_args.command == "compile":
            from tricys.workflow.cli_helper import compile_v2_cli

            cfg_file = getattr(main_args, "config", None)
            out_file = getattr(main_args, "output", None)
            as_json = getattr(main_args, "json", False)
            try:
                res_str = compile_v2_cli(cfg_file, out_file, as_json)
                if not out_file:
                    print(res_str)
            except Exception as e:
                print(f"Error compiling config: {e}", file=sys.stderr)
                sys.exit(1)
            return
        elif main_args.command == "validate":
            config_file = getattr(main_args, "config", None)
            if not config_file:
                for candidate in ["config.yaml", "config.yml", "config.json"]:
                    if os.path.exists(candidate):
                        config_file = candidate
                        break
                else:
                    print(
                        "Error: No configuration file specified. Use -c <config>.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            from tricys.utils.config_utils import basic_prepare_config

            try:
                basic_prepare_config(config_file)
                print(f"Config is valid ✅ ({config_file})")
                return
            except Exception as e:
                print(
                    f"Config validation error in '{config_file}': {e}", file=sys.stderr
                )
                sys.exit(1)

    # 2. Determine config path (explicit -c or default)
    config_path = main_args.config
    if not config_path:
        for candidate in ["config.yaml", "config.yml", "config.json"]:
            if os.path.exists(candidate):
                print(
                    f"INFO: No command or config specified, using default: {candidate}"
                )
                config_path = candidate
                break
        else:
            # No command, no -c, no default config found -> show help and exit
            parser.print_help(sys.stderr)
            sys.exit(1)

    # 3. Handle config-based dispatch using the determined config_path
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        if config_path.endswith((".yaml", ".yml")):
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
    except Exception as e:
        print(
            f"Error reading or parsing config file '{config_path}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine base directory for resolving relative paths in config_data
    base_dir = os.path.dirname(os.path.abspath(config_path))

    # Check for TRICYS 2.0 configuration
    from tricys.core.schema import is_v2_config

    if is_v2_config(config_data):
        print(f"INFO: Detected TRICYS 2.0 configuration ({config_path}).")
        task_cfg = config_data.get("task", {})
        task_type = task_cfg.get("type", "simulate")

        if main_args.enhanced:
            task_cfg["parallel"] = True
            print("INFO: Enhanced parallel mode enabled via command line flag.")
        if main_args.turbo:
            task_cfg["parallel"] = True
            task_cfg["max_workers"] = os.cpu_count() or 4
            print(f"INFO: Turbo mode enabled! Using {task_cfg['max_workers']} workers.")

        print(f"INFO: Dispatching TRICYS 2.0 task '{task_type}' to simulation engine.")

        diagnose_path = (
            "diagnostics.json" if getattr(main_args, "diagnose", False) else None
        )
        with DiagnosticsBarrier(
            output_path=diagnose_path,
            context_snapshot={"config_path": config_path, "task_type": task_type},
            exit_on_error=True if getattr(main_args, "diagnose", False) else False,
        ):
            ctx = simulation_main(
                config_data,
                base_dir=base_dir,
                export_csv=main_args.csv,
            )

        # Handle visualizer auto-launch if configured
        vis_cfg = config_data.get("visualizer", {})
        if vis_cfg.get("launch_on_finish", False):
            print("INFO: 'launch_on_finish' is enabled. Launching visualizer...")
            try:
                from tricys.visualizer.main import start as visualizer_main

                vis_argv = [original_argv[0]]
                if (
                    ctx
                    and hasattr(ctx, "results_store_path")
                    and ctx.results_store_path
                ):
                    vis_argv.append(ctx.results_store_path)
                if "port" in vis_cfg:
                    vis_argv.extend(["--port", str(vis_cfg["port"])])
                if "host" in vis_cfg:
                    vis_argv.extend(["--host", str(vis_cfg["host"])])
                sys.argv = vis_argv
                visualizer_main()
            except Exception as e:
                print(f"WARNING: Visualizer launch failed: {e}", file=sys.stderr)
        return

    # Legacy 1.0 configuration handling
    is_analysis = "sensitivity_analysis" in config_data and config_data.get(
        "sensitivity_analysis", {}
    ).get("enabled", False)

    if "simulation" not in config_data:
        config_data["simulation"] = {}

    # 4. Handle --enhanced and --turbo overrides for 1.0
    if main_args.enhanced:
        config_data["simulation"]["execute_mode"] = "enhanced"
        config_data["simulation"]["concurrent"] = True
        print("INFO: Enhanced mode enabled via command line flag.")

    if main_args.turbo:
        config_data["simulation"]["maximize_workers"] = True
        print("INFO: Turbo mode enabled! Using maximum available resources.")

    if is_analysis:
        print(
            "INFO: Detected 'sensitivity_analysis' in config. Running unified analysis pipeline."
        )
        simulation_main(
            config_data,
            base_dir=base_dir,
            export_csv=main_args.csv,
        )
    elif "co_simulation" in config_data:
        print(
            "INFO: Detected 'co_simulation' in config. Running co-simulation workflow."
        )
        cosim_main(
            config_data,
            base_dir=base_dir,
            export_csv=main_args.csv,
        )
    else:
        print(
            "INFO: No 'sensitivity_analysis' detected in config. Running standard simulation workflow."
        )
        simulation_main(
            config_data,
            base_dir=base_dir,
            export_csv=main_args.csv,
        )
    return


if __name__ == "__main__":
    main()
