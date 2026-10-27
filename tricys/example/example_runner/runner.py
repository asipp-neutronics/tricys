#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRICYS V2.0 Example Runner.

A modern unified runner specifically designed for TRICYS Schema 2.0 industrial
benchmark examples (v2_examples).
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict


class TricysV2ExampleRunner:
    """TRICYS V2.0 Example Runner (Schema 2.0)."""

    def __init__(self, no_vis_default: bool = False):
        """Initialize the V2 example runner."""
        # Force UTF-8 encoding for stdout/stderr to support emojis on Windows
        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass

        # Locate package paths
        self.runner_dir = Path(__file__).resolve().parent
        self.package_root = self.runner_dir.parent
        self.example_base_dir = self.package_root / "example_data"
        self.v2_example_dir = self.example_base_dir / "v2_examples"

        self.workspace_dir = Path.cwd()
        self.test_example_base_dir = self.workspace_dir / "test_example"
        self.test_example_v2_dir = self.test_example_base_dir / "v2_examples"
        self.no_vis_default = no_vis_default

        # Automatically scan and generate example configurations
        self.examples = self._scan_examples()

    def _scan_examples(self) -> Dict[str, Dict[str, Any]]:
        """Read and auto-discover V2.0 examples from v2_examples.

        Returns:
            dict: Example configuration dictionary keyed by numeric ID string.
        """
        examples: Dict[str, Dict[str, Any]] = {}
        if not self.v2_example_dir.exists():
            print(f"⚠️  警告: V2 示例目录不存在: {self.v2_example_dir}")
            return examples

        # 1. Read metadata from example_runner.json if available
        json_meta: Dict[str, Dict[str, Any]] = {}
        config_file = self.v2_example_dir / "example_runner.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("examples", []):
                        cfg_name = item.get("config", "")
                        if cfg_name:
                            json_meta[cfg_name] = item
            except Exception as e:
                print(f"⚠️  警告: 读取 example_runner.json 失败: {e}")

        # 2. Discover all *.yaml files in v2_examples
        yaml_files = sorted(
            [
                p
                for p in self.v2_example_dir.glob("*.yaml")
                if p.is_file() and not p.name.startswith(".")
            ]
        )

        counter = 1
        for yaml_path in yaml_files:
            cfg_name = yaml_path.name
            meta = json_meta.get(cfg_name, {})

            # Fallback extraction of task_type and description from YAML if missing
            task_type = meta.get("task_type", "")
            desc = meta.get("description", "")
            name = meta.get("name", "")

            if not task_type or not desc or not name:
                try:
                    import yaml

                    with open(yaml_path, "r", encoding="utf-8") as f:
                        cfg_content = yaml.safe_load(f) or {}
                    if not task_type:
                        task_type = cfg_content.get("task", {}).get("type", "simulate")
                    if not name:
                        title = (
                            cfg_content.get("visualizer", {})
                            .get("report", {})
                            .get("title", "")
                        )
                        name = (
                            title if title else yaml_path.stem.replace("_", " ").title()
                        )
                    if not desc:
                        desc = f"Schema 2.0 {task_type} 工业算例: {yaml_path.name}"
                except Exception:
                    task_type = task_type or "simulate"
                    name = name or yaml_path.stem
                    desc = desc or f"V2 算例 {yaml_path.name}"

            item_id = str(counter)
            examples[item_id] = {
                "id": item_id,
                "name": name,
                "task_type": task_type.upper(),
                "config": cfg_name,
                "path": ".",
                "full_path": str(yaml_path),
                "description": desc,
            }
            counter += 1

        return examples

    def show_menu(self) -> None:
        """Display available TRICYS V2 example menu."""
        print("\n" + "═" * 68)
        print(f"{'TRICYS V2.0 现代工业算例运行器 (Schema 2.0)':^60}")
        print("═" * 68 + "\n")

        if not self.examples:
            print("❌ 未发现任何 TRICYS V2 示例！")
            print(f"请检查目录: {self.v2_example_dir}")
        else:
            for key, ex in self.examples.items():
                badge = f"[{ex['task_type']}]"
                print(f"  {key:>2}. {badge:<12} {ex['name']}")
                print(f"      📝 {ex['description']}")
                print(f"      ⚙️  {ex['config']}")
                print("─" * 68)

        print("\n" + "─" * 68)
        print("  0. 退出程序  |  h. 显示帮助  |  s. 重新扫描")
        print("  c <序号>. 编译Pipeline DAG  |  v <序号>. 校验配置语法")
        print("─" * 68 + "\n")

    def copy_example(self, example_info: Dict[str, Any], no_vis: bool = False) -> bool:
        """Copy V2 example and auxiliary dependency models into test_example directory.

        Args:
            example_info: Example configuration dict.
            no_vis: Whether to disable launch_on_finish in copied config.

        Returns:
            bool: True if copy succeeded, False otherwise.
        """
        try:
            self.test_example_base_dir.mkdir(parents=True, exist_ok=True)
            self.test_example_v2_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 68)
            print("📋 正在准备算例运行环境...")
            print("-" * 68)

            # 1. Copy shared model folders (example_*) to test_example_base_dir
            for item in self.example_base_dir.glob("example_*"):
                if item.is_dir():
                    dest_path = self.test_example_base_dir / item.name
                    if not dest_path.exists():
                        print(f"   📦 同步依赖模型库: {item.name}")
                        shutil.copytree(item, dest_path)

            # 2. Copy V2 auxiliary directories (plugins, cosim models) to test_example/v2_examples
            aux_dirs = ["plugins", "10_cosim_parameter_injection"]
            for aux in aux_dirs:
                src_aux = self.v2_example_dir / aux
                if src_aux.exists() and src_aux.is_dir():
                    dest_aux = self.test_example_v2_dir / aux
                    if dest_aux.exists():
                        shutil.rmtree(dest_aux)
                    shutil.copytree(src_aux, dest_aux)

            # 3. Copy target YAML config file
            src_config = self.v2_example_dir / example_info["config"]
            dest_config = self.test_example_v2_dir / example_info["config"]

            if not src_config.exists():
                print(f"❌ 源配置文件不存在: {src_config}")
                return False

            if no_vis:
                # Patch visualizer.launch_on_finish to False
                import yaml

                with open(src_config, "r", encoding="utf-8") as f:
                    cfg_obj = yaml.safe_load(f)
                if isinstance(cfg_obj, dict) and "visualizer" in cfg_obj:
                    cfg_obj["visualizer"]["launch_on_finish"] = False
                with open(dest_config, "w", encoding="utf-8") as f:
                    yaml.dump(cfg_obj, f, allow_unicode=True, sort_keys=False)
            else:
                shutil.copy2(src_config, dest_config)

            print(f"   📂 算例工作目录: {self.test_example_v2_dir}")
            print(f"   📄 目标配置文件: {dest_config.name}")
            print("✅ 算例运行环境就绪")
            return True

        except PermissionError:
            print("❌ 权限错误: 无法访问或复制文件，请检查文件写入权限")
            return False
        except Exception as e:
            print(f"❌ 准备算例环境失败: {e}")
            return False

    def validate_example(self, example_info: Dict[str, Any]) -> bool:
        """Validate V2 example configuration syntax and schema."""
        cfg_path = self.v2_example_dir / example_info["config"]
        print(f"\n🔍 正在校验算例配置: {example_info['name']} ({cfg_path.name})")

        from tricys.utils.config_utils import basic_prepare_config

        try:
            runtime_cfg, _ = basic_prepare_config(str(cfg_path))
            task_type = runtime_cfg.get("task", {}).get("type", "unknown")
            print(f"✅ 语法与结构校验通过！任务类型: [{task_type.upper()}]")
            return True
        except Exception as e:
            print(f"❌ 配置校验失败: {e}")
            return False

    def compile_example(self, example_info: Dict[str, Any]) -> bool:
        """Compile V2 example configuration into DAG pipeline without running."""
        cfg_path = self.v2_example_dir / example_info["config"]
        print(f"\n⚙️ 正在编译算例 Pipeline DAG: {example_info['name']}")
        print("─" * 68)

        from tricys.workflow.cli_helper import compile_v2_cli

        try:
            dag_str = compile_v2_cli(str(cfg_path))
            print(dag_str)
            print("─" * 68)
            print("✅ Pipeline DAG 编译成功！")
            return True
        except Exception as e:
            print(f"❌ Pipeline DAG 编译失败: {e}")
            return False

    def run_command(
        self,
        example_info: Dict[str, Any],
        use_enhanced: bool = False,
        turbo: bool = False,
        csv: bool = False,
        dry_run: bool = False,
        no_vis: bool = False,
    ) -> bool:
        """Execute the tricys simulation command for the V2 example.

        Args:
            example_info: Example configuration dict.
            use_enhanced: Force enhanced mode.
            turbo: Maximize processes.
            csv: Export CSV results.
            dry_run: Validate and compile without starting simulation.
            no_vis: Disable visualizer launch.

        Returns:
            bool: True if execution succeeded.
        """
        config_path = self.test_example_v2_dir / example_info["config"]
        if not config_path.exists():
            print(f"❌ 运行目录中的配置文件不存在: {config_path}")
            return False

        if dry_run:
            print("\n🔬 [Dry-Run 模式] 校验并编译 Pipeline，不执行仿真积分...")
            v_ok = self.validate_example(example_info)
            c_ok = self.compile_example(example_info)
            return v_ok and c_ok

        # Determine CLI command
        tricys_cmd = shutil.which("tricys")
        if tricys_cmd:
            cmd = [tricys_cmd, "-c", str(config_path)]
        else:
            cmd = [sys.executable, "-m", "tricys", "-c", str(config_path)]

        if use_enhanced:
            cmd.append("--enhanced")
        if turbo:
            cmd.append("--turbo")
        if csv:
            cmd.append("--csv")

        print("\n" + "=" * 68)
        print("🚀 开始执行 TRICYS V2.0 仿真算例")
        print("=" * 68)
        print(f"📂 工作目录: {self.test_example_v2_dir}")
        print(f"💻 执行命令: {' '.join(cmd)}")
        mode_str = (
            "🔥 Turbo" if turbo else ("⚡ Enhanced" if use_enhanced else " Standard")
        )
        print(f"⚙️ 运行模式: {mode_str}")
        print("=" * 68 + "\n")

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.test_example_v2_dir,
                capture_output=False,
                text=True,
            )
            elapsed = time.time() - start_time
            print("\n" + "=" * 68)
            if result.returncode == 0:
                print(f"✅ 算例执行成功！总耗时: {elapsed:.2f} 秒")
                print(f"📊 结果与产物位于: {self.test_example_v2_dir}")
                return True
            else:
                print(f"❌ 算例执行失败，返回码: {result.returncode}")
                return False

        except FileNotFoundError:
            print("❌ 未能找到 Python 或 tricys 运行环境")
            return False
        except Exception as e:
            print(f"❌ 执行命令时发生未预期异常: {e}")
            return False

    def run_example(
        self,
        choice: str,
        use_enhanced: bool = False,
        turbo: bool = False,
        csv: bool = False,
        dry_run: bool = False,
        no_vis: bool = False,
    ) -> bool:
        """Run the specified V2 example by choice ID."""
        if choice not in self.examples:
            print(f"\n❌ 无效的选择: '{choice}'")
            return False

        example_info = self.examples[choice]
        actual_no_vis = no_vis or self.no_vis_default

        if not self.copy_example(example_info, no_vis=actual_no_vis):
            return False

        success = self.run_command(
            example_info,
            use_enhanced=use_enhanced,
            turbo=turbo,
            csv=csv,
            dry_run=dry_run,
            no_vis=actual_no_vis,
        )

        if success:
            print(f"\n🎉 算例 [{example_info['id']}] {example_info['name']} 完成")
        else:
            print(f"\n⚠️ 算例 [{example_info['id']}] {example_info['name']} 异常中断")

        return success

    def show_help(self) -> None:
        """Display help information for TRICYS V2 example runner."""
        help_text = """
══════════════════════════════════════════════════════════════════════
               TRICYS V2.0 工业算例运行器 使用指南 (Schema 2.0)
══════════════════════════════════════════════════════════════════════

  功能概览:
    本运行器专为 TRICYS V2.0 设计，提供全套现代 Schema 2.0 工业级基准算例：
    • [SWEEP]       参数扫描、启动氚自持评估、脉宽边界扫描
    • [SIMULATE]    单堆长周期脉冲响应、二分法 Required_TBR 求解、质量审计
    • [SENSITIVITY] 高维参数 Sobol 全局灵敏度分析与一阶/总阶方差分解
    • [COSIM]       两阶段离线查表、FMI/SSP 在线步进同步、CFETR 参数注入稳态仿真

  交互式快捷键:
    • 1-10:        运行对应的算例编号
    • c <序号>:     编译该算例的 Schema 2.0 Pipeline DAG (不执行仿真)
    • v <序号>:     校验该算例配置文件的语法与模式完整性
    • s:           重新扫描 v2_examples 目录
    • h:           显示此帮助信息
    • 0 或 q:       退出程序

  命令行参数 (支持非交互式调用):
    • --list                列出所有可用的 V2 算例
    • --run <ID>            直接运行指定编号的算例
    • --compile <ID>        编译并输出指定算例的 DAG 结构
    • --validate <ID>       校验指定算例语法
    • --dry-run             准备环境并校验/编译，不执行长周期仿真
    • --no-vis              禁止仿真完成后自动拉起可视化界面
    • --enhanced / --turbo  开启并行或极致加速模式

══════════════════════════════════════════════════════════════════════
        """.strip()
        print(help_text)

    def interactive_loop(self) -> None:
        """Run interactive menu loop."""
        while True:
            self.show_menu()
            try:
                raw_input = input(
                    f"\n请输入操作指令 (1-{len(self.examples)}/c <ID>/v <ID>/h/s/0): "
                ).strip()

                if not raw_input:
                    continue

                if raw_input in ["0", "q", "quit", "exit"]:
                    print("\n👋 感谢使用 TRICYS V2.0，再见！")
                    break

                if raw_input.lower() == "h":
                    self.show_help()
                    continue

                if raw_input.lower() == "s":
                    print("\n🔄 正在重新扫描 V2 算例目录...")
                    self.examples = self._scan_examples()
                    continue

                # Compile command: c <ID>
                if raw_input.lower().startswith("c "):
                    parts = raw_input.split()
                    if len(parts) >= 2 and parts[1] in self.examples:
                        self.compile_example(self.examples[parts[1]])
                    else:
                        print(f"❌ 未知算例序号: {raw_input}")
                    continue

                # Validate command: v <ID>
                if raw_input.lower().startswith("v "):
                    parts = raw_input.split()
                    if len(parts) >= 2 and parts[1] in self.examples:
                        self.validate_example(self.examples[parts[1]])
                    else:
                        print(f"❌ 未知算例序号: {raw_input}")
                    continue

                # Direct selection: e.g. "1" or "01"
                norm_choice = str(int(raw_input)) if raw_input.isdigit() else raw_input
                if norm_choice in self.examples:
                    self.run_example(norm_choice)
                    while True:
                        cont = (
                            input("\n是否继续体验其他 V2 算例? (y/n, 默认 y): ")
                            .strip()
                            .lower()
                        )
                        if cont in ["y", "yes", "是", "Y", ""]:
                            break
                        elif cont in ["n", "no", "否", "N"]:
                            print("\n👋 程序退出。")
                            return
                else:
                    print(
                        f"\n❌ 无效的输入 '{raw_input}'，请输入 1-{len(self.examples)}、h 或 0 退出。"
                    )

            except KeyboardInterrupt:
                print("\n\nℹ️ 用户中断，退出运行器。")
                break
            except Exception as e:
                print(f"\n❌ 发生异常: {e}")


# Aliases for backward compatibility
TricysAllTestRunner = TricysV2ExampleRunner
TricysTestRunner = TricysV2ExampleRunner
TricysAnaTestRunner = TricysV2ExampleRunner


def main(argv=None) -> None:
    """Main CLI entrypoint for TRICYS V2 example runner."""
    parser = argparse.ArgumentParser(
        description="TRICYS V2.0 Example Runner (Schema 2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available V2 examples and exit."
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run specific example by ID (e.g. --run 1) non-interactively.",
    )
    parser.add_argument(
        "--compile",
        type=str,
        default=None,
        help="Compile and display DAG pipeline for specific example ID.",
    )
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Validate configuration schema for specific example ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare workspace and validate/compile without running simulation.",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable automatic visualizer launch on completion.",
    )
    parser.add_argument(
        "--enhanced", action="store_true", help="Enable enhanced parallel mode."
    )
    parser.add_argument(
        "--turbo", action="store_true", help="Enable turbo process mode."
    )
    parser.add_argument(
        "--csv", action="store_true", help="Export simulation results to CSV."
    )

    args = parser.parse_args(argv)
    runner = TricysV2ExampleRunner(no_vis_default=args.no_vis)

    if args.list:
        print("\n" + "=" * 68)
        print("📋 可用 TRICYS V2 现代化工业算例列表:")
        print("=" * 68)
        for k, ex in runner.examples.items():
            print(f"  [{k:>2}] [{ex['task_type']:<11}] {ex['name']}")
            print(f"       配置: {ex['config']}")
        print("=" * 68 + "\n")
        sys.exit(0)

    if args.compile:
        target = str(int(args.compile)) if args.compile.isdigit() else args.compile
        if target in runner.examples:
            ok = runner.compile_example(runner.examples[target])
            sys.exit(0 if ok else 1)
        else:
            print(f"❌ 未找到算例: {args.compile}")
            sys.exit(1)

    if args.validate:
        target = str(int(args.validate)) if args.validate.isdigit() else args.validate
        if target in runner.examples:
            ok = runner.validate_example(runner.examples[target])
            sys.exit(0 if ok else 1)
        else:
            print(f"❌ 未找到算例: {args.validate}")
            sys.exit(1)

    if args.run:
        target = str(int(args.run)) if args.run.isdigit() else args.run
        if target in runner.examples:
            ok = runner.run_example(
                target,
                use_enhanced=args.enhanced,
                turbo=args.turbo,
                csv=args.csv,
                dry_run=args.dry_run,
                no_vis=args.no_vis,
            )
            sys.exit(0 if ok else 1)
        else:
            print(f"❌ 未找到算例: {args.run}")
            sys.exit(1)

    # Interactive mode
    runner.interactive_loop()


if __name__ == "__main__":
    main()
