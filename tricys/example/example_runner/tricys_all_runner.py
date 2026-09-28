#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tricys All-in-One Example Runner
A unified runner for running all tricys command examples (BASIC and ANALYSIS)
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


class TricysAllTestRunner:
    """Tricys All-in-One Example Runner"""

    def __init__(self):
        """Initialize the runner"""
        # Force UTF-8 encoding for stdout/stderr to support emojis on Windows
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")

        # Locate package root directory
        self.runner_dir = Path(__file__).parent
        self.package_root = self.runner_dir.parent
        self.example_base_dir = self.package_root / "example_data"
        self.workspace_dir = Path.cwd()  # Run in CWD
        self.test_example_base_dir = self.workspace_dir / "test_example"

        # Automatically scan and generate example configurations
        self.examples = self._scan_examples()

    def _scan_examples(self):
        """
        Read all example configurations from example/basic/ and example/analysis/

        Returns:
            dict: Combined example configuration dictionary
        """
        examples = {}
        counter = 1
        example_types = ["basic", "analysis"]

        print("\n" + "=" * 60)
        print("🔄 正在扫描所有示例目录 (basic/ & analysis/)...")
        print("=" * 60 + "\n")

        for example_type in example_types:
            config_file = self.example_base_dir / example_type / "example_runner.json"
            if not config_file.exists():
                print(
                    f"⚠️  警告: {example_type.upper()} 示例配置文件不存在: {config_file}"
                )
                continue

            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                print(
                    f"📦 [{example_type.upper()}] {config_data.get('description', '')}"
                )
                print("-" * 60)

                examples_list = config_data.get("examples", [])

                for example_config in examples_list:
                    if not example_config.get("enabled", True):
                        # print(f"  ⏸️  [跳过] {example_config.get('name', 'Unknown')}")
                        continue

                    example_path = (
                        self.example_base_dir / example_type / example_config["path"]
                    )
                    config_path = example_path / example_config["config"]

                    if not config_path.exists():
                        print(f"  ⚠️  [缺失] {example_config['name']} ({config_path})")
                        continue

                    # The 'command' from JSON is no longer needed, but we keep it for compatibility
                    examples[str(counter)] = {
                        "name": example_config["name"],
                        "type": example_type,
                        "path": example_config["path"],
                        "config": example_config["config"],
                        "command": example_config.get(
                            "command", "tricys"
                        ),  # Default to tricys
                        "description": example_config["description"],
                    }

                    print(f"  ✅ {counter}. {example_config['name']}")
                    counter += 1
                print()

            except json.JSONDecodeError as e:
                print(f"❌ {example_type.upper()} 的 JSON 解析错误: {e}")
                print(f"   请检查 {config_file} 文件格式")
            except Exception as e:
                print(f"❌ 读取 {example_type.upper()} 配置文件时出错: {e}")

        print("=" * 60)
        print(f"🎉 扫描完成: 共加载 {len(examples)} 个示例")
        print("=" * 60 + "\n")
        return examples

    def show_menu(self):
        """Display available example menu"""
        print("\n" + "=" * 60)
        print(f"{'TRICYS 统一示例运行器':^56}")
        print("=" * 60 + "\n")

        if not self.examples:
            print("❌ 未发现任何示例")
            print("请检查 tricys/example/example_data 目录是否存在配置文件")
        else:
            for key, example in self.examples.items():
                print(f"  {key}. [{example['type'].upper()}] {example['name']}")
                print(f"     📝 {example['description']}")
                # print(f"     ⚙️  {example['config']}")
                print("-" * 60)

        print("\n" + "-" * 60)
        print("  0. 退出程序  |  h. 显示帮助  |  s. 重新扫描")
        print("-" * 60 + "\n")

    def copy_example(self, example_info):
        """
        Copy example folder to test_example directory

        Args:
            example_info: Example information dictionary

        Returns:
            bool: Whether copy is successful
        """
        try:
            source_path = (
                self.example_base_dir / example_info["type"] / example_info["path"]
            )

            if not source_path.exists():
                print(f"❌ 示例路径不存在: {source_path}")
                return False

            self.test_example_dir = (
                self.test_example_base_dir / example_info["type"] / example_info["path"]
            )

            if self.test_example_dir.exists():
                # print("─" * 50)
                # print(f"🧹 正在清理旧的测试目录: {self.test_example_dir}")
                shutil.rmtree(self.test_example_dir)

            self.test_example_base_dir.mkdir(exist_ok=True)

            print("\n" + "=" * 60)
            print("📋 正在准备环境...")
            print("-" * 60)
            print(f"   📂 源目录: {source_path}")
            print(f"   🎯 目标目录: {self.test_example_dir}")

            shutil.copytree(source_path, self.test_example_dir)

            # Also copy all 'example_*' subdirectories from the 'example_data' directory
            example_root = self.package_root / "example_data"
            for item in example_root.glob("example_*"):
                if item.is_dir():
                    dest_path = self.test_example_base_dir / item.name
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(item, dest_path)

            config_file = self.test_example_dir / example_info["config"]
            if not config_file.exists():
                print(f"⚠️  警告: 配置文件不存在: {config_file}")
                return False

            print(f"✅ 示例文件已复制到: {self.test_example_dir}")
            return True

        except PermissionError:
            print("❌ 权限错误: 无法访问或复制文件")
            print("💡 请以管理员权限运行程序")
            return False
        except Exception as e:
            print(f"❌ 复制示例文件失败: {e}")
            return False

    def run_command(self, example_info, use_enhanced=False):
        """
        Run tricys command

        Args:
            example_info: Example information dictionary
            use_enhanced: Whether to enable enhanced mode

        Returns:
            bool: Whether command execution is successful
        """
        try:
            config_path = self.test_example_dir / example_info["config"]

            if not config_path.exists():
                print(f"❌ 配置文件不存在: {config_path}")
                return False

            # The main 'tricys' command automatically detects the workflow from the config file.
            cmd = ["tricys", "-c", str(config_path)]

            if use_enhanced:
                cmd.append("--enhanced")

            print("\n" + "=" * 60)
            print("🚀 开始执行仿真命令")
            print("=" * 60)
            print(f"📂 工作目录: {self.test_example_dir}")
            print(f"💻 执行命令: {' '.join(cmd)}")
            print(
                f"⚡ 运行模式: {'🔥 Enhanced (Compile Once)' if use_enhanced else '🐢 Standard'}"
            )
            print("=" * 60 + "\n")

            start_time = time.time()

            result = subprocess.run(
                cmd,
                cwd=self.test_example_dir,
                capture_output=False,
                text=True,
            )

            print("\n" + "=" * 60)

            if result.returncode == 0:
                execution_time = time.time() - start_time
                print(f"✅ 命令执行成功，执行时间: {execution_time:.2f} 秒")
                return True
            else:
                print(f"❌ 命令执行失败，返回码: {result.returncode}")
                return False

        except FileNotFoundError:
            print("❌ 找不到命令 'tricys'")
            print("💡 请确保已正确安装Tricys:")
            print("   pip install -e .")
            print("   或者")
            print("   pip install tricys")
            return False
        except Exception as e:
            print(f"❌ 执行命令时发生错误: {e}")
            return False

    def run_example(self, choice):
        """
        Run specified example

        Args:
            choice: User selected example number

        Returns:
            bool: Whether example execution is successful
        """
        if choice not in self.examples:
            print("\n❌ 无效的选择")
            return False

        example_info = self.examples[choice]

        # Default to Enhanced mode without prompting
        use_enhanced = True

        if not self.copy_example(example_info):
            return False

        success = self.run_command(example_info, use_enhanced=use_enhanced)

        if success:
            print(f"\n✅ 示例 '{example_info['name']}' 运行完成")
            if self.test_example_dir.exists():
                print(f"📊 结果文件位于: {self.test_example_dir}")
        else:
            print(f"\n❌ 示例 '{example_info['name']}' 运行失败")

        return success

    def show_help(self):
        """Display help information"""
        help_text = """
════════════════════════════════════════════════════════════
                    TRICYS 统一示例运行器帮助
════════════════════════════════════════════════════════════

  使用说明:
    1. 选择要运行的示例编号。
    2. 程序会自动复制示例文件到 test_example 目录。
    3. 执行 `tricys -c <配置文件>` 命令。
    4. 程序会根据配置文件内容自动识别并运行 `basic` 或 `analysis` 工作流。
    5. 查看运行结果和日志输出。

  示例类型说明:
    • [BASIC]:    基础仿真任务，如参数扫描、并发仿真等。
    • [ANALYSIS]: 复杂分析任务，如敏感性分析、TBR搜索等。

  注意事项:
    • 确保已正确安装 Tricys 和相关依赖 (`pip install -e .`)。
    • 运行前会清理 test_example 目录中对应的旧示例。
    • 结果文件保存在 test_example 目录中。

  快捷键:
    • h: 显示此帮助信息
    • s: 重新扫描示例目录
    • 0: 退出程序
    • Ctrl+C: 强制退出

════════════════════════════════════════════════════════════
        """.strip()
        print(help_text)

    def main(self):
        """Main program loop"""

        while True:
            self.show_menu()

            try:
                choice = input(
                    "\n请输入选择 (0-{}/h/s): ".format(len(self.examples))  # noqa
                ).strip()

                if choice == "0":
                    break

                if choice in self.examples:
                    self.run_example(choice)

                    while True:
                        continue_choice = (
                            input("\n是否继续运行其他示例? (y/n, 默认y): ")
                            .strip()
                            .lower()
                        )
                        if continue_choice in ["y", "yes", "是", "Y", ""]:
                            break
                        elif continue_choice in ["n", "no", "否", "N"]:
                            return
                elif choice.lower() == "h":
                    self.show_help()
                elif choice.lower() == "s":
                    print("\n🔄 正在重新扫描示例目录...")
                    self.examples = self._scan_examples()
                else:
                    print(
                        "\n❌ 无效的选择，请输入 0-{}、h 或 s".format(  # noqa
                            len(self.examples)
                        )
                    )

            except KeyboardInterrupt:
                print("\n\nℹ️  用户中断，程序退出")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")


def main():
    """Main function entry point"""
    try:
        runner = TricysAllTestRunner()
        runner.main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见!")
    except Exception as e:
        print(f"\n❌ 程序发生未预期错误: {e}")
        print("💡 请检查环境配置或联系开发者")


if __name__ == "__main__":
    main()
