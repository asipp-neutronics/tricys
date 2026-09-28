#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tricys BASIC Example Runner
A BASIC sample configuration specifically designed for running tricys commands
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


class TricysTestRunner:
    """Tricys BASIC Example Runner"""

    def __init__(self):
        """Initialize the runner"""
        # Force UTF-8 encoding for stdout/stderr to support emojis on Windows
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        # Locate path to example data within package
        self.runner_dir = Path(__file__).parent
        self.package_root = self.runner_dir.parent
        self.workspace_dir = Path.cwd()
        self.example_dir = self.package_root / "example_data" / "basic"
        self.test_example_base_dir = Path.cwd() / "test_example"

        # Automatically scan and generate example configurations
        self.examples = self._scan_examples()

    def _scan_examples(self):
        """
        Read BASIC example configurations from example/basic/example_runner.json

        Returns:
            dict: Example configuration dictionary
        """
        examples = {}

        # Read BASIC example configuration file
        config_file = self.example_dir / "example_runner.json"
        if not config_file.exists():
            print(f"⚠️  警告: BASIC示例配置文件不存在: {config_file}")
            print("请创建 example/basic/example_runner.json 文件")
            return examples

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            print("\n" + "=" * 60)
            print("🔄 正在扫描 BASIC 示例目录...")
            print("=" * 60 + "\n")

            print(f"📦 [BASIC] {config_data.get('description', '')}")
            print("-" * 60)

            examples_list = config_data.get("examples", [])
            counter = 1

            for example_config in examples_list:
                # Check if example is enabled
                if not example_config.get("enabled", True):
                    # print(f"  ⏸️  [跳过] {example_config.get('name', 'Unknown')}")
                    continue

                # Check if configuration file exists
                example_path = self.example_dir / example_config["path"]
                config_path = example_path / example_config["config"]

                if not config_path.exists():
                    print(f"  ⚠️  [缺失] {example_config['name']} ({config_path})")
                    continue

                examples[str(counter)] = {
                    "name": example_config["name"],
                    "path": example_config["path"],
                    "config": example_config["config"],
                    "command": example_config.get("command", "tricys"),
                    "description": example_config["description"],
                }

                print(f"  ✅ {counter}. {example_config['name']}")
                counter += 1

            print("-" * 60)
            print(f"🎉 扫描完成: 共加载 {len(examples)} 个 BASIC 示例")
            print("=" * 60 + "\n")

        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
            print("请检查 example_runner.json 文件格式")
        except Exception as e:
            print(f"❌ 读取配置文件时出错: {e}")

        return examples

    def show_menu(self):
        """Display available example menu"""
        print("\n" + "=" * 60)
        print(f"{'Tricys BASIC 示例运行器':^56}")
        print("=" * 60 + "\n")

        if not self.examples:
            print("❌ 未发现任何BASIC示例")
            print("请检查 tricys/example/example_data/basic 目录是否存在配置文件")
        else:
            for key, example in self.examples.items():
                print(f"  {key}. {example['name']}")
                print(f"     📝 {example['description']}")
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
            source_path = self.example_dir / example_info["path"]

            # Check if source path exists
            if not source_path.exists():
                print(f"❌ 示例路径不存在: {source_path}")
                return False

            # Create corresponding subdirectory based on example type
            example_type = (
                example_info["path"].split("/")[0]
                if "/" in example_info["path"]
                else example_info["path"]
            )
            self.test_example_dir = self.test_example_base_dir / "basic" / example_type

            # If corresponding test_example subdirectory exists, delete it first
            if self.test_example_dir.exists():
                print("─" * 50)
                print(f"🧹 正在清理旧的测试目录: {self.test_example_dir}")
                shutil.rmtree(self.test_example_dir)

            # Create base directory
            self.test_example_base_dir.mkdir(exist_ok=True)

            # Copy entire example directory
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

            # Verify if key files exist
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

            # Record start time
            start_time = time.time()

            # Switch to test_example directory to execute command
            result = subprocess.run(
                cmd,
                cwd=self.test_example_dir,
                capture_output=False,  # Allow real-time output
                text=True,
            )

            # Calculate execution time

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
            print("❌ 无效的选择")
            return False

        example_info = self.examples[choice]

        # Ask for enhanced mode
        print("\n" + "-" * 30)
        enhanced_input = (
            input("是否启用 Enhanced 模式 (Compile Once)? (y/n, 默认y): ")
            .strip()
            .lower()
        )
        use_enhanced = enhanced_input in ["", "y", "yes", "是"]

        # 1. Copy example files
        if not self.copy_example(example_info):
            return False

        # 2. Run command
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
                    Tricys BASIC 示例运行器帮助
════════════════════════════════════════════════════════════

  使用说明:
    1. 选择要运行的BASIC示例编号。
    2. 程序会自动复制示例文件到 test_example 目录。
    3. 执行 `tricys -c <配置文件>` 命令。
    4. 程序会自动识别为 basic 工作流并执行。
    5. 查看运行结果和日志输出。

  BASIC功能特性:
    • 参数扫描: 支持多维参数扫描和批量仿真
    • 配置驱动: 通过JSON配置文件定义仿真参数
    • 结果输出: 自动生成CSV结果文件和日志
    • 并发控制: 可配置并发度和执行策略

  注意事项:
    • 确保已正确安装 Tricys 和相关依赖 (`pip install -e .`)。
    • 运行前会清理 test_example 目录。
    • 结果文件保存在 test_example 目录中。
    • BASIC模式通常适用于批量仿真和自动化任务

  快捷键:
    • h: 显示此帮助信息
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
                    "\n请输入选择 (0-{}/h/s): ".format(len(self.examples))
                ).strip()

                if choice == "0":
                    break

                if choice in self.examples:
                    self.run_example(choice)

                    # Ask whether to continue
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
                    # Rescan examples
                    print("\n🔄 正在重新扫描示例目录...")
                    self.examples = self._scan_examples()
                    if self.examples:
                        print(f"✅ 重新扫描完成，发现 {len(self.examples)} 个示例")
                    else:
                        print("❌ 未发现任何示例")
                else:
                    print(
                        "\n❌ 无效的选择，请输入 0-{}、h 或 s".format(
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
        runner = TricysTestRunner()
        runner.main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见!")
    except Exception as e:
        print(f"\n❌ 程序发生未预期错误: {e}")
        print("💡 请检查环境配置或联系开发者")


if __name__ == "__main__":
    main()
