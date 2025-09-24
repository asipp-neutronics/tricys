#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tricys CLI Example Runner
A CLI sample configuration specifically designed for running tricys commands
"""
import os
import shutil
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime


class TricysTestRunner:
    """Tricys CLI Example Runner"""    
    def __init__(self):
        """Initialize the runner"""
        # Locate project root directory from script/example_runner directory
        self.script_dir = Path(__file__).parent.parent.parent
        self.workspace_dir = self.script_dir
        self.example_dir = self.workspace_dir / "example" / "cli"
        self.test_example_base_dir = self.workspace_dir / "test_example"
        
        # Automatically scan and generate example configurations
        self.examples = self._scan_examples()
    
    def _scan_examples(self):
        """
        Read CLI example configurations from example/cli/example_runner.json
        
        Returns:
            dict: Example configuration dictionary
        """
        examples = {}
        
        # Read CLI example configuration file
        config_file = self.example_dir / "example_runner.json"
        if not config_file.exists():
            print(f"⚠️  警告: CLI示例配置文件不存在: {config_file}")
            print("请创建 example/cli/example_runner.json 文件")
            return examples
        
        try:            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            print("\n" + "="*60 + "\n")

            print(f"📄 正在读取CLI示例配置: {config_data.get('description', '')}")
            
            examples_list = config_data.get('examples', [])
            counter = 1
            
            for example_config in examples_list:
                # Check if example is enabled
                if not example_config.get('enabled', True):
                    print(f"  ⏸️  跳过禁用的示例: {example_config.get('name', 'Unknown')}")
                    continue
                
                # Check if configuration file exists
                example_path = self.example_dir / example_config['path']
                config_path = example_path / example_config['config']
                
                if not config_path.exists():
                    print(f"  ⚠️  跳过缺失配置文件的示例: {example_config['name']} ({config_path})")
                    continue
                
                examples[str(counter)] = {
                    "name": example_config['name'],
                    "path": example_config['path'],
                    "config": example_config['config'],
                    "command": example_config['command'],
                    "description": example_config['description'],
                }
                
                print(f"  ✅ 加载示例: {example_config['name']}")
                counter += 1
            
            print(f"🎉 成功加载 {len(examples)} 个CLI示例")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
            print("请检查 example_runner.json 文件格式")
        except Exception as e:
            print(f"❌ 读取配置文件时出错: {e}")
        
        return examples

    
    def show_menu(self):
        """Display available example menu"""
        print("\n" + "="*60)
        print("         Tricys CLI 示例运行器")
        print("="*60 + "\n")
        
        if not self.examples:
            print("❌ 未发现任何CLI示例")
            print("请检查 example/cli 目录是否存在配置文件")
        else:
            for key, example in self.examples.items():
                print(f"  {key}. {example['name']}")
                print(f"     描述: {example['description']}")
                print(f"     配置: {example['config']}")
                print(f"     命令: {example['command']}")
                print()
        
        print("  0. 退出程序")
        print("  h. 显示帮助信息")
        print("  s. 重新扫描示例目录\n")
        print("="*60 )
    
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
            example_type = example_info["path"].split('/')[0] if '/' in example_info["path"] else example_info["path"]
            self.test_example_dir = self.test_example_base_dir / "cli" / example_type
            
            # If corresponding test_example subdirectory exists, delete it first
            if self.test_example_dir.exists():
                print("─" * 50)
                print(f"🧹 正在清理旧的测试目录: {self.test_example_dir}")
                shutil.rmtree(self.test_example_dir)
            
            # Create base directory
            self.test_example_base_dir.mkdir(exist_ok=True)
            
            # Copy entire example directory
            print("─" * 50)
            print(f"📋 正在复制示例目录...")
            print(f"   从: {source_path}")
            print(f"   到: {self.test_example_dir}")
            
            shutil.copytree(source_path, self.test_example_dir)
            
            # Verify if key files exist
            config_file = self.test_example_dir / example_info["config"]
            if not config_file.exists():
                print(f"⚠️  警告: 配置文件不存在: {config_file}")
                return False
            
            print(f"✅ 示例文件已复制到: {self.test_example_dir}")
            return True
            
        except PermissionError:
            print(f"❌ 权限错误: 无法访问或复制文件")
            print("💡 请以管理员权限运行程序")
            return False
        except Exception as e:
            print(f"❌ 复制示例文件失败: {e}")
            return False
    
    def run_command(self, example_info):
        """
        Run tricys command
        
        Args:
            example_info: Example information dictionary
            
        Returns:
            bool: Whether command execution is successful
        """
        try:
            config_path = self.test_example_dir / example_info["config"]
            command = example_info["command"]
            
            # Check if configuration file exists
            if not config_path.exists():
                print(f"❌ 配置文件不存在: {config_path}")
                return False
            
            # Build command
            cmd = [command, "-c", str(config_path)]
            
            print(f"\n📂 工作目录: {self.test_example_dir}")
            print("─" * 50)
            
            # Record start time
            start_time = time.time()
            
            # Switch to test_example directory to execute command
            result = subprocess.run(
                cmd,
                cwd=self.test_example_dir,
                capture_output=False,  # Allow real-time output
                text=True
            )
            
            # Calculate execution time
            
            print("─" * 50)
            
            if result.returncode == 0:
                execution_time = time.time() - start_time
                print(f"✅ 命令执行成功，执行时间: {execution_time:.2f} 秒")
                return True
            else:
                print(f"❌ 命令执行失败，返回码: {result.returncode}")
                return False
                
        except FileNotFoundError:
            print(f"❌ 找不到命令 '{command}'")
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
        
        # 1. Copy example files
        if not self.copy_example(example_info):
            return False
        
        # 2. Run command
        success = self.run_command(example_info)
        
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
                    Tricys CLI 示例运行器帮助                   
════════════════════════════════════════════════════════════

  使用说明:                                                    
    1. 选择要运行的CLI示例编号 (1-4)                           
    2. 程序会自动复制示例文件到 test_example 目录               
    3. 执行相应的 tricys 命令                                  
    4. 查看运行结果和日志输出                                   
                                                              
  CLI示例类型说明:                                            
    • 并发仿真: 使用多线程并行执行多个仿真任务                  
    • 非并发仿真: 串行执行仿真任务                              
    • 协同仿真: 集成外部仿真软件的联合仿真                        
    • 并发协同仿真: 并行执行多个协同仿真任务                    
                                                               
  CLI功能特性:                                                
    • 参数扫描: 支持多维参数扫描和批量仿真                      
    • 配置驱动: 通过JSON配置文件定义仿真参数                   
    • 结果输出: 自动生成CSV结果文件和日志                       
    • 并发控制: 可配置并发度和执行策略                          
                                                               
  注意事项:                                                    
    • 确保已正确安装 Tricys 和相关依赖                        
    • 运行前会清理 test_example 目录                          
    • 结果文件保存在 test_example 目录中                      
    • CLI模式通常适用于批量仿真和自动化任务                    
                                                               
  快捷键:                                                      
    • h: 显示此帮助信息                                        
    • 0: 退出程序                                              
    • Ctrl+C: 强制退出                                      

════════════════════════════════════════════════════════════
        """
        print(help_text)
    

    

    
    def main(self):
        """Main program loop"""

        while True:
            self.show_menu()
            
            try:
                choice = input("\n请输入选择 (0-{}/h/s): ".format(len(self.examples))).strip()
                
                if choice == "0":
                    break
                
                if choice in self.examples:
                    self.run_example(choice)
                    
                    # Ask whether to continue
                    while True:
                        continue_choice = input("\n是否继续运行其他示例? (y/n, 默认y): ").strip().lower()
                        if continue_choice in ['y', 'yes', '是', 'Y']:
                            break
                        elif continue_choice in ['n', 'no', '否', 'N']:
                            return
                        else:
                            continue_choice = 'y'
                            break
                elif choice.lower() == 'h':
                    self.show_help()
                elif choice.lower() == 's':
                    # Rescan examples
                    print("\n🔄 正在重新扫描示例目录...")
                    self.examples = self._scan_examples()
                    if self.examples:
                        print(f"✅ 重新扫描完成，发现 {len(self.examples)} 个示例")
                    else:
                        print("❌ 未发现任何示例")
                else:
                    print("\n❌ 无效的选择，请输入 0-{}、h 或 s".format(len(self.examples)))
                    
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