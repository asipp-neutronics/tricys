import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sys
import os
import math
from run_simulation import run_parameter_sweep

class SimulationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fusion Simulation Parameter Sweep")
        self.root.geometry("450x600")  # 增加高度以容纳新输入框
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Model Name:").grid(row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar(value="FFCAS.Cycle")
        tk.Entry(self.root, textvariable=self.model_var).grid(row=0, column=1, padx=5, pady=5)

        # 参数 A 设置（固定值）
        param_A_frame = ttk.LabelFrame(self.root, text="Parameter A Settings (Fixed Values)", padding=10)
        param_A_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        tk.Label(param_A_frame, text="Parameter A Name:").grid(row=0, column=0, padx=5, pady=5)
        self.param_A_name_var = tk.StringVar(value="blanket.T")
        tk.Entry(param_A_frame, textvariable=self.param_A_name_var).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(param_A_frame, text="Values (comma-separated):").grid(row=1, column=0, padx=5, pady=5)
        self.param_A_values_var = tk.StringVar(value="6,12,18,24")
        tk.Entry(param_A_frame, textvariable=self.param_A_values_var).grid(row=1, column=1, padx=5, pady=5)

        # 参数 B 设置（扫描范围）
        param_B_frame = ttk.LabelFrame(self.root, text="Parameter B Settings (Sweep)", padding=10)
        param_B_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        tk.Label(param_B_frame, text="Parameter B Name:").grid(row=0, column=0, padx=5, pady=5)
        self.param_B_name_var = tk.StringVar(value="blanket.TBR")
        tk.Entry(param_B_frame, textvariable=self.param_B_name_var).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(param_B_frame, text="Min Value:").grid(row=1, column=0, padx=5, pady=5)
        self.min_var = tk.DoubleVar(value=1.05)
        tk.Entry(param_B_frame, textvariable=self.min_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(param_B_frame, text="Max Value:").grid(row=2, column=0, padx=5, pady=5)
        self.max_var = tk.DoubleVar(value=1.15)
        tk.Entry(param_B_frame, textvariable=self.max_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(param_B_frame, text="Number of Steps:").grid(row=3, column=0, padx=5, pady=5)
        self.steps_var = tk.IntVar(value=6)
        tk.Entry(param_B_frame, textvariable=self.steps_var, width=10).grid(row=3, column=1, padx=5, pady=5)

        # 辅助输入框：期望步长
        tk.Label(param_B_frame, text="Desired Step Size (Optional):").grid(row=4, column=0, padx=5, pady=5)
        self.desired_step_size_var = tk.DoubleVar(value=0.02)
        tk.Entry(param_B_frame, textvariable=self.desired_step_size_var, width=10).grid(row=4, column=1, padx=5, pady=5)

        tk.Button(param_B_frame, text="Calculate Steps", command=self.calculate_steps).grid(row=5, column=1, pady=5)

        tk.Label(self.root, text="Stop Time (s):").grid(row=3, column=0, padx=5, pady=5)
        self.stop_time_var = tk.DoubleVar(value=5000.0)
        tk.Entry(self.root, textvariable=self.stop_time_var).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(self.root, text="Number of Sim Steps:").grid(row=4, column=0, padx=5, pady=5)
        self.num_sim_steps_var = tk.IntVar(value=5000)  # 新增仿真步数输入框，默认 5000
        tk.Entry(self.root, textvariable=self.num_sim_steps_var).grid(row=4, column=1, padx=5, pady=5)

        tk.Label(self.root, text="Step Size (s):").grid(row=5, column=0, padx=5, pady=5)
        self.sim_step_size_var = tk.DoubleVar(value=1.0)  # 初始值设为 1.0，运行时根据步数计算
        tk.Entry(self.root, textvariable=self.sim_step_size_var, state='readonly').grid(row=5, column=1, padx=5, pady=5)  # 只读

        tk.Button(self.root, text="Run Simulation", command=self.run_simulation).grid(row=6, column=1, pady=10)

    def calculate_steps(self):
        try:
            min_val = self.min_var.get()
            max_val = self.max_var.get()
            desired_step_size = self.desired_step_size_var.get()

            if min_val is None or max_val is None or desired_step_size <= 0:
                raise ValueError("Invalid min/max value or desired step size.")
            if min_val > max_val:
                raise ValueError("Min value must be less than or equal to Max value.")
            if desired_step_size > (max_val - min_val):
                raise ValueError("Desired step size must be smaller than the range (Max - Min).")

            # 计算步数：ceil((max_val - min_val) / desired_step_size) + 1
            steps = math.ceil((max_val - min_val) / desired_step_size) + 1
            if steps < 2:
                steps = 2  # 至少需要2个点（包含两端）

            # 直接更新 Number of Steps 输入框
            self.steps_var.set(steps)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate steps: {str(e)}")

    def run_simulation(self):
        try:
            model_name = self.model_var.get()
            param_A_name = self.param_A_name_var.get()
            param_A_values_str = self.param_A_values_var.get()
            param_B_name = self.param_B_name_var.get()
            min_val = self.min_var.get()
            max_val = self.max_var.get()
            steps = self.steps_var.get()
            stop_time = self.stop_time_var.get()
            num_sim_steps = self.num_sim_steps_var.get()

            # 验证输入
            if not param_A_name or not param_B_name:
                raise ValueError("Parameter names cannot be empty.")
            if not param_A_values_str:
                raise ValueError("Parameter A values cannot be empty.")
            if min_val is None or max_val is None or steps < 2:
                raise ValueError("Invalid parameter B range or number of steps (must be at least 2).")
            if min_val > max_val:
                raise ValueError("Min value must be less than or equal to Max value.")
            if num_sim_steps < 2:
                raise ValueError("Number of simulation steps must be at least 2.")

            # 根据仿真步数计算 sim_step_size
            sim_step_size = stop_time / num_sim_steps
            self.sim_step_size_var.set(sim_step_size)

            # 解析参数 A 的值（逗号分隔的字符串转列表）
            param_A_vals = [float(val.strip()) for val in param_A_values_str.split(',')]
            param_A_values = {param_A_name: param_A_vals}

            # 使用步数计算参数 B 的扫描范围（闭区间）
            param_B_vals = np.linspace(min_val, max_val, steps)
            # 计算实际步长并提示
            actual_step_size = (max_val - min_val) / (steps - 1) if steps > 1 else 0
            param_B_sweep = {param_B_name: param_B_vals}

            package_path = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/FFCAS/package.mo"
            temp_dir = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp"
            result_path = run_parameter_sweep(package_path, model_name, param_A_values, param_B_sweep, stop_time, sim_step_size, temp_dir)

            # 自动绘制 Start-up Tritium Inventory 折线图
            plot_path = self.plot_startup_inventory(result_path, param_A_name, param_B_name)

            # 合并提示信息
            result_filename = os.path.basename(result_path)
            plot_filename = os.path.basename(plot_path)
            messagebox.showinfo("Success", f"Simulation completed successfully!\nResults saved as: {result_filename}\nActual step size for Parameter B: {actual_step_size:.4f}\nPlot saved as: {plot_filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")

    def plot_startup_inventory(self, csv_path, param_A_name, param_B_name):
        # 延迟加载库
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 设置绘图风格
        sns.set(style='whitegrid')

        # 只读取需要的列
        param_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        param_columns = [col for col in param_columns if col != 'time']  # 排除 time 列
        df = pd.read_csv(csv_path, usecols=param_columns)

        # 解析列名，提取参数 A 和参数 B 的值，并计算 Start-up Tritium Inventory
        param_A_values = {}
        for col in param_columns:
            # 列名格式：paramA=valA_paramB=valB
            parts = col.split('_')
            param_A_part = parts[0]  # paramA=valA
            param_B_part = parts[1]  # paramB=valB
            param_A_val = float(param_A_part.split('=')[1])
            param_B_val = float(param_B_part.split('=')[1])
            
            # 使用 NumPy 高效计算初始值和最小值
            col_data = df[col].to_numpy()
            initial_value = col_data[0]  # 第一行的值
            min_value = np.min(col_data)  # 最小值
            startup_inventory = initial_value - min_value
            
            if param_A_val not in param_A_values:
                param_A_values[param_A_val] = []
            param_A_values[param_A_val].append((param_B_val, startup_inventory))

        # 绘制折线图
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette('tab10', len(param_A_values))

        for i, (param_A_val, data) in enumerate(param_A_values.items()):
            # 按参数 B 排序
            data_sorted = sorted(data, key=lambda x: x[0])
            param_B_vals = [x[0] for x in data_sorted]
            startup_inventories = [x[1] for x in data_sorted]
            
            # 绘制折线
            plt.plot(param_B_vals, startup_inventories, marker='o', label=f"{param_A_name}={param_A_val:.3f}", color=colors[i], linewidth=1.5)

        plt.xlabel(param_B_name)
        plt.ylabel('Start-up Tritium Inventory')
        plt.title(f'Start-up Tritium Inventory vs {param_B_name} for Different {param_A_name}')
        plt.legend(loc='best', fontsize=8)
        plt.grid(True)
        plt.margins(x=0.05, y=0.1)

        # 保存图像
        save_dir = 'D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp'
        png_path = os.path.join(save_dir, f'startup_tritium_inventory_{param_A_name}_vs_{param_B_name}.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        return png_path

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), sys.exit(0)])
    root.mainloop()