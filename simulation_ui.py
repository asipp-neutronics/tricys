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
        self.root.geometry("400x550")
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Model Name:").grid(row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar(value="FFCAS.Cycle")
        tk.Entry(self.root, textvariable=self.model_var).grid(row=0, column=1, padx=5, pady=5)

        # 参数 A 设置（固定值）
        param_A_frame = ttk.LabelFrame(self.root, text="Parameter A Settings (Fixed Values)", padding=10)
        param_A_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        tk.Label(param_A_frame, text="Parameter A Name:").grid(row=0, column=0, padx=5, pady=5)
        self.param_A_name_var = tk.StringVar(value="blanket.A")
        tk.Entry(param_A_frame, textvariable=self.param_A_name_var).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(param_A_frame, text="Values (comma-separated):").grid(row=1, column=0, padx=5, pady=5)
        self.param_A_values_var = tk.StringVar(value="1.0,1.1,1.2")
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

        tk.Label(self.root, text="Step Size (s):").grid(row=4, column=0, padx=5, pady=5)
        self.sim_step_size_var = tk.DoubleVar(value=1.0)
        tk.Entry(self.root, textvariable=self.sim_step_size_var).grid(row=4, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Run Simulation", command=self.run_simulation).grid(row=5, column=1, pady=10)

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
            sim_step_size = self.sim_step_size_var.get()

            # 验证输入
            if not param_A_name or not param_B_name:
                raise ValueError("Parameter names cannot be empty.")
            if not param_A_values_str:
                raise ValueError("Parameter A values cannot be empty.")
            if min_val is None or max_val is None or steps < 2:
                raise ValueError("Invalid parameter B range or number of steps (must be at least 2).")
            if min_val > max_val:
                raise ValueError("Min value must be less than or equal to Max value.")

            # 解析参数 A 的值（逗号分隔的字符串转列表）
            param_A_vals = [float(val.strip()) for val in param_A_values_str.split(',')]
            param_A_values = {param_A_name: param_A_vals}

            # 使用步数计算参数 B 的扫描范围（闭区间）
            param_B_vals = np.linspace(min_val, max_val, steps)
            # 计算实际步长并提示
            actual_step_size = (max_val - min_val) / (steps - 1) if steps > 1 else 0
            step_notice = f"\nActual step size for Parameter B: {actual_step_size:.4f}"
            param_B_sweep = {param_B_name: param_B_vals}

            package_path = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/FFCAS/package.mo"
            temp_dir = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp"
            result_path = run_parameter_sweep(package_path, model_name, param_A_values, param_B_sweep, stop_time, sim_step_size, temp_dir)

            # 提取文件名（不含路径）
            result_filename = os.path.basename(result_path)
            messagebox.showinfo("Success", f"Simulation completed successfully!\nResults saved as: {result_filename}{step_notice}")
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), sys.exit(0)])
    root.mainloop()