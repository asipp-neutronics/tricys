"""本模块提供运行参数扫描模拟的图形界面。"""

import logging
import math
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from tricys.manager.config_manager import config_manager
from tricys.manager.logger_manager import logger_manager

from tricys.simulation.sweep_simulation import run_parameter_sweep
from tricys.utils.plot_utils import plot_startup_inventory

# Add project root to sys.path to allow absolute imports from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
logger = logging.getLogger(__name__)


class SimulationUI:
    """用于配置和运行聚变模拟参数扫描的GUI。"""

    def __init__(self, root: tk.Tk):
        """
        初始化模拟UI。

        参数:
            root (tk.Tk): 根Tkinter窗口。
        """
        self.root = root
        self.root.title("Fusion Simulation Parameter Sweep")
        self.root.geometry("450x600")
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.config = config_manager
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Model Name:").grid(row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar(
            value=self.config.get("simulation.model_name", "example.Cycle")
        )
        tk.Entry(self.root, textvariable=self.model_var).grid(
            row=0, column=1, padx=5, pady=5
        )

        param_A_frame = ttk.LabelFrame(
            self.root, text="Parameter A Settings (Fixed Values)", padding=10
        )
        param_A_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        tk.Label(param_A_frame, text="Parameter A Name:").grid(
            row=0, column=0, padx=5, pady=5
        )
        self.param_A_name_var = tk.StringVar(value="blanket.T")
        tk.Entry(param_A_frame, textvariable=self.param_A_name_var).grid(
            row=0, column=1, padx=5, pady=5
        )

        tk.Label(param_A_frame, text="Values (comma-separated):").grid(
            row=1, column=0, padx=5, pady=5
        )
        self.param_A_values_var = tk.StringVar(value="6,12,18,24")
        tk.Entry(param_A_frame, textvariable=self.param_A_values_var).grid(
            row=1, column=1, padx=5, pady=5
        )

        param_B_frame = ttk.LabelFrame(
            self.root, text="Parameter B Settings (Sweep)", padding=10
        )
        param_B_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        tk.Label(param_B_frame, text="Parameter B Name:").grid(
            row=0, column=0, padx=5, pady=5
        )
        self.param_B_name_var = tk.StringVar(value="blanket.TBR")
        tk.Entry(param_B_frame, textvariable=self.param_B_name_var).grid(
            row=0, column=1, padx=5, pady=5
        )

        tk.Label(param_B_frame, text="Min Value:").grid(row=1, column=0, padx=5, pady=5)
        self.min_var = tk.DoubleVar(value=1.05)
        tk.Entry(param_B_frame, textvariable=self.min_var, width=10).grid(
            row=1, column=1, padx=5, pady=5
        )

        tk.Label(param_B_frame, text="Max Value:").grid(row=2, column=0, padx=5, pady=5)
        self.max_var = tk.DoubleVar(value=1.15)
        tk.Entry(param_B_frame, textvariable=self.max_var, width=10).grid(
            row=2, column=1, padx=5, pady=5
        )

        tk.Label(param_B_frame, text="Number of Steps:").grid(
            row=3, column=0, padx=5, pady=5
        )
        self.steps_var = tk.IntVar(value=6)
        tk.Entry(param_B_frame, textvariable=self.steps_var, width=10).grid(
            row=3, column=1, padx=5, pady=5
        )

        tk.Label(param_B_frame, text="Desired Step Size (Optional):").grid(
            row=4, column=0, padx=5, pady=5
        )
        self.desired_step_size_var = tk.DoubleVar(value=0.02)
        tk.Entry(param_B_frame, textvariable=self.desired_step_size_var, width=10).grid(
            row=4, column=1, padx=5, pady=5
        )

        tk.Button(
            param_B_frame, text="Calculate Steps", command=self.calculate_steps
        ).grid(row=5, column=1, pady=5)

        tk.Label(self.root, text="Stop Time (s):").grid(row=3, column=0, padx=5, pady=5)
        self.stop_time_var = tk.DoubleVar(value=5000.0)
        tk.Entry(self.root, textvariable=self.stop_time_var).grid(
            row=3, column=1, padx=5, pady=5
        )

        tk.Label(self.root, text="Number of Sim Steps:").grid(
            row=4, column=0, padx=5, pady=5
        )
        self.num_sim_steps_var = tk.IntVar(value=5000)
        tk.Entry(self.root, textvariable=self.num_sim_steps_var).grid(
            row=4, column=1, padx=5, pady=5
        )

        tk.Label(self.root, text="Step Size (s):").grid(row=5, column=0, padx=5, pady=5)
        self.sim_step_size_var = tk.DoubleVar(value=1.0)
        tk.Entry(self.root, textvariable=self.sim_step_size_var, state="readonly").grid(
            row=5, column=1, padx=5, pady=5
        )

        tk.Button(self.root, text="Run Simulation", command=self.run_simulation).grid(
            row=6, column=1, pady=10
        )

    def calculate_steps(self):
        """根据期望的步长计算步数。"""
        try:
            min_val = self.min_var.get()
            max_val = self.max_var.get()
            desired_step_size = self.desired_step_size_var.get()

            if min_val is None or max_val is None or desired_step_size <= 0:
                raise ValueError("Invalid min/max value or desired step size.")
            if min_val > max_val:
                raise ValueError("Min value must be less than or equal to Max value.")
            if desired_step_size > (max_val - min_val):
                raise ValueError(
                    "Desired step size must be smaller than the range (Max - Min)."
                )

            steps = math.ceil((max_val - min_val) / desired_step_size) + 1
            if steps < 2:
                steps = 2

            self.steps_var.set(steps)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate steps: {str(e)}")

    def run_simulation(self):
        """运行参数扫描模拟并绘制结果。"""
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

            # Validate inputs
            if not param_A_name or not param_B_name:
                raise ValueError("Parameter names cannot be empty.")
            if not param_A_values_str:
                raise ValueError("Parameter A values cannot be empty.")
            if min_val is None or max_val is None or steps < 2:
                raise ValueError(
                    "Invalid parameter B range or number of steps (must be at least 2)."
                )
            if min_val > max_val:
                raise ValueError("Min value must be less than or equal to Max value.")
            if num_sim_steps < 2:
                raise ValueError("Number of simulation steps must be at least 2.")

            # Calculate simulation step size
            sim_step_size = stop_time / num_sim_steps
            self.sim_step_size_var.set(sim_step_size)

            # Parse Parameter A values
            param_A_vals = [float(val.strip()) for val in param_A_values_str.split(",")]
            param_A_values = {param_A_name: param_A_vals}

            # Calculate Parameter B sweep
            param_B_vals = np.linspace(min_val, max_val, steps)
            actual_step_size = (max_val - min_val) / (steps - 1) if steps > 1 else 0
            param_B_sweep = {param_B_name: param_B_vals}

            # Run simulation
            package_path = os.path.join(
                self.project_root, self.config.get("paths.package_path")
            )
            temp_dir = os.path.join(
                self.project_root, self.config.get("paths.temp_dir")
            )
            results_dir = os.path.join(
                self.project_root, self.config.get("paths.results_dir")
            )
            variableFilter = self.config.get("simulation.variableFilter", "time|sds\\.I\\[1\\]")

            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)

            result_path = run_parameter_sweep(
                package_path,
                model_name,
                param_A_values,
                param_B_sweep,
                stop_time,
                sim_step_size,
                temp_dir,
                results_dir,
                variableFilter
            )

            # Plot startup tritium inventory
            plot_path = plot_startup_inventory(
                result_path, param_A_name, param_B_name, results_dir
            )

            # Show success message
            result_filename = os.path.basename(result_path)
            plot_filename = os.path.basename(plot_path)
            messagebox.showinfo(
                "Success",
                f"Simulation completed successfully!\nResults saved as: {result_filename}\nActual step size for Parameter B: {actual_step_size:.4f}\nPlot saved as: {plot_filename}",
            )

        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")


if __name__ == "__main__":
    """Main function to initialize and run the GUI for parameter configuration."""
    root = tk.Tk()
    app = SimulationUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), sys.exit(0)])
    root.mainloop()
