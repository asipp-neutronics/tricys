import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sys
from run_simulation import run_parameter_sweep

class SimulationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fusion Simulation Parameter Sweep")
        self.root.geometry("400x350")
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Model Name:").grid(row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar(value="FFCAS.Cycle")
        tk.Entry(self.root, textvariable=self.model_var).grid(row=0, column=1, padx=5, pady=5)

        param_frame = ttk.LabelFrame(self.root, text="Parameter Settings", padding=10)
        param_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        tk.Label(param_frame, text="Parameter Name:").grid(row=0, column=0, padx=5, pady=5)
        self.param_name_var = tk.StringVar()
        tk.Entry(param_frame, textvariable=self.param_name_var).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(param_frame, text="Min Value:").grid(row=1, column=0, padx=5, pady=5)
        self.min_var = tk.DoubleVar()
        tk.Entry(param_frame, textvariable=self.min_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(param_frame, text="Max Value:").grid(row=2, column=0, padx=5, pady=5)
        self.max_var = tk.DoubleVar()
        tk.Entry(param_frame, textvariable=self.max_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(param_frame, text="Number of Steps:").grid(row=3, column=0, padx=5, pady=5)
        self.steps_var = tk.IntVar(value=20)
        tk.Entry(param_frame, textvariable=self.steps_var, width=10).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(self.root, text="Stop Time (s):").grid(row=2, column=0, padx=5, pady=5)
        self.stop_time_var = tk.DoubleVar(value=5000.0)
        tk.Entry(self.root, textvariable=self.stop_time_var).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self.root, text="Step Size (s):").grid(row=3, column=0, padx=5, pady=5)
        self.step_size_var = tk.DoubleVar(value=1.0)
        tk.Entry(self.root, textvariable=self.step_size_var).grid(row=3, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Run Simulation", command=self.run_simulation).grid(row=4, column=1, pady=10)

    def run_simulation(self):
        try:
            model_name = self.model_var.get()
            param_name = self.param_name_var.get()
            min_val = self.min_var.get()
            max_val = self.max_var.get()
            steps = self.steps_var.get()
            stop_time = self.stop_time_var.get()
            step_size = self.step_size_var.get()

            if not param_name:
                raise ValueError("Parameter name cannot be empty.")
            if min_val is None or max_val is None or steps <= 0:
                raise ValueError("Invalid parameter range or steps.")

            param_sweep = {param_name: np.linspace(min_val, max_val, steps)}

            package_path = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/FFCAS/package.mo"
            temp_dir = "D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp"
            run_parameter_sweep(package_path, model_name, param_sweep, stop_time, step_size, temp_dir)
            messagebox.showinfo("Success", "Simulation completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), sys.exit(0)])  # 关闭时退出
    root.mainloop()