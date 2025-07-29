import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from OMPython import OMCSessionZMQ
from OMPython import ModelicaSystem
import logging
from parameter_parser import get_available_parameters

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OMPython')


def get_unique_filename(base_path, filename):
    base_name, ext = os.path.splitext(filename)
    counter = 0
    new_filename = filename
    new_filepath = os.path.join(base_path, new_filename)

    while os.path.exists(new_filepath):
        counter += 1
        new_filename = f"{base_name}_{counter}{ext}"
        new_filepath = os.path.join(base_path, new_filename)

    return new_filename


def run_parameter_sweep(package_path, model_name, param_A_values, param_B_sweep, stop_time, step_size, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)

    mod = None
    try:
        logger.info(f"Creating ModelicaSystem for {model_name}")
        mod = ModelicaSystem(fileName=package_path, modelName=model_name)
        logger.info("Building model")
        mod.buildModel()
        error = mod.getconn.sendExpression("getErrorString()")
        if error:
            raise RuntimeError(f"Model build failed: {error}")

        param_A_name = list(param_A_values.keys())[0]
        param_A_vals = param_A_values[param_A_name]
        param_B_name = list(param_B_sweep.keys())[0]
        param_B_vals = param_B_sweep[param_B_name]

        counter = 0
        for param_A_val in param_A_vals:
            for param_B_val in param_B_vals:
                logger.info(
                    f"Simulating: {param_A_name}={param_A_val}, {param_B_name}={param_B_val}")
                mod.setParameters(
                    [f"{param_A_name}={param_A_val}", f"{param_B_name}={param_B_val}"])
                mod.setSimulationOptions([
                    f"stopTime={stop_time}",
                    "tolerance=1e-6",
                    "outputFormat=csv",
                    "variableFilter=time|sds\\.I\\[1\\]",
                    f"stepSize={step_size}"
                ])
                base_filename = f"simulation_results_{counter}.csv"
                output_csv = os.path.join(
                    temp_dir, get_unique_filename(temp_dir, base_filename))
                logger.info(f"Simulation output file: {output_csv}")
                mod.simulate(resultfile=output_csv)
                error = mod.getconn.sendExpression("getErrorString()")
                if error:
                    raise RuntimeError(f"Simulation failed: {error}")
                if not os.path.exists(output_csv):
                    raise FileNotFoundError(
                        f"Simulation output file not created: {output_csv}")
                counter += 1

        combined_df = None
        original_csv_files = []
        rises_info = []
        counter = 0
        for param_A_val in param_A_vals:
            for param_B_val in param_B_vals:
                csv_file = os.path.join(
                    temp_dir, f"simulation_results_{counter}.csv")
                if os.path.exists(csv_file):
                    original_csv_files.append(csv_file)
                    df = pd.read_csv(csv_file)
                    if 'time' not in df or 'sds.I[1]' not in df:
                        logger.warning(
                            f"Invalid CSV format in {csv_file}: {df.columns}")
                        continue
                    if combined_df is None:
                        combined_df = df[['time']].copy()
                    column_name = f"{param_A_name}={param_A_val:.3f}_{param_B_name}={param_B_val:.3f}"
                    combined_df[column_name] = df['sds.I[1]']
                    data = df['sds.I[1]'].to_numpy()
                    diffs = np.diff(data)
                    mid = len(diffs) // 2
                    rises = any(diffs[:mid] < 0) and any(diffs[mid:] > 0)
                    rises_info.append({
                        param_A_name: param_A_val,
                        "blanket.TBR": param_B_val,
                        "rises": rises
                    })
                else:
                    logger.warning(f"CSV file {csv_file} not found.")
                    continue
                counter += 1

        if combined_df is None:
            raise ValueError("No valid simulation results found")

        base_combined_filename = f"{param_A_name.replace('.', '_')}_{param_B_name.replace('.', '_')}.csv"
        combined_csv_path = os.path.join(
            temp_dir, get_unique_filename(temp_dir, base_combined_filename))
        combined_df.to_csv(combined_csv_path, index=False)
        logger.info(f"Combined CSV file saved to: {combined_csv_path}")

        rises_csv_path = os.path.join(
            temp_dir, get_unique_filename(temp_dir, "rises_info.csv"))
        pd.DataFrame(rises_info).to_csv(rises_csv_path, index=False)
        logger.info(f"Rises info saved to: {rises_csv_path}")

        for csv_file in original_csv_files:
            try:
                os.remove(csv_file)
            except:
                logger.warning(f"Failed to remove temporary file: {csv_file}")

        return combined_csv_path, rises_csv_path

    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        raise
    finally:
        if mod is not None:
            try:
                mod.getconn.sendExpression("quit()")
            except Exception as e:
                logger.warning(f"Failed to close ModelicaSystem: {str(e)}")
            try:
                del mod
            except:
                pass


class SimulationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TBR Analysis")
        self.root.geometry("450x650")
        self.package_path = "./example/package.mo"
        self.available_params = get_available_parameters(
            self.package_path, "example.Cycle")
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Model Name:").grid(
            row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar(value="example.Cycle")
        tk.Entry(self.root, textvariable=self.model_var).grid(
            row=0, column=1, padx=5, pady=5)

        param_frame = ttk.LabelFrame(
            self.root, text="Parameter Selection", padding=10)
        param_frame.grid(row=1, column=0, columnspan=2,
                         padx=5, pady=5, sticky="nsew")

        tk.Label(param_frame, text="Parameter Name:").grid(
            row=0, column=0, padx=5, pady=5)
        self.param_var = tk.StringVar(value="i_iss.T" if "i_iss.T" in self.available_params else (
            self.available_params[0] if self.available_params else ""))
        ttk.Combobox(param_frame, textvariable=self.param_var, values=self.available_params,
                     state="readonly").grid(row=0, column=1, padx=5, pady=5)

        tk.Label(param_frame, text="Values (comma-separated):").grid(row=1,
                                                                     column=0, padx=5, pady=5)
        self.param_values_var = tk.StringVar(value="12.0")
        tk.Entry(param_frame, textvariable=self.param_values_var).grid(
            row=1, column=1, padx=5, pady=5)

        tk.Label(param_frame, text="Tip: Check console for parameters").grid(
            row=2, column=0, columnspan=2, padx=5, pady=5)

        tk.Label(self.root, text="Stop Time (s):").grid(
            row=2, column=0, padx=5, pady=5)
        self.stop_time_var = tk.DoubleVar(value=5000.0)
        tk.Entry(self.root, textvariable=self.stop_time_var).grid(
            row=2, column=1, padx=5, pady=5)
        tk.Label(self.root, text="Number of Intervals:").grid(
            row=3, column=0, padx=5, pady=5)
        self.num_intervals_var = tk.IntVar(value=5000)
        tk.Entry(self.root, textvariable=self.num_intervals_var).grid(
            row=3, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Run Sweep and Plot",
                  command=self.run_sweep).grid(row=4, column=1, pady=10)

    def run_sweep(self):
        try:
            model_name = self.model_var.get()
            param_name = self.param_var.get().strip()
            param_values_str = self.param_values_var.get()
            stop_time = self.stop_time_var.get()
            num_intervals = self.num_intervals_var.get()

            if not model_name:
                raise ValueError("Model name cannot be empty.")
            if not param_name:
                raise ValueError("Parameter name cannot be empty.")
            if not param_values_str:
                raise ValueError("Parameter values cannot be empty.")
            if stop_time <= 0:
                raise ValueError("Stop time must be positive.")
            if num_intervals < 2:
                raise ValueError("Number of intervals must be at least 2.")

            try:
                param_values = [float(x.strip())
                                for x in param_values_str.split(',')]
                if not param_values:
                    raise ValueError("No valid parameter values provided.")
            except ValueError:
                raise ValueError(
                    "Parameter values must be comma-separated numbers (e.g., 12.0 or 6.0,12.0).")

            temp_dir = "./temp"
            if not os.path.exists(self.package_path):
                raise FileNotFoundError(
                    f"Package file not found: {self.package_path}")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            param_A_values = {param_name: param_values}
            param_B_sweep = {"blanket.TBR": np.arange(1.01, 1.25, 0.01)}
            step_size = stop_time / num_intervals

            logger.info(
                f"Starting sweep: {param_name}={param_values}, TBR range=[1.01, 1.25]")
            result_path, rises_csv_path = run_parameter_sweep(
                package_path=self.package_path,
                model_name=model_name,
                param_A_values=param_A_values,
                param_B_sweep=param_B_sweep,
                stop_time=stop_time,
                step_size=step_size,
                temp_dir=temp_dir
            )

            logger.info(f"Plotting results from {result_path}")
            plot_paths = self.plot_results(
                result_path, param_name, param_values, stop_time, temp_dir)

            result_filename = os.path.basename(result_path)
            rises_filename = os.path.basename(rises_csv_path)
            plot_filenames = [os.path.basename(p) for p in plot_paths]
            messagebox.showinfo(
                "Success",
                f"Sweep completed!\nResults saved as: {result_filename}\nRises info saved as: {rises_filename}\n"
                f"Plots saved as: {', '.join(plot_filenames)}"
            )

        except Exception as e:
            logger.error(f"Sweep failed: {str(e)}")
            messagebox.showerror(
                "Error", f"Sweep failed: {str(e)}. Check console for available parameters.")

    def plot_results(self, csv_path, param_name, param_values, stop_time, temp_dir):
        try:
            df = pd.read_csv(csv_path)
            logger.info(
                f"CSV loaded: {csv_path}, shape={df.shape}, columns={df.columns.tolist()}")
            time = df['time'].to_numpy()

            if len(time) < 2:
                raise ValueError("CSV contains insufficient time points")

            max_time = stop_time
            for column in df.columns[1:]:
                data = df[column].to_numpy()
                if len(data) < 3:
                    logger.warning(
                        f"Skipping column {column}: insufficient data")
                    continue
                diffs = np.diff(data)
                for i in range(1, len(diffs)):
                    if i > len(diffs)//2 and diffs[i] > 0:
                        rise_time = time[i]
                        max_time = min(max_time, rise_time * 1.5)
                        logger.info(
                            f"Rise detected for {column} at {rise_time:.2f}s")
                        break

            time_mask = time <= max_time
            if not any(time_mask):
                logger.warning("No data within max_time, using full range")
                time_mask = np.ones_like(time, dtype=bool)
            time = time[time_mask]
            df = df.iloc[time_mask]

            plot_paths = []
            curves_per_plot = 5
            sns.set(style='whitegrid')
            for param_val in param_values:
                param_columns = [col for col in df.columns[1:]
                                 if f"{param_name}={param_val:.3f}" in col]
                for i in range(0, len(param_columns), curves_per_plot):
                    plt.figure(figsize=(10, 6))
                    colors = sns.color_palette("tab20", min(
                        curves_per_plot, len(param_columns) - i))
                    for idx, column in enumerate(param_columns[i:i+curves_per_plot]):
                        tbr_value = float(column.split("blanket.TBR=")[1])
                        plt.plot(
                            time, df[column], label=f"TBR={tbr_value:.2f}", color=colors[idx], linewidth=1.0)

                    plt.xlabel("Time (s)")
                    plt.ylabel("sds.I[1]")
                    plt.title(
                        f"sds.I[1] vs Time for {param_name}={param_val:.2f} (Group {i//curves_per_plot + 1})")
                    plt.legend(bbox_to_anchor=(1.05, 1),
                               loc='upper left', fontsize=6)
                    plt.grid(True)
                    plt.tight_layout()

                    safe_param_name = param_name.replace('.', '_')
                    png_path = os.path.join(temp_dir, get_unique_filename(
                        temp_dir, f"sds_I1_sweep_{safe_param_name}_{param_val:.2f}_group_{i//curves_per_plot}.png"))
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_paths.append(png_path)
                    logger.info(f"Plot saved: {png_path}")

            return plot_paths

        except Exception as e:
            logger.error(f"Plotting failed: {str(e)}")
            raise


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), sys.exit(0)])
    root.mainloop()
