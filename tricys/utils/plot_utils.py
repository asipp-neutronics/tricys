"""Utility functions for plotting simulation results.

This module provides functions to generate plots from the simulation output
CSV files, such as visualizing startup tritium inventory or time-series data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_startup_inventory(
    csv_path: str, param_A_name: str, param_B_name: str, save_dir: str
) -> str:
    """Plots the startup tritium inventory based on simulation results.

    This function reads a combined CSV file from a parameter sweep, calculates
    the startup tritium inventory for each run, and plots it as a function of
    two varied parameters.

    Args:
        csv_path (str): The path to the combined simulation results CSV file.
        param_A_name (str): The name of the first parameter (used for grouping lines).
        param_B_name (str): The name of the second parameter (used as the x-axis).
        save_dir (str): The directory where the output plot image will be saved.

    Returns:
        str: The path to the saved plot image file.
    """
    # Set plotting style
    sns.set(style="whitegrid")

    # Read only necessary columns
    param_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    param_columns = [col for col in param_columns if col != "time"]
    df = pd.read_csv(csv_path, usecols=param_columns)

    # Parse column names and calculate startup tritium inventory
    param_A_values = {}
    for col in param_columns:
        parts = col.split("_")
        param_A_part = parts[0]
        param_B_part = parts[1]
        param_A_val = float(param_A_part.split("=")[1])
        param_B_val = float(param_B_part.split("=")[1])

        col_data = df[col].to_numpy()
        initial_value = col_data[0]
        min_value = np.min(col_data)
        startup_inventory = initial_value - min_value

        if param_A_val not in param_A_values:
            param_A_values[param_A_val] = []
        param_A_values[param_A_val].append((param_B_val, startup_inventory))

    # Plot line graph
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("tab10", len(param_A_values))

    for i, (param_A_val, data) in enumerate(param_A_values.items()):
        data_sorted = sorted(data, key=lambda x: x[0])
        param_B_vals = [x[0] for x in data_sorted]
        startup_inventories = [x[1] for x in data_sorted]

        plt.plot(
            param_B_vals,
            startup_inventories,
            marker="o",
            label=f"{param_A_name}={param_A_val:.3f}",
            color=colors[i],
            linewidth=1.5,
        )

    plt.xlabel(param_B_name)
    plt.ylabel("Start-up Tritium Inventory")
    plt.title(
        f"Start-up Tritium Inventory vs {param_B_name} for Different {param_A_name}"
    )
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)
    plt.margins(x=0.05, y=0.1)

    # Save plot
    png_path = os.path.join(
        save_dir, f"startup_tritium_inventory_{param_A_name}_vs_{param_B_name}.png"
    )
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    return png_path


def plot_results(csv_path, param_name, param_values, stop_time, temp_dir):
    """Plots time-series results from a simulation sweep.

    This function reads a CSV file, automatically adjusts the time axis to focus
    on the relevant startup period, and generates a series of plots for different
    parameter groups.

    Args:
        csv_path (str): Path to the simulation results CSV file.
        param_name (str): The name of the primary parameter being swept.
        param_values (list): The list of values used for the primary parameter.
        stop_time (float): The simulation stop time, used as an upper bound for plotting.
        temp_dir (str): The directory where the output plot images will be saved.

    Returns:
        list: A list of paths to the generated plot images.

    Raises:
        ValueError: If the CSV contains insufficient data for plotting.
    """
    try:
        df = pd.read_csv(csv_path)
        time = df["time"].to_numpy()

        if len(time) < 2:
            raise ValueError("CSV contains insufficient time points")

        max_time = stop_time
        for column in df.columns[1:]:
            data = df[column].to_numpy()
            if len(data) < 3:
                continue
            diffs = np.diff(data)
            for i in range(1, len(diffs)):
                if i > len(diffs) // 2 and diffs[i] > 0:
                    rise_time = time[i]
                    max_time = min(max_time, rise_time * 1.5)
                    break

        time_mask = time <= max_time
        if not any(time_mask):
            time_mask = np.ones_like(time, dtype=bool)
        time = time[time_mask]
        df = df.iloc[time_mask]

        plot_paths = []
        curves_per_plot = 5
        sns.set(style="whitegrid")
        for param_val in param_values:
            param_columns = [
                col for col in df.columns[1:] if f"{param_name}={param_val:.3f}" in col
            ]
            for i in range(0, len(param_columns), curves_per_plot):
                plt.figure(figsize=(10, 6))
                colors = sns.color_palette(
                    "tab20", min(curves_per_plot, len(param_columns) - i)
                )
                for idx, column in enumerate(param_columns[i : i + curves_per_plot]):
                    tbr_value = float(column.split("blanket.TBR=")[1])
                    plt.plot(
                        time,
                        df[column],
                        label=f"TBR={tbr_value:.2f}",
                        color=colors[idx],
                        linewidth=1.0,
                    )

                plt.xlabel("Time (s)")
                plt.ylabel("sds.I[1]")
                plt.title(
                    f"sds.I[1] vs Time for {param_name}={param_val:.2f} (Group {i//curves_per_plot + 1})"
                )
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)
                plt.grid(True)
                plt.tight_layout()

                safe_param_name = param_name.replace(".", "_")
                png_path = os.path.join(
                    temp_dir,
                    f"sds_I1_sweep_{safe_param_name}_{param_val:.2f}_group_{i//curves_per_plot}.png",
                )
                plt.savefig(png_path, dpi=300, bbox_inches="tight")
                plt.close()
                plot_paths.append(png_path)

        return plot_paths

    except Exception:
        raise
