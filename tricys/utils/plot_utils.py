"""Utility functions for plotting simulation results.

This module provides functions to generate plots from the simulation output
CSV files, such as visualizing startup tritium inventory or time-series data.
"""

import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_analysis_plots(
    summary_df: pd.DataFrame, analysis_case: dict, save_dir: str
) -> list:
    """
    Generates and saves plots based on the sensitivity analysis summary.

    Args:
        summary_df: DataFrame containing the summarized analysis results.
        analysis_case: Configuration for the analysis cases, which can specify
                        a 'plot_type' of 'line' or 'bar', and optionally
                        'combine_plots' to merge all plots into subplots.
        save_dir: Directory to save the plot images.

    Returns:
        A list of paths to the saved plot images.
    """
    if summary_df.empty:
        return []

    analysis_cases = [analysis_case]
    # Set enhanced plotting theme and style
    sns.set_theme(style="whitegrid", palette="husl")
    plt.style.use("seaborn-v0_8-darkgrid")

    # Custom color palettes for different plot types
    line_colors = sns.color_palette("viridis", 10)
    bar_colors = sns.color_palette("plasma", 10)

    # Check if any case requests combined plots
    combine_plots = any(case.get("combine_plots", False) for case in analysis_cases)

    # Collect all valid plot configurations
    valid_plots = []
    for case in analysis_cases:
        case_name = case["name"]
        x_var = case["independent_variable"]
        y_vars = case["dependent_variables"]
        plot_type = case.get("plot_type", "bar")

        if x_var not in summary_df.columns:
            print(
                f"Warning: Independent variable '{x_var}' not found in summary data for case '{case_name}'. Skipping."
            )
            continue

        for y_var in y_vars:
            if y_var not in summary_df.columns:
                print(
                    f"Warning: Dependent variable '{y_var}' not found in summary data for case '{case_name}'. Skipping."
                )
                continue

            valid_plots.append(
                {
                    "case_name": case_name,
                    "x_var": x_var,
                    "y_var": y_var,
                    "plot_type": plot_type,
                }
            )

    if not valid_plots:
        return []

    if combine_plots:
        # Generate combined subplot figure
        return _generate_combined_plots(
            summary_df, valid_plots, save_dir, line_colors, bar_colors
        )
    else:
        # Generate individual plots (original behavior)
        return _generate_individual_plots(
            summary_df, valid_plots, save_dir, line_colors, bar_colors
        )


def _generate_combined_plots(
    summary_df: pd.DataFrame,
    valid_plots: list,
    save_dir: str,
    line_colors: list,
    bar_colors: list,
) -> list:
    """
    Generate a single combined figure with multiple subplots.
    """
    n_plots = len(valid_plots)
    if n_plots == 0:
        return []

    # Calculate optimal subplot layout
    if n_plots == 1:
        rows, cols = 1, 1
    elif n_plots == 2:
        rows, cols = 1, 2
    elif n_plots <= 4:
        rows, cols = 2, 2
    elif n_plots <= 6:
        rows, cols = 2, 3
    elif n_plots <= 9:
        rows, cols = 3, 3
    else:
        rows = int(np.ceil(np.sqrt(n_plots)))
        cols = int(np.ceil(n_plots / rows))

    # Create the combined figure
    fig_width = max(16, cols * 5)
    fig_height = max(12, rows * 4)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Ensure axes is always a 2D array for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    # Set overall figure properties
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Sensitivity Analysis - Combined Results",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    # Generate each subplot
    for idx, plot_config in enumerate(valid_plots):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        _create_subplot(summary_df, plot_config, ax, line_colors, bar_colors)

    # Hide unused subplots
    for idx in range(n_plots, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)

    # Save the combined figure
    combined_filename = "combined_analysis_plots.png"
    save_path = os.path.join(save_dir, combined_filename)
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)

    print(f"Generated combined analysis plot: {save_path}")
    return [save_path]


def _generate_individual_plots(
    summary_df: pd.DataFrame,
    valid_plots: list,
    save_dir: str,
    line_colors: list,
    bar_colors: list,
) -> list:
    """
    Generate individual plot files (original behavior).
    """
    plot_paths = []

    for plot_config in valid_plots:
        # Create individual figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("white")

        _create_subplot(summary_df, plot_config, ax, line_colors, bar_colors)

        # Add case name as subtitle if available
        case_name = plot_config["case_name"]
        if case_name:
            fig.suptitle(
                f"Analysis Case: {case_name}", fontsize=12, style="italic", alpha=0.7
            )

        # Adjust layout
        plt.tight_layout(pad=2.0)

        # Save individual plot
        x_var = plot_config["x_var"]
        y_var = plot_config["y_var"]
        plot_type = plot_config["plot_type"]
        plot_filename = f"{plot_type}_{y_var}_vs_{x_var}.png"
        save_path = os.path.join(save_dir, plot_filename)

        plt.savefig(
            save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        plt.close(fig)
        plot_paths.append(save_path)
        print(f"Generated enhanced analysis plot: {save_path}")

    return plot_paths


def _create_subplot(
    summary_df: pd.DataFrame, plot_config: dict, ax, line_colors: list, bar_colors: list
) -> None:
    """
    Create a single subplot based on the plot configuration.
    """
    x_var = plot_config["x_var"]
    y_var = plot_config["y_var"]
    plot_type = plot_config["plot_type"]

    # Set background color
    ax.set_facecolor("#f8f9fa")

    if plot_type == "line":
        # Enhanced line plot
        sns.lineplot(
            data=summary_df,
            x=x_var,
            y=y_var,
            marker="o",
            markersize=6,
            linewidth=2.5,
            color=line_colors[0],
            alpha=0.8,
            ax=ax,
        )

        # Add data point annotations (smaller for subplots)
        for i, row in summary_df.iterrows():
            ax.annotate(
                f"{row[y_var]:.2f}",
                (row[x_var], row[y_var]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                alpha=0.7,
            )

        title = f"Line: {y_var} vs. {x_var}"

    elif plot_type == "bar":
        # Enhanced bar plot
        bars = sns.barplot(
            data=summary_df,
            x=x_var,
            y=y_var,
            hue=x_var,
            palette=bar_colors[: len(summary_df)],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
            legend=False,
            ax=ax,
        )

        # Add value labels on bars (smaller for subplots)
        for i, bar in enumerate(bars.patches):
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=8,
                )

        title = f"Bar: {y_var} vs. {x_var}"
        ax.tick_params(axis="x", rotation=45, labelsize=9)

    else:
        title = f"Unknown: {y_var} vs. {x_var}"

    # Set title and labels
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel(x_var, fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel(y_var, fontsize=11, fontweight="bold", labelpad=8)

    # Improve grid appearance
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(1.2)

    # Improve tick appearance
    ax.tick_params(
        axis="both", which="major", labelsize=9, colors="#333333", length=4, width=0.8
    )


def plot_sweep_time_series(
    csv_path: str,
    save_dir: str,
    y_var_name: Union[str, List[str]],
    independent_var_name: str,
    independent_var_alias: str = None,
) -> str:
    """
    Filter columns containing specified variables from the CSV file of scan results and plot all time series on a single graph.

    This function is used for single-parameter scans when variables are recorded over time.
    Plot columns whose names contain any of the y_var_name(s).

    Args:
        csv_path (str): Path to the scan result CSV file.
        save_dir (str): Directory to save the image.
        y_var_name (Union[str, List[str]]): Name(s) of the Y-axis variable(s) (e.g., "sds.I[1]" or ["sds.I[1]", "blanket.TBR"]).
        independent_var_name (str): Full name of the scan parameter
                                    (e.g., "tep_fep.to_SDS_Fraction[1]").
        independent_var_al
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find results file at {csv_path}")
        return None

    time = df["time"]
    plot_alias = (
        independent_var_alias if independent_var_alias else independent_var_name
    )

    # Handle both string and list inputs for y_var_name
    if isinstance(y_var_name, str):
        y_var_names = [y_var_name]
    else:
        y_var_names = y_var_name

    # Filter out columns containing any of the y_var_name(s)
    y_var_columns = []
    for y_var in y_var_names:
        y_var_columns.extend(
            [col for col in df.columns if col != "time" and y_var in col]
        )

    # Remove duplicates while preserving order
    y_var_columns = list(dict.fromkeys(y_var_columns))

    if not y_var_columns:
        print(
            f"Warning: No columns found containing any of {y_var_names} in {csv_path}"
        )
        return None

    print(
        f"Found {len(y_var_columns)} columns containing {y_var_names}: {y_var_columns}"
    )

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("viridis", len(y_var_columns))

    for i, column in enumerate(y_var_columns):
        plt.plot(time, df[column], label=column, color=colors[i], linewidth=1.5)

    plt.xlabel("Time (hrs)")
    plt.ylabel(", ".join(y_var_names))
    plt.title(f"Time Evolution of {', '.join(y_var_names)} vs. {plot_alias}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    safe_y_vars = "_".join(
        [var.replace(".", "_").replace("[", "").replace("]", "") for var in y_var_names]
    )
    safe_param = plot_alias.replace(".", "_").replace("[", "").replace("]", "")
    png_path = os.path.join(save_dir, f"sweep_{safe_y_vars}_vs_{safe_param}.png")

    try:
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"Successfully generated sweep plot: {png_path}")
        return png_path
    except Exception as e:
        print(f"Error saving plot: {e}")
        plt.close()
        return None
