"""Utility functions for plotting simulation results.

This module provides functions to generate plots from the simulation output
CSV files, such as visualizing startup tritium inventory or time-series data.
"""

import os
import re
from typing import Any, Dict, List, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _format_label(label: str) -> str:
    """Formats a label for display, replacing underscores/dots with spaces and capitalizing each word."""
    if not isinstance(label, str):
        return label
    label = label.replace("_", " ")
    # Replace dots that are not part of a number with a space
    label = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", label)
    return label


def _find_unit_config(var_name: str, unit_map: dict) -> dict | None:
    """
    Finds the unit configuration for a variable name from the unit_map.
    1. Checks for an exact match.
    2. Checks if the last part of a dot-separated name matches.
    3. Checks for a simple substring containment as a fallback, matching longest keys first.
    """
    if not unit_map or not var_name:
        return None

    # 1. Exact match
    if var_name in unit_map:
        return unit_map[var_name]

    # 2. Last component match (e.g., 'pulse.power' matches 'power')
    components = var_name.split(".")
    if len(components) > 1:
        last_component = components[-1]
        if last_component in unit_map:
            return unit_map[last_component]

    # 3. Substring match (longest key first to be safer)
    for key in sorted(unit_map.keys(), key=len, reverse=True):
        if key in var_name:
            return unit_map[key]

    return None


def _format_number_for_display(value):
    """
    Format a number for display with appropriate decimal places:
    - If the value is a whole number, show no decimal places
    - If the absolute value is >= 100, show 1 decimal place
    - If the absolute value is >= 10, show 2 decimal places
    - If the absolute value is >= 1, show 3 decimal places
    - If the absolute value is >= 0.1, show 4 decimal places
    - If the absolute value is < 0.1, show 5 decimal places
    - If the value is very small (< 0.0001), use scientific notation
    """
    if np.isnan(value) or np.isinf(value):
        return str(value)

    abs_value = abs(value)

    if abs_value == 0:
        return "0"
    elif abs_value >= 100:
        return f"{value:.1f}"
    elif abs_value >= 10:
        return f"{value:.2f}"
    elif abs_value >= 1:
        return f"{value:.3f}"
    else:
        # For very small numbers, use scientific notation
        return f"{value:.2e}"


def _generate_multi_required_plot(
    summary_df: pd.DataFrame,
    case: dict,
    required_cols: list,
    base_metric_name: str,
    save_dir: str,
    unit_map: dict = None,
) -> list:
    """
    Generates a figure with subplots for 'Required_***' metrics,
    where each subplot corresponds to a unique combination of simulation parameters (hue_vars).
    """
    x_var = case["independent_variable"]
    case_sim_params = case.get("default_simulation_values", {})
    hue_vars = sorted(list(case_sim_params.keys()))

    x_var_label = _format_label(x_var)
    base_metric_name_label = _format_label(base_metric_name)

    # Apply units to labels if unit_map is provided
    if unit_map:
        x_config = _find_unit_config(x_var, unit_map)
        if x_config and x_config.get("unit"):
            x_var_label = f'{x_var_label} ({x_config["unit"]})'

        base_config = _find_unit_config(base_metric_name, unit_map)
        if base_config and base_config.get("unit"):
            base_metric_name_label = f'{base_metric_name_label} ({base_config["unit"]})'

    if not hue_vars:
        # If no hue_vars, create a single plot with all required_cols.
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        axes = [ax]
        plot_groups = [("All Data", summary_df)]
    else:
        # Group data by unique combinations of hue_vars
        plot_groups = list(summary_df.groupby(hue_vars))
        n_plots = len(plot_groups)
        if n_plots == 0:
            return []

        # Determine layout
        if n_plots <= 2:
            rows, cols = 1, n_plots
        elif n_plots <= 4:
            rows, cols = 2, 2
        else:
            rows = int(np.ceil(np.sqrt(n_plots)))
            cols = int(np.ceil(n_plots / rows))

        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 7, rows * 5), squeeze=False
        )
        axes = axes.flatten()

    line_colors = sns.color_palette("viridis", len(required_cols))

    # Create a subplot for each group
    for idx, (group_name, group_df) in enumerate(plot_groups):
        if idx >= len(axes):
            break
        ax = axes[idx]

        plot_df = group_df.copy()

        # Apply data conversion for x-axis
        if unit_map:
            x_config = _find_unit_config(x_var, unit_map)
            if x_config:
                factor = x_config.get("conversion_factor")
                if factor and pd.api.types.is_numeric_dtype(plot_df[x_var]):
                    plot_df[x_var] = plot_df[x_var] / float(factor)

        # Plot each required_col as a line in the subplot
        for i, req_col in enumerate(required_cols):
            # Create a clean label for the legend
            legend_label = req_col.replace(base_metric_name, "").strip("()")
            if not legend_label:
                legend_label = req_col

            # Apply data conversion for y-axis
            if unit_map:
                base_config = _find_unit_config(base_metric_name, unit_map)
                if base_config:
                    factor = base_config.get("conversion_factor")
                    if factor and pd.api.types.is_numeric_dtype(plot_df[req_col]):
                        plot_df[req_col] = plot_df[req_col] / float(factor)

            sns.lineplot(
                data=plot_df,
                x=x_var,
                y=req_col,
                ax=ax,
                color=line_colors[i],
                label=_format_label(legend_label),
                marker="o",
                markersize=6,
                linewidth=2.0,
            )

        # Subplot titles and labels
        if hue_vars:
            if isinstance(group_name, tuple):
                title = ", ".join(
                    f"{_format_label(k)}={v}" for k, v in zip(hue_vars, group_name)
                )
            else:
                title = f"{_format_label(hue_vars[0])}={group_name}"
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title(
                f"Dependence of {base_metric_name_label} on {x_var_label}", fontsize=12
            )

        ax.set_xlabel(x_var_label, fontsize=12)
        ax.set_ylabel(base_metric_name_label, fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        # Dynamically determine the legend title from associated metric columns
        legend_title = "Constraint"  # A more descriptive default
        search_pattern = f"_for_{base_metric_name}"
        for col in summary_df.columns:
            if search_pattern in col:
                metric_name = col.split(search_pattern)[0]
                legend_title = _format_label(metric_name)
                break

        ax.legend(title=legend_title)

    # Hide unused axes
    for i in range(len(plot_groups), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(
        save_dir, f"multi_{base_metric_name}_analysis_by_param.svg"
    )
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"Generated multi-metric analysis plot by parameter: {save_path}")
    return [save_path]


def generate_analysis_plots(
    summary_df: pd.DataFrame, analysis_case: dict, save_dir: str, unit_map: dict = None
) -> list:
    """
    Generates and saves plots based on the sensitivity analysis summary.
    This function first generates dedicated plots for all 'Required_***' metrics,
    then handles plotting for all other standard metrics.

    Args:
        summary_df: DataFrame containing the summarized analysis results.
        analysis_case: Configuration for the analysis cases.
        save_dir: Directory to save the plot images.
        unit_map: Optional dictionary for unit conversion and labeling.

    Returns:
        A list of paths to the saved plot images.
    """
    if summary_df.empty:
        return []

    analysis_cases = [analysis_case]  # Keep as a list for consistency
    sns.set_theme(style="whitegrid")
    line_colors = sns.color_palette("viridis", 10)

    plot_paths = []

    # If unit_map is not provided, initialize as empty dict
    if unit_map is None:
        unit_map = {}

    # --- 1. Handle ALL 'Required_***' plots first and separately ---
    all_required_vars_from_config = {
        var
        for case in analysis_cases
        for var in case.get("dependent_variables", [])
        if var.startswith("Required_")
    }

    for req_var in all_required_vars_from_config:
        # Find all actual columns in the dataframe for this base name
        matching_cols = sorted(
            [
                c
                for c in summary_df.columns
                if c == req_var or c.startswith(req_var + "(")
            ]
        )

        if not matching_cols:
            continue

        case_for_plot = analysis_cases[0]

        if len(matching_cols) > 1:
            # Multi-value case -> generate a multi-subplot figure
            multi_plot_paths = _generate_multi_required_plot(
                summary_df,
                case_for_plot,
                matching_cols,
                req_var,
                save_dir,
                unit_map=unit_map,
            )
            plot_paths.extend(multi_plot_paths)
        elif len(matching_cols) == 1:
            # Single-value case -> generate a single, individual plot
            plot_config = {
                "case_name": case_for_plot["name"],
                "x_var": case_for_plot["independent_variable"],
                "y_var": matching_cols[0],
                "plot_type": "line",
                "hue_vars": sorted(
                    list(case_for_plot.get("default_simulation_values", {}).keys())
                ),
            }
            single_plot_path = _generate_individual_plots(
                summary_df, [plot_config], save_dir, line_colors, unit_map=unit_map
            )
            plot_paths.extend(single_plot_path)

    # --- 2. Handle all other (non-Required) plots ---

    # Collect plot configurations for remaining standard variables
    valid_plots_for_combine = []
    for case in analysis_cases:
        case_name = case["name"]
        x_var = case["independent_variable"]

        # Filter out the Required_*** vars that we just plotted
        y_vars = [
            v
            for v in case.get("dependent_variables", [])
            if not v.startswith("Required_")
        ]

        if x_var not in summary_df.columns:
            print(
                f"Warning: Independent variable '{x_var}' not found in summary data for case '{case_name}'. Skipping."
            )
            continue

        case_sim_params = case.get("default_simulation_values", {})
        hue_vars = sorted(list(case_sim_params.keys()))

        for y_var in y_vars:
            if y_var not in summary_df.columns:
                print(
                    f"Warning: Dependent variable '{y_var}' not found in summary data for case '{case_name}'. Skipping."
                )
                continue

            # We assume standard metrics have a 1-to-1 name match in the dataframe
            valid_plots_for_combine.append(
                {
                    "case_name": case_name,
                    "x_var": x_var,
                    "y_var": y_var,
                    "plot_type": "line",
                    "hue_vars": hue_vars,
                }
            )

    if valid_plots_for_combine:
        combine_plots = any(case.get("combine_plots", False) for case in analysis_cases)
        generated_paths = []
        if combine_plots:
            generated_paths = _generate_combined_plots(
                summary_df,
                valid_plots_for_combine,
                save_dir,
                line_colors,
                unit_map=unit_map,
            )
        else:
            # If not combining, plot them individually anyway
            generated_paths = _generate_individual_plots(
                summary_df,
                valid_plots_for_combine,
                save_dir,
                line_colors,
                unit_map=unit_map,
            )
        plot_paths.extend(generated_paths)

    return plot_paths


def _generate_combined_plots(
    summary_df: pd.DataFrame,
    valid_plots: list,
    save_dir: str,
    line_colors: list,
    unit_map: dict = None,
) -> list:
    """
    Generate a single combined figure with multiple subplots.
    Layout rules:
    - Max 2 plots per row.
    - For odd numbers of plots > 1, the last plot is centered and spans the full width.
    """
    n_plots = len(valid_plots)
    if n_plots == 0:
        return []

    # If there is only one plot, delegate to the individual plot generator
    # which is optimized for a single, larger plot.
    if n_plots == 1:
        return _generate_individual_plots(
            summary_df, valid_plots, save_dir, line_colors, unit_map=unit_map
        )

    axes_list = []
    is_odd = n_plots % 2 == 1

    # Determine layout based on number of plots
    if is_odd and n_plots > 1:
        # Custom layout for odd numbers (3, 5, 7...)
        rows = int(np.ceil(n_plots / 2))
        cols = 2
        fig = plt.figure(figsize=(cols * 8, rows * 5))
        gs = fig.add_gridspec(
            rows, cols, height_ratios=[1] * rows, hspace=0.3, wspace=0.2
        )

        # Add all but the last plot
        for i in range(n_plots - 1):
            ax = fig.add_subplot(gs[i // cols, i % cols])
            axes_list.append(ax)

        # Add the last plot, spanning the full width of the last row
        ax = fig.add_subplot(gs[rows - 1, :])
        axes_list.append(ax)

    else:
        # General layout for even numbers and a single plot
        cols = 2
        rows = int(np.ceil(n_plots / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 8, rows * 5), squeeze=False
        )
        axes_list = axes.flatten()

    # Set overall figure properties
    fig.patch.set_facecolor("white")
    # x_var_label = _format_label(valid_plots[0]["x_var"])

    # Generate each subplot
    for idx, plot_config in enumerate(valid_plots):
        ax = axes_list[idx]
        _create_subplot(
            summary_df, plot_config, ax, line_colors, idx, unit_map=unit_map
        )

    # Hide any unused axes (only relevant for even-number layouts)
    for i in range(n_plots, len(axes_list)):
        axes_list[i].set_visible(False)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)

    # Save the combined figure
    combined_filename = "combined_analysis_plots.svg"
    save_path = os.path.join(save_dir, combined_filename)
    plt.savefig(
        save_path,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)

    print(f"Generated combined analysis plot: {save_path}")
    return [save_path]


def _generate_individual_plots(
    summary_df: pd.DataFrame,
    valid_plots: list,
    save_dir: str,
    line_colors: list,
    unit_map: dict = None,
) -> list:
    """
    Generate individual plot files (original behavior).
    """
    plot_paths = []

    for idx, plot_config in enumerate(valid_plots):
        # Create individual figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("white")

        _create_subplot(
            summary_df, plot_config, ax, line_colors, idx, unit_map=unit_map
        )

        # Adjust layout
        plt.tight_layout(pad=2.0)

        # Save individual plot
        x_var = plot_config["x_var"]
        y_var = plot_config["y_var"]
        plot_type = plot_config["plot_type"]
        plot_filename = f"{plot_type}_{y_var}_vs_{x_var}.svg"
        save_path = os.path.join(save_dir, plot_filename)

        plt.savefig(
            save_path,
            format="svg",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        plot_paths.append(save_path)
        print(f"Generated enhanced analysis plot: {save_path}")

    return plot_paths


def _create_subplot(
    summary_df: pd.DataFrame,
    plot_config: dict,
    ax,
    line_colors: list,
    plot_index: int,
    unit_map: dict = None,
) -> None:
    """
    Creates and beautifies a single subplot for sensitivity analysis results.

    This function handles plotting one dependent variable against an independent
    variable. If 'hue_vars' are provided in plot_config, it will draw
    multiple curves for different parameter combinations.

    Args:
        summary_df: DataFrame containing the data to plot.
        plot_config: Dictionary with plot details ('x_var', 'y_var', 'hue_vars').
        ax: Matplotlib axes object to draw the plot on.
        line_colors: A list of colors to use for the plot lines.
        plot_index: The index of the plot, used to select a color.
        unit_map: Optional dictionary for unit conversion and labeling.
    """
    x_var = plot_config["x_var"]
    y_var = plot_config["y_var"]
    hue_vars = plot_config.get("hue_vars", [])

    # Format labels for display
    x_var_label = _format_label(x_var)
    y_var_display = _format_label(y_var)

    # Create a local copy for plotting to avoid changing the original DataFrame
    plot_vars = [x_var, y_var] + hue_vars
    plot_data = summary_df[plot_vars].copy().dropna()

    # Initialize labels
    final_y_var_label = y_var_display
    final_x_var_label = x_var_label

    # Apply unit conversions and formatting based on unit_map
    if unit_map:
        # Process Y-axis variable
        y_config = _find_unit_config(y_var, unit_map)
        if y_config:
            unit = y_config.get("unit")
            factor = y_config.get("conversion_factor")
            if factor and pd.api.types.is_numeric_dtype(plot_data[y_var]):
                plot_data[y_var] = plot_data[y_var] / float(factor)
            if unit:
                final_y_var_label = f"{y_var_display} ({unit})"

        # Process X-axis variable
        x_config = _find_unit_config(x_var, unit_map)
        if x_config:
            unit = x_config.get("unit")
            factor = x_config.get("conversion_factor")
            if factor and pd.api.types.is_numeric_dtype(plot_data[x_var]):
                plot_data[x_var] = plot_data[x_var] / float(factor)
            if unit:
                final_x_var_label = f"{x_var_label} ({unit})"
    else:
        # Fallback to old hard-coded logic if no unit_map is provided
        if y_var in ["Doubling_Time", "Self_Sufficiency_Time"]:
            plot_data[y_var] = plot_data[y_var] / 24
            final_y_var_label = f"{y_var_display} (days)"
        elif y_var == "Startup_Inventory":
            plot_data[y_var] = plot_data[y_var] / 1000.0
            final_y_var_label = f"{y_var_display} (kg)"

    # --- Plotting Logic ---
    num_curves = 1  # Default for a single curve
    if hue_vars:
        # Set up hue column and legend title based on number of hue variables
        if len(hue_vars) == 1:
            hue_col = hue_vars[0]
        else:
            # For multiple hue variables, create a string-based column for the hue
            hue_col = ", ".join(hue_vars)
            plot_data[hue_col] = plot_data[hue_vars].apply(
                lambda row: ", ".join(map(str, row)), axis=1
            )

        num_curves = len(plot_data[hue_col].unique())

        # Define distinct markers and line styles for better visual separation
        markers_cycle = ["o", "s", "X", "D", "^", "v", "P", "*"]
        dashes_cycle = [(1, 0), (5, 5), (2, 2), (5, 2, 2, 2), (3, 5, 1, 5)]

        sns.lineplot(
            data=plot_data,
            x=x_var,
            y=y_var,
            hue=hue_col,
            style=hue_col,
            markers=markers_cycle,
            dashes=dashes_cycle,
            palette="viridis",
            linewidth=2.5,
            markersize=8,
            alpha=0.9,
            ax=ax,
        )
    else:
        # Original behavior: single line plot
        color = line_colors[plot_index % len(line_colors)]
        sns.lineplot(
            data=plot_data,
            x=x_var,
            y=y_var,
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=color,
            alpha=0.9,
            ax=ax,
        )

    # Add data point annotations only if the number of curves is manageable
    if num_curves <= 4:
        for i, row in plot_data.iterrows():
            ax.annotate(
                _format_number_for_display(row[y_var]),
                (row[x_var], row[y_var]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                alpha=0.75,
            )

    title = f"Dependence of {final_y_var_label} on {final_x_var_label}"

    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel(final_x_var_label, fontsize=14)
    ax.set_ylabel(final_y_var_label, fontsize=14)

    # Set grid style and legend
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    if hue_vars:
        # Get the legend object and adjust its properties
        legend = ax.get_legend()
        if legend:
            original_title = legend.get_title().get_text()
            legend.set_title(
                _format_label(original_title), prop={"size": 10, "weight": "bold"}
            )
            plt.setp(legend.get_texts(), fontsize=10)


def plot_sweep_time_series(
    csv_path: str,
    save_dir: str,
    y_var_name: Union[str, List[str]],
    independent_var_name: str,
    independent_var_alias: str = None,
    default_params: Dict[str, Any] = None,
) -> str:
    """
    Generates a single figure with two subplots: an overall time-series view and a
    zoomed-in view around the minimum point of the curves. The time axis is in days.
    The overall view hides data points for a curve if they exceed twice its initial value.

    Args:
        csv_path (str): Path to the scan result CSV file.
        save_dir (str): Directory to save the image.
        y_var_name (Union[str, List[str]]): Name(s) of the Y-axis variable(s).
        independent_var_name (str): Full name of the scan parameter.
        independent_var_alias (str): Alias for the scan parameter for cleaner plot titles.
        default_params (Dict[str, Any], optional): A dictionary of default parameters.
            If provided, only curves matching these parameters will be plotted.

    Returns:
        The path to the saved plot image, or None on failure.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find results file at {csv_path}")
        return None

    if "time" not in df.columns:
        print(f"Error: 'time' column not found in {csv_path}")
        return None

    # Convert time from hours to days
    time_days = df["time"] / 24

    # Use alias if provided, otherwise format the original name
    raw_plot_alias = (
        independent_var_alias if independent_var_alias else independent_var_name
    )
    # plot_alias = _format_label(raw_plot_alias)

    if isinstance(y_var_name, str):
        y_var_names = [y_var_name]
    else:
        y_var_names = y_var_name

    y_var_columns = []
    for y_var in y_var_names:
        y_var_columns.extend(
            [col for col in df.columns if col != "time" and y_var in col]
        )
    y_var_columns = list(dict.fromkeys(y_var_columns))

    # If default_params are provided, filter columns to only plot baseline curves
    if default_params:
        filtered_columns = []
        for col in y_var_columns:
            try:
                param_str = col.split("&", 1)[1]
                col_params = dict(p.split("=", 1) for p in param_str.split("&"))

                # Check if all default_params match the parameters in the column name
                is_match = all(
                    col_params.get(key) == str(val)
                    for key, val in default_params.items()
                )

                if is_match:
                    filtered_columns.append(col)
            except IndexError:
                # This column does not have parameters in its name, so it can't be a match
                continue
        y_var_columns = filtered_columns

    if not y_var_columns:
        print(
            f"Warning: No columns found containing any of {y_var_names} in {csv_path} that match the criteria."
        )
        return None

    # Convert y-axis data from grams to kilograms
    for col in y_var_columns:
        df[col] = df[col] / 1000.0

    # Generate clean labels (values only) for the legend
    plot_labels = []
    for col in y_var_columns:
        label = col
        try:
            param_parts = col.split("&")[1:]
            for part in param_parts:
                if part.startswith(independent_var_name + "="):
                    label = part.split("=", 1)[1]  # Extract just the value
                    break
        except IndexError:
            pass  # No parameters in name, use full column name
        plot_labels.append(label)

    print(
        f"Found {len(y_var_columns)} columns to plot containing {y_var_names}: {y_var_columns}"
    )

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("plasma", len(y_var_columns))

    # Create a figure with two subplots (overall and zoom)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 16), sharex=False, gridspec_kw={"height_ratios": [2, 1]}
    )
    y_var_names_formatted = [_format_label(y) for y in y_var_names]

    min_y_global = float("inf")
    min_x_global = float("inf")

    # Define the y-axis label with units
    y_label = f"{', '.join(y_var_names_formatted)} (kg)"

    # --- Subplot 1: Overall View ---
    for i, column in enumerate(y_var_columns):
        y_data = df[column]

        # For the global view, mask data that is more than 2x the initial value
        if not y_data.empty:
            initial_value = y_data.iloc[0]
            threshold = 2 * initial_value
            y_masked = y_data.where(y_data <= threshold)
        else:
            y_masked = y_data

        ax1.plot(
            time_days,
            y_masked,
            label=plot_labels[i],
            color=colors[i],
            linewidth=1.2,
            alpha=0.85,
        )

        # Calculations for zoom window should use the original, unmasked data
        if not y_data.empty:
            min_idx = y_data.idxmin()
            current_min_y = y_data.loc[min_idx]
            if current_min_y < min_y_global:
                min_y_global = current_min_y
                min_x_global = time_days.loc[min_idx]

    ax1.set_ylabel(y_label, fontsize=14)
    ax1.set_title(
        "Overall View (Data exceeding 2x initial value is hidden)", fontsize=12
    )
    ax1.legend(loc="best", title=_format_label(independent_var_name))
    ax1.grid(True)

    # --- Subplot 2: Zoomed-in View (uses original data) ---
    if min_y_global != float("inf") and np.isfinite(min_y_global):
        for i, column in enumerate(y_var_columns):
            # Plot original, unmasked data in the zoom plot
            ax2.plot(
                time_days,
                df[column],
                label=plot_labels[i],
                color=colors[i],
                linewidth=1.8,
                alpha=0.9,
            )

        # Define the zoom window from t=0 to a bit after the minimum
        x1 = 0
        x2 = min_x_global + 2  # Show 2 days past the minimum

        # Filter the DataFrame to the new x-range to find the y-range
        zoom_mask = (time_days >= x1) & (time_days <= x2)
        df_zoom_range = df[zoom_mask]

        # Find y-min and y-max within this specific range
        y_min_in_range = df_zoom_range[y_var_columns].min().min()
        y_max_in_range = df_zoom_range[y_var_columns].max().max()

        # Add padding to the y-axis
        y_padding = (y_max_in_range - y_min_in_range) * 0.05
        y1 = y_min_in_range - y_padding
        y2 = y_max_in_range + y_padding

        ax2.set_xlim(x1, x2)
        ax2.set_ylim(y1, y2)

        ax2.set_xlabel("Time (days)", fontsize=14)
        ax2.set_ylabel(y_label, fontsize=14)
        ax2.set_title("Detailed View (t=0 to Post-Minimum)", fontsize=12)
        ax2.grid(True, linestyle="--")

        # Add a rectangle to the main plot to indicate the new zoom area
        rect = patches.Rectangle(
            (x1, y1),
            (x2 - x1),
            (y2 - y1),
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            linestyle="--",
            alpha=0.7,
        )
        ax1.add_patch(rect)
    else:
        # If no zoom, hide the second subplot
        ax2.set_visible(False)

    ax1.set_xlabel("Time (days)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

    # --- Save Figure ---
    safe_y_vars = "_".join(
        [var.replace(".", "_").replace("[", "").replace("]", "") for var in y_var_names]
    )
    safe_param = raw_plot_alias.replace(".", "_").replace("[", "").replace("]", "")
    svg_path = os.path.join(save_dir, f"sweep_{safe_y_vars}_vs_{safe_param}.svg")

    try:
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"Successfully generated combined sweep plot: {svg_path}")
        return svg_path
    except Exception as e:
        print(f"Error saving plot: {e}")
        return None
    finally:
        plt.close(fig)
