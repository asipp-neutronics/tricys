import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_startup_inventory(csv_path: str, param_A_name: str, param_B_name: str) -> str:
    """
    Plot the startup tritium inventory based on simulation results.

    Args:
        csv_path (str): Path to the combined CSV file.
        param_A_name (str): Name of Parameter A.
        param_B_name (str): Name of Parameter B.

    Returns:
        str: Path to the saved plot image.
    """
    # Set plotting style
    sns.set(style='whitegrid')

    # Read only necessary columns
    param_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    param_columns = [col for col in param_columns if col != 'time']
    df = pd.read_csv(csv_path, usecols=param_columns)

    # Parse column names and calculate startup tritium inventory
    param_A_values = {}
    for col in param_columns:
        parts = col.split('_')
        param_A_part = parts[0]
        param_B_part = parts[1]
        param_A_val = float(param_A_part.split('=')[1])
        param_B_val = float(param_B_part.split('=')[1])
        
        col_data = df[col].to_numpy()
        initial_value = col_data[0]
        min_value = np.min(col_data)
        startup_inventory = initial_value - min_value
        
        if param_A_val not in param_A_values:
            param_A_values[param_A_val] = []
        param_A_values[param_A_val].append((param_B_val, startup_inventory))

    # Plot line graph
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette('tab10', len(param_A_values))

    for i, (param_A_val, data) in enumerate(param_A_values.items()):
        data_sorted = sorted(data, key=lambda x: x[0])
        param_B_vals = [x[0] for x in data_sorted]
        startup_inventories = [x[1] for x in data_sorted]
        
        plt.plot(param_B_vals, startup_inventories, marker='o', label=f"{param_A_name}={param_A_val:.3f}", color=colors[i], linewidth=1.5)

    plt.xlabel(param_B_name)
    plt.ylabel('Start-up Tritium Inventory')
    plt.title(f'Start-up Tritium Inventory vs {param_B_name} for Different {param_A_name}')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.margins(x=0.05, y=0.1)

    # Save plot
    save_dir = 'D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/data'
    png_path = os.path.join(save_dir, f'startup_tritium_inventory_{param_A_name}_vs_{param_B_name}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    return png_path