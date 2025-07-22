import os
import pandas as pd
from .file_utils import get_unique_filename

def combine_simulation_results(param_A_values: dict, param_B_sweep: dict, temp_dir: str, output_csv_files: list) -> str:
    """
    Combine simulation results from multiple CSV files into a single CSV file.

    Args:
        param_A_values (dict): Dictionary with parameter A name and list of values.
        param_B_sweep (dict): Dictionary with parameter B name and sweep values.
        temp_dir (str): Directory containing temporary CSV files.
        output_csv_files (list): List of paths to temporary CSV files.

    Returns:
        str: Path to the combined CSV file.
    """
    # Get parameter A and B names and values
    param_A_name = list(param_A_values.keys())[0]
    param_A_vals = param_A_values[param_A_name]
    param_B_name = list(param_B_sweep.keys())[0]
    param_B_vals = param_B_sweep[param_B_name]

    # Combine results
    combined_df = None
    counter = 0
    for param_A_val in param_A_vals:
        for param_B_val in param_B_vals:
            csv_file = output_csv_files[counter]
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if combined_df is None:
                    combined_df = df[['time']].copy()
                column_name = f"{param_A_name}={param_A_val:.3f}_{param_B_name}={param_B_val:.3f}"
                combined_df[column_name] = df['sds.I[1]']
            else:
                print(f"Warning: CSV file {csv_file} not found.")
            counter += 1

    # Save combined results
    base_combined_filename = f"{param_A_name}_{param_B_name}.csv"
    combined_csv_path = get_unique_filename(temp_dir, base_combined_filename)
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined CSV file saved to: {combined_csv_path}")

    # Clean up temporary files
    for csv_file in output_csv_files:
        os.remove(csv_file)

    return combined_csv_path