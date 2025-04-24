import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# !! Important: Update these paths to match your directory structure !!
INPUT_DIR = "/Users/suryatejachalla/Research/reentry-amr-modeling/data/final-tle-data" # Directory with original CSV files
OUTPUT_DIR = "/Users/suryatejachalla/Research/reentry-amr-modeling/data/raw-fitting" # Directory with processed CSV files
NUM_POINTS_PROCESSED = 90 # Number of points used in the processing script

# Columns to plot
COLUMNS_TO_PLOT = [
    'mean_motion', 'eccentricity', 'inclination', 'bstar',
    'mean_motion_deriv', 'semi_major_axis', 'orbital_period', 'perigee_alt'
]
# --- End Configuration ---

def plot_comparison(norad_id, input_dir, output_dir):
    """
    Reads original and processed data for a given NORAD ID and plots
    a comparison for specified orbital parameters.

    Args:
        norad_id (str): The NORAD Catalog ID of the object.
        input_dir (str): Path to the directory containing original data files.
        output_dir (str): Path to the directory containing processed data files.
    """
    original_file = os.path.join(input_dir, f"{norad_id}.csv")
    processed_file = os.path.join(output_dir, f"{norad_id}.csv")

    # --- File Existence Check ---
    if not os.path.exists(original_file):
        print(f"Error: Original data file not found for NORAD ID {norad_id} at {original_file}")
        return
    if not os.path.exists(processed_file):
        print(f"Error: Processed data file not found for NORAD ID {norad_id} at {processed_file}")
        return

    print(f"Loading data for NORAD ID: {norad_id}")
    print(f"Original file: {original_file}")
    print(f"Processed file: {processed_file}")

    try:
        # --- Load Data ---
        df_original = pd.read_csv(original_file)
        df_processed = pd.read_csv(processed_file)

        # --- Data Validation ---
        # Check if processed data has the expected number of points
        if len(df_processed) != NUM_POINTS_PROCESSED:
            print(f"Warning: Processed data for {norad_id} has {len(df_processed)} points, expected {NUM_POINTS_PROCESSED}.")

        # Check if required columns exist in original data
        missing_original_cols = [col for col in COLUMNS_TO_PLOT if col not in df_original.columns]
        if missing_original_cols:
            print(f"Error: Missing required columns in original file {original_file}: {missing_original_cols}")
            return

        # Check if required columns exist in processed data (should be guaranteed by the processing script)
        missing_processed_cols = [col for col in COLUMNS_TO_PLOT if col not in df_processed.columns]
        if missing_processed_cols:
            print(f"Error: Missing required columns in processed file {processed_file}: {missing_processed_cols}")
            # This indicates an issue with the processing script or the file itself
            return

        # --- Plotting Setup ---
        num_plots = len(COLUMNS_TO_PLOT)
        # Adjust grid size as needed, aiming for roughly square layout
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols # Calculate required rows

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
        axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

        # --- Generate Plots ---
        for i, column in enumerate(COLUMNS_TO_PLOT):
            ax = axes[i]

            # Original data indices (0 to N-1)
            original_indices = np.arange(len(df_original))
            # Processed data indices (0 to NUM_POINTS_PROCESSED-1)
            processed_indices = np.arange(len(df_processed))

            # Plot original data (using circles for distinct points)
            ax.plot(original_indices, df_original[column], 'o-', markersize=3, alpha=0.7, label='Original Data', color='skyblue')

            # Plot processed data (using a solid line)
            # We need to scale the x-axis of the processed data to roughly match the original index range for better visual comparison
            # Create scaled indices for processed data that span the same range as original indices
            scaled_processed_indices = np.linspace(original_indices.min(), original_indices.max(), len(df_processed))
            ax.plot(scaled_processed_indices, df_processed[column], '.-', linewidth=2, label='Processed Data', color='red')

            # Set plot titles and labels
            ax.set_title(column.replace('_', ' ').title())
            ax.set_xlabel("Data Point Index (Original Timeline)")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        # --- Final Touches ---
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"TLE Parameter Comparison for NORAD ID: {norad_id}", fontsize=16, y=1.02) # Add overall title
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
        plt.show() # Display the plots

    except pd.errors.EmptyDataError:
        print(f"Error: One of the files for NORAD ID {norad_id} is empty.")
    except Exception as e:
        print(f"An unexpected error occurred while plotting for NORAD ID {norad_id}: {e}")

if __name__ == "__main__":
    # --- User Input ---
    while True:
        try:
            norad_id_input = input("Enter the NORAD ID to plot (or 'quit' to exit): ").strip()
            if norad_id_input.lower() == 'quit':
                break
            # Basic validation: check if it's a number (NORAD IDs are typically integers)
            int(norad_id_input) # Raises ValueError if not an integer
            plot_comparison(norad_id_input, INPUT_DIR, OUTPUT_DIR)
        except ValueError:
            print("Invalid input. Please enter a numeric NORAD ID.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
