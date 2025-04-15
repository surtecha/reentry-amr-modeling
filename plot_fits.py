import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
TLE_DATA_FOLDER = '/Users/suryatejachalla/Research/Re-entry-Prediction/Code/Orbital-Influence/tle_data_output'
ML_DATA_FILE = '/Users/suryatejachalla/Research/Re-entry-Prediction/Code/Orbital-Influence/ml_ready_data.csv'

PARAMS_TO_PLOT = ['AltitudePerigee_km', 'SemiMajorAxis_km', 'Period_sec', 'MeanMotion_revday', 'ndot_TERM_from_TLE', 'Bstar', 'Inclination_deg']

# Default polynomial degree and column-specific degrees - matching feature extraction
DEFAULT_POLYNOMIAL_DEGREE = 7
COLUMN_SPECIFIC_DEGREES = {
    'Bstar': 11,  # Higher degree for Bstar due to more complex variations
    'Inclination_deg': 9
}

def plot_fit_vs_original_separate(norad_id_str):
    """
    Loads data for a given NORAD ID and plots original data vs fitted polynomial
    in separate windows for each parameter.
    """
    try:
        norad_id = int(norad_id_str)
    except ValueError:
        print(f"Error: Invalid NORAD ID '{norad_id_str}'. Please enter a number.")
        return

    # Load ML Ready Data to get Coefficients
    try:
        ml_df = pd.read_csv(ML_DATA_FILE)
        if 'Norad_id' not in ml_df.columns:
             ml_df.reset_index(inplace=True)
             if 'Norad_id' not in ml_df.columns:
                  raise ValueError("Cannot find 'Norad_id' column or index.")
        ml_df['Norad_id'] = ml_df['Norad_id'].astype(int)

        satellite_coeffs_series = ml_df[ml_df['Norad_id'] == norad_id]

        if satellite_coeffs_series.empty:
            print(f"Error: No data found for NORAD ID {norad_id} in {ML_DATA_FILE}.")
            return
        satellite_coeffs = satellite_coeffs_series.iloc[0].to_dict()

    except FileNotFoundError:
        print(f"Error: ML data file not found at {ML_DATA_FILE}.")
        return
    except Exception as e:
        print(f"Error reading {ML_DATA_FILE}: {e}")
        return

    # Load Original TLE Data
    tle_file_path = os.path.join(TLE_DATA_FOLDER, f"{norad_id}.csv")
    try:
        df_orig = pd.read_csv(tle_file_path)
        if df_orig.empty:
            print(f"Error: Original TLE file is empty: {tle_file_path}")
            return

        df_orig['Epoch'] = pd.to_datetime(df_orig['Epoch'])
        df_orig = df_orig.set_index('Epoch').sort_index()

        df_orig.dropna(subset=PARAMS_TO_PLOT, inplace=True)
        
        # Get the highest polynomial degree needed for checking data points
        max_degree = max(DEFAULT_POLYNOMIAL_DEGREE, *COLUMN_SPECIFIC_DEGREES.values())
        
        if len(df_orig) < max_degree + 2:
            print(f"Error: Insufficient valid data points ({len(df_orig)}) in original TLE file for plotting.")
            return

        time_seconds = (df_orig.index - df_orig.index.min()).total_seconds()
        time_days = time_seconds / (24.0 * 3600.0)

    except FileNotFoundError:
        print(f"Error: Original TLE data file not found: {tle_file_path}")
        return
    except Exception as e:
        print(f"Error reading original TLE file {tle_file_path}: {e}")
        return

    # Generate a Separate Plot for each parameter
    for param in PARAMS_TO_PLOT:
        # Determine the appropriate polynomial degree for this parameter
        poly_degree = COLUMN_SPECIFIC_DEGREES.get(param, DEFAULT_POLYNOMIAL_DEGREE)
        
        # Create a new figure and axes for each parameter
        fig, ax = plt.subplots(figsize=(12, 7))

        original_y = df_orig[param].values

        try:
            # Retrieve coefficients in the correct order for np.poly1d
            coeffs_p = [satellite_coeffs[f'{param}_p{j}'] for j in range(poly_degree, -1, -1)]
            residual_std = satellite_coeffs.get(f'{param}_residual_stddev', np.nan)
        except KeyError as e:
            print(f"Warning: Could not find coefficient {e} for {param}. Skipping plot for this parameter.")
            plt.close(fig)
            continue

        # Reconstruct polynomial and predict
        poly_func = np.poly1d(coeffs_p)
        fitted_y = poly_func(time_days)

        # Plotting
        ax.plot(df_orig.index, original_y, marker='.', linestyle='-', markersize=4, label='Original Data', alpha=0.7)
        ax.plot(df_orig.index, fitted_y, color='red', linestyle='--', linewidth=2, label=f'Polynomial Fit (Deg {poly_degree})')
        ax.set_ylabel(param)
        ax.set_xlabel('Epoch (UTC)')

        title_str = f"NORAD {norad_id}: {param}"
        if not np.isnan(residual_std):
             title_str += f" (Residual StdDev: {residual_std:.4g})"
        ax.set_title(title_str)

        ax.legend()
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.show(block=False)

    print(f"\nGenerated {len(PARAMS_TO_PLOT)} separate plots for NORAD ID {norad_id}.")
    print("Close plot windows to continue or enter 'quit'.")

if __name__ == "__main__":
    while True:
        norad_id_input = input("Enter NORAD ID to plot (or 'quit' to exit): ")
        if norad_id_input.lower() == 'quit':
            break
        plot_fit_vs_original_separate(norad_id_input)

    print("Exiting script.")