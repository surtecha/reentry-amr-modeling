import pandas as pd
import numpy as np
import os
# Removed Spline import
# import statsmodels.api as sm # Option 1
from statsmodels.nonparametric.smoothers_lowess import lowess # Option 2 (more direct)
from sklearn.exceptions import FitFailedWarning # Still potentially relevant for other issues
import warnings

# Configuration
TLE_DATA_FOLDER = '/Users/suryatejachalla/Research/Re-entry-Prediction/Code/Orbital-Influence/tle_data_output'
METADATA_FILE = '/Users/suryatejachalla/Research/Re-entry-Prediction/Data/data.csv'
OUTPUT_ML_FILE = 'ml_ready_data_loess.csv' # Changed output filename for LOESS

# Parameters to fit curves to
PARAMS_TO_FIT = ['AltitudePerigee_km', 'SemiMajorAxis_km', 'Period_sec', 'MeanMotion_revday', 'ndot_TERM_from_TLE', 'Bstar', 'Inclination_deg']

# Minimum number of points required for LOESS fitting (can be same as spline)
MIN_DATA_POINTS = 10

# LOESS Configuration
# The fraction of data used for each local regression.
# Smaller values -> more local detail, potentially noisy
# Larger values -> smoother curve
# This value often requires tuning based on data inspection or cross-validation.
LOESS_FRAC = 0.25 # Example value - ADJUST AS NEEDED
LOESS_ITER = 3    # Number of robustness iterations (3 is common for outlier resistance)


# --- Renamed Function ---
def extract_features_for_satellite_loess(norad_id, file_path, target_am_map):
    """
    Loads TLE data, performs LOESS smoothing, extracts features,
    and adds the target A/M ratio for a single satellite.
    """
    print(f"Processing NORAD ID: {norad_id} (LOESS, frac={LOESS_FRAC})...")
    features = {'Norad_id': norad_id}

    try:
        # Load Data
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"  Skipping {norad_id}: CSV file is empty.")
            return None

        # Convert Epoch to datetime and set as index
        df['Epoch'] = pd.to_datetime(df['Epoch'])
        df = df.set_index('Epoch').sort_index()

        # Prepare for Fitting - Check NaNs
        essential_cols = PARAMS_TO_FIT + ['Bstar', 'Eccentricity', 'Inclination_deg', 'AltitudePerigee_km', 'SemiMajorAxis_km']
        df.dropna(subset=essential_cols, inplace=True)

        if len(df) < MIN_DATA_POINTS:
             print(f"  Skipping {norad_id}: Insufficient valid data points ({len(df)} < {MIN_DATA_POINTS}) after cleaning.")
             return None

        # Create numerical time variable (days since first epoch)
        time_seconds = (df.index - df.index.min()).total_seconds()
        time_days = time_seconds / (24.0 * 3600.0)
        features['time_duration_days'] = time_days.max()
        features['number_of_tles'] = len(df)

        # --- Perform LOESS Smoothing & Extract Fit Features ---
        for param in PARAMS_TO_FIT:
            if param not in df.columns:
                print(f"  Warning: Parameter '{param}' not found in data for {norad_id}. Skipping.")
                continue

            y = df[param].values

            # Ensure y is numeric and handle potential non-numeric entries
            y = pd.to_numeric(y, errors='coerce')
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < MIN_DATA_POINTS:
                print(f"  Skipping {param} for {norad_id}: Insufficient numeric points ({valid_mask.sum()}).")
                continue

            # Ensure data is sorted by time for LOESS efficiency and correct derivative calculation
            # (df index is already sorted, so time_days and y should be aligned and sorted by time)
            y_clean = y[valid_mask]
            time_days_clean = time_days[valid_mask]

            try:
                # Fit LOESS
                # lowess returns an array: [:, 0] is sorted exog, [:, 1] is smoothed endog
                # is_sorted=True assumes time_days_clean is already sorted (which it should be)
                smoothed_output = lowess(y_clean, time_days_clean,
                                         frac=LOESS_FRAC,
                                         it=LOESS_ITER,
                                         is_sorted=True,
                                         return_sorted=True) # Ensure output is sorted by time

                # Extract sorted time and smoothed y values
                time_days_sorted = smoothed_output[:, 0]
                y_smooth = smoothed_output[:, 1]

                # Store smoothed initial and final values
                features[f'{param}_smooth_initial'] = y_smooth[0] # First point after sorting
                features[f'{param}_smooth_final'] = y_smooth[-1] # Last point after sorting

                # Calculate residual standard deviation from the smoothed fit
                # Need to ensure y_clean aligns with the sorted time_days_sorted from lowess output.
                # Since we passed is_sorted=True and time_days_clean was sorted, y_clean is already correctly ordered.
                residuals = y_clean - y_smooth
                residual_stddev = np.std(residuals)
                features[f'{param}_smooth_residual_stddev'] = residual_stddev

                # Calculate average 1st derivative (trend) using numerical differentiation
                # np.gradient needs coordinates for non-uniform spacing (time_days_sorted)
                if len(time_days_sorted) > 1: # Need at least 2 points for gradient
                    deriv1 = np.gradient(y_smooth, time_days_sorted)
                    features[f'{param}_smooth_avg_deriv1'] = np.mean(deriv1)
                else:
                    features[f'{param}_smooth_avg_deriv1'] = np.nan # Or 0? NaN seems safer

                # Note: Calculating reliable average 2nd derivative numerically is often noisy. Omitted here.

            except Exception as e:
                print(f"  Error fitting LOESS for {param} on {norad_id}: {e}")
                # Add NaN features for this parameter if fitting fails
                features[f'{param}_smooth_initial'] = np.nan
                features[f'{param}_smooth_final'] = np.nan
                features[f'{param}_smooth_residual_stddev'] = np.nan
                features[f'{param}_smooth_avg_deriv1'] = np.nan


        # Extract Other Static Features (using original potentially noisy data)
        # (These remain the same as before)
        features['mean_bstar'] = df['Bstar'].mean()
        features['final_bstar'] = df['Bstar'].iloc[-1]
        features['final_eccentricity'] = df['Eccentricity'].iloc[-1]
        features['final_inclination'] = df['Inclination_deg'].iloc[-1]
        features['initial_AltitudePerigee_km'] = df['AltitudePerigee_km'].iloc[0]
        features['final_AltitudePerigee_km'] = df['AltitudePerigee_km'].iloc[-1]
        features['initial_SemiMajorAxis_km'] = df['SemiMajorAxis_km'].iloc[0]
        features['final_SemiMajorAxis_km'] = df['SemiMajorAxis_km'].iloc[-1]

        # Add Target Variable
        if norad_id in target_am_map:
             features['A/M_ratio'] = target_am_map[norad_id]
        else:
             print(f"  Warning: No A/M ratio found for NORAD ID {norad_id}.")
             features['A/M_ratio'] = np.nan

        print(f"  Successfully extracted features for {norad_id} using LOESS.")
        return features

    except FileNotFoundError:
        print(f"  Skipping {norad_id}: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"  Skipping {norad_id}: Error during processing - {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed debug
        return None

# --- Main Execution Logic ---
# (This part remains largely the same, just update function calls and print statements)

# Load Target A/M Data
try:
    target_df = pd.read_csv(METADATA_FILE)
    target_df['Norad_id'] = target_df['Norad_id'].astype(int)

    if 'A/M' in target_df.columns:
        am_column = 'A/M'
    elif 'A/M Ratio' in target_df.columns:
        am_column = 'A/M Ratio'
    else:
        am_col_found = [col for col in target_df.columns if col.lower() == 'a/m' or col.lower() == 'a/m ratio']
        if not am_col_found:
             raise ValueError(f"Could not find 'A/M' or similar column in {METADATA_FILE}")
        am_column = am_col_found[0]
        print(f"Using column name: {am_column}")

    target_am_map = target_df.set_index('Norad_id')[am_column].to_dict()
    print(f"Loaded A/M ratios for {len(target_am_map)} objects from {METADATA_FILE}.")
except FileNotFoundError:
    print(f"Error: Metadata file not found at {METADATA_FILE}. Cannot proceed.")
    exit()
except Exception as e:
    print(f"Error loading metadata file {METADATA_FILE}: {e}")
    exit()

# Find TLE Data Files and Extract NORAD IDs
all_files = os.listdir(TLE_DATA_FOLDER)
satellite_files = {}

for filename in all_files:
    if filename.endswith('.csv'):
        try:
            base_name = filename.split('.')[0]
            norad_id_str = ''.join(filter(str.isdigit, base_name))
            if norad_id_str:
                 norad_id = int(norad_id_str)
                 satellite_files[norad_id] = os.path.join(TLE_DATA_FOLDER, filename)
            else:
                 print(f"Warning: Could not parse NORAD ID from filename: {filename}")
        except ValueError:
            print(f"Warning: Could not parse NORAD ID from filename: {filename}")

print(f"Found {len(satellite_files)} potential satellite CSV files in {TLE_DATA_FOLDER}.")

# Iterate, Extract Features, and Collect Results
all_features_list = []
processed_ids = set()

ids_to_process = sorted(list(set(target_am_map.keys()) & set(satellite_files.keys())))
# Updated print statement
print(f"\nProcessing {len(ids_to_process)} satellites found in both metadata and TLE folder using LOESS...")

for norad_id in ids_to_process:
    if norad_id in satellite_files:
        file_path = satellite_files[norad_id]
        # *** CALL THE CORRECT (RENAMED) FUNCTION ***
        satellite_features = extract_features_for_satellite_loess(norad_id, file_path, target_am_map)
        if satellite_features is not None:
            all_features_list.append(satellite_features)
        processed_ids.add(norad_id)
    else:
         print(f"Logic Error: NORAD ID {norad_id} was in ids_to_process but not in satellite_files.")


# Report missing files or target data
missing_target_ids = set(satellite_files.keys()) - set(target_am_map.keys())
missing_file_ids = set(target_am_map.keys()) - set(satellite_files.keys())

if missing_target_ids:
    print(f"\nWarning: Found CSV files for {len(missing_target_ids)} NORAD IDs without A/M ratio in {METADATA_FILE}")
if missing_file_ids:
    print(f"\nWarning: Found A/M ratios for {len(missing_file_ids)} NORAD IDs without corresponding CSV file in {TLE_DATA_FOLDER}")

# Combine into Final DataFrame
if not all_features_list:
    print("\nError: No features could be extracted for any satellite. Cannot create output file.")
else:
    ml_df = pd.DataFrame(all_features_list)
    if 'Norad_id' in ml_df.columns:
        ml_df.set_index('Norad_id', inplace=True)
    else:
        print("\nError: 'Norad_id' column missing in the generated feature list.")

    # Updated print statement
    print(f"\nCreated LOESS-based DataFrame with shape: {ml_df.shape}")
    print("Columns:", ml_df.columns.tolist())

    cols_all_nan = ml_df.columns[ml_df.isna().all()].tolist()
    if cols_all_nan:
        print(f"\nWarning: The following feature columns are entirely NaN: {cols_all_nan}")

    if 'A/M_ratio' in ml_df.columns:
        missing_am_rows = ml_df[ml_df['A/M_ratio'].isna()].index.tolist()
        if missing_am_rows:
             print(f"\nWarning: {len(missing_am_rows)} objects have missing A/M_ratio in the final DataFrame.")
    else:
        print("\nWarning: 'A/M_ratio' column not found in the final DataFrame.")


    ml_df.to_csv(OUTPUT_ML_FILE)
    # Updated print statement
    print(f"\nSuccessfully saved LOESS-based ML-ready data to: {OUTPUT_ML_FILE}")