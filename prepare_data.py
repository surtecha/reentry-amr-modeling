import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error

# Configuration
TLE_DATA_FOLDER = '/Users/suryatejachalla/Research/Re-entry-Prediction/Code/Orbital-Influence/tle_data_output'
METADATA_FILE = '/Users/suryatejachalla/Research/Re-entry-Prediction/Data/data.csv'
OUTPUT_ML_FILE = 'ml_ready_data.csv'

# Parameters to fit curves to
PARAMS_TO_FIT = ['AltitudePerigee_km', 'SemiMajorAxis_km', 'Period_sec', 'MeanMotion_revday', 'ndot_TERM_from_TLE', 'Bstar', 'Inclination_deg']

# Default polynomial degree and column-specific degrees
DEFAULT_POLYNOMIAL_DEGREE = 7
COLUMN_SPECIFIC_DEGREES = {
    'Bstar': 11,
    'Inclination_deg': 9
}

def extract_features_for_satellite(norad_id, file_path, target_am_map):
    """
    Loads TLE data, performs curve fitting, extracts features,
    and adds the target A/M ratio for a single satellite.
    """
    print(f"Processing NORAD ID: {norad_id}...")
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

        # Prepare for Fitting
        essential_cols = PARAMS_TO_FIT + ['Bstar', 'Eccentricity', 'Inclination_deg']
        df.dropna(subset=essential_cols, inplace=True)
        
        # Get the highest polynomial degree needed to ensure we have enough data points
        max_degree = max(DEFAULT_POLYNOMIAL_DEGREE, *COLUMN_SPECIFIC_DEGREES.values())
        
        if len(df) < max_degree + 2:
             print(f"  Skipping {norad_id}: Insufficient valid data points ({len(df)}) after cleaning.")
             return None

        # Create numerical time variable (days since first epoch)
        time_seconds = (df.index - df.index.min()).total_seconds()
        time_days = time_seconds / (24.0 * 3600.0)
        features['time_duration_days'] = time_days.max()
        features['number_of_tles'] = len(df)

        # Perform Curve Fitting & Extract Fit Features
        for param in PARAMS_TO_FIT:
            # Determine the appropriate polynomial degree for this parameter
            poly_degree = COLUMN_SPECIFIC_DEGREES.get(param, DEFAULT_POLYNOMIAL_DEGREE)
            
            y = df[param].values

            # Fit polynomial with parameter-specific degree
            coeffs = np.polyfit(time_days, y, poly_degree)
            poly_func = np.poly1d(coeffs)
            y_pred = poly_func(time_days)

            # Store coefficients (p0 is the last coeff from polyfit)
            for i, coeff in enumerate(reversed(coeffs)):
                features[f'{param}_p{i}'] = coeff

            # Calculate residual standard deviation
            residuals = y - y_pred
            residual_stddev = np.std(residuals)
            features[f'{param}_residual_stddev'] = residual_stddev

        # Extract Other Static Features
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

        print(f"  Successfully extracted features for {norad_id}.")
        return features

    except FileNotFoundError:
        print(f"  Skipping {norad_id}: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"  Skipping {norad_id}: Error during processing - {e}")
        return None

# Load Target A/M Data
try:
    target_df = pd.read_csv(METADATA_FILE)
    target_df['Norad_id'] = target_df['Norad_id'].astype(int)
    
    if 'A/M' in target_df.columns:
        am_column = 'A/M'
    elif 'A/M Ratio' in target_df.columns:
        am_column = 'A/M Ratio'
    else:
        raise ValueError(f"Could not find 'A/M' or similar column in {METADATA_FILE}")

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
            norad_id = int(filename.split('.')[0])
            satellite_files[norad_id] = os.path.join(TLE_DATA_FOLDER, filename)
        except ValueError:
            print(f"Warning: Could not parse NORAD ID from filename: {filename}")

print(f"Found {len(satellite_files)} potential satellite CSV files in {TLE_DATA_FOLDER}.")

# Iterate, Extract Features, and Collect Results
all_features_list = []
processed_ids = set()

ids_to_process = sorted(list(set(target_am_map.keys()) & set(satellite_files.keys())))
print(f"Processing {len(ids_to_process)} satellites found in both metadata and TLE folder...")

for norad_id in ids_to_process:
    file_path = satellite_files[norad_id]
    satellite_features = extract_features_for_satellite(norad_id, file_path, target_am_map)
    if satellite_features is not None:
        all_features_list.append(satellite_features)
    processed_ids.add(norad_id)

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
    ml_df.set_index('Norad_id', inplace=True)

    print(f"\nCreated DataFrame with shape: {ml_df.shape}")
    print("Columns:", ml_df.columns.tolist())

    cols_all_nan = ml_df.columns[ml_df.isna().all()].tolist()
    if cols_all_nan:
        print(f"\nWarning: The following feature columns are entirely NaN: {cols_all_nan}")

    missing_am_rows = ml_df[ml_df['A/M_ratio'].isna()].index.tolist()
    if missing_am_rows:
         print(f"\nWarning: {len(missing_am_rows)} objects have missing A/M_ratio in the final DataFrame.")

    ml_df.to_csv(OUTPUT_ML_FILE)
    print(f"\nSuccessfully saved ML-ready data to: {OUTPUT_ML_FILE}")