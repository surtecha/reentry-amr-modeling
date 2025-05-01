import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import re
import configparser
import argparse
from scipy import interpolate
import joblib  # For loading the scaler
import tensorflow as tf
from spacetrack import SpaceTrackClient
import warnings
import logging

# --- Configuration & Constants ---
CONFIG_FILE = 'config.ini'
SEQUENCE_LENGTH = 100  # Must match the training sequence length
FEATURES = ['apogee_altitude', 'mean_motion', 'mean_motion_derivative', 'inclination', 'eccentricity']
NUM_FEATURES = len(FEATURES)

# --- Orbital Mechanics Constants (from your first script) ---
GM = 398600441800000.0
GM13 = GM ** (1.0 / 3.0)
MRAD = 6378.137  # Earth radius in km
PI = math.pi
TPI86 = 2.0 * PI / 86400.0

# --- Suppress TensorFlow/Scipy/Spacetrack Warnings ---
# Suppress excessive TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ERROR level
tf.get_logger().setLevel('ERROR')
# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply") # From potential NaN in TLE calcs
# Suppress spacetrack INFO logs for cleaner output
logging.getLogger('spacetrack.base').setLevel(logging.WARNING)


# --- Helper Functions (Adapted from your scripts) ---

def tle_epoch_to_datetime(tle_epoch_str):
    """Converts TLE epoch string to datetime object."""
    try:
        year_short = int(tle_epoch_str[:2])
        day_of_year_fraction = float(tle_epoch_str[2:])

        if year_short < 57:
            year = 2000 + year_short
        else:
            year = 1900 + year_short

        start_of_year = datetime(year, 1, 1)
        # day_of_year_fraction includes the integer part (day number) and fractional part (time)
        # timedelta needs days, so subtract 1 from the integer part
        delta = timedelta(days=(day_of_year_fraction - 1))
        epoch_datetime = start_of_year + delta
        return epoch_datetime
    except ValueError:
        print(f"Error parsing TLE epoch string: {tle_epoch_str}")
        return None

def parse_scientific_notation(field):
    """Parses TLE scientific notation fields (like BSTAR)."""
    field = field.strip()
    # Handle explicit zero first
    if field == '00000+0' or field == '.00000+0' or field == '+00000+0' or field == '-00000+0':
         return 0.0
    # Regex for standard scientific notation like " 12345-5" or "-12345+1"
    match = re.match(r'([ +-])?(\d{5})([+-]\d)', field)
    if match:
        sign_char, mantissa_str, exponent_str = match.groups()
        sign = -1.0 if sign_char == '-' else 1.0
        mantissa = float(f'0.{mantissa_str}')
        exponent = int(exponent_str)
        return sign * mantissa * (10 ** exponent)
    else:
        # Fallback for potentially simpler formats or errors
        try:
            # Check if it's just a plain number (though not typical for BSTAR)
            return float(field)
        except ValueError:
            print(f"Warning: Could not parse scientific notation: '{field}'")
            return None # Indicate parsing failure

def parse_tle_data(tle_lines):
    """Parses raw TLE lines into a DataFrame with orbital parameters."""
    epochs = []
    epochs_utc = []
    perigee_altitudes = []
    apogee_altitudes = []
    mean_motions = []
    mean_motion_derivatives = []
    inclinations = []
    eccentricities = []
    bstars = []

    i = 0
    while i < len(tle_lines) - 1:
        line1 = tle_lines[i].strip()
        line2 = tle_lines[i + 1].strip()

        # Basic TLE format check
        if line1.startswith('1 ') and line2.startswith('2 ') and len(line1) >= 69 and len(line2) >= 69:
            try:
                # --- Line 1 Parsing ---
                epoch_str = line1[18:32].strip()
                first_deriv_str = line1[33:43].strip() # First derivative of mean motion / 2
                bstar_str = line1[53:61].strip()       # BSTAR drag term

                # --- Line 2 Parsing ---
                inclination_deg = float(line2[8:16].strip())
                eccentricity_str = line2[26:33].strip() # Assumed decimal point
                mean_motion_rev_day = float(line2[52:63].strip())

                # --- Calculations & Conversions ---
                epoch_dt = tle_epoch_to_datetime(epoch_str)
                if epoch_dt is None: # Skip if epoch parsing failed
                    i += 2
                    continue
                epoch_utc = epoch_dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

                # Handle potential missing decimal point in eccentricity
                if '.' not in eccentricity_str:
                     eccentricity = float(f'0.{eccentricity_str}')
                else:
                     eccentricity = float(eccentricity_str) # Should not happen per format, but safe check

                # First derivative is divided by 2 in TLE line 1
                # The training script multiplied by 2, so we do the same here
                # Convert from rev/day^2 to rev/day^2 (no change needed here, just naming)
                mean_motion_dot = float(first_deriv_str) * 2.0

                bstar = parse_scientific_notation(bstar_str)
                if bstar is None: # Skip if BSTAR parsing failed
                     print(f"Skipping TLE pair due to BSTAR parsing error: {bstar_str}")
                     i += 2
                     continue

                mmoti = mean_motion_rev_day
                ecc = eccentricity

                # Basic validity checks
                if mmoti <= 0 or ecc < 0 or ecc >= 1.0:
                    print(f"Skipping TLE pair due to invalid orbital elements: MM={mmoti}, e={ecc}")
                    i += 2
                    continue

                # Calculate Semi-Major Axis (SMA) in km
                # mm_rad_sec = mmoti * TPI86 # Mean motion in radians per second
                # sma_meters = GM13 / (mm_rad_sec ** (2.0 / 3.0)) # Old formula seems to use GM^(1/3) incorrectly
                # Correct formula for SMA from mean motion: a = (GM / n^2)^(1/3) where n is in rad/s
                n_rad_per_sec = mmoti * (2 * PI) / 86400.0
                sma_meters = (GM / (n_rad_per_sec**2))**(1/3.0)
                sma_km = sma_meters / 1000.0

                # Calculate Apogee and Perigee Altitude in km
                apo_km = sma_km * (1.0 + ecc)
                per_km = sma_km * (1.0 - ecc)
                apo_alt_km = apo_km - MRAD
                per_alt_km = per_km - MRAD

                # Append data
                epochs.append(epoch_dt)
                epochs_utc.append(epoch_utc)
                perigee_altitudes.append(per_alt_km)
                apogee_altitudes.append(apo_alt_km)
                mean_motions.append(mean_motion_rev_day)
                mean_motion_derivatives.append(mean_motion_dot) # Use the value * 2
                inclinations.append(inclination_deg)
                eccentricities.append(eccentricity)
                bstars.append(bstar)

                i += 2 # Move to the next TLE pair
            except ValueError as e:
                print(f"Skipping TLE pair due to ValueError during processing: {e}")
                print(f"  Line 1: {line1}")
                print(f"  Line 2: {line2}")
                i += 2
            except Exception as e:
                print(f"Skipping TLE pair due to unexpected error: {e}")
                print(f"  Line 1: {line1}")
                print(f"  Line 2: {line2}")
                i += 2
        else:
            # Move one line forward if the current pair doesn't match TLE format
            i += 1

    if not epochs:
        return pd.DataFrame() # Return empty DataFrame if no valid TLEs parsed

    # Create DataFrame
    data = pd.DataFrame({
        'epoch': epochs,
        'epoch_utc': epochs_utc,
        'apogee_altitude': apogee_altitudes,
        'perigee_altitude': perigee_altitudes,
        'mean_motion': mean_motions,
        'mean_motion_derivative': mean_motion_derivatives,
        'inclination': inclinations,
        'eccentricity': eccentricities,
        'bstar': bstars
    })

    # Sort by epoch just in case TLEs weren't perfectly ordered
    data = data.sort_values(by='epoch').reset_index(drop=True)
    return data


def parse_datetime_resample(dt_str):
    """Parses datetime strings used during resampling."""
    try:
        # Try ISO format with microseconds
        return datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        try:
            # Try ISO format without microseconds (if original data had it)
             return datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
             # Fallback for pandas default if needed (less likely)
             return pd.to_datetime(dt_str)


def resample_dataframe(df, n_points=SEQUENCE_LENGTH):
    """Resamples the DataFrame to a fixed number of points using interpolation."""
    if len(df) < 2:
        print(f"Warning: Cannot interpolate with less than 2 data points. Returning original data (length {len(df)}).")
        # If needed, pad or truncate here to exactly n_points, but interpolation isn't possible.
        # For simplicity, we'll let the calling function handle size mismatch.
        return df

    # Convert epoch_utc to numeric timestamp for interpolation
    # Use the specific parsing function to handle potential format variations
    df['timestamp'] = df['epoch_utc'].apply(lambda x: parse_datetime_resample(x).timestamp())

    # Ensure data is sorted by time
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Create interpolation parameters (0 to 1 range)
    old_indices = np.linspace(0, 1, len(df))
    new_indices = np.linspace(0, 1, n_points)

    # Create the new dataframe
    new_df = pd.DataFrame()

    # Interpolate timestamps and convert back to UTC string format
    timestamp_interp = interpolate.interp1d(old_indices, df['timestamp'].values, kind='linear', fill_value="extrapolate")
    new_timestamps = timestamp_interp(new_indices)
    new_datetimes_utc = [datetime.fromtimestamp(ts, tz=None).strftime('%Y-%m-%dT%H:%M:%S.%fZ') for ts in new_timestamps] # Use tz=None for naive datetime
    new_df['epoch_utc'] = new_datetimes_utc
    # Optional: Recreate naive 'epoch' column if needed elsewhere, though 'epoch_utc' is standard
    # new_df['epoch'] = [parse_datetime_resample(dt_str) for dt_str in new_df['epoch_utc']]

    # Interpolate all numerical columns required for the model + any others present
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    # Ensure 'timestamp' isn't accidentally interpolated if it wasn't dropped yet
    if 'timestamp' in numeric_columns:
        numeric_columns.remove('timestamp')

    for col in numeric_columns:
        if col in df.columns:
            try:
                # Use linear interpolation; extrapolate fills ends if needed
                interp_func = interpolate.interp1d(old_indices, df[col].values, kind='linear', fill_value="extrapolate")
                new_df[col] = interp_func(new_indices)
            except ValueError as e:
                 print(f"Warning: Interpolation failed for column '{col}': {e}. Filling with NaN.")
                 new_df[col] = np.nan # Or handle differently (e.g., ffill/bfill)
            except Exception as e:
                 print(f"Warning: Unexpected error during interpolation for column '{col}': {e}. Filling with NaN.")
                 new_df[col] = np.nan

    # Ensure the final dataframe has exactly n_points rows
    if len(new_df) != n_points:
        print(f"Warning: Resampling resulted in {len(new_df)} rows, expected {n_points}. Adjusting...")
        if len(new_df) > n_points:
            new_df = new_df.iloc[:n_points]
        else: # len(new_df) < n_points (shouldn't happen with linear interpolation/extrapolation)
             # Pad with the last row if needed
            while len(new_df) < n_points:
                new_df = pd.concat([new_df, new_df.iloc[[-1]]], ignore_index=True)

    return new_df


# --- Main Execution ---
def main(norad_id):
    """Fetches TLEs, preprocesses data, and predicts AMR for a given NORAD ID."""
    print(f"--- Processing NORAD ID: {norad_id} ---")

    # 1. Load Configuration
    print("Loading configuration...")
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        return
    try:
        config.read(CONFIG_FILE)
        st_identity = config['spacetrack']['identity']
        st_password = config['spacetrack']['password']
        model_path = config['files']['model_path']
        scaler_path = config['files']['scaler_path']
    except KeyError as e:
        print(f"Error: Missing key in '{CONFIG_FILE}': {e}")
        return
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        return

    # Validate existence of model and scaler files
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at '{scaler_path}'")
        return

    # 2. Fetch TLE Data from Space-Track
    print("Connecting to Space-Track...")
    try:
        st = SpaceTrackClient(identity=st_identity, password=st_password)
    except Exception as e:
        print(f"Error initializing SpaceTrackClient: {e}")
        return

    print("Fetching TLE data for the last 6 months...")
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180) # Approx 6 months
    date_range_str = f"{start_date.strftime('%Y-%m-%d')}--{end_date.strftime('%Y-%m-%d')}"

    try:
        # Fetch TLEs in chronological order (ascending epoch)
        tle_raw = st.tle(
            norad_cat_id=norad_id,
            orderby='epoch asc', # Important for correct sequence order
            epoch=date_range_str,
            format='tle'
            )

        if not tle_raw:
            print(f"No TLE data found for NORAD ID {norad_id} in the last 6 months.")
            return

        tle_lines = tle_raw.strip().splitlines()
        print(f"Fetched {len(tle_lines) // 2} TLE pairs.")

    except Exception as e:
        print(f"Error fetching TLE data from Space-Track: {e}")
        return

    # 3. Parse TLE Data
    print("Parsing TLE data...")
    df_parsed = parse_tle_data(tle_lines)

    if df_parsed.empty:
        print("No valid TLE data could be parsed.")
        return
    elif len(df_parsed) < 2:
         print(f"Insufficient valid TLE data points ({len(df_parsed)}) for resampling. Need at least 2.")
         return
    else:
        print(f"Successfully parsed {len(df_parsed)} valid TLE data points.")
        # print(df_parsed.head()) # Optional: Print head for verification

    # 4. Resample Data
    print(f"Resampling data to {SEQUENCE_LENGTH} points...")
    df_resampled = resample_dataframe(df_parsed, n_points=SEQUENCE_LENGTH)

    if len(df_resampled) != SEQUENCE_LENGTH:
         print(f"Error: Resampling did not produce exactly {SEQUENCE_LENGTH} data points (produced {len(df_resampled)}). Cannot proceed.")
         return
    elif df_resampled[FEATURES].isnull().values.any():
         print(f"Error: Resampled data contains NaN values in feature columns. Cannot proceed.")
         # Optionally, print which columns have NaN:
         # print(df_resampled.isnull().sum())
         return
    else:
         print("Resampling successful.")
         # print(df_resampled.head()) # Optional: Print head

    # 5. Load Scaler and Model
    print("Loading scaler and model...")
    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from '{scaler_path}'.")
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from '{model_path}'.")
        # model.summary() # Optional: Print model summary
    except FileNotFoundError:
        print(f"Error: Could not find model ('{model_path}') or scaler ('{scaler_path}') file.")
        return
    except Exception as e:
        print(f"Error loading scaler or model: {e}")
        return

    # 6. Prepare Data for Prediction
    print("Preparing data for prediction...")
    # Select the features the model was trained on
    sequence_data = df_resampled[FEATURES].values

    # Check shape before scaling
    if sequence_data.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
        print(f"Error: Data shape before scaling is {sequence_data.shape}, expected ({SEQUENCE_LENGTH}, {NUM_FEATURES}).")
        return

    # Scale the data using the loaded scaler
    # Scaler expects 2D input: (n_samples * seq_len, n_features)
    # In this case, n_samples is 1
    try:
        sequence_scaled = scaler.transform(sequence_data) # scaler was fit on reshaped data, so transform needs same shape
    except ValueError as e:
        print(f"Error during scaling: {e}. Check if feature columns match scaler's expected features.")
        # This can happen if FEATURES list doesn't match what the scaler was trained on.
        print(f"Scaler was trained on {scaler.n_features_in_} features.")
        return
    except Exception as e:
        print(f"Unexpected error during scaling: {e}")
        return

    # Reshape for LSTM input: (batch_size, timesteps, features) -> (1, SEQUENCE_LENGTH, NUM_FEATURES)
    sequence_final = sequence_scaled.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
    print(f"Data prepared with shape: {sequence_final.shape}")

    # 7. Predict AMR
    print("Predicting AMR...")
    try:
        predicted_amr_array = model.predict(sequence_final)
        # The prediction is likely [[value]], extract the scalar
        predicted_amr = predicted_amr_array[0][0]
        print("--- Prediction Complete ---")
        print(f"Predicted AMR for NORAD ID {norad_id}: {predicted_amr:.6f}") # Format to 6 decimal places
    except Exception as e:
        print(f"Error during prediction: {e}")

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict AMR for a given NORAD ID using TLE data.")
    parser.add_argument("norad_id", type=int, help="The NORAD Catalog ID of the satellite.")
    args = parser.parse_args()

    main(args.norad_id)


'''
[spacetrack]
identity = abc@gmail.com
password = pass

[files]
model_path = bilstm_amr_predictor.keras  
scaler_path = feature_scaler.joblib 
'''
