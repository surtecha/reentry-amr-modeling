import os
import pandas as pd
import numpy as np
import joblib # For loading scalers
import argparse
import logging
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import matplotlib
import requests
from sgp4.api import Satrec
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import time
import sys
import configparser
import traceback
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

matplotlib.use('Agg') # Use non-interactive backend

# --- Configuration ---
# Paths from the Log+StandardScaler run where models/scalers are saved
RESULTS_DIR_BEST_RUN = 'results-v2'
MODELS_DIR_BEST = os.path.join(RESULTS_DIR_BEST_RUN, 'models')
SUMMARY_FILE_BEST = os.path.join(RESULTS_DIR_BEST_RUN, 'tuning_summary.csv')
X_SCALER_PATH = os.path.join(RESULTS_DIR_BEST_RUN, 'x_scaler.joblib')
Y_SCALER_PATH = os.path.join(RESULTS_DIR_BEST_RUN, 'y_scaler.joblib') # Scaler for log-transformed y

# Path to the saved parameter forecasting model
FORECAST_MODEL_DIR = 'results_ensemble_forecast' # Directory where forecast model was saved
FORECAST_MODEL_PATH = os.path.join(FORECAST_MODEL_DIR, 'parameter_forecast_model.keras')

# Output directory for this script's results
PREDICTION_OUTPUT_DIR = 'live_prediction_output'
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

# SpaceTrack Config
SPACETRACK_CONFIG_FILE = 'config.ini'
SPACETRACK_URI = "https://www.space-track.org"
SPACETRACK_LOGIN_URL = SPACETRACK_URI + '/ajaxauth/login'
SPACETRACK_LOGOUT_URL = SPACETRACK_URI + '/ajaxauth/logout'
SPACETRACK_QUERY_URL = SPACETRACK_URI + '/basicspacedata/query'
MONTHS_PRIOR_TLE = 3 # How far back to fetch TLEs
GM_EARTH = 3.986008e14
R_EARTH_KM_FOR_PLOT = 6378.135

# Rate Limiting Parameters
MIN_REQUEST_INTERVAL_SEC = 3.0 # Be conservative
MAX_REQUESTS_PER_MINUTE = 20
MAX_REQUESTS_PER_HOUR = 250 # Lower than absolute max to be safe

# Model/Data Parameters (MUST match training configuration)
N_ENSEMBLE_MODELS = 5
N_RESAMPLED_POINTS = 90 # Target number of points after resampling
PARAMETERS_TO_USE = [
    'mean_motion', 'eccentricity', 'inclination', 'bstar',
    'mean_motion_deriv', 'semi_major_axis', 'orbital_period', 'perigee_alt'
]
FEATURE_COLUMNS = PARAMETERS_TO_USE # Use these as the final column names
NUM_FEATURES = len(FEATURE_COLUMNS)
N_FORECAST_INPUT_STEPS = 60 # Must match forecast model training
N_FORECAST_STEPS = 10

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Ensure logs go to console

# --- Helper Functions ---
# calculate_epoch_dt, parse_ndot, process_tle_data, smooth_and_resample, plot_parameter_forecasts
# (Keep these helper functions exactly as they were in the previous version)
def calculate_epoch_dt(satrec_obj):
    """Calculates datetime object from Satrec epoch fields."""
    year = satrec_obj.epochyr
    if year < 57: year += 2000
    else: year += 1900
    day = satrec_obj.epochdays
    day_int = int(day)
    frac_day = day - day_int
    base_date = datetime(year, 1, 1, tzinfo=timezone.utc)
    dt_epoch = base_date + timedelta(days=day_int - 1, seconds=frac_day * 86400.0)
    return dt_epoch

def parse_ndot(tle_line1_str):
    """Parses the n-dot/2 term robustly from TLE line 1."""
    ndot_raw = tle_line1_str[33:43].strip()
    ndot_formatted = "0.0"
    if ndot_raw:
        try:
            # Construct the scientific notation string
            sign = ''
            if ndot_raw.startswith(('-', '+')):
                sign = ndot_raw[0]
                num_part_raw = ndot_raw[1:]
            else:
                num_part_raw = ndot_raw

            # Check for exponent part, format: digits+sign+exponent_digit
            if len(num_part_raw) > 1 and num_part_raw[-2] in ['-', '+', ' '] and num_part_raw[-1].isdigit():
                num_part = num_part_raw[:-2]
                exp_sign = '-' if num_part_raw[-2] == '-' else '+'
                exp_val = num_part_raw[-1]
                ndot_formatted = f"{sign}.{num_part}e{exp_sign}{exp_val}"
            elif num_part_raw.isdigit(): # No exponent, just digits
                 ndot_formatted = f"{sign}.{num_part_raw}"
            else: # Already contains decimal or other format? Pass through.
                 ndot_formatted = ndot_raw

            # Final conversion and validation
            ndot_term = float(ndot_formatted)

        except (ValueError, IndexError):
            logging.warning(f"Could not parse ndot_term '{tle_line1_str[33:43]}'. Using 0.0.")
            ndot_term = 0.0
    return ndot_term

def process_tle_data(tles_list):
    """Parses raw TLE text list and calculates derived parameters."""
    parsed_tles = []
    parse_errors = 0
    lines = tles_list # Assume tles_list is already splitlines()
    line_idx = 0
    while line_idx < len(lines) - 1:
        line1, line2 = lines[line_idx].strip(), lines[line_idx+1].strip()
        if not line1.startswith('1 ') or not line2.startswith('2 '):
            line_idx += 1; continue
        try:
            sat = Satrec.twoline2rv(line1, line2)
            dt_epoch = calculate_epoch_dt(sat)
            ndot_term = parse_ndot(line1) # Use helper function

            # Use standard names matching the final desired columns where possible
            mean_motion_rad_min = sat.no_kozai
            mean_motion_rev_day = mean_motion_rad_min * (1440.0 / (2.0 * np.pi))

            parsed_tles.append({
                'Epoch': dt_epoch,
                'mean_motion': mean_motion_rev_day, # Rename directly
                'eccentricity': sat.ecco, # Rename
                'inclination': np.degrees(sat.inclo), # Rename
                'bstar': sat.bstar, # Rename
                'mean_motion_deriv': ndot_term, # Rename (n-dot/2)
                # Keep intermediate names for calculation clarity
                '_raan_deg': np.degrees(sat.nodeo),
                '_argp_deg': np.degrees(sat.argpo),
                '_mean_anomaly_deg': np.degrees(sat.mo),
                '_mean_motion_rad_min': mean_motion_rad_min
            })
            line_idx += 2
        except Exception as e:
            logging.warning(f"Error processing TLE pair near line {line_idx}: {e}")
            parse_errors += 1
            line_idx += 2 # Skip pair on error

    if parse_errors > 0: logging.warning(f"TLE parsing completed with {parse_errors} errors.")
    if not parsed_tles: return None

    tle_df = pd.DataFrame(parsed_tles)
    tle_df.set_index('Epoch', inplace=True)
    tle_df.sort_index(inplace=True)

    # Calculate derived parameters
    tle_df['mean_motion_rad_sec'] = tle_df['_mean_motion_rad_min'] / 60.0
    valid_n = tle_df['mean_motion_rad_sec'] > 1e-9
    tle_df['semi_major_axis'] = np.nan # Use final column name
    tle_df.loc[valid_n, 'semi_major_axis'] = ((GM_EARTH / tle_df.loc[valid_n, 'mean_motion_rad_sec']**2)**(1.0/3.0)) / 1000.0 # Directly in km

    tle_df['orbital_period'] = np.nan # Use final column name
    tle_df.loc[valid_n, 'orbital_period'] = (2.0 * np.pi / tle_df.loc[valid_n, 'mean_motion_rad_sec']) / 60.0 # Directly in minutes

    valid_sma = tle_df['semi_major_axis'].notna() & (tle_df['semi_major_axis'] > 0)
    valid_ecc = tle_df['eccentricity'].notna() & (tle_df['eccentricity'] >= 0) & (tle_df['eccentricity'] < 1)
    valid_rows_for_alt = valid_sma & valid_ecc
    tle_df['perigee_alt'] = np.nan # Use final column name
    if valid_rows_for_alt.any():
         rp_km = tle_df.loc[valid_rows_for_alt, 'semi_major_axis'] * (1.0 - tle_df.loc[valid_rows_for_alt, 'eccentricity'])
         tle_df.loc[valid_rows_for_alt, 'perigee_alt'] = rp_km - R_EARTH_KM_FOR_PLOT

    # Drop intermediate columns and rows with NaN in essential calculated columns
    intermediate_cols = ['mean_motion_rad_sec', '_mean_motion_rad_min', '_raan_deg', '_argp_deg', '_mean_anomaly_deg']
    tle_df.drop(columns=[col for col in intermediate_cols if col in tle_df.columns], inplace=True)
    essential_derived = ['semi_major_axis', 'orbital_period', 'perigee_alt']
    original_rows = len(tle_df)
    tle_df.dropna(subset=essential_derived, inplace=True)
    if len(tle_df) < original_rows: logging.info(f"Dropped {original_rows - len(tle_df)} rows due to NaN derived values.")

    return tle_df

def smooth_and_resample(df_input, target_points=90, required_cols=None):
    """Applies Savgol filter and cubic interpolation."""
    if required_cols:
        missing_cols = [col for col in required_cols if col not in df_input.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for smoothing: {missing_cols}")
        df = df_input[required_cols].copy()
    else:
        df = df_input.copy()

    if len(df) < 3: # Need at least 3 points for interpolation/smoothing
        raise ValueError(f"Insufficient data points ({len(df)}) for smoothing/resampling.")

    original_indices = np.arange(len(df))
    new_indices = np.linspace(0, len(df) - 1, target_points)
    processed_df = pd.DataFrame(index=np.arange(target_points)) # Use simple range index

    for column in df.columns:
        y = df[column].values
        # Savgol filter settings
        window_length = min(7, len(y))
        window_length = window_length if window_length % 2 != 0 else window_length - 1 # Ensure odd
        window_length = max(3, window_length) # Ensure >= 3
        poly_order = min(2, window_length - 1)

        if len(y) >= window_length:
            smoothed_y = savgol_filter(y, window_length, poly_order)
        else:
            smoothed_y = y # Not enough points to smooth

        # Interpolation
        try:
            interp_func = interp1d(original_indices, smoothed_y, kind='cubic', bounds_error=False, fill_value='extrapolate')
            new_y = interp_func(new_indices)
        except ValueError as e:
             logging.warning(f"Cubic interpolation failed for {column}: {e}. Falling back to linear.")
             interp_func = interp1d(original_indices, smoothed_y, kind='linear', bounds_error=False, fill_value='extrapolate')
             new_y = interp_func(new_indices)

        processed_df[column] = new_y

    return processed_df

def plot_parameter_forecasts(historical_data_orig, forecast_data_orig, feature_names, output_dir, obj_id):
    """Plots historical data and forecasts for each parameter."""
    n_hist_steps = historical_data_orig.shape[0]
    n_forecast_steps = forecast_data_orig.shape[0]
    n_features = historical_data_orig.shape[1]

    if n_features != len(feature_names):
        logging.error("Number of features mismatch in plot_parameter_forecasts.")
        return

    hist_time = np.arange(n_hist_steps)
    future_time = np.arange(n_hist_steps, n_hist_steps + n_forecast_steps)

    obj_plot_dir = os.path.join(output_dir, f'forecast_plots_{obj_id}')
    os.makedirs(obj_plot_dir, exist_ok=True)

    for i in range(n_features):
        feature_name = feature_names[i]
        plt.figure(figsize=(12, 6))
        plt.plot(hist_time, historical_data_orig[:, i], 'bo-', label='Smoothed/Resampled Data', markersize=4) # Updated label
        plt.plot(future_time, forecast_data_orig[:, i], 'ro--', label='Forecasted Data', markersize=4)
        plt.title(f'Parameter Forecast: {obj_id} - {feature_name}')
        plt.xlabel('Resampled Time Step (0-89) | Forecast Step (90-99)')
        plt.ylabel('Parameter Value')
        plt.legend()
        plt.grid(True, linestyle='--')
        plot_path = os.path.join(obj_plot_dir, f'forecast_{feature_name}.png')
        try:
            plt.savefig(plot_path)
        except Exception as e:
            logging.error(f"Error saving plot {plot_path}: {e}")
        plt.close()
    logging.info(f"Saved forecast plots for object {obj_id} to {obj_plot_dir}")

# --- Main Execution ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fetch TLEs, process, predict AMR, and forecast parameters for a NORAD ID.")
    parser.add_argument("norad_id", type=int, help="The NORAD Catalog ID of the space object.")
    args = parser.parse_args()

    norad_id_input = args.norad_id
    object_id = str(norad_id_input) # Use NORAD ID as object ID string
    predicted_amr_final = None # Initialize variable to store final prediction

    # --- Load SpaceTrack Credentials ---
    logging.info(f"Loading SpaceTrack credentials from {SPACETRACK_CONFIG_FILE}")
    config = configparser.ConfigParser()
    if not os.path.exists(SPACETRACK_CONFIG_FILE):
        logging.error(f"Config file '{SPACETRACK_CONFIG_FILE}' not found.")
        sys.exit(1)
    try:
        config.read(SPACETRACK_CONFIG_FILE)
        SPACETRACK_USER = config['SPACE_TRACK']['username']
        SPACETRACK_PASS = config['SPACE_TRACK']['password']
    except KeyError:
        logging.error("Username or password not found in [SPACE_TRACK] section of config file.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading config file: {e}")
        sys.exit(1)

    # --- SpaceTrack Session and Rate Limiting Init ---
    session = requests.Session()
    last_request_time = time.monotonic()
    request_times_minute = []
    request_times_hour = []
    logged_in = False

    try:
        # --- Login ---
        logging.info("Attempting Space-Track login...")
        login_data = {'identity': SPACETRACK_USER, 'password': SPACETRACK_PASS}
        login_response = session.post(SPACETRACK_LOGIN_URL, data=login_data)
        login_response.raise_for_status()
        logging.info("Space-Track login successful.")
        logged_in = True
        # Update rate limit tracking
        current_time = time.monotonic(); request_times_minute.append(current_time); request_times_hour.append(current_time); last_request_time = current_time

        # --- Fetch Latest TLE to determine date range ---
        logging.info(f"Fetching latest TLE for NORAD ID {norad_id_input}...")
        time_since_last = time.monotonic() - last_request_time
        if time_since_last < MIN_REQUEST_INTERVAL_SEC: time.sleep(MIN_REQUEST_INTERVAL_SEC - time_since_last)

        latest_tle_query_url = (f"{SPACETRACK_QUERY_URL}/class/tle/"
                                f"NORAD_CAT_ID/{norad_id_input}/orderby/EPOCH%20desc/limit/1/format/tle")
        latest_tle_response = session.get(latest_tle_query_url)
        latest_tle_response.raise_for_status()
        latest_tle_data = latest_tle_response.text.strip().splitlines()
        current_time = time.monotonic(); request_times_minute.append(current_time); request_times_hour.append(current_time); last_request_time = current_time

        if not latest_tle_data or len(latest_tle_data) < 2: raise ValueError(f"No valid latest TLE found for {norad_id_input}.")
        tle1_latest, tle2_latest = None, None
        for k in range(len(latest_tle_data) - 1):
            l1, l2 = latest_tle_data[k].strip(), latest_tle_data[k+1].strip()
            if l1.startswith('1 ') and l2.startswith('2 '): tle1_latest, tle2_latest = l1, l2; break
        if tle1_latest is None: raise ValueError("Could not parse TLE lines from latest data.")

        sat_latest = Satrec.twoline2rv(tle1_latest, tle2_latest)
        latest_epoch_dt_naive = calculate_epoch_dt(sat_latest).replace(tzinfo=None)
        END_DATE_dt = latest_epoch_dt_naive
        START_DATE_dt = END_DATE_dt - relativedelta(months=MONTHS_PRIOR_TLE)
        END_DATE_str, START_DATE_str = END_DATE_dt.strftime('%Y-%m-%d'), START_DATE_dt.strftime('%Y-%m-%d')
        logging.info(f"Latest Epoch: {latest_epoch_dt_naive.strftime('%Y-%m-%d %H:%M:%S')} UTC. Fetching history from {START_DATE_str} to {END_DATE_str}")

        # --- Fetch Historical TLEs ---
        logging.info(f"Fetching historical TLEs for NORAD ID {norad_id_input}...")
        time_since_last = time.monotonic() - last_request_time
        if time_since_last < MIN_REQUEST_INTERVAL_SEC: time.sleep(MIN_REQUEST_INTERVAL_SEC - time_since_last)

        query_url = (f"{SPACETRACK_QUERY_URL}/class/tle/NORAD_CAT_ID/{norad_id_input}/"
                     f"EPOCH/{START_DATE_str}--{END_DATE_str}/orderby/EPOCH%20asc/format/tle")
        tle_response = session.get(query_url)
        tle_response.raise_for_status()
        historical_tle_data = tle_response.text.strip().splitlines()
        current_time = time.monotonic(); request_times_minute.append(current_time); request_times_hour.append(current_time); last_request_time = current_time
        logging.info(f"Fetched {len(historical_tle_data) // 2} potential historical TLE sets.")
        if not historical_tle_data: raise ValueError("No historical TLE data found.")

        # --- Process TLE Data ---
        logging.info("Processing TLE data...")
        tle_df_processed = process_tle_data(historical_tle_data)
        if tle_df_processed is None or tle_df_processed.empty: raise ValueError("TLE processing resulted in no valid data.")
        logging.info(f"Processed TLEs into DataFrame with {len(tle_df_processed)} rows.")

        # --- Smooth and Resample ---
        logging.info(f"Smoothing and resampling to {N_RESAMPLED_POINTS} points...")
        resampled_data_df = smooth_and_resample(tle_df_processed, target_points=N_RESAMPLED_POINTS, required_cols=FEATURE_COLUMNS)
        if resampled_data_df is None or resampled_data_df.shape[0] != N_RESAMPLED_POINTS: raise ValueError(f"Resampling failed.")
        logging.info("Smoothing and resampling complete.")
        input_data_orig = resampled_data_df.values.astype(np.float32)

        # --- Load Scalers ---
        logging.info("Loading scalers...")
        x_scaler = joblib.load(X_SCALER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
        logging.info("Scalers loaded successfully.")

        # --- Scale Input Data ---
        input_data_scaled = x_scaler.transform(input_data_orig)
        input_data_scaled_batch = input_data_scaled.reshape(1, N_RESAMPLED_POINTS, NUM_FEATURES)

        # --- Load AMR Ensemble Models ---
        logging.info("Loading AMR ensemble models...")
        df_summary = pd.read_csv(SUMMARY_FILE_BEST)
        df_summary = df_summary[df_summary['status'] == 'Success'].copy()
        top_models_df = df_summary.sort_values(by=['test_rmse', 'test_mae', 'test_r2'], ascending=[True, True, False]).head(N_ENSEMBLE_MODELS)
        top_model_ids = top_models_df['combination_id'].tolist()
        ensemble_models = []
        for model_id in top_model_ids:
            model_path = os.path.join(MODELS_DIR_BEST, model_id, 'best_model.keras')
            if os.path.exists(model_path): ensemble_models.append(models.load_model(model_path))
            else: logging.warning(f"Ensemble model not found: {model_path}")
        if not ensemble_models: raise ValueError("Failed to load any AMR ensemble models.")
        logging.info(f"Loaded {len(ensemble_models)} models for AMR ensemble.")

        # --- Predict AMR ---
        logging.info("Predicting AMR using ensemble...")
        ensemble_preds_log_scaled = [model.predict(input_data_scaled_batch, verbose=0).flatten() for model in ensemble_models]
        avg_pred_log_scaled = np.mean(ensemble_preds_log_scaled, axis=0)
        avg_pred_log = y_scaler.inverse_transform(avg_pred_log_scaled.reshape(-1, 1)).flatten()
        predicted_amr_orig = np.expm1(avg_pred_log)[0]
        predicted_amr_orig = max(predicted_amr_orig, 0)
        predicted_amr_final = predicted_amr_orig # Store for final printout
        logging.info(f"--> Predicted AMR value (logged): {predicted_amr_final:.6f}") # Keep log

        # --- Load Forecasting Model ---
        logging.info("Loading parameter forecasting model...")
        if os.path.exists(FORECAST_MODEL_PATH):
            forecast_model = models.load_model(FORECAST_MODEL_PATH)
            logging.info("Parameter forecasting model loaded.")
        else:
            logging.warning(f"Forecast model not found at {FORECAST_MODEL_PATH}. Skipping forecasting.")
            forecast_model = None

        # --- Forecast Parameters ---
        if forecast_model:
            logging.info("Generating parameter forecast...")
            forecast_input_sequence = input_data_scaled[-N_FORECAST_INPUT_STEPS:, :].reshape(1, N_FORECAST_INPUT_STEPS, NUM_FEATURES)
            forecast_steps_scaled = []
            for _ in range(N_FORECAST_STEPS):
                next_step_pred_scaled = forecast_model.predict(forecast_input_sequence, verbose=0)
                forecast_steps_scaled.append(next_step_pred_scaled[0])
                next_step_pred_scaled_reshaped = next_step_pred_scaled.reshape(1, 1, NUM_FEATURES)
                forecast_input_sequence = np.concatenate([forecast_input_sequence[:, 1:, :], next_step_pred_scaled_reshaped], axis=1)

            forecast_array_scaled = np.array(forecast_steps_scaled)
            forecast_array_orig = x_scaler.inverse_transform(forecast_array_scaled)
            logging.info(f"Generated parameter forecast (original scale), shape: {forecast_array_orig.shape}")
            forecast_save_path = os.path.join(PREDICTION_OUTPUT_DIR, f'forecast_parameters_{object_id}.npy')
            np.save(forecast_save_path, forecast_array_orig)
            logging.info(f"Saved parameter forecast array to {forecast_save_path}")
            plot_parameter_forecasts(historical_data_orig=input_data_orig, forecast_data_orig=forecast_array_orig, feature_names=FEATURE_COLUMNS, output_dir=PREDICTION_OUTPUT_DIR, obj_id=object_id)
        else:
            logging.info("Skipping parameter forecasting.")

    # --- Catch specific errors and general errors ---
    except FileNotFoundError as e:
         logging.error(f"File not found error: {e}. Check paths for models, scalers, or config file.")
    except ValueError as e: # Catch specific errors raised in the code
         logging.error(f"Data processing, validation, or model loading error: {e}")
    except requests.exceptions.RequestException as e:
        logging.error(f"SpaceTrack API request failed: {e}")
    except Exception as e: # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        # --- Logout ---
        if logged_in and session:
            logging.info("Logging out from Space-Track...")
            try:
                session.get(SPACETRACK_LOGOUT_URL)
            except requests.exceptions.RequestException:
                logging.warning("Warning: Error during logout.")
            finally:
                session.close()
        elif session:
            session.close()

        # --- Final Printout of Predicted AMR ---
        print("-" * 30)
        if predicted_amr_final is not None:
            print(f"Final Predicted AMR for NORAD ID {object_id}: {predicted_amr_final:.6f}")
        else:
            print(f"AMR Prediction for NORAD ID {object_id} could not be completed due to errors.")
        print("-" * 30)
        logging.info("\n--- Live Prediction Script Finished ---")