import os
import glob
import time
import pandas as pd
import numpy as np
import joblib # For saving scalers
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration (MUST MATCH results-v2 script EXACTLY) ---
RESULTS_DIR = 'results-v2' # The directory where models ARE saved, and where scalers WILL BE saved.
RESAMPLED_DATA_FOLDER = '/Users/suryatejachalla/Research/reentry-amr-modeling/data/raw-fitting_copy'
AM_MAP_FILE = '/Users/suryatejachalla/Research/reentry-amr-modeling/data/merged_output.csv'

NORAD_COL_NAME = 'norad'
AMR_COL_NAME = 'amr'
BSTAR_COL_NAME = 'bstar'

N_INTERVALS = 90
PARAMETERS_TO_USE = [
    'mean_motion', 'eccentricity', 'inclination', 'bstar',
    'mean_motion_deriv', 'semi_major_axis', 'orbital_period', 'perigee_alt'
]
FEATURE_COLUMNS = PARAMETERS_TO_USE
NUM_FEATURES = len(FEATURE_COLUMNS)

MAX_AMR_THRESHOLD = 0.2
BSTAR_STD_THRESHOLD = 1e-8

TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42 # Crucial: MUST be the same as the training run

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Make sure results dir exists ---
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Data Loading and Initial Filtering (Copied from results-v2 script) ---
logging.info("Starting data loading and initial filtering...")
try:
    amr_df_raw = pd.read_csv(AM_MAP_FILE)
    amr_df = amr_df_raw[amr_df_raw[AMR_COL_NAME] <= MAX_AMR_THRESHOLD].copy()
    amr_df = amr_df[amr_df[AMR_COL_NAME] > 0].copy() # Ensure positive AMR
    if amr_df.empty: raise ValueError("No objects remaining after filtering")
    amr_map = pd.Series(amr_df[AMR_COL_NAME].values, index=amr_df[NORAD_COL_NAME].astype(str)).to_dict()
except Exception as e:
    logging.error(f"Error loading AMR map: {e}")
    raise

resampled_files = glob.glob(os.path.join(RESAMPLED_DATA_FOLDER, '*.csv'))
X_sequences, y_targets, processed_norad_ids = [], [], []
skipped_counts = {'amr': 0, 'shape': 0, 'cols': 0, 'nan': 0, 'bstar': 0, 'error': 0, 'amr_nonpos': 0}
start_load_time = time.time()

# --- (Identical Loading Loop as before) ---
for i, filepath in enumerate(resampled_files):
    norad_id = os.path.basename(filepath).split('.')[0]
    if norad_id not in amr_map:
        skipped_counts['amr'] += 1
        continue
    try:
        df = pd.read_csv(filepath, low_memory=False)
        if df.shape[0] != N_INTERVALS:
            skipped_counts['shape'] += 1
            continue
        if not all(col in df.columns for col in FEATURE_COLUMNS):
            skipped_counts['cols'] += 1
            continue
        if BSTAR_COL_NAME in df.columns:
            bstar_series = pd.to_numeric(df[BSTAR_COL_NAME], errors='coerce')
            if bstar_series.isnull().any():
                 skipped_counts['nan'] += 1
                 continue
            bstar_std = bstar_series.std()
            if bstar_std < BSTAR_STD_THRESHOLD:
                skipped_counts['bstar'] += 1
                continue
        else:
             skipped_counts['cols'] += 1
             continue
        sequence_data = df[FEATURE_COLUMNS].values
        if not np.issubdtype(sequence_data.dtype, np.number):
             sequence_data = sequence_data.astype(np.float32)
        if np.isnan(sequence_data).any():
            skipped_counts['nan'] += 1
            continue
        target_amr = amr_map[norad_id]
        if target_amr <= 0:
             skipped_counts['amr_nonpos'] += 1
             continue
        X_sequences.append(sequence_data)
        y_targets.append(target_amr)
        processed_norad_ids.append(norad_id)
    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}")
        skipped_counts['error'] += 1
# --- (End of Loading Loop) ---

loading_time = time.time() - start_load_time
logging.info(f"Finished loading data in {loading_time:.2f} seconds.")
logging.info(f"Successfully loaded sequences for {len(X_sequences)} objects.")
logging.info(f"Skipped counts: {skipped_counts}")

if not X_sequences:
     logging.error("No valid data loaded after filtering. Exiting.")
     exit()

X = np.array(X_sequences, dtype=np.float32)
y = np.array(y_targets, dtype=np.float32) # y is original scale here

# --- Train/Test Split (Using the SAME random_state) ---
logging.info("Splitting data into training and test sets...")
X_train, X_test, y_train_orig, y_test_orig, ids_train, ids_test = train_test_split(
    X, y, processed_norad_ids,
    test_size=TEST_SPLIT_RATIO,
    random_state=RANDOM_STATE, # Ensures the same split as before
    shuffle=True
)
logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}") # Verify sizes match original run logs if possible

# --- Recreate and Fit Scalers ---
logging.info("Recreating and fitting scalers...")

# Fit x_scaler
x_scaler = StandardScaler()
x_train_reshaped = X_train.reshape(-1, NUM_FEATURES)
x_scaler.fit(x_train_reshaped) # Fit only on the training data
logging.info("x_scaler fitted.")

# Fit y_scaler (Log1p + StandardScaler)
y_train_log = np.log1p(y_train_orig) # Log transform
y_scaler = StandardScaler()
y_scaler.fit(y_train_log.reshape(-1, 1)) # Fit only on the log-transformed training data
logging.info("y_scaler (for log-transformed data) fitted.")

# --- Save Scalers ---
X_SCALER_PATH = os.path.join(RESULTS_DIR, 'x_scaler.joblib')
Y_SCALER_PATH = os.path.join(RESULTS_DIR, 'y_scaler.joblib')

try:
    joblib.dump(x_scaler, X_SCALER_PATH)
    logging.info(f"Saved x_scaler to {X_SCALER_PATH}")
    joblib.dump(y_scaler, Y_SCALER_PATH)
    logging.info(f"Saved y_scaler to {Y_SCALER_PATH}")
except Exception as e:
    logging.error(f"Error saving scalers: {e}")

logging.info("\n--- Scaler Recreation Script Finished ---")