import os
import glob
import time
import pandas as pd
import numpy as np
import itertools
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Keep MinMaxScaler import just in case, but we'll use StandardScaler for y now
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import matplotlib # Ensure backend is suitable for saving plots without display
matplotlib.use('Agg') # Use a non-interactive backend suitable for scripts


# --- Configuration ---
# Paths from the Log+StandardScaler run
RESULTS_DIR_BEST_RUN = 'results-v2' # <--- Use results from the best run (LogStd)
SUMMARY_FILE_BEST = os.path.join(RESULTS_DIR_BEST_RUN, 'tuning_summary.csv')
MODELS_DIR_BEST = os.path.join(RESULTS_DIR_BEST_RUN, 'models')

# Original Data Paths (needed if re-running preprocessing)
RESAMPLED_DATA_FOLDER = '/Users/suryatejachalla/Research/reentry-amr-modeling/data/raw-fitting_copy'
AM_MAP_FILE = '/Users/suryatejachalla/Research/reentry-amr-modeling/data/merged_output.csv'

# Output directory for this script's results
ANALYSIS_OUTPUT_DIR = 'results_ensemble_forecast'
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)


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

# Ensemble configuration
N_ENSEMBLE_MODELS = 5

# Forecasting configuration
N_FORECAST_INPUT_STEPS = 60 # Number of past steps to use for predicting the next step
N_FORECAST_STEPS = 10      # Number of future steps to predict
FORECAST_MODEL_LSTM_UNITS = 128 # Units for the forecasting model's LSTM
FORECAST_MODEL_EPOCHS = 50 # Epochs for training forecast model (adjust as needed)
FORECAST_MODEL_BATCH_SIZE = 64
FORECAST_MODEL_LR = 0.001

TEST_SPLIT_RATIO = 0.2 # Must match the original split ratio
RANDOM_STATE = 42      # Must match the original random state

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- GPU Check --- (Copied from previous script)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        logging.info(f"Detected {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
    except RuntimeError as e:
        logging.error(f"GPU setup error: {e}")
else:
    logging.warning("No GPU detected. Running on CPU.")
print("-" * 20)


# --- 1. Load Data and Recreate Scaling ---
# Re-run the data loading and splitting part identical to results-v2 script
# to ensure we have the exact same X_train, X_test, y_train_orig, y_test_orig
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
# ... (rest of the loading loop exactly as in the previous script) ...
# --- Data Loading and Initial Filtering ---
logging.info("Starting data loading and initial filtering...")
try:
    amr_df_raw = pd.read_csv(AM_MAP_FILE)
    logging.info(f"Loaded raw A/M data for {len(amr_df_raw)} objects from {AM_MAP_FILE}")

    amr_df = amr_df_raw[amr_df_raw[AMR_COL_NAME] <= MAX_AMR_THRESHOLD].copy()
    logging.info(f"Filtered A/M data: Kept {len(amr_df)} objects with AMR <= {MAX_AMR_THRESHOLD}")

    # --- Additional filter: Ensure AMR > 0 for log transform ---
    amr_df = amr_df[amr_df[AMR_COL_NAME] > 0].copy()
    logging.info(f"Filtered non-positive AMR: Kept {len(amr_df)} objects with AMR > 0")

    if amr_df.empty:
         raise ValueError(f"No objects remaining after filtering AMR <= {MAX_AMR_THRESHOLD} and > 0")

    amr_map = pd.Series(amr_df[AMR_COL_NAME].values, index=amr_df[NORAD_COL_NAME].astype(str)).to_dict()
    logging.info(f"Created A/M map for {len(amr_map)} filtered objects.")

except FileNotFoundError:
    logging.error(f"Error: A/M mapping file not found at {AM_MAP_FILE}.")
    raise
except KeyError as e:
    logging.error(f"Error: Required column '{e}' not found in {AM_MAP_FILE}.")
    raise
except ValueError as e:
    logging.error(f"Error: {e}")
    raise

resampled_files = glob.glob(os.path.join(RESAMPLED_DATA_FOLDER, '*.csv'))
logging.info(f"Found {len(resampled_files)} resampled data files in {RESAMPLED_DATA_FOLDER}")

X_sequences = []
y_targets = []
processed_norad_ids = []
skipped_counts = {'amr': 0, 'shape': 0, 'cols': 0, 'nan': 0, 'bstar': 0, 'error': 0, 'amr_nonpos': 0}
start_load_time = time.time()

for i, filepath in enumerate(resampled_files):
    norad_id = os.path.basename(filepath).split('.')[0]

    if norad_id not in amr_map:
        skipped_counts['amr'] += 1 # Counts AMR not found or outside initial thresholds
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
             logging.warning(f"BSTAR column '{BSTAR_COL_NAME}' not found for NORAD {norad_id}, skipping BSTAR filter.")
             skipped_counts['cols'] += 1
             continue

        sequence_data = df[FEATURE_COLUMNS].values
        if not np.issubdtype(sequence_data.dtype, np.number):
             sequence_data = sequence_data.astype(np.float32)
        if np.isnan(sequence_data).any():
            skipped_counts['nan'] += 1
            continue

        # Final check: Ensure target AMR is > 0 (already done when creating amr_map, this is redundant but safe)
        target_amr = amr_map[norad_id]
        if target_amr <= 0:
             skipped_counts['amr_nonpos'] += 1 # Explicitly count non-positive AMR skip here
             continue

        X_sequences.append(sequence_data)
        y_targets.append(target_amr)
        processed_norad_ids.append(norad_id)

    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}")
        skipped_counts['error'] += 1

    # Optional: Add progress printout if loading takes long
    # if (i + 1) % 500 == 0:
    #     logging.info(f"  Processed {i+1}/{len(resampled_files)} files...")

loading_time = time.time() - start_load_time
logging.info(f"Finished loading data in {loading_time:.2f} seconds.")
logging.info(f"Successfully loaded sequences for {len(X_sequences)} objects.")
logging.info(f"Skipped counts: {skipped_counts}")

if not X_sequences:
     logging.error("No valid data loaded after filtering. Exiting.")
     exit()

X = np.array(X_sequences, dtype=np.float32)
y = np.array(y_targets, dtype=np.float32) # y is original scale here

# --- Train/Test Split ---
X_train, X_test, y_train_orig, y_test_orig, ids_train, ids_test = train_test_split(
    X, y, processed_norad_ids,
    test_size=TEST_SPLIT_RATIO,
    random_state=RANDOM_STATE,
    shuffle=True
)

# --- Scaling ---
# Scale X using StandardScaler
x_scaler = StandardScaler()
x_train_reshaped = X_train.reshape(-1, NUM_FEATURES)
x_scaler.fit(x_train_reshaped)
X_train_scaled = x_scaler.transform(x_train_reshaped).reshape(X_train.shape)
X_test_scaled = x_scaler.transform(X_test.reshape(-1, NUM_FEATURES)).reshape(X_test.shape)
logging.info("X features scaled using StandardScaler.")

# Scale y using log1p + StandardScaler
y_train_log = np.log1p(y_train_orig)
y_scaler = StandardScaler() # Scaler for log-transformed y
y_train_log_scaled = y_scaler.fit_transform(y_train_log.reshape(-1, 1)).flatten() # Needed for loading models, not retraining
logging.info("y target transformer (log1p + StandardScaler) setup complete.")

# --- 2. AMR Ensembling ---
logging.info("\n--- Starting AMR Ensemble Prediction ---")

# Load summary file to get top models
try:
    df_summary = pd.read_csv(SUMMARY_FILE_BEST)
    df_summary = df_summary[df_summary['status'] == 'Success'].copy()
    # Sort by RMSE (ascending), then MAE (ascending), then R2 (descending)
    top_models_df = df_summary.sort_values(
        by=['test_rmse', 'test_mae', 'test_r2'],
        ascending=[True, True, False]
    ).head(N_ENSEMBLE_MODELS)
    top_model_ids = top_models_df['combination_id'].tolist()
    logging.info(f"Identified Top {N_ENSEMBLE_MODELS} models for ensembling: {top_model_ids}")
except Exception as e:
    logging.error(f"Could not load or process summary file {SUMMARY_FILE_BEST}: {e}. Exiting.")
    exit()

# Load models and predict
ensemble_predictions_log_scaled = []
loaded_models_count = 0
for model_id in top_model_ids:
    model_path = os.path.join(MODELS_DIR_BEST, model_id, 'best_model.keras')
    if os.path.exists(model_path):
        try:
            logging.info(f"Loading model: {model_id}")
            model = models.load_model(model_path)
            pred_log_scaled = model.predict(X_test_scaled).flatten()
            ensemble_predictions_log_scaled.append(pred_log_scaled)
            loaded_models_count += 1
        except Exception as e:
            logging.error(f"Error loading or predicting with model {model_id}: {e}")
    else:
        logging.warning(f"Model file not found for {model_id} at {model_path}. Skipping.")

if loaded_models_count < 1:
    logging.error("No models could be loaded for the ensemble. Exiting.")
    exit()
elif loaded_models_count < N_ENSEMBLE_MODELS:
     logging.warning(f"Only loaded {loaded_models_count}/{N_ENSEMBLE_MODELS} models for ensemble.")

# Average predictions (on the log-standardized scale)
avg_pred_log_scaled = np.mean(ensemble_predictions_log_scaled, axis=0)

# Inverse transform the average prediction
# Step 1: Inverse StandardScaler
avg_pred_log = y_scaler.inverse_transform(avg_pred_log_scaled.reshape(-1, 1)).flatten()
# Step 2: Inverse log1p
y_pred_ensemble_orig = np.expm1(avg_pred_log)
# Step 3: Ensure non-negative
y_pred_ensemble_orig = np.maximum(y_pred_ensemble_orig, 0)

# Evaluate ensemble
logging.info("\n--- Ensemble Performance Metrics (Original Scale) ---")
ens_mae = mean_absolute_error(y_test_orig, y_pred_ensemble_orig)
ens_mse = mean_squared_error(y_test_orig, y_pred_ensemble_orig)
ens_rmse = np.sqrt(ens_mse)
try:
    ens_r2 = r2_score(y_test_orig, y_pred_ensemble_orig)
except ValueError:
    ens_r2 = np.nan
logging.info(f"Ensemble MAE:  {ens_mae:.5f}")
logging.info(f"Ensemble RMSE: {ens_rmse:.5f}")
logging.info(f"Ensemble R2:   {ens_r2:.5f}")

# Optional: Save ensemble predictions
# np.save(os.path.join(ANALYSIS_OUTPUT_DIR, 'ensemble_predictions_orig.npy'), y_pred_ensemble_orig)
# pd.DataFrame({'norad_id': ids_test, 'amr_actual': y_test_orig, 'amr_ensemble_pred': y_pred_ensemble_orig}).to_csv(os.path.join(ANALYSIS_OUTPUT_DIR, 'ensemble_predictions.csv'), index=False)


# --- 3. Parameter Forecasting Model ---
logging.info("\n--- Setting up Parameter Forecasting ---")

if N_INTERVALS <= N_FORECAST_INPUT_STEPS:
     logging.error(f"N_INTERVALS ({N_INTERVALS}) must be greater than N_FORECAST_INPUT_STEPS ({N_FORECAST_INPUT_STEPS})")
     exit()

# Prepare training data using sliding windows on X_train_scaled
logging.info(f"Preparing sliding window data using input steps: {N_FORECAST_INPUT_STEPS}")
X_train_fs, y_train_fs = [], []
num_train_samples = X_train_scaled.shape[0]
for i in range(num_train_samples):
    for t in range(N_INTERVALS - N_FORECAST_INPUT_STEPS):
        input_window = X_train_scaled[i, t : t + N_FORECAST_INPUT_STEPS, :]
        target_step = X_train_scaled[i, t + N_FORECAST_INPUT_STEPS, :] # Target is the next step (already scaled)
        X_train_fs.append(input_window)
        y_train_fs.append(target_step)

X_train_fs = np.array(X_train_fs, dtype=np.float32)
y_train_fs = np.array(y_train_fs, dtype=np.float32)

logging.info(f"Forecasting training data shape: X={X_train_fs.shape}, y={y_train_fs.shape}") # y should be (samples, NUM_FEATURES)

# Build forecasting model
def build_forecasting_model(input_shape, lstm_units, num_features):
    model = models.Sequential(name='parameter_forecaster')
    model.add(layers.Input(shape=input_shape, name='input_sequence'))
    # Use return_sequences=False as we only need the final state to predict the next step
    model.add(layers.LSTM(lstm_units, return_sequences=False, name='lstm_encoder'))
    # Output layer predicts all features for the *single* next time step
    model.add(layers.Dense(num_features, activation='linear', name='feature_output'))
    return model

tf.keras.backend.clear_session() # Clear session before building new model

forecast_input_shape = (N_FORECAST_INPUT_STEPS, NUM_FEATURES)
forecast_model = build_forecasting_model(forecast_input_shape, FORECAST_MODEL_LSTM_UNITS, NUM_FEATURES)
forecast_model.summary(print_fn=logging.info)

# Compile forecasting model
forecast_optimizer = optimizers.Adam(learning_rate=FORECAST_MODEL_LR)
# Use MSE loss for feature regression
forecast_model.compile(optimizer=forecast_optimizer, loss='mean_squared_error', metrics=['mae'])

# Train forecasting model
logging.info("Training the parameter forecasting model...")
forecast_history = forecast_model.fit(
    X_train_fs, y_train_fs,
    epochs=FORECAST_MODEL_EPOCHS,
    batch_size=FORECAST_MODEL_BATCH_SIZE,
    validation_split=0.1, # Use part of the windowed data for validation
    callbacks=[
        callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ],
    verbose=1
)

# Save the forecast model
forecast_model_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'parameter_forecast_model.keras')
forecast_model.save(forecast_model_path)
logging.info(f"Parameter forecasting model saved to {forecast_model_path}")

# --- 4. Generate Forecasts ---
logging.info(f"\n--- Generating {N_FORECAST_STEPS}-step Parameter Forecasts (Autoregressive) ---")

num_test_samples = X_test_scaled.shape[0]
# Array to store forecasts: (n_test_samples, n_forecast_steps, n_features)
all_forecasts_scaled = np.zeros((num_test_samples, N_FORECAST_STEPS, NUM_FEATURES))

start_forecast_gen_time = time.time()
for i in range(num_test_samples):
    # Get the last known sequence (scaled) as the initial input
    current_sequence = X_test_scaled[i, -N_FORECAST_INPUT_STEPS:, :].reshape(1, N_FORECAST_INPUT_STEPS, NUM_FEATURES)
    forecast_steps_scaled = []

    for _ in range(N_FORECAST_STEPS):
        # Predict the next step
        next_step_pred_scaled = forecast_model.predict(current_sequence, verbose=0) # Shape (1, NUM_FEATURES)
        forecast_steps_scaled.append(next_step_pred_scaled[0])

        # Update the sequence: remove the oldest step, append the prediction
        # Need to reshape prediction to (1, 1, NUM_FEATURES) for concatenation
        next_step_pred_scaled_reshaped = next_step_pred_scaled.reshape(1, 1, NUM_FEATURES)
        current_sequence = np.concatenate([current_sequence[:, 1:, :], next_step_pred_scaled_reshaped], axis=1)

    all_forecasts_scaled[i, :, :] = np.array(forecast_steps_scaled)
    if (i + 1) % 100 == 0:
         logging.info(f"  Generated forecasts for {i+1}/{num_test_samples} test samples...")


forecast_gen_time = time.time() - start_forecast_gen_time
logging.info(f"Finished generating forecasts in {forecast_gen_time:.2f}s.")
logging.info(f"Shape of scaled forecasts: {all_forecasts_scaled.shape}")

# Inverse transform forecasts using x_scaler
logging.info("Inverse transforming forecasts to original parameter scale...")
# Reshape for scaler: (samples * steps, features)
all_forecasts_scaled_reshaped = all_forecasts_scaled.reshape(-1, NUM_FEATURES)
all_forecasts_orig = x_scaler.inverse_transform(all_forecasts_scaled_reshaped)
# Reshape back: (samples, steps, features)
all_forecasts_orig = all_forecasts_orig.reshape(num_test_samples, N_FORECAST_STEPS, NUM_FEATURES)
logging.info(f"Shape of original scale forecasts: {all_forecasts_orig.shape}")

# Save the forecasts
forecasts_save_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'parameter_forecasts_orig.npy')
np.save(forecasts_save_path, all_forecasts_orig)
logging.info(f"Saved original scale forecasts to {forecasts_save_path}")
logging.warning("Accuracy of these parameter forecasts cannot be evaluated with the current dataset.")

# Optional: Plot example forecast for one object/parameter
def plot_example_forecast(X_test_orig, forecast_orig, obj_index, feature_index, feature_name, save_path):
    past_data = X_test_orig[obj_index, :, feature_index]
    future_forecast = forecast_orig[obj_index, :, feature_index]
    past_time = np.arange(N_INTERVALS)
    future_time = np.arange(N_INTERVALS, N_INTERVALS + N_FORECAST_STEPS)

    plt.figure(figsize=(12, 6))
    plt.plot(past_time, past_data, 'bo-', label='Historical Data')
    plt.plot(future_time, future_forecast, 'ro--', label='Forecasted Data')
    plt.title(f'Parameter Forecast Example: Object Index {obj_index} - {feature_name}')
    plt.xlabel('Time Step')
    plt.ylabel('Parameter Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Inverse transform X_test to original scale for plotting history
X_test_orig = x_scaler.inverse_transform(X_test_scaled.reshape(-1, NUM_FEATURES)).reshape(X_test.shape)

example_obj_index = 0
example_feature_index = PARAMETERS_TO_USE.index('perigee_alt') # Example: Perigee Altitude
example_feature_name = 'Perigee Altitude'
example_plot_path = os.path.join(ANALYSIS_OUTPUT_DIR, f'forecast_example_obj{example_obj_index}_{PARAMETERS_TO_USE[example_feature_index]}.png')
plot_example_forecast(X_test_orig, all_forecasts_orig, example_obj_index, example_feature_index, example_feature_name, example_plot_path)
logging.info(f"Saved example forecast plot to {example_plot_path}")


logging.info("\n--- Script Finished ---")