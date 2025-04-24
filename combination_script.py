import os
import glob
import time
import pandas as pd
import numpy as np
import itertools
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import matplotlib # Ensure backend is suitable for saving plots without display
matplotlib.use('Agg') # Use a non-interactive backend suitable for scripts


# --- Configuration ---
RESAMPLED_DATA_FOLDER = '/Users/suryatejachalla/Research/reentry-amr-modeling/data/raw-fitting_copy'
AM_MAP_FILE = '/Users/suryatejachalla/Research/reentry-amr-modeling/data/merged_output.csv'
RESULTS_DIR = 'results_tuning_v3' # Updated directory name

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

INITIAL_LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.2
MIN_LR = 1e-6

TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# --- Hyperparameter Grid ---
HP_NUM_LSTM_LAYERS = [1, 2, 3]
HP_LSTM_UNITS = [64, 128, 256]
HP_NUM_DENSE_LAYERS = [1, 2, 3]
HP_DENSE_UNITS = [64, 128, 256]

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Directory Setup ---
PREDICTION_SCATTER_LINEAR_DIR = os.path.join(RESULTS_DIR, 'prediction_scatter_linear')
PREDICTION_SCATTER_LOGLOG_DIR = os.path.join(RESULTS_DIR, 'prediction_scatter_loglog')
LOSS_DIR = os.path.join(RESULTS_DIR, 'loss')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
os.makedirs(PREDICTION_SCATTER_LINEAR_DIR, exist_ok=True)
os.makedirs(PREDICTION_SCATTER_LOGLOG_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- GPU Check ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        logging.info(f"Detected {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
        logging.info("GPU acceleration enabled.")
    except RuntimeError as e:
        logging.error(f"GPU setup error: {e}")
else:
    logging.warning("No GPU detected. Running on CPU.")
print("-" * 20)


# --- Data Loading and Initial Filtering ---
logging.info("Starting data loading and initial filtering...")
try:
    amr_df_raw = pd.read_csv(AM_MAP_FILE)
    logging.info(f"Loaded raw A/M data for {len(amr_df_raw)} objects from {AM_MAP_FILE}")

    amr_df = amr_df_raw[amr_df_raw[AMR_COL_NAME] <= MAX_AMR_THRESHOLD].copy()
    logging.info(f"Filtered A/M data: Kept {len(amr_df)} objects with AMR <= {MAX_AMR_THRESHOLD}")

    if amr_df.empty:
         raise ValueError(f"No objects remaining after filtering AMR <= {MAX_AMR_THRESHOLD}")

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
skipped_counts = {'amr': 0, 'shape': 0, 'cols': 0, 'nan': 0, 'bstar': 0, 'error': 0}
start_load_time = time.time()

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
             logging.warning(f"BSTAR column '{BSTAR_COL_NAME}' not found for NORAD {norad_id}, skipping BSTAR filter.")
             skipped_counts['cols'] += 1
             continue

        sequence_data = df[FEATURE_COLUMNS].values
        if not np.issubdtype(sequence_data.dtype, np.number):
             sequence_data = sequence_data.astype(np.float32)
        if np.isnan(sequence_data).any():
            skipped_counts['nan'] += 1
            continue

        # Final check: ensure target AMR is non-negative for log plot compatibility later
        target_amr = amr_map[norad_id]
        if target_amr <= 0:
            skipped_counts['amr'] += 1 # Skip objects with non-positive AMR
            continue

        X_sequences.append(sequence_data)
        y_targets.append(target_amr)
        processed_norad_ids.append(norad_id)

    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}")
        skipped_counts['error'] += 1

    if (i + 1) % 500 == 0:
        logging.info(f"  Processed {i+1}/{len(resampled_files)} files...")

loading_time = time.time() - start_load_time
logging.info(f"Finished loading data in {loading_time:.2f} seconds.")
logging.info(f"Successfully loaded sequences for {len(X_sequences)} objects.")
logging.info(f"Skipped counts: {skipped_counts}")
total_skipped = sum(skipped_counts.values())
logging.info(f"Total skipped: {total_skipped}")

if not X_sequences:
     logging.error("No valid data loaded after filtering. Exiting.")
     exit()

X = np.array(X_sequences, dtype=np.float32)
y = np.array(y_targets, dtype=np.float32)

logging.info(f"Shape of X (sequences): {X.shape}")
logging.info(f"Shape of y (targets): {y.shape}")


# --- Train/Test Split ---
logging.info("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, processed_norad_ids,
    test_size=TEST_SPLIT_RATIO,
    random_state=RANDOM_STATE,
    shuffle=True
)
logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")


# --- Scaling ---
logging.info("Scaling data...")
# Scale X
x_scaler = StandardScaler()
x_train_reshaped = X_train.reshape(-1, NUM_FEATURES)
x_scaler.fit(x_train_reshaped)
x_train_scaled_reshaped = x_scaler.transform(x_train_reshaped)
X_train_scaled = x_train_scaled_reshaped.reshape(X_train.shape)

x_test_reshaped = X_test.reshape(-1, NUM_FEATURES)
x_test_scaled_reshaped = x_scaler.transform(x_test_reshaped)
X_test_scaled = x_test_scaled_reshaped.reshape(X_test.shape)

# Scale y using MinMaxScaler
y_scaler = MinMaxScaler()
y_train_reshaped = y_train.reshape(-1, 1)
y_scaler.fit(y_train_reshaped)
y_train_scaled = y_scaler.transform(y_train_reshaped).flatten()

y_test_reshaped = y_test.reshape(-1, 1)
y_test_scaled = y_scaler.transform(y_test_reshaped).flatten()
logging.info("Scaling complete using StandardScaler for X and MinMaxScaler for y.")


# --- Model Building Function ---
def build_dynamic_model(num_lstm_layers, lstm_units, num_dense_layers, dense_units, input_shape):
    model = models.Sequential(name=f"lstm_{num_lstm_layers}x{lstm_units}_dense_{num_dense_layers}x{dense_units}")
    model.add(layers.Input(shape=input_shape, name='input_sequence'))

    for i in range(num_lstm_layers):
        is_last_lstm = (i == num_lstm_layers - 1)
        model.add(layers.LSTM(lstm_units, return_sequences=not is_last_lstm, name=f'lstm_{i+1}'))

    for i in range(num_dense_layers):
        model.add(layers.Dense(dense_units, activation='relu', name=f'dense_{i+1}'))

    model.add(layers.Dense(1, activation='linear', name='amr_output'))
    return model


# --- Plotting Functions ---
def plot_training_history(history, file_path, combination_id, best_epoch):
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Training History: {combination_id}', fontsize=14)

    ax1.plot(epochs, hist['loss'], 'bo-', label='Training Loss (MSE)')
    ax1.plot(epochs, hist['val_loss'], 'ro--', label='Validation Loss (MSE)')
    if best_epoch:
        ax1.axvline(best_epoch, color='grey', linestyle=':', label=f'Best Epoch ({best_epoch})')
    ax1.set_title('Training and Validation Loss (MSE)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, hist['mae'], 'bo-', label='Training MAE')
    ax2.plot(epochs, hist['val_mae'], 'ro--', label='Validation MAE')
    if best_epoch:
         ax2.axvline(best_epoch, color='grey', linestyle=':', label=f'Best Epoch ({best_epoch})')
    ax2.set_title('Training and Validation MAE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(file_path)
    plt.close(fig)


def plot_predictions_base(y_true, y_pred, file_path, combination_id, metrics, scale_type):
    # Filter out non-positive values for log scale compatibility, and NaN predictions
    valid_indices = (y_true > 0) & (y_pred > 0) & (~np.isnan(y_pred))
    if not np.any(valid_indices):
        logging.warning(f"No valid positive points for {scale_type} plot: {combination_id}")
        # Create a blank plot or just skip
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.text(0.5, 0.5, 'No valid data points for plotting', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f"Predicted vs Actual A/M: {combination_id}\n({scale_type.capitalize()} Scale) - No Data")
        plt.savefig(file_path)
        plt.close(fig)
        return

    y_true_filt = y_true[valid_indices]
    y_pred_filt = y_pred[valid_indices]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(y_true_filt, y_pred_filt, alpha=0.6, edgecolors='k', s=50)
    ax.set_xlabel("Actual A/M Ratio")
    ax.set_ylabel("Predicted A/M Ratio")

    stats_text = (f"MAE: {metrics['mae']:.4f}\n"
                  f"RMSE: {metrics['rmse']:.4f}\n"
                  f"R²: {metrics['r2']:.4f}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    ax.grid(True, linestyle='--', alpha=0.6)

    if scale_type == 'linear':
        limit_min, limit_max = 0, 0.2
        ax.set_title(f"Predicted vs Actual A/M: {combination_id}\n(Test Set - Linear Scale)")
        ax.plot([limit_min, limit_max], [limit_min, limit_max], 'r--', lw=2, label='Ideal Fit (y=x)')
        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)
        ax.set_aspect('equal', adjustable='box')

    elif scale_type == 'loglog':
        limit_min, limit_max = 0.01, 0.2 # Define log limits
        ax.set_title(f"Predicted vs Actual A/M: {combination_id}\n(Test Set - LogLog Scale)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Ensure plot limits contain the data, adjusting if necessary
        actual_min = max(limit_min, y_true_filt.min() * 0.8) if len(y_true_filt) > 0 else limit_min
        actual_max = min(limit_max, y_true_filt.max() * 1.2) if len(y_true_filt) > 0 else limit_max
        pred_min = max(limit_min, y_pred_filt.min() * 0.8) if len(y_pred_filt) > 0 else limit_min
        pred_max = min(limit_max, y_pred_filt.max() * 1.2) if len(y_pred_filt) > 0 else limit_max

        final_min = max(actual_min, pred_min) # Ensure the diagonal covers plotted area
        final_max = min(actual_max, pred_max)

        # Use the predefined log limits primarily, but allow shrinking if data range is smaller
        plot_limit_min = limit_min
        plot_limit_max = limit_max

        ax.plot([plot_limit_min, plot_limit_max], [plot_limit_min, plot_limit_max], 'r--', lw=2, label='Ideal Fit (y=x)')
        ax.set_xlim(plot_limit_min, plot_limit_max)
        ax.set_ylim(plot_limit_min, plot_limit_max)
        # Aspect ratio 'equal' might not look right on log scale, use 'auto' or adjust manually if needed
        # ax.set_aspect('equal', adjustable='box') # Often not desired for log-log

    ax.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close(fig)


# --- Grid Search Loop ---
input_shape = (N_INTERVALS, NUM_FEATURES)
results_summary = []
hp_combinations = list(itertools.product(HP_NUM_LSTM_LAYERS, HP_LSTM_UNITS, HP_NUM_DENSE_LAYERS, HP_DENSE_UNITS))
total_combinations = len(hp_combinations)

logging.info(f"Starting Grid Search over {total_combinations} combinations...")
overall_start_time = time.time()

for i, (num_lstm, lstm_u, num_dense, dense_u) in enumerate(hp_combinations):
    combination_id = f"lstm_{num_lstm}x{lstm_u}_dense_{num_dense}x{dense_u}"
    logging.info(f"\n--- [{i+1}/{total_combinations}] Running Combination: {combination_id} ---")

    model_dir = os.path.join(MODELS_DIR, combination_id)
    os.makedirs(model_dir, exist_ok=True)
    model_file_path = os.path.join(model_dir, 'best_model.keras')
    loss_plot_path = os.path.join(LOSS_DIR, f"{combination_id}.png")
    scatter_plot_linear_path = os.path.join(PREDICTION_SCATTER_LINEAR_DIR, f"{combination_id}.png")
    scatter_plot_loglog_path = os.path.join(PREDICTION_SCATTER_LOGLOG_DIR, f"{combination_id}.png")

    tf.keras.backend.clear_session()

    try:
        # Build & Compile
        model = build_dynamic_model(num_lstm, lstm_u, num_dense, dense_u, input_shape)
        optimizer = optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        # Callbacks
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
        model_checkpoint_cb = callbacks.ModelCheckpoint(filepath=model_file_path, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=0)
        reduce_lr_cb = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=1)
        training_callbacks_list = [early_stopping_cb, model_checkpoint_cb, reduce_lr_cb]

        # Train
        logging.info(f"Training model: {combination_id}")
        start_train_time = time.time()
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=training_callbacks_list,
            verbose=0
        )
        train_time = time.time() - start_train_time
        logging.info(f"Training finished in {train_time:.2f}s.")

        # Load best model
        logging.info(f"Loading best weights from {model_file_path}")
        model = models.load_model(model_file_path)

        # Evaluate
        logging.info(f"Evaluating model: {combination_id} on Test Set")
        y_pred_scaled = model.predict(X_test_scaled).flatten()

        # Inverse transform predictions
        # Handle potential NaN/Inf from scaling very small numbers if MinMaxScaler clips them
        y_pred_scaled_reshaped = y_pred_scaled.reshape(-1, 1)
        # Replace NaNs or Infs before inverse transform if necessary, although MinMaxScaler is usually robust
        # y_pred_scaled_reshaped = np.nan_to_num(y_pred_scaled_reshaped)
        y_pred_original = y_scaler.inverse_transform(y_pred_scaled_reshaped).flatten()
        # Ensure predictions are non-negative after inverse transform
        y_pred_original = np.maximum(y_pred_original, 0) # Clip any potentially negative results

        # Calculate metrics using original scale y_test and the non-negative inverse-transformed predictions
        test_mae = mean_absolute_error(y_test, y_pred_original)
        test_mse = mean_squared_error(y_test, y_pred_original)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_pred_original)
        test_metrics = {'mae': test_mae, 'mse': test_mse, 'rmse': test_rmse, 'r2': test_r2}
        logging.info(f"Test Metrics: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R2={test_r2:.4f}")

        # Plotting
        best_epoch_index = np.argmin(history.history['val_loss'])
        best_epoch = best_epoch_index + 1
        plot_training_history(history, loss_plot_path, combination_id, best_epoch)
        plot_predictions_base(y_test, y_pred_original, scatter_plot_linear_path, combination_id, test_metrics, scale_type='linear')
        plot_predictions_base(y_test, y_pred_original, scatter_plot_loglog_path, combination_id, test_metrics, scale_type='loglog')
        logging.info(f"Saved plots to respective directories.")

        # Log results
        results_summary.append({
            'combination_id': combination_id,
            'num_lstm_layers': num_lstm,
            'lstm_units': lstm_u,
            'num_dense_layers': num_dense,
            'dense_units': dense_u,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'training_time_s': train_time,
            'best_epoch': best_epoch,
            'final_val_loss': history.history['val_loss'][best_epoch_index],
            'status': 'Success' # Added status
        })

    except Exception as e:
        logging.error(f"!!! Error occurred during combination: {combination_id} !!!")
        logging.error(f"Error details: {e}", exc_info=True)
        results_summary.append({
             'combination_id': combination_id,
             'num_lstm_layers': num_lstm,
             'lstm_units': lstm_u,
             'num_dense_layers': num_dense,
             'dense_units': dense_u,
             'test_mae': np.nan, 'test_rmse': np.nan, 'test_r2': np.nan,
             'training_time_s': np.nan, 'best_epoch': np.nan, 'final_val_loss': np.nan,
             'status': 'Failed'
        })


# --- Save Summary ---
summary_df = pd.DataFrame(results_summary)
summary_file_path = os.path.join(RESULTS_DIR, 'tuning_summary.csv')
summary_df.to_csv(summary_file_path, index=False)

overall_time = time.time() - overall_start_time
logging.info(f"\n--- Grid Search Complete ---")
logging.info(f"Total combinations processed: {len(results_summary)} / {total_combinations}")
logging.info(f"Total time: {overall_time / 60:.2f} minutes")
logging.info(f"Results summary saved to: {summary_file_path}")
logging.info(f"Models and plots saved in: {RESULTS_DIR}")