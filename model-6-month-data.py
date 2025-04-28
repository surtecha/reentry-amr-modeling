import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
# Import necessary preprocessing tools
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # For handling NaNs
import matplotlib.pyplot as plt
from scipy import stats
# Import R2 score for evaluation
from sklearn.metrics import r2_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory paths
input_dir = 'tle-nn-input'
model_dir = 'models_improved' # Use a new directory for improved model
plots_dir = 'plots_improved' # Use a new directory for improved plots

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# --- Configuration / Hyperparameters ---
# Define features and target
selected_features = ['perigee_alt', 'mean_motion', 'mean_motion_deriv', 'bstar', 'semi_major_axis']
target_variable = 'amr'
amr_threshold = 0.2
sequence_length = 180  # 180 points per object

# Model hyperparameters
lstm_units_1 = 128 # Increased units
lstm_units_2 = 64
dense_units = 64  # Increased units
dropout_rate = 0.3 # Added dropout
learning_rate = 0.001 # Initial learning rate
batch_size = 32 # Slightly increased batch size
epochs = 150 # Increased epochs slightly, rely on early stopping
early_stopping_patience = 25 # Increased patience
validation_split_ratio = 0.2

# --- Function to load and prepare data (Revised) ---
def prepare_data_improved():
    # Load AMR data
    try:
        amr_data = pd.read_csv('/Users/suryatejachalla/Research/reentry-amr-modeling/data/merged_output.csv')
    except FileNotFoundError:
        print("Error: merged_output.csv not found. Please check the path.")
        return None, None, None, None, None, None # Return None values

    # Filter based on AMR threshold
    amr_data_filtered = amr_data[amr_data[target_variable] <= amr_threshold].copy()
    print(f"Total objects found: {len(amr_data)}")
    print(f"Number of objects with {target_variable} <= {amr_threshold}: {len(amr_data_filtered)}")
    if len(amr_data_filtered) == 0:
        print("Error: No objects left after filtering by AMR. Check threshold or data.")
        return None, None, None, None, None, None

    X_all = []
    y_all = []
    object_ids_processed = []

    # Read sequence data for each valid object
    for _, row in amr_data_filtered.iterrows():
        norad_id = int(row['norad']) # Ensure integer NORAD ID
        amr_value = row[target_variable]

        file_path = os.path.join(input_dir, f"{norad_id}_resampled.csv")
        if not os.path.exists(file_path):
            print(f"Warning: File not found for NORAD ID {norad_id}. Skipping.")
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Warning: Could not read file for NORAD ID {norad_id}: {e}. Skipping.")
            continue

        # Verify sequence length
        if len(df) != sequence_length:
            print(f"Warning: NORAD ID {norad_id} has {len(df)} points (expected {sequence_length}). Skipping.")
            continue

        # Check if all selected features exist
        if not all(feature in df.columns for feature in selected_features):
            missing = [f for f in selected_features if f not in df.columns]
            print(f"Warning: Missing features {missing} for NORAD ID {norad_id}. Skipping.")
            continue

        # Extract features
        sequence_data = df[selected_features].values
        X_all.append(sequence_data)
        y_all.append(amr_value)
        object_ids_processed.append(norad_id)

    if not X_all:
        print("Error: No valid sequences found after processing files.")
        return None, None, None, None, None, None

    # Convert to numpy arrays
    X_all = np.array(X_all, dtype=np.float32) # Ensure float32 for TF
    y_all = np.array(y_all, dtype=np.float32).reshape(-1, 1) # Reshape y for scaler

    print(f"Successfully loaded {len(X_all)} sequences.")
    print(f"Shape of X_all before split: {X_all.shape}")
    print(f"Shape of y_all before split: {y_all.shape}")

    # Split data into training and validation sets FIRST
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=validation_split_ratio,
        random_state=42 # Ensure reproducible split
    )

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # --- Initialize Scalers and Imputers ---
    feature_scalers = {}
    feature_imputers = {}

    # Choose scaler/imputer based on feature characteristics
    for i, feature in enumerate(selected_features):
        if feature in ['perigee_alt', 'mean_motion', 'semi_major_axis']:
            # Assume roughly normal distribution, use mean for imputation
            feature_imputers[feature] = SimpleImputer(strategy='mean')
            feature_scalers[feature] = StandardScaler()
        elif feature in ['mean_motion_deriv', 'bstar']:
            # Potential outliers, use median for imputation
            feature_imputers[feature] = SimpleImputer(strategy='median')
            feature_scalers[feature] = RobustScaler()
        else: # Default fallback
            feature_imputers[feature] = SimpleImputer(strategy='mean')
            feature_scalers[feature] = StandardScaler()

    # Target scaler
    target_scaler = MinMaxScaler()

    # --- Fit and Transform Training Data ---
    X_train_scaled = np.zeros_like(X_train)
    for i, feature in enumerate(selected_features):
        # Reshape feature data: (n_samples * sequence_length, 1)
        feature_data_train = X_train[:, :, i].reshape(-1, 1)

        # 1. Impute NaNs/Infs (Fit and Transform on Train)
        feature_data_train = feature_imputers[feature].fit_transform(feature_data_train)

        # 2. Scale (Fit and Transform on Train)
        scaled_feature_train = feature_scalers[feature].fit_transform(feature_data_train)

        # Reshape back: (n_samples, sequence_length)
        X_train_scaled[:, :, i] = scaled_feature_train.reshape(X_train.shape[0], sequence_length)
        print(f"Processed training data for feature: {feature}")


    # --- Transform Validation Data ---
    X_val_scaled = np.zeros_like(X_val)
    for i, feature in enumerate(selected_features):
        # Reshape feature data: (n_samples * sequence_length, 1)
        feature_data_val = X_val[:, :, i].reshape(-1, 1)

        # 1. Impute NaNs/Infs (Transform only, using imputer fitted on train)
        feature_data_val = feature_imputers[feature].transform(feature_data_val)

        # 2. Scale (Transform only, using scaler fitted on train)
        scaled_feature_val = feature_scalers[feature].transform(feature_data_val)

        # Reshape back: (n_samples, sequence_length)
        X_val_scaled[:, :, i] = scaled_feature_val.reshape(X_val.shape[0], sequence_length)
        print(f"Processed validation data for feature: {feature}")


    # --- Scale Target Variable (y) ---
    # Fit on training data
    y_train_scaled = target_scaler.fit_transform(y_train)
    # Transform validation data
    y_val_scaled = target_scaler.transform(y_val)

    # Combine scalers for easy access later
    scalers = {'features': feature_scalers, 'target': target_scaler}
    imputers = {'features': feature_imputers} # Store imputers too if needed

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scalers, imputers


# --- Create the Improved Model ---
def create_improved_model(input_shape):
    model = models.Sequential()

    # Input Layer - consider masking if sequences have variable length (not needed here)
    # model.add(layers.Masking(mask_value=0., input_shape=input_shape)) # Example if padding used

    # Bidirectional LSTM layers with Dropout
    # Use return_sequences=True for all but the last LSTM layer if stacking
    model.add(layers.Bidirectional(layers.LSTM(lstm_units_1, return_sequences=True), input_shape=input_shape))
    model.add(layers.Dropout(dropout_rate)) # Dropout after LSTM

    model.add(layers.Bidirectional(layers.LSTM(lstm_units_2)))
    model.add(layers.Dropout(dropout_rate)) # Dropout after LSTM

    # Dense layers with Batch Normalization and Dropout
    model.add(layers.Dense(dense_units))
    model.add(layers.BatchNormalization()) # Normalize before activation
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate)) # Dropout after activation

    # Output layer (single neuron for regression)
    model.add(layers.Dense(1))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae']) # Mean Squared Error for regression

    model.summary() # Print model summary
    return model

# --- Train the Model (Revised Callbacks) ---
def train_model_improved(X_train, y_train, X_val, y_val):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_improved_model(input_shape)

    # Define callbacks
    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', # Monitor validation loss
        patience=early_stopping_patience,
        restore_best_weights=True, # Restore weights from the epoch with the best val_loss
        verbose=1
    )

    # Model Checkpoint (optional, as EarlyStopping restores best weights)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, 'best_model_improved.keras'), # Use .keras format
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Reduce Learning Rate on Plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # Reduce LR by a factor of 5
        patience=10,  # Reduce LR if no improvement for 10 epochs
        min_lr=1e-6, # Minimum learning rate
        verbose=1
    )

    print("\n--- Starting Model Training ---")
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint, reduce_lr], # Add reduce_lr callback
        verbose=1 # Show progress bar
    )
    print("--- Model Training Finished ---")

    # Load the best weights found by early stopping/checkpoint
    # model.load_weights(os.path.join(model_dir, 'best_model_improved.keras')) # Redundant if restore_best_weights=True

    return model, history

# --- Plot training history (No changes needed) ---
def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Validation')
    plt.title('Model Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to: {save_path}")
    plt.close()

# --- Plot predictions vs actual (Added R-squared) ---
def plot_predictions(model, X_val, y_val_scaled, scalers, save_path):
    # Get predictions (scaled)
    y_pred_scaled = model.predict(X_val)

    # Inverse transform to get original scale
    y_val_orig = scalers['target'].inverse_transform(y_val_scaled).flatten()
    y_pred_orig = scalers['target'].inverse_transform(y_pred_scaled).flatten()

    # Create scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y_val_orig, y_pred_orig, alpha=0.6, edgecolors='k', s=50) # Enhanced visuals

    # Plot y=x reference line
    min_val = min(min(y_val_orig), min(y_pred_orig)) * 0.95 # Adjust limits slightly
    max_val = max(max(y_val_orig), max(y_pred_orig)) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit (y=x)')

    # Calculate metrics
    correlation, _ = stats.pearsonr(y_val_orig, y_pred_orig)
    mse = np.mean((y_val_orig - y_pred_orig) ** 2)
    mae = np.mean(np.abs(y_val_orig - y_pred_orig))
    r2 = r2_score(y_val_orig, y_pred_orig) # Calculate R-squared

    # Add metrics to plot title
    plt.title(f'Predicted vs Actual AMR (Validation Set)\n'
              f'Correlation: {correlation:.4f} | R²: {r2:.4f} | MSE: {mse:.6f} | MAE: {mae:.6f}',
              fontsize=12)
    plt.xlabel('Actual AMR', fontsize=12)
    plt.ylabel('Predicted AMR', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # Ensure equal scaling for x and y axes
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend()

    plt.savefig(save_path)
    print(f"Prediction scatter plot saved to: {save_path}")
    plt.close()

    return y_val_orig, y_pred_orig

# --- Main execution ---
def main():
    print("--- Starting Data Preparation ---")
    # Prepare data using the improved function
    X_train, y_train, X_val, y_val, scalers, imputers = prepare_data_improved()

    # Check if data loading was successful
    if X_train is None:
        print("Data preparation failed. Exiting.")
        return

    print("\n--- Starting Model Creation and Training ---")
    # Create and train the improved model
    model, history = train_model_improved(X_train, y_train, X_val, y_val)

    print("\n--- Plotting Training History ---")
    history_plot_path = os.path.join(plots_dir, 'training_history_improved.png')
    plot_training_history(history, history_plot_path)

    print("\n--- Evaluating Model and Plotting Predictions ---")
    predictions_plot_path = os.path.join(plots_dir, 'prediction_scatter_improved.png')
    y_val_orig, y_pred_orig = plot_predictions(model, X_val, y_val, scalers, predictions_plot_path)

    # Optional: Print some sample predictions
    print("\n--- Sample Predictions (Actual vs Predicted) ---")
    sample_indices = np.random.choice(len(y_val_orig), min(10, len(y_val_orig)), replace=False)
    for i in sample_indices:
        print(f"Sample {i}: Actual={y_val_orig[i]:.6f}, Predicted={y_pred_orig[i]:.6f}")

    print("\n--- Script Finished Successfully ---")

if __name__ == "__main__":
    main()