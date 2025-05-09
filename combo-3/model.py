import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, BatchNormalization, Concatenate, Reshape, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import glob
import matplotlib.pyplot as plt

DECAY_DATA_DIR = 'decay-analysis'
MERGED_DATA_FILE = 'merged_output.csv'
OUTPUT_DIR = 'model_output'
MODEL_WEIGHTS = 'amr_prediction_model.keras'

os.makedirs(OUTPUT_DIR, exist_ok=True)

merged_df = pd.read_csv(MERGED_DATA_FILE)
merged_df = merged_df[(merged_df['amr'] >= 0.001) & (merged_df['amr'] <= 0.009)]

valid_norads = set(merged_df['norad'].astype(str).tolist())

all_files = glob.glob(os.path.join(DECAY_DATA_DIR, '*.csv'))

input_sequences = []
norad_ids = []

sequence_cols = ['semi_major_axis', 'bstar', 'eccentricity', 'mean_motion', 'apogee_altitude']
seq_length = 100

for file in all_files:
    norad_id = os.path.basename(file).split('.')[0]
    
    if norad_id in valid_norads:
        try:
            df = pd.read_csv(file)
            
            if len(df) == seq_length and all(col in df.columns for col in sequence_cols):
                sequence_data = []
                
                for col in sequence_cols:
                    seq = df[col].values
                    sequence_data.append(seq)
                
                input_sequences.append(sequence_data)
                norad_ids.append(norad_id)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

print(f"Loaded {len(input_sequences)} valid sequences")

if not input_sequences:
    raise ValueError("No valid sequences found")

input_sequences = np.array(input_sequences)
norad_ids = np.array(norad_ids)

target_values = []
for norad in norad_ids:
    amr = merged_df[merged_df['norad'].astype(str) == norad]['amr'].values[0]
    target_values.append(amr)

target_values = np.array(target_values)

X_train, X_val, y_train, y_val, norad_train, norad_val = train_test_split(
    input_sequences, target_values, norad_ids, test_size=0.2, random_state=42
)

feature_scalers = []
for i in range(len(sequence_cols)):
    if sequence_cols[i] == 'bstar':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
    
    flat_features = X_train[:, i, :].flatten().reshape(-1, 1)
    scaler.fit(flat_features)
    feature_scalers.append(scaler)

for i in range(len(sequence_cols)):
    for j in range(len(X_train)):
        X_train[j, i, :] = feature_scalers[i].transform(X_train[j, i, :].reshape(-1, 1)).flatten()
    
    for j in range(len(X_val)):
        X_val[j, i, :] = feature_scalers[i].transform(X_val[j, i, :].reshape(-1, 1)).flatten()

y_train_log = np.log10(y_train).reshape(-1, 1)
y_val_log = np.log10(y_val).reshape(-1, 1)

target_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = target_scaler.fit_transform(y_train_log).flatten()
y_val_scaled = target_scaler.transform(y_val_log).flatten()

def build_model(seq_length, num_features):
    inputs = []
    encoded_features = []
    
    for i in range(num_features):
        input_seq = Input(shape=(seq_length,), name=f'input_{i}')
        inputs.append(input_seq)
        
        reshaped = Reshape((seq_length, 1))(input_seq)
        
        lstm1 = Bidirectional(LSTM(64, return_sequences=True))(reshaped)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(0.2)(lstm1)
        
        lstm2 = Bidirectional(LSTM(32))(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(0.2)(lstm2)
        
        dense1 = Dense(16, activation='relu')(lstm2)
        encoded_features.append(dense1)
    
    concat = Concatenate()(encoded_features)
    
    x = Dense(32, activation='relu')(concat)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    
    output = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

model = build_model(seq_length, len(sequence_cols))

input_data_train = [X_train[:, i, :] for i in range(len(sequence_cols))]
input_data_val = [X_val[:, i, :] for i in range(len(sequence_cols))]

checkpoint = ModelCheckpoint(
    os.path.join(OUTPUT_DIR, MODEL_WEIGHTS),
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    input_data_train, 
    y_train_scaled,
    validation_data=(input_data_val, y_val_scaled),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1
)

best_model = load_model(os.path.join(OUTPUT_DIR, MODEL_WEIGHTS))

y_pred_scaled = best_model.predict(input_data_val)
y_pred_log = target_scaler.inverse_transform(y_pred_scaled)
y_pred = np.power(10, y_pred_log)

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.7)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
plt.xlabel('True AMR')
plt.ylabel('Predicted AMR')
plt.title('AMR Prediction Performance')
plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_performance.png'))

results_df = pd.DataFrame({
    'norad': norad_val,
    'true_amr': y_val,
    'predicted_amr': y_pred.flatten()
})
results_df.to_csv(os.path.join(OUTPUT_DIR, 'validation_results.csv'), index=False)

input_feature_names = []
for col in sequence_cols:
    with open(os.path.join(OUTPUT_DIR, f'{col}_scaler.pkl'), 'wb') as f:
        import pickle
        pickle.dump(feature_scalers[sequence_cols.index(col)], f)
    input_feature_names.append(col)

with open(os.path.join(OUTPUT_DIR, 'target_scaler.pkl'), 'wb') as f:
    import pickle
    pickle.dump(target_scaler, f)

model_metadata = {
    'input_features': input_feature_names,
    'sequence_length': seq_length,
    'amr_range': [0.001, 0.009],
    'target_transform': 'log10'
}

pd.DataFrame([model_metadata]).to_json(os.path.join(OUTPUT_DIR, 'model_metadata.json'), orient='records')

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))

print(f"Model training complete. Best model saved to {os.path.join(OUTPUT_DIR, MODEL_WEIGHTS)}")