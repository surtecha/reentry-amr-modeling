import os
import configparser
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import math
import pickle
import json
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings('ignore')

GM = 398600441800000.0
GM13 = GM ** (1.0/3.0)
MRAD = 6378.137
PI = math.pi
TPI86 = 2.0 * PI / 86400.0

MODEL_DIR = 'model_output'
TEMP_DIR = 'temp_data'
OUTPUT_DIR = 'prediction_results'

# Number of bootstrap samples for uncertainty estimation
BOOTSTRAP_SAMPLES = 10

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_config():
    config = configparser.ConfigParser()
    config.read('config.ini')

    if 'spacetrack' not in config:
        raise ValueError("Missing 'spacetrack' section in config.ini")

    if 'username' not in config['spacetrack'] or 'password' not in config['spacetrack']:
        raise ValueError("Missing username or password in config.ini")

    return config['spacetrack']['username'], config['spacetrack']['password']

def login_to_spacetrack(username, password):
    session = requests.Session()
    login_url = "https://www.space-track.org/ajaxauth/login"
    credentials = {"identity": username, "password": password}

    response = session.post(login_url, data=credentials)

    if response.status_code != 200:
        raise ConnectionError(f"Failed to login to Space-Track: {response.status_code}")

    return session

def fetch_tle_data(session, norad_id, months=6):
    print(f"Fetching TLE data for NORAD ID {norad_id} (6 months from latest TLE)...")

    latest_query_url = (
        f"https://www.space-track.org/basicspacedata/query/class/tle/"
        f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH desc/limit/1/format/tle"
    )

    latest_response = session.get(latest_query_url)

    if latest_response.status_code != 200 or not latest_response.text.strip():
        raise ConnectionError(f"Failed to fetch latest TLE: {latest_response.status_code}")

    latest_tle_lines = latest_response.text.strip().split('\n')
    if len(latest_tle_lines) < 2:
        raise ValueError("Invalid TLE format returned")

    line1 = latest_tle_lines[0].strip()
    if not line1.startswith('1 '):
        raise ValueError(f"Invalid TLE format. Expected line to start with '1 ', got: {line1[:2]}")

    epoch_str = line1[18:32].strip()
    try:
        end_date = tle_epoch_to_datetime(epoch_str)
        print(f"Latest TLE epoch: {end_date}")
    except Exception as e:
        raise ValueError(f"Failed to parse epoch from TLE: {str(e)}, epoch string: '{epoch_str}'")

    start_date = end_date - timedelta(days=30*months)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Fetching TLEs from {start_str} to {end_str} (based on latest TLE epoch)")

    query_url = (
        f"https://www.space-track.org/basicspacedata/query/class/tle/"
        f"NORAD_CAT_ID/{norad_id}/EPOCH/{start_str}--{end_str}/"
        f"orderby/EPOCH/format/tle"
    )

    response = session.get(query_url)

    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch data: {response.status_code}")

    tle_data = response.text

    if latest_response.text.strip() not in tle_data:
        print("Appending latest TLE that wasn't in the date range...")
        tle_data = tle_data + "\n" + latest_response.text.strip()

    return tle_data

def tle_epoch_to_datetime(tle_epoch_str):
    year_short = int(tle_epoch_str[:2])
    day_of_year_fraction = float(tle_epoch_str[2:])

    if year_short < 57:
        year = 2000 + year_short
    else:
        year = 1900 + year_short

    start_of_year = datetime(year, 1, 1)
    delta_days = timedelta(days=(day_of_year_fraction - 1))
    epoch_datetime = start_of_year + delta_days
    return epoch_datetime

def parse_scientific_notation(field):
    field = field.strip()
    match = re.match(r'([ +-])?(\d+)([+-]\d)', field)
    if match:
        sign_char, mantissa_str, exponent_str = match.groups()
        sign = -1.0 if sign_char == '-' else 1.0
        mantissa = float(f'0.{mantissa_str}')
        exponent = int(exponent_str)
        return sign * mantissa * (10 ** exponent)
    else:
        try:
            if float(field) == 0.0:
                return 0.0
        except ValueError:
            pass
        return None

def parse_tle_data(tle_data, norad_id):
    epochs = []
    epochs_utc = []
    perigee_altitudes = []
    apogee_altitudes = []
    mean_motions = []
    mean_motion_derivatives = []
    inclinations = []
    eccentricities = []
    bstars = []
    semi_major_axes = []

    lines = tle_data.strip().split('\n')

    i = 0
    while i < len(lines) - 1:
        line1 = lines[i].strip()
        line2 = lines[i+1].strip()

        if line1.startswith('1 ') and line2.startswith('2 '):
            try:
                epoch_str = line1[18:32].strip()
                epoch_dt = tle_epoch_to_datetime(epoch_str)
                epoch_utc = epoch_dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

                first_deriv_str = line1[33:43].strip()
                bstar_str = line1[53:61].strip()

                inclination_deg = float(line2[8:16].strip())
                eccentricity_str = line2[26:33].strip()
                mean_motion_rev_day = float(line2[52:63].strip())

                eccentricity = float(f'0.{eccentricity_str}')
                first_deriv = float(first_deriv_str) * 2.0
                bstar = parse_scientific_notation(bstar_str)

                if bstar is None:
                    i += 2
                    continue

                mmoti = mean_motion_rev_day
                ecc = eccentricity

                if mmoti <= 0:
                    i += 2
                    continue

                sma = GM13 / ((TPI86 * mmoti) ** (2.0 / 3.0)) / 1000.0
                apo = sma * (1.0 + ecc) - MRAD
                per = sma * (1.0 - ecc) - MRAD

                epochs.append(epoch_dt)
                epochs_utc.append(epoch_utc)
                perigee_altitudes.append(per)
                apogee_altitudes.append(apo)
                mean_motions.append(mean_motion_rev_day)
                mean_motion_derivatives.append(first_deriv)
                inclinations.append(inclination_deg)
                eccentricities.append(eccentricity)
                bstars.append(bstar)
                semi_major_axes.append(sma)

                i += 2
            except Exception as e:
                print(f"Error parsing TLE entry: {str(e)}")
                i += 2
        else:
            i += 1

    if not epochs:
        raise ValueError("No valid TLE entries found")

    data = pd.DataFrame({
        'epoch': epochs,
        'epoch_utc': epochs_utc,
        'apogee_altitude': apogee_altitudes,
        'perigee_altitude': perigee_altitudes,
        'mean_motion': mean_motions,
        'mean_motion_derivative': mean_motion_derivatives,
        'inclination': inclinations,
        'eccentricity': eccentricities,
        'bstar': bstars,
        'semi_major_axis': semi_major_axes
    })

    data = data.sort_values(by='epoch').reset_index(drop=True)

    raw_filepath = os.path.join(TEMP_DIR, f"{norad_id}_raw.csv")
    data.to_csv(raw_filepath, index=False)

    print(f"Parsed {len(data)} TLE entries")
    return data

def adaptive_resampling(df, column, target_points=200):
    if len(df) < 4:
        return df
    
    df = df.copy()
    first_timestamp = df['epoch'].min()
    df['epoch_seconds'] = (df['epoch'] - first_timestamp).dt.total_seconds()
    
    df['weight'] = abs(df[column].diff()) + 0.1
    df['weight'].fillna(df['weight'].mean(), inplace=True)
    
    cumulative_weights = df['weight'].cumsum()
    total_weight = cumulative_weights.iloc[-1]
    
    new_cum_weights = np.linspace(0, total_weight, target_points)
    
    indices = []
    for w in new_cum_weights:
        idx = np.argmin(abs(cumulative_weights - w))
        indices.append(idx)
    
    resampled_df = df.iloc[indices].copy()
    resampled_df.sort_values('epoch_seconds', inplace=True)
    
    resampled_df = resampled_df.drop(columns=['epoch_seconds', 'weight'])
    
    return resampled_df

def process_tle_data(df, norad_id):
    print("Processing TLE data...")

    if len(df) < 10:
        raise ValueError("Not enough TLE data points (minimum 10 required)")

    df = df.sort_values('epoch').reset_index(drop=True)

    # Instead of segmenting data, just use the most recent data (last 100 points or all if less)
    n_points = min(100, len(df))
    working_df = df.iloc[-n_points:].copy().reset_index(drop=True)

    columns_to_resample = ['perigee_altitude', 'semi_major_axis', 'bstar', 'eccentricity', 'mean_motion', 'apogee_altitude']
    resampled_dfs = []
    
    for column in columns_to_resample:
        if column in working_df.columns:
            resampled_df = adaptive_resampling(working_df, column, target_points=200)
            resampled_dfs.append(resampled_df[['epoch', column]])
    
    resampled_df = resampled_dfs[0]
    for other_df in resampled_dfs[1:]:
        resampled_df = pd.merge_asof(resampled_df, other_df, on='epoch')
    
    if 'bstar' in df.columns:
        bstar_values = []
        for timestamp in resampled_df['epoch']:
            closest_idx = np.argmin(abs(working_df['epoch'] - timestamp))
            bstar_values.append(working_df['bstar'].iloc[closest_idx])
        resampled_df['bstar'] = bstar_values

    processed_filepath = os.path.join(TEMP_DIR, f"{norad_id}_processed.csv")
    resampled_df.to_csv(processed_filepath, index=False)

    plot_comparison(df, working_df, resampled_df, norad_id)

    return resampled_df

def plot_comparison(original_df, working_df, resampled_df, norad_id):
    print("Generating analysis plots...")

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    plot_cols = [('perigee_altitude', 0, 0), ('apogee_altitude', 0, 1),
                ('inclination', 1, 0), ('bstar', 1, 1),
                ('semi_major_axis', 2, 0), ('mean_motion', 2, 1)]

    for col, row, col_idx in plot_cols:
        if col in original_df.columns:
            ax = axes[row, col_idx]

            ax.plot(original_df['epoch'], original_df[col], 'o', alpha=0.3, label='Original', color='blue')

            if working_df is not None:
                ax.plot(working_df['epoch'], working_df[col], 'o', alpha=0.7, label='Working Dataset', color='green')

            if resampled_df is not None and col in resampled_df.columns:
                ax.plot(resampled_df['epoch'], resampled_df[col], '-', label='Resampled', color='red', linewidth=2)

            ax.set_title(col.replace('_', ' ').title())
            ax.legend()

    plt.suptitle(f"Orbital Parameter Analysis for NORAD ID: {norad_id}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{norad_id}_analysis.png"))
    plt.close()

def load_model_and_scalers():
    print("Loading model and scalers...")

    model_path = os.path.join(MODEL_DIR, 'amr_prediction_model.keras')
    model = load_model(model_path)

    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)[0]

    feature_names = metadata['input_features']
    seq_length = metadata['sequence_length']

    feature_scalers = []
    for feature in feature_names:
        scaler_path = os.path.join(MODEL_DIR, f'{feature}_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            feature_scalers.append(scaler)

    target_scaler_path = os.path.join(MODEL_DIR, 'target_scaler.pkl')
    with open(target_scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)

    return model, feature_names, feature_scalers, target_scaler, metadata

def prepare_model_input(df, feature_names, feature_scalers, seq_length):
    print("Preparing model input...")

    if len(df) != seq_length:
        raise ValueError(f"Expected {seq_length} data points, but got {len(df)}")

    input_sequences = []

    for i, feature in enumerate(feature_names):
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in data")

        seq = df[feature].values

        scaler, is_bstar = feature_scalers[i]
        
        if is_bstar:
            log_values = np.log(np.abs(seq) + 1e-10) * np.sign(seq)
            seq_scaled = scaler.transform(log_values.reshape(-1, 1)).flatten()
        else:
            seq_scaled = scaler.transform(seq.reshape(-1, 1)).flatten()
            
        input_sequences.append(seq_scaled)

    return input_sequences

def predict_amr_with_uncertainty(model, input_sequences, target_scaler, metadata, bootstrap_samples=BOOTSTRAP_SAMPLES):
    print(f"Predicting AMR value with uncertainty ({bootstrap_samples} bootstrap samples)...")

    predictions = []
    
    # Original prediction first
    model_input = [seq.reshape(1, -1) for seq in input_sequences]
    prediction_scaled = model.predict(model_input)
    prediction_log = target_scaler.inverse_transform(prediction_scaled)[0][0]
    prediction_amr = np.power(10, prediction_log)
    predictions.append(prediction_amr)
    
    # Bootstrap predictions
    for i in range(bootstrap_samples):
        bootstrap_inputs = []
        for seq in input_sequences:
            # Generate bootstrap sample by randomly choosing from input with replacement
            bootstrap_indices = np.random.choice(len(seq), size=len(seq), replace=True)
            bootstrap_seq = seq[bootstrap_indices]
            bootstrap_inputs.append(bootstrap_seq.reshape(1, -1))
        
        bootstrap_prediction_scaled = model.predict(bootstrap_inputs)
        bootstrap_prediction_log = target_scaler.inverse_transform(bootstrap_prediction_scaled)[0][0]
        bootstrap_prediction_amr = np.power(10, bootstrap_prediction_log)
        predictions.append(bootstrap_prediction_amr)
    
    predictions = np.array(predictions)
    
    # Calculate statistics
    mean_prediction = np.mean(predictions)
    std_deviation = np.std(predictions)
    
    # Calculate confidence intervals (for 95% confidence)
    lower_bound = np.percentile(predictions, 2.5)
    upper_bound = np.percentile(predictions, 97.5)
    
    # Calculate relative uncertainty
    relative_uncertainty = (std_deviation / mean_prediction) * 100
    
    result = {
        'amr': mean_prediction,
        'std_dev': std_deviation,
        'relative_uncertainty': relative_uncertainty,
        'ci_lower_95': lower_bound,
        'ci_upper_95': upper_bound,
        'all_predictions': predictions
    }
    
    return result

def plot_uncertainty_distribution(predictions, norad_id):
    plt.figure(figsize=(10, 6))
    
    # Create histogram of bootstrap predictions
    plt.hist(predictions['all_predictions'], bins=30, alpha=0.7, color='blue', density=True)
    
    # Add vertical lines for mean and confidence intervals
    plt.axvline(predictions['amr'], color='red', linestyle='-', linewidth=2, label=f"Mean: {predictions['amr']:.8f}")
    plt.axvline(predictions['ci_lower_95'], color='orange', linestyle='--', linewidth=2, 
                label=f"95% CI Lower: {predictions['ci_lower_95']:.8f}")
    plt.axvline(predictions['ci_upper_95'], color='orange', linestyle='--', linewidth=2,
                label=f"95% CI Upper: {predictions['ci_upper_95']:.8f}")
    
    plt.title(f'AMR Prediction Distribution for NORAD ID: {norad_id}\nRelative Uncertainty: {predictions["relative_uncertainty"]:.2f}%', fontsize=14)
    plt.xlabel('AMR Value (m²/kg)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{norad_id}_uncertainty_distribution.png"))
    plt.close()

def run_prediction(norad_id):
    try:
        username, password = parse_config()

        session = login_to_spacetrack(username, password)

        tle_data = fetch_tle_data(session, norad_id)

        df = parse_tle_data(tle_data, norad_id)

        processed_df = process_tle_data(df, norad_id)

        model, feature_names, feature_scalers, target_scaler, metadata = load_model_and_scalers()

        input_sequences = prepare_model_input(processed_df, feature_names, feature_scalers, len(processed_df))

        # Get prediction with uncertainty
        prediction_results = predict_amr_with_uncertainty(model, input_sequences, target_scaler, metadata)
        
        # Plot uncertainty distribution
        plot_uncertainty_distribution(prediction_results, norad_id)

        print("\n" + "="*70)
        print(f"AMR Prediction Results for NORAD ID: {norad_id}")
        print("="*70)
        print(f"Predicted AMR: {prediction_results['amr']:.8f} m²/kg")
        print(f"Standard Deviation: {prediction_results['std_dev']:.8f} m²/kg")
        print(f"Relative Uncertainty: {prediction_results['relative_uncertainty']:.2f}%")
        print(f"95% Confidence Interval: [{prediction_results['ci_lower_95']:.8f}, {prediction_results['ci_upper_95']:.8f}] m²/kg")
        print(f"Analysis plots saved to: {os.path.join(OUTPUT_DIR, f'{norad_id}_analysis.png')}")
        print(f"Uncertainty distribution saved to: {os.path.join(OUTPUT_DIR, f'{norad_id}_uncertainty_distribution.png')}")
        print("="*70)

        result = {
            'norad_id': norad_id,
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_amr': float(prediction_results['amr']),
            'std_deviation': float(prediction_results['std_dev']),
            'relative_uncertainty': float(prediction_results['relative_uncertainty']),
            'ci_lower_95': float(prediction_results['ci_lower_95']),
            'ci_upper_95': float(prediction_results['ci_upper_95']),
            'data_points_used': len(processed_df),
            'timespan_days': (processed_df['epoch'].max() - processed_df['epoch'].min()).days,
            'bootstrap_samples': BOOTSTRAP_SAMPLES
        }

        result_df = pd.DataFrame([result])
        result_df.to_csv(os.path.join(OUTPUT_DIR, f"{norad_id}_amr_prediction.csv"), index=False)

        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    norad_id = 6073
    run_prediction(norad_id)
