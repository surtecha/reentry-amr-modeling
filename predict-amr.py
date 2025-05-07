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
import time
from scipy import interpolate
from tensorflow.keras.models import load_model
import warnings
import argparse

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants for TLE processing
GM = 398600441800000.0
GM13 = GM ** (1.0/3.0)
MRAD = 6378.137
PI = math.pi
TPI86 = 2.0 * PI / 86400.0

# Directories
MODEL_DIR = 'model_output'
TEMP_DIR = 'temp_data'
OUTPUT_DIR = 'prediction_results'

# Create necessary directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_config():
    """Parse the config.ini file for Space-Track credentials"""
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    if 'spacetrack' not in config:
        raise ValueError("Missing 'spacetrack' section in config.ini")
    
    if 'username' not in config['spacetrack'] or 'password' not in config['spacetrack']:
        raise ValueError("Missing username or password in config.ini")
    
    return config['spacetrack']['username'], config['spacetrack']['password']

def login_to_spacetrack(username, password):
    """Login to Space-Track and return a session"""
    session = requests.Session()
    login_url = "https://www.space-track.org/ajaxauth/login"
    credentials = {"identity": username, "password": password}
    
    response = session.post(login_url, data=credentials)
    
    if response.status_code != 200:
        raise ConnectionError(f"Failed to login to Space-Track: {response.status_code}")
    
    return session

def fetch_tle_data(session, norad_id, months=6):
    """Fetch TLE data for a specific NORAD ID for the last N months"""
    print(f"Fetching TLE data for NORAD ID {norad_id} (last {months} months)...")
    
    # Calculate the date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30*months)
    
    # Format dates for the query
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Build the query URL
    query_url = (
        f"https://www.space-track.org/basicspacedata/query/class/tle/"
        f"NORAD_CAT_ID/{norad_id}/EPOCH/{start_str}--{end_str}/"
        f"orderby/EPOCH/format/tle"
    )
    
    response = session.get(query_url)
    
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch data: {response.status_code}")
    
    return response.text

def tle_epoch_to_datetime(tle_epoch_str):
    """Convert TLE epoch string to datetime object"""
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
    """Parse scientific notation fields in TLE"""
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
    """Parse TLE data into a pandas DataFrame"""
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
    
    # Save raw data
    raw_filepath = os.path.join(TEMP_DIR, f"{norad_id}_raw.csv")
    data.to_csv(raw_filepath, index=False)
    
    print(f"Parsed {len(data)} TLE entries")
    return data

def detect_maneuver_or_change(df, column='perigee_altitude', threshold=10):
    """
    Detect significant changes in orbital parameters that might indicate a maneuver or decay start.
    Returns indices where significant changes occur.
    """
    changes = []
    if len(df) < 3:
        return changes
    
    # Calculate rate of change
    df['gradient'] = df[column].diff() / df['epoch'].diff().dt.total_seconds()
    
    # Calculate standard deviation of gradient
    std_grad = df['gradient'].std()
    if pd.isna(std_grad) or std_grad == 0:
        return changes
    
    # Identify points where gradient exceeds threshold * standard deviation
    for i in range(1, len(df)-1):
        if abs(df['gradient'].iloc[i]) > threshold * std_grad:
            changes.append(i)
    
    # Remove the temporary column
    df.drop('gradient', axis=1, inplace=True)
    
    return changes

def segment_data(df):
    """
    Segment the data based on detected maneuvers or changes.
    Returns a list of dataframes, each representing a segment.
    """
    # Detect changes in perigee altitude (most sensitive to decay)
    change_indices = detect_maneuver_or_change(df, 'perigee_altitude')
    
    # If no significant changes, try apogee
    if not change_indices:
        change_indices = detect_maneuver_or_change(df, 'apogee_altitude')
    
    # If still no changes, try mean motion
    if not change_indices:
        change_indices = detect_maneuver_or_change(df, 'mean_motion', threshold=5)
    
    # Sort and deduplicate change points
    change_indices = sorted(set(change_indices))
    
    # Create segments
    segments = []
    start_idx = 0
    
    for idx in change_indices:
        if idx - start_idx > 2:  # Ensure segment has at least 3 points
            segments.append(df.iloc[start_idx:idx+1])
        start_idx = idx + 1
    
    # Add the final segment
    if len(df) - start_idx > 2:
        segments.append(df.iloc[start_idx:])
    
    return segments

def select_decay_segment(segments):
    """
    Select the segment that most likely represents orbital decay.
    Typically the last segment with decreasing altitude or increasing mean motion.
    """
    if not segments:
        return None
    
    # First check if the last segment shows decreasing altitude
    last_segment = segments[-1]
    if len(last_segment) >= 5:
        # Check if there's a clear downward trend in perigee
        first_perigee = last_segment['perigee_altitude'].iloc[0]
        last_perigee = last_segment['perigee_altitude'].iloc[-1]
        
        if last_perigee < first_perigee and (first_perigee - last_perigee) / first_perigee > 0.05:
            return last_segment
    
    # If the last segment doesn't show clear decay, check all segments
    for segment in reversed(segments):  # Check from the last segment backward
        if len(segment) >= 5:
            # Check for consistent decay pattern
            perigee_trend = segment['perigee_altitude'].iloc[-1] - segment['perigee_altitude'].iloc[0]
            mean_motion_trend = segment['mean_motion'].iloc[-1] - segment['mean_motion'].iloc[0]
            
            # Decay is characterized by decreasing altitude and increasing mean motion
            if perigee_trend < 0 and mean_motion_trend > 0:
                return segment
    
    # If no segment shows clear decay, return the last segment if it has enough points
    if len(segments[-1]) >= 5:
        return segments[-1]
    elif len(segments) > 1 and len(segments[-2]) >= 5:
        return segments[-2]
    
    # Last resort: return the longest segment
    return max(segments, key=len) if segments else None

def adaptive_resampling(df, target_points=100):
    """
    Adaptively resample the data to focus more points on rapid changes.
    Uses a weighted approach that increases point density where changes are faster.
    """
    if len(df) < 4:
        return df  # Not enough points for resampling
    
    # Convert epoch to numeric for calculations
    df = df.copy()
    first_timestamp = df['epoch'].min()
    df['epoch_seconds'] = (df['epoch'] - first_timestamp).dt.total_seconds()
    
    # Calculate importance weights based on rate of change of perigee_altitude
    df['weight'] = abs(df['perigee_altitude'].diff()) + 0.1  # Add small constant to avoid zeros
    # Fill NA values with mean
    df['weight'].fillna(df['weight'].mean(), inplace=True)
    
    # Calculate cumulative weights
    cumulative_weights = df['weight'].cumsum()
    total_weight = cumulative_weights.iloc[-1]
    
    # Create points distributed according to weights
    new_cum_weights = np.linspace(0, total_weight, target_points)
    
    # Find corresponding indices in original data
    indices = []
    for w in new_cum_weights:
        idx = np.argmin(abs(cumulative_weights - w))
        indices.append(idx)
    
    # Create the resampled dataframe
    resampled_df = df.iloc[indices].copy()
    resampled_df.sort_values('epoch_seconds', inplace=True)
    
    # Clean up temporary columns
    resampled_df = resampled_df.drop(columns=['epoch_seconds', 'weight'])
    
    return resampled_df

def process_tle_data(df, norad_id):
    """Process TLE data using the same techniques as in decay-analysis.py"""
    print("Processing TLE data...")
    
    # Skip processing if too few points
    if len(df) < 10:
        raise ValueError("Not enough TLE data points (minimum 10 required)")
    
    # Sort by epoch to ensure chronological order
    df = df.sort_values('epoch').reset_index(drop=True)
    
    # Segment the data
    segments = segment_data(df)
    
    # Select the segment that best represents decay
    decay_segment = select_decay_segment(segments)
    
    if decay_segment is None or len(decay_segment) < 5:
        # If no good decay segment, take the last N points or all if fewer
        n_points = min(100, len(df))
        decay_segment = df.iloc[-n_points:].copy().reset_index(drop=True)
    
    # Apply adaptive resampling to focus on areas of rapid change
    resampled_df = adaptive_resampling(decay_segment, target_points=100)
    
    # Save processed data
    processed_filepath = os.path.join(TEMP_DIR, f"{norad_id}_processed.csv")
    resampled_df.to_csv(processed_filepath, index=False)
    
    # Create comparison plot
    plot_comparison(df, decay_segment, resampled_df, norad_id)
    
    return resampled_df

def plot_comparison(original_df, decay_df, resampled_df, norad_id):
    """Create plots comparing original data, detected decay segment, and resampled data."""
    print("Generating analysis plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot columns
    plot_cols = [('perigee_altitude', 0, 0), ('apogee_altitude', 0, 1), 
                ('inclination', 1, 0), ('bstar', 1, 1),
                ('semi_major_axis', 2, 0), ('mean_motion', 2, 1)]
    
    for col, row, col_idx in plot_cols:
        if col in original_df.columns:
            ax = axes[row, col_idx]
            
            # Plot original data
            ax.plot(original_df['epoch'], original_df[col], 'o', alpha=0.3, label='Original', color='blue')
            
            # Plot decay segment
            if decay_df is not None:
                ax.plot(decay_df['epoch'], decay_df[col], 'o', alpha=0.7, label='Decay Segment', color='green')
            
            # Plot resampled data
            if resampled_df is not None:
                ax.plot(resampled_df['epoch'], resampled_df[col], '-', label='Resampled', color='red', linewidth=2)
            
            ax.set_title(col.replace('_', ' ').title())
            ax.legend()
    
    plt.suptitle(f"Orbital Parameter Analysis for NORAD ID: {norad_id}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{norad_id}_analysis.png"))
    plt.close()

def load_model_and_scalers():
    """Load the trained model and associated scalers"""
    print("Loading model and scalers...")
    
    # Load model
    model_path = os.path.join(MODEL_DIR, 'amr_prediction_model.keras')
    model = load_model(model_path)
    
    # Load metadata to get feature names
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)[0]
    
    feature_names = metadata['input_features']
    seq_length = metadata['sequence_length']
    
    # Load feature scalers
    feature_scalers = []
    for feature in feature_names:
        scaler_path = os.path.join(MODEL_DIR, f'{feature}_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            feature_scalers.append(scaler)
    
    # Load target scaler
    target_scaler_path = os.path.join(MODEL_DIR, 'target_scaler.pkl')
    with open(target_scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    return model, feature_names, feature_scalers, target_scaler

def prepare_model_input(df, feature_names, feature_scalers, seq_length):
    """Prepare the processed data for model input"""
    print("Preparing model input...")
    
    # Check if we have enough data points
    if len(df) != seq_length:
        raise ValueError(f"Expected {seq_length} data points, but got {len(df)}")
    
    # Extract features
    input_sequences = []
    
    for i, feature in enumerate(feature_names):
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in data")
        
        seq = df[feature].values
        
        # Apply the appropriate scaler
        seq_scaled = feature_scalers[i].transform(seq.reshape(-1, 1)).flatten()
        input_sequences.append(seq_scaled)
    
    return input_sequences

def predict_amr(model, input_sequences, target_scaler):
    """Use the model to predict AMR value"""
    print("Predicting AMR value...")
    
    # Reshape input for model
    model_input = [seq.reshape(1, -1) for seq in input_sequences]
    
    # Make prediction
    prediction_scaled = model.predict(model_input)
    
    # Inverse transform to get actual AMR
    prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]
    
    return prediction

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fetch TLE data and predict AMR for a satellite')
    parser.add_argument('norad_id', type=str, help='NORAD ID of the satellite')
    args = parser.parse_args()
    
    norad_id = args.norad_id
    
    try:
        # Parse configuration
        username, password = parse_config()
        
        # Login to Space-Track
        session = login_to_spacetrack(username, password)
        
        # Fetch TLE data
        tle_data = fetch_tle_data(session, norad_id)
        
        # Parse TLE data
        df = parse_tle_data(tle_data, norad_id)
        
        # Process the data
        processed_df = process_tle_data(df, norad_id)
        
        # Load model and scalers
        model, feature_names, feature_scalers, target_scaler = load_model_and_scalers()
        
        # Prepare input for the model
        input_sequences = prepare_model_input(processed_df, feature_names, feature_scalers, len(processed_df))
        
        # Predict AMR
        amr_prediction = predict_amr(model, input_sequences, target_scaler)
        
        # Output results
        print("\n" + "="*50)
        print(f"AMR Prediction Results for NORAD ID: {norad_id}")
        print("="*50)
        print(f"Predicted AMR: {amr_prediction:.8f} m²/kg")
        print(f"Analysis plots saved to: {os.path.join(OUTPUT_DIR, f'{norad_id}_analysis.png')}")
        print("="*50)
        
        # Save result to file
        result = {
            'norad_id': norad_id,
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_amr': float(amr_prediction),
            'data_points_used': len(processed_df),
            'timespan_days': (processed_df['epoch'].max() - processed_df['epoch'].min()).days
        }
        
        result_df = pd.DataFrame([result])
        result_df.to_csv(os.path.join(OUTPUT_DIR, f"{norad_id}_amr_prediction.csv"), index=False)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()