import os
import pandas as pd
import numpy as np
from scipy import interpolate
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

INPUT_DIR = '6-month-csv'
OUTPUT_DIR = 'decay-analysis'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_maneuver_or_change(df, column='perigee_altitude', threshold=10):
    changes = []
    if len(df) < 3:
        return changes
    
    df['gradient'] = df[column].diff() / df['epoch'].diff().dt.total_seconds()
    
    std_grad = df['gradient'].std()
    if pd.isna(std_grad) or std_grad == 0:
        return changes
    
    for i in range(1, len(df)-1):
        if abs(df['gradient'].iloc[i]) > threshold * std_grad:
            changes.append(i)
    
    df.drop('gradient', axis=1, inplace=True)
    
    return changes

def segment_data(df):
    change_indices = detect_maneuver_or_change(df, 'perigee_altitude')
    
    if not change_indices:
        change_indices = detect_maneuver_or_change(df, 'apogee_altitude')
    
    if not change_indices:
        change_indices = detect_maneuver_or_change(df, 'mean_motion', threshold=5)
    
    change_indices = sorted(set(change_indices))
    
    segments = []
    start_idx = 0
    
    for idx in change_indices:
        if idx - start_idx > 2:
            segments.append(df.iloc[start_idx:idx+1])
        start_idx = idx + 1
    
    if len(df) - start_idx > 2:
        segments.append(df.iloc[start_idx:])
    
    return segments

def select_decay_segment(segments):
    if not segments:
        return None
    
    last_segment = segments[-1]
    if len(last_segment) >= 5:
        first_perigee = last_segment['perigee_altitude'].iloc[0]
        last_perigee = last_segment['perigee_altitude'].iloc[-1]
        
        if last_perigee < first_perigee and (first_perigee - last_perigee) / first_perigee > 0.05:
            return last_segment
    
    for segment in reversed(segments):
        if len(segment) >= 5:
            perigee_trend = segment['perigee_altitude'].iloc[-1] - segment['perigee_altitude'].iloc[0]
            mean_motion_trend = segment['mean_motion'].iloc[-1] - segment['mean_motion'].iloc[0]
            
            if perigee_trend < 0 and mean_motion_trend > 0:
                return segment
    
    if len(segments[-1]) >= 5:
        return segments[-1]
    elif len(segments) > 1 and len(segments[-2]) >= 5:
        return segments[-2]
    
    return max(segments, key=len) if segments else None

def adaptive_resampling(df, target_points=100):
    if len(df) < 4:
        return df
    
    df = df.copy()
    first_timestamp = df['epoch'].min()
    df['epoch_seconds'] = (df['epoch'] - first_timestamp).dt.total_seconds()
    
    df['weight'] = abs(df['perigee_altitude'].diff()) + 0.1
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

def process_file(filename):
    try:
        input_path = os.path.join(INPUT_DIR, filename)
        df = pd.read_csv(input_path, parse_dates=['epoch'])
        
        if len(df) < 10:
            return None, None
        
        df = df.sort_values('epoch').reset_index(drop=True)
        
        segments = segment_data(df)
        
        decay_segment = select_decay_segment(segments)
        
        if decay_segment is None or len(decay_segment) < 5:
            n_points = min(100, len(df))
            decay_segment = df.iloc[-n_points:].copy().reset_index(drop=True)
        
        resampled_df = adaptive_resampling(decay_segment, target_points=100)
        
        return df, resampled_df
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None, None

def plot_comparison(original_df, decay_df, resampled_df, filename, save_dir):
    norad_id = filename.split('.')[0]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    plot_cols = [('perigee_altitude', 0, 0), ('apogee_altitude', 0, 1), 
                ('inclination', 1, 0), ('bstar', 1, 1),
                ('semi_major_axis', 2, 0), ('mean_motion', 2, 1)]
    
    for col, row, col_idx in plot_cols:
        if col in original_df.columns:
            ax = axes[row, col_idx]
            
            ax.plot(original_df['epoch'], original_df[col], 'o', alpha=0.3, label='Original', color='blue')
            
            if decay_df is not None:
                ax.plot(decay_df['epoch'], decay_df[col], 'o', alpha=0.7, label='Decay Segment', color='green')
            
            if resampled_df is not None:
                ax.plot(resampled_df['epoch'], resampled_df[col], '-', label='Resampled', color='red', linewidth=2)
            
            ax.set_title(col.replace('_', ' ').title())
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{norad_id}_analysis.png"))
    plt.close()

def main():
    plot_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    merged_df = pd.read_csv('merged_output.csv')
    valid_norads = set(merged_df[(merged_df['amr'] >= 0.001) & (merged_df['amr'] <= 0.01)]['norad'].astype(str))
    
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.csv'):
            norad_id = filename.split('.')[0]
            
            if norad_id in valid_norads:
                print(f"Processing {norad_id}...")
                
                original_df, resampled_df = process_file(filename)
                
                if original_df is not None and resampled_df is not None:
                    segments = segment_data(original_df)
                    decay_segment = select_decay_segment(segments)
                    
                    output_path = os.path.join(OUTPUT_DIR, filename)
                    resampled_df.to_csv(output_path, index=False)
            else:
                print(f"Skipping {norad_id}: Not in AMR range or not found in merged_output.csv")
    
    print(f"Analysis complete. Results saved to '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()