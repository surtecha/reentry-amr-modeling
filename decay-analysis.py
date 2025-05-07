import os
import pandas as pd
import numpy as np
from scipy import interpolate
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

INPUT_DIR = '6-month-csv'
OUTPUT_DIR = 'decay-analysis'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def process_file(filename):
    """
    Process a single TLE data file to identify and analyze decay patterns.
    """
    try:
        # Read the CSV file
        input_path = os.path.join(INPUT_DIR, filename)
        df = pd.read_csv(input_path, parse_dates=['epoch'])
        
        # Skip files with too few points
        if len(df) < 10:
            return None, None
        
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
        
        return df, resampled_df
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None, None

def plot_comparison(original_df, decay_df, resampled_df, filename, save_dir):
    """
    Create plots comparing original data, detected decay segment, and resampled data.
    """
    norad_id = filename.split('.')[0]
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
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{norad_id}_analysis.png"))
    plt.close()

def main():
    """
    Main function to process all TLE data files.
    """
    # Create directory for plots
    plot_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.csv'):
            norad_id = filename.split('.')[0]
            print(f"Processing {norad_id}...")
            
            # Process the file
            original_df, resampled_df = process_file(filename)
            
            if original_df is not None and resampled_df is not None:
                # Determine decay segment (for plotting)
                segments = segment_data(original_df)
                decay_segment = select_decay_segment(segments)
                
                # Save resampled data
                output_path = os.path.join(OUTPUT_DIR, filename)
                resampled_df.to_csv(output_path, index=False)
                
                # Generate plots for visualization
                plot_comparison(original_df, decay_segment, resampled_df, filename, plot_dir)
    
    print(f"Analysis complete. Results saved to '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()