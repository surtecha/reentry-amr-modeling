import os
import pandas as pd
import numpy as np
from scipy import interpolate
import warnings
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

INPUT_DIR = '6-month-csv'
OUTPUT_DIR = '6-month-resampled'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def resample_to_100_points(df):
    """
    Resample a dataframe to exactly 100 points using cubic interpolation.
    Handles edge cases and ensures all patterns in the data are preserved.
    """
    # Convert epoch to numeric for interpolation (seconds since first timestamp)
    first_timestamp = df['epoch'].min()
    df['epoch_seconds'] = (df['epoch'] - first_timestamp).dt.total_seconds()
    
    # Get the original data points
    x_orig = df['epoch_seconds'].values
    
    # Create a new set of 100 evenly spaced points
    x_new = np.linspace(x_orig.min(), x_orig.max(), 100)
    
    # Create a new dataframe for resampled data
    resampled_df = pd.DataFrame()
    
    # Add the original epoch corresponding to the resampled points
    td_seconds = pd.to_timedelta(x_new, unit='s')
    resampled_df['epoch'] = first_timestamp + td_seconds
    
    # Generate the epoch_utc string representation
    resampled_df['epoch_utc'] = resampled_df['epoch'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    # Handle interpolation for all numeric columns except epoch-related ones
    numeric_columns = [col for col in df.columns if col not in ['epoch', 'epoch_utc', 'epoch_seconds']]
    
    # Different interpolation methods based on data characteristics
    for col in numeric_columns:
        y_orig = df[col].values
        
        # Check if we need to handle potential issues
        if np.ptp(y_orig) < 1e-10:  # If data is nearly constant
            # For nearly constant data, use linear interpolation to avoid oscillations
            resampled_df[col] = np.full_like(x_new, np.mean(y_orig))
        else:
            # For most data, use cubic spline interpolation
            # Use k=3 for cubic spline when we have enough points, otherwise reduce order
            k = min(3, len(x_orig) - 1)
            if k < 1:
                # If too few points, just replicate the data
                resampled_df[col] = y_orig[0] if len(y_orig) > 0 else np.nan
            else:
                # Create a spline function
                try:
                    cs = interpolate.CubicSpline(x_orig, y_orig)
                    resampled_df[col] = cs(x_new)
                except Exception:
                    # Fallback to linear interpolation if cubic fails
                    try:
                        f = interpolate.interp1d(x_orig, y_orig, kind='linear', bounds_error=False,
                                                fill_value=(y_orig[0], y_orig[-1]))
                        resampled_df[col] = f(x_new)
                    except Exception:
                        # Last resort: nearest neighbor
                        f = interpolate.interp1d(x_orig, y_orig, kind='nearest', bounds_error=False,
                                               fill_value=(y_orig[0], y_orig[-1]))
                        resampled_df[col] = f(x_new)
                        
        # Ensure we don't have invalid values (NaN, inf) after interpolation
        if any(~np.isfinite(resampled_df[col])):
            # Replace with nearest valid value
            mask = ~np.isfinite(resampled_df[col])
            resampled_df.loc[mask, col] = np.interp(
                x_new[mask], 
                x_orig, 
                y_orig,
                left=y_orig[0],
                right=y_orig[-1]
            )
    
    # Clean up
    resampled_df = resampled_df.drop(columns=['epoch_seconds'], errors='ignore')
    
    return resampled_df

# Process each CSV file
file_count = 0
for filename in os.listdir(INPUT_DIR):
    if filename.endswith('.csv'):
        try:
            # Read the CSV file
            input_path = os.path.join(INPUT_DIR, filename)
            df = pd.read_csv(input_path, parse_dates=['epoch'])
            
            # Skip files with fewer than 4 points (minimum needed for cubic interpolation)
            if len(df) < 4:
                continue
                
            # Resample to exactly 100 points
            resampled_df = resample_to_100_points(df)
            
            # Save the resampled data
            output_path = os.path.join(OUTPUT_DIR, filename)
            resampled_df.to_csv(output_path, index=False)
            
            file_count += 1
            
            # Optional validation plotting (can be commented out in production)
            
            # Uncomment this block to generate validation plots
            if file_count <= 20:  # Only plot first few files
                fig, axes = plt.subplots(3, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                plot_cols = ['perigee_altitude', 'apogee_altitude', 'inclination', 
                             'bstar', 'semi_major_axis', 'mean_motion']
                
                for i, col in enumerate(plot_cols):
                    if i < len(axes) and col in df.columns:
                        axes[i].plot(df['epoch'], df[col], 'o', label='Original')
                        axes[i].plot(resampled_df['epoch'], resampled_df[col], '-', label='Resampled')
                        axes[i].set_title(col)
                        axes[i].legend()
                
                plt.tight_layout()
                plt.savefig(f"validation_{filename[:-4]}.png")
                plt.close()
            
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

print(f"Resampling complete. Processed {file_count} files. Resampled data saved to '{OUTPUT_DIR}' directory.")