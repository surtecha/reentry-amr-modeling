import os
import pandas as pd
import numpy as np
from scipy import interpolate
import glob
import datetime

# Directory paths
input_dir = 'tle-6-months'
output_dir = 'tle-nn-input'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all CSV files in the input directory
csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

# Process each CSV file
for csv_file in csv_files:
    # Extract NORAD ID from filename
    norad_id = os.path.basename(csv_file).split('.')[0]
    print(f"Processing NORAD ID: {norad_id}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if data exists
    if df.empty:
        print(f"Warning: No data found for NORAD ID {norad_id}. Skipping.")
        continue
    
    # Convert date strings to datetime objects and then to numeric values for interpolation
    df['date'] = pd.to_datetime(df['date'])
    # Calculate days since the first date
    first_date = df['date'].min()
    df['days_since_start'] = [(date - first_date).total_seconds() / (24 * 3600) for date in df['date']]
    
    # Sort by date
    df = df.sort_values('days_since_start')
    
    # Get the number of available data points
    num_points = len(df)
    
    # Create a DataFrame for resampled data
    resampled_df = pd.DataFrame()
    
    # Get the columns to interpolate (excluding 'date')
    columns_to_interpolate = [col for col in df.columns if col not in ['date', 'days_since_start']]
    
    # Handle different cases based on number of data points
    if num_points >= 180:
        # Case 1: More than 180 points - use direct interpolation
        # Create evenly spaced points across the original data range
        x_new = np.linspace(df['days_since_start'].min(), df['days_since_start'].max(), 180)
        
        # Interpolate each column
        for col in columns_to_interpolate:
            # Create interpolation function
            f = interpolate.interp1d(df['days_since_start'], df[col], kind='linear', bounds_error=False, fill_value='extrapolate')
            # Apply interpolation
            resampled_df[col] = f(x_new)
            
        # Convert days_since_start back to dates
        resampled_dates = [first_date + datetime.timedelta(days=float(x)) for x in x_new]
        resampled_df['date'] = resampled_dates
        
    elif num_points > 1:
        # Case 2: Between 2 and 179 points - interpolate to create 180 points
        # Create evenly spaced points across the original data range
        x_new = np.linspace(df['days_since_start'].min(), df['days_since_start'].max(), 180)
        
        # Interpolate each column
        for col in columns_to_interpolate:
            # Create interpolation function
            f = interpolate.interp1d(df['days_since_start'], df[col], kind='linear', bounds_error=False, fill_value='extrapolate')
            # Apply interpolation
            resampled_df[col] = f(x_new)
            
        # Convert days_since_start back to dates
        resampled_dates = [first_date + datetime.timedelta(days=float(x)) for x in x_new]
        resampled_df['date'] = resampled_dates
        
    elif num_points == 1:
        # Case 3: Only one data point - duplicate it 180 times
        print(f"Warning: Only one data point for NORAD ID {norad_id}. Duplicating to create 180 points.")
        
        single_point = df.iloc[0]
        
        # Create a DataFrame with 180 duplicated rows
        resampled_df = pd.DataFrame([single_point.to_dict()] * 180)
        
        # Create evenly spaced dates over a 6-month period
        start_date = pd.to_datetime(single_point['date'])
        end_date = start_date + datetime.timedelta(days=180)
        resampled_dates = pd.date_range(start=start_date, end=end_date, periods=180)
        resampled_df['date'] = resampled_dates
        
    else:
        # Case 4: No data points - skip this file
        print(f"Error: No valid data points for NORAD ID {norad_id}. Skipping.")
        continue
    
    # Rearrange columns to put date first
    columns_order = ['date'] + [col for col in resampled_df.columns if col != 'date']
    resampled_df = resampled_df[columns_order]
    
    # Save the resampled data
    output_file = os.path.join(output_dir, f"{norad_id}_resampled.csv")
    resampled_df.to_csv(output_file, index=False)
    
    print(f"Completed processing NORAD ID {norad_id}: Created {len(resampled_df)} resampled points.")

print("All files processed successfully.")