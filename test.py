import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import glob

def process_file(file_path, output_dir, num_points=90):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Filter to include only the required columns
        required_columns = [
            'mean_motion', 'eccentricity', 'inclination', 'bstar',
            'mean_motion_deriv', 'semi_major_axis', 'orbital_period', 'perigee_alt'
        ]
        
        # Check if all required columns exist in the dataframe
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in {file_path}: {missing_columns}")
        
        # Select only the required columns
        df = df[required_columns]
        
        # Create indices for the original data points (assuming they're evenly spaced in time)
        original_indices = np.arange(len(df))
        
        # Create new evenly spaced indices for interpolation
        new_indices = np.linspace(0, len(df) - 1, num_points)
        
        # Create a new dataframe to store processed data
        processed_df = pd.DataFrame()
        
        # Process each column
        for column in df.columns:
            # Get the data for the current column
            y = df[column].values
            
            # Apply Savitzky-Golay filter if we have enough points
            window_length = min(7, len(y) - 2 if len(y) % 2 == 0 else len(y) - 1)
            # Ensure window_length is odd
            if window_length % 2 == 0:
                window_length -= 1
            # Ensure window_length is at least 3
            window_length = max(3, window_length)
            # Ensure polyorder is less than window_length
            poly_order = min(2, window_length - 1)
            
            if len(y) >= window_length:
                smoothed_y = savgol_filter(y, window_length, poly_order)
            else:
                smoothed_y = y
            
            # Interpolate to get evenly spaced points
            interp_func = interp1d(original_indices, smoothed_y, kind='cubic', bounds_error=False, fill_value='extrapolate')
            new_y = interp_func(new_indices)
            
            # Add to the processed dataframe
            processed_df[column] = new_y
        
        # Extract object_id from filename
        object_id = os.path.basename(file_path).split('.')[0]
        
        # Save the processed data
        output_path = os.path.join(output_dir, f"{object_id}.csv")
        processed_df.to_csv(output_path, index=False)
        
        return True, object_id
    except Exception as e:
        object_id = os.path.basename(file_path).split('.')[0]
        return False, object_id, str(e)

def main():
    # Input and output directories
    input_dir = "/Users/suryatejachalla/Research/reentry-amr-modeling/data/final-tle-data"
    output_dir = "raw-fitting"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # Initialize counters
    successful = 0
    failed = 0
    failed_files = []
    
    # Process each file
    total_files = len(csv_files)
    for i, file_path in enumerate(csv_files, 1):
        result = process_file(file_path, output_dir)
        
        if result[0]:
            successful += 1
            print(f"Processed {i}/{total_files}: {result[1]}")
        else:
            failed += 1
            failed_files.append((result[1], result[2]))
            print(f"Failed to process {i}/{total_files}: {result[1]} - Error: {result[2]}")
    
    # Print statistics
    print("\n--- Processing Statistics ---")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {successful}")
    print(f"Failed to process: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for failed_file, error in failed_files:
            print(f"- {failed_file}: {error}")

if __name__ == "__main__":
    main()