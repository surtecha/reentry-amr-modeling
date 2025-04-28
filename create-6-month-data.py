import os
import csv
import math
import datetime
import pandas as pd

# Constants
GM = 398600441800000.0
GM13 = GM ** (1.0/3.0)
MRAD = 6378.137
PI = 3.14159265358979
TPI86 = 2.0 * PI / 86400.0

# Create output directory if it doesn't exist
os.makedirs('tle-6-months', exist_ok=True)

# Read the merged_output.csv file
merged_data = pd.read_csv('/Users/suryatejachalla/Research/reentry-amr-modeling/data/merged_output.csv')

# Function to parse TLE epoch to datetime
def parse_tle_epoch(epoch_str):
    # TLE epoch format is YYDDD.FRACDAY
    # First 2 digits are the year (add 1900 if yy >= 57, otherwise add 2000)
    # Next 3 digits are the day of the year
    # The fractional part is the time of the day
    
    year = int(epoch_str[0:2])
    if year >= 57:  # Epoch/Space age starts in 1957
        year += 1900
    else:
        year += 2000
    
    day_of_year = int(epoch_str[2:5])
    frac_day = float("0." + epoch_str[6:]) if '.' in epoch_str else 0
    
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year-1 + frac_day)
    return date

# Function to extract parameters from TLE
def extract_parameters_from_tle(line1, line2):
    # Parse TLE line 1
    norad_id = int(line1[2:7])
    epoch = line1[18:32].strip()
    mean_motion_deriv = float(line1[33:43])
    bstar = float(f"{line1[53:54]}.{line1[54:59]}e{line1[59:61]}") if line1[53:54] != ' ' else 0.0
    
    # Parse TLE line 2
    inclination = float(line2[8:16])
    eccentricity = float(f"0.{line2[26:33]}")
    mean_motion = float(line2[52:63])
    
    # Calculate derived parameters
    sma = GM13 / ((TPI86 * mean_motion) ** (2.0 / 3.0)) / 1000.0  # semi-major axis in km
    perigee_alt = sma * (1.0 - eccentricity) - MRAD  # perigee altitude in km
    smak = sma * 1000.0  # semi-major axis in meters
    orbital_period = 2.0 * PI * ((smak ** 3.0) / GM) ** (0.5)  # orbital period in seconds
    
    # Format datetime from epoch - store as consistent string format
    date = parse_tle_epoch(epoch)
    date_str = date.strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        'date': date_str,
        'perigee_alt': perigee_alt,
        'mean_motion': mean_motion,
        'mean_motion_deriv': mean_motion_deriv,
        'eccentricity': eccentricity,
        'inclination': inclination,
        'bstar': bstar,
        'semi_major_axis': sma,
        'orbital_period': orbital_period
    }

# Process each object in the merged_output.csv
for _, row in merged_data.iterrows():
    norad_id = row['norad']
    amr = row['amr']
    
    # Path to the TLE file
    tle_file_path = os.path.join('objects', f"{norad_id}.txt")
    
    # Check if the file exists
    if not os.path.exists(tle_file_path):
        print(f"Warning: TLE file for NORAD ID {norad_id} not found.")
        continue
    
    # Read the TLE file
    with open(tle_file_path, 'r') as file:
        lines = file.readlines()
    
    # Group lines into TLE entries (each entry is 2 lines)
    tle_entries = []
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            line1 = lines[i].strip().replace('\\', '')
            line2 = lines[i+1].strip().replace('\\', '')
            
            # Skip if not a valid TLE format
            if not (line1.startswith('1 ') and line2.startswith('2 ')):
                continue
            
            try:
                params = extract_parameters_from_tle(line1, line2)
                # Store the datetime object for sorting
                params['datetime_obj'] = parse_tle_epoch(line1[18:32].strip())
                tle_entries.append(params)
            except Exception as e:
                print(f"Error processing TLE for NORAD ID {norad_id}: {e}")
    
    # Sort TLE entries by date
    tle_entries.sort(key=lambda x: x['datetime_obj'])
    
    # Get the last 6 months of data
    if tle_entries:
        latest_date = tle_entries[-1]['datetime_obj']
        six_months_ago = latest_date - datetime.timedelta(days=180)
        
        # Filter entries from the last 6 months
        recent_entries = [entry for entry in tle_entries if entry['datetime_obj'] >= six_months_ago]
        
        # Remove the datetime_obj field before saving to CSV
        for entry in recent_entries:
            del entry['datetime_obj']
        
        # Save to CSV
        output_file = os.path.join('tle-6-months', f"{norad_id}.csv")
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['date', 'perigee_alt', 'mean_motion', 'mean_motion_deriv', 
                         'eccentricity', 'inclination', 'bstar', 'semi_major_axis', 'orbital_period']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in recent_entries:
                writer.writerow(entry)
        
        print(f"Processed NORAD ID {norad_id}: {len(recent_entries)} entries in the last 6 months.")