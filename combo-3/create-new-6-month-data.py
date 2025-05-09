import os
import csv
import pandas as pd
from datetime import datetime, timedelta
import re
import shutil

def parse_tle_epoch(tle_line):
    """Extract and convert TLE epoch to datetime object."""
    # TLE epoch format: YYDDD.DDDDDDDD (year, day of year, and fractional portion)
    epoch_str = tle_line[18:32].strip()
    
    year = int(epoch_str[:2])
    if year < 57:  # Cutoff for 21st century (adjust as needed)
        year += 2000
    else:
        year += 1900
    
    day_of_year = float(epoch_str[2:])
    integer_day = int(day_of_year)
    fractional_day = day_of_year - integer_day
    
    date = datetime(year, 1, 1) + timedelta(days=integer_day-1)
    seconds = int(fractional_day * 86400)  # Convert to seconds
    
    return date + timedelta(seconds=seconds)

def extract_last_6_months_tles(file_path):
    """Extract TLEs from the last 6 months from a file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Clean lines and remove any extra whitespace
        lines = [line.strip() for line in lines]
        
        # Ensure we have complete TLE sets (pairs of lines)
        tle_pairs = []
        for i in range(0, len(lines), 2):
            if i+1 < len(lines) and lines[i].startswith('1 ') and lines[i+1].startswith('2 '):
                tle_pairs.append((lines[i], lines[i+1]))
        
        # Sort TLE pairs by epoch date (newest last)
        tle_pairs.sort(key=lambda pair: parse_tle_epoch(pair[0]))
        
        # If there are no TLE pairs, return empty list
        if not tle_pairs:
            return []
        
        # Get the newest epoch date
        newest_date = parse_tle_epoch(tle_pairs[-1][0])
        
        # Calculate cutoff date (6 months before the newest date)
        cutoff_date = newest_date - timedelta(days=180)
        
        # Filter TLEs to only include those from the last 6 months
        recent_tles = []
        for pair in tle_pairs:
            epoch_date = parse_tle_epoch(pair[0])
            if epoch_date >= cutoff_date:
                recent_tles.append(pair)
        
        return recent_tles
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    # Create output directory if it doesn't exist
    output_dir = '6-month-data'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove existing directory
    os.makedirs(output_dir)
    
    # Read bad IDs from file
    bad_ids = set()
    try:
        with open('bad_id.txt', 'r') as f:
            for line in f:
                # Extract the NORAD ID from each line and add to the set
                norad_id = line.strip()
                if norad_id:  # Skip empty lines
                    bad_ids.add(norad_id)
        print(f"Loaded {len(bad_ids)} bad IDs to skip")
    except Exception as e:
        print(f"Error reading bad_id.txt: {e}")
    
    # Read NORAD IDs from CSV files
    norad_ids = set()
    
    # Try reading from merged_output.csv
    try:
        df1 = pd.read_csv('merged_output.csv')
        if 'norad' in df1.columns:
            norad_ids.update(df1['norad'].astype(str).tolist())
    except Exception as e:
        print(f"Error reading merged_output.csv: {e}")
    
    # Try reading from medium_bc.csv
    try:
        df2 = pd.read_csv('medium_bc.csv')
        if 'norad' in df2.columns:
            norad_ids.update(df2['norad'].astype(str).tolist())
    except Exception as e:
        print(f"Error reading medium_bc.csv: {e}")
    
    # Process each NORAD ID
    processed_count = 0
    skipped_bad_id_count = 0
    skipped_insufficient_tles_count = 0
    
    for norad_id in norad_ids:
        try:
            # Clean and pad NORAD ID
            norad_id = str(norad_id).strip()
            
            # Skip if in bad_id.txt
            if norad_id in bad_ids:
                skipped_bad_id_count += 1
                print(f"Skipping {norad_id}: Listed in bad_id.txt")
                continue
            
            # Find matching file in objects folder
            tle_file_path = os.path.join('/Users/suryatejachalla/Research/cosmos482/objects', f"{norad_id}.txt")
            
            # Extract TLEs from the last 6 months
            recent_tle_pairs = extract_last_6_months_tles(tle_file_path)
            
            # Skip if less than 100 TLEs
            if len(recent_tle_pairs) < 100:
                skipped_insufficient_tles_count += 1
                print(f"Skipping {norad_id}: Only {len(recent_tle_pairs)} TLEs found (minimum 100 required)")
                continue
            
            # Write TLEs to output file
            output_file_path = os.path.join(output_dir, f"{norad_id}.txt")
            with open(output_file_path, 'w') as out_file:
                for line1, line2 in recent_tle_pairs:
                    out_file.write(f"{line1}\n{line2}\n")
            
            processed_count += 1
            print(f"Processed {norad_id}: Found {len(recent_tle_pairs)} TLEs in the last 6 months")
        
        except Exception as e:
            print(f"Error processing NORAD ID {norad_id}: {e}")
    
    print(f"Completed! TLE data for the last 6 months has been saved to the '{output_dir}' folder.")
    print(f"Summary: {processed_count} objects processed, {skipped_bad_id_count} objects skipped (in bad_id.txt), {skipped_insufficient_tles_count} objects skipped (less than 100 TLEs)")

if __name__ == "__main__":
    main()