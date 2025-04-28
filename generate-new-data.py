import os
import re
import glob
from datetime import datetime, timedelta
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
import pandas as pd
import shutil

def parse_tle_epoch(line1):
    """Extract epoch date from TLE line 1."""
    try:
        # Extract epoch year and day of year
        epoch_year = int(line1[18:20])
        epoch_day = float(line1[20:32])
        
        # Adjust year to full format
        if epoch_year < 57:  # Arbitrary cutoff for the century
            year = 2000 + epoch_year
        else:
            year = 1900 + epoch_year
            
        # Calculate date
        base_date = datetime(year, 1, 1)
        date = base_date + timedelta(days=epoch_day - 1)
        
        return date
    except:
        return None

def extract_norad_id(line1):
    """Extract NORAD ID from TLE line 1."""
    match = re.search(r'^\d\s+(\d+)', line1)
    if match:
        return int(match.group(1))
    return None

def read_tle_file(filepath):
    """Read TLE data from a file and return a dictionary of NORAD IDs to TLE pairs."""
    tle_data = {}
    
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        # Clean lines
        lines = [line.strip() for line in lines]
        
        # Process lines in pairs
        i = 0
        while i < len(lines) - 1:
            line1 = lines[i]
            line2 = lines[i+1]
            
            # Check if these lines form a valid TLE pair
            if line1.startswith('1 ') and line2.startswith('2 '):
                norad_id = extract_norad_id(line1)
                
                if norad_id:
                    # Get epoch for sorting
                    epoch = parse_tle_epoch(line1)
                    
                    # Add to dictionary, with epoch for sorting
                    if norad_id not in tle_data:
                        tle_data[norad_id] = []
                    
                    tle_data[norad_id].append((epoch, line1, line2))
                
            i += 1
        
        return tle_data
    
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return {}

def create_norad_id_file(norad_id, tle_list, output_dir):
    """Create or update a file for a specific NORAD ID with sorted TLEs."""
    filepath = os.path.join(output_dir, f"{norad_id}.txt")
    existing_tles = []
    
    # Check if file already exists and read existing TLEs
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()
            
            # Clean lines
            lines = [line.strip() for line in lines]
            
            # Process lines in pairs
            i = 0
            while i < len(lines) - 1:
                if lines[i].startswith('1 ') and lines[i+1].startswith('2 '):
                    epoch = parse_tle_epoch(lines[i])
                    existing_tles.append((epoch, lines[i], lines[i+1]))
                i += 1
    
    # Combine existing and new TLEs
    all_tles = existing_tles + tle_list
    
    # Remove duplicates (keeping the first occurrence)
    unique_tles = {}
    for epoch, line1, line2 in all_tles:
        if epoch is not None:
            # Create a key that uniquely identifies this TLE data
            # Using both epoch and a hash of the content
            key = f"{epoch}_{hash(line1+line2)}"
            if key not in unique_tles:
                unique_tles[key] = (epoch, line1, line2)
    
    # Sort by epoch
    sorted_tles = sorted([v for v in unique_tles.values()], key=lambda x: x[0] if x[0] else datetime.min)
    
    # Write to file
    with open(filepath, 'w') as file:
        for _, line1, line2 in sorted_tles:
            file.write(f"{line1}\n{line2}\n")
    
    return len(sorted_tles)

def process_tle_files(input_dir, output_dir):
    """Process all TLE files in the input directory and organize by NORAD ID."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all text files in the input directory
    file_paths = glob.glob(os.path.join(input_dir, "*.txt"))
    
    # Track statistics
    total_files_processed = 0
    total_objects_found = 0
    total_tles_processed = 0
    processed_norad_ids = set()
    
    print(f"Found {len(file_paths)} TLE files to process in {input_dir}")
    
    # Process each file
    for file_path in file_paths:
        try:
            filename = os.path.basename(file_path)
            print(f"Processing {filename}...")
            
            # Read TLE data from file
            tle_data = read_tle_file(file_path)
            
            # Create individual files for each NORAD ID
            for norad_id, tle_list in tle_data.items():
                tle_count = create_norad_id_file(norad_id, tle_list, output_dir)
                
                total_tles_processed += len(tle_list)
                
                if norad_id not in processed_norad_ids:
                    processed_norad_ids.add(norad_id)
                    total_objects_found += 1
                
                print(f"  - NORAD ID {norad_id}: {len(tle_list)} TLEs (total: {tle_count})")
            
            total_files_processed += 1
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Files processed: {total_files_processed}")
    print(f"Unique objects found: {total_objects_found}")
    print(f"Total TLEs processed: {total_tles_processed}")
    print(f"Output directory: {output_dir}")

def main():
    input_dir = "/Users/suryatejachalla/Research/reentry-amr-modeling/data/raw-tle-data-space-track"
    output_dir = "objects"
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return
    
    # Process TLE files
    process_tle_files(input_dir, output_dir)

if __name__ == "__main__":
    main()