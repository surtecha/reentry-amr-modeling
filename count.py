import os
import csv
import glob

# Define the folder path
folder_path = '/Users/suryatejachalla/Research/Re-entry-Prediction/Code/Orbital-Influence/tle_data_output'

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"Error: Folder '{folder_path}' does not exist.")
    exit(1)

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

if not csv_files:
    print(f"No CSV files found in folder '{folder_path}'.")
    exit(0)

print(f"Checking {len(csv_files)} CSV files for those with fewer than 100 records...")

# Check each CSV file
small_files = []

for file_path in csv_files:
    filename = os.path.basename(file_path)
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # Count the rows
            row_count = sum(1 for row in reader) - 1  # Subtract 1 for header row
            
            if row_count < 50:
                small_files.append((filename, row_count))
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")

# Print results
if small_files:
    print("\nFiles with fewer than 50 records:")
    for filename, count in small_files:
        print(f"{filename}: {count} records")
    print(f"\nTotal: {len(small_files)} files with fewer than 100 records")
else:
    print("\nNo files with fewer than 50 records found.")