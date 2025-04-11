import pandas as pd
import numpy as np
import os
import glob 
import matplotlib.pyplot as plt
import seaborn as sns # For prettier plots

# --- Configuration ---
TLE_CSV_DIRECTORY = 'tle_data_output' 
MIN_ROWS_THRESHOLD = 75 
SUMMARY_CSV_FILE = 'decay_object_classification_summary.csv' 
# Directory to save the generated plots
PLOT_OUTPUT_DIRECTORY = 'analysis_plots' 

# --- Define Categorization Functions (same as before) ---

def categorize_eccentricity(ecc):
    if pd.isna(ecc): return "Unknown"
    elif ecc < 0.01: return "Near-Circular"
    elif ecc < 0.1: return "Low Eccentricity"
    elif ecc < 0.3: return "Moderate Eccentricity"
    else: return "High Eccentricity"

def categorize_altitude(perigee_km):
    if pd.isna(perigee_km): return "Unknown"
    elif perigee_km < 250: return "Very Low LEO"
    elif perigee_km < 500: return "Low LEO"
    elif perigee_km < 1000: return "Mid LEO"
    else: return "Higher Orbit (Decaying)"

def categorize_inclination(inc_deg):
    if pd.isna(inc_deg): return "Unknown"
    elif inc_deg < 30: return "Low Inclination"
    elif inc_deg < 75: return "Mid Inclination"
    elif inc_deg < 110: return "High Inclination"
    else: return "Retrograde"

# --- Main Analysis Script ---

print(f"Starting analysis of CSV files in '{TLE_CSV_DIRECTORY}'...")
csv_files = glob.glob(os.path.join(TLE_CSV_DIRECTORY, '*.csv'))

if not csv_files:
    print(f"Error: No CSV files found in the directory '{TLE_CSV_DIRECTORY}'.")
    exit()

print(f"Found {len(csv_files)} CSV files.")

analysis_results = []
processed_count = 0
skipped_count = 0

# Create plot directory if it doesn't exist
os.makedirs(PLOT_OUTPUT_DIRECTORY, exist_ok=True)
print(f"Plots will be saved in '{PLOT_OUTPUT_DIRECTORY}/'")

# --- File Processing Loop (similar to before) ---
for csv_file_path in csv_files:
    filename = os.path.basename(csv_file_path)
    try:
        norad_id = int(filename.split('.')[0]) 
    except (ValueError, IndexError):
        # print(f"  Warning: Could not parse NORAD ID from filename '{filename}'. Skipping.")
        skipped_count += 1
        continue

    try:
        try:
             df = pd.read_csv(csv_file_path, parse_dates=['Epoch'], index_col='Epoch')
        except ValueError as e:
             # print(f"  Warning: Error reading or parsing dates in {filename}: {e}. Trying without date parsing.")
             df = pd.read_csv(csv_file_path) 
             if 'Epoch' in df.columns:
                 df.set_index('Epoch', inplace=True)
             else:
                 # print(f"  Critical Error: 'Epoch' column not found in {filename}. Skipping.")
                 skipped_count += 1
                 continue

        if len(df) < MIN_ROWS_THRESHOLD:
            # print(f"  Skipping {filename}: Only {len(df)} rows.")
            skipped_count += 1
            continue

        required_cols = ['Eccentricity', 'AltitudePerigee_km', 'AltitudeApogee_km', 'Inclination_deg', 'Bstar']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             # print(f"  Warning: Skipping {filename}. Missing required columns: {', '.join(missing_cols)}")
             skipped_count += 1
             continue
             
        df.sort_index(inplace=True)

        # Calculate Metrics
        median_eccentricity = df['Eccentricity'].median()
        mean_eccentricity = df['Eccentricity'].mean()
        min_eccentricity = df['Eccentricity'].min()
        max_eccentricity = df['Eccentricity'].max()
        final_eccentricity = df['Eccentricity'].iloc[-1] if not df.empty else np.nan 

        median_perigee = df['AltitudePerigee_km'].median()
        median_apogee = df['AltitudeApogee_km'].median()
        final_perigee = df['AltitudePerigee_km'].iloc[-1] if not df.empty else np.nan

        median_inclination = df['Inclination_deg'].median()
        median_bstar = df['Bstar'].median()
        eccentricity_range = max_eccentricity - min_eccentricity

        # Categorize
        ecc_category = categorize_eccentricity(median_eccentricity)
        alt_category = categorize_altitude(median_perigee)
        inc_category = categorize_inclination(median_inclination)

        # Store Results
        analysis_results.append({
            'NORAD_ID': norad_id,
            'FileName': filename,
            'Num_TLEs': len(df),
            'Median_Eccentricity': median_eccentricity,
            'Mean_Eccentricity': mean_eccentricity,
            'Min_Eccentricity': min_eccentricity,
            'Max_Eccentricity': max_eccentricity,
            'Range_Eccentricity': eccentricity_range,
            'Final_Eccentricity': final_eccentricity,
            'Median_Perigee_km': median_perigee,
            'Median_Apogee_km': median_apogee,
            'Final_Perigee_km': final_perigee,
            'Median_Inclination_deg': median_inclination,
            'Median_Bstar': median_bstar,
            'Eccentricity_Category': ecc_category,
            'Altitude_Category': alt_category,
            'Inclination_Category': inc_category
        })
        processed_count += 1

    except FileNotFoundError:
        # print(f"  Error: File not found: {csv_file_path}. Skipping.")
        skipped_count += 1
    except pd.errors.EmptyDataError:
         # print(f"  Warning: File is empty: {csv_file_path}. Skipping.")
         skipped_count += 1
    except Exception as e:
        print(f"!! Unexpected Error processing file {filename} (NORAD ID: {norad_id}): {e}")
        import traceback
        traceback.print_exc() 
        skipped_count += 1
        
# --- Process Results ---
print("\n--- Analysis Summary ---")
print(f"Total files found: {len(csv_files)}")
print(f"Files processed (>= {MIN_ROWS_THRESHOLD} rows): {processed_count}")
print(f"Files skipped (threshold, errors, missing cols): {skipped_count}")

if not analysis_results:
    print("\nNo objects met the criteria for analysis. No summary or plots generated.")
    exit() # Exit if no data

# Create Summary DataFrame
summary_df = pd.DataFrame(analysis_results)
summary_df.set_index('NORAD_ID', inplace=True)
summary_df.sort_index(inplace=True)

# Save the raw summary data
try:
    summary_df.to_csv(SUMMARY_CSV_FILE)
    print(f"\nFull summary data saved to '{SUMMARY_CSV_FILE}'")
except Exception as e:
    print(f"\nError saving summary CSV: {e}")

# --- Print Overall Statistics ---
print("\n--- Overall Statistics for Processed Objects ---")
# Use describe for numerical columns - selecting key ones
print("Distribution of Key Metrics:")
print(summary_df[['Median_Eccentricity', 'Median_Perigee_km', 'Median_Inclination_deg', 'Median_Bstar']].describe())

print("\nCategory Counts:")
print("\nEccentricity Categories:")
print(summary_df['Eccentricity_Category'].value_counts())
print("\nAltitude Categories (based on Median Perigee):")
print(summary_df['Altitude_Category'].value_counts())
print("\nInclination Categories:")
print(summary_df['Inclination_Category'].value_counts())

# --- Generate and Save Plots ---
print("\n--- Generating Plots ---")
sns.set_theme(style="whitegrid") # Set a nice theme for plots

# 1. Distribution Plots
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1) # 2 rows, 2 columns, 1st plot
sns.histplot(summary_df['Median_Eccentricity'], kde=True, bins=20)
plt.title('Distribution of Median Eccentricity')
plt.xlabel('Median Eccentricity')

plt.subplot(2, 2, 2)
sns.histplot(summary_df['Median_Perigee_km'], kde=True, bins=20)
plt.title('Distribution of Median Perigee Altitude (km)')
plt.xlabel('Median Perigee (km)')

plt.subplot(2, 2, 3)
sns.histplot(summary_df['Median_Inclination_deg'], kde=True, bins=20)
plt.title('Distribution of Median Inclination (degrees)')
plt.xlabel('Median Inclination (deg)')

plt.subplot(2, 2, 4)
sns.histplot(summary_df['Median_Bstar'], kde=False) # KDE might be noisy for Bstar
plt.title('Distribution of Median B* Drag Term')
plt.xlabel('Median B*')
plt.xlim(summary_df['Median_Bstar'].quantile(0.01), summary_df['Median_Bstar'].quantile(0.99)) # Zoom if outliers

plt.tight_layout() # Adjust layout to prevent overlap
plot_filename = os.path.join(PLOT_OUTPUT_DIRECTORY, 'distributions.png')
plt.savefig(plot_filename)
print(f"Saved: {plot_filename}")
plt.close() # Close the figure to free memory

# 2. Categorical Count Plots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1) # 1 row, 3 columns, 1st plot
sns.countplot(y=summary_df['Eccentricity_Category'], order=summary_df['Eccentricity_Category'].value_counts().index)
plt.title('Object Count by Eccentricity Category')
plt.xlabel('Number of Objects')
plt.ylabel('Category')

plt.subplot(1, 3, 2)
sns.countplot(y=summary_df['Altitude_Category'], order=summary_df['Altitude_Category'].value_counts().index)
plt.title('Object Count by Altitude Category')
plt.xlabel('Number of Objects')
plt.ylabel('') # Remove redundant label

plt.subplot(1, 3, 3)
sns.countplot(y=summary_df['Inclination_Category'], order=summary_df['Inclination_Category'].value_counts().index)
plt.title('Object Count by Inclination Category')
plt.xlabel('Number of Objects')
plt.ylabel('') # Remove redundant label

plt.tight_layout()
plot_filename = os.path.join(PLOT_OUTPUT_DIRECTORY, 'category_counts.png')
plt.savefig(plot_filename)
print(f"Saved: {plot_filename}")
plt.close()

# 3. Relationship Plot (Example: Perigee vs Eccentricity, colored by category)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=summary_df, 
    x='Median_Eccentricity', 
    y='Median_Perigee_km', 
    hue='Eccentricity_Category', # Color points by category
    alpha=0.7 # Add transparency if points overlap
)
plt.title('Median Perigee vs. Median Eccentricity')
plt.xlabel('Median Eccentricity')
plt.ylabel('Median Perigee Altitude (km)')
# Optional: Limit axes if needed based on data range
# plt.xlim(0, max(0.2, summary_df['Median_Eccentricity'].max())) 
# plt.ylim(min(100, summary_df['Median_Perigee_km'].min()), summary_df['Median_Perigee_km'].max())

plt.tight_layout()
plot_filename = os.path.join(PLOT_OUTPUT_DIRECTORY, 'perigee_vs_eccentricity.png')
plt.savefig(plot_filename)
print(f"Saved: {plot_filename}")
plt.close()


# --- You can add more plots here! ---
# Example: Inclination vs Perigee
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=summary_df, 
    x='Median_Inclination_deg', 
    y='Median_Perigee_km', 
    hue='Inclination_Category', # Color points by category
    alpha=0.7 
)
plt.title('Median Perigee vs. Median Inclination')
plt.xlabel('Median Inclination (degrees)')
plt.ylabel('Median Perigee Altitude (km)')

plt.tight_layout()
plot_filename = os.path.join(PLOT_OUTPUT_DIRECTORY, 'perigee_vs_inclination.png')
plt.savefig(plot_filename)
print(f"Saved: {plot_filename}")
plt.close()


print("\n--- Plot generation finished ---")
# plt.show() # Use this if you want plots to pop up interactively instead of just saving