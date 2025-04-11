import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
TLE_CSV_DIRECTORY = 'tle_data_output'
MIN_ROWS_THRESHOLD = 75 # Use a slightly higher threshold for rate calculation? Maybe 100?
MIN_TIME_DIFF_SEC = 60 # Avoid dividing by zero or tiny time diffs (e.g., > 1 minute)

# --- Function to Process Single Object ---
def analyze_object_decay_rate(csv_path, norad_id):
    try:
        # Read data, parse dates, set index, sort
        df = pd.read_csv(csv_path, parse_dates=['Epoch'], index_col='Epoch')
        df.sort_index(inplace=True)

        if len(df) < MIN_ROWS_THRESHOLD:
            return None # Skip if not enough data points

        # Check for required columns
        required_cols = ['MeanMotion_revday', 'AltitudePerigee_km', 'Eccentricity', 'Bstar']
        if not all(col in df.columns for col in required_cols):
             # print(f"Warning: Missing required columns in {os.path.basename(csv_path)}. Skipping.")
             return None

        # Calculate differences between consecutive rows
        df['Time_Diff_sec'] = df.index.to_series().diff().dt.total_seconds()
        df['MM_Diff'] = df['MeanMotion_revday'].diff()

        # Filter out rows with insufficient time difference or NaN diffs
        valid_rates = df[(df['Time_Diff_sec'] >= MIN_TIME_DIFF_SEC) & df['MM_Diff'].notna()]

        if len(valid_rates) < MIN_ROWS_THRESHOLD / 2: # Need enough valid rate points
            # print(f"Warning: Not enough valid rate points for {norad_id}. Skipping.")
            return None

        # Calculate instantaneous rate (change in Mean Motion per day)
        # Convert Time_Diff_sec to days for rate unit consistency
        valid_rates['Rate_MM_per_day'] = (valid_rates['MM_Diff'] / (valid_rates['Time_Diff_sec'] / 86400.0))

        # --- Calculate Median Rate and Parameters ---
        # Use median of the valid instantaneous rates
        median_rate_mm = valid_rates['Rate_MM_per_day'].median()

        # Calculate median parameters from the original dataframe (over the whole period)
        median_perigee = df['AltitudePerigee_km'].median()
        median_ecc = df['Eccentricity'].median()
        median_bstar = df['Bstar'].median()

        # Basic sanity check on rate
        if pd.isna(median_rate_mm) or median_rate_mm < -1e-3: # Expect positive or very small negative
              # print(f"Warning: Suspicious median rate ({median_rate_mm:.2e}) for {norad_id}. Skipping.")
              return None

        return {
            'NORAD_ID': norad_id,
            'Median_Rate_MM_per_day': median_rate_mm,
            'Median_Perigee_km': median_perigee,
            'Median_Eccentricity': median_ecc,
            'Median_Bstar': median_bstar,
            'Num_Valid_Rates': len(valid_rates) # Info about data quality
        }

    except Exception as e:
        print(f"!! Error processing {os.path.basename(csv_path)}: {e}")
        return None

# --- Main Loop ---
print("Starting combined decay rate analysis...")
csv_files = glob.glob(os.path.join(TLE_CSV_DIRECTORY, '*.csv'))
decay_analysis_data = []
processed_files = 0
skipped_files = 0

for csv_file_path in csv_files:
    filename = os.path.basename(csv_file_path)
    try:
        norad_id = int(filename.split('.')[0])
        result = analyze_object_decay_rate(csv_file_path, norad_id)
        if result:
            decay_analysis_data.append(result)
            processed_files += 1
        else:
            skipped_files += 1
    except (ValueError, IndexError):
        skipped_files += 1
        continue
    except Exception as e:
         print(f"!! Outer loop error for {filename}: {e}")
         skipped_files += 1


print(f"\nProcessed {processed_files} objects for decay rate analysis.")
print(f"Skipped {skipped_files} objects (insufficient data, errors, etc.).")

if not decay_analysis_data:
    print("No objects had sufficient data for combined analysis. Exiting.")
    exit()

# --- Create DataFrame and Analyze ---
decay_analysis_df = pd.DataFrame(decay_analysis_data)
decay_analysis_df.set_index('NORAD_ID', inplace=True)

print("\n--- Combined Decay Analysis DataFrame (First 5 rows) ---")
print(decay_analysis_df.head())

# Calculate and visualize correlation matrix
print("\n--- Correlation Matrix ---")
# Select columns for correlation
cols_for_corr = ['Median_Rate_MM_per_day', 'Median_Perigee_km', 'Median_Eccentricity', 'Median_Bstar']
corr_matrix = decay_analysis_df[cols_for_corr].corr()
print(corr_matrix)

# Visualize heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Median Decay Rate and Orbital Parameters')
plt.tight_layout()

# Save the plot
plot_filename = 'decay_rate_correlation_heatmap.png'
plt.savefig(plot_filename)
print(f"\nCorrelation heatmap saved to '{plot_filename}'")
# plt.show() # Optional: display plot interactively
plt.close()

print("\n--- Analysis Interpretation ---")
print("Key correlations to examine:")
print(f"- Rate vs. Perigee: {corr_matrix.loc['Median_Rate_MM_per_day', 'Median_Perigee_km']:.3f} (Expected strong negative)")
print(f"- Rate vs. Eccentricity: {corr_matrix.loc['Median_Rate_MM_per_day', 'Median_Eccentricity']:.3f} (Expected positive)")
print(f"- Rate vs. Bstar: {corr_matrix.loc['Median_Rate_MM_per_day', 'Median_Bstar']:.3f} (Expected positive)")
print("\nNote: These correlations DO NOT account for external factors like solar activity.")
print("Higher POSITIVE Rate_MM means FASTER decay (mean motion increases more rapidly).")

# --- Save the analysis data ---
summary_filename = 'decay_rate_summary.csv'
decay_analysis_df.to_csv(summary_filename)
print(f"Combined decay analysis data saved to '{summary_filename}'")

print("\n--- Combined Analysis Finished ---")