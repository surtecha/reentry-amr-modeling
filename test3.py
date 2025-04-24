import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import linregress
import shutil # For moving files

def analyze_and_filter_files(input_dir="r/Users/suryatejachalla/Research/reentry-amr-modeling/data/raw-fitting_copy", removed_dir_suffix="_removed_by_filter"):
    """
    Analyzes processed CSV files and filters out those that don't meet
    specific trend criteria for orbital parameters (perigee decrease,
    mean motion/derivative increase, BSTAR variation).

    Args:
        input_dir (str): Directory containing the processed CSV files.
        removed_dir_suffix (str): Suffix appended to input_dir to create
                                   the directory for removed files.
    """

    # --- Configuration Thresholds ---
    # Min data points required for trend analysis
    MIN_DATA_POINTS = 5

    # BSTAR Analysis: Standard deviation below this is considered 'flat'
    BSTAR_FLATNESS_THRESHOLD = 1e-6

    # --- End Configuration ---

    output_dir = input_dir
    removed_dir = f"{input_dir}{removed_dir_suffix}"

    # Create directory for removed files if it doesn't exist
    os.makedirs(removed_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))

    total_files = len(csv_files)
    removed_count = 0
    kept_count = 0
    removal_reasons = {}

    print(f"Starting analysis of {total_files} files in '{output_dir}'...")

    for i, file_path in enumerate(csv_files, 1):
        object_id = os.path.basename(file_path).split('.')[0]
        print(f"Analyzing {i}/{total_files}: {object_id}...", end=" ")

        remove_file = False
        reasons = []

        try:
            df = pd.read_csv(file_path)

            # Check for minimum data points
            if len(df) < MIN_DATA_POINTS:
                reasons.append(f"Insufficient data points ({len(df)} < {MIN_DATA_POINTS})")
                remove_file = True
            else:
                # Time axis for regression
                x_time = np.arange(len(df))

                # --- 1. Perigee Altitude Check ---
                perigee_alt = df['perigee_alt'].values
                # Check for NaN/Inf
                if np.isnan(perigee_alt).any() or np.isinf(perigee_alt).any():
                     reasons.append("NaN/Inf in perigee_alt")
                     remove_file = True
                # Check for constant value (avoid division by zero / linregress issues)
                elif np.all(perigee_alt == perigee_alt[0]):
                     reasons.append("Perigee altitude is constant")
                     remove_file = True # Constant perigee doesn't decrease
                else:
                    # Perform linear regression
                    slope, intercept, r_value, p_value, std_err = linregress(x_time, perigee_alt)

                    # Ensure perigee trend is decreasing (slope < 0)
                    if slope >= 0:
                        reasons.append(f"Perigee not decreasing (slope={slope:.2e})")
                        remove_file = True

                    # ## REMOVED CHECK: Perigee Fluctuation (R-squared) ##
                    # elif r_squared < PERIGEE_MIN_R_SQUARED:
                    #      reasons.append(f"Perigee fluctuates too much (R^2={r_squared:.2f} < {PERIGEE_MIN_R_SQUARED})")
                    #      remove_file = True

                # --- 2. Mean Motion / Derivative Check ---
                # Only check if not already marked for removal
                if not remove_file:
                    mean_motion = df['mean_motion'].values
                    mean_motion_deriv = df['mean_motion_deriv'].values
                    mm_increasing = False
                    mmd_increasing = False

                    # Check Mean Motion trend (if not constant and not NaN/Inf)
                    if not np.all(mean_motion == mean_motion[0]) and not (np.isnan(mean_motion).any() or np.isinf(mean_motion).any()):
                         slope_mm, _, _, _, _ = linregress(x_time, mean_motion)
                         if slope_mm > 0:
                             mm_increasing = True

                    # Check Mean Motion Derivative trend (if not constant and not NaN/Inf)
                    if not np.all(mean_motion_deriv == mean_motion_deriv[0]) and not (np.isnan(mean_motion_deriv).any() or np.isinf(mean_motion_deriv).any()):
                         slope_mmd, _, _, _, _ = linregress(x_time, mean_motion_deriv)
                         if slope_mmd > 0:
                              mmd_increasing = True

                    # Require Mean Motion or its derivative to be increasing
                    if not (mm_increasing or mmd_increasing):
                        reasons.append("Neither Mean Motion nor Derivative increasing")
                        remove_file = True

                # --- 3. BSTAR Check ---
                 # Only check if not already marked for removal
                if not remove_file:
                    bstar = df['bstar'].values
                    # Check for NaN/Inf first
                    if np.isnan(bstar).any() or np.isinf(bstar).any():
                        reasons.append("NaN/Inf in bstar")
                        remove_file = True
                    else:
                        # Ensure BSTAR shows variation (std dev > threshold)
                        if bstar.std() < BSTAR_FLATNESS_THRESHOLD:
                            reasons.append(f"BSTAR is flat (std_dev={bstar.std():.2e} < {BSTAR_FLATNESS_THRESHOLD})")
                            remove_file = True

        except FileNotFoundError:
            print(f"\nError: File not found {file_path}")
            continue # Skip to next file
        except pd.errors.EmptyDataError:
            print(f"\nError: File is empty {file_path}")
            reasons.append("File is empty")
            remove_file = True # Mark for removal
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            reasons.append(f"Processing error: {e}")
            remove_file = True # Mark for removal

        # --- Action: Move or Keep ---
        if remove_file:
            removed_count += 1
            removal_reasons[object_id] = reasons
            try:
                dest_path = os.path.join(removed_dir, os.path.basename(file_path))
                shutil.move(file_path, dest_path)
                print(f"Removed ({', '.join(reasons)})")
            except Exception as e:
                print(f"\nError moving file {file_path} to {removed_dir}: {e}")
                # Decrement count if move failed, log error
                removed_count -= 1
                kept_count += 1
        else:
            kept_count += 1
            print("Kept")

    # --- Print Statistics ---
    print("\n--- Filtering Statistics ---")
    print(f"Total files analyzed: {total_files}")
    print(f"Files kept: {kept_count}")
    print(f"Files removed: {removed_count}")

    if removed_count > 0:
        print("\nReasons for removal:")
        # Sort reasons by object_id for consistent output
        for object_id in sorted(removal_reasons.keys()):
             print(f"- {object_id}: {'; '.join(removal_reasons[object_id])}")
        print(f"\nRemoved files moved to: '{removed_dir}'")

if __name__ == "__main__":
    # Set the directory containing the processed CSV files
    processed_data_directory = "/Users/suryatejachalla/Research/reentry-amr-modeling/data/raw-fitting_copy"
    analyze_and_filter_files(input_dir=processed_data_directory)