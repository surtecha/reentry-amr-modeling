import requests
import pandas as pd
import numpy as np
from sgp4.api import Satrec
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import time
import sys
import configparser
import os
import traceback # Added for better error reporting if needed

# --- Configuration ---
config = configparser.ConfigParser()
config.read('config.ini') # Assumes file exists and is valid

SPACETRACK_USER = config['SPACE_TRACK']['username']
SPACETRACK_PASS = config['SPACE_TRACK']['password']

SPACETRACK_URI = "https://www.space-track.org"
SPACETRACK_LOGIN_URL = SPACETRACK_URI + '/ajaxauth/login'
SPACETRACK_LOGOUT_URL = SPACETRACK_URI + '/ajaxauth/logout'
SPACETRACK_QUERY_URL = SPACETRACK_URI + '/basicspacedata/query'

# --- Parameters ---
INPUT_CSV_FILE = '/Users/suryatejachalla/Research/Re-entry-Prediction/Data/data.csv' # Assumes file exists
NORAD_COLUMN_NAME = 'Norad_id' # Assumes column exists
OUTPUT_DIRECTORY = 'tle_data_output' # Directory to save CSV files
MONTHS_PRIOR = 3
GM_EARTH = 3.986008e14
R_EARTH_KM_FOR_PLOT = 6378.135

# --- Rate Limiting Parameters ---
MIN_REQUEST_INTERVAL_SEC = 3.0
MAX_REQUESTS_PER_MINUTE = 25
MAX_REQUESTS_PER_HOUR = 280
HOUR_SLEEP_DURATION_SEC = 3610

# --- Helper Function ---
def calculate_epoch_dt(satrec_obj):
    """Calculates datetime object from Satrec epoch fields."""
    year = satrec_obj.epochyr
    if year < 57: year += 2000
    else: year += 1900
    day = satrec_obj.epochdays
    day_int = int(day)
    frac_day = day - day_int
    # Ensure base date is UTC
    base_date = datetime(year, 1, 1, tzinfo=timezone.utc)
    # Calculate the final datetime object, preserving UTC timezone
    dt_epoch = base_date + timedelta(days=day_int - 1, seconds=frac_day * 86400.0)
    return dt_epoch

# --- Main Script ---

# 1. Read NORAD IDs from CSV
input_df = pd.read_csv(INPUT_CSV_FILE)
norad_ids = input_df[NORAD_COLUMN_NAME].unique().tolist()
norad_ids = [int(nid) for nid in norad_ids if pd.notna(nid)]
print(f"Found {len(norad_ids)} unique NORAD IDs in '{INPUT_CSV_FILE}'.")

if not norad_ids:
    print("No valid NORAD IDs found. Exiting.")
    sys.exit(0)

# 2. Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
print(f"Output CSV files will be saved in '{OUTPUT_DIRECTORY}/'")

# 3. Initialize Session and Rate Limiting Variables
session = requests.Session()
last_request_time = time.monotonic()
request_times_minute = []
request_times_hour = []
logged_in = False

try:
    # 4. Login to Space-Track (once)
    print("Attempting Space-Track login...")
    login_data = {'identity': SPACETRACK_USER, 'password': SPACETRACK_PASS}
    login_response = session.post(SPACETRACK_LOGIN_URL, data=login_data)
    login_response.raise_for_status() # Still check HTTP errors for login
    print("Space-Track login successful.")
    logged_in = True
    current_time = time.monotonic()
    request_times_minute.append(current_time)
    request_times_hour.append(current_time)
    last_request_time = current_time

    # 5. Loop through each NORAD ID
    total_ids = len(norad_ids)
    for i, norad_id in enumerate(norad_ids):
        print(f"\n--- Processing NORAD ID {norad_id} ({i+1}/{total_ids}) ---")

        # Inner try block for individual satellite processing robustness
        try:
            # --- Rate Limiting Check ---
            current_time = time.monotonic()
            # Remove timestamps older than 1 minute / 1 hour
            request_times_minute = [t for t in request_times_minute if current_time - t < 60]
            request_times_hour = [t for t in request_times_hour if current_time - t < 3600]

            # Check per-minute limit
            if len(request_times_minute) >= MAX_REQUESTS_PER_MINUTE:
                wait_time_minute = 60 - (current_time - request_times_minute[0]) + 1 # Wait until the oldest request expires + 1s buffer
                print(f"  Minute rate limit hit ({len(request_times_minute)} requests). Waiting {wait_time_minute:.1f}s...")
                time.sleep(wait_time_minute)
                # Re-evaluate time and clear lists after waiting
                current_time = time.monotonic()
                request_times_minute = [t for t in request_times_minute if current_time - t < 60]
                request_times_hour = [t for t in request_times_hour if current_time - t < 3600]

             # Check per-hour limit
            if len(request_times_hour) >= MAX_REQUESTS_PER_HOUR:
                 # Find the time until the oldest request is an hour old
                 wait_time_hour = 3600 - (current_time - request_times_hour[0]) + 5 # Wait until oldest request expires + 5s buffer
                 print(f"  Hourly rate limit hit ({len(request_times_hour)} requests). Waiting {wait_time_hour:.1f}s (approx {wait_time_hour/60:.1f} min)...")
                 time.sleep(wait_time_hour)
                 # Re-evaluate time and clear lists after waiting
                 current_time = time.monotonic()
                 request_times_minute = [t for t in request_times_minute if current_time - t < 60]
                 request_times_hour = [t for t in request_times_hour if current_time - t < 3600]

            # Check minimum interval
            time_since_last = current_time - last_request_time
            if time_since_last < MIN_REQUEST_INTERVAL_SEC:
                wait_interval = MIN_REQUEST_INTERVAL_SEC - time_since_last
                time.sleep(wait_interval)

            # --- Fetch Latest TLE ---
            print(f"  Fetching latest TLE...")
            latest_tle_query_url = (f"{SPACETRACK_QUERY_URL}/class/tle/"
                                    f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20desc/limit/1/format/tle")
            try:
                latest_tle_response = session.get(latest_tle_query_url)
                latest_tle_response.raise_for_status()
                latest_tle_data = latest_tle_response.text.strip()
                # Update rate limit counters *after* successful request
                current_time = time.monotonic()
                request_times_minute.append(current_time)
                request_times_hour.append(current_time)
                last_request_time = current_time
            except requests.exceptions.RequestException as e:
                print(f"  Error fetching latest TLE for {norad_id}: {e}. Skipping ID.")
                continue # Skip to next NORAD ID

            # --- Parse Latest TLE and Determine Date Range ---
            if not latest_tle_data or len(latest_tle_data.splitlines()) < 2:
                print(f"  No valid latest TLE found for {norad_id}. Skipping ID.")
                continue

            latest_lines = latest_tle_data.splitlines()
            tle_line1_latest, tle_line2_latest = None, None
            for k in range(len(latest_lines) - 1):
                 line1_cand, line2_cand = latest_lines[k].strip(), latest_lines[k+1].strip()
                 if line1_cand.startswith('1 ') and line2_cand.startswith('2 '):
                      tle_line1_latest, tle_line2_latest = line1_cand, line2_cand
                      break # Found the TLE pair
            if not tle_line1_latest:
                 print(f"  Could not parse TLE lines from latest data for {norad_id}. Skipping ID.")
                 continue

            try:
                sat_latest = Satrec.twoline2rv(tle_line1_latest, tle_line2_latest)
                # Get latest epoch and remove timezone for relativedelta calculation
                latest_epoch_dt_naive = calculate_epoch_dt(sat_latest).replace(tzinfo=None)
            except Exception as e:
                print(f"  Error parsing latest TLE object for {norad_id}: {e}. Skipping ID.")
                continue

            END_DATE_dt = latest_epoch_dt_naive
            START_DATE_dt = END_DATE_dt - relativedelta(months=MONTHS_PRIOR)
            END_DATE_str = END_DATE_dt.strftime('%Y-%m-%d')
            START_DATE_str = START_DATE_dt.strftime('%Y-%m-%d')
            print(f"  Latest Epoch: {latest_epoch_dt_naive.strftime('%Y-%m-%d %H:%M:%S')} UTC. Range: {START_DATE_str} to {END_DATE_str}")

            # --- Rate Limiting Check (Before Historical) ---
            # (Repeat the check - essential before *each* request)
            current_time = time.monotonic()
            request_times_minute = [t for t in request_times_minute if current_time - t < 60]
            request_times_hour = [t for t in request_times_hour if current_time - t < 3600]
            if len(request_times_minute) >= MAX_REQUESTS_PER_MINUTE:
                wait_time_minute = 60 - (current_time - request_times_minute[0]) + 1
                print(f"  Minute rate limit hit ({len(request_times_minute)} requests). Waiting {wait_time_minute:.1f}s...")
                time.sleep(wait_time_minute)
                current_time = time.monotonic() # Update time after sleep
                request_times_minute = [t for t in request_times_minute if current_time - t < 60] # Reclean
                request_times_hour = [t for t in request_times_hour if current_time - t < 3600] # Reclean

            if len(request_times_hour) >= MAX_REQUESTS_PER_HOUR:
                 wait_time_hour = 3600 - (current_time - request_times_hour[0]) + 5
                 print(f"  Hourly rate limit hit ({len(request_times_hour)} requests). Waiting {wait_time_hour:.1f}s...")
                 time.sleep(wait_time_hour)
                 current_time = time.monotonic()
                 request_times_minute = [t for t in request_times_minute if current_time - t < 60]
                 request_times_hour = [t for t in request_times_hour if current_time - t < 3600]

            time_since_last = current_time - last_request_time
            if time_since_last < MIN_REQUEST_INTERVAL_SEC:
                wait_interval = MIN_REQUEST_INTERVAL_SEC - time_since_last
                time.sleep(wait_interval)

            # --- Fetch Historical TLEs ---
            print(f"  Fetching historical TLEs...")
            query_url = (f"{SPACETRACK_QUERY_URL}/class/tle/NORAD_CAT_ID/{norad_id}/"
                         f"EPOCH/{START_DATE_str}--{END_DATE_str}/orderby/EPOCH%20asc/format/tle")
            try:
                tle_response = session.get(query_url)
                tle_response.raise_for_status()
                tle_data = tle_response.text
                 # Update rate limit counters *after* successful request
                current_time = time.monotonic()
                request_times_minute.append(current_time)
                request_times_hour.append(current_time)
                last_request_time = current_time
                print(f"  Fetched {len(tle_data.splitlines()) // 2} potential historical sets.")
            except requests.exceptions.RequestException as e:
                print(f"  Error fetching historical TLEs for {norad_id}: {e}. Skipping ID.")
                continue # Skip to next NORAD ID

            # --- Parse Historical TLEs ---
            print("  Parsing historical TLEs...")
            lines = tle_data.strip().splitlines()
            tles = []
            line_idx = 0
            parse_errors = 0
            while line_idx < len(lines) - 1:
                line1, line2 = lines[line_idx].strip(), lines[line_idx+1].strip()
                if not line1.startswith('1 ') or not line2.startswith('2 '):
                    line_idx += 1
                    continue
                try:
                    sat = Satrec.twoline2rv(line1, line2)
                    dt_epoch = calculate_epoch_dt(sat) # This now returns UTC datetime

                    # --- CORRECTED N-DOT/2 PARSING (from User's original) ---
                    ndot_raw = line1[33:43].strip()
                    ndot_formatted = "0.0" # Default value
                    if ndot_raw: # Process only if not empty
                        if '.' not in ndot_raw and len(ndot_raw) > 0: # Check length > 0 and no existing decimal
                            sign = ''
                            if ndot_raw.startswith(('-', '+')):
                                sign = ndot_raw[0]
                                num_part_raw = ndot_raw[1:] # Use separate var for raw number part
                            else:
                                num_part_raw = ndot_raw # No sign explicitly given

                            exp_sign = ''
                            exp_val = ''
                            num_part = num_part_raw # This will hold number part without exponent

                            # Check for exponent using '-', '+', or space delimiter AT THE END
                            if len(num_part_raw) > 1: # Need at least one digit and exponent part
                                last_char = num_part_raw[-1]
                                second_last_char = num_part_raw[-2]
                                if last_char.isdigit():
                                    if second_last_char in ['-', '+', ' ']:
                                         num_part = num_part_raw[:-2]
                                         exp_sign = '-' if second_last_char == '-' else '+'
                                         exp_val = last_char
                                    # else: no exponent found, num_part remains num_part_raw

                            if exp_val: # If an exponent was found
                                ndot_formatted = f"{sign}.{num_part}e{exp_sign}{exp_val}"
                            elif num_part: # No exponent found, just add decimal
                                 ndot_formatted = f"{sign}.{num_part}"
                            # Handle edge cases from parsing
                            if not num_part and exp_val:
                                ndot_formatted = "0.0"
                            elif not num_part and not exp_val and sign:
                                 ndot_formatted = "0.0"

                        else: # Already has decimal or is simple (e.g., '0')
                             ndot_formatted = ndot_raw
                             # Minimal validation for simple cases like just '+' or '-'
                             if ndot_formatted == '+' or ndot_formatted == '-':
                                 ndot_formatted = "0.0"

                    # Now parse the potentially corrected scientific notation string
                    try:
                        ndot_term = float(ndot_formatted)
                    except ValueError:
                        # print(f"  Warning: Could not parse ndot_term '{line1[33:43]}' (processed as '{ndot_formatted}') for TLE epoch {dt_epoch}. Setting to 0.")
                        ndot_term = 0.0
                        parse_errors += 1
                    # --- END OF CORRECTED N-DOT/2 PARSING ---

                    mean_motion_rad_min = sat.no_kozai
                    mean_motion_rev_day = mean_motion_rad_min * (1440.0 / (2.0 * np.pi))
                    tles.append({
                        'Epoch': dt_epoch, # Keep timezone-aware datetime from helper
                        'MeanMotion_rad_min': mean_motion_rad_min,
                        'MeanMotion_revday': mean_motion_rev_day,
                        'Eccentricity': sat.ecco,
                        'Inclination_deg': np.degrees(sat.inclo),
                        'RAAN_deg': np.degrees(sat.nodeo),
                        'ArgPerigee_deg': np.degrees(sat.argpo),
                        'MeanAnomaly_deg': np.degrees(sat.mo),
                        'Bstar': sat.bstar,
                        'ndot_TERM_from_TLE': ndot_term, # Use the parsed term
                    })
                    line_idx += 2
                except Exception as e:
                    # print(f"  ERROR processing TLE pair: {e}. Lines:\n{line1}\n{line2}") # Optional debug
                    parse_errors += 1
                    line_idx += 2 # Skip pair on error

            if parse_errors > 0: print(f"  Parsing completed with {parse_errors} errors.")
            if not tles:
                print(f"  No valid historical TLEs parsed for {norad_id}. Skipping save.")
                continue
            print(f"  Parsed {len(tles)} TLE sets.")
            tle_df_raw = pd.DataFrame(tles)

            # --- Calculate Derived Parameters ---
            print("  Calculating derived parameters...")
            tle_df_raw['MeanMotion_rad_sec'] = tle_df_raw['MeanMotion_rad_min'] / 60.0
            n_rad_per_sec = tle_df_raw['MeanMotion_rad_sec']
            valid_n = n_rad_per_sec > 1e-9
            tle_df_raw['SemiMajorAxis_m'] = np.nan
            tle_df_raw.loc[valid_n, 'SemiMajorAxis_m'] = (GM_EARTH / n_rad_per_sec[valid_n]**2)**(1.0/3.0)
            tle_df_raw['SemiMajorAxis_km'] = tle_df_raw['SemiMajorAxis_m'] / 1000.0
            tle_df_raw['Period_sec'] = np.nan
            tle_df_raw.loc[valid_n, 'Period_sec'] = 2.0 * np.pi / n_rad_per_sec[valid_n]
            valid_sma = tle_df_raw['SemiMajorAxis_m'].notna() & (tle_df_raw['SemiMajorAxis_m'] > 0)
            valid_ecc = tle_df_raw['Eccentricity'].notna() & (tle_df_raw['Eccentricity'] >= 0) & (tle_df_raw['Eccentricity'] < 1)
            valid_rows_for_alt = valid_sma & valid_ecc
            tle_df_raw['AltitudePerigee_km'] = np.nan
            tle_df_raw['AltitudeApogee_km'] = np.nan
            if valid_rows_for_alt.any():
                 rp_m = tle_df_raw.loc[valid_rows_for_alt, 'SemiMajorAxis_m'] * (1.0 - tle_df_raw.loc[valid_rows_for_alt, 'Eccentricity'])
                 ra_m = tle_df_raw.loc[valid_rows_for_alt, 'SemiMajorAxis_m'] * (1.0 + tle_df_raw.loc[valid_rows_for_alt, 'Eccentricity'])
                 tle_df_raw.loc[valid_rows_for_alt, 'AltitudePerigee_km'] = (rp_m / 1000.0) - R_EARTH_KM_FOR_PLOT
                 tle_df_raw.loc[valid_rows_for_alt, 'AltitudeApogee_km'] = (ra_m / 1000.0) - R_EARTH_KM_FOR_PLOT

            original_rows = len(tle_df_raw)
            essential_cols = ['SemiMajorAxis_km', 'AltitudePerigee_km', 'AltitudeApogee_km', 'Period_sec']
            cols_to_check = [col for col in essential_cols if col in tle_df_raw.columns]
            tle_df = tle_df_raw.dropna(subset=cols_to_check).copy()
            dropped_rows = original_rows - len(tle_df)
            if dropped_rows > 0: print(f"  Dropped {dropped_rows} rows due to NaN derived values.")
            if tle_df.empty:
                print("  No valid data remains after calculating derived parameters. Skipping save.")
                continue

            # Set index *before* saving, ensures 'Epoch' column is the index
            tle_df.set_index('Epoch', inplace=True)
            tle_df.sort_index(inplace=True)

            # --- Save DataFrame to CSV ---
            output_filename = os.path.join(OUTPUT_DIRECTORY, f"{norad_id}.csv")
            # Save with index=True to include the 'Epoch' index column
            tle_df.to_csv(output_filename, index=True)
            print(f"  Successfully saved data to '{output_filename}'")

        except Exception as e: # Catch-all for unexpected errors during single satellite processing
             print(f"!! Unexpected Error processing NORAD ID {norad_id}: {e}")
             traceback.print_exc() # Print detailed traceback
             print(f"!! Skipping NORAD ID {norad_id} due to this error.")

finally:
    # 6. Logout
    if logged_in and session:
        print("\nLogging out from Space-Track...")
        try:
            session.get(SPACETRACK_LOGOUT_URL) # Attempt logout
            print("Logout successful.")
        except requests.exceptions.RequestException as e:
             print(f"Warning: Error during logout: {e}") # Non-fatal
        finally:
             session.close() # Ensure session is closed
    elif session:
         session.close() # Also close if login failed but session object exists

print("\n--- Script finished ---")