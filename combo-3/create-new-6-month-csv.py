import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import re

GM = 398600441800000.0
GM13 = GM ** (1.0/3.0)
MRAD = 6378.137
PI = math.pi
TPI86 = 2.0 * PI / 86400.0

INPUT_DIR = '6-month-data'
OUTPUT_DIR = '6-month-csv'

def tle_epoch_to_datetime(tle_epoch_str):
    year_short = int(tle_epoch_str[:2])
    day_of_year_fraction = float(tle_epoch_str[2:])
    
    if year_short < 57:
        year = 2000 + year_short
    else:
        year = 1900 + year_short
    
    start_of_year = datetime(year, 1, 1)
    delta_days = timedelta(days=(day_of_year_fraction - 1))
    epoch_datetime = start_of_year + delta_days
    return epoch_datetime

def parse_scientific_notation(field):
    field = field.strip()
    match = re.match(r'([ +-])?(\d+)([+-]\d)', field)
    if match:
        sign_char, mantissa_str, exponent_str = match.groups()
        sign = -1.0 if sign_char == '-' else 1.0
        mantissa = float(f'0.{mantissa_str}')
        exponent = int(exponent_str)
        return sign * mantissa * (10 ** exponent)
    else:
        try:
            if float(field) == 0.0:
                return 0.0
        except ValueError:
            pass
        return None

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".txt"):
        norad_id = filename[:-4]
        filepath = os.path.join(INPUT_DIR, filename)
        
        epochs = []
        epochs_utc = []
        perigee_altitudes = []
        apogee_altitudes = []
        mean_motions = []
        mean_motion_derivatives = []
        inclinations = []
        eccentricities = []
        bstars = []
        semi_major_axes = []  # Added list for semi-major axis values
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines) - 1:
                line1 = lines[i].strip()
                line2 = lines[i+1].strip()
                
                if line1.startswith('1 ') and line2.startswith('2 '):
                    try:
                        epoch_str = line1[18:32].strip()
                        epoch_dt = tle_epoch_to_datetime(epoch_str)
                        epoch_utc = epoch_dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                        
                        first_deriv_str = line1[33:43].strip()
                        bstar_str = line1[53:61].strip()
                        
                        inclination_deg = float(line2[8:16].strip())
                        eccentricity_str = line2[26:33].strip()
                        mean_motion_rev_day = float(line2[52:63].strip())
                        
                        eccentricity = float(f'0.{eccentricity_str}')
                        first_deriv = float(first_deriv_str) * 2.0
                        bstar = parse_scientific_notation(bstar_str)
                        
                        if bstar is None:
                            i += 2
                            continue
                        
                        mmoti = mean_motion_rev_day
                        ecc = eccentricity
                        
                        if mmoti <= 0:
                            i += 2
                            continue
                        
                        sma = GM13 / ((TPI86 * mmoti) ** (2.0 / 3.0)) / 1000.0
                        apo = sma * (1.0 + ecc) - MRAD
                        per = sma * (1.0 - ecc) - MRAD
                        
                        epochs.append(epoch_dt)
                        epochs_utc.append(epoch_utc)
                        perigee_altitudes.append(per)
                        apogee_altitudes.append(apo)
                        mean_motions.append(mean_motion_rev_day)
                        mean_motion_derivatives.append(first_deriv)
                        inclinations.append(inclination_deg)
                        eccentricities.append(eccentricity)
                        bstars.append(bstar)
                        semi_major_axes.append(sma)  
                        
                        i += 2
                    except Exception:
                        i += 2
                else:
                    i += 1
                    
            if epochs:
                data = pd.DataFrame({
                    'epoch': epochs,
                    'epoch_utc': epochs_utc,
                    'apogee_altitude': apogee_altitudes,
                    'perigee_altitude': perigee_altitudes,
                    'mean_motion': mean_motions,
                    'mean_motion_derivative': mean_motion_derivatives,
                    'inclination': inclinations,
                    'eccentricity': eccentricities,
                    'bstar': bstars,
                    'semi_major_axis': semi_major_axes 
                })
                
                data = data.sort_values(by='epoch').reset_index(drop=True)
                
                output_filepath = os.path.join(OUTPUT_DIR, f"{norad_id}.csv")
                data.to_csv(output_filepath, index=False)
                
        except Exception:
            continue

print(f"Processing complete. CSV files saved to '{OUTPUT_DIR}' directory.")