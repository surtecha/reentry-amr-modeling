import os
import re
import argparse
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
import matplotlib.dates as mdates

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

def calculate_perigee_altitude(line1, line2):
    """
    Calculate perigee altitude from TLE data.
    Returns altitude in kilometers above Earth's surface.
    """
    try:
        # Parse the TLE data
        satellite = twoline2rv(line1, line2, wgs72)
        
        # Semi-major axis (in km)
        a = satellite.a * satellite.radiusearthkm
        
        # Eccentricity
        e = satellite.ecco
        
        # Perigee altitude (km above Earth's surface)
        perigee_altitude = a * (1 - e) - satellite.radiusearthkm
        
        return perigee_altitude
    except Exception as e:
        print(f"Error calculating perigee altitude: {e}")
        return None

def read_tle_file_for_norad(norad_id, objects_dir):
    """Read TLE data for a specific NORAD ID and return a list of TLEs with epochs."""
    filepath = os.path.join(objects_dir, f"{norad_id}.txt")
    
    if not os.path.exists(filepath):
        print(f"No TLE data found for NORAD ID {norad_id}")
        return []
    
    tle_data = []
    
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
                epoch = parse_tle_epoch(line1)
                
                if epoch:
                    tle_data.append((epoch, line1, line2))
            
            i += 1
        
        # Sort by epoch
        tle_data.sort(key=lambda x: x[0])
        
        return tle_data
    
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return []

def filter_tle_data_by_time_window(tle_data, months):
    """Filter TLE data to include only entries within the specified number of months from the latest entry."""
    if not tle_data:
        return []
    
    # Find the latest epoch
    latest_epoch = max(tle_data, key=lambda x: x[0])[0]
    
    # Calculate the cutoff date
    cutoff_date = latest_epoch - timedelta(days=30.44 * months)  # Approximate days in a month
    
    # Filter the data
    filtered_data = [entry for entry in tle_data if entry[0] >= cutoff_date]
    
    return filtered_data

def plot_perigee_altitude(norad_id, perigee_data, months):
    """Plot perigee altitude over time for a specific NORAD ID."""
    if not perigee_data:
        print("No data to plot.")
        return
    
    # Extract dates and perigee values
    dates = [data[0] for data in perigee_data]
    perigee_values = [data[1] for data in perigee_data]
    
    # Create a figure and axis
    plt.figure(figsize=(12, 6))
    plt.plot(dates, perigee_values, marker='o', linestyle='-', markersize=4)
    
    # Add title and labels
    plt.title(f"Perigee Altitude for NORAD ID {norad_id} (Last {months} months)")
    plt.xlabel("Date")
    plt.ylabel("Perigee Altitude (km)")
    
    # Format the date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    if perigee_values:
        avg_perigee = np.mean(perigee_values)
        min_perigee = min(perigee_values)
        max_perigee = max(perigee_values)
        
        stats_text = f"Min: {min_perigee:.2f} km\nMax: {max_perigee:.2f} km\nAvg: {avg_perigee:.2f} km"
        plt.figtext(0.02, 0.02, stats_text, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the figure
    output_filename = f"perigee_altitude_norad_{norad_id}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_filename}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot perigee altitude for a satellite by NORAD ID')
    parser.add_argument('norad_id', type=int, help='NORAD ID of the satellite')
    parser.add_argument('--months', type=float, default=6.0, help='Time window in months (default: 6.0)')
    parser.add_argument('--objects-dir', type=str, default='objects', help='Directory containing TLE data files')
    
    args = parser.parse_args()
    
    # Check if objects directory exists
    if not os.path.exists(args.objects_dir):
        print(f"Error: Objects directory '{args.objects_dir}' not found.")
        return
    
    # Read TLE data for the specified NORAD ID
    print(f"Reading TLE data for NORAD ID {args.norad_id}...")
    tle_data = read_tle_file_for_norad(args.norad_id, args.objects_dir)
    
    if not tle_data:
        print(f"No TLE data found for NORAD ID {args.norad_id}.")
        return
    
    print(f"Found {len(tle_data)} TLE entries for NORAD ID {args.norad_id}.")
    
    # Filter TLE data by time window
    filtered_data = filter_tle_data_by_time_window(tle_data, args.months)
    print(f"Filtered to {len(filtered_data)} entries within the last {args.months} months.")
    
    # Calculate perigee altitude for each TLE
    perigee_data = []
    for epoch, line1, line2 in filtered_data:
        perigee = calculate_perigee_altitude(line1, line2)
        if perigee is not None:
            perigee_data.append((epoch, perigee))
    
    print(f"Calculated perigee altitude for {len(perigee_data)} TLE entries.")
    
    # Plot the data
    plot_perigee_altitude(args.norad_id, perigee_data, args.months)

if __name__ == "__main__":
    main()