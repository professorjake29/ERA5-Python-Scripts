# -*- coding: utf-8 -*-
"""
Created on Wed May  7 07:29:22 2025

@author: jakew
"""
import os
import cdsapi
import time

# Function to Get the current 
# working directory
def current_path():
    print("Current working directory before")
    print(os.getcwd())
    print()
 
 
# Driver's code
# Printing CWD before
current_path()
 
# Changing the CWD
os.chdir('./Downloads')
 
# Printing CWD after
current_path()

# Initialize the CDS API client
client = cdsapi.Client()

# List of lake-effect snow case dates and times
# Format: (year, month, day, time)
cases = [
    ('2004', '01', '01', '00:00'),
    ('2004', '01', '05', '12:00'),
    ('2004', '01', '10', '06:00'),
    # Add more cases here...
]

# Loop through the cases and download the data
for i, (year, month, day, time_str) in enumerate(cases, start=1):
    filename = f"era5_vort_1000_500hPa_{year}{month}{day}_{time_str.replace(':', '')}.nc"
    
    request = {
        "product_type": "reanalysis",
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
            "geopotential"
        ],
        "pressure_level": ["1000", "500"],
        "year": [year],
        "month": [month],
        "day": [day],
        "time": [time_str],
        "format": "netcdf",
    }

    print(f"\n[{i}/{len(cases)}] Requesting data for {year}-{month}-{day} {time_str}...")

    try:
        client.retrieve("reanalysis-era5-pressure-levels", request).download(filename)
        print(f"✓ Saved as {filename}")
    except Exception as e:
        print(f"✗ Failed for {year}-{month}-{day} {time_str}: {e}")

    # Optional: sleep 2 seconds between requests to avoid hitting API limits
    time.sleep(2)
