# -*- coding: utf-8 -*-
"""
Created on Sun May 18 12:06:00 2025

@author: jakew
"""

import os
import cdsapi
import time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metpy.calc as mpcalc
from metpy.units import units
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl

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
    filename = f"era5_temp_z_850hPa_{year}{month}{day}_{time_str.replace(':', '')}.nc"
    
    request = {
        "product_type": "reanalysis",
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
            "geopotential",
            "temperature"
        ],
        "pressure_level": ["850"],
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

# List your files
file_list = [
    "era5_temp_z_850hPa_20040101_0000.nc",
    "era5_temp_z_850hPa_20040105_1200.nc",
    "era5_temp_z_850hPa_20040110_0600.nc"
]

datasets = []
for f in file_list:
    ds = xr.open_dataset(f)

    # Drop time by squeezing or selecting first index if needed
    # We assume time is a singleton in each file
    ds_clean = ds.squeeze("valid_time", drop=True)
    
    # Drop the time coordinate entirely
    ds_clean = ds_clean.drop_vars("valid_time", errors="ignore")
    
    datasets.append(ds_clean)

# Concatenate along a fake dimension for compositing
composite = xr.concat(datasets, dim="composite_index").mean(dim="composite_index")
    
composite.to_netcdf("era5_composite_850_temp.nc",mode = 'w',engine="netcdf4",format="NETCDF4")

z850 = composite['z'].sel(pressure_level=850, method='nearest')
z850gh = z850 / 9.8 

z_850 = z850gh.sel(latitude=slice(65, 25), longitude=slice(250, 310))

level = 850
t850 = composite['t'].sel(pressure_level=level) - 273.15  # Convert to Celsius
u850 = composite['u'].sel(pressure_level=level)
v850 = composite['v'].sel(pressure_level=level)

v_850 = v850.sel(latitude=slice(65, 25), longitude=slice(250, 310)) * units("m/s") 
u_850 = u850.sel(latitude=slice(65, 25), longitude=slice(250, 310)) * units("m/s")
t_850 = u850.sel(latitude=slice(65, 25), longitude=slice(250, 310)) * units("°C")

v_850 = v_850[:].squeeze()
u_850 = u_850[:].squeeze()
t_850 = t_850[:].squeeze()

# Add units as attributes
u_850.attrs['units'] = 'm/s'
v_850.attrs['units'] = 'm/s'
t_850.attrs['units'] = '°C'

lat = composite['latitude']
lon = composite['longitude']
lat_na = u_850.latitude
lon_na = u_850.longitude
lon2d, lat2d = np.meshgrid(lon_na, lat_na)

# Subsample for wind barbs
skip = 10

# Create figure
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-110, -60, 25, 65], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES, linewidth=0.5)

# Contour shaded temperature
cmap = plt.cm.coolwarm
temp_plot = ax.contourf(lon2d, lat2d, t_850, levels=np.arange(-30, 35, 2), cmap=cmap, extend='both')

# Add color bar
cbar = plt.colorbar(temp_plot, orientation='horizontal', pad=0.05, aspect=40, shrink=0.8)
cbar.set_label("Temperature (°C)")

# Contour geopotential height (every 30 m)
height_contours = ax.contour(lon2d, lat2d, z_850, levels=np.arange(np.nanmin(z_850), np.nanmax(z_850), 30), colors='black', linewidths=1)
ax.clabel(height_contours, inline=True, fontsize=8, fmt='%d')

# Plot wind barbs
ax.barbs(lon2d[::skip, ::skip], lat2d[::skip, ::skip], 
         u_850.values[::skip, ::skip], v_850.values[::skip, ::skip],
         transform=ccrs.PlateCarree(), length=5, linewidth = 0.5)

# Title
plt.title("850 mb Temperature (shaded), Wind Barbs, and Geopotential Height (contours)")

############################## 850 mb temperature advection map ################################################

dx, dy = mpcalc.lat_lon_grid_deltas(lon2d, lat2d)

# Stack as a list of 2D arrays with shape (2, ny, nx)
wind = np.array([u_850.values, v_850.values]) * units('m/s')

# Calculate temperature advection using metpy function
adv = mpcalc.advection(t_850, u=u_850, v=v_850, dx=dx, dy=dy, longitude=lon_na, latitude=lat_na) 
adv_plot = adv * 1e4

# Subsample for wind barbs
skip = 10

#  visually de-emphasize weak advection from -2 to 2 10^-4 k/s

clevs_adv = np.concatenate([
    np.arange(-20, -2, 2), 
    np.array([-2, 2]), 
    np.arange(4, 22, 2)
])

base_cmap = plt.get_cmap('RdBu_r', len(clevs_adv) - 1)
# Convert to list and insert white in the center
colors = base_cmap(np.linspace(0, 1, len(clevs_adv) - 1))
middle_idx = np.where((clevs_adv[:-1] < 2) & (clevs_adv[1:] > -2))[0]
for idx in middle_idx:
    colors[idx] = (1.0, 1.0, 1.0, 1.0)  # RGBA for white

# Create new colormap and norm
custom_cmap = ListedColormap(colors)
norm = BoundaryNorm(clevs_adv, ncolors=custom_cmap.N)

# Create figure
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-110, -60, 25, 65], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES, linewidth=0.5)

# Contour shaded temperature
# cmap = plt.cm.coolwarm
cmap = plt.get_cmap("RdBu_r")  # Red for warm advection, blue for cold
temp_plot = ax.contourf(lon2d, lat2d, adv_plot, levels=np.arange(-20, 21, 2), cmap= custom_cmap, norm = norm, extend='both', transform=ccrs.PlateCarree())

# Add color bar
cbar = plt.colorbar(temp_plot, orientation='horizontal', pad=0.05, aspect=40, shrink=0.8)
cbar.set_label(r'Temperature Advection ($10^{-4}$ K/s)')

# Contour 850 mb temperature in dashed grey
temp_levels = np.arange(-40, 41, 2)  # adjust range as needed

temp_contours = ax.contour(
    lon_na, lat_na, t_850, levels=temp_levels,
    colors='grey', linestyles='dashed',
    linewidths=1, transform=ccrs.PlateCarree()
)

# Contour geopotential height (every 30 m)
height_contours = ax.contour(lon2d, lat2d, z_850, levels=np.arange(np.nanmin(z_850), np.nanmax(z_850), 30), colors='black', linewidths=1)
ax.clabel(height_contours, inline=True, fontsize=8, fmt='%d')

# Plot wind barbs
ax.barbs(lon2d[::skip, ::skip], lat2d[::skip, ::skip], 
         u_850.values[::skip, ::skip], v_850.values[::skip, ::skip],
         transform=ccrs.PlateCarree(), length=5, linewidth = 0.5)

# Title
plt.title("850 mb Temperature Advection (shaded), Wind Barbs, and Geopotential Height (contours)")





























