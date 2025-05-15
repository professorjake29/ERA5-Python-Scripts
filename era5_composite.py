# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:28:13 2025

@author: jakew
"""

import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.chdir('./Downloads')

# List your files
file_list = [
    "era5_vort_1000_500hPa_20040101_0000.nc",
    "era5_vort_1000_500hPa_20040105_1200.nc",
    "era5_vort_1000_500hPa_20040110_0600.nc"
]

# Load data
ds = xr.open_dataset("era5_vort_1000_500hPa_20040101_0000.nc").metpy.parse_cf()
ds = ds.squeeze("valid_time", drop=True)
dt = xr.open_dataset("era5_vort_1000_500hPa_20040105_1200.nc").metpy.parse_cf()
dt = dt.squeeze("valid_time", drop=True)
dr = xr.open_dataset("era5_vort_1000_500hPa_20040110_0600.nc").metpy.parse_cf()
dr = dr.squeeze("valid_time", drop=True)



# Extract variables
zs = ds['z'].sel(pressure_level=500, method='nearest')
zt = dt['z'].sel(pressure_level=500, method='nearest')
zr = dr['z'].sel(pressure_level=500, method='nearest')

zs.sel(latitude = 50, longitude = 50).item() # Extract value only

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
    
composite.to_netcdf("era5_composite.nc",mode = 'w',engine="netcdf4",format="NETCDF4")
dc = xr.open_dataset("era5_composite.nc").metpy.parse_cf()


z500 = composite['z'].sel(pressure_level=500, method='nearest')
z500gh = z500 / 9.8

z1000 = composite['z'].sel(pressure_level=1000, method='nearest')
z1000gh = z1000 / 9.8

z_500 = z500gh.sel(latitude=slice(65, 25), longitude=slice(250, 310))
z_1000 = z1000gh.sel(latitude=slice(65, 25), longitude=slice(250, 310))


# Set up the map
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot height contours
contours = ax.contour(z_500.longitude, z_500.latitude, z_500,
                      levels=range(5000, 6000, 60), colors='black', linewidths=1)

ax.clabel(contours, inline=1, fontsize=10)

# Add features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.set_extent([-110, -60, 25, 65], crs=ccrs.PlateCarree())

plt.title("500 hPa Geopotential Height (m) - CONUS")
plt.savefig("500mb_heights_conus.png", dpi=600, bbox_inches="tight")
plt.show()

# Overlapy map

# Create figure and axes
fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([-110, -60, 25, 65], crs=ccrs.PlateCarree())

# Manually set the ticks you want to display
xticks = [-110, -100, -90, -80, -70, -60]
yticks = [25, 35, 45, 55, 65]

# Set manual ticks (no gridlines)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())

# Add formatted labels
ax.set_xticklabels([f"{abs(x)}°W" for x in xticks], fontsize=10)
ax.set_yticklabels([f"{y}°N" for y in yticks], fontsize=10)

# Remove gridlines if they exist
ax.grid(False)

# Plot shaded 1000 mb height
cf = ax.contourf(z_1000.longitude, z_1000.latitude, z_1000,
                 levels=30, cmap="nipy_spectral", extend = "both")

# Add colorbar
#cbar = plt.colorbar(cf, ax=ax, orientation="vertical", shrink = 0.92, pad=0.05, aspect=50) # aspect controls thickness (larger = thinner for horizontal bars).
cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", shrink = 0.7, pad=0.04, aspect = 60) # pad moves the color bar closer to the plot (smaller = closer)


#cbar.set_label("1000 mb Geopotential Height (m)")

# Plot 500 mb height contours
cs = ax.contour(z_500.longitude, z_500.latitude, z_500,
                levels=range(5000, 6000, 60), colors="black", linewidths=1.5)
# ax.clabel(cs, fmt="%d", fontsize=10)

clabels = plt.clabel(cs, fmt='%d', colors='black', inline_spacing=5, use_clabeltext=True)

# Contour labels with white boxes black text
for t in clabels:
    t.set_bbox({'facecolor': 'white', 'pad': 2})
    t.set_fontweight('medium')



# Add map features
ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
ax.add_feature(cfeature.BORDERS.with_scale("50m"))
ax.add_feature(cfeature.STATES.with_scale("50m"))

# Title and save
plt.title("ERA5 500-hPa NAM Geopotential Heights (m), 1000-hPa NAM Geopotential Heights (m)", fontsize=9, loc = 'left')
plt.savefig("500mb_over_1000mb_heights.png", dpi=400, bbox_inches="tight")
plt.show()