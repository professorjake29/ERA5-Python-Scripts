# -*- coding: utf-8 -*-
"""
Created on Sun May 18 10:12:20 2025

@author: jakew
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

os.chdir('./Downloads')
dc = xr.open_dataset("era5_composite.nc").metpy.parse_cf()
lons = dc['longitude'].values
lats = dc['latitude'].values


# Create meshgrid of lat/lon
lon2d, lat2d = np.meshgrid(lons, lats)

# Set up figure
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add basic features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.set_extent([-90, -70, 40, 50], crs=ccrs.PlateCarree())  # Adjust to your domain

# Plot grid points
ax.scatter(lon2d, lat2d, s=0.5, color='black', transform=ccrs.PlateCarree())

# Titles and show
plt.title("ERA5 Grid Points")
plt.show()