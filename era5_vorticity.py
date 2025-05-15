# -*- coding: utf-8 -*-
"""
Created on Wed May 14 01:33:17 2025

@author: jakew
"""

import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import metpy.calc as mpcalc
import numpy as np
from metpy.units import units
from matplotlib.colors import ListedColormap, BoundaryNorm



os.chdir('./Downloads')


dc = xr.open_dataset("era5_composite.nc").metpy.parse_cf()
lon = dc['longitude']
lat = dc['latitude']


z500 = dc['z'].sel(pressure_level=500, method='nearest')
z500gh = z500 / 9.8

u500 = dc['u'].sel(pressure_level=500, method='nearest').squeeze() * units('m/s')
v500 = dc['v'].sel(pressure_level=500, method='nearest').squeeze() * units('m/s')
u500 = u500.metpy.quantify()
v500 = v500.metpy.quantify()

# Add units as attributes
u500.attrs['units'] = 'm/s'
v500.attrs['units'] = 'm/s'

# Get data over North America

u500_na = u500.sel(latitude=slice(55, 25), longitude=slice(230, 300))
v500_na = v500.sel(latitude=slice(55, 25), longitude=slice(230, 300))
z500_na = z500gh.sel(latitude=slice(55, 25), longitude=slice(230, 300))
lat_na = u500_na.latitude
lon_na = u500_na.longitude

print("Any NaNs in u?", np.isnan(u500.values).any())
print("Any NaNs in v?", np.isnan(v500.values).any())
print("Any NaNs in longitude?", np.isnan(dc['longitude'].values).any())
print("Any NaNs in latitude?", np.isnan(dc['latitude'].values).any())

dx, dy = mpcalc.lat_lon_grid_deltas(lon_na.values, lat_na.values)
f = mpcalc.coriolis_parameter(np.deg2rad(lat_na))

print("dx valid?", np.isfinite(dx).all())
print("dy valid?", np.isfinite(dy).all())

vort = mpcalc.absolute_vorticity(u500_na.squeeze(), v500_na.squeeze(), dx=dx, dy=dy, latitude=lat_na, longitude = lon_na)
vort = vort * 1e5 * units('1/s')

## Set up vorticity plotting

# Define boundaries
boundaries = list(range(-40, 1, 10)) + list(range(5, 60, 5))  # [-40, -30, ..., 0, 5, ..., 55]

# Define colors
grayscale = ['black', 'dimgray', 'gray', 'lightgray', 'white']
colorscale = [
    'white', 'yellow', 'orange', 'red', 'pink',
    'purple', 'darkblue', 'blue', 'lightskyblue', 'deepskyblue', 'cyan', 'lightblue'
]

# Combine colors (remove duplicate white)
colors = grayscale + colorscale[1:]

# Create the colormap and norm
cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries, ncolors=len(colors))


# Set up map
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([-110, -60, 25, 55], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES, linewidth=0.5)

# Plot vorticity (shaded)
cf = ax.contourf(lon_na, lat_na, vort, levels=boundaries, cmap=cmap, norm = norm, transform=ccrs.PlateCarree(), extend="neither")

# Plot geopotential height (contoured)
cs = ax.contour(lon_na, lat_na, z500_na.squeeze(), levels=np.arange(4800, 6000, 60), colors='black', linewidths=1)
ax.clabel(cs, fmt="%d", fontsize=10, inline=True)

# Plot wind barbs (every 5th point to declutter)
skip = (slice(None, None, 5), slice(None, None, 5))
ax.barbs(
    lon_na.values[skip[1]],
    lat_na.values[skip[0]],
    u500_na.squeeze().values[skip],
    v500_na.squeeze().values[skip],
    length=5,
    color='black',
    linewidth=0.5
)

# Add colorbar for vorticity
# cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
cbar = plt.colorbar(
    cf,
    ax=ax,
    orientation='horizontal',
    ticks=boundaries,
    extend='neither',
    shrink=0.85,
    pad=0.04,
    aspect=40
)

cbar.set_label("500 mb Absolute Vorticity (10⁻⁵ s⁻¹)")

# Custom lat/lon labels
xticks = [-110, -100, -90, -80, -70, -60]
yticks = [25, 35, 45, 55]
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())
ax.set_xticklabels([f"{abs(x)}°W" for x in xticks], fontsize=10)
ax.set_yticklabels([f"{y}°N" for y in yticks], fontsize=10)

# Final formatting
plt.title("500 mb Vorticity (shaded), Geopotential Height (contoured), and Wind (barbs)", fontsize=14)
plt.tight_layout()
plt.show()
