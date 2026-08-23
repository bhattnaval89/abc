"""
config.py
---------
Put your NASA FIRMS MAP_KEY here. This file is imported by
src/real_data_loader.py when you run it directly for testing.

Get a free key at: https://firms.modaps.eosdis.nasa.gov/api/map_key/
"""

FIRMS_MAP_KEY = "d72bbda43f66c027d2730fc1e61e0612"

# Bounding box for the region you want to monitor: (west, south, east, north)
# in decimal degrees. Default below roughly covers the Jamshedpur industrial
# belt, India - change it to your own region of interest.
BBOX = (86.05, 22.65, 86.35, 22.95)
