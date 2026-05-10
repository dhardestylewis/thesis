"""
scratch_geocode_spatial_fallback.py
===================================
For cases that failed or timed out in the Census Geocoder, use a KDTree to attach
the same Census Tract GEOID as the nearest physical case that *was* successfully geocoded.

Uses:
  Data/Panel/geo/case_geoid_lookup.csv (the 2000 successful matches)
  Data/interim/stage_c_features_raw.parquet (for lat/lon of missing cases)

Appends missing cases to case_geoid_lookup.csv with source = 'spatial_fallback'.
"""
import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree

OUT_PATH = 'Data/Panel/geo/case_geoid_lookup.csv'

print("[1/3] Loading geocoded cases...")
# Load known (geocoded)
gl = pd.read_csv(OUT_PATH, low_memory=False, dtype={'geoid_tract': str, 'geoid_bg': str})
gl = gl[gl['geoid_tract'].notna()].copy()
gl['geoid_tract'] = gl['geoid_tract'].astype(str).str.split('.').str[0]
if 'geoid_bg' in gl.columns:
    gl['geoid_bg'] = gl['geoid_bg'].astype(str).str.split('.').str[0]
known_ids = set(gl.case_id.astype(str))

# Read lat/lons from stage_c to get coordinates for the known and missing cases
sc = pd.read_parquet('Data/interim/stage_c_features_raw.parquet')
sc = sc[['case_id', 'latitude', 'longitude']].dropna().copy()
sc['case_id'] = sc['case_id'].astype(str)

# Attach lat/lon to known
known_coords = gl.merge(sc, on='case_id', how='inner')
print(f"  Valid geocoded cases with lat/lon: {len(known_coords)}")

print("[2/3] Building KDTree on successful geocoder matches...")
coords_rad = np.radians(known_coords[['latitude', 'longitude']].values)
tree = cKDTree(coords_rad)

print("[3/3] Performing spatial fallback for remaining cases...")
# We only care about target cases 2007-2025
sc_all = pd.read_parquet('Data/interim/stage_c_features_raw.parquet')
sc_all['year'] = pd.to_datetime(sc_all['as_of_date']).dt.year
cases = sc_all[sc_all['year'].notna() & sc_all['year'].between(2007, 2025)].copy()
has_ll = cases['latitude'].notna() & cases['longitude'].notna()
target = cases[has_ll][['case_id', 'latitude', 'longitude']].copy()
target['case_id'] = target['case_id'].astype(str)

missing = target[~target['case_id'].isin(known_ids)].copy()
print(f"  Remaining cases to spatial-fill: {len(missing)}")

fallback_results = []
EARTH_R_M = 6_371_000

for _, row in missing.iterrows():
    pt = np.radians([[row['latitude'], row['longitude']]])
    dist_rad, idx = tree.query(pt, k=1)
    dist_m = dist_rad[0] * EARTH_R_M
    
    # We'll generously allow up to 2000m since tracts are large, but usually it's much closer
    if dist_m <= 2000:
        nearest_row = known_coords.iloc[idx[0]]
        fallback_results.append({
            'case_id': row['case_id'],
            'geoid_tract': nearest_row['geoid_tract'],
            'geoid_bg': nearest_row.get('geoid_bg', nearest_row['geoid_tract'] + '0'),
            'source': f'spatial_fallback_{dist_m:.0f}m'
        })

print(f"  Successfully matched {len(fallback_results)} cases via spatial fallback!")

if fallback_results:
    fb_df = pd.DataFrame(fallback_results)
    fb_df.to_csv(OUT_PATH, mode='a', header=False, index=False)
    print(f"  Appended to {OUT_PATH}")

print("Done.")
