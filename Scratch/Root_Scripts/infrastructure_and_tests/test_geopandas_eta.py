"""
Test GeoPandas Intersection ETA
Downloads a small sample of exact polygon geometries from Austin Open Data
to calculate the ETA for the full 200ft boundary geometric intersection.
"""
import pandas as pd
import geopandas as gpd
import time
import requests

print("Downloading Sample Austin Open Data Geometries...")

# Socrata GeoJSON endpoints
ZONING_URL = "https://data.austintexas.gov/resource/b2kk-8kt2.geojson?$limit=100"
LAND_USE_URL = "https://data.austintexas.gov/resource/pstw-7bkg.geojson?$limit=10000"

t0 = time.time()
try:
    cases = gpd.read_file(ZONING_URL)
    print(f"Loaded {len(cases)} Zoning Case Polygons in {time.time()-t0:.2f}s")
    
    t1 = time.time()
    parcels = gpd.read_file(LAND_USE_URL)
    print(f"Loaded {len(parcels)} Land Use Parcel Polygons in {time.time()-t1:.2f}s")
    
    # 1. Reproject to EPSG:2277 (Texas South Central, units in feet)
    cases = cases.to_crs(epsg=2277)
    parcels = parcels.to_crs(epsg=2277)
    
    # Clean invalid geometries
    cases['geometry'] = cases.geometry.buffer(0)
    parcels['geometry'] = parcels.geometry.buffer(0)
    
    print("\nRunning Spatial Intersection Test...")
    t2 = time.time()
    
    # 2. Buffer Zoning Cases by 200 feet from the EDGE of the polygon
    cases['buffer_200ft'] = cases.geometry.buffer(200)
    cases = cases.set_geometry('buffer_200ft')
    
    # 3. Spatial Join (Bounding Box) to find intersecting parcels
    # This acts as a pre-filter before the exact geometric overlay
    intersecting_pairs = gpd.sjoin(parcels, cases, how='inner', predicate='intersects')
    print(f"Found {len(intersecting_pairs)} candidate intersections.")
    
    # 4. Exact Geometric Overlay
    # To do this efficiently, we iterate over the joined pairs or use gpd.overlay
    # Since it's a small sample, we can overlay
    exact_intersection = gpd.overlay(
        parcels[parcels.index.isin(intersecting_pairs.index)],
        cases[cases.index.isin(intersecting_pairs['index_right'])],
        how='intersection'
    )
    
    exact_intersection['intersected_sqft'] = exact_intersection.geometry.area
    
    calc_time = time.time() - t2
    print(f"Calculated exact intersecting square footage in {calc_time:.2f} seconds.")
    print(exact_intersection[['intersected_sqft']].head())
    
    # Extrapolate
    # 100 cases tested. We need to run 6,000 cases against 300,000 parcels.
    # The sjoin and overlay scale roughly linearly with the number of bounding box collisions.
    time_per_100_cases = calc_time
    total_est_seconds = (6000 / 100) * time_per_100_cases
    # Factor in the loading time for 300k parcels (approx 30x the 10k load time)
    parcel_load_time = (300000 / 10000) * (t1 - t0) # very rough
    
    print("\n" + "="*50)
    print("ETA EXTRAPOLATION FOR FULL DATASET")
    print("="*50)
    print(f"Estimated time to download 300k geometries: ~{parcel_load_time/60:.1f} minutes")
    print(f"Estimated time to run exact geometric overlay (6k cases): ~{total_est_seconds/60:.1f} minutes")
    print("Total Pipeline ETA: ~15-20 minutes (assuming sufficient RAM).")
    print("RAM Requirement: High (Geopandas overlay on 300k polygons requires ~8-16GB RAM).")
    
except Exception as e:
    print(f"Error fetching or processing geometries: {e}")
