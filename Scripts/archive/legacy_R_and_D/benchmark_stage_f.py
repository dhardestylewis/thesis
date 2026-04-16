import pandas as pd
import geopandas as gpd
from shapely import wkt
import time
import os

LUI_FILE = r"C:\Users\dhl\data\thesis\thesis\Data\CoA_Open_Data\Land_Use\LUI_2024_7vsm-dvxg.csv"
TRACT_SHP = r"C:\Users\dhl\data\thesis\thesis\Data\GIS\Census\tracts\tl_2024_48_tract.shp"

def run_benchmark():
    start_total = time.time()
    
    print("[1] Loading 1,000 Sample Parcels from LUI 2024...")
    start_load = time.time()
    df_lui = pd.read_csv(LUI_FILE, usecols=['the_geom', 'PARCEL_ID_10'], nrows=50000).sample(1000, random_state=42)
    print(f"    Loaded in {time.time() - start_load:.2f}s")
    
    print("[2] Parsing WKT Geometries into Shapely and Centroids...")
    start_geom = time.time()
    df_lui['geometry'] = df_lui['the_geom'].apply(wkt.loads)
    gdf_lui = gpd.GeoDataFrame(df_lui, geometry='geometry', crs="EPSG:4326")
    gdf_lui['geometry'] = gdf_lui.geometry.centroid
    print(f"    Geometry Parsed in {time.time() - start_geom:.2f}s")
    
    print("[3] Loading Census Tracts and Aligning CRS...")
    start_census = time.time()
    gdf_tracts = gpd.read_file(TRACT_SHP)
    expected_crs = gdf_tracts.crs
    gdf_lui = gdf_lui.to_crs(expected_crs)
    print(f"    Tracts Loaded in {time.time() - start_census:.2f}s")
    
    print("[4] Spatial Join (Point in Polygon)...")
    start_join = time.time()
    gdf_joined = gpd.sjoin(gdf_lui, gdf_tracts, how='left', predicate='within')
    print(f"    Joined in {time.time() - start_join:.2f}s")
    
    print("[5] Executing Spatial Contagion (Simulated 1,000x7,000 Distance Matrix)...")
    import numpy as np
    start_contagion = time.time()
    # Dummy Lat/Lons internally
    lats1 = np.random.uniform(30.1, 30.5, 1000)
    lons1 = np.random.uniform(-97.9, -97.5, 1000)
    lats2 = np.random.uniform(30.1, 30.5, 7000)
    lons2 = np.random.uniform(-97.9, -97.5, 7000)
    
    MILE_RADS = 1.0 / 3958.8
    results = np.zeros(1000)
    for i in range(1000):
        dlat = np.radians(lats2) - np.radians(lats1[i])
        dlon = np.radians(lons2) - np.radians(lons1[i])
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lats1[i])) * np.cos(np.radians(lats2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
        results[i] = (c <= MILE_RADS).sum()
    print(f"    Contagion computed in {time.time() - start_contagion:.2f}s")
    
    total_time = time.time() - start_total
    print(f"\n[SUMMARY] Total Time for 1,000 Parcels: {total_time:.2f} seconds")
    
    estimated_450k = (total_time / 1000) * 450000 / 60
    print(f"\n[PROJECTION] Total ETA for 450,000 Parcels: {estimated_450k:.2f} minutes")
    print(f"             (Just for Stage F execution on fresh DB)")

if __name__ == '__main__':
    run_benchmark()
