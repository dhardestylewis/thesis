import geopandas as gpd
import pandas as pd
import os

RAW_GEOJSON = r"c:\Users\dhl\data\Thesis\thesis\Data\CoA_Open_Data\Zoning_Cases_Raw_Download.geojson"
OUT_CSV = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"

def extract_base_universe():
    print(f"Loading raw City of Austin GeoJSON: {RAW_GEOJSON}")
    gdf = gpd.read_file(RAW_GEOJSON)
    
    # Project to Texas Central EPSG 2277 to calculate square footage
    print("Projecting to EPSG:2277 to calculate accurate shape_area (sqft)...")
    gdf = gdf.to_crs(epsg=2277)
    gdf['shape_area'] = gdf.geometry.area
    
    print(f"Extracted {len(gdf)} raw cases. Stripping spatial geometries to create the base CSV canvas...")
    df = pd.DataFrame(gdf.drop(columns=['geometry']))
    
    # Optional: Basic cleaning so downstream scripts don't fail
    if 'case_number' in df.columns:
        df['case_number'] = df['case_number'].str.strip()
    
    # Save the blank canvas
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Successfully initialized the master pipeline target: {OUT_CSV}")

if __name__ == "__main__":
    extract_base_universe()
