import geopandas as gpd
import pandas as pd
import os

RAW_GEOJSON = r"c:\Users\dhl\data\Thesis\thesis\Data\CoA_Open_Data\Zoning_Cases_Raw_Download.geojson"
POLYGONS_GEOJSON = r"c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\zoning_cases_master_polygons.geojson"
OUT_CSV = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"

def extract_base_universe():
    print(f"Loading raw City of Austin GeoJSON (Points): {RAW_GEOJSON}")
    gdf_pts = gpd.read_file(RAW_GEOJSON)

    print(f"Loading master polygons GeoJSON: {POLYGONS_GEOJSON}")
    if os.path.exists(POLYGONS_GEOJSON):
        gdf_poly = gpd.read_file(POLYGONS_GEOJSON)
        # Project polygons to Texas Central EPSG 2277 to calculate square footage
        print("Projecting polygons to EPSG:2277 to calculate accurate shape_area (sqft)...")
        gdf_poly = gdf_poly.to_crs(epsg=2277)
        gdf_poly['shape_area'] = gdf_poly.geometry.area
        
        # Merge shape_area into the base points dataset
        # Avoid geometry collision by joining purely as pandas DataFrames on case_number
        poly_areas = gdf_poly[['case_number', 'shape_area']].drop_duplicates(subset=['case_number'])
        gdf = gdf_pts.merge(poly_areas, on='case_number', how='left')
    else:
        print("WARNING: Master polygons not found. Proceeding with Point geometry (0.0 area).")
        gdf = gdf_pts.to_crs(epsg=2277)
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
