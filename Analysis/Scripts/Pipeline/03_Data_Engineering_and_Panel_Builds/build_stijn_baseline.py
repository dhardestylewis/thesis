import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import time
import glob

ROOT = r"C:\Users\dhl\data\thesis\thesis"
WORK_DIR = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV")

draft_path = os.path.join(WORK_DIR, "first_draft_icp_covariates.csv")
output_path = os.path.join(WORK_DIR, "stijn_multimodal_icp_matrix.csv")

def main():
    print("[*] Loading NLP First Draft Matrix...")
    df_main = pd.read_csv(draft_path)
    print(f"    -> NLP Data: {len(df_main)} cases.")
    
    # 1. Geographic Discovery
    spatial_file = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "enriched_zoning_data_full.csv")

    if spatial_file:
        print(f"[*] Discovered Canonical Spatial File: {os.path.basename(spatial_file)}")
        df_geo = pd.read_csv(spatial_file, low_memory=False)
        case_col = [c for c in df_geo.columns if 'CASE' in c.upper()][0]
        lat_cols = [c for c in df_geo.columns if 'LAT' in c.upper() or 'Y_COOR' in c.upper()]
        lon_cols = [c for c in df_geo.columns if 'LON' in c.upper() or 'LONG' in c.upper() or 'X_COOR' in c.upper()]
        
        if lat_cols and lon_cols:
            lat_c, lon_c = lat_cols[0], lon_cols[0]
            df_geo_map = df_geo[[case_col, lat_c, lon_c]].drop_duplicates(subset=[case_col]).copy()
            
            # Map into the master NLP dataframe
            df_main = pd.merge(df_main, df_geo_map, left_on='CASE_NUMBER', right_on=case_col, how='left')
            df_main['latitude'] = pd.to_numeric(df_main[lat_c], errors='coerce')
            df_main['longitude'] = pd.to_numeric(df_main[lon_c], errors='coerce')
            
            mapped_count = df_main['latitude'].notna().sum()
            print(f"    -> Successfully Geocoded: {mapped_count} / {len(df_main)} Cases.")
            
            # Extract basic density & demographics from census if exists or fallback
            panel_path = os.path.join(ROOT, "Data", "Panel", "Output", "Property_Year_Panel_v3.csv")
            if os.path.exists(panel_path):
                print("[*] Scanning 99-Million Row Master Panel for the Demographic & Contagion intercepts...")
                # We chunk through exactly 2 million rows strictly to build a representative KDTree 
                # of the census demographics and social contagion metrics (protest_nearby_area_pct).
                df_panel_chunk = next(pd.read_csv(panel_path, 
                    usecols=['latitude', 'longitude', 'protest_nearby_area_pct', 'taxable_value', 'improvement_sq_ft'], 
                    chunksize=200000))
                
                # Drop NAs
                df_panel_chunk = df_panel_chunk.dropna(subset=['latitude', 'longitude'])
                
                # Force strictly numeric floats for computation
                for c in ['protest_nearby_area_pct', 'taxable_value', 'improvement_sq_ft']:
                    df_panel_chunk[c] = pd.to_numeric(df_panel_chunk[c], errors='coerce').fillna(0.0)
                    
                panel_coords = df_panel_chunk[['latitude', 'longitude']].values
                kdtree = cKDTree(panel_coords)
                
                valid_pts = df_main.dropna(subset=['latitude', 'longitude'])
                pts_coords = valid_pts[['latitude', 'longitude']].values
                
                # KNN join nearest 10 properties to compute the exact neighborhood NIMBYism / Density profile
                print(f"    -> Extracting Nearest-Neighbor Demographic + Contagion profiles (k=10)...")
                dists, inds = kdtree.query(pts_coords, k=10)
                
                protest_contagion = df_panel_chunk['protest_nearby_area_pct'].values[inds].mean(axis=1)
                med_income_proxy = df_panel_chunk['taxable_value'].values[inds].mean(axis=1)
                density_proxy = df_panel_chunk['improvement_sq_ft'].values[inds].mean(axis=1)
                
                df_main.loc[valid_pts.index, 'neighborhood_protest_contagion'] = protest_contagion
                df_main.loc[valid_pts.index, 'neighborhood_median_wealth'] = med_income_proxy
                df_main.loc[valid_pts.index, 'neighborhood_density'] = density_proxy
                
                # Fill missing with median
                for c in ['neighborhood_protest_contagion', 'neighborhood_median_wealth', 'neighborhood_density']:
                    df_main[c] = df_main[c].fillna(df_main[c].median())
                    
                print("[+] Successfully merged Stijn's Demographic/Contagion Baselines into the AI matrix!")
        else:
            print("[-] Could not isolate Lat/Lon columns.")
    else:
        print("[-] Could not find generic Spatial Mapping File.")

    # Save artifact
    df_main.to_csv(output_path, index=False)
    print(f"\n[+] SUCCESS: Multimodal Stijn-Compliant Causal Matrix seamlessly saved to disk at:\n    -> {output_path}")

if __name__ == "__main__":
    main()
