import pandas as pd
import geopandas as gpd
import numpy as np
import time
import json
import os

def build_spatial_vectors(petitions, tcad, cases_gdf, props=None, out_dir=r"Data/Protest_Petitions"):
    signed_cases = petitions['case_number'].unique()
    
    print(f"2. Computing spatial distance vectors for {len(signed_cases)} protested cases...")
    results = []
    
    t0 = time.time()
    for idx, case in enumerate(signed_cases):
        if case not in cases_gdf.index:
            continue
            
        case_geom = cases_gdf.loc[case].geometry
        if isinstance(case_geom, pd.Series):
            case_geom = case_geom.iloc[0]
            
        # Get signers for this case
        raw_signers = petitions[petitions['case_number'] == case]['tcad_id'].dropna().astype(str)
        signers = raw_signers.str.replace(r'\.0$', '', regex=True).str.replace('-', '', regex=False).unique()
        
        valid_signers = [s for s in signers if s in tcad.index]
        if not valid_signers:
            continue
            
        # Get signer polygons
        signer_geoms = tcad.loc[valid_signers]
        if isinstance(signer_geoms, pd.Series):
            signer_geoms = gpd.GeoDataFrame(geometry=signer_geoms)
            
        # Compute exact distance from case BOUNDARY to signer POLYS
        # GeoPandas distance computes shortest distance between the two geometries
        distances = signer_geoms.distance(case_geom).values
        
        # Build vectors
        dist_vector = np.round(distances, 2).tolist()
        
        res = {
            'case_number': case,
            'signer_distance_vector': json.dumps(dist_vector),
            'min_signer_dist': np.min(distances),
            'max_signer_dist': np.max(distances),
            'median_signer_dist': np.median(distances),
            'signers_within_200ft': np.sum(distances <= 200),
            'signers_outside_200ft': np.sum(distances > 200),
            'unofficial_protest_intensity': len(distances) # total signers regardless of distance
        }
        results.append(res)
        
        if idx % 50 == 0:
            print(f"   Processed {idx}/{len(signed_cases)} cases...")
            
    res_df = pd.DataFrame(results)
    print(f"   Completed in {time.time() - t0:.1f}s")
    
    print("3. Merging with exact geometric percentages...")
    geo = pd.read_csv(r'Data/Protest_Petitions/petition_summary_spatial_true.csv')
    
    merged = pd.merge(geo, res_df, on='case_number', how='left')
    
    out_path = r'Data/Protest_Petitions/advanced_geometric_petition_intensity.csv'
    merged.to_csv(out_path, index=False)
    print(f"Saved advanced spatial vectors to {out_path}")

if __name__ == "__main__":
    build_spatial_vectors()
