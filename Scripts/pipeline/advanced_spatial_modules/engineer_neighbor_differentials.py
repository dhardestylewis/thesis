import pandas as pd
import geopandas as gpd
import numpy as np
import time
import os
from config.paths import DATA_DIR, PANEL_DIR, PROTEST_PETITIONS_DIR, GIS_DIR, ZONING_CASES_DIR


def build_neighbor_differentials(petitions, tcad, cases_gdf, props=None, out_dir=PROTEST_PETITIONS_DIR):
    signed_cases = petitions['case_number'].unique()
    print(f"2. Computing neighbor differentials for {len(signed_cases)} protested cases...")
    
    # We will buffer cases by 200ft up front
    print("Buffering cases...")
    cases_to_process = cases_gdf[cases_gdf.index.isin(signed_cases)].copy()
    cases_to_process['geometry'] = cases_to_process.geometry.buffer(200)
    
    print("Spatial joining buffered cases to TCAD parcels...")
    t0 = time.time()
    # Find all TCAD parcels that intersect the 200ft buffers
    joined = gpd.sjoin(tcad, cases_to_process.reset_index(), how='inner', predicate='intersects')
    print(f"SJOIN completed in {time.time()-t0:.1f}s. Found {len(joined)} parcel-buffer intersections.")
    
    results = []
    
    for idx, case in enumerate(cases_to_process.index):
        # All parcels within 200ft
        neighbors = joined[joined['case_number'] == case].index.astype(str).unique()
        if len(neighbors) == 0:
            continue
            
        # Fix float string bug (10003.0 -> '10003') while supporting dashes
        raw_signers = petitions[petitions['case_number'] == case]['tcad_id'].dropna().astype(str)
        signers = set(raw_signers.str.replace(r'\.0$', '', regex=True).str.replace('-', '', regex=False).unique())
        
        # Partition into cohorts
        protesting_ids = [n for n in neighbors if n in signers]
        silent_ids = [n for n in neighbors if n not in signers]
        
        # Look up properties
        protesting_props = props.reindex(protesting_ids).dropna(subset=['lui_general_land_use'])
        silent_props = props.reindex(silent_ids).dropna(subset=['lui_general_land_use'])
        
        # Aggregations
        res = {'case_number': case}
        
        # Single Family
        res['protesting_pct_single_family'] = (protesting_props['lui_general_land_use'] == 'Single Family').mean() if len(protesting_props) > 0 else 0
        res['silent_pct_single_family'] = (silent_props['lui_general_land_use'] == 'Single Family').mean() if len(silent_props) > 0 else 0
        
        # Commercial
        res['protesting_pct_commercial'] = (protesting_props['lui_general_land_use'] == 'Commercial').mean() if len(protesting_props) > 0 else 0
        res['silent_pct_commercial'] = (silent_props['lui_general_land_use'] == 'Commercial').mean() if len(silent_props) > 0 else 0
        
        # Multifamily
        res['protesting_pct_multifamily'] = (protesting_props['lui_general_land_use'] == 'Multifamily').mean() if len(protesting_props) > 0 else 0
        res['silent_pct_multifamily'] = (silent_props['lui_general_land_use'] == 'Multifamily').mean() if len(silent_props) > 0 else 0
        
        # Parcel Size
        res['protesting_mean_parcel_sqft'] = protesting_props['lui_shape_area'].mean() if len(protesting_props) > 0 else 0
        res['silent_mean_parcel_sqft'] = silent_props['lui_shape_area'].mean() if len(silent_props) > 0 else 0
        
        results.append(res)
        
        if idx % 50 == 0:
            print(f"   Processed {idx}/{len(cases_to_process)} cases...")
            
    res_df = pd.DataFrame(results).fillna(0)
    print(f"Completed in {time.time() - t0:.1f}s")
    
    out_path = r'Data\Protest_Petitions\neighbor_differentials.csv'
    res_df.to_csv(out_path, index=False)
    print(f"Saved neighbor differentials to {out_path}")
    
    print("3. Merging into Advanced Petition Panel...")
    adv_path = (PROTEST_PETITIONS_DIR / "advanced_geometric_petition_intensity.csv")
    adv = pd.read_csv(adv_path)
    
    # Drop existing differential columns if they exist (for reruns)
    cols_to_drop = [c for c in adv.columns if c in res_df.columns and c != 'case_number']
    adv = adv.drop(columns=cols_to_drop)
    
    merged = pd.merge(adv, res_df, on='case_number', how='left').fillna(0)
    merged.to_csv(adv_path, index=False)
    print("Merged and saved advanced geometric petition intensity.")

if __name__ == "__main__":
    build_neighbor_differentials()
