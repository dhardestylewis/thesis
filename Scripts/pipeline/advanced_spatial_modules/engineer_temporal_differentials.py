import pandas as pd
import geopandas as gpd
import numpy as np
import time
import os
from config.paths import DATA_DIR, PANEL_DIR, PROTEST_PETITIONS_DIR, GIS_DIR, ZONING_CASES_DIR


def determine_friction(case_zoning, neighbor_lu):
    if not isinstance(case_zoning, str) or not isinstance(neighbor_lu, str):
        return 0.0
    case_zoning = case_zoning.upper()
    
    # Commercial / Industrial case
    case_is_commercial = any(x in case_zoning for x in ['CS', 'GR', 'W', 'I', 'CH', 'LI', 'MI'])
    # Multifamily case
    case_is_mf = any(x in case_zoning for x in ['MF', 'V', 'PDA'])
    
    # Friction triggers
    if (case_is_commercial or case_is_mf) and ('Single Family' in neighbor_lu):
        return 1.0 # High friction (Density/Commercial vs SF)
    
    if case_is_commercial and ('Multifamily' in neighbor_lu):
        return 1.0 # High friction (Commercial vs MF)
        
    return 0.0

def build_temporal_differentials(petitions, tcad, cases_gdf, props=None, out_dir=PROTEST_PETITIONS_DIR):
    print("Loading 2021 Property Universe...")
    props_2021 = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\Panel\parcel\property_universe.csv', dtype={'standardized_tcad_id': str})
    props_2021['standardized_tcad_id'] = props_2021['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    props_2021 = props_2021.set_index('standardized_tcad_id')
    
    print("Loading case metadata...")
    case_meta = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\enriched_zoning_data_causal.csv', low_memory=False)
    case_meta = case_meta.set_index('case_number')
    
    print("Loading 2016 LDB...")
    props_2016 = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\CoA_Open_Data\LDB_2016_4nsn-uea6.csv', low_memory=False, dtype={'PID_10': str})
    # LDB 2016 might have PID_10 or PROP_ID as TCAD ID
    # Usually PROP_ID is the 6-digit one, PID_10 is the 10-digit one
    props_2016['standardized_tcad_id'] = props_2016['PID_10'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    # Deduplicate
    props_2016 = props_2016.drop_duplicates(subset=['standardized_tcad_id']).set_index('standardized_tcad_id')
    
    signed_cases = petitions['case_number'].unique()
    print(f"2. Computing temporal and zoning deltas for {len(signed_cases)} protested cases...")
    
    cases_to_process = cases_gdf[cases_gdf.index.isin(signed_cases)].copy()
    cases_to_process['geometry'] = cases_to_process.geometry.buffer(200)
    
    print("Spatial joining buffered cases to TCAD parcels...")
    t0 = time.time()
    joined = gpd.sjoin(tcad, cases_to_process.reset_index(), how='inner', predicate='intersects')
    print(f"SJOIN completed in {time.time()-t0:.1f}s.")
    
    results = []
    
    for idx, case in enumerate(cases_to_process.index):
        neighbors = joined[joined['case_number'] == case].index.astype(str).unique()
        if len(neighbors) == 0:
            continue
            
        raw_signers = petitions[petitions['case_number'] == case]['tcad_id'].dropna().astype(str)
        signers = set(raw_signers.str.replace(r'\.0$', '', regex=True).str.replace('-', '', regex=False).unique())
        
        # Partition
        clean_neighbors = [str(n).replace('-', '') for n in neighbors]
        protesting_ids = [n for n in clean_neighbors if n in signers]
        silent_ids = [n for n in clean_neighbors if n not in signers]
        
        # Determine case date
        case_date = '2021-01-01' # fallback
        case_zoning = ''
        if case in case_meta.index:
            row = case_meta.loc[case]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            case_date = str(row.get('application_start_date', '2021-01-01'))
            case_zoning = str(row.get('proposed_zoning', ''))
            
        is_pre_2021 = case_date < '2021-01-01'
        
        # Pull properties temporally
        if is_pre_2021:
            p_props = props_2016.reindex(protesting_ids)
            s_props = props_2016.reindex(silent_ids)
            lu_col = 'GEN_LU_DESC'
            sqft_col = 'SHAPE_Area'
        else:
            p_props = props_2021.reindex(protesting_ids)
            s_props = props_2021.reindex(silent_ids)
            lu_col = 'lui_general_land_use'
            sqft_col = 'lui_shape_area'
            
        p_props = p_props.dropna(subset=[lu_col])
        s_props = s_props.dropna(subset=[lu_col])
        
        res = {'case_number': case}
        
        # SF
        res['temporal_protesting_pct_sf'] = (p_props[lu_col].astype(str).str.contains('Single Family', na=False)).mean() if len(p_props) > 0 else 0
        res['temporal_silent_pct_sf'] = (s_props[lu_col].astype(str).str.contains('Single Family', na=False)).mean() if len(s_props) > 0 else 0
        
        # Commercial
        res['temporal_protesting_pct_com'] = (p_props[lu_col].astype(str).str.contains('Commercial', na=False)).mean() if len(p_props) > 0 else 0
        res['temporal_silent_pct_com'] = (s_props[lu_col].astype(str).str.contains('Commercial', na=False)).mean() if len(s_props) > 0 else 0
        
        # Multifamily
        res['temporal_protesting_pct_mf'] = (p_props[lu_col].astype(str).str.contains('Multifamily', na=False)).mean() if len(p_props) > 0 else 0
        res['temporal_silent_pct_mf'] = (s_props[lu_col].astype(str).str.contains('Multifamily', na=False)).mean() if len(s_props) > 0 else 0
        
        # Zoning Friction Delta
        p_frictions = p_props[lu_col].apply(lambda lu: determine_friction(case_zoning, lu))
        s_frictions = s_props[lu_col].apply(lambda lu: determine_friction(case_zoning, lu))
        
        res['delta_protesting_friction'] = p_frictions.mean() if len(p_frictions) > 0 else 0
        res['delta_silent_friction'] = s_frictions.mean() if len(s_frictions) > 0 else 0
        
        results.append(res)
        
        if idx % 50 == 0:
            print(f"   Processed {idx}/{len(cases_to_process)} cases...")
            
    res_df = pd.DataFrame(results).fillna(0)
    print(f"Completed in {time.time() - t0:.1f}s")
    
    out_path = r'Data\Protest_Petitions\temporal_differentials.csv'
    res_df.to_csv(out_path, index=False)
    print(f"Saved temporal differentials to {out_path}")
    
    print("3. Merging into Advanced Petition Panel...")
    adv_path = (PROTEST_PETITIONS_DIR / "advanced_geometric_petition_intensity.csv")
    adv = pd.read_csv(adv_path)
    
    cols_to_drop = [c for c in adv.columns if c in res_df.columns and c != 'case_number']
    adv = adv.drop(columns=cols_to_drop)
    
    merged = pd.merge(adv, res_df, on='case_number', how='left').fillna(0)
    merged.to_csv(adv_path, index=False)
    print("Merged and saved advanced geometric petition intensity.")

if __name__ == "__main__":
    build_temporal_differentials()
