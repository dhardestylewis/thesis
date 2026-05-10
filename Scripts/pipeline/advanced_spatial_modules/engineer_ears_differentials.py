import pandas as pd
import geopandas as gpd
import numpy as np
import time
import os
from pathlib import Path

# Mapping State Property Tax Board codes to standardized classes
def get_standardized_lu(ears_code):
    if not isinstance(ears_code, str):
        return 'Unknown'
    code = ears_code.upper().strip()
    if code.startswith('A'):
        return 'Single Family'
    elif code.startswith('B'):
        return 'Multifamily'
    elif code.startswith('F') or code.startswith('C'):
        return 'Commercial'
    else:
        return 'Other'

def determine_friction(case_zoning, standardized_lu):
    if not isinstance(case_zoning, str) or not isinstance(standardized_lu, str):
        return 0.0
    case_zoning = case_zoning.upper()
    
    # Commercial / Industrial case
    case_is_commercial = any(x in case_zoning for x in ['CS', 'GR', 'W', 'I', 'CH', 'LI', 'MI'])
    # Multifamily case
    case_is_mf = any(x in case_zoning for x in ['MF', 'V', 'PDA'])
    
    # Friction triggers
    if (case_is_commercial or case_is_mf) and ('Single Family' in standardized_lu):
        return 1.0 # High friction (Density/Commercial vs SF)
    
    if case_is_commercial and ('Multifamily' in standardized_lu):
        return 1.0 # High friction (Commercial vs MF)
        
    return 0.0

def load_ears_year(year):
    # Fallback to nearest available year if out of bounds (EARS has 2018-2025)
    if year < 2018: year = 2018
    if year > 2025: year = 2025
    
    ears_path = rf'C:\Users\dhl\data\Thesis\thesis\Data\Raw\EARS\ears_{year}.csv'
    if not os.path.exists(ears_path):
        print(f"  [!] Missing EARS {year}. Falling back to 2021.")
        ears_path = r'C:\Users\dhl\data\Thesis\thesis\Data\Raw\EARS\ears_2021.csv'
        
    df = pd.read_csv(ears_path, dtype={'account_number': str}, low_memory=False)
    # EARS accounts should already be zero-padded by extraction script, but strictly enforce
    df['account_number'] = df['account_number'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    
    # Create standardized LU column
    df['standardized_lu'] = df['land_use_code'].apply(get_standardized_lu)
    df = df.drop_duplicates(subset=['account_number']).set_index('account_number')
    
    # Identify the sqft column (usually improvement_sq_ft)
    if 'improvement_sq_ft' in df.columns:
        df['sqft'] = pd.to_numeric(df['improvement_sq_ft'], errors='coerce').fillna(0)
    else:
        df['sqft'] = 0
        
    return df

def build_ears_differentials(petitions, tcad, cases_gdf, props=None, out_dir=r"Data/Protest_Petitions"):
    print("Pre-caching EARS longitudinal panels...")
    
    print("Loading case metadata...")
    case_meta = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\enriched_zoning_data_causal.csv', low_memory=False)
    case_meta = case_meta.set_index('case_number')
    
    ears_cache = {}
    
    signed_cases = petitions['case_number'].unique()
    print(f"2. Computing exact longitudinal neighbor differentials for {len(signed_cases)} protested cases...")
    
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
        
        # Determine exact year of case
        case_date = '2021-01-01'
        case_zoning = ''
        if case in case_meta.index:
            row = case_meta.loc[case]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            case_date = str(row.get('application_start_date', '2021-01-01'))
            case_zoning = str(row.get('proposed_zoning', ''))
            
        try:
            case_year = int(case_date[:4])
            ears_year = case_year - 1 # Lag by 1 year to prevent post-treatment leakage
        except:
            ears_year = 2020
            
        # Load the EXACT EARS year from cache
        if ears_year not in ears_cache:
            ears_cache[ears_year] = load_ears_year(ears_year)
            
        props = ears_cache[ears_year]
        
        p_props = props.reindex(protesting_ids)
        s_props = props.reindex(silent_ids)
        
        p_props = p_props.dropna(subset=['standardized_lu'])
        s_props = s_props.dropna(subset=['standardized_lu'])
        
        res = {'case_number': case}
        
        # --- SPATIAL BASE (Replaces neighbor_differentials) ---
        res['protesting_pct_single_family'] = (p_props['standardized_lu'] == 'Single Family').mean() if len(p_props) > 0 else 0
        res['silent_pct_single_family'] = (s_props['standardized_lu'] == 'Single Family').mean() if len(s_props) > 0 else 0
        
        res['protesting_pct_commercial'] = (p_props['standardized_lu'] == 'Commercial').mean() if len(p_props) > 0 else 0
        res['silent_pct_commercial'] = (s_props['standardized_lu'] == 'Commercial').mean() if len(s_props) > 0 else 0
        
        res['protesting_pct_multifamily'] = (p_props['standardized_lu'] == 'Multifamily').mean() if len(p_props) > 0 else 0
        res['silent_pct_multifamily'] = (s_props['standardized_lu'] == 'Multifamily').mean() if len(s_props) > 0 else 0
        
        res['protesting_mean_parcel_sqft'] = p_props['sqft'].mean() if len(p_props) > 0 else 0
        res['silent_mean_parcel_sqft'] = s_props['sqft'].mean() if len(s_props) > 0 else 0
        
        # --- TEMPORAL DIFFERENTIALS (Replaces temporal_differentials) ---
        # Since this is perfectly aligned, temporal features are identical to the base spatial features
        res['temporal_protesting_pct_sf'] = res['protesting_pct_single_family']
        res['temporal_silent_pct_sf'] = res['silent_pct_single_family']
        res['temporal_protesting_pct_com'] = res['protesting_pct_commercial']
        res['temporal_silent_pct_com'] = res['silent_pct_commercial']
        res['temporal_protesting_pct_mf'] = res['protesting_pct_multifamily']
        res['temporal_silent_pct_mf'] = res['silent_pct_multifamily']
        
        # Zoning Friction Delta
        p_frictions = p_props['standardized_lu'].apply(lambda lu: determine_friction(case_zoning, lu))
        s_frictions = s_props['standardized_lu'].apply(lambda lu: determine_friction(case_zoning, lu))
        
        res['delta_protesting_friction'] = p_frictions.mean() if len(p_frictions) > 0 else 0
        res['delta_silent_friction'] = s_frictions.mean() if len(s_frictions) > 0 else 0
        
        results.append(res)
        
        if idx % 50 == 0:
            print(f"   Processed {idx}/{len(cases_to_process)} cases...")
            
    res_df = pd.DataFrame(results).fillna(0)
    print(f"Completed in {time.time() - t0:.1f}s")
    
    out_path = r'Data\Protest_Petitions\ears_differentials.csv'
    res_df.to_csv(out_path, index=False)
    print(f"Saved EARS longitudinal differentials to {out_path}")
    
    print("3. Merging into Advanced Petition Panel...")
    adv_path = r'Data/Protest_Petitions/advanced_geometric_petition_intensity.csv'
    adv = pd.read_csv(adv_path)
    
    cols_to_drop = [c for c in adv.columns if c in res_df.columns and c != 'case_number']
    adv = adv.drop(columns=cols_to_drop)
    
    merged = pd.merge(adv, res_df, on='case_number', how='left').fillna(0)
    merged.to_csv(adv_path, index=False)
    print("Merged and saved advanced geometric petition intensity using EARS data.")

if __name__ == "__main__":
    build_ears_differentials()
