import os
import sys
import numpy as np
import pandas as pd
import warnings
import re
warnings.filterwarnings('ignore')

try:
    from surgeo import SurgeoModel
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "surgeo"])
    from surgeo import SurgeoModel

def extract_surname(name_str):
    if not isinstance(name_str, str) or not name_str.strip():
        return None
    name = name_str.strip().upper()
    for kw in ['LLC', 'INC', 'CORP', 'TRUST', 'LP', 'LTD', 'ASSOC', 'BANK', 'FUND', 'HOMES']:
        if kw in name: return None
    if ',' in name:
        return name.split(',')[0].strip()
    parts = name.split()
    if parts: return parts[0].strip()
    return None

def extract_zip(situs):
    if not isinstance(situs, str): return None
    match = re.search(r'\b(\d{5})\b', situs)
    return match.group(1) if match else None

def build_bisg_case_features():
    print("[+] Starting BISG Case-Level Aggregation Pipeline...")
    ROOT = r"C:\Users\dhl\data\thesis\thesis"
    
    panel_file = os.path.join(ROOT, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
    map_file   = os.path.join(ROOT, "Data", "Warehouse_As_Of", "Build", "case_buffer_map.csv")
    h0_file    = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
    
    print("[+] Loading Buffer Map and Property Panel Names...")
    b_map = pd.read_csv(map_file, usecols=['CASE_NUMBER', 'neighbor_tcad_id'], dtype=str)
    
    # We only care about standardized_tcad_id, owner_name, situs_city_state_zip to save memory
    p_names = pd.read_csv(panel_file, usecols=['standardized_tcad_id', 'owner_name', 'situs_city_state_zip'], dtype=str)
    
    # Drop duplicates because the same property in different years has the same name usually, or take the latest
    p_names = p_names.drop_duplicates(subset=['standardized_tcad_id'], keep='last')
    
    print(f"    Map keys: {len(b_map):,} | Property IDs: {len(p_names):,}")
    
    merged = pd.merge(b_map, p_names, left_on='neighbor_tcad_id', right_on='standardized_tcad_id', how='inner')
    print(f"    Successfully matched {len(merged):,} case-neighbor pairs.")
    
    # Run BISG
    merged['_surname'] = merged['owner_name'].apply(extract_surname)
    merged['_zip'] = merged['situs_city_state_zip'].apply(extract_zip)
    
    has_both = merged['_surname'].notna() & merged['_zip'].notna()
    n_valid = has_both.sum()
    print(f"[+] Valid surname+zip pairs across all neighbors: {n_valid:,} ({(n_valid/len(merged))*100:.1f}%)")
    
    valid_df = merged[has_both].copy()
    
    model = SurgeoModel()
    # It takes Series: zip, zcta, surname
    res = model.get_probabilities(valid_df['_surname'], valid_df['_zip'])
    
    # Map surgeo results back
    mapping = {
        'white': 'bisg_white_200ft',
        'black': 'bisg_black_200ft',
        'api': 'bisg_asian_200ft',
        'hispanic': 'bisg_hispanic_200ft'
    }
    for k, v in mapping.items():
        valid_df[v] = res[k].values
        
    print("[+] Aggregating BISG to Case Level...")
    agg_cols = list(mapping.values())
    case_bisg = valid_df.groupby('CASE_NUMBER')[agg_cols].mean().reset_index()
    
    print(f"    Generated BISG features for {len(case_bisg):,} unique zoning cases.")
    
    # Merge back into H0_Filing_Master_Enriched
    print("[+] Merging into H0_Filing Master Matrix...")
    h0_master = pd.read_csv(h0_file, low_memory=False)
    
    # Remove old bisg cols if they exist
    for col in agg_cols:
        if col in h0_master.columns:
            h0_master = h0_master.drop(columns=[col])
            
    # H0 may use 'case_number' or 'CASE_NUMBER'
    merge_key = 'case_number' if 'case_number' in h0_master.columns else 'CASE_NUMBER'
    case_bisg = case_bisg.rename(columns={'CASE_NUMBER': merge_key})
    
    h0_master = pd.merge(h0_master, case_bisg, on=merge_key, how='left')
    
    # Impute missing with census baseline if possible, or 0
    for col in agg_cols:
        h0_master[col] = h0_master[col].fillna(h0_master[col].mean())
        
    h0_master.to_csv(h0_file, index=False)
    print("[+] Pipeline Complete! Updated H0_Filing_Master_Enriched.csv successfully with new Demographic 200ft Buffer Features.")
    
if __name__ == "__main__":
    build_bisg_case_features()
