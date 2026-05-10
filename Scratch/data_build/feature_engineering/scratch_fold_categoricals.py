import pandas as pd
import numpy as np
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data')
WH_DIR = os.path.join(DATA, 'Warehouse_As_Of')

def fold_categoricals():
    v2_path = os.path.join(WH_DIR, 'H0_Filing_Master_Enriched_v2_OmniLagged.csv')
    zoning_path = os.path.join(DATA, 'Zoning_Cases', 'Processed_Data', 'CSV', 'enriched_zoning_data_causal.csv')
    
    print("[*] Loading final V2 Omni-Lag matrix...")
    df = pd.read_csv(v2_path, low_memory=False)
    v2_base_cols = len(df.columns)
    
    print("[*] Loading source Categorical Zoning File...")
    zdf = pd.read_csv(zoning_path, low_memory=False)
    
    # Isolate string categorical data 
    cat_cols = ['case_number', 'case_type', 'proposed_zoning', 'existing_zoning', 
                'proposed_land_use', 'existing_land_use', 'owner_fullname', 
                'owner_organization_name', 'description_of_work']
                
    # Normalize case number for strict joining
    zdf['case_number'] = zdf['Case Number'].fillna(zdf.get('case_number', pd.Series(dtype=str))).astype(str).str.strip().str.upper()
    
    # Drop duplicates by case_number keeping last
    zdf = zdf.drop_duplicates(subset=['case_number'], keep='last')
    
    # Isolate relevant columns and prefix to avoid completely overwriting anything active
    zdf_subset = zdf[[c for c in cat_cols if c in zdf.columns]].copy()
    rename_mapping = {c: f"raw_{c}" for c in zdf_subset.columns if c != 'case_number'}
    zdf_subset.rename(columns=rename_mapping, inplace=True)
    
    print("[*] Folding string categoricals onto V2 Matrix...")
    df = df.merge(zdf_subset, on='case_number', how='left')
    
    missing_desc = df['raw_description_of_work'].isna().sum()
    print(f"    Missing Description of Work strings: {missing_desc}/{len(df)} ({missing_desc/len(df)*100:.1f}%)")
    
    df.to_csv(v2_path, index=False)
    print(f"[*] Done! Appended {len(df.columns) - v2_base_cols} native CoA string columns natively into V2.")

if __name__ == '__main__':
    fold_categoricals()
