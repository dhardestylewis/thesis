import pandas as pd
import numpy as np
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data")

H0_BASE = os.path.join(DATA, "Warehouse_As_Of", "H0_Filing_Complete.csv")
ZONING_CAUSAL = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "CSV", "enriched_zoning_data_causal.csv")
ENRICHED_PANEL = os.path.join(DATA, "Panel", "Output", "Property_Year_Panel_Enriched.csv")
OUT_FILE = os.path.join(DATA, "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")

def run():
    print("Loading H0 Structural Geometries...")
    h0 = pd.read_csv(H0_BASE)
    
    print("Loading ZONING_CAUSAL to map TCAD IDs...")
    crosswalk = pd.read_csv(ZONING_CAUSAL, usecols=['Case Number', 'TCAD ID'])
    
    # Clean strings
    crosswalk['case_number'] = crosswalk['Case Number'].astype(str).str.strip().str.upper()
    crosswalk = crosswalk.drop_duplicates(subset=['case_number']).copy()
    
    # Clean TCAD ID
    crosswalk['TCAD ID'] = crosswalk['TCAD ID'].astype(str).str.replace(r'[- ]', '', regex=True).str.lstrip('0')
    
    # Extrapolate calendar year securely using physical RegEx on the C14-YYYY strings
    crosswalk['year'] = pd.to_numeric(crosswalk['case_number'].str.extract(r'C\d+[A-Z]*-(\d{4})')[0], errors='coerce')
    # Fallback to general year regex if C14 format misses
    crosswalk['year'] = crosswalk['year'].fillna(pd.to_numeric(crosswalk['case_number'].str.extract(r'((?:19|20)\d\d)')[0], errors='coerce'))
    crosswalk['year'] = crosswalk['year'].fillna(2020) # absolute baseline target

    
    # Join crosswalk onto H0
    df = h0.merge(crosswalk[['case_number', 'TCAD ID']], on='case_number', how='left')
    
    print("Loading the massive Enriched Master Panel...")
    panel = pd.read_csv(ENRICHED_PANEL, low_memory=False)
    
    # We only care about joining onto properties that exist in H0, so standardizing panel:
    panel['standardized_tcad_id'] = panel['standardized_tcad_id'].astype(str).str.replace(r'[- ]', '', regex=True).str.lstrip('0')
    panel['year'] = pd.to_numeric(panel['year'], errors='coerce')
    
    # Merge Enriched Panel features to df using TCAD ID and YEAR (the proven 'As-Of' engine)
    print("Executing Time-Stamped As-Of Empirical Joins...")
    final = df.merge(panel, left_on=['TCAD ID', 'year'], right_on=['standardized_tcad_id', 'year'], how='left')
    
    print(f"Final Enriched Matrix dynamically expanded to {final.shape[1]} columns for {len(final)} unique Zoning Cases.")
    
    # ----------------------------------------------------
    # NEW FEATURE: Spatial Contagion (Historical Protests)
    # ----------------------------------------------------
    print("Calculating endogenous spatial contagion (past protested zoning cases)...")
    if 'latitude' in final.columns and 'longitude' in final.columns:
        lats = np.radians(final['latitude'].fillna(0).values)
        lons = np.radians(final['longitude'].fillna(0).values)
        years = final['year'].values
        protests = final['is_protested'].fillna(0).values
        
        contagion_1yr = np.zeros(len(final))
        contagion_3yr = np.zeros(len(final))
        MILE_RADS = 1.0 / 3958.8 # 1 mile radius in radians
        
        for i in range(len(final)):
            if lats[i] == 0: continue
            
            dlat = lats - lats[i]
            dlon = lons - lons[i]
            a = np.sin(dlat/2)**2 + np.cos(lats[i]) * np.cos(lats) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
            mask_dist = c <= MILE_RADS
            
            mask_past = (years < years[i])
            mask_protested = (protests == 1)
            
            mask_1yr = mask_past & (years >= (years[i] - 1)) & mask_dist & mask_protested
            mask_3yr = mask_past & (years >= (years[i] - 3)) & mask_dist & mask_protested
            
            contagion_1yr[i] = mask_1yr.sum()
            contagion_3yr[i] = mask_3yr.sum()
            
        final['spatial_contagion_1yr'] = contagion_1yr
        final['spatial_contagion_3yr'] = contagion_3yr
        print(f"Endogenous clustering calculated. Avg 3-yr contagion: {contagion_3yr.mean():.3f} per case")
    else:
        print("[!] Warning: Missing latitude/longitude. Spatial contagion failed.")
    # ----------------------------------------------------
    
    # Validate missing merges
    missing = final['standardized_tcad_id'].isna().sum()
    print(f"Warning: {missing} cases failed to match to the background Enriched Panel.")
    
    final.to_csv(OUT_FILE, index=False)
    print(f"Architectural execution complete. V2 Warehouse physical output bound to {OUT_FILE}.")

if __name__ == '__main__':
    run()
