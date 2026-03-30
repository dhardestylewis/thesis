"""
causal_did.py
=============
Priority 3: Causal / Econometric Analysis 
Part 2: Does protest prevent development? (DiD: residential vs commercial / protested vs not)

Extracts TCADs involved in zoning cases, matches them against the Property-Year panel, 
and computes the average improvement / market value pre and post case.
Outputs an event study plot showing the trajectory of appraised values.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import ast

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Econometrics")
os.makedirs(OUT_DIR, exist_ok=True)

ZONING_CSV = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "enriched_zoning_data_updated.csv")
PET_CSV = os.path.join(DATA, "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv")
PANEL_CSV = os.path.join(DATA, "Panel", "Output", "Property_Year_Panel_v3.csv")

def extract_tcads(val):
    if pd.isna(val) or val == '[]': return []
    try:
        if isinstance(val, str):
            res = ast.literal_eval(val)
            if isinstance(res, list): return res
    except:
        pass
    return []

def analyze_did():
    print("=== PRIORITY 3: DID DEVELOPMENT ANALYSIS ===")
    
    # 1. Load Zoning Cases
    zoning = pd.read_csv(ZONING_CSV, low_memory=False)
    
    # Simple classification of Residential vs Commercial
    def classify_use(use):
        if pd.isna(use): return 'Unknown'
        use_lower = str(use).lower()
        if 'family' in use_lower or 'residential' in use_lower or 'condo' in use_lower:
            return 'Residential'
        if 'commercial' in use_lower or 'retail' in use_lower or 'office' in use_lower or 'mu' in use_lower:
            return 'Commercial'
        return 'Other'
        
    # Get Year from case number e.g. C14-2007-0131
    zoning['case_number'] = zoning['Case Number'].fillna(zoning['case_number'])
    zoning['Year'] = zoning['case_number'].str.extract(r'C\d+[A-Z]*-(\d{4})')[0].astype(float)
    zoning['group'] = zoning['proposed_land_use'].apply(classify_use)
    
    # Use standardized_tcad_id
    zoning['tcad_list'] = zoning['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    
    cases_explode = zoning[['case_number', 'Year', 'group', 'tcad_list', 'approval_date']].dropna(subset=['tcad_list', 'Year'])
    
    # 2. Join with Protest Data
    pet = pd.read_csv(PET_CSV)
    protested_cases = set(pet['case_number'].str.strip())
    cases_explode['is_protested'] = cases_explode['case_number'].str.strip().isin(protested_cases)
    
    # We only want Approved cases to see development conditional on approval
    cases_explode = cases_explode[cases_explode['approval_date'].notna()]
    
    # Dedup TCADs - keep the earliest case if a parcel is involved in multiple
    cases_dedup = cases_explode.sort_values(by=['tcad_list', 'Year']).drop_duplicates(subset=['tcad_list'])
    
    target_tcads = set(cases_dedup['tcad_list'])
    print(f"Identified {len(target_tcads)} unique parcels involved in APPROVED zoning cases.")
    
    # 3. Read Panel Data in chunks and extract relevant traces
    # Property_Year_Panel_v3 schema: standardized_tcad_id, year, imprv_val, market_value
    print("Scanning panel database (this may take a minute)...")
    
    chunksize = 1000000
    panel_chunks = []
    
    try:
        # Check first row to get column names safely
        head = pd.read_csv(PANEL_CSV, nrows=1)
        
        # We need the ID column name. Could be 'standardized_tcad_id' or 'prop_id' or 'TCAD_ID'
        id_col = 'standardized_tcad_id'
        if id_col not in head.columns:
            for col in head.columns:
                if 'id' in col.lower() or 'tcad' in col.lower():
                    id_col = col
                    break
                    
        print(f"Using ID column: {id_col}")
        
        usecols = [id_col, 'year', 'appraised_value']
        # add total_market_value if available
        if 'total_market_value' in head.columns: usecols.append('total_market_value')
        
        for i, chunk in enumerate(pd.read_csv(PANEL_CSV, chunksize=chunksize, usecols=usecols, low_memory=False)):
            if i % 3 == 0: print(f"Processing row {i*chunksize}...")
            
            # Ensure ID is formatted the same
            chunk[id_col] = chunk[id_col].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
            
            # Filter
            mask = chunk[id_col].isin(target_tcads)
            panel_chunks.append(chunk[mask])
    
    except Exception as e:
        print(f"Error reading panel: {e}")
        return
        
    df_panel = pd.concat(panel_chunks, ignore_index=True)
    df_panel.rename(columns={id_col: 'tcad_list'}, inplace=True)
    
    # Clean money columns
    for money_col in [c for c in df_panel.columns if 'val' in c.lower()]:
        if df_panel[money_col].dtype == object:
            df_panel[money_col] = pd.to_numeric(df_panel[money_col].str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    
    print(f"Extracted {len(df_panel)} panel records for targeted parcels.")
    
    # 4. Merge panel traces with the case characteristics
    merged = pd.merge(df_panel, cases_dedup, on='tcad_list', how='inner')
    
    # Calculate relative time (t)
    # Ensure years are numeric
    merged['year'] = pd.to_numeric(merged['year'], errors='coerce')
    merged['case_year'] = pd.to_numeric(merged['Year'], errors='coerce')
    merged = merged.dropna(subset=['year', 'case_year'])
    
    merged['relative_year'] = merged['year'] - merged['case_year']
    
    # Restrict to window T-3 to T+5
    did_window = merged[(merged['relative_year'] >= -3) & (merged['relative_year'] <= 5)].copy()
    
    # Calculate mean trajectory by group
    grouped = did_window.groupby(['is_protested', 'group', 'relative_year'])['appraised_value'].mean().reset_index()
    
    grouped.to_csv(os.path.join(OUT_DIR, "did_trajectories.csv"), index=False)
    
    # 5. Plotting Event Study
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    groups = ['Residential', 'Commercial']
    for i, g in enumerate(groups):
        ax = axes[i]
        
        # Protested
        p_data = grouped[(grouped['is_protested'] == True) & (grouped['group'] == g)]
        if not p_data.empty:
            ax.plot(p_data['relative_year'], p_data['appraised_value']/1e6, 
                    marker='o', linestyle='-', color='crimson', linewidth=2, label='Protested')
            
        # Unprotested
        u_data = grouped[(grouped['is_protested'] == False) & (grouped['group'] == g)]
        if not u_data.empty:
            ax.plot(u_data['relative_year'], u_data['appraised_value']/1e6, 
                    marker='s', linestyle='--', color='steelblue', linewidth=2, label='Not Protested')
                    
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
        ax.set_title(f"Event Study: {g} Parcels")
        ax.set_xlabel("Years Relative to Zoning Case")
        if i == 0: ax.set_ylabel("Average TCAD Improvement Value ($ Millions)")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(range(-3, 6))
        
    fig.suptitle("Impact of Protest on Development Outcomes (T-3 to T+5)", fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig11_did_development.png"), dpi=150)
    plt.close()
    
    print("Saved fig11_did_development.png and did_trajectories.csv")
    
if __name__ == "__main__":
    analyze_did()
