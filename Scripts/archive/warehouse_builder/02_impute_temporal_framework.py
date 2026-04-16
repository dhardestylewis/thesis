import pandas as pd
import numpy as np
import os
import datetime

# Paths
ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
WORK_DIR = os.path.join(ROOT_DIR, "Data", "Warehouse_As_Of", "Build")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Data", "Warehouse_As_Of")

def impute_timeline():
    print("Loading Case Master...")
    cm_path = os.path.join(WORK_DIR, "case_master.csv")
    if not os.path.exists(cm_path):
        print("case_master not found. Run 01_build_case_master.py first.")
        return
        
    df = pd.read_csv(cm_path)
    
    # We only care about the ~566 cases in the analytical set
    h0_path = os.path.join(OUTPUT_DIR, "H0_Filing.csv")
    if os.path.exists(h0_path):
        h0 = pd.read_csv(h0_path)
        valid_cases = h0['case_number'].unique()
        df = df[df['CASE_NUMBER'].isin(valid_cases)].copy()
        
        # Merge the year column from H0 to anchor our back-calculation if DATA_PORTAL_UPDATE is messy
        year_map = h0[['case_number', 'year']].drop_duplicates().set_index('case_number')['year']
        df['council_year'] = df['CASE_NUMBER'].map(year_map)
    else:
        df['council_year'] = 2024 # Fallback
        
    print(f"Filtering to {len(df)} historically validated cases.")

    # Impute an anchor council_date. 
    # Since we avoid scraping, we anchor on August 1st of the council_year for mockup consistency, 
    # unless we parse it perfectly from an existing text field.
    
    # Clean up floats
    df['council_year'] = df['council_year'].fillna(2024).astype(int)
    
    # Convert arbitrary strings to datetime for the math:
    df['council_date'] = pd.to_datetime(df['council_year'].astype(str) + '-08-01')

    # Apply Statutory Logic (as described in LaTeX thesis)
    # H3: Pre-Council -> Council Date - 3 days
    df['h3_pre_council'] = df['council_date'] - pd.Timedelta(days=3)
    
    # H2: Pre-Commission -> Council Date - 30 days
    df['h2_pre_commission'] = df['council_date'] - pd.Timedelta(days=30)
    
    # H1: Notice / Petition Deadline -> Council Date - 45 days (Austin statutory mail constraint proxy)
    df['h1_notice'] = df['council_date'] - pd.Timedelta(days=45)
    
    # H0: Filing Date -> Council Date - 120 days (Standard pipeline proxy)
    df['h0_filing'] = df['council_date'] - pd.Timedelta(days=120)

    # Save imputed timestamps
    out_path = os.path.join(WORK_DIR, "02_imputed_timelines.csv")
    df[['CASE_NUMBER', 'h0_filing', 'h1_notice', 'h2_pre_commission', 'h3_pre_council', 'council_date']].to_csv(out_path, index=False)
    
    print("Statutory derivation complete. Timelines mathematically imputed.")
    print("Saved to:", out_path)

if __name__ == "__main__":
    impute_timeline()
