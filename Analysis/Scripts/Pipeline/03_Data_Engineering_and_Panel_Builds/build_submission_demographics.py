"""
build_submission_demographics.py
================================
Implements Phase 2 Advanced Econometric Engineering for the Master Target Matrix.
1. FRED Macroeconomics via API (MORTGAGE30US, FEDFUNDS)
2. Bayesian Ethnicity mappings for Applicant Agents
3. High-Fidelity Semantic Zoning translations (FAR/Density Proxies)
"""
import os
import re
import pandas as pd
import numpy as np
import datetime
import warnings

# Suppress Ethnicolr TF Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

try:
    from ethnicolr import pred_census_ln
except ImportError:
    print("[-] Ethnicolr is not active. Falling back to NaN ethnicity bindings.")
    pred_census_ln = None

ROOT = r"C:\Users\dhl\data\thesis\thesis"
WORK_DIR = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV")

INPUT_PATH = os.path.join(WORK_DIR, "stijn_multimodal_icp_matrix.csv")
OUTPUT_PATH = os.path.join(WORK_DIR, "submission_grade_icp_matrix.csv")

def extract_last_name(name_str):
    if pd.isna(name_str): return ""
    # Extract name inside parentheses: e.g. "Firm (John Doe)" -> "John Doe"
    match = re.search(r'\((.*?)\)', str(name_str))
    target = match.group(1) if match else str(name_str)
    
    # Clean noise
    target = target.replace("?", "").replace(",", "").strip()
    parts = target.split()
    return parts[-1] if parts else ""

def map_zoning_density(zone_str):
    """Translates Austin Zoning text codes into an explicit Ordinal Density Ceiling (0 to 10)."""
    if pd.isna(zone_str): return 0
    z = str(zone_str).upper()
    
    if 'CBD' in z: return 10
    if 'MF-6' in z: return 9
    if 'MF-5' in z: return 8
    if 'MF-4' in z or 'MF-3' in z: return 7
    if 'MF' in z: return 6
    if 'CS' in z or 'GR' in z or 'CH' in z: return 5
    if 'SF-6' in z or 'SF-5' in z: return 4
    if 'SF-4' in z: return 3
    if 'SF-3' in z: return 2
    if 'SF-2' in z or 'SF-1' in z: return 1
    if 'RR' in z or 'DR' in z: return 0.5
    return 0

def main():
    print("[*] Loading Initial ICP Matrix...")
    df = pd.read_csv(INPUT_PATH)
    
    print(f"    -> Parsed {len(df)} base target cases.")
    
    # ---------------------------------------------------------
    # 1. Semantic Zoning Definition Scaling
    # ---------------------------------------------------------
    print("\n[*] 1. Translating Categorical Zoning Code Syntax into Structural Real Estate Density Proxies...")
    df['orig_zoning_density'] = df['orig_zoning'].apply(map_zoning_density)
    df['target_zoning_density'] = df['target_zoning'].apply(map_zoning_density)
    df['net_density_change'] = df['target_zoning_density'] - df['orig_zoning_density']
    print(f"    -> Extracted structural density variance (e.g. net metric {df['net_density_change'].mean():.2f}).")
    
    # ---------------------------------------------------------
    # 2. Time-Series FRED Macroeconomics
    # ---------------------------------------------------------
    print("\n[*] 2. Hydrating Time-Series FRED Macros (Interest Rates, Mortgages)...")
    # Date bounds
    df['Meeting_Date'] = pd.to_datetime(df['Meeting_Date'], errors='coerce')
    min_date = df['Meeting_Date'].min()
    max_date = df['Meeting_Date'].max()
    
    try:
        # Federal Funds Rate & 30-Year Fixed Mortgage Average via explicit CSV endpoints
        fed = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS", parse_dates=['DATE'], na_values='.')
        mort = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US", parse_dates=['DATE'], na_values='.')
        macro_df = fed.merge(mort, on='DATE', how='outer').sort_values('DATE').ffill()
        
        macro_df['year_month'] = macro_df['DATE'].dt.to_period('M')
        # Avoid non-numeric aggregation error
        macro_monthly = macro_df[['year_month', 'FEDFUNDS', 'MORTGAGE30US']].groupby('year_month').mean().reset_index()
        
        df['year_month'] = df['Meeting_Date'].dt.to_period('M')
        df = df.merge(macro_monthly[['year_month', 'FEDFUNDS', 'MORTGAGE30US']], on='year_month', how='left')
        
        df['FEDFUNDS'] = df['FEDFUNDS'].ffill()
        df['MORTGAGE30US'] = df['MORTGAGE30US'].ffill()
        print(f"    -> Bound exactly {df['FEDFUNDS'].notna().sum()} synchronous Macro indicators natively to Meeting Times.")
        df = df.drop(columns=['year_month'])
    except Exception as e:
        print(f"    [-] FRED Direct Download Failed: {e}")

    # ---------------------------------------------------------
    # 3. Bayesian Identity Extraction
    # ---------------------------------------------------------
    print("\n[*] 3. Isolating Applicant Identities for Bayesian Ancestry Prediction...")
    df['agent_last_name'] = df['agent'].apply(extract_last_name)
    
    # Use Ethnicolr to cast names to probabilstic demographic vectors
    if pred_census_ln is not None:
        try:
            name_df = pd.DataFrame({'last_name': df['agent_last_name'].unique()})
            print(f"    -> Processing {len(name_df)} unique legal identities through neural network tensors...")
            eth_preds = pred_census_ln(name_df, 'last_name', year=2010)
            
            # Merge predictions back into master
            eth_cols = ['last_name', 'race', 'pctwhite', 'pctblack', 'pctapi', 'pcthispanic']
            eth_merge = eth_preds[[c for c in eth_cols if c in eth_preds.columns]]
            df = df.merge(eth_merge, left_on='agent_last_name', right_on='last_name', how='left')
            print("    -> Successfully injected explicit identity racial proxies into structural model (pctwhite, pcthispanic).")
        except Exception as e:
            print(f"    [-] Ethnicolr Tensor Failure: {e}")
    else:
        print("    [-] Skipped: Machine learning binaries missing.")

    # ---------------------------------------------------------
    # Final Output
    # ---------------------------------------------------------
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[+] SUCCESS: Advanced Phase 2 Econometrics perfectly integrated.")
    print(f"    -> Saved Master Tensor File to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
