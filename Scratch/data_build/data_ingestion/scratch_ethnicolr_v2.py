import os
import pandas as pd
import numpy as np
import re
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

def extract_last_name(name_str):
    if pd.isna(name_str): return ""
    match = re.search(r'\((.*?)\)', str(name_str))
    target = match.group(1) if match else str(name_str)
    
    target = target.replace("?", "").replace(",", "").strip()
    parts = target.split()
    return parts[-1] if parts else ""

def execute_ethnicolr_injection():
    try:
        from ethnicolr import pred_census_ln
    except ImportError:
        print("[-] Ethnicolr failed to load even after installation attempts. Aborting.")
        return
        
    v2_path = os.path.join(DATA, 'H0_Filing_Master_Enriched_v2_OmniLagged.csv')
    print("[*] Loading Master Matrix for Neural Identity Parsing...")
    df = pd.read_csv(v2_path, low_memory=False)
    
    if 'raw_owner_fullname' not in df.columns:
        print("[-] 'raw_owner_fullname' column missing. Did you fold categoricals yet?")
        return
        
    base_cols = len(df.columns)
    
    print("[*] Isolating Applicant Identities for Bayesian Ancestry Prediction...")
    df['agent_last_name'] = df['raw_owner_fullname'].apply(extract_last_name)
    
    name_df = pd.DataFrame({'last_name': df['agent_last_name'].unique()})
    name_df = name_df[name_df['last_name'] != ""]
    
    print(f"[*] Processing {len(name_df)} unique legal identities through neural network tensors...")
    eth_preds = pred_census_ln(name_df, 'last_name', year=2010)
    
    eth_cols = ['last_name', 'race', 'pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']
    eth_merge = eth_preds[[c for c in eth_cols if c in eth_preds.columns]]

    # Map variables smoothly
    rename_map = {c: f"eth_owner_{c}" for c in eth_merge.columns if c != 'last_name'}
    eth_merge.rename(columns=rename_map, inplace=True)
    
    # Merge onto main dataframe
    df = df.merge(eth_merge, left_on='agent_last_name', right_on='last_name', how='left')
    
    # Clean up processing artifacts
    df.drop(columns=['agent_last_name', 'last_name'], inplace=True, errors='ignore')
    
    df.to_csv(v2_path, index=False)
    print("\n[+] SUCCESS: Bayesian identity probabilities firmly injected.")
    print(f"    -> Extracted fields: {list(rename_map.values())}")
    print(f"    -> Master Matrix expanded from {base_cols} to {len(df.columns)} columns.")

if __name__ == '__main__':
    execute_ethnicolr_injection()
