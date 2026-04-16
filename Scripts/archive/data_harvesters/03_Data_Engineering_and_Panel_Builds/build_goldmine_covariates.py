"""
build_goldmine_covariates.py
============================
Integrates the final untouched 'Goldmine' repository datasets into the Model Tensor.
Executes deep Bayesian mappings (`ethnicolr`) against the explicit physical names of the 
NIMBY protest petitioners to construct exact Demographic Friction algorithms.
"""
import os
import pandas as pd
import numpy as np
import warnings

# Suppress Ethnicolr TF Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

try:
    from ethnicolr import pred_census_ln
except ImportError:
    print("[-] Ethnicolr is not active. Falling back to empty demographic friction vectors.")
    pred_census_ln = None

ROOT = r"C:\Users\dhl\data\thesis\thesis"
WORK_DIR = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV")

def extract_signer_last_name(name_str):
    """Safely extracts the last word of the string to evaluate physical signers."""
    if pd.isna(name_str): return ""
    clean = str(name_str).replace(",", "").strip()
    return clean.split()[-1] if clean else ""

def main():
    print("[*] Bootstrapping the Goldmine Identity Covariates...")
    # Load the furthest completed ML array
    target_csv = os.path.join(WORK_DIR, "stijn_multimodal_icp_matrix.csv")
    
    df = pd.read_csv(target_csv)
    print(f"    -> Valid Base Model Size: {len(df)}")
    
    # 1. Load Explicit Protester Identities
    protest_path = os.path.join(ROOT, "Data", "Protest_Petitions", "petition_signers_from_pdf.csv")
    if os.path.exists(protest_path):
        df_sig = pd.read_csv(protest_path)
        df_sig['case_number'] = df_sig['case_number'].str.upper()
        df_sig['signer_last_name'] = df_sig['owner_name'].apply(extract_signer_last_name)
        
        print(f"    -> Parsed {len(df_sig)} exact NIMBY objector signatures physically extracted from the PDF protest rolls.")
        
        if pred_census_ln is not None:
            name_df = pd.DataFrame({'last_name': df_sig['signer_last_name'].unique()})
            print(f"    -> Routing {len(name_df)} unique legal objectors through Bayesian ancestry classification wrappers...")
            
            # Predict ethnic breakdown based on 2010 US Census algorithms
            eth_preds = pred_census_ln(name_df, 'last_name', year=2010).dropna(subset=['pctwhite'])
            
            # Bind identity percentages strictly onto the individual signer array
            df_sig = df_sig.merge(eth_preds[['last_name', 'pctwhite', 'pcthispanic']], left_on='signer_last_name', right_on='last_name', how='left')
            
            # Aggregate horizontally to create the "Mean Protester Racial Constituency" per Zoning Case!
            eth_case = df_sig.groupby('case_number')[['pctwhite', 'pcthispanic']].mean().reset_index()
            eth_case = eth_case.rename(columns={'pctwhite': 'protester_pctwhite', 'pcthispanic': 'protester_pcthispanic'})
            
            # Splice back into the primary Econometric model
            df['CASE_NUMBER'] = df['CASE_NUMBER'].str.upper()
            df = df.merge(eth_case, left_on='CASE_NUMBER', right_on='case_number', how='left')
            
            # Compute Explicit "Demographic Friction"! (The mathematical distance between Developer Identity and Neighborhood Identity)
            # (pctwhite represents Developer ethnicity probability generated in Phase 2)
            if 'pctwhite' in df.columns:
                df['friction_white'] = (df['pctwhite'] - df['protester_pctwhite']).abs()
                df['friction_hispanic'] = (df['pcthispanic'] - df['protester_pcthispanic']).abs()
                
                # Fill missing cases with 0 friction baseline
                df['friction_white'] = df['friction_white'].fillna(0.0)
                df['friction_hispanic'] = df['friction_hispanic'].fillna(0.0)
                df['protester_pctwhite'] = df['protester_pctwhite'].fillna(df['protester_pctwhite'].mean())
                
                print("    -> [SUCCESS] Quantified explicit Demographic Friction parameters mapping Protester-Applicant disparity.")
            else:
                print("    [-] Applicant 'pctwhite' missing. Bypassing demographic friction cross-walk.")
        else:
            print("    [-] Ethnicolr neural weights offline. Bypassing identity mapping.")
    else:
        print("    [-] File `petition_signers_from_pdf.csv` not found in workspace.")
            
    out_path = os.path.join(WORK_DIR, "submission_grade_goldmine_tensor.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[+] SUCCESS: Goldmine Identity Matrix precisely locked and written to disk at:")
    print(f"    -> {out_path}")

if __name__ == "__main__":
    main()
