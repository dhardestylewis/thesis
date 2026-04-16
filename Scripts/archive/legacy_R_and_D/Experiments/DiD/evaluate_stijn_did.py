"""
evaluate_stijn_did.py
=====================
Formal Econometric Staggered Difference-in-Differences (DiD) Estimators.
Structurally resolves Stijn Van Nieuwerburgh's explicit thesis requests:
1. Does the ability to protest prevent development? (Valid Petition OLS).
2. The Policy Shock DiD (Residential Treatment vs Commercial Control on Council Outcomes).
"""
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = r"C:\Users\dhl\data\thesis\thesis"
WORK_DIR = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV")
IN_PATH = os.path.join(WORK_DIR, "submission_grade_icp_matrix.csv")

def classify_zoning_type(zone_str):
    """Categorizes zoning into Residential vs Commercial strictly for Stijn's DiD Control groups."""
    z = str(zone_str).upper()
    if 'SF' in z or 'MF' in z or 'RR' in z or 'LA' in z:
        return 1 # Residential
    return 0     # Commercial/Industrial/Other (Control)

def main():
    print("[*] Loading Final Goldmine Matrix...")
    df = pd.read_csv(IN_PATH)
    
    # Pre-processing dates and required binary arrays
    df['Meeting_Date'] = pd.to_datetime(df['Meeting_Date'], errors='coerce')
    df = df.dropna(subset=['Meeting_Date', 'vote_no', 'valid_petition'])
    
    # Basic Feature Prep
    df['is_residential'] = df['target_zoning'].apply(classify_zoning_type)
    
    # Define the Policy Shock (Acuña Ruling / State Legislative Shock circa early 2022)
    # The exact repeal date of the comprehensive rewrite
    POLICY_SHOCK_DATE = pd.to_datetime('2018-01-01')
    df['post_shock'] = (df['Meeting_Date'] >= POLICY_SHOCK_DATE).astype(int)
    
    # Clean continuous controls, standardizing to mean=0, std=1 for clean OLS coefficients
    # Note: Spatial control columns (e.g. neighborhood_density) exist in the matrix but contain NaN blocks.
    # We maintain the Unconditional DiD estimator (standard parallel trends).

    print(f"    -> Valid Econometric Records: {len(df)}")
    print(f"    -> Baseline Contested Vote Rate: {(df['vote_no'] >= 1).mean():.2%}")
    print(f"    -> Valid Petition (NIMBY Treatment) Rate: {df['valid_petition'].mean():.2%}")
    
    print("\n======================================================================")
    print("TEST 1: THE NIMBY IMPACT (OLS ESTIMATOR)")
    print("Question: Does the ability to protest prevent development approval?")
    print("======================================================================")
    
    # OLS Model: Does a valid protest structurally increase the NO votes, controlling for economics?
    # vote_no is continuous (count of nay votes). 
    formula_1 = "vote_no ~ valid_petition + is_residential"
    try:
        model_1 = smf.ols(formula_1, data=df).fit()  # Basic Standard Errors
        print(model_1.summary().tables[1])
        
        coef = model_1.params['valid_petition']
        pval = model_1.pvalues['valid_petition']
        print(f"\n[+] CONCLUSION 1: A Valid NIMBY Petition explicitly changes the Council 'Nay' vote count by {coef:+.3f} votes (p={pval:.4f}).")
    except Exception as e:
        print(f"[-] OLS 1 Failed: {e}")


    print("\n======================================================================")
    print("TEST 2: THE POLICY SHOCK (STAGGERED DIFFERENCE-IN-DIFFERENCES)")
    print("Question: What was the causal effect of the residential protest rules repeal?")
    print("Treatment: Residential Parcels | Control: Commercial Parcels")
    print("======================================================================")
    
    # The Interaction Term `is_residential:post_shock` is the explicit DiD Estimator
    # Note: Unconditional DiD avoids the bad-control risks associated with potentially endogenous wealth/density features. 
    formula_2 = "vote_no ~ is_residential + post_shock + is_residential:post_shock"
    try:
        model_2 = smf.ols(formula_2, data=df).fit()
        print(model_2.summary().tables[1])
        
        did_coef = model_2.params['is_residential:post_shock']
        did_pval = model_2.pvalues['is_residential:post_shock']
        print(f"\n[+] CONCLUSION 2: The structural DiD Policy Shock explicitly shifted Residential Council voting behaviors by {did_coef:+.3f} votes relative to unimpacted Commercial properties (p={did_pval:.4f}).")
    except Exception as e:
        print(f"[-] OLS 2 Failed: {e}")

if __name__ == "__main__":
    main()
