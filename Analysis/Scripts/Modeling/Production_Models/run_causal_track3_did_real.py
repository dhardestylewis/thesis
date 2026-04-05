import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from artifact_registry import ROOT_DIR, TraceabilityRegistry as AR

ROOT = str(ROOT_DIR)
DATA_H0 = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
VOTE_DATA = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "submission_grade_goldmine_tensor.csv")
OUT_DIR = str(AR.TRACK3_METRICS)
os.makedirs(OUT_DIR, exist_ok=True)

def run_track3():
    print("==============================================")
    print(" TRACK 3: Authentic Causal Study 2 (DiD)")
    print("==============================================")
    
    if not os.path.exists(DATA_H0) or not os.path.exists(VOTE_DATA):
        print("[-] Required data sources not found. Track 3 cannot proceed.")
        return
        
    df_h0 = pd.read_csv(DATA_H0, low_memory=False)
    # Merge available actual voting outcome variables
    df_votes = pd.read_csv(VOTE_DATA, usecols=['CASE_NUMBER', 'vote_no', 'vote_yes'])
    # Deduplicate: aggregate per-member votes to one row per case
    df_votes = df_votes.groupby('CASE_NUMBER', as_index=False).agg({'vote_no': 'sum', 'vote_yes': 'sum'})
    df = df_h0.merge(df_votes, left_on='case_number', right_on='CASE_NUMBER', how='inner')
    
    if df.empty or 'vote_no' not in df.columns:
        print("[!] No authentic vote variables found. Halting Track 3 to prevent synthetic generation.")
        with open(os.path.join(OUT_DIR, "did_results.txt"), "w") as f:
            f.write("Track 3 aborted: No authentic dependent variables (vote_no) successfully joined.")
        return

    # Impute missing votes as 0 (uncontested consent items typically pass unanimously)
    df['vote_no'] = df['vote_no'].fillna(0)
    
    # Establish applicability (Treatment group)
    if 'zoning_code' in df.columns:
        df['is_residential'] = df['zoning_code'].astype(str).str.contains('SF-', na=False).astype(int)
    else:
        df['is_residential'] = df['property_category_code'].astype(str).str.startswith('A', na=False).astype(int)
    
    home1_time = 2024.0
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['post_home_phase1'] = (df['year'] >= home1_time).astype(int)
    df['treated'] = df['is_residential'] * df['post_home_phase1']
    
    out_lines = [
        "Track 3: Authentic Event Study DiD (Actual Voting Records)",
        "========================================================"
    ]
    
    out_lines.append("1. STATIC TWFE BASELINE (Average Treatment Effect on Vote_No Outcome)")
    model_static = smf.ols("vote_no ~ treated + is_residential + C(year)", data=df)
    try:
        res_static = model_static.fit(cov_type='HC1')
        out_lines.append(res_static.summary().tables[1].as_text())
        try:
            est = res_static.params['treated']
            se = res_static.bse['treated']
            p = res_static.pvalues['treated']
            conf_int = res_static.conf_int().loc['treated']
            
            sig_text = "statistically significant" if p < 0.05 else "statistically insignificant"
            dir_text = "increase" if est > 0 else "decrease"
            
            from Utilities_and_Logs.lib_metrics import update_metric
            update_metric("metricDiDVotes", f"+{est:.3f}" if est > 0 else f"{est:.3f}")
            update_metric("metricDiDVotingShift", f"{abs(est)/11.0 * 100:.0f}\\%" if est != 0 else "0\\%")
            update_metric("metricDiDCI", f"[{conf_int[0]:.2f}, {conf_int[1]:.2f}]")
            update_metric("metricDiDSE", f"{se:.3f}")
            update_metric("metricDiDSignificanceText", sig_text)
            update_metric("metricDiDDirectionText", dir_text)
        except Exception as e:
            print(f"    [!] Macro Telemetry Export Failed: {e}")
    except Exception as e:
        out_lines.append(f"Model failed to converge due to missing temporal variance. Reason: {str(e)}")
        
    out_lines.append("")
    out_lines.append("[!] Note: This is an authentic causal regression run purely on extracted vote distributions. Previous versions systematically mapped np.random against drafting expectations. This output represents empirical ground truth irrespective of manuscript claims.")
        
    with open(os.path.join(OUT_DIR, "did_results.txt"), "w") as f:
        f.write("\n".join(out_lines))
        
    print("[+] Track 3 Complete using actual `vote_no`. Fabricated `np.random` architectures explicitly purged.")

if __name__ == '__main__':
    run_track3()
