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
DATA_PETITION = os.path.join(ROOT, "Data", "Protest_Petitions", "petition_summary_from_pdf.csv")
COA_RAW = os.path.join(ROOT, "Data", "CoA_Open_Data", "Zoning", "ZC_current_edir-dcnf.csv")
OUT_DIR = str(AR.TRACK2_METRICS)
os.makedirs(OUT_DIR, exist_ok=True)

def run_rd(df, bw, threshold=0.20, exclude_donut=None):
    """Run local linear regression RD with triangular kernel."""
    df_temp = df.copy()
    if exclude_donut:
        # Exclude observations within [threshold - donut, threshold + donut]
        df_temp = df_temp[~df_temp['signed_area_share'].between(threshold - exclude_donut, threshold + exclude_donut)]
        
    df_temp['running_var'] = df_temp['signed_area_share'] - threshold
    df_temp['post_threshold'] = (df_temp['running_var'] >= 0).astype(int)
    
    # Restrict to bandwidth
    df_bw = df_temp[df_temp['running_var'].abs() <= bw].copy()
    
    # Triangular weight
    df_bw['weight'] = 1 - (df_bw['running_var'].abs() / bw)
    
    if len(df_bw) < 10:
        return None
        
    model = smf.wls("days_delayed ~ running_var * post_threshold", data=df_bw, weights=df_bw['weight'])
    try:
        res = model.fit(cov_type='HC1')
        return res
    except:
        return None

def run_track2():
    print("==============================================")
    print(" TRACK 2: Authentic Causal Study 1 (RD)")
    print("==============================================")
    
    if not os.path.exists(DATA_H0) or not os.path.exists(DATA_PETITION) or not os.path.exists(COA_RAW):
        print("[-] Required data sources not found.")
        return
        
    df_h0 = pd.read_csv(DATA_H0, low_memory=False)
    df_pet = pd.read_csv(DATA_PETITION, low_memory=False)
    df_coa = pd.read_csv(COA_RAW, low_memory=False)
    
    # Merge H0 filing base with real extracted petition signature percentages
    df = df_h0.merge(df_pet[['case_number', 'signer_pct']], on='case_number', how='left')
    
    # Fill cases without petitions as 0%
    df['signed_area_share'] = df['signer_pct'].fillna(0)
    
    # Calculate days_delayed dynamically using root Open Data chronology missing from Warehouse Builder
    df_coa['start'] = pd.to_datetime(df_coa['APPLICATION_START_DATE'], errors='coerce')
    df_coa['end'] = pd.to_datetime(df_coa['FINAL_DATE'], errors='coerce')
    df_coa['days_delayed_raw'] = (df_coa['end'] - df_coa['start']).dt.days
    
    # Keep only logically sound delays (dropping negatives or infinite/missing)
    valid_dates = df_coa.dropna(subset=['CASE_NUMBER', 'days_delayed_raw'])
    valid_dates = valid_dates[valid_dates['days_delayed_raw'] >= 0]
    
    # Map back into df payload
    delay_map = valid_dates.set_index('CASE_NUMBER')['days_delayed_raw'].to_dict()
    df['days_delayed'] = df['case_number'].map(delay_map)
    
    # Drop rows without timeline dependencies for this specific modeling track
    df = df.dropna(subset=['days_delayed'])
    
    # Covariates for continuity test
    df['cov_parcel_size'] = df.get('calculated_acres', df.get('acres', 0)) # Fallback if acres isn't universally mapped
    df['cov_base_zoning'] = df.get('zoning_code', 'Unmapped').astype('category').cat.codes
    
    out_lines = [
        "Track 2: Authentic Regression Discontinuity (Real Timeline Delay Extraction)",
        "========================================================"
    ]
    
    # 1. Main RD Model (Bandwidth 0.10)
    res_main = run_rd(df, bw=0.10)
    if res_main is None:
        out_lines.append("[!] Insufficient variance locally to run bandwidth 0.10. Array too sparse when dropping tracking nulls.")
        print("[!] Track 2 Failed: Insufficient density directly on the bandwidth.")
        with open(os.path.join(OUT_DIR, "rd_results.txt"), "w") as f:
            f.write("\n".join(out_lines))
        return

    out_lines.append(f"Empirical Analytical N (Complete timelines mapping): {len(df)}")
    out_lines.append(f"Average City Pipeline Delay: {df['days_delayed'].mean():.1f} days")
    out_lines.append("")
        
    out_lines.append("1. MAIN LOCAL LINEAR RD ESTIMATE (Bandwidth = 0.10, Triangular Kernel)")
    out_lines.append(res_main.summary().tables[1].as_text())
    out_lines.append("")
    try:
        est = res_main.params['post_threshold']
        se = res_main.bse['post_threshold']
        p = res_main.pvalues['post_threshold']
        conf_int = res_main.conf_int().loc['post_threshold']
        
        sig_text = "statistically significant" if p < 0.05 else "statistically insignificant"
        dir_text = "delay" if est > 0 else "acceleration"
        
        from Utilities_and_Logs.lib_metrics import update_metric
        update_metric("metricRDDelay", f"+{est:.1f}" if est > 0 else f"{est:.1f}")
        update_metric("metricRDDelayWeeks", f"{est/7.0:.1f}")
        update_metric("metricRDCI", f"[{conf_int[0]:.2f}, {conf_int[1]:.2f}]")
        update_metric("metricRDSE", f"{se:.2f}")
        update_metric("metricRDSignificanceText", sig_text)
        update_metric("metricRDDirectionText", dir_text)
    except Exception as e:
        print(f"    [!] Macro Telemetry Export Failed: {e}")
    
    # 2. Bandwidth Sensitivity
    out_lines.append("2. BANDWIDTH SENSITIVITY")
    out_lines.append(f"{'Bandwidth':<15} | {'Estimate':<10} | {'Std. Err':<10} | {'P-Value':<10}")
    out_lines.append("-" * 55)
    for bw in [0.05, 0.10, 0.15, 0.20]:
        res_bw = run_rd(df, bw=bw)
        if res_bw:
            try:
                est = res_bw.params['post_threshold']
                se = res_bw.bse['post_threshold']
                p = res_bw.pvalues['post_threshold']
                out_lines.append(f"{bw:<15.2f} | {est:<10.3f} | {se:<10.3f} | {p:<10.3f}")
            except Exception:
                out_lines.append(f"{bw:<15.2f} | Error computing constraint")
    out_lines.append("")
    
    # 3. Covariate Continuity Tests
    out_lines.append("3. COVARIATE CONTINUITY TESTS (At threshold = 0.20, Bandwidth = 0.10)")
    out_lines.append(f"{'Covariate':<20} | {'Post- Pre- Diff':<15} | {'P-Value':<10}")
    out_lines.append("-" * 55)
    bw = 0.10
    df_bw = df[(df['signed_area_share'] - 0.20).abs() <= bw]
    post = df_bw[df_bw['signed_area_share'] >= 0.20]
    pre = df_bw[df_bw['signed_area_share'] < 0.20]
    for cov in ['cov_parcel_size', 'cov_base_zoning']:
        from scipy.stats import ttest_ind
        stat, pval = ttest_ind(post[cov], pre[cov], equal_var=False)
        diff = post[cov].mean() - pre[cov].mean()
        out_lines.append(f"{cov:<20} | {diff:<15.3f} | {pval:<10.3f}")
    out_lines.append("")

    # 4. Density Test (McCrary proxy around threshold)
    out_lines.append("4. DENSITY TEST (Check for manipulation around the cutoff)")
    n_pre = len(df[(df['signed_area_share'] >= 0.15) & (df['signed_area_share'] < 0.20)])
    n_post = len(df[(df['signed_area_share'] >= 0.20) & (df['signed_area_share'] <= 0.25)])
    out_lines.append(f"Observations [0.15, 0.20): {n_pre}")
    out_lines.append(f"Observations [0.20, 0.25]: {n_post}")
    # Simple binomial test proxy
    try:
        from scipy.stats import binomtest
        p_binom = binomtest(n_post, n_pre + n_post, p=0.5).pvalue
    except (ImportError, ValueError):
        try:
            from scipy.stats import binom_test
            p_binom = binom_test(n_post, n_pre + n_post, p=0.5)
        except ValueError:
            p_binom = 1.0 # Defaults if 0 occurrences 
    out_lines.append(f"Binomial Test P-Value (H0: smooth density): {p_binom:.3f}")
    out_lines.append("")

    # 5. Placebo Cutoffs
    out_lines.append("5. PLACEBO CUTOFF TESTS (Bandwidth = 0.10)")
    out_lines.append(f"{'Placebo Cutoff':<15} | {'Estimate':<10} | {'Std. Err':<10} | {'P-Value':<10}")
    out_lines.append("-" * 55)
    for cut in [0.10, 0.15, 0.25, 0.30]:
        res_placebo = run_rd(df, bw=0.10, threshold=cut)
        if res_placebo:
            try:
                est = res_placebo.params['post_threshold']
                se = res_placebo.bse['post_threshold']
                p = res_placebo.pvalues['post_threshold']
                out_lines.append(f"{cut:<15.2f} | {est:<10.3f} | {se:<10.3f} | {p:<10.3f}")
            except KeyError:
                out_lines.append(f"{cut:<15.2f} | Insufficient variation")
    out_lines.append("")

    # 6. Donut RD
    out_lines.append("6. DONUT RD (Excluding observations +/- 0.01 around the cutoff)")
    res_donut = run_rd(df, bw=0.10, exclude_donut=0.01)
    if res_donut:
        try:
            est = res_donut.params['post_threshold']
            se = res_donut.bse['post_threshold']
            p = res_donut.pvalues['post_threshold']
            out_lines.append(f"Estimate: {est:.3f} (SE: {se:.3f}, p={p:.3f})")
        except:
             out_lines.append("Donut calculation omitted due to sparse local vectors.")
    else:
        out_lines.append("Insufficient data for Donut RD.")
    
    with open(os.path.join(OUT_DIR, "rd_results.txt"), "w") as f:
        f.write("\n".join(out_lines))
        
    print(res_main.summary().tables[1])
    print("[+] Track 2 Complete. Robust Diagnostics appended. Output saved to Track2_Causal/rd_results.txt")

if __name__ == '__main__':
    run_track2()
