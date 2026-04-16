import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing.csv")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Track2_Causal")
os.makedirs(OUT_DIR, exist_ok=True)

def run_rd(df, bw, threshold=0.20, exclude_donut=None):
    """Run local linear regression RD with triangular kernel."""
    df_temp = df.copy()
    if exclude_donut:
        # Exclude observations within [threshold - donut, threshold + donut]
        df_temp = df_temp[~df_temp['signed_area_share'].between(threshold - exclude_donut, threshold + exclude_donut)]
        
    df_temp['running_var'] = df_temp['signed_area_share'] - threshold
    df_temp['post_threshold'] = (df_temp['running_var'] > 0).astype(int)
    
    # Restrict to bandwidth
    df_bw = df_temp[df_temp['running_var'].abs() <= bw].copy()
    
    # Triangular weight
    df_bw['weight'] = 1 - (df_bw['running_var'].abs() / bw)
    
    if len(df_bw) < 10:
        return None
        
    model = smf.wls("days_delayed ~ running_var * post_threshold", data=df_bw, weights=df_bw['weight'])
    try:
        res = model.fit()
        return res
    except:
        return None

def run_track2():
    print("Running Track 2: Regression Discontinuity (ILLUSTRATIVE SYNTHETIC DATA)")
    if not os.path.exists(DATA):
        print("Data not found.")
        return
        
    df = pd.read_csv(DATA)
    
    # Mock continuous signed-area share around the 20% (0.20) threshold based on protest
    np.random.seed(42)
    df['signed_area_share'] = np.where(df['is_protested'] == 1, 
                                       np.random.uniform(0.15, 0.45, len(df)), 
                                       np.random.uniform(0.0, 0.199, len(df)))
    
    # Mock outcome: Days delayed (Null effect around the threshold, base effect from protest flag)
    # We add a slight endogenous correlation up to the threshold, but no jump at 0.20.
    df['days_delayed'] = 30 + 50 * df['is_protested'] + 10 * df['signed_area_share'] + np.random.normal(0, 10, len(df))
    
    # Mock covariates
    df['cov_parcel_size'] = np.random.lognormal(mean=0, sigma=1, size=len(df))
    df['cov_base_zoning'] = np.random.randint(1, 5, size=len(df))
    
    out_lines = [
        "TRACK 2 REGRESSION DISCONTINUITY DIAGNOSTICS (ILLUSTRATIVE SYNTHETIC DATA)",
        "==========================================================================",
        "Note: Real 'signed_area_share' is not currently in the warehouse.",
        "These results demonstrate the proposed methodology and diagnostics battery.\n"
    ]
    
    # 1. Main RD Model (Bandwidth 0.10)
    res_main = run_rd(df, bw=0.10)
    out_lines.append("1. MAIN LOCAL LINEAR RD ESTIMATE (Bandwidth = 0.10, Triangular Kernel)")
    out_lines.append(res_main.summary().tables[1].as_text())
    out_lines.append("")
    
    # 2. Bandwidth Sensitivity
    out_lines.append("2. BANDWIDTH SENSITIVITY")
    out_lines.append(f"{'Bandwidth':<15} | {'Estimate':<10} | {'Std. Err':<10} | {'P-Value':<10}")
    out_lines.append("-" * 55)
    for bw in [0.05, 0.10, 0.15, 0.20]:
        res_bw = run_rd(df, bw=bw)
        if res_bw:
            est = res_bw.params['post_threshold']
            se = res_bw.bse['post_threshold']
            p = res_bw.pvalues['post_threshold']
            out_lines.append(f"{bw:<15.2f} | {est:<10.3f} | {se:<10.3f} | {p:<10.3f}")
    out_lines.append("")
    
    # 3. Covariate Continuity Tests
    out_lines.append("3. COVARIATE CONTINUITY TESTS (At threshold = 0.20, Bandwidth = 0.10)")
    out_lines.append(f"{'Covariate':<20} | {'Post- Pre- Diff':<15} | {'P-Value':<10}")
    out_lines.append("-" * 55)
    bw = 0.10
    df_bw = df[(df['signed_area_share'] - 0.20).abs() <= bw]
    post = df_bw[df_bw['signed_area_share'] > 0.20]
    pre = df_bw[df_bw['signed_area_share'] <= 0.20]
    for cov in ['cov_parcel_size', 'cov_base_zoning']:
        from scipy.stats import ttest_ind
        stat, pval = ttest_ind(post[cov], pre[cov], equal_var=False)
        diff = post[cov].mean() - pre[cov].mean()
        out_lines.append(f"{cov:<20} | {diff:<15.3f} | {pval:<10.3f}")
    out_lines.append("")

    # 4. Density Test (McCrary proxy around threshold)
    out_lines.append("4. DENSITY TEST (Check for manipulation around the cutoff)")
    n_pre = len(df[(df['signed_area_share'] >= 0.15) & (df['signed_area_share'] < 0.20)])
    n_post = len(df[(df['signed_area_share'] > 0.20) & (df['signed_area_share'] <= 0.25)])
    out_lines.append(f"Observations [0.15, 0.20): {n_pre}")
    out_lines.append(f"Observations (0.20, 0.25]: {n_post}")
    # Simple binomial test proxy
    try:
        from scipy.stats import binomtest
        p_binom = binomtest(n_post, n_pre + n_post, p=0.5).pvalue
    except ImportError:
        from scipy.stats import binom_test
        p_binom = binom_test(n_post, n_pre + n_post, p=0.5)
    out_lines.append(f"Binomial Test P-Value: {p_binom:.3f}")
    out_lines.append("")

    # 5. Placebo Cutoffs
    out_lines.append("5. PLACEBO CUTOFF TESTS (Bandwidth = 0.10)")
    out_lines.append(f"{'Placebo Cutoff':<15} | {'Estimate':<10} | {'Std. Err':<10} | {'P-Value':<10}")
    out_lines.append("-" * 55)
    for cut in [0.10, 0.15, 0.25, 0.30]:
        res_placebo = run_rd(df, bw=0.10, threshold=cut)
        if res_placebo:
            est = res_placebo.params['post_threshold']
            se = res_placebo.bse['post_threshold']
            p = res_placebo.pvalues['post_threshold']
            out_lines.append(f"{cut:<15.2f} | {est:<10.3f} | {se:<10.3f} | {p:<10.3f}")
    out_lines.append("")

    # 6. Donut RD
    out_lines.append("6. DONUT RD (Excluding observations +/- 0.01 around the cutoff)")
    res_donut = run_rd(df, bw=0.10, exclude_donut=0.01)
    if res_donut:
        est = res_donut.params['post_threshold']
        se = res_donut.bse['post_threshold']
        p = res_donut.pvalues['post_threshold']
        out_lines.append(f"Estimate: {est:.3f} (SE: {se:.3f}, p={p:.3f})")
    else:
        out_lines.append("Insufficient data for Donut RD.")
    
    with open(os.path.join(OUT_DIR, "rd_diagnostics.txt"), "w") as f:
        f.write("\n".join(out_lines))
        
    print("\n".join(out_lines))
    print("Track 2 Complete. Output saved to rd_diagnostics.txt")

if __name__ == '__main__':
    # Add binom_test compatibility for newer scipy versions if needed
    try:
        from scipy.stats import binom_test
    except ImportError:
        def binom_test(x, n, p):
            import scipy.stats as st
            return st.binomtest(x, n, p).pvalue
            
    run_track2()
