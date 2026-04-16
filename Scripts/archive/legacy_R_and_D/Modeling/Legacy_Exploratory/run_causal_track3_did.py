import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing.csv")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Track3_Causal")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_illustrative_data(df):
    """Generate mock data for HOME Phase 1 and 2 to demonstrate DiD event study."""
    np.random.seed(123)
    
    # We will simulate data across multiple quarters leading up to and after policy implementation.
    # We use tracking quarters instead of years for finer granularity.
    df = df.copy()
    
    # Let's say we have cases from 2018 to 2025.
    if 'year' not in df.columns:
        df['year'] = np.random.randint(2018, 2026, len(df))
    
    # Assign quarters
    df['quarter'] = np.random.randint(1, 5, len(df))
    df['time_t'] = df['year'] + (df['quarter'] - 1) / 4.0
    
    # HOME Phase 1 implemented Q1 2024
    home1_time = 2024.0
    
    # HOME Phase 2 implemented Q3 2024
    home2_time = 2024.5
    
    # Treatment assignments
    df['eligible_home_1'] = np.random.choice([0, 1], p=[0.5, 0.5], size=len(df))
    df['eligible_home_2'] = np.where(df['eligible_home_1'] == 1, 
                                     np.random.choice([0, 1], p=[0.7, 0.3], size=len(df)), 0)
    
    # Event time relative to implementation (in quarters)
    df['rel_time_home1'] = np.floor((df['time_t'] - home1_time) * 4)
    df['rel_time_home2'] = np.floor((df['time_t'] - home2_time) * 4)
    
    # Add dummies for event study leads and lags (-4 to +4 quarters)
    for k in range(-4, 5):
        if k != -1: # exclude T-1 baseline
            prefix = "m" if k < 0 else "p"
            num = abs(k)
            df[f'leadlag_h1_{prefix}{num}'] = (df['rel_time_home1'] == k).astype(int) * df['eligible_home_1']
            
    # Mock Outcome: council_dissent
    # Pre-trends are flat (0). Post-trends rise gradually for phase 1.
    post_effect = np.where(df['rel_time_home1'] >= 0, 0.5 * df['rel_time_home1'], 0) * df['eligible_home_1']
    df['council_dissent'] = 1.0 + 0.2 * df['eligible_home_1'] + post_effect + np.random.normal(0, 1.5, len(df))
    
    # Post dummies for standard TWFE
    df['post_home1'] = (df['time_t'] >= home1_time).astype(int)
    df['post_home2'] = (df['time_t'] >= home2_time).astype(int)
    
    return df

def run_track3():
    print("Running Track 3: Event Study DiD (ILLUSTRATIVE SYNTHETIC DATA)")
    if not os.path.exists(DATA):
        print("Data not found.")
        return
        
    df_raw = pd.read_csv(DATA)
    df = generate_illustrative_data(df_raw)
    
    out_lines = [
        "TRACK 3 DIFFERENCE-IN-DIFFERENCES DIAGNOSTICS (ILLUSTRATIVE SYNTHETIC DATA)",
        "==========================================================================",
        "Note: Valid case-level policy applicability flags for HOME Phase 1 & 2 are pending.",
        "These results demonstrate the dynamic Callaway-Sant'Anna style event study methodology.",
        ""
    ]
    
    # 1. Static TWFE (Simple DiD) for HOME Phase 1
    out_lines.append("1. STATIC TWFE BASELINE (HOME Phase 1 Average Treatment Effect)")
    df['treat_home1'] = df['eligible_home_1'] * df['post_home1']
    mod_static = smf.ols("council_dissent ~ treat_home1 + eligible_home_1 + post_home1", data=df)
    res_static = mod_static.fit(cov_type='HC1')
    out_lines.append(res_static.summary().tables[1].as_text())
    out_lines.append("")
    
    # 2. Dynamic Event Study for HOME Phase 1
    out_lines.append("2. DYNAMIC EVENT STUDY (HOME Phase 1)")
    # build formula
    leads_lags = []
    for k in range(-4, 5):
        if k != -1:
            prefix = "m" if k < 0 else "p"
            num = abs(k)
            leads_lags.append(f"leadlag_h1_{prefix}{num}")
            
    formula_vars = " + ".join(leads_lags) + " + eligible_home_1 + C(time_t)"
    mod_dyn = smf.ols(f"council_dissent ~ {formula_vars}", data=df)
    res_dyn = mod_dyn.fit(cov_type='HC1')
    
    out_lines.append(f"{'Relative Quarter':<20} | {'Estimate':<10} | {'Std. Err':<10} | {'P-Value':<10}")
    out_lines.append("-" * 55)
    
    # Build array for pre-trend test
    pre_trend_vars = [f"leadlag_h1_m{abs(k)}" for k in range(-4, 0) if k != -1]
    
    for k in range(-4, 5):
        if k == -1:
            out_lines.append(f"T-1 (Baseline)       | {'0.000':<10} | {'0.000':<10} | {'---':<10}")
            continue
        prefix = "m" if k < 0 else "p"
        num = abs(k)
        var = f"leadlag_h1_{prefix}{num}"
        est = res_dyn.params[var]
        se = res_dyn.bse[var]
        p = res_dyn.pvalues[var]
        # Labeling format
        label = f"T{k}" if k < 0 else f"T+{k}"
        out_lines.append(f"{label:<20} | {est:<10.3f} | {se:<10.3f} | {p:<10.3f}")
        
    out_lines.append("")
    
    # 3. Pre-trend joint F-test
    out_lines.append("3. IDENTIFICATION: PRE-TREND JOINT F-TEST")
    test_str = " = ".join(pre_trend_vars) + " = 0"
    try:
        f_test = res_dyn.f_test(test_str)
        out_lines.append(f"Joint F-test p-value for pre-treatment periods (T-4, T-3, T-2) = 0: {f_test.pvalue:.3f}")
        if f_test.pvalue > 0.05:
            out_lines.append("Conclusion: Fail to reject null hypothesis of parallel pre-trends (Assumption holds).")
        else:
            out_lines.append("Conclusion: Reject null hypothesis of parallel pre-trends (Assumption violated).")
    except Exception as e:
        out_lines.append(f"Could not compute F-test: {e}")
        
    out_lines.append("\n4. HOME PHASE 2 DEGENERACY CHECK")
    out_lines.append("Phase 2 enacted late 2024. Checking post-treatment observation counts.")
    post_phase2 = len(df[(df['eligible_home_2'] == 1) & (df['post_home2'] == 1)])
    out_lines.append(f"Available Phase 2 Post-Treatment Observations: {post_phase2}")
    if post_phase2 < 50:
        out_lines.append("Conclusion: Insufficient statistical power to estimate Phase 2 effect (Degenerate).")
        out_lines.append("Analysis constrained to HOME Phase 1 until further data collection.")
        
    with open(os.path.join(OUT_DIR, "did_diagnostics.txt"), "w") as f:
        f.write("\n".join(out_lines))
        
    print("\n".join(out_lines))
    print("Track 3 Complete. Output saved to did_diagnostics.txt")

if __name__ == '__main__':
    run_track3()
