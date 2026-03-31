import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass


ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA_H0 = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
DATA_PETITION = os.path.join(ROOT, "Data", "Protest_Petitions", "petition_summary_from_pdf.csv")
COA_RAW = os.path.join(ROOT, "Data", "CoA_Open_Data", "Zoning", "ZC_current_edir-dcnf.csv")
OUT_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter5")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_exhibits():
    print("[*] Rendering Authentic F16: RD Scatterplot Execution...")
    
    if not os.path.exists(DATA_H0) or not os.path.exists(DATA_PETITION) or not os.path.exists(COA_RAW):
        print("[-] Required data sources not found.")
        return
        
    df_h0 = pd.read_csv(DATA_H0, low_memory=False)
    df_pet = pd.read_csv(DATA_PETITION, low_memory=False)
    df_coa = pd.read_csv(COA_RAW, low_memory=False)
    
    df = df_h0.merge(df_pet[['case_number', 'signer_pct']], on='case_number', how='left')
    df['signed_area_share'] = df['signer_pct'].fillna(0)
    
    df_coa['start'] = pd.to_datetime(df_coa['APPLICATION_START_DATE'], errors='coerce')
    df_coa['end'] = pd.to_datetime(df_coa['FINAL_DATE'], errors='coerce')
    df_coa['days_delayed_raw'] = (df_coa['end'] - df_coa['start']).dt.days
    
    valid_dates = df_coa.dropna(subset=['CASE_NUMBER', 'days_delayed_raw'])
    valid_dates = valid_dates[valid_dates['days_delayed_raw'] >= 0]
    
    delay_map = valid_dates.set_index('CASE_NUMBER')['days_delayed_raw'].to_dict()
    df['days_delayed'] = df['case_number'].map(delay_map)
    df = df.dropna(subset=['days_delayed'])
    
    # Isolate analysis bound: [0, 0.40] for neighborhood visualization 
    df_plot = df[(df['signed_area_share'] >= 0.0) & (df['signed_area_share'] <= 0.40)].copy()
    
    if len(df_plot) < 20: # Sparse density guard
        print("    [!] F16 Failed. Array lacked sufficient mass within plotting envelope.")
        return
        
    # Create bin-averages
    bins = np.linspace(0, 0.40, 40)
    df_plot['bin'] = pd.cut(df_plot['signed_area_share'], bins=bins)
    df_binned = df_plot.groupby('bin', observed=True)['days_delayed'].mean().reset_index()
    # Handle Interval extract for plotting X axis natively
    df_binned['signed_area_share'] = df_binned['bin'].apply(lambda x: x.mid).astype(float)
    
    # Fit Lowess left and right of the 20% cut line natively
    import statsmodels.api as sm
    left = df_plot[df_plot['signed_area_share'] < 0.20].sort_values('signed_area_share')
    right = df_plot[df_plot['signed_area_share'] >= 0.20].sort_values('signed_area_share')
    
    lowess_left = sm.nonparametric.lowess(left['days_delayed'], left['signed_area_share'], frac=0.4)
    lowess_right = sm.nonparametric.lowess(right['days_delayed'], right['signed_area_share'], frac=0.4)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot true binned averages
    ax.scatter(df_binned['signed_area_share'], df_binned['days_delayed'], 
               color='gray', alpha=0.6, s=50, label='Bin Average (True Observation)')
               
    # Plot Local regressions natively mapped 
    ax.plot(lowess_left[:, 0], lowess_left[:, 1], color='darkblue', linewidth=2.5, label='Local Linear Fit')
    ax.plot(lowess_right[:, 0], lowess_right[:, 1], color='darkred', linewidth=2.5)
    
    ax.axvline(x=0.20, color='black', linestyle='--', linewidth=2, label='Statutory Cutoff (20%)')
    
    ax.set_ylabel('Empirical Days Delayed (Application to Ordinance)')
    ax.set_xlabel('Signed Petition Area Proportion')
    ax.set_title('Regression Discontinuity at the 20% Protest Petition Threshold', fontsize=14, pad=15)
    ax.set_xlim(0, 0.40)
    
    # Dynamic scaling using underlying variance
    y_min, y_max = df_binned['days_delayed'].min() * 0.8, df_binned['days_delayed'].max() * 1.2
    if pd.isna(y_min): y_min = 0
    if pd.isna(y_max) or y_max < 150: y_max = 500
    ax.set_ylim(y_min, y_max)
    
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    out_path = os.path.join(OUT_DIR, "F16_Petition_RD.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"    [+] Successfully produced Authentic F16 via absolute timeline extraction array.")

plot_f16 = generate_exhibits

if __name__ == "__main__":
    generate_exhibits()
