import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA_PATH = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
OUT_PATH = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter5")
os.makedirs(OUT_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH, on_bad_lines='skip', low_memory=False)

# Clean dates and targets
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

target_col = next((c for c in ['is_protested', 'organized_opposition', 'opposition'] if c in df.columns), None)
df['Petition_Hazard'] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)

if 'ldb_council_district' in df.columns:
    df['District'] = pd.to_numeric(df['ldb_council_district'], errors='coerce')
elif 'council_district_x' in df.columns:
    df['District'] = pd.to_numeric(df['council_district_x'], errors='coerce')
elif 'council_district_y' in df.columns:
    df['District'] = pd.to_numeric(df['council_district_y'], errors='coerce')
elif 'council_district' in df.columns:
    df['District'] = pd.to_numeric(df['council_district'], errors='coerce')
else:
    df['District'] = 1

# Target districts that ACTUALLY flipped ideology in 2022
treated_districts = [4, 9]
results = []
elections = [2014, 2016, 2018, 2020, 2022]

for elec_year in elections:
    slice_df = df[(df['year'] >= elec_year - 1) & (df['year'] <= elec_year + 1)].copy()
    
    if len(slice_df) < 50:
        continue
        
    slice_df['Post_Election'] = (slice_df['year'] >= elec_year + 1).astype(int)
    slice_df['Treated_District'] = slice_df['District'].isin(treated_districts).astype(int)
    
    formula = "Petition_Hazard ~ Treated_District + Post_Election + Treated_District:Post_Election"
    
    try:
        model = smf.ols(formula, data=slice_df).fit()
        coef = model.params['Treated_District:Post_Election']
        se = model.bse['Treated_District:Post_Election']
        p_val = model.pvalues['Treated_District:Post_Election']
        
        results.append({
            'Election': elec_year,
            'DiD_Coefficient': coef,
            'SE': se,
            'p_value': p_val
        })
    except Exception as e:
        pass

res_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))

for idx, row in res_df.iterrows():
    color = 'darkred' if row['p_value'] <= 0.1 else 'gray'
    plt.errorbar(row['Election'], row['DiD_Coefficient'], yerr=1.96*row['SE'], 
                 fmt='o', color=color, ecolor=color, capsize=6, elinewidth=2, 
                 markeredgewidth=2, markersize=8, mfc='white')

plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.title('Placebo Falsification: Petition Filing Rate by Election Cycle', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('DiD Coefficient ($\\Delta$ Petition Filing Rate)', fontsize=12)
plt.xlabel('Municipal Electoral Cycle', fontsize=12)
plt.xticks(elections)

plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

out_file = os.path.join(OUT_PATH, "Electoral_Placebo_DiD.png")
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {out_file}")
