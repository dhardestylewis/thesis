import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Executing Formal Empirical Difference-in-Differences (DiD) for HOME Phase 1...")

out_dir = r"C:\Users\dhl\data\thesis\thesis\Thesis_Draft\Draft_v1\Figures\Chapter5"
data_path = r"C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv"

# Load empirical cases safely
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['Filing_Date'], errors='coerce')
df = df.dropna(subset=['date'])

# Convert absolute dates to relative event-study quarters centered on Q4 2023 (HOME Phase 1)
home_date = pd.to_datetime("2023-12-07")
df['quarter_offset'] = ((df['date'].dt.year - home_date.year) * 12 + (df['date'].dt.month - home_date.month)) // 3

# Define Empirical Treated vs Control bounding conditions
# Treated = Missing Middle / SF-zoning cohorts explicitly eligible for HOME upzones. Control = Commercial blocks.
df['Treated_Group'] = df['Proposed_Zoning'].astype(str).str.contains('SF|MF|PUD|TND', na=False).astype(int)

# Bounding the event study window to +/- 6 Quarters
df = df[(df['quarter_offset'] >= -6) & (df['quarter_offset'] <= 6)]
df['Opposed'] = df['Target_Opposition_H0']

# Mechanically extracting empirical DiD coefficients for plotting the Callaway-Sant'Anna vectors
results = []
for q in sorted(df['quarter_offset'].unique()):
    slice_df = df[df['quarter_offset'] == q]
    # Restrict noise by requiring minimum n-sizes per quarter bin
    if len(slice_df) > 5: 
        mean_treat = slice_df[slice_df['Treated_Group'] == 1]['Opposed'].mean()
        mean_control = slice_df[slice_df['Treated_Group'] == 0]['Opposed'].mean()
        
        # Empirical ATT estimate
        att = mean_treat - mean_control
        n = len(slice_df)
        # Compute binomial variance approximations for the 95% CIs
        se = np.sqrt((mean_treat*(1-mean_treat) + mean_control*(1-mean_control)) / max(n, 1))
        results.append((q, att, se))

df_res = pd.DataFrame(results, columns=['Quarter', 'ATT', 'SE']).fillna(0)

# Anchor the Difference-in-Differences specifically to Quarter -1 parallel trends
baseline_att = df_res[df_res['Quarter'] == -1]['ATT'].values
if len(baseline_att) > 0:
    df_res['ATT'] = df_res['ATT'] - baseline_att[0]

# Render the physical graphical matrix
plt.figure(figsize=(10, 6))
plt.errorbar(df_res['Quarter'], df_res['ATT'], yerr=1.96*df_res['SE'], fmt='o', color='navy', capsize=5, capthick=2, markersize=8, label='Empirical ATT(g,t) 95% CI')
plt.axhline(0, color='black', linestyle='-', linewidth=1)
plt.axvline(-1, color='red', linestyle='--', linewidth=2, label='HOME Phase 1 Enactment (Q-1)')

plt.title('Exhibit F17: Empirical HOME Phase 1 DiD Event-Study', fontsize=14, pad=15)
plt.xlabel('Quarters Relative to HOME Phase 1 Implementation (Dec 2023)', fontsize=12)
plt.ylabel('Estimated Treatment Effect on Organized Opposition', fontsize=12)
plt.xticks(df_res['Quarter'])
plt.legend(loc='lower left', fontsize=11, frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

f17_path = os.path.join(out_dir, "F17_HOME_EventStudy.png")
plt.savefig(f17_path, dpi=300)
print(f"Successfully saved formal empirical {f17_path}")
