import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA_PATH = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
OUT_PATH = os.path.join(ROOT, "Analysis", "Output", "Track1_Predictive", "Figures")
os.makedirs(OUT_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH, on_bad_lines='skip', low_memory=False)

target_col = next((c for c in ['is_protested', 'organized_opposition', 'opposition'] if c in df.columns), None)
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['gross_site_area_acres'] = pd.to_numeric(df['gross_site_area_acres'], errors='coerce').fillna(0)
df['delta_max_far'] = pd.to_numeric(df['delta_max_far'], errors='coerce').fillna(0)

# Filter bounds
df = df[(df['year'] >= 2007) & (df['year'] <= 2024)]

def derive_6_tier(row):
    far = row['delta_max_far']
    acres = row['gross_site_area_acres']
    if acres > 3 and far > 1.5: return "PUD"
    if far > 1.0: return "Mixed-Use"
    if far > 0.5: return "Multifamily"
    if far > 0.1 and acres < 1.0: return "Missing-Middle"
    if far > 0: return "Rezoning"
    return "By-Right Infill"

df['Typology'] = df.apply(derive_6_tier, axis=1)

# Group by Year and Typology
grouped = df.groupby(['year', 'Typology'])[target_col].agg(['mean', 'count']).reset_index()

# Filter noise (points with < 10 cases in a given year are hidden to prevent 100% spikes from n=1)
grouped.loc[grouped['count'] < 5, 'mean'] = np.nan

# Okabe-Ito friendly
palette = {
    "PUD": "#E69F00",
    "Mixed-Use": "#56B4E9",
    "Multifamily": "#009E73",
    "Missing-Middle": "#F0E442",
    "Rezoning": "#0072B2",
    "By-Right Infill": "#D55E00"
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Top Panel: Rate
sns.lineplot(data=grouped, x='year', y='mean', hue='Typology', palette=palette, marker='o', linewidth=2.5, ax=ax1)
ax1.set_title("Valid Protest Petition Rate by Project Type (2007-2024)", fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel("Observed Petition Incidence Rate", fontsize=12)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(title="Zoning Typology", bbox_to_anchor=(1.05, 1), loc='upper left')

# Bottom Panel: Volume
pivot_counts = grouped.pivot(index='year', columns='Typology', values='count').fillna(0)
cols = [c for c in palette.keys() if c in pivot_counts.columns]
pivot_counts = pivot_counts[cols]

bottom = np.zeros(len(pivot_counts))
for col in cols:
    ax2.bar(pivot_counts.index, pivot_counts[col], bottom=bottom, color=palette[col], label=col, width=0.85)
    bottom += pivot_counts[col]

ax2.set_ylabel("Total Cases (Volume)", fontsize=12)
ax2.set_xlabel("Application Year", fontsize=12)
ax2.set_xticks(np.arange(2007, 2025, 2))
ax2.grid(True, linestyle='--', alpha=0.6, axis='y')

plt.tight_layout()

save_path = os.path.join(OUT_PATH, 'Typology_Temporal_Incidence.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Track1_Predictive", "Figures")
os.makedirs(out_dir, exist_ok=True)
repo_path = os.path.join(out_dir, "Typology_Temporal_Incidence.png")
plt.savefig(repo_path, dpi=300, bbox_inches='tight')
print(f"Saved temporal incidence to {save_path} & {repo_path}")
