import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as plt_sns
from catboost import CatBoostClassifier, Pool

plt.style.use('dark_background')
plt_sns.set_palette("husl")
plt_sns.set_context("talk")

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
ARTIFACT_DIR = r'C:\Users\dhl\.gemini\antigravity\brain\13179483-b897-4efa-b65b-6259cbe2174e'

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched_v2_OmniLagged.csv'), low_memory=False)

def map_zoning_density(zone_str):
    if pd.isna(zone_str): return 0
    if 'CBD' in str(zone_str).upper(): return 10
    return 0
if 'zoning_code' in df.columns:
    df['zoning_density_score'] = df['zoning_code'].apply(map_zoning_density)

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
cat_cols = [c for c in df.columns if c.startswith('raw_')]

# Safely extract numeric features
X_raw = df.drop(columns=[c for c in (drop_cols + future_features + cat_cols) if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
y = df['is_protested'].values

print(f"[*] Training Core Attribution Engine on {X_raw.shape[1]} Omni-Lag features...")
model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
model.fit(X_raw, y)

print("[*] Extracting Exact TreeSHAP Marginal Values (this may take a moment)...")
shap_values = model.get_feature_importance(type='ShapValues', data=Pool(X_raw, label=y))
# SHAP array is (samples, features + 1 base_value). We take the mean absolute SHAP across all cases for global importance.
global_shap_importance = np.mean(np.abs(shap_values[:, :-1]), axis=0)

fi_df = pd.DataFrame({'Feature': X_raw.columns, 'Importance': global_shap_importance})

parsed_rows = []
for _, row in fi_df.iterrows():
    f = row['Feature']
    imp = row['Importance']
    
    # Extract temporal offset logic
    lag_match = re.search(r'lag_(\d+)yr', f)
    if lag_match:
        offset = f"Lag {lag_match.group(1)} Yr"
        base_name = f.replace(f'_lag_{lag_match.group(1)}yr', '').replace('district_', '')
    else:
        offset = "Base / Current"
        base_name = f
        
    # Clean up names for visualization
    base_name = base_name.replace('agg_', '').replace('ears_', '').replace('_median', '').replace('acs2_', '')
    parsed_rows.append({'Base_Feature': base_name, 'Temporal_Offset': offset, 'Importance': imp})

mdf = pd.DataFrame(parsed_rows)

# Group by the exact Base Feature across all its lag iterations to find the Top 25 structural drivers
top_structural = mdf.groupby('Base_Feature')['Importance'].sum().sort_values(ascending=False).head(25).index

filtered = mdf[mdf['Base_Feature'].isin(top_structural)]
pivot = filtered.pivot_table(index='Base_Feature', columns='Temporal_Offset', values='Importance', aggfunc='sum').fillna(0)

# Sort temporally explicitly
col_order = ['Base / Current', 'Lag 1 Yr', 'Lag 2 Yr', 'Lag 3 Yr', 'Lag 4 Yr', 'Lag 5 Yr', 'Lag 6 Yr']
valid_cols = [c for c in col_order if c in pivot.columns]
pivot = pivot[valid_cols]

# Sort rows to put heaviest absolute features visually on top, then normalize by row to expose the temporal distribution
pivot['TOTAL'] = pivot.sum(axis=1)
pivot = pivot.sort_values('TOTAL', ascending=False)
absolute_totals = pivot['TOTAL'].copy()
pivot = pivot.drop(columns=['TOTAL'])

# Row-normalize to expose temporal structural shifts
pivot = pivot.div(pivot.sum(axis=1), axis=0)

filtered.to_csv(os.path.join(DATA, 'Meta_Attribution_PostClustered.csv'), index=False)

fig, ax = plt.subplots(figsize=(14, 12), facecolor="#121212")
fig.patch.set_facecolor('#121212')
plt_sns.heatmap(pivot, cmap='mako', annot=False, fmt=".2f", linewidths=.5, ax=ax, cbar_kws={'label': 'Row-Normalized Relative SHAP Energy (0 to 1)'})
ax.set_title("Post-Clustered Meta-Attribution Map (Relative Temporal Energy)", color='white', size=16, pad=20)
ax.set_ylabel("Geospatial & Semantic Raw Architectures", color='white', size=12)
ax.set_xlabel("Temporal Offsets", color='white', size=12)
ax.tick_params(colors='lightgray')
for spine in ax.spines.values():
    spine.set_visible(False)
    
plt.tight_layout()
out_path = os.path.join(ARTIFACT_DIR, 'plot_final_meta_attribution.png')
fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close(fig)

print(f"\n[*] Success! Meta-Attribution matrix successfully engineered.")
print(f"    Saved raw matrix data to: Meta_Attribution_PostClustered.csv")
print(f"    Saved visual explicitly to: {out_path}")
