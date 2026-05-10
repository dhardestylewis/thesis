import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

plt.style.use('dark_background')
sns.set_palette("husl")
sns.set_context("talk", rc={"axes.grid": True, "grid.color": "#333", "grid.linestyle": "--"})

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
DRAFT_DIR = os.path.join(ROOT, 'Thesis_Draft')
ARTIFACT_DIR = r'C:\Users\dhl\.gemini\antigravity\brain\13179483-b897-4efa-b65b-6259cbe2174e'

def map_zoning_density(zone_str):
    if pd.isna(zone_str): return 0
    z = str(zone_str).upper()
    if 'CBD' in z: return 10
    if 'MF-6' in z: return 9
    if 'MF-5' in z: return 8
    if 'MF-4' in z or 'MF-3' in z: return 7
    if 'MF' in z: return 6
    if 'CS' in z or 'GR' in z or 'CH' in z: return 5
    if 'SF-6' in z or 'SF-5' in z: return 4
    if 'SF-4' in z: return 3
    if 'SF-3' in z: return 2
    if 'SF-2' in z or 'SF-1' in z: return 1
    if 'RR' in z or 'DR' in z: return 0.5
    return 0

print("[*] Loading production V2 dataset for Attribution...")
df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched_v2.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested']).sort_values('year')

if 'zoning_code' in df.columns:
    df['zoning_density_score'] = df['zoning_code'].apply(map_zoning_density)

if 'latitude' in df.columns and 'longitude' in df.columns:
    df['latitude'] = np.round(df['latitude'], 2)
    df['longitude'] = np.round(df['longitude'], 2)

for col in ['gross_site_area_acres', 'improvement_sq_ft', 'total_market_value', 'appraised_value']:
    if col in df.columns:
        try:
            df[col] = pd.qcut(df[col].replace(0, np.nan), q=10, labels=False, duplicates='drop').fillna(0)
        except Exception:
            pass

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
X_raw = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_raw)
y = df['is_protested'].values

print(f"[*] Training core CatBoost model on full V2 dataset ({X_sc.shape[1]} features)...")
model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
model.fit(X_sc, y)

print("[*] Extracting raw specific attributions...")
raw_fi = model.get_feature_importance(type='LossFunctionChange', data=Pool(X_sc, label=y))
fi_df = pd.DataFrame({'Feature': X_raw.columns, 'Importance': raw_fi}).sort_values('Importance', ascending=False)

# -------------------------------------------------------------
# 1. Post-Clustered (Raw Specific Variables) Plot
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8), facecolor="#121212")
ax.set_facecolor("#121212")
sns.barplot(data=fi_df.head(20), x='Importance', y='Feature', palette='Blues_r', ax=ax)
ax.set_title("Top 20 Post-Clustered Spatial & Appraised Variables (V2 Matrix)", color='white', pad=20)
ax.set_xlabel("Loss Function Change (Importance)", color='white')
ax.set_ylabel("")
ax.tick_params(colors='lightgray')
plt.tight_layout()
out_post = os.path.join(ARTIFACT_DIR, 'plot_final_postclustered_v2.png')
fig.savefig(out_post, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------------
# 2. Pre-Clustered (Semantic Rollup) Plot
# -------------------------------------------------------------
feature_clusters = {
    'Zoning Density Limits': ['zoning_density_score', 'delta_max_height_ft', 'delta_max_far', 'delta_max_bldg_cov_pct', 'delta_min_lot_sqft'],
    'Parcel Dimensions': ['gross_site_area_acres', 'land_acres', 'agg_land_acres_total', 'latitude', 'longitude'],
    'Appraisal / Valuation': ['total_market_value', 'appraised_value', 'agg_appraised_value_median', 'ldb_appraised_val'],
    'Structure Scale': ['improvement_sq_ft', 'agg_improvement_sq_ft_median', 'ldb_imprv_sqft'],
    'Structure Age': ['year_built', 'agg_year_built_median', 'ears_year_built'],
    'Local Demographics': ['acs2_total_population', 'acs_total_population'],
    'Local Economic Class': ['acs2_median_household_income', 'acs_median_household_income'],
    'Housing Tenure': ['acs2_renter_occupied_units', 'acs2_owner_occupied_units', 'acs_renter_occupied_units']
}

cluster_scores = defaultdict(float)
for _, row in fi_df.iterrows():
    f_name = row['Feature']
    score = row['Importance']
    matched = False
    for group, feats in feature_clusters.items():
        if any(f_name == f or f_name.startswith(f + '_lag') for f in feats):
            cluster_scores[group] += score
            matched = True
            break
    if not matched:
        cluster_scores['Unclustered Context Lags'] += score

cdf = pd.DataFrame(list(cluster_scores.items()), columns=['Semantic Group', 'Consolidated Importance'])
cdf = cdf.sort_values('Consolidated Importance', ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 8), facecolor="#121212")
ax2.set_facecolor("#121212")
sns.barplot(data=cdf, x='Consolidated Importance', y='Semantic Group', palette='rocket', ax=ax2)
ax2.set_title("Pre-Clustered Semantic Attribution (Macro Dimensions)", color='white', pad=20)
ax2.set_xlabel("Consolidated Loss Function Change", color='white')
ax2.set_ylabel("")
ax2.tick_params(colors='lightgray')
plt.tight_layout()
out_pre = os.path.join(ARTIFACT_DIR, 'plot_final_preclustered_v2.png')
fig2.savefig(out_pre, dpi=300, facecolor=fig2.get_facecolor(), bbox_inches='tight')
plt.close(fig2)

print("\nSaved visuals:")
print(f"  {out_post}")
print(f"  {out_pre}")
