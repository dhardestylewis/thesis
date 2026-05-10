import pandas as pd
import numpy as np
import os
import re
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as plt_sns
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')
plt.style.use('dark_background')
plt_sns.set_palette("husl")
plt_sns.set_context("talk")

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
ARTIFACT_DIR = r'C:\Users\dhl\.gemini\antigravity\brain\13179483-b897-4efa-b65b-6259cbe2174e'

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched_v2_OmniLagged.csv'), low_memory=False)

def map_zoning_density(z):
    if pd.isna(z): return 0
    if 'CBD' in str(z).upper(): return 10
    return 0

if 'zoning_code' in df.columns:
    df['zoning_density_score'] = df['zoning_code'].apply(map_zoning_density)

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
cat_cols = [c for c in df.columns if c.startswith('raw_')]

X_raw = df.drop(columns=[c for c in (drop_cols + future_features + cat_cols) if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
y = df['is_protested'].values

raw_columns = list(X_raw.columns)
base_to_cols = defaultdict(list)

# Group variables logically
for col in raw_columns:
    lag_match = re.search(r'_lag_(\d+)yr', col)
    if lag_match:
        base = col.replace(f'_lag_{lag_match.group(1)}yr', '').replace('district_', '')
        base_to_cols[base].append(col)
    else:
        base = col.replace('district_', '')
        base_to_cols[base].append(col)

print(f"[*] Found {len(base_to_cols)} core base features spanning {len(raw_columns)} lag permutations.")

# NATIVELY CLUSTER TEMPORAL DIMENSIONS VIA MEAN (Moving Average Consolidation)
X_clustered = pd.DataFrame()
for base_feat, cols in base_to_cols.items():
    X_clustered[base_feat] = X_raw[cols].mean(axis=1)

print(f"[*] Raw Features successfully consolidated into {X_clustered.shape[1]} Native Temporal Clusters via Mean Aggregation.")

print("[*] Training Base CatBoost Learner natively on Clustered Arrays...")
model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
model.fit(X_clustered, y)

print("[*] Extracting Pre-Clustered SHAP Attributions (TreeSHAP)...")
shap_values = model.get_feature_importance(type='ShapValues', data=Pool(X_clustered, label=y))
global_shap_importance = np.mean(np.abs(shap_values[:, :-1]), axis=0)

shap_df = pd.DataFrame({'Pre-Clustered Feature Block': X_clustered.columns, 'Mean Absolute SHAP Score': global_shap_importance})
shap_df = shap_df.sort_values('Mean Absolute SHAP Score', ascending=False)

top_shap = shap_df.head(20)

fig, ax = plt.subplots(figsize=(12, 10), facecolor="#121212")
fig.patch.set_facecolor('#121212')

plt_sns.barplot(data=top_shap, x='Mean Absolute SHAP Score', y='Pre-Clustered Feature Block', palette='mako', ax=ax)
ax.set_title("Pre-Clustered SHAP Attribution (Native Dimension Means)", color='white', size=16, pad=20)
ax.set_xlabel("Mean Absolute TreeSHAP Offset", color='white', size=12)
ax.set_ylabel("", color='white')
ax.tick_params(colors='lightgray')

for spine in ax.spines.values():
    spine.set_visible(False)
    
plt.tight_layout()
out_path = os.path.join(ARTIFACT_DIR, 'plot_final_native_clustered_shap.png')
fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')

print(f"\n[*] Flawless Native Pre-Clustered SHAP successfully exacted.")
print(f"    Graph exported to: {out_path}")
