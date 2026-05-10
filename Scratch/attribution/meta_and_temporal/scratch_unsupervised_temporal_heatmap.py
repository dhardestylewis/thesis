import pandas as pd
import numpy as np
import os
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as plt_sns
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')
plt.style.use('dark_background')
plt_sns.set_palette("husl")
plt_sns.set_context("talk")

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
ARTIFACT_DIR = r'C:\Users\dhl\.gemini\antigravity\brain\13179483-b897-4efa-b65b-6259cbe2174e'

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched_v2_OmniLagged.csv'), low_memory=False)

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
cat_cols = [c for c in df.columns if c.startswith('raw_')]

# Isolate all mathematical vectors
X_raw = df.drop(columns=[c for c in (drop_cols + future_features + cat_cols) if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
X_raw = X_raw.loc[:, X_raw.var() > 0]
y = df['is_protested'].values
feature_names = np.array(X_raw.columns)

print(f"[*] Beginning Unsupervised Math Constraints on {len(feature_names)} blind metrics.")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Automatically determine K via PCA
pca = PCA(n_components=0.95, random_state=42)
pca.fit(X_scaled)
optimal_k = pca.n_components_
print(f"[*] Auto-K = {optimal_k} mathematical clusters.")

agg = FeatureAgglomeration(n_clusters=optimal_k, metric='euclidean', linkage='ward')
X_clustered_raw = agg.fit_transform(X_scaled)

def get_plain_english_name(top_variables_str):
    vt = top_variables_str.lower()
    if 'income' in vt or 'poverty' in vt or 'rent' in vt or 'owner' in vt or 'race' in vt or 'population' in vt: return "Demographic & Wealth Geography"
    if 'value' in vt or 'apprais' in vt: return "Property Valuation Trajectories"
    if 'height' in vt or 'far' in vt or 'cov' in vt or 'zoning' in vt: return "Physical Density Constraints"
    if 'sqft' in vt or 'area' in vt or 'acre' in vt or 'lot' in vt: return "Parcel Topology & Geometries"
    if 'age' in vt or 'year_built' in vt or 'built' in vt: return "Structural Age Constraints"
    if 'council' in vt or 'district' in vt: return "Political Boundary Markers"
    if 'lat' in vt or 'lon' in vt: return "Absolute Spatial Coordinates"
    if 'lui' in vt or 'land_use' in vt: return "Municipal Land Use Flags"
    if 'protest' in vt or 'opposition' in vt: return "Aggregated Dissent Contagion"
    return "Secondary Administrative Domains"

cluster_labels = agg.labels_
cluster_names = []
cluster_to_temporal_breakdown = {}

for cluster_id in range(optimal_k):
    member_idx = np.where(cluster_labels == cluster_id)[0]
    member_names = feature_names[member_idx]
    
    variances = np.var(X_scaled[:, member_idx], axis=0)
    top_3_idx = np.argsort(variances)[::-1][:3]
    top_3_names = member_names[top_3_idx]
    
    semantic_base = get_plain_english_name(" ".join(top_3_names))
    c_name = f"{semantic_base} [Cluster {cluster_id+1}]"
    cluster_names.append(c_name)
    
    # Analyze the constituent temporal offsets of this mathematical cluster
    temporal_counts = {
        'Base / Current': 0, 'Lag 1 Yr': 0, 'Lag 2 Yr': 0, 'Lag 3 Yr': 0,
        'Lag 4 Yr': 0, 'Lag 5 Yr': 0, 'Lag 6 Yr': 0
    }
    
    for fname in member_names:
        lag_match = re.search(r'_lag_(\d+)yr', fname)
        if lag_match:
            offset = f"Lag {lag_match.group(1)} Yr"
        else:
            offset = "Base / Current"
            
        if offset in temporal_counts:
            temporal_counts[offset] += 1
            
    # Convert counts to proportions (since FeatureAgglomeration uses mean, variance weights are conceptually flat)
    total_features = sum(temporal_counts.values())
    temporal_props = {k: v/max(1, total_features) for k, v in temporal_counts.items()}
    cluster_to_temporal_breakdown[c_name] = temporal_props

X_clustered = pd.DataFrame(X_clustered_raw, columns=cluster_names)

print("[*] Training Base CatBoost Engine on exact orthogonal clusters...")
model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
model.fit(X_clustered, y)

print("[*] Extracting Multi-Collinearity Safe TreeSHAP Matrices...")
shap_values = model.get_feature_importance(type='ShapValues', data=Pool(X_clustered, label=y))
global_shap_importance = np.mean(np.abs(shap_values[:, :-1]), axis=0)

shap_df = pd.DataFrame({'Mathematical Domain Vector': cluster_names, 'Mean Absolute SHAP Score': global_shap_importance})
shap_df = shap_df.sort_values('Mean Absolute SHAP Score', ascending=False)
top_15_clusters = shap_df.head(15)['Mathematical Domain Vector'].tolist()

# Construct output pivot table mapping Domain Vector -> Temporal Offset Attribution
pivot_rows = []
for idx, row in shap_df.head(15).iterrows():
    c_name = row['Mathematical Domain Vector']
    total_shap = row['Mean Absolute SHAP Score']
    props = cluster_to_temporal_breakdown[c_name]
    
    for offset, prop in props.items():
        # Distribute the un-diluted SHAP total computationally across the historical arrays forming the domain linearly
        pivot_rows.append({'Mathematical Domain Vector': c_name, 'Temporal_Offset': offset, 'Attribution': total_shap * prop})

pivot_df = pd.DataFrame(pivot_rows)
pivot = pivot_df.pivot_table(index='Mathematical Domain Vector', columns='Temporal_Offset', values='Attribution', aggfunc='sum').fillna(0)

# Sort temporally explicitly
col_order = ['Base / Current', 'Lag 1 Yr', 'Lag 2 Yr', 'Lag 3 Yr', 'Lag 4 Yr', 'Lag 5 Yr', 'Lag 6 Yr']
valid_cols = [c for c in col_order if c in pivot.columns]
pivot = pivot[valid_cols]

# Match the sort order of the highest-value absolute SHAP clusters natively
pivot = pivot.loc[top_15_clusters]

# Row Normalize identically to the previous Meta-Attribution Map
pivot_normalized = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)

fig, ax = plt.subplots(figsize=(14, 12), facecolor="#121212")
fig.patch.set_facecolor('#121212')

plt_sns.heatmap(pivot_normalized, cmap='mako', annot=False, fmt=".2f", linewidths=.5, ax=ax, cbar_kws={'label': 'Row-Normalized Distributed SHAP Energy (0 to 1)'})
ax.set_title("Unsupervised Mathematical Domain: Temporal Inertia Heatmap", color='white', size=16, pad=20)
ax.set_ylabel("Geospatial & Semantic Raw Architectures", color='white', size=12)
ax.set_xlabel("Temporal Offsets", color='white', size=12)
ax.tick_params(colors='lightgray')

for spine in ax.spines.values():
    spine.set_visible(False)
    
plt.tight_layout()
out_path = os.path.join(ARTIFACT_DIR, 'plot_final_unsupervised_temporal_heatmap.png')
fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')

print(f"[*] Native Temporal Extrapolation extracted.")
print(f"    Graph exported rigidly to: {out_path}")
