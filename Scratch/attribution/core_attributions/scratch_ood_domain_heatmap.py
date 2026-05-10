import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as plt_sns
from collections import defaultdict
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
plt.style.use('dark_background')
plt_sns.set_palette("husl")
plt_sns.set_context("talk")

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
ARTIFACT_DIR = r'C:\Users\dhl\.gemini\antigravity\brain\13179483-b897-4efa-b65b-6259cbe2174e'

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched_v2_OmniLagged.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested']).sort_values('year')

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
cat_cols = [c for c in df.columns if c.startswith('raw_')]

# Spatial coordinate blocking limits memorization globally natively
if 'latitude' in df.columns and 'longitude' in df.columns:
    df['latitude'] = np.round(df['latitude'], 2)
    df['longitude'] = np.round(df['longitude'], 2)

X_raw = df.drop(columns=[c for c in (drop_cols + future_features + cat_cols) if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
X_raw = X_raw.loc[:, X_raw.var() > 0]
y = df['is_protested'].values
years = df['year'].values
feature_names = np.array(X_raw.columns)

print(f"[*] Extracting Unsupervised Arrays globally statically on {len(feature_names)} blind metrics.")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=0.95, random_state=42)
pca.fit(X_scaled)
optimal_k = pca.n_components_
print(f"[*] PCA Dimensional Bound isolated globally to exactly K={optimal_k} mathematical clusters.")

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

for cluster_id in range(optimal_k):
    member_idx = np.where(cluster_labels == cluster_id)[0]
    member_names = feature_names[member_idx]
    variances = np.var(X_scaled[:, member_idx], axis=0)
    top_3_names = member_names[np.argsort(variances)[::-1][:3]]
    c_name = f"{get_plain_english_name(' '.join(top_3_names))} [C{cluster_id+1}]"
    cluster_names.append(c_name)

X_clustered = pd.DataFrame(X_clustered_raw, columns=cluster_names)

print("[*] Formulating OOD Matrix Walk-Forward anchors dynamically...")
anchors = [2018, 2019, 2020, 2021]
offsets = [1, 2, 3, 4, 5]

# Aggregation dictionaries
domain_offset_shap_weighted_sum = defaultdict(lambda: defaultdict(float))
domain_offset_test_count = defaultdict(lambda: defaultdict(int))
domain_global_importance = defaultdict(float)

for anchor in anchors:
    train_mask = years <= anchor
    if train_mask.sum() == 0: continue
    
    print(f"    Train Boundary [<= {anchor}] deployed across 4 Model Families.")
    
    models_to_test = [
        ('CatBoost', CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)),
        ('LightGBM', LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)),
        ('XGBoost', XGBClassifier(n_estimators=100, max_depth=6, eval_metric='logloss', random_state=42)),
        ('RandomForest', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1))
    ]
    
    for m_name, model in models_to_test:
        X_train_val = X_clustered[train_mask].values
        y_train_val = y[train_mask]
        
        model.fit(X_train_val, y_train_val)
        
        # Configure explainer securely across class
        if m_name == 'CatBoost':
            pass # handled natively
        else:
            explainer = shap.TreeExplainer(model)
        
        for offset in offsets:
            test_year = anchor + offset
            test_mask = years == test_year
            N_test = test_mask.sum()
            
            if N_test < 50:
                continue
                
            # Extract independent Multi-Collinearity SHAP values mapping
            X_test_val = X_clustered[test_mask].values
            if m_name == 'CatBoost':
                shap_matrix = model.get_feature_importance(type='ShapValues', data=Pool(X_test_val, label=y[test_mask]))[:, :-1]
            else:
                shap_matrix = explainer.shap_values(X_test_val)
                if isinstance(shap_matrix, list):
                    shap_matrix = shap_matrix[1] # positive boundary
                elif isinstance(shap_matrix, np.ndarray) and len(shap_matrix.shape) == 3:
                    shap_matrix = shap_matrix[:, :, 1]
                    
            abs_mean_shap_test = np.mean(np.abs(shap_matrix), axis=0)
            
            # Ensure the test arrays mathematically collapsed cleanly into 1D scalar bounds natively
            if hasattr(abs_mean_shap_test, "shape") and len(abs_mean_shap_test.shape) > 1:
                abs_mean_shap_test = np.squeeze(abs_mean_shap_test)
            
            offset_lbl = f"+{offset}yr"
            for idx, d_name in enumerate(cluster_names):
                domain_offset_shap_weighted_sum[d_name][offset_lbl] += (abs_mean_shap_test[idx] * N_test)
                # Count tests cleanly to balance global matrix mapping
                domain_offset_test_count[d_name][offset_lbl] += N_test

# Compute Absolute Expected Exact Math Expectations
pivot_rows = []
for d_name in cluster_names:
    total_shap_allocated = 0
    for offset_lbl in [f"+{o}yr" for o in offsets]:
        t_count = domain_offset_test_count[d_name][offset_lbl]
        if t_count > 0:
            exact_val = domain_offset_shap_weighted_sum[d_name][offset_lbl] / t_count
            pivot_rows.append({'Mathematical Domain Vector': d_name, 'Predictive Test Horizon': offset_lbl, 'Attribution': exact_val})
            total_shap_allocated += exact_val
            
    # Track the global volume of Domain Performance for clipping visually
    domain_global_importance[d_name] = total_shap_allocated
    
pivot_df = pd.DataFrame(pivot_rows)
pivot = pivot_df.pivot_table(index='Mathematical Domain Vector', columns='Predictive Test Horizon', values='Attribution', aggfunc='sum').fillna(0)

# Sort temporally explicitly mapped
col_order = [f"+{o}yr" for o in offsets]
valid_cols = [c for c in col_order if c in pivot.columns]
pivot = pivot[valid_cols]

# Truncate to the strictly resilient mathematical structures mapping perfectly cleanly top 15 ranks organically
top_15_domains = sorted(domain_global_importance, key=domain_global_importance.get, reverse=True)[:15]
pivot = pivot.loc[top_15_domains]

# Row-Normalization isolates the topological decay directly without volume distortion bounds
pivot_normalized = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)

fig, ax = plt.subplots(figsize=(14, 12), facecolor="#121212")
fig.patch.set_facecolor('#121212')

plt_sns.heatmap(pivot_normalized, cmap='mako', annot=False, fmt=".2f", linewidths=.5, ax=ax, cbar_kws={'label': 'Row-Normalized Exact SHAP Predictive Energy (0 to 1)'})
ax.set_title("Unsupervised Mathematical Domains: OOD Horizon Degradation", color='white', size=16, pad=20)
ax.set_ylabel("Pure Unsupervised Geometries", color='white', size=12)
ax.set_xlabel("Out-Of-Distribution Look-Forward Limits", color='white', size=12)
ax.tick_params(colors='lightgray')

for spine in ax.spines.values():
    spine.set_visible(False)
    
plt.tight_layout()
out_path = os.path.join(ARTIFACT_DIR, 'plot_final_ood_domain_heatmap.png')
fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')

print(f"[*] Native OOD Extrapolation exacted.")
print(f"    Graph exported rigidly to: {out_path}")
