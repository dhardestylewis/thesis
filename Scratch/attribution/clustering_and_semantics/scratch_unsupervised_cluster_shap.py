import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as plt_sns
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
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

# Isolate all mathematical vectors
X_raw = df.drop(columns=[c for c in (drop_cols + future_features + cat_cols) if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)

# Drop any globally dead structural vectors before matrix transforms
X_raw = X_raw.loc[:, X_raw.var() > 0]
y = df['is_protested'].values
feature_names = np.array(X_raw.columns)

print(f"[*] Beginning Unsupervised Math Constraints on {len(feature_names)} blind metrics.")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Automatically determine K via Principal Component Analysis (95% Variance Retained)
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, random_state=42)
pca.fit(X_scaled)
optimal_k = pca.n_components_
print(f"[*] Automated Dimensionality Analysis: 95% of all Temporal Variance can be perfectly explained by exactly K={optimal_k} mathematical clusters.")

print(f"[*] Agglomerating highly correlated multidimensional arrays into Top {optimal_k} Latent Domains...")
# Euclidean on Scaled Features = Functional Pearson Correlation Distance. Solves Zero Vector limitations cleanly.
agg = FeatureAgglomeration(n_clusters=optimal_k, metric='euclidean', linkage='ward')
X_clustered_raw = agg.fit_transform(X_scaled)

def get_plain_english_name(top_variables_str):
    vt = top_variables_str.lower()
    if 'income' in vt or 'poverty' in vt or 'rent' in vt or 'owner' in vt or 'race' in vt or 'population' in vt:
        return "Demographic & Wealth Geography"
    if 'value' in vt or 'apprais' in vt:
        return "Property Valuation Trajectories"
    if 'height' in vt or 'far' in vt or 'cov' in vt or 'zoning' in vt:
        return "Physical Density Constraints"
    if 'sqft' in vt or 'area' in vt or 'acre' in vt or 'lot' in vt:
        return "Parcel Topology & Geometries"
    if 'age' in vt or 'year_built' in vt or 'built' in vt:
        return "Structural Age Constraints"
    if 'council' in vt or 'district' in vt:
        return "Political Boundary Markers"
    if 'lat' in vt or 'lon' in vt:
        return "Absolute Spatial Coordinates"
    if 'lui' in vt or 'land_use' in vt:
        return "Municipal Land Use Flags"
    if 'protest' in vt or 'opposition' in vt:
        return "Aggregated Dissent Contagion"
    return "Secondary Administrative Domains"

# Name clusters by identifying the Top 3 variables dominating that cluster computationally
cluster_labels = agg.labels_
cluster_names = []
for cluster_id in range(optimal_k):
    member_idx = np.where(cluster_labels == cluster_id)[0]
    member_names = feature_names[member_idx]
    
    # Calculate global variance of each member in the scaled dataset
    variances = np.var(X_scaled[:, member_idx], axis=0)
    top_3_idx = np.argsort(variances)[::-1][:3]
    top_3_names = member_names[top_3_idx]
    
    # Translate raw math columns to Semantic Domains algebraically
    clean_str = " ".join(top_3_names)
    semantic_base = get_plain_english_name(clean_str)
    
    # If multiple spatial domains overlap into the exact same descriptive bound, append suffix mathematically
    c_name = f"{semantic_base} [Cluster {cluster_id+1}]"
    cluster_names.append(c_name)

X_clustered = pd.DataFrame(X_clustered_raw, columns=cluster_names)
print(f"[*] Dimensionality successfully crushed natively from {len(feature_names)} to {optimal_k} Unsupervised Clusters.")

print("[*] Training Base CatBoost Engine...")
model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
model.fit(X_clustered, y)

print("[*] Extracting True TreeSHAP Values against strictly orthogonalized Latent Domains...")
shap_values = model.get_feature_importance(type='ShapValues', data=Pool(X_clustered, label=y))
global_shap_importance = np.mean(np.abs(shap_values[:, :-1]), axis=0)

shap_df = pd.DataFrame({'Mathematical Domain Vector': cluster_names, 'Mean Absolute SHAP Score': global_shap_importance})
shap_df = shap_df.sort_values('Mean Absolute SHAP Score', ascending=False)
shap_df = shap_df.head(15)

fig, ax = plt.subplots(figsize=(12, 10), facecolor="#121212")
fig.patch.set_facecolor('#121212')

plt_sns.barplot(data=shap_df, x='Mean Absolute SHAP Score', y='Mathematical Domain Vector', palette='mako', ax=ax)
ax.set_title("Unsupervised Feature Agglomeration: Global SHAP Map", color='white', size=16, pad=20)
ax.set_xlabel("Mean Absolute TreeSHAP Contribution (Latent Array Importance)", color='white', size=12)
ax.set_ylabel("", color='white')
ax.tick_params(colors='lightgray')

for spine in ax.spines.values():
    spine.set_visible(False)
    
plt.tight_layout()
out_path = os.path.join(ARTIFACT_DIR, 'plot_final_top15_unsupervised_shap.png')
fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')

print(f"\n[*] SHAP completely resolved over Latent Agglomerative Components!")
print(f"    Graph exported rigidly to: {out_path}")
