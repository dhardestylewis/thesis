import pandas as pd
import numpy as np
import os
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched_v2_OmniLagged.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested']).sort_values('year')

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
cat_cols = [c for c in df.columns if c.startswith('raw_')]

if 'latitude' in df.columns and 'longitude' in df.columns:
    df['latitude'] = np.round(df['latitude'], 2)
    df['longitude'] = np.round(df['longitude'], 2)

X_raw = df.drop(columns=[c for c in (drop_cols + future_features + cat_cols) if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
X_raw = X_raw.loc[:, X_raw.var() > 0]
feature_names = np.array(X_raw.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=0.95, random_state=42)
pca.fit(X_scaled)
optimal_k = pca.n_components_

agg = FeatureAgglomeration(n_clusters=optimal_k, metric='euclidean', linkage='ward')
agg.fit(X_scaled)
cluster_labels = agg.labels_

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

output_md = "# Mathematically Generated Domain Composition Dictionary\n\nThis artifact documents exactly what physical parameters the Unsupervised Feature Agglomeration strictly compressed together geometrically.\n\n"

for cluster_id in range(optimal_k):
    member_idx = np.where(cluster_labels == cluster_id)[0]
    member_names = feature_names[member_idx]
    
    variances = np.var(X_scaled[:, member_idx], axis=0)
    variance_sorting = np.argsort(variances)[::-1]
    top_3_names = member_names[variance_sorting[:3]]
    all_sorted_names = member_names[variance_sorting]
    
    semantic_base = get_plain_english_name(' '.join(top_3_names))
    c_name = f"**{semantic_base} [Cluster {cluster_id+1}]**"
    
    # Only map clusters that might be relevant or interesting to avoid 20 pages of markdown, just output all!
    output_md += f"## {c_name} (Contains {len(member_idx)} Native Features)\n"
    output_md += "Top 10 highest-variance mathematical anchors driving this domain:\n"
    for i, name in enumerate(all_sorted_names[:10]):
        output_md += f"- `{name}`\n"
        
    output_md += "\n"

with open(r'C:\Users\dhl\.gemini\antigravity\brain\13179483-b897-4efa-b65b-6259cbe2174e\unsupervised_domain_dictionary.md', 'w') as f:
    f.write(output_md)

print("Dictionary dynamically formulated securely!")
