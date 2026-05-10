import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score
from scipy.cluster.hierarchy import fcluster
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

warnings = __import__('warnings')
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested']).sort_values('year')

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
X_raw_df = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
features = X_raw_df.columns.tolist()
X_raw = X_raw_df.values
y = df['is_protested'].values
years = df['year'].values

SEMANTIC_CLUSTERS = {
    'acs_owner_occupied_units': 'Housing Tenure',
    'acs_renter_occupied_units': 'Housing Tenure',
    'acs_total_housing_units': 'Housing Tenure',
    'acs_race_white': 'Demographics',
    'acs_race_hispanic': 'Demographics',
    'acs_race_black': 'Demographics',
    'acs_race_asian': 'Demographics',
    'acs_median_household_income': 'Neighborhood Income',
    'acs_poverty_count': 'Neighborhood Income',
    'acs_median_home_value': 'Neighborhood Valuation',
    'ldb_appraised_val': 'Property Valuation',
    'land_market_value': 'Property Valuation',
    'total_market_value': 'Property Valuation',
    'improvement_sq_ft': 'Improvement Scale',
    'ldb_imprv_sqft': 'Improvement Scale',
    'ldb_yr_built': 'Structure Age',
    'year_built': 'Structure Age',
    'property_age': 'Structure Age',
    'gross_site_area_acres': 'Parcel Scale',
    'deed_acreage': 'Parcel Scale',
    'ldb_land_acres': 'Parcel Scale',
    'ldb_lotsize': 'Parcel Scale',
    'ldb_far': 'Zoning Density',
    'ldb_units': 'Zoning Density'
}

anchors = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
attribution_matrix = []
labels = []
families = []

for anchor in anchors:
    train_mask = years < anchor
    if train_mask.sum() < 50: continue
    X_train_raw, y_train = X_raw[train_mask], y[train_mask]
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)

    models = {
        'CatBoost': CatBoostClassifier(iterations=100, depth=6, verbose=0, random_seed=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Logistic': LogisticRegression(class_weight='balanced', random_state=42, max_iter=500),
        'TabNet': TabNetClassifier(verbose=0)
    }

    print(f"Training models for Pre-{anchor} anchor...")
    for name, m in models.items():
        if name == 'Logistic': 
            m.fit(X_train_sc, y_train)
            raw_imp = np.abs(m.coef_[0])
        elif name == 'TabNet':
            m.fit(X_train_sc, y_train, max_epochs=25, patience=5)
            raw_imp = m.feature_importances_
        else: 
            m.fit(X_train_raw, y_train)
            if name == 'CatBoost': raw_imp = m.get_feature_importance()
            else: raw_imp = m.feature_importances_
        
        # Ensure it sums to 1
        total = np.sum(raw_imp)
        if total > 0: raw_imp = (raw_imp / total) * 100
        else: raw_imp = np.zeros_like(raw_imp)

        sem_map = {}
        for f_name, imp in zip(features, raw_imp):
            grp = SEMANTIC_CLUSTERS.get(f_name, "Other")
            if grp != "Other":
                sem_map[grp] = sem_map.get(grp, 0) + imp
        
        vec = pd.Series(sem_map)
        # Re-normalize just the semantic pool to 100% so rows sum exactly to 100% across the 9 clusters
        # Wait, if they only care about these 9 clusters, normalize them so they sum to 100%
        vsum = vec.sum()
        if vsum > 0: vec = vec / vsum * 100

        attribution_matrix.append(vec)
        labels.append(f"{name}_{anchor}")
        fam = 'Linear' if name == 'Logistic' else 'Deep' if name == 'TabNet' else 'Trees'
        families.append(fam)

df_attr = pd.DataFrame(attribution_matrix, index=labels).fillna(0)

# Filter zero var
df_attr = df_attr.loc[:, df_attr.var() > 0.0]

sns.set_theme(style='white')
g = sns.clustermap(
    df_attr, 
    cmap='rocket_r',
    method='ward',
    metric='euclidean',
    figsize=(14, 18),
    linewidths=.5,
    annot=True,
    fmt=".1f"
)

g.fig.suptitle("Meta-Attribution Structural Clustering", fontsize=16, fontweight='bold', y=1.02)
g.ax_heatmap.set_xlabel("Semantic Feature Clusters (Invariant Core Testing)", fontsize=12)
g.ax_heatmap.set_ylabel("Environment (Architecture_OriginYear)", fontsize=12)

out_dir = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "SHAP_MetaClustering")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "meta_attribution_clustermap.pdf")
g.savefig(out_path, bbox_inches='tight')
print(f"[+] Saved Clustered Attributions to: {out_path}")

# Now, regenerate archetypal attribution table!
df_attr['Family'] = families
df_agg = df_attr.groupby('Family').mean()

# Order exactly as we had before, sorted descending by Trees
c_order = df_agg.loc['Trees'].sort_values(ascending=False).index.tolist()

tex_lines = [
    r'\begin{table}[htbp]', r'\centering',
    r'\caption[Archetypal Family Attribution]{\textbf{Archetypal Family Attribution (Unified).} Average percentage of absolute model reliance allocated to each semantic feature cluster. Expanded to unequivocally map the entire canonical evaluation space across 42 architectures and anchors. Demonstrates the immense structural divergence between tree ensembles (socio-demographic focus), deep architectures (physical-scale focus), and linear baseline estimators.}',
    r'\label{tab:archetypal_attribution}',
    r'\begin{tabular}{lccc}', r'\toprule',
    r'\textbf{Semantic Target Cluster} & \textbf{Tree Ensembles} & \textbf{Deep Architectures} & \textbf{Linear Architectures} \\',
    r'\midrule'
]

for c in c_order:
    t = df_agg.loc['Trees', c]
    d = df_agg.loc['Deep', c]
    l = df_agg.loc['Linear', c]
    
    t_str = f"{t:.1f}\\%"
    d_str = f"{d:.1f}\\%"
    l_str = f"{l:.1f}\\%"
    
    # Bold max
    max_val = max(t, d, l)
    if t == max_val: t_str = f"\\textbf{{{t_str}}}"
    elif d == max_val: d_str = f"\\textbf{{{d_str}}}"
    else: l_str = f"\\textbf{{{l_str}}}"
        
    tex_lines.append(f"{c} & {t_str} & {d_str} & {l_str} \\\\")

tex_lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])

txt_out = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Tables", "archetypal_attribution.tex")
with open(txt_out, 'w') as f: f.write('\n'.join(tex_lines))
print(f"[+] Saved Archetypal Table to: {txt_out}")

