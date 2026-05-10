import pandas as pd, numpy as np, os
from catboost import CatBoostRanker, Pool
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def assign_bin(x):
    if x == 0.0: return 0
    elif x <= 0.05: return 1
    elif x <= 0.20: return 2
    else: return 3

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
DRAFT_DIR = os.path.join(ROOT, "Thesis_Draft")

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['council_district'] = df['council_district'] if 'council_district' in df.columns else df.get('council_district_x', 1)
df['council_district'] = df['council_district'].fillna(1).astype(str)

df = df.dropna(subset=['year', 'is_protested', 'case_number'])

try:
    pet = pd.read_csv(os.path.join(ROOT, "Data", "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv"))
    df = df.merge(pet[['case_number', 'signer_pct']], on='case_number', how='left')
    df['signed_area_share'] = (df['signer_pct'] / 100.0).fillna(0.0)
except Exception as e:
    raise RuntimeError("Missing Petition file.")

le = LabelEncoder()
df['group_id'] = le.fit_transform(df['council_district'].astype(str) + "_" + df['year'].astype(str))
df['Bin_Relevance'] = df['signed_area_share'].apply(assign_bin)
df = df.sort_values(by=['group_id']).reset_index(drop=True)

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'signer_pct', 'signed_area_share', 'group_id', 'Binary_Target', 'Bin_Relevance', 'original_idx']
fut_feat = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']

X_raw_df = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

# Discretize continuous targets exactly reproducing the successful limits
for col in [c for c in ['ldb_appraised_val', 'gross_site_area_acres', 'ldb_lotsize'] if c in X_raw_df.columns]:
    dt = DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
    X_col = X_raw_df[[col]].values
    dt.fit(X_col, df['is_protested'].values)
    X_raw_df[col] = dt.apply(X_col)

X_raw = X_raw_df.values
y_bin = df['Bin_Relevance'].values
groups = df['group_id'].values
feature_names = X_raw_df.columns.tolist()

print("[*] Training Universal Relational Oracle (YetiRank)...")
master_pool = Pool(X_raw, label=y_bin, group_id=groups)

# Train universally strictly learning internal group relativistic physics
ranker = CatBoostRanker(iterations=500, depth=6, loss_function='YetiRank', random_seed=42, verbose=100)
ranker.fit(master_pool)

print("[*] Extracting Native Relational Topography Bounds...")
importances = ranker.get_feature_importance(master_pool, type='LossFunctionChange')
# Some arrays output negative change based on loss function mapping; standardize natively for absolute influence
imp_abs = np.abs(importances)
imp_norm = 100.0 * (imp_abs / np.sum(imp_abs))

attr_df = pd.DataFrame({'Feature': feature_names, 'Relational_Importance_Pct': imp_norm})
attr_df = attr_df.sort_values(by='Relational_Importance_Pct', ascending=False)
out_csv = os.path.join(DRAFT_DIR, "LTR_Feature_Attribution.csv")
attr_df.to_csv(out_csv, index=False)
print(f"Dumped LTR Relational Attribution array strictly out to: {out_csv}")

# Plot
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(12, 10))
top_25 = attr_df.head(25)
sns.barplot(data=top_25, x='Relational_Importance_Pct', y='Feature', palette="rocket")
plt.title("The Spatial Laws of Zoning Opposition\nRelational Feature Attribution via YetiRank mapping Domain-Invariant Geometry", fontsize=14, weight='bold')
plt.xlabel("Relative Ranking Importance (%)", fontsize=12)
plt.ylabel("Extracted Geographic Variables", fontsize=12)
plt.tight_layout()

out_png = os.path.join(DRAFT_DIR, "plot_ltr_attribution.png")
plt.savefig(out_png, dpi=300)
plt.close()
print(f"Dumped Relational Plot out to: {out_png}")
