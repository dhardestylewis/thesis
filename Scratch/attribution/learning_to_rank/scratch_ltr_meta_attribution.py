import pandas as pd, numpy as np, os
from catboost import CatBoostRanker, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold
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
df['Binary_Target'] = (df['signed_area_share'] >= 0.20).astype(int)
df = df.sort_values(by=['group_id']).reset_index(drop=True)

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'signer_pct', 'signed_area_share', 'group_id', 'Binary_Target', 'Bin_Relevance', 'original_idx']
fut_feat = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']

X_raw_df = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

# Discretize continuous targets explicitly mirroring successful limits
for col in [c for c in ['ldb_appraised_val', 'gross_site_area_acres', 'ldb_lotsize'] if c in X_raw_df.columns]:
    dt = DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
    X_col = X_raw_df[[col]].values
    dt.fit(X_col, df['is_protested'].values)
    X_raw_df[col] = dt.apply(X_col)

X_raw = X_raw_df.values
y_bin = df['Bin_Relevance'].values
y_abs = df['signed_area_share'].values
y_bool = df['Binary_Target'].values
groups = df['group_id'].values

print("[*] Generating Master K-Fold OOF Predictions natively to cleanly extract Meta-Features...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_p = np.zeros(len(X_raw))
meta_a = np.zeros(len(X_raw))

for trn, val in kf.split(X_raw):
    b_c = LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1).fit(X_raw[trn], y_bool[trn])
    meta_p[val] = b_c.predict_proba(X_raw[val])[:, 1]
    
    b_a = LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1).fit(X_raw[trn], y_abs[trn])
    meta_a[val] = b_a.predict(X_raw[val])

# Combine explicitly mapping exact architecture failure matrix bounds
X_meta = np.hstack((X_raw, meta_p.reshape(-1, 1), meta_a.reshape(-1, 1)))
meta_feature_names = X_raw_df.columns.tolist() + ['META_Probability_Discrete_Cliff', 'META_Regression_Continuous_Float']

print("[*] Training Master Poisoned Relational Oracle (YetiRank Meta-Stack)...")
meta_pool = Pool(X_meta, label=y_bin, group_id=groups)

ranker = CatBoostRanker(iterations=500, depth=6, loss_function='YetiRank', random_seed=42, verbose=100)
ranker.fit(meta_pool)

print("[*] Extracting Native Meta-Attribution Poison Bounds...")
importances = ranker.get_feature_importance(meta_pool, type='LossFunctionChange')
imp_abs = np.abs(importances)
imp_norm = 100.0 * (imp_abs / np.sum(imp_abs))

attr_df = pd.DataFrame({'Feature': meta_feature_names, 'Relational_Importance_Pct': imp_norm})
attr_df = attr_df.sort_values(by='Relational_Importance_Pct', ascending=False)
out_csv = os.path.join(DRAFT_DIR, "LTR_Meta_Feature_Attribution.csv")
attr_df.to_csv(out_csv, index=False)
print(f"Dumped LTR Poison Feature array strictly out to: {out_csv}")

# Plot
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(12, 10))
top_25 = attr_df.head(25)

# Color the specific meta-features bright Red visually highlighting their toxic takeover capability natively
colors = ['red' if 'META_' in str(x) else '#5dade2' for x in top_25['Feature']]
sns.barplot(data=top_25, x='Relational_Importance_Pct', y='Feature', palette=colors)
plt.title("The Spatial Poison: Meta-Ranker Feature Attribution\nVisualizing continuous regression arrays structurally overriding geographic logic", fontsize=14, weight='bold')
plt.xlabel("Relative Ranking Importance (%)", fontsize=12)
plt.ylabel("Extracted Geographic Variables + Meta-Stacks", fontsize=12)
plt.tight_layout()

out_png = os.path.join(DRAFT_DIR, "plot_ltr_meta_attribution.png")
plt.savefig(out_png, dpi=300)
plt.close()
print(f"Dumped Relational Poison Plot out to: {out_png}")
