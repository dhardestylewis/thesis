import pandas as pd, numpy as np, os
from sklearn.metrics import ndcg_score
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

def median_absolute_percentage_error(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0: return np.nan
    return np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def assign_bin(x):
    if x == 0.0: return 0
    elif x <= 0.05: return 1
    elif x <= 0.20: return 2
    else: return 3

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

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
df['Binary_Target'] = (df['signed_area_share'] > 0).astype(int)

df['original_idx'] = np.arange(len(df))
df = df.sort_values(by=['group_id'])

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'signer_pct', 'signed_area_share', 'group_id', 'Bin_Relevance', 'Binary_Target', 'original_idx']
fut_feat = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']

X_raw_df = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
for col in [c for c in ['ldb_appraised_val', 'gross_site_area_acres', 'ldb_lotsize'] if c in X_raw_df.columns]:
    dt = DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
    X_col = X_raw_df[[col]].values
    dt.fit(X_col, df['is_protested'].values)
    X_raw_df[col] = dt.apply(X_col)

# ABSOLUTE MAPPING (LOCAL GEOMETRY STRICTLY)
X_raw = X_raw_df.values
y_reg = df['signed_area_share'].values
y_bin = df['Bin_Relevance'].values
y_binary = df['Binary_Target'].values
groups = df['group_id'].values
years = df['year'].values

eval_years = [2021, 2022, 2023, 2024]
anchor = 2020
train_mask = years < anchor
train_idx_arr = np.where(train_mask)[0]

results = []
print("[*] Processing Ultimate Topological Mega-Stack...")

kf = KFold(n_splits=4, shuffle=True, random_state=42)

meta_prob = np.zeros(train_mask.sum())
meta_rank_abs = np.zeros(train_mask.sum())
meta_rank_bin = np.zeros(train_mask.sum())

print("---> Training Exhaustive K-Fold Base Geometries...")
for trn, val in kf.split(train_idx_arr):
    real_trn = train_idx_arr[trn]
    real_val = train_idx_arr[val]
    
    # Base 1: Classification Boolean Vector
    c_base = CatBoostClassifier(iterations=100, depth=6, verbose=0)
    c_base.fit(X_raw[real_trn], y_binary[real_trn])
    meta_prob[val] = c_base.predict_proba(X_raw[real_val])[:, 1]
    
    # Base 2: YetiRank Absolute Continuity Vector
    r_abs = LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1, random_state=42)
    r_abs.fit(X_raw[real_trn], y_reg[real_trn])
    meta_rank_abs[val] = r_abs.predict(X_raw[real_val])

    # Base 3: YetiRank Ordinal Bucket Vector
    r_bin = LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1, random_state=100)
    r_bin.fit(X_raw[real_trn], y_bin[real_trn])
    meta_rank_bin[val] = r_bin.predict(X_raw[real_val])


# Final Extracted Base Models trained universally on all anchor data for test predictions
c_final = CatBoostClassifier(iterations=150, depth=6, verbose=0).fit(X_raw[train_mask], y_binary[train_mask])
r_abs_final = LGBMRegressor(n_estimators=150, max_depth=6, verbose=-1).fit(X_raw[train_mask], y_reg[train_mask])
r_bin_final = LGBMRegressor(n_estimators=150, max_depth=6, verbose=-1).fit(X_raw[train_mask], y_bin[train_mask])

# Stack Feature Map
X_meta_train = np.hstack((X_raw[train_mask], meta_prob.reshape(-1,1), meta_rank_abs.reshape(-1,1), meta_rank_bin.reshape(-1,1)))

print("---> Mapping Unconstrained Hyperparameter Depths...")

# Evaluate Baseline without Stack 
b_reg = LGBMRegressor(n_estimators=200, max_depth=6, objective='poisson', random_state=42, verbose=-1)
b_reg.fit(X_raw[train_mask], y_reg[train_mask])

# The Meta Regression Maps
models = {
    'Stacked_Depth6': LGBMRegressor(n_estimators=200, max_depth=6, objective='poisson', random_state=42, verbose=-1),
    'Stacked_Depth12': LGBMRegressor(n_estimators=300, max_depth=12, objective='poisson', random_state=42, verbose=-1),
    'Stacked_Unconstrained': LGBMRegressor(n_estimators=400, max_depth=-1, num_leaves=128, objective='poisson', random_state=42, verbose=-1)
}

for m_name, m_inst in models.items():
    m_inst.fit(X_meta_train, y_reg[train_mask])

for test_year in eval_years:
    test_mask = years == test_year
    if test_mask.sum() == 0: continue
    
    X_t, y_t = X_raw[test_mask], y_reg[test_mask]
    
    # True Base prediction
    preds_base = np.clip(b_reg.predict(X_t), 0, 1)
    results.append({'Architecture': 'Baseline (No Stack)', 'Year': test_year, 'MdAPE': median_absolute_percentage_error(y_t, preds_base)})
    
    # Meta Features construction for test
    t_prob = c_final.predict_proba(X_t)[:, 1]
    t_abs = r_abs_final.predict(X_t)
    t_bin = r_bin_final.predict(X_t)
    X_meta_test = np.hstack((X_t, t_prob.reshape(-1,1), t_abs.reshape(-1,1), t_bin.reshape(-1,1)))
    
    for m_name, m_inst in models.items():
        preds_meta = np.clip(m_inst.predict(X_meta_test), 0, 1)
        meta_mdape = median_absolute_percentage_error(y_t, preds_meta)
        results.append({'Architecture': f"Meta_{m_name}", 'Year': test_year, 'MdAPE': meta_mdape})

res_df = pd.DataFrame(results)

print("\n=== The Ultimate Topological Stacking Error Bound ===")
pivot = res_df.pivot_table(index='Architecture', columns='Year', values='MdAPE')

# Order rows to show progressive stack depth
sorted_index = ['Baseline (No Stack)', 'Meta_Stacked_Depth6', 'Meta_Stacked_Depth12', 'Meta_Stacked_Unconstrained']
pivot = pivot.reindex(sorted_index)
print(pivot.round(3).to_markdown())

