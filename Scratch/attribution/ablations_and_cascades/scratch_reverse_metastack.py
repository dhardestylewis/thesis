import pandas as pd, numpy as np, os
from sklearn.metrics import ndcg_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

def compute_prauc(y_true, y_pred):
    if len(np.unique(y_true)) < 2: return np.nan
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return auc(r, p)

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
df['Binary_Target'] = (df['signed_area_share'] >= 0.20).astype(int)

df['original_idx'] = np.arange(len(df))
df = df.sort_values(by=['group_id'])

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'signer_pct', 'signed_area_share', 'group_id', 'Binary_Target', 'original_idx']
fut_feat = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']

X_raw_df = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
for col in [c for c in ['ldb_appraised_val', 'gross_site_area_acres', 'ldb_lotsize'] if c in X_raw_df.columns]:
    dt = DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
    X_col = X_raw_df[[col]].values
    dt.fit(X_col, df['is_protested'].values)
    X_raw_df[col] = dt.apply(X_col)

# NATIVE ABSOLUTE PROPERTIES
X_raw = X_raw_df.values
y_reg = df['signed_area_share'].values
y_binary = df['Binary_Target'].values
groups = df['group_id'].values
years = df['year'].values

eval_years = [2021, 2022, 2023, 2024]
anchor = 2020
train_mask = years < anchor
train_idx_arr = np.where(train_mask)[0]

results = []
print("[*] Processing Reverse Meta-Stack Topology...")

kf = KFold(n_splits=4, shuffle=True, random_state=42)

meta_prob = np.zeros(train_mask.sum())
meta_rank = np.zeros(train_mask.sum())
meta_reg = np.zeros(train_mask.sum())

print("---> Training Exhaustive OOF Baselines...")
for trn, val in kf.split(train_idx_arr):
    real_trn = train_idx_arr[trn]
    real_val = train_idx_arr[val]
    
    # Base 1: Classification Boolean Vector
    c_base = CatBoostClassifier(iterations=100, depth=6, verbose=0)
    c_base.fit(X_raw[real_trn], y_binary[real_trn])
    meta_prob[val] = c_base.predict_proba(X_raw[real_val])[:, 1]
    
    # Base 2: YetiRank Relational Vector
    r_base = LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1, random_state=42)
    r_base.fit(X_raw[real_trn], y_reg[real_trn])
    meta_rank[val] = r_base.predict(X_raw[real_val])

    # Base 3: LightGBM Regressor (Absolute Volume)
    reg_base = LGBMRegressor(n_estimators=100, max_depth=6, objective='poisson', verbose=-1, random_state=42)
    reg_base.fit(X_raw[real_trn], y_reg[real_trn])
    meta_reg[val] = reg_base.predict(X_raw[real_val])


# Final Full Models explicitly for the OOS feature building
c_final = CatBoostClassifier(iterations=150, depth=6, verbose=0).fit(X_raw[train_mask], y_binary[train_mask])
r_final = LGBMRegressor(n_estimators=150, max_depth=6, verbose=-1).fit(X_raw[train_mask], y_reg[train_mask])
reg_final = LGBMRegressor(n_estimators=150, max_depth=6, objective='poisson', verbose=-1).fit(X_raw[train_mask], y_reg[train_mask])

print("---> Mapping Inverse Mega-Stacks (Classification & LTR)...")

# --- CLASSIFICATION STACK 
X_meta_cls_trn = np.hstack((X_raw[train_mask], meta_rank.reshape(-1,1), meta_reg.reshape(-1,1)))
b_cls = CatBoostClassifier(iterations=200, depth=6, random_seed=42, verbose=0).fit(X_raw[train_mask], y_binary[train_mask])
m_cls = CatBoostClassifier(iterations=200, depth=6, random_seed=42, verbose=0).fit(X_meta_cls_trn, y_binary[train_mask])

for test_year in eval_years:
    test_mask = years == test_year
    if test_mask.sum() == 0: continue
    X_t, y_t = X_raw[test_mask], y_binary[test_mask]
    
    p_b = b_cls.predict_proba(X_t)[:, 1]
    results.append({'Topology': 'Classifier_Base', 'Year': test_year, 'Metric': 'PR-AUC', 'Score': compute_prauc(y_t, p_b)})
    
    t_rank = r_final.predict(X_t)
    t_reg = reg_final.predict(X_t)
    X_meta_test = np.hstack((X_t, t_rank.reshape(-1,1), t_reg.reshape(-1,1)))
    
    p_m = m_cls.predict_proba(X_meta_test)[:, 1]
    results.append({'Topology': 'Classifier_Meta_Stacked', 'Year': test_year, 'Metric': 'PR-AUC', 'Score': compute_prauc(y_t, p_m)})

# --- LTR YETIRANK STACK
X_meta_rnk_trn = np.hstack((X_raw[train_mask], meta_prob.reshape(-1,1), meta_reg.reshape(-1,1)))
b_rnk = CatBoostRanker(iterations=200, depth=6, loss_function='YetiRank', random_seed=42, verbose=0).fit(Pool(X_raw[train_mask], y_reg[train_mask], group_id=groups[train_mask]))
m_rnk = CatBoostRanker(iterations=200, depth=6, loss_function='YetiRank', random_seed=42, verbose=0).fit(Pool(X_meta_rnk_trn, y_reg[train_mask], group_id=groups[train_mask]))

for test_year in eval_years:
    test_mask = years == test_year
    if test_mask.sum() == 0: continue
    X_t, y_t, g_t = X_raw[test_mask], y_reg[test_mask], groups[test_mask]
    
    # Base
    preds_b = b_rnk.predict(X_t)
    ndcgs_b = []
    for g in np.unique(g_t):
        idx = (g_t == g)
        if y_t[idx].sum() > 0 and len(y_t[idx]) > 1: ndcgs_b.append(ndcg_score([y_t[idx]], [preds_b[idx]]))
    results.append({'Topology': 'Ranker_Base', 'Year': test_year, 'Metric': 'NDCG', 'Score': np.mean(ndcgs_b) if len(ndcgs_b)>0 else np.nan})
    
    # Meta
    t_prob = c_final.predict_proba(X_t)[:, 1]
    t_reg = reg_final.predict(X_t)
    X_meta_rnk_test = np.hstack((X_t, t_prob.reshape(-1,1), t_reg.reshape(-1,1)))
    
    preds_m = m_rnk.predict(X_meta_rnk_test)
    ndcgs_m = []
    for g in np.unique(g_t):
        idx = (g_t == g)
        if y_t[idx].sum() > 0 and len(y_t[idx]) > 1: ndcgs_m.append(ndcg_score([y_t[idx]], [preds_m[idx]]))
    results.append({'Topology': 'Ranker_Meta_Stacked', 'Year': test_year, 'Metric': 'NDCG', 'Score': np.mean(ndcgs_m) if len(ndcgs_m)>0 else np.nan})


res_df = pd.DataFrame(results)

print("\n=== Reverse Meta-Stack Topology Matrix ===")
print("Evaluating if Continuous/Ranking features arbitrarily inflate natively strong Models.")

for metric in ['PR-AUC', 'NDCG']:
    sub = res_df[res_df['Metric'] == metric]
    pivot = sub.pivot_table(index='Topology', columns='Year', values='Score')
    print(f"\nMetric: {metric}")
    print(pivot.round(3).to_markdown())

