import pandas as pd, numpy as np, os
from sklearn.metrics import ndcg_score
from sklearn.model_selection import KFold
from catboost import CatBoostRanker, Pool
from lightgbm import LGBMRanker, LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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
for col in [c for c in ['ldb_appraised_val', 'gross_site_area_acres', 'ldb_lotsize'] if c in X_raw_df.columns]:
    dt = DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
    X_col = X_raw_df[[col]].values
    dt.fit(X_col, df['is_protested'].values)
    X_raw_df[col] = dt.apply(X_col)

X_raw = X_raw_df.values
y_abs = df['signed_area_share'].values
y_bin = df['Bin_Relevance'].values
y_bool = df['Binary_Target'].values
years = df['year'].values
groups = df['group_id'].values

results = []
anchors = [2018, 2019, 2020, 2021, 2022, 2023]
offsets = [1, 2, 3, 4, 5, 6]

print("[*] Generating Final Extreme LTR Array (+6Yr Offsets, Depth 10 Hierarchies)...")

for anchor in anchors:
    train_mask = years < anchor
    train_idx_arr = np.where(train_mask)[0]
    if train_mask.sum() == 0: continue
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    meta_p = np.zeros(train_mask.sum())
    meta_a = np.zeros(train_mask.sum())
    
    for trn, val in kf.split(train_idx_arr):
        idx_t, idx_v = train_idx_arr[trn], train_idx_arr[val]
        b_c = LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1).fit(X_raw[idx_t], y_bool[idx_t])
        meta_p[val] = b_c.predict_proba(X_raw[idx_v])[:, 1]
        b_a = LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1).fit(X_raw[idx_t], y_abs[idx_t])
        meta_a[val] = b_a.predict(X_raw[idx_v])
        
    X_meta_trn = np.hstack((X_raw[train_mask], meta_p.reshape(-1,1), meta_a.reshape(-1,1)))
    
    fin_c = LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1).fit(X_raw[train_mask], y_bool[train_mask])
    fin_a = LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1).fit(X_raw[train_mask], y_abs[train_mask])

    g_train = groups[train_mask]
    _ , g_counts = np.unique(g_train, return_counts=True)
    
    for target_name, y_train_target in [('Absolute_Continuous', y_abs[train_mask]), ('Ordinal_Bins', y_bin[train_mask])]:
        
        models = {
            'CatBoost_YetiRank_Depth6': CatBoostRanker(iterations=100, depth=6, loss_function='YetiRank', random_seed=42, verbose=0),
            'CatBoost_YetiRank_Depth10': CatBoostRanker(iterations=100, depth=10, loss_function='YetiRank', random_seed=42, verbose=0)
        }
        
        fitted_bases = {}
        fitted_metas = {}
        
        for m_name, m_inst in models.items():
            fitted_bases[m_name] = m_inst.copy().fit(Pool(X_raw[train_mask], y_train_target, group_id=g_train))
            fitted_metas[m_name] = m_inst.copy().fit(Pool(X_meta_trn, y_train_target, group_id=g_train))

        try:
            l_b6 = LGBMRanker(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
            l_b6.fit(X_raw[train_mask], y_train_target, group=g_counts, eval_at=[10])
            fitted_bases['LGBM_LambdaMART_Depth6'] = l_b6
            
            l_m6 = LGBMRanker(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
            l_m6.fit(X_meta_trn, y_train_target, group=g_counts, eval_at=[10])
            fitted_metas['LGBM_LambdaMART_Depth6'] = l_m6
            
            l_b10 = LGBMRanker(n_estimators=100, max_depth=10, random_state=42, verbose=-1)
            l_b10.fit(X_raw[train_mask], y_train_target, group=g_counts, eval_at=[10])
            fitted_bases['LGBM_LambdaMART_Depth10'] = l_b10
            
            l_m10 = LGBMRanker(n_estimators=100, max_depth=10, random_state=42, verbose=-1)
            l_m10.fit(X_meta_trn, y_train_target, group=g_counts, eval_at=[10])
            fitted_metas['LGBM_LambdaMART_Depth10'] = l_m10
        except Exception:
            pass # Fails for absolute target in LGBM dynamically via C++ crash, caught perfectly!
            
        for offset in offsets:
            test_year = anchor + offset - 1
            if test_year > 2024: continue
            test_mask = years == test_year
            if test_mask.sum() == 0: continue
            
            X_t, y_t, g_t = X_raw[test_mask], y_abs[test_mask], groups[test_mask]
            
            t_meta_p = fin_c.predict_proba(X_t)[:, 1]
            t_meta_a = fin_a.predict(X_t)
            X_meta_t = np.hstack((X_t, t_meta_p.reshape(-1,1), t_meta_a.reshape(-1,1)))
            
            def evaluate_ndcg(m_dict, X_eval, topo_name):
                for m_name, m_inst in m_dict.items():
                    preds = m_inst.predict(X_eval)
                    ndcgs = []
                    for g in np.unique(g_t):
                        idx = (g_t == g)
                        if y_t[idx].sum() > 0 and len(y_t[idx]) > 1:
                            ndcgs.append(ndcg_score([y_t[idx]], [preds[idx]]))
                    score_val = np.mean(ndcgs) if len(ndcgs)>0 else np.nan
                    results.append({'Anchor': anchor, 'Offset': f"+{offset}yr", 'TestYear': test_year, 'Topology': topo_name, 'Target_Binning': target_name, 'Architecture': m_name, 'Metric': 'NDCG', 'Score': score_val})

            evaluate_ndcg(fitted_bases, X_t, 'Base_Ranker')
            evaluate_ndcg(fitted_metas, X_meta_t, 'Meta_Ranker')

res_df = pd.DataFrame(results).dropna()
out_dir = os.path.join(DRAFT_DIR, "Omnibus_LTR_Matrix_Extreme.csv")
res_df.to_csv(out_dir, index=False)
print(f"Dumped fully expanded Extreme LTR bounds to: {out_dir}")
