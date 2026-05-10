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

# Load Reference Empirical Array
try:
    omnibus_df = pd.read_csv(os.path.join(DRAFT_DIR, "Omnibus_LTR_Matrix_Extreme.csv"))
except Exception:
    raise RuntimeError("Missing Omnibus_LTR_Matrix_Extreme.csv reference file to provide NDCG empirical grounding.")

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['council_district'] = df['council_district'] if 'council_district' in df.columns else df.get('council_district_x', 1)
df['council_district'] = df['council_district'].fillna(1).astype(str)
df = df.dropna(subset=['year', 'is_protested', 'case_number'])

try:
    pet = pd.read_csv(os.path.join(ROOT, "Data", "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv"))
    df = df.merge(pet[['case_number', 'signer_pct']], on='case_number', how='left')
    df['signed_area_share'] = (df['signer_pct'] / 100.0).fillna(0.0)
except Exception:
    raise RuntimeError("Missing Petition file.")

le = LabelEncoder()
df['group_id'] = le.fit_transform(df['council_district'].astype(str) + "_" + df['year'].astype(str))
df['Bin_Relevance'] = df['signed_area_share'].apply(assign_bin)
df['Binary_Target'] = (df['signed_area_share'] >= 0.20).astype(int)
df = df.sort_values(by=['group_id']).reset_index(drop=True)

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'signer_pct', 'signed_area_share', 'group_id', 'Binary_Target', 'Bin_Relevance', 'original_idx']
fut_feat = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
X_raw_df = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

# Discretize natively
for col in [c for c in ['ldb_appraised_val', 'gross_site_area_acres', 'ldb_lotsize'] if c in X_raw_df.columns]:
    dt = DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
    X_col = X_raw_df[[col]].values
    dt.fit(X_col, df['is_protested'].values)
    X_raw_df[col] = dt.apply(X_col)

# Setup master dictionaries uniquely capturing weighted impact mappings
feature_names_base = X_raw_df.columns.tolist()
feature_names_meta = feature_names_base + ['META_Probability_Discrete_Cliff', 'META_Regression_Continuous_Float']

master_importance_dict = {f: 0.0 for f in feature_names_meta}
total_weight = 0.0

X_raw = X_raw_df.values
y_abs = df['signed_area_share'].values
y_bool = df['Binary_Target'].values
years = df['year'].values
groups = df['group_id'].values

anchors = [2018, 2019, 2020, 2021, 2022, 2023]
offsets = [1, 2, 3, 4, 5, 6]
architectures = ['CatBoost_YetiRank_Depth6', 'CatBoost_YetiRank_Depth10']

print("[*] Launching Master Performance-Weighted Empirical Engine...")

for anchor in anchors:
    train_mask = years < anchor
    train_idx_arr = np.where(train_mask)[0]
    if train_mask.sum() == 0: continue
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    meta_p = np.zeros(train_mask.sum())
    meta_a = np.zeros(train_mask.sum())
    
    for trn, val in kf.split(train_idx_arr):
        b_c = LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1).fit(X_raw[train_idx_arr[trn]], y_bool[train_idx_arr[trn]])
        meta_p[val] = b_c.predict_proba(X_raw[train_idx_arr[val]])[:, 1]
        b_a = LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1).fit(X_raw[train_idx_arr[trn]], y_abs[train_idx_arr[trn]])
        meta_a[val] = b_a.predict(X_raw[train_idx_arr[val]])
        
    X_meta_trn = np.hstack((X_raw[train_mask], meta_p.reshape(-1,1), meta_a.reshape(-1,1)))
    y_trn = y_abs[train_mask]
    g_trn = groups[train_mask]
    
    # Evaluate CatBoost Base and Meta
    for arch_name, depth in [('CatBoost_YetiRank_Depth6', 6), ('CatBoost_YetiRank_Depth10', 10)]:
        
        # 1. BASELINE PIPELINE
        b_ranker = CatBoostRanker(iterations=100, depth=depth, loss_function='YetiRank', random_seed=42, verbose=0)
        b_pool = Pool(X_raw[train_mask], label=y_trn, group_id=g_trn)
        b_ranker.fit(b_pool)
        
        b_imp_raw = b_ranker.get_feature_importance(b_pool, type='LossFunctionChange')
        b_imp = 100.0 * (np.abs(b_imp_raw) / np.sum(np.abs(b_imp_raw)))
        
        # Pull matching theoretical performance for Base offsets tracking spatial survival natively
        b_perf_mask = (omnibus_df['Anchor'] == anchor) & (omnibus_df['Architecture'] == arch_name) & (omnibus_df['Topology'] == 'Base_Ranker') & (omnibus_df['Target_Binning'] == 'Absolute_Continuous')
        for offset in offsets:
            o_row = omnibus_df[b_perf_mask & (omnibus_df['Offset'] == f"+{offset}yr")]
            if len(o_row) == 0: continue
            valid_ndcg = max(0.0, float(o_row['Score'].values[0])) # Floor failure bounds directly tracking purely to zero weight cleanly
            
            for f_idx, f_name in enumerate(feature_names_base):
                master_importance_dict[f_name] += (b_imp[f_idx] * valid_ndcg)
            total_weight += valid_ndcg

        # 2. META PIPELINE
        m_ranker = CatBoostRanker(iterations=100, depth=depth, loss_function='YetiRank', random_seed=42, verbose=0)
        m_pool = Pool(X_meta_trn, label=y_trn, group_id=g_trn)
        m_ranker.fit(m_pool)
        
        m_imp_raw = m_ranker.get_feature_importance(m_pool, type='LossFunctionChange')
        m_imp = 100.0 * (np.abs(m_imp_raw) / np.sum(np.abs(m_imp_raw)))
        
        m_perf_mask = (omnibus_df['Anchor'] == anchor) & (omnibus_df['Architecture'] == arch_name) & (omnibus_df['Topology'] == 'Meta_Ranker') & (omnibus_df['Target_Binning'] == 'Absolute_Continuous')
        for offset in offsets:
            o_row = omnibus_df[m_perf_mask & (omnibus_df['Offset'] == f"+{offset}yr")]
            if len(o_row) == 0: continue
            valid_ndcg = max(0.0, float(o_row['Score'].values[0]))
            
            for f_idx, f_name in enumerate(feature_names_meta):
                master_importance_dict[f_name] += (m_imp[f_idx] * valid_ndcg)
            total_weight += valid_ndcg

print("[*] Normalizing extracted matrix boundaries universally dynamically...")
for k in master_importance_dict.keys():
    master_importance_dict[k] = master_importance_dict[k] / total_weight if total_weight>0 else 0.0

attr_df = pd.DataFrame(list(master_importance_dict.items()), columns=['Feature', 'Weighted_Relational_Importance'])
attr_df = attr_df.sort_values(by='Weighted_Relational_Importance', ascending=False)

out_csv = os.path.join(DRAFT_DIR, "LTR_Weighted_Meta_Attribution.csv")
attr_df.to_csv(out_csv, index=False)
print(f"Dumped fully normalized Weighted LTR Combinatorial Matrix natively out to: {out_csv}")

# Plot
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.figure(figsize=(14, 10))
top_25 = attr_df.head(25)

colors = ['red' if 'META_' in str(x) else '#9b59b6' for x in top_25['Feature']]
sns.barplot(data=top_25, x='Weighted_Relational_Importance', y='Feature', palette=colors)
plt.title("Performance-Weighted Meta-Attribution Map\nEmpirically normalized YetiRank Feature importance bounding half-decade Survival Physics", fontsize=14, weight='bold')
plt.xlabel("NDCG-Weighted Relational Importance Index", fontsize=12)
plt.ylabel("Extracted Geographic Variables + Topologies", fontsize=12)
plt.tight_layout()

out_png = os.path.join(DRAFT_DIR, "plot_ltr_weighted_meta_attribution.png")
plt.savefig(out_png, dpi=300)
plt.close()
print(f"Dumped Weighted Relational Matrix plot definitively to: {out_png}")
