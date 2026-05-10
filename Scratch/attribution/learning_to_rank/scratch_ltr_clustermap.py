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

try:
    omnibus_df = pd.read_csv(os.path.join(DRAFT_DIR, "Omnibus_LTR_Matrix_Extreme.csv"))
except Exception:
    raise RuntimeError("Missing Omnibus_LTR_Matrix_Extreme.csv")

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['council_district'] = df['council_district'] if 'council_district' in df.columns else df.get('council_district_x', 1)
df['council_district'] = df['council_district'].fillna(1).astype(str)
df = df.dropna(subset=['year', 'is_protested', 'case_number'])

pet = pd.read_csv(os.path.join(ROOT, "Data", "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv"))
df = df.merge(pet[['case_number', 'signer_pct']], on='case_number', how='left')
df['signed_area_share'] = (df['signer_pct'] / 100.0).fillna(0.0)

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

feature_names_base = X_raw_df.columns.tolist()
feature_names_meta = feature_names_base + ['META_Probability_Discrete_Cliff', 'META_Regression_Continuous_Float']

# 2D Matrix for Clustermap: [Feature] x [Arch_Topology]
matrix_dict = {}

X_raw = X_raw_df.values
y_abs = df['signed_area_share'].values
y_bool = df['Binary_Target'].values
years = df['year'].values
groups = df['group_id'].values

anchors = [2018, 2019, 2020, 2021, 2022, 2023]
offsets = [1, 2, 3, 4, 5, 6]

print("[*] Launching 2D Master Performance-Weighted Empirical Clustermap Extraction...")

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
        topo_name = f"{arch_name}_Base_Ranker"
        if topo_name not in matrix_dict: matrix_dict[topo_name] = {f: 0.0 for f in feature_names_meta}
        
        b_ranker = CatBoostRanker(iterations=100, depth=depth, loss_function='YetiRank', random_seed=42, verbose=0)
        b_pool = Pool(X_raw[train_mask], label=y_trn, group_id=g_trn)
        b_ranker.fit(b_pool)
        
        b_imp_raw = b_ranker.get_feature_importance(b_pool, type='LossFunctionChange')
        b_imp = 100.0 * (np.abs(b_imp_raw) / np.sum(np.abs(b_imp_raw)))
        
        b_perf_mask = (omnibus_df['Anchor'] == anchor) & (omnibus_df['Architecture'] == arch_name) & (omnibus_df['Topology'] == 'Base_Ranker') & (omnibus_df['Target_Binning'] == 'Absolute_Continuous')
        for offset in offsets:
            o_row = omnibus_df[b_perf_mask & (omnibus_df['Offset'] == f"+{offset}yr")]
            if len(o_row) == 0: continue
            valid_ndcg = max(0.0, float(o_row['Score'].values[0]))
            for f_idx, f_name in enumerate(feature_names_base):
                matrix_dict[topo_name][f_name] += (b_imp[f_idx] * valid_ndcg)

        # 2. META PIPELINE
        topo_name_meta = f"{arch_name}_Meta_Ranker"
        if topo_name_meta not in matrix_dict: matrix_dict[topo_name_meta] = {f: 0.0 for f in feature_names_meta}
        
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
                matrix_dict[topo_name_meta][f_name] += (m_imp[f_idx] * valid_ndcg)

print("[*] Normalizing 2D matrix arrays natively...")
# Convert to DataFrame
df_map = pd.DataFrame(matrix_dict)
df_map = df_map.div(df_map.sum(axis=0), axis=1) * 100.0 # Normalize safely to percentage per architecture
df_map = df_map.fillna(0.0)

# Drop any features that are useless universally (< 0.1 importance anywhere) to keep clustermap readable
df_map = df_map[df_map.max(axis=1) > 0.1]

out_csv = os.path.join(DRAFT_DIR, "LTR_Clustermap_Meta_Attribution.csv")
df_map.to_csv(out_csv, index=True)
print(f"Dumped fully normalized 2D Clustermap Matrix to: {out_csv}")

# Plot Clustermap natively identically matching thesis Fig 7
sns.set_theme(style="white", context="paper", font_scale=1.0)
cg = sns.clustermap(
    df_map, 
    cmap="mako", 
    figsize=(14, 12), 
    linewidths=.5, 
    annot=False, 
    cbar_kws={'label': 'NDCG-Weighted Relational Importance (%)'}
)
cg.fig.suptitle("Performance-Weighted Relational Clustermap\nHierarchical mapping of YetiRank geographic laws distinctly tracking across Base and Meta architectures", 
                fontsize=14, weight='bold', y=1.02)

out_png = os.path.join(DRAFT_DIR, "plot_ltr_clustermap_meta_attribution.png")
cg.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()
print(f"Dumped 2D Clustermap visualization safely to: {out_png}")
