import pandas as pd, numpy as np, os
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

def median_absolute_percentage_error(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0: return np.nan
    return np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def compute_prauc(y_true, y_pred):
    if len(np.unique(y_true)) < 2: return np.nan
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return auc(r, p)

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

X_raw = X_raw_df.values
y_abs = df['signed_area_share'].values
y_bool = df['Binary_Target'].values
years = df['year'].values

results = []
anchors = [2018, 2019, 2020, 2021, 2022, 2023]
offsets = [1, 2, 3]

# Non-Tree Models
models_reg = {
    'Ridge_L2': Ridge(alpha=1.0, random_state=42),
    'MLP_Regressor': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
}
models_cls = {
    'LogisticReg': LogisticRegression(max_iter=500, random_state=42),
    'MLP_Classifier': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
}

print("[*] Generating Non-Tree Omnibus Omega Array (Deep Learning & Generalized Linear)...")

for anchor in anchors:
    train_mask = years < anchor
    train_idx_arr = np.where(train_mask)[0]
    if train_mask.sum() == 0: continue
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    meta_p = np.zeros(train_mask.sum())
    meta_a = np.zeros(train_mask.sum())
    
    # 1. Base OOF Generate (Scaling explicitly per fold)
    for trn, val in kf.split(train_idx_arr):
        idx_t, idx_v = train_idx_arr[trn], train_idx_arr[val]
        
        scaler = StandardScaler()
        X_t_s = scaler.fit_transform(X_raw[idx_t])
        X_v_s = scaler.transform(X_raw[idx_v])
        
        b_c = LogisticRegression(max_iter=500, random_state=42).fit(X_t_s, y_bool[idx_t])
        meta_p[val] = b_c.predict_proba(X_v_s)[:, 1]
        
        b_a = Ridge(random_state=42).fit(X_t_s, y_abs[idx_t])
        meta_a[val] = b_a.predict(X_v_s)
        
    # Scale full training data once for finalizing test bounds
    full_scaler = StandardScaler()
    X_trn_s = full_scaler.fit_transform(X_raw[train_mask])
    X_meta_trn_s = np.hstack((X_trn_s, meta_p.reshape(-1,1), meta_a.reshape(-1,1)))
    
    fin_c = LogisticRegression(max_iter=500, random_state=42).fit(X_trn_s, y_bool[train_mask])
    fin_a = Ridge(random_state=42).fit(X_trn_s, y_abs[train_mask])

    for offset in offsets:
        test_year = anchor + offset - 1
        if test_year > 2024: continue
        test_mask = years == test_year
        if test_mask.sum() == 0: continue
            
        # Scale test geometry identically to training distribution
        X_t_s = full_scaler.transform(X_raw[test_mask])
        
        t_meta_p = fin_c.predict_proba(X_t_s)[:, 1]
        t_meta_a = fin_a.predict(X_t_s)
        X_meta_t_s = np.hstack((X_t_s, t_meta_p.reshape(-1,1), t_meta_a.reshape(-1,1)))

        # A. Execute REGRESSION OMNIBUS 
        y_test_abs = y_abs[test_mask]
        for m_name, m_inst in models_reg.items():
            # Standard
            m_inst.fit(X_trn_s, y_abs[train_mask])
            preds_base = np.clip(m_inst.predict(X_t_s), 0, 1)
            results.append({
                'Anchor': anchor, 'Offset': f"+{offset}yr", 'TestYear': test_year, 
                'Topology': 'Base_Regression', 'Target_Binning': 'Absolute_Continuous',
                'Architecture': m_name, 'Metric': 'MdAPE', 'Score': median_absolute_percentage_error(y_test_abs, preds_base)
            })
            # Stacked
            m_inst.fit(X_meta_trn_s, y_abs[train_mask])
            preds_meta = np.clip(m_inst.predict(X_meta_t_s), 0, 1)
            results.append({
                'Anchor': anchor, 'Offset': f"+{offset}yr", 'TestYear': test_year, 
                'Topology': 'Meta_Regression', 'Target_Binning': 'Absolute_Continuous',
                'Architecture': m_name, 'Metric': 'MdAPE', 'Score': median_absolute_percentage_error(y_test_abs, preds_meta)
            })
            
        # B. Execute CLASSIFICATION OMNIBUS 
        y_test_bool = y_bool[test_mask]
        for m_name, m_inst in models_cls.items():
            # Standard
            m_inst.fit(X_trn_s, y_bool[train_mask])
            preds_base = m_inst.predict_proba(X_t_s)[:, 1]
            results.append({
                'Anchor': anchor, 'Offset': f"+{offset}yr", 'TestYear': test_year, 
                'Topology': 'Base_Classifier', 'Target_Binning': 'Boolean_Legal',
                'Architecture': m_name, 'Metric': 'PR-AUC', 'Score': compute_prauc(y_test_bool, preds_base)
            })
            # Stacked
            m_inst.fit(X_meta_trn_s, y_bool[train_mask])
            preds_meta = m_inst.predict_proba(X_meta_t_s)[:, 1]
            results.append({
                'Anchor': anchor, 'Offset': f"+{offset}yr", 'TestYear': test_year, 
                'Topology': 'Meta_Classifier', 'Target_Binning': 'Boolean_Legal',
                'Architecture': m_name, 'Metric': 'PR-AUC', 'Score': compute_prauc(y_test_bool, preds_meta)
            })

res_df = pd.DataFrame(results).dropna()
out_dir = os.path.join(DRAFT_DIR, "Omnibus_Nontree_Matrix.csv")
res_df.to_csv(out_dir, index=False)
print(f"Dumped fully exhaustive Non-Tree permutations to: {out_dir}")
