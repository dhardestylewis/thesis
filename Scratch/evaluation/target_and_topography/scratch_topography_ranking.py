import pandas as pd, numpy as np, os
from sklearn.metrics import f1_score, ndcg_score
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from xgboost import XGBRanker
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

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
    raise RuntimeError("Missing Petition file for continuous analysis.")

# Create Query Groups for LTR
# LTR requires queries to be sorted grouped blocks.
le = LabelEncoder()
# A query group will be a unique combination of District and Year to simulate localized competitiveness
df['group_id_str'] = df['council_district'].astype(str) + "_" + df['year'].astype(str)
df['group_id'] = le.fit_transform(df['group_id_str'])

# Sort required for Ranker
df = df.sort_values(by=['group_id'])

# Create Ordinal Bins
def assign_bin(x):
    if x == 0.0: return 0
    elif x < 0.05: return 1
    elif x < 0.20: return 2
    else: return 3

df['Ordinal_Bin'] = df['signed_area_share'].apply(assign_bin)

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'signer_pct', 'signed_area_share', 'group_id_str', 'group_id', 'Ordinal_Bin']
fut_feat = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']

X_raw_df = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

phys_floats = ['ldb_appraised_val', 'land_market_value', 'total_market_value', 'gross_site_area_acres', 'deed_acreage', 'ldb_land_acres', 'ldb_lotsize', 'improvement_sq_ft', 'ldb_imprv_sqft']
to_discretize = [c for c in phys_floats if c in X_raw_df.columns]
for col in to_discretize:
    dt = DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
    X_col = X_raw_df[[col]].values
    dt.fit(X_col, df['is_protested'].values)
    X_raw_df[col] = dt.apply(X_col)

X_raw = X_raw_df.values
years = df['year'].values
anchor = 2020
train_mask = years < anchor

# Ensure arrays
y_reg = df['signed_area_share'].values
y_bin = df['Ordinal_Bin'].values
groups = df['group_id'].values

eval_years = [2021, 2022, 2023, 2024]
results = []
print(f"[*] Executing Ranking Target Topographies on Pre-{anchor} Anchor...\n")

# --- TOPOGRAPHY 1: MULTICLASS ORDINAL CLASSIFICATION ---
print("---> Evaluating Ordinal Bin-Ranking")
clf = CatBoostClassifier(iterations=200, depth=6, loss_function='MultiClass', random_seed=42, verbose=0)
clf.fit(X_raw[train_mask], y_bin[train_mask])

for test_year in eval_years:
    test_mask = years == test_year
    if test_mask.sum() == 0: continue
    X_test, y_test = X_raw[test_mask], y_bin[test_mask]
    
    # Predict bins
    preds = clf.predict(X_test)
    f1_mac = f1_score(y_test, preds, average='macro')
    
    results.append({
        'Task': 'MultiClass_Bins',
        'Model': 'CatBoost_Ordinal',
        'Evaluate_Year': test_year,
        'Metric': round(f1_mac, 3)
    })

# --- TOPOGRAPHY 2: LEARNING TO RANK (NDCG) ---
print("---> Evaluating LTR (Learning-To-Rank) YetiRank")
# Pool for Train
train_pool = Pool(
    data=X_raw[train_mask],
    label=y_reg[train_mask],
    group_id=groups[train_mask]
)

ranker = CatBoostRanker(iterations=200, depth=6, loss_function='YetiRank', random_seed=42, verbose=0)
ranker.fit(train_pool)

# Evaluate locally via NDCG
for test_year in eval_years:
    test_mask = years == test_year
    if test_mask.sum() == 0: continue
    X_test, y_test_r = X_raw[test_mask], y_reg[test_mask]
    g_test = groups[test_mask]
    
    # Calculate Mean NDCG across groups (districts) in the test year
    ndcgs = []
    preds = ranker.predict(X_test)
    for g in np.unique(g_test):
        idx = (g_test == g)
        if y_test_r[idx].sum() > 0 and len(y_test_r[idx]) > 1: # Requires at least one positive resistance
            ndcg = ndcg_score([y_test_r[idx]], [preds[idx]])
            ndcgs.append(ndcg)
            
    mean_ndcg = np.mean(ndcgs) if len(ndcgs) > 0 else 0.0
    
    results.append({
        'Task': 'NDCG_Ranking',
        'Model': 'CatBoost_YetiRank',
        'Evaluate_Year': test_year,
        'Metric': round(mean_ndcg, 3)
    })

res_df = pd.DataFrame(results)

print("\n=== Ordinal Bin-Ranking (Classification Macro-F1) ===")
bins_out = res_df[res_df['Task'] == 'MultiClass_Bins']
print(bins_out.pivot_table(index='Model', columns='Evaluate_Year', values='Metric'))

print("\n=== Learning-to-Rank (Mean NDCG by Query Group) ===")
rank_out = res_df[res_df['Task'] == 'NDCG_Ranking']
print(rank_out.pivot_table(index='Model', columns='Evaluate_Year', values='Metric'))

