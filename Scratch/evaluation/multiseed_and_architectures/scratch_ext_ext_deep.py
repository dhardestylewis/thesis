import pandas as pd, numpy as np, os
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested']).sort_values('year')

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
X_raw_df = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

phys_floats = ['ldb_appraised_val', 'land_market_value', 'total_market_value', 'gross_site_area_acres', 'deed_acreage', 'ldb_land_acres', 'ldb_lotsize', 'improvement_sq_ft', 'ldb_imprv_sqft']
to_discretize = [c for c in phys_floats if c in X_raw_df.columns]
if len(to_discretize) > 0:
    for col in to_discretize:
        dt = DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
        X_col = X_raw_df[[col]].values
        dt.fit(X_col, df['is_protested'].values)
        X_raw_df[col] = dt.apply(X_col)

X_raw = X_raw_df.values
y = df['is_protested'].values
years = df['year'].values

anchor = 2020
train_mask = years < anchor

X_train_raw = X_raw[train_mask]
y_train = y[train_mask]

# Extremely Deep + Extremely Regularized
models = {
    'CatBoost_RegExtDeep': CatBoostClassifier(iterations=500, depth=14, l2_leaf_reg=50.0, random_seed=42, verbose=0),
    'RandomForest_RegExtDeep': RandomForestClassifier(n_estimators=500, max_depth=25, min_samples_leaf=100, min_samples_split=200, random_state=42),
    'LightGBM_RegExtDeep': LGBMClassifier(n_estimators=500, max_depth=-1, num_leaves=512, reg_lambda=50.0, reg_alpha=10.0, min_child_samples=100, random_state=42, verbose=-1)
}

print(f"[*] Training 'Regularized Extremely Deep' Configurations on Pre-{anchor} Anchor...")
fitted = {}
for name, m in models.items():
    m.fit(X_train_raw, y_train)
    fitted[name] = m

eval_years = [2021, 2022, 2023, 2024]
results = []
print("[*] Evaluating OOS Longitudinal Drift...")
for test_year in eval_years:
    test_mask = years == test_year
    if test_mask.sum() == 0: continue
    X_test_raw, y_test = X_raw[test_mask], y[test_mask]
    
    for name, m in fitted.items():
        p = m.predict_proba(X_test_raw)[:, 1]
        prauc = average_precision_score(y_test, p)
        results.append({
            'Model': name,
            'Evaluate_Year': test_year,
            'PRAUC': round(prauc, 3)
        })

res_df = pd.DataFrame(results)
print("\n=== Regularized Extremely Deep OOS Performance ===")
pivot = res_df.pivot_table(index='Model', columns='Evaluate_Year', values='PRAUC')
print(pivot.to_markdown())

