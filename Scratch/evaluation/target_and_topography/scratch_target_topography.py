import pandas as pd, numpy as np, os
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested', 'case_number']).sort_values('year')

# Merge in Area Share from Petitions
try:
    pet = pd.read_csv(os.path.join(ROOT, "Data", "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv"))
    df = df.merge(pet[['case_number', 'signer_pct']], on='case_number', how='left')
    df['signed_area_share'] = (df['signer_pct'] / 100.0).fillna(0.0)
except Exception as e:
    print("Warning: Petition file not found. Using is_protested proxy for Regression.", e)
    df['signed_area_share'] = df['is_protested'] * 0.25 # arbitrary placeholder if missing

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'signer_pct', 'signed_area_share']
fut_feat = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']

X_raw_df = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

# Discretization Filter (Parity with Gauntlet)
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

# Ensure Contagion is boolean
df['spatial_contagion_1yr'] = (df['spatial_contagion_1yr'] > 0).astype(int)
df['spatial_contagion_3yr'] = (df['spatial_contagion_3yr'] > 0).astype(int)

targets = {
    'Regression_AreaShare': df['signed_area_share'].values,
    'Classification_Contagion1yr': df['spatial_contagion_1yr'].values,
    'Classification_Contagion3yr': df['spatial_contagion_3yr'].values
}

results = []
print(f"[*] Executing Target Topography Analysis on Pre-{anchor} Anchor...")

for target_name, y in targets.items():
    print(f"\n---> Evaluating Target Surface: {target_name}")
    y_train = y[train_mask]
    
    if 'Regression' in target_name:
        models = {
            'CatBoost_Default': CatBoostRegressor(iterations=100, depth=6, random_seed=42, verbose=0),
            'CatBoost_ExtDeep': CatBoostRegressor(iterations=300, depth=14, random_seed=42, verbose=0),
            'RandomForest_Default': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        }
    else:
        models = {
            'CatBoost_Default': CatBoostClassifier(iterations=100, depth=6, random_seed=42, verbose=0),
            'CatBoost_ExtDeep': CatBoostClassifier(iterations=300, depth=14, random_seed=42, verbose=0),
            'RandomForest_Default': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        }

    fitted = {}
    for name, m in models.items():
        m.fit(X_raw[train_mask], y_train)
        fitted[name] = m

    for test_year in [2021, 2022, 2023, 2024]:
        test_mask = years == test_year
        if test_mask.sum() == 0: continue
            
        X_test, y_test = X_raw[test_mask], y[test_mask]
        
        for name, m in fitted.items():
            if 'Regression' in target_name:
                preds = m.predict(X_test)
                # Cap negative predictions bounds to strictly positive zero floor
                preds = np.clip(preds, 0, 1)
                metric_val = r2_score(y_test, preds)
                metric_name = 'OOS_R2'
            else:
                preds = m.predict_proba(X_test)[:, 1]
                metric_val = average_precision_score(y_test, preds)
                metric_name = 'OOS_PRAUC'
                
            results.append({
                'Target': target_name,
                'Model': name,
                'Evaluate_Year': test_year,
                'Metric_Type': metric_name,
                'Metric_Value': round(metric_val, 3)
            })

res_df = pd.DataFrame(results)

print("\n--- Output Matrices ---")
for t in res_df['Target'].unique():
    print(f"\n### {t}")
    sub = res_df[res_df['Target'] == t]
    pivot = sub.pivot_table(index='Model', columns='Evaluate_Year', values='Metric_Value')
    print(pivot.to_markdown())

