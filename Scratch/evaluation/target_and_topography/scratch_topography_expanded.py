import pandas as pd, numpy as np, os
from sklearn.metrics import average_precision_score, mean_absolute_error, root_mean_squared_error, median_absolute_error
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Helper function to manually calculate MdAPE avoiding zero-division
def median_absolute_percentage_error(y_true, y_pred):
    # MdAPE is structurally undefined or infinite when y_true = 0.
    # We will filter to only observe predicting error on actual strictly positive property resistance
    mask = y_true > 0
    if mask.sum() == 0: return np.nan
    return np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested', 'case_number']).sort_values('year')

try:
    pet = pd.read_csv(os.path.join(ROOT, "Data", "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv"))
    df = df.merge(pet[['case_number', 'signer_pct']], on='case_number', how='left')
    df['signed_area_share'] = (df['signer_pct'] / 100.0).fillna(0.0)
except Exception as e:
    raise RuntimeError("Missing Petition file for continuous analysis.")

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

# Ensure Target Variable arrays
y_reg = df['signed_area_share'].values
y_class_any = (df['signed_area_share'] > 0.0).astype(int).values
y_class_5 = (df['signed_area_share'] >= 0.05).astype(int).values
y_class_10 = (df['signed_area_share'] >= 0.10).astype(int).values
y_class_20 = (df['signed_area_share'] >= 0.20).astype(int).values

y_class_targets = {
    'Any Protest (>0%)': y_class_any,
    'Minor Protest (>=5%)': y_class_5,
    'Strong Protest (>=10%)': y_class_10,
    'Valid Petition (>=20%)': y_class_20
}

eval_years = [2021, 2022, 2023, 2024]
results = []
print(f"[*] Executing Expanded Topography Analysis on Pre-{anchor} Anchor...")

print("\n---> Evaluating Regression Errors (Zero-Inflated)")
# Train the specialized regressors
reg_models = {
    'CatBoost_Regressor': CatBoostRegressor(iterations=100, depth=6, random_seed=42, verbose=0),
    'XGBoost_Tweedie': XGBRegressor(n_estimators=100, max_depth=6, random_state=42, objective='reg:tweedie', tweedie_variance_power=1.5),
    'LightGBM_Poisson': LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, objective='poisson', verbose=-1)
}

# Only train on Pre-2020
X_train = X_raw[train_mask]
y_train_reg = y_reg[train_mask]

fitted_regs = {}
for name, m in reg_models.items():
    m.fit(X_train, y_train_reg)
    fitted_regs[name] = m

for test_year in eval_years:
    test_mask = years == test_year
    if test_mask.sum() == 0: continue
    X_test, y_test = X_raw[test_mask], y_reg[test_mask]
    
    for name, m in fitted_regs.items():
        preds = np.clip(m.predict(X_test), 0, 1) # Bounds
        
        rmse = root_mean_squared_error(y_test, preds)
        mdae = median_absolute_error(y_test, preds)
        mdape = median_absolute_percentage_error(y_test, preds)
        
        results.append({
            'Target': 'Regression_Metrics',
            'Model': name,
            'Evaluate_Year': test_year,
            'RMSE': round(rmse, 3),
            'MdAE': round(mdae, 4),
            'MdAPE_nonzero': round(mdape, 3) if not np.isnan(mdape) else None
        })

print("\n---> Evaluating Classification Gradient Thresholds")
cb_clf = CatBoostClassifier(iterations=100, depth=6, random_seed=42, verbose=0)
for t_name, y_c in y_class_targets.items():
    cb_clf.fit(X_train, y_c[train_mask])
    
    for test_year in eval_years:
        test_mask = years == test_year
        if test_mask.sum() == 0: continue
        X_test, y_test = X_raw[test_mask], y_c[test_mask]
        
        preds = cb_clf.predict_proba(X_test)[:, 1]
        prauc = average_precision_score(y_test, preds)
        
        results.append({
            'Target': 'Classification_Thresholds',
            'Model': t_name,
            'Evaluate_Year': test_year,
            'RMSE': round(prauc, 3), # RE-using this column for PR-AUC to keep df flat
            'MdAE': None,
            'MdAPE_nonzero': None
        })

res_df = pd.DataFrame(results)

print("\n=== Expanded Zero-Inflated Regression Errors ===")
reg_out = res_df[res_df['Target'] == 'Regression_Metrics']
print(reg_out.pivot_table(index='Model', columns='Evaluate_Year', values=['RMSE', 'MdAE']))

print("\n=== Median Absolute Percentage Error (MdAPE) on strictly Non-Zero targets ===")
print(reg_out.pivot_table(index='Model', columns='Evaluate_Year', values='MdAPE_nonzero'))

print("\n=== Classification Degradation Across Exact Area Thresholds (PR-AUC) ===")
clf_out = res_df[res_df['Target'] == 'Classification_Thresholds']
print(clf_out.pivot_table(index='Model', columns='Evaluate_Year', values='RMSE'))

