import pandas as pd, numpy as np, os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, median_absolute_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

def median_absolute_percentage_error(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0: return np.nan
    return np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['council_district'] = df['council_district'] if 'council_district' in df.columns else df.get('council_district_x', 1)
df['council_district'] = df['council_district'].fillna(1).astype(str)

df = df.dropna(subset=['year', 'is_protested', 'case_number']).sort_values('year')

try:
    pet = pd.read_csv(os.path.join(ROOT, "Data", "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv"))
    df = df.merge(pet[['case_number', 'signer_pct']], on='case_number', how='left')
    df['signed_area_share'] = (df['signer_pct'] / 100.0).fillna(0.0)
except Exception as e:
    raise RuntimeError("Missing Petition file for continuous analysis.")

# MACRO AGGREGATION: Calculate rolling cumulative sum geometries per district
dist_agg = df.groupby(['council_district', 'year'])['signed_area_share'].mean().reset_index()

# Construct forward spatial rolling timelines
for lead in [1, 2, 3]:
    target_dict = {}
    for d in dist_agg['council_district'].unique():
        sub = dist_agg[dist_agg['council_district'] == d].copy().sort_values('year')
        sub[f'Lead_{lead}yr_Mean_Share'] = sub['signed_area_share'].shift(-lead)
        # Store back
        for _, row in sub.dropna(subset=[f'Lead_{lead}yr_Mean_Share']).iterrows():
            target_dict[(d, row['year'])] = row[f'Lead_{lead}yr_Mean_Share']
            
    # Map back to individual properties
    df[f'Target_{lead}yr'] = df.apply(lambda r: target_dict.get((r['council_district'], r['year']), np.nan), axis=1)

# We can safely analyze properties that have target maps
train_mask = df['year'] < 2020

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'signer_pct', 'signed_area_share', 'Target_1yr', 'Target_2yr', 'Target_3yr']
fut_feat = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']

X_raw_df = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

# Discretization Filter
phys_floats = ['ldb_appraised_val', 'land_market_value', 'total_market_value', 'gross_site_area_acres', 'deed_acreage', 'ldb_land_acres', 'ldb_lotsize', 'improvement_sq_ft', 'ldb_imprv_sqft']
to_discretize = [c for c in phys_floats if c in X_raw_df.columns]
for col in to_discretize:
    dt = DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
    X_col = X_raw_df[[col]].values
    dt.fit(X_col, df['is_protested'].values)
    X_raw_df[col] = dt.apply(X_col)

X_raw = X_raw_df.values
years = df['year'].values

eval_years = [2021, 2022, 2023, 2024]
results = []
print("[*] Executing Multi-Horizon Topography Analysis on Pre-2020 Anchor...")

models = {
    'CatBoost_Regressor': CatBoostRegressor(iterations=100, depth=6, random_seed=42, verbose=0),
    'XGBoost_Tweedie': XGBRegressor(n_estimators=100, max_depth=6, random_state=42, objective='reg:tweedie', tweedie_variance_power=1.5),
    'LightGBM_Poisson': LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, objective='poisson', verbose=-1)
}

for lead in [1, 2, 3]:
    target_col = f'Target_{lead}yr'
    print(f"\n---> Evaluating Macroscopic Target: {target_col}")
    
    # Isolate valid rows for this horizon
    valid_mask = ~df[target_col].isna()
    X_valid = X_raw[valid_mask]
    y_valid = df[target_col].values[valid_mask]
    yr_valid = years[valid_mask]
    tr_mask = yr_valid < 2020
    
    if tr_mask.sum() == 0:
        continue
        
    fitted = {}
    for name, m in models.items():
        m.fit(X_valid[tr_mask], y_valid[tr_mask])
        fitted[name] = m
        
    for test_year in eval_years:
        te_mask = yr_valid == test_year
        if te_mask.sum() == 0: continue
            
        X_test, y_test = X_valid[te_mask], y_valid[te_mask]
        
        for name, m in fitted.items():
            preds = np.clip(m.predict(X_test), 0, 1) # Bounds percentage
            rmse = root_mean_squared_error(y_test, preds)
            mdape = median_absolute_percentage_error(y_test, preds)
            
            results.append({
                'Horizon': f"+{lead} Year",
                'Model': name,
                'Evaluate_Year': test_year,
                'RMSE': round(rmse, 3),
                'MdAPE_nonzero': round(mdape, 3) if not np.isnan(mdape) else None
            })

res_df = pd.DataFrame(results)

print("\n=== Multi-Horizon Regression Out-of-Sample MdAPE ===")
print("Tracking absolute magnitude deviation against strictly non-zero aggregate targets.")

for p in ["+1 Year", "+2 Year", "+3 Year"]:
    print(f"\n--- HORIZON: {p} ---")
    sub = res_df[res_df['Horizon'] == p]
    if len(sub) > 0:
        print(sub.pivot_table(index='Model', columns='Evaluate_Year', values='MdAPE_nonzero'))
    else:
        print("Insufficient future data runway for spatial horizon.")
