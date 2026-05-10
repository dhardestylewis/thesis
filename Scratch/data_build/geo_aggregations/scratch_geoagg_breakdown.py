import pandas as pd, numpy as np, os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, median_absolute_error
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

# Ensure boundaries exist
df['council_district'] = df['council_district'] if 'council_district' in df.columns else df.get('council_district_x', 1)
df['council_district'] = df['council_district'].fillna(1).astype(str)
df['situs_city_state_zip'] = df['situs_city_state_zip'].fillna('Unknown_Zip').astype(str)
df['zoning_case_GEOID'] = df['zoning_case_GEOID'].fillna('Unknown_Tract').astype(str)

df = df.dropna(subset=['year', 'is_protested', 'case_number']).sort_values('year')

try:
    pet = pd.read_csv(os.path.join(ROOT, "Data", "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv"))
    df = df.merge(pet[['case_number', 'signer_pct']], on='case_number', how='left')
    df['signed_area_share'] = (df['signer_pct'] / 100.0).fillna(0.0)
except Exception as e:
    raise RuntimeError("Missing Petition file for continuous analysis.")

print("[*] Engineering Forward Geographic Timelines...")

geo_scales = {
    'Macro (Council District)': 'council_district',
    'Meso (Postal Zipcode)': 'situs_city_state_zip',
    'Micro (Census Tract)': 'zoning_case_GEOID'
}

# Engineer Target Variables
for scale_name, geo_col in geo_scales.items():
    agg = df.groupby([geo_col, 'year'])['signed_area_share'].mean().reset_index()
    target_dict = {}
    for g in agg[geo_col].unique():
        sub = agg[agg[geo_col] == g].copy().sort_values('year')
        sub['Lead_1yr'] = sub['signed_area_share'].shift(-1)
        for _, row in sub.dropna(subset=['Lead_1yr']).iterrows():
            target_dict[(g, row['year'])] = row['Lead_1yr']
    
    df[f'Target_{scale_name}'] = df.apply(lambda r: target_dict.get((r[geo_col], r['year']), np.nan), axis=1)

# Ensure Individual level
df['Target_Absolute (Individual Property)'] = df['signed_area_share']

# Strip features to match earlier array format
drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'signer_pct', 'signed_area_share', 'situs_city_state_zip', 'zoning_case_GEOID', 'Target_Macro (Council District)', 'Target_Meso (Postal Zipcode)', 'Target_Micro (Census Tract)', 'Target_Absolute (Individual Property)']
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
print("\n[*] Executing GeoAgg Breakdown Topography on Pre-2020 Anchor...")

scales_to_test = [
    'Target_Macro (Council District)',
    'Target_Meso (Postal Zipcode)',
    'Target_Micro (Census Tract)',
    'Target_Absolute (Individual Property)'
]

for target_col in scales_to_test:
    scale_label = target_col.split('_')[1]
    
    valid_mask = ~df[target_col].isna()
    X_valid = X_raw[valid_mask]
    y_valid = df[target_col].values[valid_mask]
    yr_valid = years[valid_mask]
    tr_mask = yr_valid < 2020
    
    if tr_mask.sum() == 0: continue
        
    m = LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, objective='poisson', verbose=-1)
    m.fit(X_valid[tr_mask], y_valid[tr_mask])
    
    for test_year in eval_years:
        te_mask = yr_valid == test_year
        if te_mask.sum() == 0: continue
            
        X_test, y_test = X_valid[te_mask], y_valid[te_mask]
        preds = np.clip(m.predict(X_test), 0, 1) 
        
        mdape = median_absolute_percentage_error(y_test, preds)
        
        results.append({
            'Geographic_Layer': scale_label,
            'Evaluate_Year': test_year,
            'MdAPE': round(mdape, 3) if not np.isnan(mdape) else None
        })

res_df = pd.DataFrame(results)

print("\n=== Topography GeoAgg Breakdown ===")
print("Tracking percentage error deviation by shrinking Spatial Geometric Boundaries.")
pivot = res_df.pivot_table(index='Geographic_Layer', columns='Evaluate_Year', values='MdAPE')

# Order the index properly for visual presentation
sorted_index = ['Macro (Council District)', 'Meso (Postal Zipcode)', 'Micro (Census Tract)', 'Absolute (Individual Property)']
pivot = pivot.reindex(sorted_index)
print(pivot.to_markdown())

