import pandas as pd, numpy as np, os
from sklearn.metrics import median_absolute_error
from lightgbm import LGBMRegressor
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

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'date', 'application_start_date', 'final_date', 'council_district_x', 'signer_pct']
fut_feat = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']

df = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').fillna(0)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if 'signed_area_share' in numeric_cols: numeric_cols.remove('signed_area_share')
if 'year' in numeric_cols: numeric_cols.remove('year')

print("[*] Engineering Forward Geographic Timelines with Comprehensive Descriptive Variance...")

geo_scales = {
    'Macro (Council District)': 'council_district',
    'Meso (Postal Zipcode)': 'situs_city_state_zip',
    'Micro (Census Tract)': 'zoning_case_GEOID'
}

eval_years = [2021, 2022, 2023, 2024]
results = []

for scale_name, geo_col in geo_scales.items():
    print(f"---> Processing Exhaustive Descriptive Stats: {scale_name}")
    
    # 1. Exhaustive Descriptive Statistics Aggregation
    agg_funcs = ['mean', 'std', 'min', 'max', 'count']
    agg_feats = df.groupby([geo_col, 'year'])[numeric_cols].agg(agg_funcs)
    
    # Flatten multi-index columns
    agg_feats.columns = [f"{c[0]}_{c[1]}" for c in agg_feats.columns]
    agg_feats = agg_feats.reset_index().fillna(0) # Fills NaNs from std where count=1
    
    feature_cols = [c for c in agg_feats.columns if c not in [geo_col, 'year', 'signed_area_share']]
    
    # 2. Aggregate Target
    agg_target = df.groupby([geo_col, 'year'])['signed_area_share'].mean().reset_index()
    
    # 3. Create Future Timeline target
    target_dict = {}
    for g in agg_target[geo_col].unique():
        sub = agg_target[agg_target[geo_col] == g].copy().sort_values('year')
        sub['Lead_1yr'] = sub['signed_area_share'].shift(-1)
        for _, row in sub.dropna(subset=['Lead_1yr']).iterrows():
            target_dict[(g, row['year'])] = row['Lead_1yr']
            
    # 4. Map Target safely to the Aggregated Feature matrix
    agg_feats['Target_1yr'] = agg_feats.apply(lambda r: target_dict.get((r[geo_col], r['year']), np.nan), axis=1)
    
    # Execution
    valid = agg_feats.dropna(subset=['Target_1yr'])
    if len(valid) == 0: continue
        
    X_valid = valid[feature_cols].values
    y_valid = valid['Target_1yr'].values
    yr_valid = valid['year'].values
    
    train_mask = yr_valid < 2020
    if train_mask.sum() == 0: continue
        
    m = LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, objective='poisson', verbose=-1)
    m.fit(X_valid[train_mask], y_valid[train_mask])
    
    for test_year in eval_years:
        test_mask = yr_valid == test_year
        if test_mask.sum() == 0: continue
        
        preds = np.clip(m.predict(X_valid[test_mask]), 0, 1)
        mdape = median_absolute_percentage_error(y_valid[test_mask], preds)
        
        results.append({
            'Geographic_Layer': scale_name.split(' (')[0] + ' (' + scale_name.split(' (')[1],
            'Evaluate_Year': test_year,
            'MdAPE': round(mdape, 3) if not np.isnan(mdape) else None
        })

# Lastly, compute Individual Absolute scale without leads
print("---> Processing: Absolute (Individual Property Baseline)")
m = LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, objective='poisson', verbose=-1)
X_ind = df[numeric_cols].values
y_ind = df['signed_area_share'].values
yr_ind = df['year'].values
m.fit(X_ind[yr_ind < 2020], y_ind[yr_ind < 2020])

for test_year in eval_years:
    test_mask = yr_ind == test_year
    if test_mask.sum() == 0: continue
    preds = np.clip(m.predict(X_ind[test_mask]), 0, 1)
    mdape = median_absolute_percentage_error(y_ind[test_mask], preds)
    results.append({
        'Geographic_Layer': 'Absolute (Individual Property)',
        'Evaluate_Year': test_year,
        'MdAPE': round(mdape, 3) if not np.isnan(mdape) else None
    })

res_df = pd.DataFrame(results)

print("\n=== Comprehensive Descriptive Topography GeoAgg Breakdown ===")
print("Tracking percentage error deviation utilizing exhaustive statistical matrices (mean, std, min, max, count).")
pivot = res_df.pivot_table(index='Geographic_Layer', columns='Evaluate_Year', values='MdAPE')

sorted_index = ['Macro (Council District)', 'Meso (Postal Zipcode)', 'Micro (Census Tract)', 'Absolute (Individual Property)']
pivot = pivot.reindex(sorted_index)
print(pivot.to_markdown())

