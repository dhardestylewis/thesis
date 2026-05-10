import pandas as pd, numpy as np, os
from sklearn.metrics import median_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

def median_absolute_percentage_error(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0: return np.nan
    return np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def q25(x): return x.quantile(0.25)
def q75(x): return x.quantile(0.75)

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

geo_scales = {
    'Macro (Council District)': 'council_district',
    'Meso (Postal Zipcode)': 'situs_city_state_zip',
    'Micro (Census Tract)': 'zoning_case_GEOID'
}

anchors = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
results = []

print("[*] Processing Ultimate GeoAgg Mega-Matrix...")

models_dict = {
    'LGBM_Def': LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, objective='poisson', verbose=-1),
    'LGBM_High': LGBMRegressor(n_estimators=150, max_depth=10, random_state=42, objective='poisson', verbose=-1),
    'XGB_Def': XGBRegressor(n_estimators=100, max_depth=6, random_state=42, objective='reg:tweedie', tweedie_variance_power=1.5),
    'XGB_High': XGBRegressor(n_estimators=150, max_depth=10, random_state=42, objective='reg:tweedie', tweedie_variance_power=1.5),
    'CAT_Def': CatBoostRegressor(iterations=100, depth=6, random_seed=42, verbose=0),
    'CAT_High': CatBoostRegressor(iterations=150, depth=10, random_seed=42, verbose=0)
}

for scale_name, geo_col in geo_scales.items():
    print(f"---> Constructing Robust Pipeline: {scale_name}")
    
    agg_funcs = ['median', 'std', q25, q75, 'count']
    agg_feats = df.groupby([geo_col, 'year'])[numeric_cols].agg(agg_funcs)
    agg_feats.columns = [f"{c[0]}_{c[1]}" for c in agg_feats.columns]
    agg_feats = agg_feats.reset_index().fillna(0)
    
    feature_cols = [c for c in agg_feats.columns if c not in [geo_col, 'year', 'signed_area_share']]
    
    # EXACT EXPLICIT CHRONOLOGICAL MAPPING
    agg_target = df.groupby([geo_col, 'year'])['signed_area_share'].mean().reset_index()
    target_dict = agg_target.set_index([geo_col, 'year'])['signed_area_share'].to_dict()
    
    # Create Temporal Offsets
    for lead in [1, 2, 3]:
        agg_feats[f'Target_{lead}yr'] = agg_feats.apply(lambda r: target_dict.get((r[geo_col], r['year'] + lead), np.nan), axis=1)

    for lead in [1, 2, 3]:
        tar_col = f'Target_{lead}yr'
        valid = agg_feats.dropna(subset=[tar_col])
        if len(valid) == 0: continue
            
        X_valid = valid[feature_cols].values
        y_valid = valid[tar_col].values
        yr_valid = valid['year'].values
        
        for anchor in anchors:
            train_mask = yr_valid < anchor
            if train_mask.sum() == 0: continue
            
            for m_name, m_inst in models_dict.items():
                m_inst.fit(X_valid[train_mask], y_valid[train_mask])
                
                anchor_mdape = []
                for test_year in range(anchor, 2026):
                    test_mask = yr_valid == test_year
                    if test_mask.sum() == 0: continue
                    
                    preds = np.clip(m_inst.predict(X_valid[test_mask]), 0, 1)
                    mdape = median_absolute_percentage_error(y_valid[test_mask], preds)
                    if not np.isnan(mdape):
                        anchor_mdape.append(mdape)
                        
                mean_mdape = np.mean(anchor_mdape) if len(anchor_mdape) > 0 else np.nan
                
                results.append({
                    'Layer': scale_name.split(' (')[0],
                    'Architecture': m_name,
                    'Anchor': anchor,
                    'Offset': f"+{lead} Yr",
                    'MdAPE': mean_mdape
                })

# Individual Absolute Sequence
print("---> Processing: Absolute (Individual Property)")
m_inst = models_dict['LGBM_Def'] 
X_ind = df[numeric_cols].values
y_ind = df['signed_area_share'].values
yr_ind = df['year'].values

for anchor in anchors:
    train_mask = yr_ind < anchor
    if train_mask.sum() == 0: continue
    
    m_inst.fit(X_ind[train_mask], y_ind[train_mask])
    anchor_mdape = []
    
    for test_year in range(anchor, 2026):
        test_mask = yr_ind == test_year
        if test_mask.sum() == 0: continue
        preds = np.clip(m_inst.predict(X_ind[test_mask]), 0, 1)
        mdape = median_absolute_percentage_error(y_ind[test_mask], preds)
        if not np.isnan(mdape):
            anchor_mdape.append(mdape)
            
    mean_mdape = np.mean(anchor_mdape) if len(anchor_mdape) > 0 else np.nan
    results.append({
        'Layer': 'Absolute',
        'Architecture': 'LGBM_Def',
        'Anchor': anchor,
        'Offset': "Self",
        'MdAPE': mean_mdape
    })

res_df = pd.DataFrame(results).dropna()

print("\n=== The Ultimate GeoAgg Mega-Matrix: Mean Error Breakdown by Spatial Layer ===")
# Group tightly to present mean architecture baseline
final_agg = res_df.groupby(['Layer', 'Offset'])['MdAPE'].mean().unstack()

# Print specifically ordered bounds
sorted_index = ['Macro', 'Meso', 'Micro', 'Absolute']
final_agg = final_agg.reindex([s for s in sorted_index if s in final_agg.index])
print(final_agg.round(3).to_markdown())

# Anchor Tracking check 
print("\n=== Anchor Exhaustion Check (Pre-2018 to Pre-2024 Average MdAPE across models) ===")
anch_agg = res_df.groupby(['Anchor', 'Layer'])['MdAPE'].mean().unstack()
anch_agg = anch_agg.reindex([s for s in sorted_index if s in anch_agg.columns], axis=1)
print(anch_agg.round(3).to_markdown())

