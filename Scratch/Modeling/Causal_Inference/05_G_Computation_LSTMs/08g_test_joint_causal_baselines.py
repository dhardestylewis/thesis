import os
from pathlib import Path
import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from catboost import CatBoostRegressor, CatBoostClassifier
import time

# Root configuration
ROOT = Path(__file__).resolve().parents[4]
PANEL_PATH = ROOT / "Data/Panel/biweekly_panel.csv"

def fraction_01(s):
    x = pd.to_numeric(s, errors='coerce').fillna(0.0)
    if x.quantile(0.99) > 1.0:
        x = x / 100.0
    return x.clip(0.0, 1.0)

print("Loading biweekly panel...", flush=True)
df = pd.read_csv(PANEL_PATH, low_memory=False)

# Load raw zoning data for dates
print("Loading raw zoning dates...", flush=True)
zoning_path = ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_land_use_merged_data.csv"
zoning_df = pd.read_csv(zoning_path, low_memory=False)
zoning_df['start'] = pd.to_datetime(zoning_df['application_start_date'], errors='coerce')
zoning_df['end'] = pd.to_datetime(zoning_df['status_date'], errors='coerce')
zoning_df['days_to_resolution'] = (zoning_df['end'] - zoning_df['start']).dt.days
zoning_df['days_to_resolution'] = zoning_df['days_to_resolution'].clip(0, 3650)
zoning_dates = zoning_df[['case_number', 'days_to_resolution']].drop_duplicates('case_number')

# Load Socrata case statuses
print("Loading Socrata case statuses...", flush=True)
status_path = ROOT / "Data/Zoning_Cases/Processed_Data/CSV/zoning_case_statuses.csv"
status_df = pd.read_csv(status_path, low_memory=False)

# Collapse panel to cross-sectional
print("Collapsing to cross-sectional...", flush=True)
cs = df.groupby('case_number').agg({
    'petition_pct_this_period': 'max',
    'Delta_Approved_Height': 'last',
    'Delta_Requested_Height': 'last',
    'latitude': 'first',
    'longitude': 'first',
    'median_household_income': 'first',
    'race_white': 'first',
    'renter_share': 'first',
    'year': 'first',
    'cumulative_min_signer_dist': 'max',
    'cumulative_signers_outside_200ft': 'max',
    'cumulative_protester_embed_dim1': 'max',
    'cumulative_protester_embed_dim2': 'max',
    'cumulative_petition_attempted': 'max',
    'cumulative_mobilization_failure': 'max'
}).reset_index()

# SURVIVOR BIAS PATCH
mask_withdrawn = cs['Delta_Requested_Height'].notna() & cs['Delta_Approved_Height'].isna()
cs.loc[mask_withdrawn, 'Delta_Approved_Height'] = 0

cs = pd.merge(cs, zoning_dates, on='case_number', how='left')
cs = pd.merge(cs, status_df[['case_number', 'detailed_status']], on='case_number', how='left')

cs['petition_dose'] = fraction_01(cs['petition_pct_this_period'])
cs['Height_Attrition'] = cs['Delta_Requested_Height'] - cs['Delta_Approved_Height']
cs['Withdrawal_Binary'] = (cs['detailed_status'] == 'Withdrawn').astype(float)

# Impute minor missingness
for c in ['median_household_income', 'race_white', 'renter_share']:
    cs[c] = cs[c].fillna(cs[c].median())
for c in ['cumulative_min_signer_dist', 'cumulative_signers_outside_200ft', 'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2', 'cumulative_petition_attempted', 'cumulative_mobilization_failure']:
    cs[c] = cs[c].fillna(0.0)

confounders = [
    'Delta_Requested_Height', 'latitude', 'longitude', 
    'median_household_income', 'race_white', 'renter_share',
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure'
]

cs = cs.dropna(subset=confounders + ['Delta_Approved_Height', 'Height_Attrition', 'petition_dose', 'days_to_resolution', 'year'])

TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
t = 0.20 # Using a 20% petition threshold for the test

print("\n--- RUNNING JOINT BASELINES ACROSS ALL YEARS ---")
model_y_multi = CatBoostRegressor(iterations=100, depth=4, loss_function='MultiRMSE', task_type='GPU', verbose=0)
model_t_gpu = CatBoostClassifier(iterations=100, depth=4, task_type='GPU', verbose=0)
model_y_gpu = CatBoostRegressor(iterations=100, depth=4, task_type='GPU', verbose=0)

results = []

for year_cutoff in TEST_YEARS:
    print(f"\n=== Walk-Forward Cutoff: {year_cutoff} ===", flush=True)
    
    train_mask = cs['year'] < year_cutoff
    test_mask = cs['year'] == year_cutoff

    cs_tr = cs[train_mask]
    cs_te = cs[test_mask]
    
    if len(cs_te) == 0:
        continue

    X_tr = cs_tr[confounders].values
    X_te = cs_te[confounders].values
    D_tr = (cs_tr['petition_dose'] >= t).astype(float).values
    D_te = (cs_te['petition_dose'] >= t).astype(float).values
    
    if D_tr.sum() < 5:
        print(f"Skipping: Only {D_tr.sum()} treated cases.")
        continue

    # --- ARCHITECTURE 1: MULTI-OUTPUT ECONML ---
    surv_mask_tr = ~cs_tr['detailed_status'].isin(['Withdrawn', 'Denied', 'Expired', 'VOID'])
    surv_mask_te = ~cs_te['detailed_status'].isin(['Withdrawn', 'Denied', 'Expired', 'VOID'])

    Y_tr_joint = cs_tr.loc[surv_mask_tr, ['Height_Attrition', 'days_to_resolution']].values
    X_tr_surv = X_tr[surv_mask_tr]
    D_tr_surv = D_tr[surv_mask_tr]
    
    X_te_surv = X_te[surv_mask_te]

    cf_joint = CausalForestDML(model_y=model_y_multi, model_t=model_t_gpu, discrete_treatment=True, n_estimators=100, random_state=42)
    try:
        cf_joint.fit(Y_tr_joint, D_tr_surv, X=X_tr_surv)
        cate_multi_te = cf_joint.effect(X_te_surv) # N x 2 array
        multi_height_cate = cate_multi_te[:, 0].mean()
        multi_delay_cate = cate_multi_te[:, 1].mean()
    except Exception as e:
        print("Multi-Output Failed:", e)
        multi_height_cate = np.nan
        multi_delay_cate = np.nan

    # --- ARCHITECTURE 2: HURDLE PIPELINE ---
    Y_tr_withd = cs_tr['Withdrawal_Binary'].values
    cf_withd = CausalForestDML(model_y=model_y_gpu, model_t=model_t_gpu, discrete_treatment=True, n_estimators=100, random_state=42)
    
    try:
        cf_withd.fit(Y_tr_withd, D_tr, X=X_tr)
        cate_withd_te = cf_withd.effect(X_te)
        withd_cate = cate_withd_te.mean()
    except Exception as e:
        withd_cate = np.nan

    base_surv = 1.0 - cs_te['Withdrawal_Binary'].mean()
    base_delay = cs_te.loc[surv_mask_te, 'days_to_resolution'].mean()
    
    expected_delay_untreated = base_surv * base_delay
    expected_delay_treated = (base_surv - withd_cate) * (base_delay + multi_delay_cate)
    joint_hurdle_cate = expected_delay_treated - expected_delay_untreated

    print(f"  Multi-Output Height CATE: {multi_height_cate:.1f} ft")
    print(f"  Multi-Output Delay CATE:  +{multi_delay_cate:.1f} days")
    print(f"  Withdrawal Penalty CATE:  +{withd_cate*100:.1f}%")
    print(f"  Joint Expected Delay CATE: {joint_hurdle_cate:.1f} days")

print("\nDone!")
