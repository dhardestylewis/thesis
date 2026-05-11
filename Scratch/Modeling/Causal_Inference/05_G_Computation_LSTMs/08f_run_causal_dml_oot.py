import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, brier_score_loss

# Root configuration to match thesis pipeline
ROOT = Path(__file__).resolve().parents[4] # thesis root
PANEL_PATH = ROOT / "Data/Panel/biweekly_panel.csv"

OUT_DIR = ROOT / "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = OUT_DIR / "causal_baseline_oot_metrics.csv"

def fraction_01(s):
    x = pd.to_numeric(s, errors='coerce').fillna(0.0)
    if x.quantile(0.99) > 1.0:
        x = x / 100.0
    return x.clip(0.0, 1.0)

print("Loading biweekly panel...", flush=True)
df = pd.read_csv(PANEL_PATH, low_memory=False)

# Load raw zoning data for dates
print("Loading raw zoning dates...", flush=True)
zoning_path = r'c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\zoning_land_use_merged_data.csv'
zoning_df = pd.read_csv(zoning_path, low_memory=False)
zoning_df['start'] = pd.to_datetime(zoning_df['application_start_date'], errors='coerce')
zoning_df['end'] = pd.to_datetime(zoning_df['status_date'], errors='coerce')
zoning_df['days_to_resolution'] = (zoning_df['end'] - zoning_df['start']).dt.days
zoning_df['days_to_resolution'] = zoning_df['days_to_resolution'].clip(0, 3650) # Cap at 10 years
zoning_dates = zoning_df[['case_number', 'days_to_resolution']].drop_duplicates('case_number')

# Load Socrata case statuses
print("Loading Socrata case statuses...", flush=True)
status_path = r'c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\zoning_case_statuses.csv'
status_df = pd.read_csv(status_path, low_memory=False)

# Collapse panel to cross-sectional (Baseline model only)
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

# SURVIVOR BIAS PATCH: If Delta_Requested_Height is known but Delta_Approved_Height is missing,
# it means the case was withdrawn or denied (developer got 0 extra height above base zoning).
mask_withdrawn = cs['Delta_Requested_Height'].notna() & cs['Delta_Approved_Height'].isna()
cs.loc[mask_withdrawn, 'Delta_Approved_Height'] = 0

# Merge dates and statuses
cs = pd.merge(cs, zoning_dates, on='case_number', how='left')
cs = pd.merge(cs, status_df[['case_number', 'detailed_status']], on='case_number', how='left')

cs['petition_dose'] = fraction_01(cs['petition_pct_this_period'])
cs['concession_binary'] = (cs['Delta_Approved_Height'] > 0).astype(float)
cs['Height_Attrition'] = cs['Delta_Requested_Height'] - cs['Delta_Approved_Height']
cs['Withdrawal_Binary'] = (cs['detailed_status'] == 'Withdrawn').astype(float)
cs['Denial_Binary'] = (cs['detailed_status'] == 'Denied').astype(float)

# Impute minor missingness in census demographics
for c in ['median_household_income', 'race_white', 'renter_share']:
    cs[c] = cs[c].fillna(cs[c].median())

# Impute petition embeddings (0 for no petition)
for c in ['cumulative_min_signer_dist', 'cumulative_signers_outside_200ft', 'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2', 'cumulative_petition_attempted', 'cumulative_mobilization_failure']:
    cs[c] = cs[c].fillna(0.0)

confounders = [
    'Delta_Requested_Height', 'latitude', 'longitude', 
    'median_household_income', 'race_white', 'renter_share',
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure'
]

# Drop ANY remaining NaNs to prevent EconML from crashing
cs = cs.dropna(subset=confounders + ['Delta_Approved_Height', 'Height_Attrition', 'concession_binary', 'petition_dose', 'days_to_resolution', 'year'])

TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
# To capture up to the max density, we test up to 0.40. Beyond 0.40 data is too sparse.
THRESHOLDS = np.arange(0.05, 0.45, 0.05)
# Including baseline "Any Petition"
THRESHOLDS = [0.001] + list(THRESHOLDS)

TARGETS = {
    'concession_binary': 'Binary_Upzone',
    'Height_Attrition': 'Continuous_Attrition',
    'days_to_resolution': 'Continuous_Time_Delay',
    'Withdrawal_Binary': 'Binary_Withdrawal',
    'Denial_Binary': 'Binary_Denial'
}

results = []

for year_cutoff in TEST_YEARS:
    print(f"\n=== Walk-Forward Cutoff: {year_cutoff} ===", flush=True)
    
    train_mask = cs['year'] < year_cutoff
    test_mask = cs['year'] == year_cutoff
    
    cs_tr = cs[train_mask]
    cs_te = cs[test_mask]
    
    if len(cs_te) == 0:
        print(f"  No test data for {year_cutoff}, skipping.")
        continue
        
    X_tr = cs_tr[confounders].values
    X_te = cs_te[confounders].values
    
    for t in THRESHOLDS:
        print(f"  Testing Threshold >= {t:.3f}...")
        
        D_tr = (cs_tr['petition_dose'] >= t).astype(float).values
        D_te = (cs_te['petition_dose'] >= t).astype(float).values
        
        # We need enough treated cases in train to split a forest, and at least some variance in test
        if D_tr.sum() < 5:
            print(f"    Skipping: Only {D_tr.sum()} treated cases in training set (< 20{year_cutoff}).")
            continue
            
        for target_col, target_name in TARGETS.items():
            if target_name == "Continuous_Time_Delay":
                # SURVIVOR BIAS PATCH: Only evaluate delay for cases that survived.
                # Defeated cases (withdrawn/denied) have maxed-out NaN delays that skew the continuous target.
                surv_mask_tr = ~cs_tr['detailed_status'].isin(['Withdrawn', 'Denied', 'Expired', 'VOID'])
                surv_mask_te = ~cs_te['detailed_status'].isin(['Withdrawn', 'Denied', 'Expired', 'VOID'])
                
                Y_tr = cs_tr.loc[surv_mask_tr, target_col].values
                Y_te = cs_te.loc[surv_mask_te, target_col].values
                D_tr_cur = D_tr[surv_mask_tr]
                D_te_cur = D_te[surv_mask_te]
                X_tr_cur = X_tr[surv_mask_tr]
                X_te_cur = X_te[surv_mask_te]
            else:
                Y_tr = cs_tr[target_col].values
                Y_te = cs_te[target_col].values
                D_tr_cur = D_tr
                D_te_cur = D_te
                X_tr_cur = X_tr
                X_te_cur = X_te
            
            # Need enough treated cases in current subset
            if D_tr_cur.sum() < 5:
                print(f"    Skipping {target_name}: Only {D_tr_cur.sum()} treated cases in training subset.")
                continue
            
            # Setup CF
            cf = CausalForestDML(
                model_y=LGBMRegressor(max_depth=3, min_child_samples=5),
                model_t=LGBMClassifier(max_depth=3, min_child_samples=5),
                discrete_treatment=True,
                n_estimators=100,
                random_state=42
            )
            
            try:
                cf.fit(Y_tr, D_tr_cur, X=X_tr_cur)
                cate_te = cf.effect(X_te_cur)
                
                # The EconML score() computes the Out-of-Sample R-Scorer (negative MSE of the treatment effect)
                # Note: if D_te has no variance (i.e. no treated cases in the holdout), score() will fail.
                if D_te_cur.sum() >= 1 and D_te_cur.sum() < len(D_te_cur):
                    r_scorer = cf.score(Y_te, D_te_cur, X=X_te_cur)
                else:
                    r_scorer = np.nan
                
                # --- EXPLICIT NUISANCE MODEL EVALUATION ---
                baseline_y = LGBMRegressor(max_depth=3, min_child_samples=5, random_state=42)
                baseline_t = LGBMClassifier(max_depth=3, min_child_samples=5, random_state=42)
                
                baseline_y.fit(X_tr_cur, Y_tr)
                baseline_t.fit(X_tr_cur, D_tr_cur)
                
                Y_pred = baseline_y.predict(X_te_cur)
                D_pred = baseline_t.predict_proba(X_te_cur)[:, 1]
                
                t_brier = brier_score_loss(D_te_cur, D_pred)
                t_roc = roc_auc_score(D_te_cur, D_pred) if (D_te_cur.sum() >= 1 and D_te_cur.sum() < len(D_te_cur)) else np.nan
                
                y_r2 = np.nan
                y_mae = np.nan
                y_roc = np.nan
                y_brier = np.nan
                
                if target_name in ["Continuous_Attrition", "Continuous_Time_Delay"]:
                    y_r2 = r2_score(Y_te, Y_pred)
                    y_mae = mean_absolute_error(Y_te, Y_pred)
                else:
                    y_brier = brier_score_loss(Y_te, Y_pred)
                    y_roc = roc_auc_score(Y_te, Y_pred) if (Y_te.sum() >= 1 and Y_te.sum() < len(Y_te)) else np.nan

                results.append({
                    "Test_Year": year_cutoff,
                    "Target": target_name,
                    "Threshold": t,
                    "N_Train": len(Y_tr),
                    "N_Train_Treated": D_tr_cur.sum(),
                    "N_Test": len(Y_te),
                    "N_Test_Treated": D_te_cur.sum(),
                    "OOT_Mean_CATE": cate_te.mean(),
                    "OOT_R_Scorer": r_scorer,
                    "Propensity_ROC": t_roc,
                    "Propensity_Brier": t_brier,
                    "Outcome_R2": y_r2,
                    "Outcome_MAE": y_mae,
                    "Outcome_ROC": y_roc,
                    "Outcome_Brier": y_brier
                })
                
            except Exception as e:
                print(f"    [{target_name}] FAILED: {e}")

if results:
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(res_df)} OOT causal metric rows to {OUT_CSV}")
else:
    print("\nNo results generated.")
