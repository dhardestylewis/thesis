"""
08i_train_and_save_model.py

Trains the Causal Forest using CONTINUOUS treatment so we can dynamically
evaluate the CATE for any arbitrary Petition Dose on the fly.
Saves the trained models, the base X matrix, and the geometry FlatGeobuf.
"""

import pandas as pd
import numpy as np
import warnings
import joblib
warnings.filterwarnings('ignore')
from pathlib import Path
from econml.dml import CausalForestDML
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor

model_y_multi = MultiOutputRegressor(GradientBoostingRegressor(max_depth=4, n_estimators=100))
model_t_cont = GradientBoostingRegressor(max_depth=4, n_estimators=100)
model_y_bin = GradientBoostingRegressor(max_depth=4, n_estimators=100)
survival_clf = GradientBoostingClassifier(max_depth=4, n_estimators=100)

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")

print("Loading cleaned cross-sectional panel...", flush=True)
cs_path = ROOT / "Data/Panel/cross_sectional_dml_panel.csv"
cs = pd.read_csv(cs_path)

# Sort temporally by case_number (which contains the year) to enable TimeSeriesSplit
cs = cs.sort_values('case_number').reset_index(drop=True)

demo_cols = ['median_household_income', 'race_white', 'race_black', 'race_hispanic', 
             'renter_share', 'rent_burden', 'total_population', 'median_age',
             'appraised_value', 'building_age', 'mortgage_rate_30yr', 
             'fed_funds_rate', 'local_unemployment_rate',
             'knn_petition_rate_1km', 'dist_petition_rate_lag1']

confounders = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'knn_petition_rate_1km', 'dist_petition_rate_lag1',
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor'
]

# ── HURDLE MODEL: Predict baseline probability of withdrawal ─────────────────
print("\nTraining Phase 1 Hurdle (Withdrawal Propensity)...", flush=True)
X = cs[confounders].values
Y_withd = cs['Withdrawal_Binary'].values
survival_clf.fit(X, Y_withd)
cs['P_withdraw'] = survival_clf.predict_proba(X)[:, 1]

# Expand confounders for the Joint Model to include the survival hurdle
joint_confounders = confounders + ['P_withdraw']

# ── 2. Fit CONTINUOUS Causal Forest ──────────────────────────────────────
print("\nFitting Continuous Causal Forest on historical cases...", flush=True)
T_cont = cs['petition_dose'].values

surv_mask = (
    ~cs['detailed_status'].isin(['Withdrawn', 'Denied', 'Expired', 'VOID']) &
    cs['days_to_resolution'].notna() &
    cs['Height_Attrition'].notna()
)
cs_surv = cs[surv_mask]
X_surv_joint = cs_surv[joint_confounders].values
Y_surv_joint = cs_surv[['Height_Attrition', 'days_to_resolution']].values
T_cont_surv = cs_surv['petition_dose'].values

print(f"  Joint model training N:      {len(cs_surv):,}", flush=True)
print(f"  Withdrawal model training N: {len(cs):,}", flush=True)

# Use strict TimeSeriesSplit for Out-of-Time (OOT) causal validation
tscv = TimeSeriesSplit(n_splits=5)

cf_joint = CausalForestDML(
    model_y=model_y_multi, model_t=model_t_cont,
    discrete_treatment=False, n_estimators=1000, 
    cv=tscv, random_state=42
)
cf_withd = CausalForestDML(
    model_y=model_y_bin, model_t=model_t_cont,
    discrete_treatment=False, n_estimators=1000,
    cv=tscv, random_state=42
)

cf_joint.fit(Y_surv_joint, T_cont_surv, X=X_surv_joint)
cf_withd.fit(Y_withd, T_cont, X=X)

# ── 4. Save Artifacts ──────────────────────────────────────────────────────
models_out = ROOT / "Data/Zoning_Cases/causal_models.pkl"
print(f"Saving models to {models_out}...", flush=True)
joblib.dump({
    'cf_joint': cf_joint,
    'cf_withd': cf_withd,
    'survival_clf': survival_clf
}, models_out)

print("Phase 1 Persistence complete!", flush=True)
