"""
08p_falsification_permutation.py

Production Causal Falsification Pipeline (Permutation).
Shuffles the treatment (petition dose) across cases. If the identification
is valid and the signal is not just global noise, the ATE must collapse
to zero when the pairing between (T) and (Y) is broken.
"""

import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Root resolution for production Scripts/pipeline/ location
ROOT = Path(__file__).resolve().parents[2]

print("\n--- RUNNING PERMUTED TREATMENT (SHUFFLED PLACEBO) TEST ---", flush=True)

panel_path = ROOT / "Data/Panel/cross_sectional_dml_panel.csv"
if not panel_path.exists():
    raise FileNotFoundError(f"Missing DML panel at {panel_path}. Run 08g_causal_panel_prep.py first.")

cs = pd.read_csv(panel_path)

# 1. Define Confounders (EX-ANTE ONLY)
ex_ante = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor'
]

# 2. Train Hurdle for selection correction
print("Training Hurdle Model for selection correction...", flush=True)
clf = GradientBoostingClassifier(n_estimators=100, max_depth=4)
clf.fit(cs[ex_ante].values, cs['Withdrawal_Binary'].values)
cs['P_withdraw'] = clf.predict_proba(cs[ex_ante].values)[:, 1]

# Filter survived cases
cs_surv = cs[cs['Withdrawal_Binary'] == 0].dropna(subset=['Height_Attrition', 'log_days_to_resolution']).copy()

safe_hist = ['knn_petition_rate_1km', 'dist_petition_rate_lag1']
confounders = ex_ante + safe_hist + ['P_withdraw']

# 3. Shuffle the treatment!
print("Shuffling treatment labels (petition dose)...", flush=True)
cs_surv['shuffled_dose'] = cs_surv['petition_dose'].sample(frac=1.0, random_state=42).values

# 4. Fit Causal Forest on Shuffled Data
model_y = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=4))
model_t = GradientBoostingRegressor(n_estimators=100, max_depth=4)

cf_placebo = CausalForestDML(
    model_y=model_y, model_t=model_t,
    discrete_treatment=False, n_estimators=500,
    random_state=42
)

X = cs_surv[confounders].values
T_shuffled = cs_surv['shuffled_dose'].values
Y = cs_surv[['Height_Attrition', 'log_days_to_resolution']].values

print("Fitting Causal Forest on permuted treatments...", flush=True)
cf_placebo.fit(Y, T_shuffled, X=X)

# 5. Evaluate ATE
ate = cf_placebo.ate(X)
ate_lb, ate_ub = cf_placebo.ate_interval(X, alpha=0.05)

print(f"\n--- Permuted Treatment Results ---")
print(f"  Height Attrition ATE: {ate[0]:.4f} [95% CI: {ate_lb[0]:.4f}, {ate_ub[0]:.4f}]")
print(f"  Log-Delay ATE:        {ate[1]:.4f} [95% CI: {ate_lb[1]:.4f}, {ate_ub[1]:.4f}]")

def check_pass(lb, ub, name):
    if lb <= 0 <= ub:
        print(f"  ✅ PASS: {name} signal collapsed to zero as expected.")
    else:
        print(f"  ❌ FAIL: {name} still shows a 'causal' effect with shuffled data! (Global bias detected)")

check_pass(ate_lb[0], ate_ub[0], "Height")
check_pass(ate_lb[1], ate_ub[1], "Log-Delay")

print("\nPermutation testing complete!", flush=True)
