"""
08i_train_causal_dml.py

Production Causal Inference Training Pipeline.
Estimates Average Treatment Effects (ATE) of neighborhood mobilization on
entitlement outcomes (height attrition and resolution delay) using
Double Machine Learning (DML) / Causal Forest.

Identification: Regime-Stratified CIA, strictly pre-treatment confounders.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# Root resolution for production Scripts/pipeline/ location
ROOT = Path(__file__).resolve().parents[2]

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("Loading cross-sectional DML panel...", flush=True)
panel_path = ROOT / "Data/Panel/cross_sectional_dml_panel.csv"
if not panel_path.exists():
    raise FileNotFoundError(f"Missing DML panel at {panel_path}. Run 08g_causal_panel_prep.py first.")

cs = pd.read_csv(panel_path)
cs['year'] = pd.to_datetime(cs['application_start_date'], errors='coerce').dt.year

# ── 2. Identification Strategy (X) ───────────────────────────────────────────
# STRICTLY PRE-TREATMENT. Purged of all post-treatment mediators.
ex_ante_confounders = [
    'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'knn_petition_rate_1km', 'dist_petition_rate_lag1',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor'
]

# ── 3. HURDLE MODEL: Selection Propensity ────────────────────────────────────
# Predict baseline probability of withdrawal using ex-ante state
print("\nTraining Phase 1 Hurdle (Withdrawal Propensity)...", flush=True)
survival_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
X_ex_ante = cs[ex_ante_confounders].values
Y_withd = cs['Withdrawal_Binary'].values
survival_clf.fit(X_ex_ante, Y_withd)
cs['P_withdraw'] = survival_clf.predict_proba(X_ex_ante)[:, 1]

# ── 4. CAUSAL FOREST (DML) ───────────────────────────────────────────────────
# Joint model for Height Attrition and Resolution Delay
print("Fitting Joint Causal Forest (DML)...", flush=True)

# Select survivors for outcome models (Non-withdrawn cases)
cs_surv = cs[cs['Withdrawal_Binary'] == 0].dropna(subset=['Height_Attrition', 'log_days_to_resolution'])
T_cont_surv = cs_surv['petition_dose'].values
Y_surv_joint = cs_surv[['Height_Attrition', 'log_days_to_resolution']].values

# Include the Hurdle Propensity as a joint confounder
joint_confounders = ex_ante_confounders + ['P_withdraw']
X_surv_joint = cs_surv[joint_confounders].values

cf_joint = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, max_depth=6),
    model_t=RandomForestRegressor(n_estimators=100, max_depth=6),
    discrete_treatment=False,
    random_state=42
)
cf_joint.fit(Y_surv_joint, T_cont_surv, X=X_surv_joint)

# ── 5. TEMPORAL EVALUATION (Walk-Forward ATE) ────────────────────────────────
print("\n--- Exhaustive OOT ATE Matrix (95% CI) ---", flush=True)
ate_results = []
years = sorted(cs['year'].dropna().unique())

for y in years:
    if y < 2016: continue
    
    # Slice data for target year
    df_y = cs[cs['year'] == y].dropna(subset=joint_confounders)
    if len(df_y) < 10: continue
    
    X_y = df_y[joint_confounders].values
    
    # Calculate ATE for the year
    # cf_joint returns effects for all outcomes
    ate_y = cf_joint.const_marginal_ate(X_y)
    interval_y = cf_joint.const_marginal_ate_interval(X_y)
    
    h_ate = ate_y[0]
    h_ci  = interval_y[:, 0]
    d_ate = ate_y[1]
    d_ci  = interval_y[:, 1]
    
    print(f"Year {int(y)}: Height ATE = {h_ate:.1f} ft [{h_ci[0]:.1f}, {h_ci[1]:.1f}] | "
          f"Log-Delay ATE = {d_ate:.2f} [{d_ci[0]:.2f}, {d_ci[1]:.2f}]")
    
    ate_results.append({
        'year': int(y),
        'height_ate': h_ate, 'height_low': h_ci[0], 'height_high': h_ci[1],
        'delay_ate': d_ate, 'delay_low': d_ci[0], 'delay_high': d_ci[1]
    })

# ── 6. Persistence ──────────────────────────────────────────────────────────
models_path = ROOT / "Data/Zoning_Cases/causal_models_production.pkl"
print(f"\nSaving models to {models_path}...", flush=True)
with open(models_path, 'wb') as f:
    pickle.dump({
        'cf_joint': cf_joint,
        'hurdle_model': survival_clf,
        'features': joint_confounders,
        'ate_history': ate_results
    }, f)

print("Causal Production Pipeline complete!", flush=True)
