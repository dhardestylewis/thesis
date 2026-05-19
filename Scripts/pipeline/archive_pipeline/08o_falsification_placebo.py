"""
08o_falsification_placebo.py

Production Causal Falsification Pipeline.
Tests for "spurious effects" on immutable physical features (placebos).
If the identification strategy is valid, neighborhood mobilization should have
zero causal effect on geographic slope or building age.
"""

import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Root resolution for production Scripts/pipeline/ location
ROOT = Path(__file__).resolve().parents[2]

print("\n--- RUNNING CAUSAL FALSIFICATION (PLACEBO) TESTS ---", flush=True)

panel_path = ROOT / "Data/Panel/cross_sectional_dml_panel.csv"
if not panel_path.exists():
    raise FileNotFoundError(f"Missing DML panel at {panel_path}. Run 08g_causal_panel_prep.py first.")

cs = pd.read_csv(panel_path)

# 1. Define Confounders (EX-ANTE ONLY)
ex_ante = [
    'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'fire_hazard_severity', 'is_imagine_corridor'
]
# Note: we exclude slope_degree and building_age from confounders when testing them as outcomes
safe_hist = ['knn_petition_rate_1km', 'dist_petition_rate_lag1']
confounders = ex_ante + safe_hist

def run_placebo(target_col):
    print(f"\nTesting Outcome Placebo: {target_col}...")
    
    # We use a simple Causal Forest on the whole dataset to check for global leakage
    model_y = GradientBoostingRegressor(n_estimators=100, max_depth=4)
    model_t = GradientBoostingRegressor(n_estimators=100, max_depth=4)
    
    cf = CausalForestDML(
        model_y=model_y, model_t=model_t,
        discrete_treatment=False, n_estimators=500,
        random_state=42
    )
    
    X = cs[confounders].values
    T = cs['petition_dose'].values
    Y = cs[target_col].values
    
    cf.fit(Y, T, X=X)
    
    ate = cf.ate(X)
    ate_lb, ate_ub = cf.ate_interval(X, alpha=0.05)
    
    print(f"  Result for {target_col}:")
    ate_val = ate[0] if hasattr(ate, "__len__") else ate
    lb_val = ate_lb[0] if hasattr(ate_lb, "__len__") else ate_lb
    ub_val = ate_ub[0] if hasattr(ate_ub, "__len__") else ate_ub
    
    print(f"    ATE: {ate_val:.4f} [95% CI: {lb_val:.4f}, {ub_val:.4f}]")
    
    if lb_val <= 0 <= ub_val:
        print(f"    ✅ PASS: CI crosses zero. No false causal signal detected.")
    else:
        print(f"    ❌ FAIL: Detected a statistically significant effect on an immutable feature!")

run_placebo('slope_degree')
run_placebo('building_age')

print("\nFalsification testing complete!", flush=True)
