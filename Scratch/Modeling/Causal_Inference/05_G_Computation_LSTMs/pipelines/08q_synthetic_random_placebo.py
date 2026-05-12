import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")
cs = pd.read_csv(ROOT / "Data/Panel/cross_sectional_dml_panel.csv")

# 1. Inject Synthetic Noise
np.random.seed(42)
cs['synthetic_random_noise'] = np.random.normal(0, 1, size=len(cs))

# 2. Define Confounders
ex_ante = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor'
]
safe_hist = ['knn_petition_rate_1km', 'dist_petition_rate_lag1']
confounders = ex_ante + safe_hist

print("\n--- RUNNING SYNTHETIC RANDOM PLACEBO TEST ---", flush=True)
print("Logic: Proven identification requires ZERO signal on a mathematically orthogonal random feature.")

# 3. Fit Causal Forest on Synthetic Outcome
model_y = GradientBoostingRegressor(n_estimators=100, max_depth=4)
model_t = GradientBoostingRegressor(n_estimators=100, max_depth=4)

cf_synthetic = CausalForestDML(
    model_y=model_y, model_t=model_t,
    discrete_treatment=False, n_estimators=500,
    random_state=42
)

X = cs[confounders].values
T = cs['petition_dose'].values
Y_noise = cs['synthetic_random_noise'].values

print("Fitting Causal Forest on pure Gaussian noise...", flush=True)
cf_synthetic.fit(Y_noise, T, X=X)

# 4. Evaluate ATE
ate = cf_synthetic.ate(X)
ate_lb, ate_ub = cf_synthetic.ate_interval(X, alpha=0.05)

# Handle potential scalar return
ate_val = ate[0] if hasattr(ate, "__len__") else ate
lb_val = ate_lb[0] if hasattr(ate_lb, "__len__") else ate_lb
ub_val = ate_ub[0] if hasattr(ate_ub, "__len__") else ate_ub

print(f"\n--- Synthetic Placebo Results ---")
print(f"  Outcome: Gaussian Noise N(0,1)")
print(f"  ATE: {ate_val:.4f} [95% CI: {lb_val:.4f}, {ub_val:.4f}]")

if lb_val <= 0 <= ub_val:
    print(f"  ✅ PASS: CI crosses zero. No hallucinated effect on random noise.")
else:
    print(f"  ❌ FAIL: The estimator detected a spurious effect on pure random junk! (Critical validity failure)")

print("\nSynthetic testing complete!", flush=True)
