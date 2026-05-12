import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")
cs = pd.read_csv(ROOT / "Data/Panel/cross_sectional_dml_panel.csv")
cs['year'] = pd.to_datetime(cs['application_start_date'], errors='coerce').dt.year

# 1. Define Confounders
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

print("\n--- Generating Overlap Diagnostics ---", flush=True)

# 2. Calculate Treatment Propensity Scores
# We use a regressor since treatment (dose) is continuous
model_t = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
X = cs[confounders].values
T = cs['petition_dose'].values
model_t.fit(X, T)
cs['propensity_score'] = model_t.predict(X)

# 3. Check for Support Collapse by Year
years = [2018, 2020, 2022, 2024]
plt.figure(figsize=(15, 10))

for i, y in enumerate(years):
    subset = cs[cs['year'] == y]
    if len(subset) == 0: continue
    
    plt.subplot(2, 2, i+1)
    plt.hist(subset['propensity_score'], bins=30, alpha=0.5, label='Predicted Dose (Propensity)', density=True)
    plt.hist(subset['petition_dose'], bins=30, alpha=0.5, label='Actual Dose', density=True)
    plt.title(f"Treatment Overlap in Year {y}")
    plt.xlabel("Petition Dose (0.0 to 1.0)")
    plt.legend()

plt.tight_layout()
diag_plot = ROOT / "Data/Zoning_Cases/treatment_overlap_diagnostics.png"
plt.savefig(diag_plot)
print(f"Saved overlap plot to {diag_plot}", flush=True)

# 4. Effective Sample Size calculation
# ESS = (\sum w)^2 / \sum w^2  where w = 1 / P(T|X)
# For continuous DML, it's more complex, but we can look at the variance of the residuals
residuals = T - cs['propensity_score']
print(f"\nGlobal Treatment Residual Variance: {np.var(residuals):.4f}")
for y in sorted(cs['year'].unique()):
    y_res = residuals[cs['year'] == y]
    if len(y_res) > 0:
        print(f"  Year {y}: Variance = {np.var(y_res):.4f} (N={len(y_res)})")

print("\nDiagnostics complete!", flush=True)
