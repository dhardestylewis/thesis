"""
08r_referee_robustness_pack.py

"Referee-Proof" Robustness Suite for Causal Inference.
Implements the targeted sensitivity checks required to satisfy top-tier 
economic journal standards, focusing on model stability across nuisance 
learners, seed variation, treatment definitions, and feature ablations.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, LogisticRegressionCV
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Root resolution for production Scripts/pipeline/ location
ROOT = Path(__file__).resolve().parents[2]

# ── 1. Load Data ─────────────────────────────────────────────────────────────
panel_path = ROOT / "Data/Panel/cross_sectional_dml_panel.csv"
if not panel_path.exists():
    raise FileNotFoundError(f"Missing DML panel at {panel_path}. Run 08g_causal_panel_prep.py first.")

cs = pd.read_csv(panel_path)
# Survivor subset for outcome models (Primary: Height_Attrition)
cs_surv = cs[cs['Withdrawal_Binary'] == 0].dropna(subset=['Height_Attrition'])

ex_ante = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'knn_petition_rate_1km', 'dist_petition_rate_lag1',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor'
]

robustness_results = []

def record_result(test_group, variant, h_ate, h_interval=None):
    res = {
        'Test_Group': test_group,
        'Variant': variant,
        'Height_ATE': h_ate
    }
    if h_interval is not None:
        res['Height_LB'], res['Height_UB'] = h_interval
    robustness_results.append(res)
    print(f"[{test_group}] {variant:20} | Height ATE: {h_ate:6.2f}")

# ── 2. NUISANCE MODEL SENSITIVITY ────────────────────────────────────────────
print("\n--- 3.1 Nuisance Model Sensitivity ---")

nuisances = {
    'Linear/Lasso': (LassoCV(), LassoCV()),
    'RandomForest': (RandomForestRegressor(n_estimators=100, max_depth=6), RandomForestRegressor(n_estimators=100, max_depth=6)),
    'GradBoost': (GradientBoostingRegressor(n_estimators=100, max_depth=4), GradientBoostingRegressor(n_estimators=100, max_depth=4))
}

for name, (m_y, m_t) in nuisances.items():
    cf = CausalForestDML(model_y=m_y, model_t=m_t, discrete_treatment=False, random_state=42)
    X = cs_surv[ex_ante].values
    T = cs_surv['petition_dose'].values
    Y = cs_surv['Height_Attrition'].values
    cf.fit(Y, T, X=X)
    ate = cf.ate(X)
    record_result('Nuisance_Sensitivity', name, np.atleast_1d(ate)[0])

# ── 3. SEED STABILITY (20 SEEDS) ─────────────────────────────────────────────
print("\n--- 3.2 Multi-Seed Stability (N=20) ---", flush=True)
seed_ates = []
for s in range(20):
    print(f"  Fitting Seed {s+1}/20...", end="\r", flush=True)
    cf = CausalForestDML(random_state=s)
    cf.fit(Y, T, X=X)
    seed_ates.append(np.atleast_1d(cf.ate(X))[0])

print("\n  Seed Stability Fits Complete.", flush=True)
seed_ates = np.array(seed_ates)
h_mean, h_std = seed_ates.mean(), seed_ates.std()
h_pos_pct = (seed_ates > 0).mean() * 100

record_result('Seed_Stability', f'Mean (N=20, {h_pos_pct:.0f}% pos)', h_mean)
record_result('Seed_Stability', 'Std Dev', h_std)

# ── 4. TREATMENT ROBUSTNESS ──────────────────────────────────────────────────
print("\n--- 3.5 Treatment Definition Robustness ---", flush=True)

treatments = {
    'Continuous (Base)': cs_surv['petition_dose'].values,
    'Log-Transformed': np.log1p(cs_surv['petition_dose'].values),
    'Winsorized (95th)': np.clip(cs_surv['petition_dose'].values, 0, np.percentile(cs_surv['petition_dose'].values, 95)),
    'Binary (>0.20)': (cs_surv['petition_dose'] > 0.20).astype(float)
}

for name, T_variant in treatments.items():
    print(f"  Evaluating {name}...", flush=True)
    is_discrete = 'Binary' in name
    cf = CausalForestDML(discrete_treatment=is_discrete, random_state=42)
    cf.fit(Y, T_variant, X=X)
    ate = cf.ate(X)
    record_result('Treatment_Robustness', name, np.atleast_1d(ate)[0])

# ── 5. FEATURE ABLATION STUDY ────────────────────────────────────────────────
print("\n--- 3.4 Feature Ablations ---", flush=True)

ablations = {
    'Baseline (Full X)': ex_ante,
    'No Spatial': [f for f in ex_ante if f not in ['latitude', 'longitude', 'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor']],
    'No Macro': [f for f in ex_ante if f not in ['mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate']],
    'No History/Lags': [f for f in ex_ante if f not in ['knn_petition_rate_1km', 'dist_petition_rate_lag1']],
    'No Demographics': [f for f in ex_ante if f not in ['median_household_income', 'race_white', 'race_black', 'race_hispanic', 'renter_share', 'rent_burden', 'total_population', 'median_age']]
}

for name, X_cols in ablations.items():
    print(f"  Evaluating Ablation: {name}...", flush=True)
    X_v = cs_surv[X_cols].values
    cf = CausalForestDML(random_state=42)
    cf.fit(Y, T, X=X_v)
    ate = cf.ate(X_v)
    record_result('Feature_Ablation', name, np.atleast_1d(ate)[0])

# ── 6. Consolidation & Export ────────────────────────────────────────────────
out_df = pd.DataFrame(robustness_results)
out_path = ROOT / "Data/Zoning_Cases/referee_robustness_matrix.csv"
out_df.to_csv(out_path, index=False)

print(f"\nRobustness Pack Complete. Matrix saved to: {out_path}", flush=True)

# Generate a "Referee-Ready" Summary Text
print("\n--- ROBUSTNESS SUMMARY FOR MANUSCRIPT ---")
print(f"1. Sign Stability (Height): {h_pos_pct:.1f}% of seeds yielded positive ATE.")
print(f"2. Nuisance Range (Height): [{out_df[out_df['Test_Group']=='Nuisance_Sensitivity']['Height_ATE'].min():.2f}, {out_df[out_df['Test_Group']=='Nuisance_Sensitivity']['Height_ATE'].max():.2f}]")
print(f"3. Ablation Max Delta: {np.abs(out_df[out_df['Test_Group']=='Feature_Ablation']['Height_ATE'].diff()).max():.2f}")
