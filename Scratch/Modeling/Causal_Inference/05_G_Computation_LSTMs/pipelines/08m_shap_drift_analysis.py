import pandas as pd
import numpy as np
import shap
from catboost import CatBoostClassifier
from pathlib import Path

ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")
cs = pd.read_csv(ROOT / "Data/Panel/cross_sectional_dml_panel.csv")
cs['year'] = pd.to_datetime(cs['application_start_date'], errors='coerce').dt.year

ex_ante = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor'
]

def analyze_regime(name, data):
    print(f"\n--- SHAP Drift Analysis: {name} ---")
    X = data[ex_ante]
    Y = data['Withdrawal_Binary']
    
    model = CatBoostClassifier(iterations=200, depth=4, verbose=0, random_seed=42)
    model.fit(X, Y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Get mean absolute SHAP for importance
    importances = np.abs(shap_values).mean(0)
    feat_imp = pd.Series(importances, index=ex_ante).sort_values(ascending=False)
    print(feat_imp.head(10))
    return feat_imp

pre_2020 = cs[cs['year'] < 2020]
post_2020 = cs[cs['year'] >= 2020]

imp_pre = analyze_regime("Pre-2020 (Historical)", pre_2020)
imp_post = analyze_regime("Post-2020 (Modern Regime)", post_2020)

# Detect Inversions
print("\n--- Potential Political Realignment (Importance Shift) ---")
merged = pd.concat([imp_pre.rename('Pre'), imp_post.rename('Post')], axis=1)
merged['Shift'] = merged['Post'] - merged['Pre']
print(merged.sort_values('Shift', ascending=False).head(5))
print("\nTop Dropped Features:")
print(merged.sort_values('Shift', ascending=True).head(5))
