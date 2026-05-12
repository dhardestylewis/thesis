import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from catboost import CatBoostRegressor, CatBoostClassifier

print("--- Evaluating Institutional Regime Change (Pre-2020 vs Post-2020) ---", flush=True)

# 1. Load Data
cs = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\Panel\cross_sectional_dml_panel.csv')
cs['year'] = pd.to_datetime(cs['application_start_date'], errors='coerce').dt.year
cs = cs.dropna(subset=['year'])

ex_ante = [
    'Delta_Requested_Height', 'latitude', 'longitude',
    'median_household_income', 'race_white', 'race_black', 'race_hispanic',
    'renter_share', 'rent_burden', 'total_population', 'median_age',
    'appraised_value', 'building_age',
    'mortgage_rate_30yr', 'fed_funds_rate', 'local_unemployment_rate',
    'fire_hazard_severity', 'slope_degree', 'is_imagine_corridor'
]

post_trt = [
    'knn_petition_rate_1km', 'dist_petition_rate_lag1',
    'cumulative_min_signer_dist', 'cumulative_signers_outside_200ft',
    'cumulative_protester_embed_dim1', 'cumulative_protester_embed_dim2',
    'cumulative_petition_attempted', 'cumulative_mobilization_failure'
]
confounders = ex_ante + post_trt
for c in confounders: cs[c] = cs[c].fillna(cs[c].median())

# Create Surviving subset
cs_surv = cs[cs['Withdrawal_Binary'] == 0].dropna(subset=['Height_Attrition', 'days_to_resolution']).copy()

# Filter strict cohorts
pre_cs = cs[(cs['year'] >= 2010) & (cs['year'] <= 2019)]
post_cs = cs[(cs['year'] >= 2020) & (cs['year'] <= 2024)]

pre_surv = cs_surv[(cs_surv['year'] >= 2010) & (cs_surv['year'] <= 2019)]
post_surv = cs_surv[(cs_surv['year'] >= 2020) & (cs_surv['year'] <= 2024)]

def evaluate_regime(train_name, test_name, train_cs, test_cs, train_surv, test_surv):
    print(f"\n======================================")
    print(f"TRAIN ON: {train_name} | TEST ON: {test_name}")
    print(f"======================================")
    
    # --- Hurdle Classifier (ROC-AUC) ---
    X_train_ex = train_cs[ex_ante].values
    Y_train_w = train_cs['Withdrawal_Binary'].values
    
    X_test_ex = test_cs[ex_ante].values
    Y_test_w = test_cs['Withdrawal_Binary'].values
    
    clf = CatBoostClassifier(iterations=150, depth=4, verbose=0, random_seed=42, eval_metric='AUC')
    clf.fit(X_train_ex, Y_train_w)
    roc = roc_auc_score(Y_test_w, clf.predict_proba(X_test_ex)[:, 1])
    print(f"Phase 1 Hurdle (Withdrawal) ROC-AUC: {roc:.3f}")
    
    # Inject P_withdraw into the Joint sets
    train_surv_copy = train_surv.copy()
    test_surv_copy = test_surv.copy()
    train_surv_copy['P_withdraw'] = clf.predict_proba(train_surv_copy[ex_ante].values)[:, 1]
    test_surv_copy['P_withdraw'] = clf.predict_proba(test_surv_copy[ex_ante].values)[:, 1]
    
    joint_c = confounders + ['P_withdraw']
    
    X_train_j = train_surv_copy[joint_c].values
    Y_train_h = train_surv_copy['Height_Attrition'].values
    Y_train_d = train_surv_copy['days_to_resolution'].values
    T_train = train_surv_copy['petition_dose'].values
    
    X_test_j = test_surv_copy[joint_c].values
    Y_test_h = test_surv_copy['Height_Attrition'].values
    Y_test_d = test_surv_copy['days_to_resolution'].values
    T_test = test_surv_copy['petition_dose'].values
    
    # --- Nuisance Performance ---
    mod_h = CatBoostRegressor(iterations=150, depth=6, verbose=0, random_seed=42).fit(X_train_j, Y_train_h)
    mod_d = CatBoostRegressor(iterations=150, depth=6, verbose=0, random_seed=42).fit(X_train_j, Y_train_d)
    mod_t = CatBoostRegressor(iterations=150, depth=6, verbose=0, random_seed=42).fit(X_train_j, T_train)
    
    mae_h = mean_absolute_error(Y_test_h, mod_h.predict(X_test_j))
    mae_d = mean_absolute_error(Y_test_d, mod_d.predict(X_test_j))
    r2_t = r2_score(T_test, mod_t.predict(X_test_j))
    
    print(f"Phase 2 Nuisance: Height MAE = {mae_h:.1f} ft (True Test Mean = {Y_test_h.mean():.1f})")
    print(f"Phase 2 Nuisance: Delay MAE = {mae_d:.1f} days (True Test Mean = {Y_test_d.mean():.1f})")
    print(f"Phase 2 Nuisance: Treatment Propensity (Protest Dose) R2 = {r2_t:.3f}")

evaluate_regime("Pre-2020 Cohort (2010-2019)", "Post-2020 Cohort (2020-2024)", pre_cs, post_cs, pre_surv, post_surv)
evaluate_regime("Post-2020 Cohort (2020-2024)", "Pre-2020 Cohort (2010-2019)", post_cs, pre_cs, post_surv, pre_surv)
