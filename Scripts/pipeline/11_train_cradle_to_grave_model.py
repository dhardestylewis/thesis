import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error, classification_report, mean_squared_error
import shap

# Configuration
BASE_DIR = r"C:\Users\dhl\data\Thesis\thesis"
DATA_DIR = os.path.join(BASE_DIR, "Data")
X_PATH = os.path.join(DATA_DIR, "Panel", "cross_sectional_dml_panel.csv")
Y_PATH = os.path.join(DATA_DIR, "Panel", "cradle_to_grave_dataset.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "Model_Outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_merge_data():
    print("Loading Features (X) and Targets (Y)...")
    try:
        df_x = pd.read_csv(X_PATH)
        df_y = pd.read_csv(Y_PATH)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None
        
    df_merged = pd.merge(df_x, df_y, on='case_number', how='inner')
    print(f"Merged Dataset: {len(df_merged)} records.")
    
    exclude_cols = ['case_number', 'tcad_id', 'latitude', 'longitude', 'is_abandoned', 'roi_pct', 'value_created', 
                    'permit_issue_year', 'permit_final_year', 'declared_cost', 'inflation_adjusted_cost', 
                    'stabilized_ears_year', 'stabilized_imprv_value', 'final_land_use_code', 'final_year_built',
                    'application_year', 'gross_site_area_acres', 'zoning_friction_active', 'proposed_zoning']
    
    features = [c for c in df_x.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_x[c])]
    df_merged[features] = df_merged[features].fillna(df_merged[features].median())
    return df_merged, features

def train_stage_1_classification_3way(df, features):
    print("\n--- STAGE 1: REALIZATION PROBABILITY (3-Way Split) ---")
    
    # Outer Temporal Split (Predicting the Post-COVID Reality)
    df_past = df[df['application_year'] <= 2020].copy()
    df_future = df[df['application_year'] > 2020].copy()
    
    # Inner Random Split (Cross-Sectional validation controlling for time)
    X_inner = df_past[features]
    y_inner = df_past['is_abandoned']
    X_train, X_inner_test, y_train, y_inner_test = train_test_split(X_inner, y_inner, test_size=0.2, random_state=42, stratify=y_inner)
    
    X_outer_test = df_future[features]
    y_outer_test = df_future['is_abandoned']
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Evaluate Inner Split (Random)
    probs_inner = clf.predict_proba(X_inner_test)[:, 1]
    auc_inner = roc_auc_score(y_inner_test, probs_inner)
    
    # Evaluate Outer Split (Temporal OOT)
    probs_outer = clf.predict_proba(X_outer_test)[:, 1]
    auc_outer = roc_auc_score(y_outer_test, probs_outer)
    
    print(f"Inner Random AUC  (Train <= 2020): {auc_inner:.3f}")
    print(f"Outer Temporal AUC (Test > 2020) : {auc_outer:.3f}")
    
    return clf

def train_stage_2_regression_3way(df, features):
    print("\n--- STAGE 2: CONDITIONAL ROI (Composite 50/50 Inner Split) ---")
    
    df_stab = df[df['is_abandoned'] == 0].copy()
    df_stab = df_stab[(df_stab['roi_pct'] > -100) & (df_stab['roi_pct'] < 1000)]
    
    # Outer Temporal Split
    df_past = df_stab[df_stab['application_year'] <= 2020].copy()
    df_future = df_stab[df_stab['application_year'] > 2020].copy()
    
    # --- 50/50 Composite Inner Validation ---
    # Temporal half: The most recent 2 years of the inner block (2019-2020)
    inner_temporal_test = df_past[df_past['application_year'] >= 2019]
    n_temporal = len(inner_temporal_test)
    
    # Cross-sectional half: Random sample of equal size from the older history (< 2019)
    inner_train_pool = df_past[df_past['application_year'] < 2019]
    inner_cross_test = inner_train_pool.sample(n=n_temporal, random_state=42)
    
    # The actual inner training set is what remains
    inner_train = inner_train_pool.drop(inner_cross_test.index)
    
    # The composite inner test set (50% cross-sectional interpolation, 50% temporal near-term extrapolation)
    inner_composite_test = pd.concat([inner_temporal_test, inner_cross_test])
    
    X_train = inner_train[features]
    y_train = inner_train['roi_pct']
    
    X_inner_comp_test = inner_composite_test[features]
    y_inner_comp_test = inner_composite_test['roi_pct']
    
    X_outer_test = df_future[features]
    y_outer_test = df_future['roi_pct']
    
    reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    
    # Evaluate Inner Composite (50/50)
    preds_inner_comp = reg.predict(X_inner_comp_test)
    r2_inner_comp = r2_score(y_inner_comp_test, preds_inner_comp)
    mae_inner_comp = mean_absolute_error(y_inner_comp_test, preds_inner_comp)
    
    # We can also evaluate the two halves of the inner composite separately!
    preds_inner_cross = reg.predict(inner_cross_test[features])
    r2_inner_cross = r2_score(inner_cross_test['roi_pct'], preds_inner_cross)
    
    preds_inner_temp = reg.predict(inner_temporal_test[features])
    r2_inner_temp = r2_score(inner_temporal_test['roi_pct'], preds_inner_temp)
    
    # Evaluate Outer Split (Temporal OOT)
    preds_outer = reg.predict(X_outer_test)
    r2_outer = r2_score(y_outer_test, preds_outer)
    mae_outer = mean_absolute_error(y_outer_test, preds_outer)
    
    print(f"\n--- Inner Composite Validation (50% Cross / 50% Temporal Near-Term) ---")
    print(f"R-Squared (Composite): {r2_inner_comp:.3f}")
    print(f"MAE (Composite):       {mae_inner_comp:.1f}% ROI")
    print(f"  -> Breakdown: Cross-Sectional Half R2 = {r2_inner_cross:.3f}")
    print(f"  -> Breakdown: Near-Term Temporal Half R2 = {r2_inner_temp:.3f}")
    
    print(f"\n--- Outer Temporal Validation (Extrapolating strictly > 2020) ---")
    print(f"R-Squared: {r2_outer:.3f}")
    print(f"MAE:       {mae_outer:.1f}% ROI")
    
    # Extract SHAP on the final model to show the drivers of the Inner regime
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer.shap_values(X_inner_comp_test)
    shap_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({'feature': features, 'importance_roi': shap_abs}).sort_values('importance_roi', ascending=False)
    
    return reg, shap_df

def main():
    result = load_and_merge_data()
    if not result: return
    df, features = result
    
    train_stage_1_classification_3way(df, features)
    reg, shap_reg = train_stage_2_regression_3way(df, features)
    
    print("\n--- TOP CAUSAL DRIVERS (Within Macro Regime) ---")
    print(shap_reg.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
