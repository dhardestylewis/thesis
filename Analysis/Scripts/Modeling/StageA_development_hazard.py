import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
import gc

print("==============================================")
print(" STAGE A: Development-Occurrence Hazard Model")
print("==============================================")

try:
    print("[*] Loading FULL 282k Parcel Panel (v3.csv)...")
    # Load in chunks or full memory if machine supports it. We'll load full to allow groupby shifts.
    panel_df = pd.read_csv('Data/Panel/Output/Property_Year_Panel_v3.csv', low_memory=False)
    print(f"    Loaded Full Panel Shape: {panel_df.shape}")
    
    print("[*] Loading historical zoning cases...")
    case_tcad = pd.read_csv('Data/Warehouse_As_Of/H0_Filing_Master_Enriched.csv', low_memory=False)
    
    # Force types
    case_tcad['standardized_tcad_id'] = case_tcad['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    panel_df['standardized_tcad_id'] = panel_df['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
    
    # Establish base event dictionary
    case_years = case_tcad[['standardized_tcad_id', 'year']].drop_duplicates()
    case_years['event'] = 1
    
    print("[*] Constructing Site-Time Horizons (1-5 Year Forward Windows)...")
    merged = panel_df.merge(case_years, on=['standardized_tcad_id', 'year'], how='left')
    merged['event'] = merged['event'].fillna(0).astype(int)
    
    # Sort for time-series shifting
    merged = merged.sort_values(['standardized_tcad_id', 'year'])
    
    # Calculate Hazard horizons (H = 1 to 3 years out)
    merged['event_next_1yr'] = merged.groupby('standardized_tcad_id')['event'].shift(-1).fillna(0)
    merged['event_next_2yr'] = merged.groupby('standardized_tcad_id')['event'].shift(-2).fillna(0)
    merged['event_next_3yr'] = merged.groupby('standardized_tcad_id')['event'].shift(-3).fillna(0)
    
    # Target definitions
    horizons = {
        'H=4 Quarters (1 Yr)': 'event_next_1yr',
        'H=8 Quarters (2 Yr)': 'event_next_2yr',
        'H=12 Quarters (3 Yr)': 'event_next_3yr'
    }
    
    exclude_cols = ['standardized_tcad_id', 'year', 'event', 'event_next_1yr', 'event_next_2yr', 'event_next_3yr', 'case_filed']
    features = [c for c in panel_df.columns if c not in exclude_cols]
    
    num_cols = merged[features].select_dtypes(include=[np.number]).columns
    X = merged[num_cols].fillna(0)
    
    # Memory cleanup
    del panel_df
    gc.collect()

    # ----------------------------------------------------
    # ACADEMIC GAUNTLET LOOP (All Horizons)
    # ----------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import RandomizedSearchCV

    # 1. Classical Domain Heuristic (ILR < 1.0)
    print("\n[*] Establishing Domain Heuristic (Highest & Best Use)...")
    safe_land_val = merged['land_market_value'].replace(0, 1)
    merged['ILR'] = merged['improvement_market_value'] / safe_land_val
    merged['Heuristic_ILR'] = (merged['ILR'] < 1.0).astype(int)

    # Pre-scale Economist Features once
    econ_features = ['ILR', 'year_built', 'land_market_value', 'improvement_market_value']
    X_econ = merged[econ_features].fillna(0)
    scaler = StandardScaler()
    X_econ_scaled = scaler.fit_transform(X_econ)

    output_cols = ['standardized_tcad_id', 'year', 'Heuristic_ILR']

    for horizon_name, target_col in horizons.items():
        print(f"\n==============================================")
        print(f" EVALUATING GAUNTLET HORIZON: {horizon_name}")
        print(f"==============================================")
        h_tag = horizon_name.split()[0] # e.g., 'H=4'
        y = merged[target_col]
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        
        # Eval ILR Heuristic for this horizon
        ilr_auc = average_precision_score(y, merged['Heuristic_ILR'])
        print(f"    [+] Domain Heuristic (ILR < 1.0) PR-AUC:   {ilr_auc:.4f}")

        # 2. Logistic Regression (Econometric Baseline)
        try:
            print("    [+] Training Logistic Regression Econometric Baseline...")
            lr = LogisticRegression(max_iter=500, class_weight='balanced')
            lr.fit(X_econ_scaled, y)
            merged[f'Prob_LR_{h_tag}'] = lr.predict_proba(X_econ_scaled)[:, 1]
            lr_auc = average_precision_score(y, merged[f'Prob_LR_{h_tag}'])
            print(f"    [+] Econometric Baseline (Logistic) PR-AUC: {lr_auc:.4f}")
        except Exception as e:
            print(f"    [-] Logistic baseline failed: {e}")
            merged[f'Prob_LR_{h_tag}'] = 0.0
            
        # 3. LightGBM Challenger (Unconstrained Library Auto-Heuristics)
        print(f"    [+] Initializing Unconstrained Global LightGBM...")
        lgbm_base = LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        lgbm_base.fit(X, y)
        merged[f'Prob_LGBM_{h_tag}'] = lgbm_base.predict_proba(X)[:, 1]
        lgbm_auc = average_precision_score(y, merged[f'Prob_LGBM_{h_tag}'])
        print(f"    [+] Operational PR-AUC (LightGBM):          {lgbm_auc:.4f}")

        # 4. CatBoost Challenger (Unconstrained Library Auto-Heuristics)
        print(f"    [+] Initializing Unconstrained Global CatBoost...")
        from catboost import CatBoostClassifier
        cb_base = CatBoostClassifier(scale_pos_weight=pos_weight, verbose=0, random_seed=42, thread_count=-1)
        cb_base.fit(X, y)
        merged[f'Prob_{h_tag}'] = cb_base.predict_proba(X)[:, 1]
        pr_auc = average_precision_score(y, merged[f'Prob_{h_tag}'])
        print(f"    [+] Operational PR-AUC (CatBoost):          {pr_auc:.4f}")
        
        # Append target columns to output schema
        output_cols.extend([f'Prob_LR_{h_tag}', f'Prob_LGBM_{h_tag}', f'Prob_{h_tag}'])
    
    # Save results
    merged[output_cols].to_csv('Analysis/Output/Track0_Predictive/stage_a_hazard_results.csv', index=False)
    print("\n[+] Saved multi-horizon multi-model probabilities to: Analysis/Output/Track0_Predictive/stage_a_hazard_results.csv")
    
except Exception as e:
    import traceback
    traceback.print_exc()

print("Stage A Complete.")
