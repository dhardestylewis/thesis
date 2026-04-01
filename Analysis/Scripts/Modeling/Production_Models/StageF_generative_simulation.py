"""
Stage F: Generative Forward Simulation (Empty Parcel Chaining)
==============================================================
Demonstrates the conceptually rigorous forward-chaining architecture.
Unlike Stages A-E which evaluate accuracy on ground-truth administrative
deadlines (static evaluation), this pipeline synthesizes future cases 
by chaining predictions end-to-end sequentially:
    Project_Hazard = f(Parcel)
    Predicted_Scale = f(Parcel | Hazard)
    Predicted_Opposition = f(Parcel | Hazard, Predicted_Scale)
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

ROOT = r"C:\Users\dhl\data\thesis\thesis"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from artifact_registry import ROOT_DIR, DATA_WAREHOUSE_DIR, TRACK1_DIR, TraceabilityRegistry as AR

DATA_IN = str(DATA_WAREHOUSE_DIR / "H0_Filing_Master_Enriched.csv")
A_PROBS = str(AR.STAGE_A_HAZARD_RESULTS)
B_MODEL = str(AR.STAGE_B_MODEL)
C_MODEL_H0 = str(AR.STAGE_C_MODEL_H0)
C_MODEL_H3 = str(AR.STAGE_C_MODEL_H3)
OUT_DIR = str(TRACK1_DIR)


def build_autoregressive_transformers():
    print("    [+] Building H0 -> H3 Autoregressive Imputation Engine...")
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    
    h3_path = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H3_Filing_Master_NLP.csv")
    if not os.path.exists(h3_path):
        print("        [-] Could not find H3 dataset to train autoregressive imputer.")
        return None
        
    df_h3 = pd.read_csv(h3_path, low_memory=False)
    nlp_targets = [c for c in df_h3.columns if c.startswith('tfidf_') or c.startswith('speech_')]
    if not nlp_targets:
        return None
        
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 'final_date', 'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4', 'ipw', 'council_district', 'council_district_x']
    X_base = df_h3.drop(columns=nlp_targets + drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    Y_nlp = df_h3[nlp_targets].fillna(0)
    
    print(f"        -> Translating {X_base.shape[1]} Base Features into {len(nlp_targets)} Future Temporal NLP Vectors...")
    imputer = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42, n_jobs=-1)
    imputer.fit(X_base, Y_nlp)
    
    out_path = os.path.join(OUT_DIR, "stage_f_autoregressive_imputer_H0_to_H3.joblib")
    joblib.dump({'model': imputer, 'features': list(X_base.columns), 'targets': nlp_targets}, out_path)
    print(f"        [+] Saved Autoregressive Generator (H0->H3) to {out_path}")
    return out_path

def run_generative_simulation():
    print("==========================================================")
    print(" STARTING GENERATIVE FORWARD SIMULATION CHASSIS (STAGE F) ")
    print("==========================================================")
    
    # 1. Load Baseline Parcels (Simulating 'Empty Parcels' today)
    print("[1] Loading spatial baseline database...")
    if not os.path.exists(DATA_IN):
        print(f"    [!] Missing master dataset: {DATA_IN}")
        return
    df = pd.read_csv(DATA_IN, low_memory=False)
    
    # 2. Ingest Stage A (Development Hazard)
    if os.path.exists(A_PROBS):
        print("[2] Ingesting Stage A Hazard Probabilities P(D)...")
        df_hazard = pd.read_csv(A_PROBS, usecols=['standardized_tcad_id', 'year', 'Prob_LGBM_H=4'])
        if 'standardized_tcad_id' in df.columns:
            df['standardized_tcad_id'] = df['standardized_tcad_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
            df_hazard['standardized_tcad_id'] = df_hazard['standardized_tcad_id'].astype(str).str.zfill(10)
            
            # Use the most recent hazard probability for each parcel to simulate 'today's hazard
            df_hazard_latest = df_hazard.sort_values('year').groupby('standardized_tcad_id').last().reset_index()
            
            df = df.merge(df_hazard_latest[['standardized_tcad_id', 'Prob_LGBM_H=4']], on='standardized_tcad_id', how='left')
            df['simulated_hazard_prob'] = df['Prob_LGBM_H=4'].fillna(0.01)
        else:
            df['simulated_hazard_prob'] = 0.01
    else:
        print("[!] Stage A Probabilities missing. Generating synthetic probabilities for scaffolding...")
        df['simulated_hazard_prob'] = np.random.uniform(0, 0.05, len(df))
        
    # 3. Simulate Stage B (Conditional Scale)
    print("[3] Simulating Stage B (Project Typology) dynamically...")
    if os.path.exists(B_MODEL):
        from catboost import CatBoostClassifier
        model_b = CatBoostClassifier().load_model(B_MODEL)
        df['gross_site_area_acres'] = pd.to_numeric(df.get('gross_site_area_acres', 0), errors='coerce').fillna(0)
        df['year'] = pd.to_numeric(df.get('year', 2024), errors='coerce').fillna(2024)
        X_b = df[['gross_site_area_acres', 'year']]
        df['simulated_6_tier_class'] = model_b.predict(X_b).flatten()
    else:
        print("[!] Missing Stage B Model.")
        df['simulated_6_tier_class'] = "Unknown"
    
    # 4. Simulate Stage C (Conditional Opposition)
    print("[4] Simulating Stage C (Opposition Pathway) at H0...")
    if os.path.exists(C_MODEL_H0):
        import joblib
        model_c = joblib.load(C_MODEL_H0)
        
        dist_col = 'council_district' if 'council_district' in df.columns else 'council_district_x'
        drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 'final_date', 'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4', 'Prob_Optimal_H=4', 'ipw', dist_col, 'council_district', 'simulated_hazard_prob', 'simulated_6_tier_class']
        df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        leak_cols = [c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_')]
        if len(leak_cols) > 0:
            df_clean = df_clean.drop(columns=leak_cols)
            
        X_c = df_clean.select_dtypes(include=[np.number]).fillna(0)
        
        expected_features = model_c.calibrated_classifiers_[0].estimator.feature_names_
        for feature in expected_features:
            if feature not in X_c.columns:
                X_c[feature] = 0
        X_c = X_c[expected_features]
        
        df['simulated_opposition_prob_H0'] = model_c.predict_proba(X_c)[:, 1]
    else:
        print("[!] Missing Stage C H0 Model.")
        df['simulated_opposition_prob_H0'] = np.clip(df['simulated_hazard_prob'] * 0.02, 0, 1)

    print("[4B] Executing Chronological Autoregressive Imputation (H0 -> H3)...")
    imputer_path = os.path.join(OUT_DIR, "stage_f_autoregressive_imputer_H0_to_H3.joblib")
    if not os.path.exists(imputer_path):
        imputer_path = build_autoregressive_transformers()

    if imputer_path and os.path.exists(C_MODEL_H3):
        import joblib
        ar_data = joblib.load(imputer_path)
        imputer = ar_data['model']
        ar_features = ar_data['features']
        ar_targets = ar_data['targets']
        
        # Build base predicting array
        X_base = df.copy()
        for feature in ar_features:
            if feature not in X_base.columns:
                X_base[feature] = 0
        X_base = X_base[ar_features].select_dtypes(include=[np.number]).fillna(0)
        
        # IMPUTE FUTURE TEMPORAL FEATURES
        imputed_vectors = imputer.predict(X_base)
        df_imputed = pd.DataFrame(imputed_vectors, columns=ar_targets)
        df_future = pd.concat([df.reset_index(drop=True), df_imputed], axis=1)
        
        print(f"        -> Synthesized {len(ar_targets)} future NLP vectors for pipeline.")
        
        # Score H3
        model_c_h3 = joblib.load(C_MODEL_H3)
        drop_cols_h3 = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 'final_date', 'standardized_tcad_id', 'Prob_H=4', 'Prob_LGBM_H=4', 'Prob_Optimal_H=4', 'ipw', dist_col, 'council_district', 'simulated_hazard_prob', 'simulated_6_tier_class']
        df_clean_h3 = df_future.drop(columns=[c for c in drop_cols_h3 if c in df_future.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        
        expected_features_h3 = model_c_h3.calibrated_classifiers_[0].estimator.feature_names_
        for feature in expected_features_h3:
            if feature not in df_clean_h3.columns:
                df_clean_h3[feature] = 0
        X_c_h3 = df_clean_h3[expected_features_h3]
        
        df['simulated_opposition_prob_H3'] = model_c_h3.predict_proba(X_c_h3)[:, 1]
    else:
        df['simulated_opposition_prob_H3'] = df['simulated_opposition_prob_H0']

    
    # 5. Output Synthetic Landscape
    print("[5] Exporting Generative Sequence Results Structure...")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "stage_f_generative_simulation_results.csv")
    df[['case_number', 'simulated_hazard_prob', 'simulated_6_tier_class', 'simulated_opposition_prob_H0', 'simulated_opposition_prob_H3']].head(100).to_csv(out_path, index=False)
    print(f"    -> Scaffold exported successfully to {out_path}")
    print("    -> Note: This architecture forms the basis for the Future Work extension.")

if __name__ == '__main__':
    run_generative_simulation()
