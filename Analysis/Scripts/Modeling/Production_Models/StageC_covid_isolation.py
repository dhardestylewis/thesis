import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
import shap

PATH = r"C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv"

def evaluate_covid_shock():
    print("Loading data...")
    df = pd.read_csv(PATH, low_memory=False)
    
    # Isolate Covid Shock (2021-2022)
    covid_df = df[(df['year'] >= 2021) & (df['year'] <= 2022)].copy()
    print(f"Isolated {len(covid_df)} cases from 2021-2022.")
    
    if len(covid_df) == 0:
        print("No cases found in this range!")
        return

    # Clean Features
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 
                'TCAD ID', 'date', 'application_start_date', 'final_date', 'standardized_tcad_id', 
                'Prob_H=4', 'Prob_LGBM_H=4', 'Prob_CB_H=4', 'Prob_Optimal_H=4', 'ipw', 'council_district']
    
    df_clean = covid_df.drop(columns=[c for c in drop_cols if c in covid_df.columns], errors='ignore')
    
    # Strip Temporal Leakage (NLP vectors)
    leak_cols = [c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_')]
    df_clean = df_clean.drop(columns=leak_cols)
    
    X_raw = df_clean.select_dtypes(include=[np.number]).fillna(0)
    y = covid_df['is_protested'].fillna(0).astype(int)
    
    print(f"Training on {X_raw.shape[1]} features.")
    
    # K-Fold CV specifically on 2021-2022
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pr_aucs = []
    
    # Setup for SHAP
    final_model = CatBoostClassifier(iterations=250, auto_class_weights='Balanced', depth=6, random_seed=42, verbose=0)
    
    for train_idx, test_idx in kf.split(X_raw):
        X_tr, X_te = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model = CatBoostClassifier(iterations=250, auto_class_weights='Balanced', depth=6, random_seed=42, verbose=0)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_te)[:, 1]
        
        pr = average_precision_score(y_te, preds)
        pr_aucs.append(pr)
        
    print(f"COVID-Regime Internal Cross-Validation PR-AUC: {np.mean(pr_aucs):.4f}")
    
    # Now train on full COVID dataset for SHAP to understand what features dominate this regime
    final_model.fit(X_raw, y)
    
    explainer = shap.TreeExplainer(final_model)
    shap_vals = explainer.shap_values(X_raw)
    
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({'Feature': X_raw.columns, 'Mean_Abs_SHAP': mean_abs_shap})
    shap_df = shap_df.sort_values(by='Mean_Abs_SHAP', ascending=False).head(20)
    
    print("\n--- TOP 20 FEATURES EXCLUSIVE TO THE COVID SHOCK REGIME ---")
    print(shap_df.to_string(index=False))

if __name__ == '__main__':
    evaluate_covid_shock()
