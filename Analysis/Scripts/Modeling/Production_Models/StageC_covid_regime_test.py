import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

PATH = r"C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv"

def test_covid_regime_exclusivity():
    df = pd.read_csv(PATH, low_memory=False)
    
    # Feature Engineering
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 
                'TCAD ID', 'date', 'application_start_date', 'final_date', 'standardized_tcad_id', 
                'Prob_H=4', 'Prob_LGBM_H=4', 'Prob_CB_H=4', 'Prob_Optimal_H=4', 'ipw', 'council_district']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    leak_cols = [c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_')]
    df_clean = df_clean.drop(columns=leak_cols)
    
    X_raw = df_clean.select_dtypes(include=[np.number]).fillna(0)
    y = df['is_protested'].fillna(0).astype(int)
    
    # Define Regimes
    mask_pre = df['year'] <= 2019
    mask_covid = (df['year'] >= 2021) & (df['year'] <= 2022)
    mask_post = df['year'] >= 2023
    
    print("\n--- TRAINING ON COVID SHOCK REGIME (2021-2022) ---")
    model = CatBoostClassifier(iterations=300, auto_class_weights='Balanced', depth=6, random_seed=42, verbose=0)
    
    X_cov_tr = X_raw[mask_covid]
    y_cov_tr = y[mask_covid]
    
    # Let's train on 80% of covid, test on 20% covid, and test on 100% pre/post
    split_idx = int(len(X_cov_tr) * 0.8)
    X_cov_train, X_cov_test = X_cov_tr.iloc[:split_idx], X_cov_tr.iloc[split_idx:]
    y_cov_train, y_cov_test = y_cov_tr.iloc[:split_idx], y_cov_tr.iloc[split_idx:]
    
    model.fit(X_cov_train, y_cov_train)
    
    # Evaluate In-Distribution (COVID)
    p_covid = model.predict_proba(X_cov_test)[:, 1]
    if y_cov_test.sum() > 0:
        pr_covid = average_precision_score(y_cov_test, p_covid)
        print(f"Test on COVID holdout (In-Distribution): {pr_covid:.4f}")
    
    # Evaluate Out-Of-Distribution (Pre-COVID)
    p_pre = model.predict_proba(X_raw[mask_pre])[:, 1]
    pr_pre = average_precision_score(y[mask_pre], p_pre)
    print(f"Test on Pre-2020 ERA (Out-Of-Distribution): {pr_pre:.4f}")
    
    # Evaluate Out-Of-Distribution (Post-COVID)
    p_post = model.predict_proba(X_raw[mask_post])[:, 1]
    pr_post = average_precision_score(y[mask_post], p_post)
    print(f"Test on Post-2023 ERA (Out-Of-Distribution): {pr_post:.4f}")

if __name__ == '__main__':
    test_covid_regime_exclusivity()
