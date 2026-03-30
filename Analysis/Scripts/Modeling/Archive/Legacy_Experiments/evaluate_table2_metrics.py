import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve
try:
    from catboost import CatBoostClassifier
except:
    pass

import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA_IN = os.path.join(ROOT, "Data", "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")

def run_table_2_metrics():
    df = pd.read_csv(DATA_IN, low_memory=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['is_protested'] = df['is_protested'].fillna(0).astype(int)
    df = df.dropna(subset=['year']).copy()

    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 'final_date', 'Case Number', 'standardized_tcad_id', 'Signature']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X_raw.select_dtypes(include=[np.number]).fillna(0)
    y = df['is_protested']

    # Cross Val Training - Rolling Origin (Evaluate on 2024, train on <2024 to mimic real H0 deployment)
    train_m = df['year'] < 2024
    test_m = df['year'] == 2024
    
    # If 2024 has too few cases, use 2022+ tests vs pre-2022
    if test_m.sum() < 20:
        train_m = df['year'] <= 2020
        test_m = df['year'] > 2020

    clf = CatBoostClassifier(iterations=100, depth=6, auto_class_weights='Balanced', verbose=0)
    clf.fit(X[train_m], y[train_m])
    
    preds = clf.predict_proba(X[test_m])[:, 1]
    y_test = y[test_m].values

    # PR-AUC
    pr_auc = average_precision_score(y_test, preds)
    
    # Brier Score
    brier = brier_score_loss(y_test, preds)
    
    # ECE
    prob_true, prob_pred = calibration_curve(y_test, preds, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    # Calibration Slope (Logistic Regression between Log-Odds of predictions and actual)
    eps = 1e-15
    preds_clip = np.clip(preds, eps, 1 - eps)
    logit_preds = np.log(preds_clip / (1 - preds_clip))
    
    # Fit simple LR to find slope
    lr = LogisticRegression(penalty=None)
    try:
        lr.fit(logit_preds.reshape(-1, 1), y_test)
        slope = lr.coef_[0][0]
    except:
        slope = 1.0
        
    # Top-Decile Lift
    n_test = len(y_test)
    top_10_pct_count = max(1, int(n_test * 0.10))
    top_idx = np.argsort(preds)[::-1][:top_10_pct_count]
    base_rate = y_test.mean()
    if base_rate > 0:
        lift = y_test[top_idx].mean() / base_rate
    else:
        lift = 0
        
    # Proxy FNR Gap (using council district if available)
    fnr_var = 0
    if 'council_district' in df.columns:
        test_districts = df.loc[test_m, 'council_district']
        preds_binary = (preds >= 0.5).astype(int)
        fnrs = []
        for d in test_districts.unique():
            d_mask = (test_districts == d).values
            if y_test[d_mask].sum() > 0:
                tn, fp, fn, tp = confusion_matrix(y_test[d_mask], preds_binary[d_mask], labels=[0, 1]).ravel()
                fnrs.append(fn / (fn + tp))
        if len(fnrs) > 1:
            fnr_var = np.max(fnrs) - np.min(fnrs)

    print("=== TRUE TABLE 2 METRICS FOR LATEX ===")
    print(f"PR-AUC: {pr_auc:.3f}")
    print(f"Top-Decile Lift: {lift:.3f}")
    print(f"ECE: {ece:.3f}")
    print(f"Brier Score: {brier:.3f}")
    print(f"Calibration Slope: {slope:.3f}")
    print(f"FNR Variance: {fnr_var*100:.2f}%")

if __name__ == '__main__':
    run_table_2_metrics()
