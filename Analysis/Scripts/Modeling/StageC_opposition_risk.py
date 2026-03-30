import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier
except ImportError:
    pass

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Track1_Predictive")
os.makedirs(OUT_DIR, exist_ok=True)

def compute_ece(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    if len(prob_true) == 0: return 0
    return np.mean(np.abs(prob_true - prob_pred))

def extract_metrics(y_true, y_pred, name=""):
    if len(np.unique(y_true)) < 2:
        return {'Model': name, 'PR-AUC': np.nan, 'ROC-AUC': np.nan, 'Brier': np.nan, 'ECE': np.nan}
    return {
        'Model': name,
        'PR-AUC': average_precision_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred),
        'Brier': brier_score_loss(y_true, y_pred),
        'ECE': compute_ece(y_true, y_pred)
    }

def process_horizon(path, horizon_name):
    print(f"\n==============================================")
    print(f" HORIZON MULTI-TRACK EXECUTION: {horizon_name} ")
    print(f"==============================================")
    
    df = pd.read_csv(path, low_memory=False)
    
    # Strictly map to the 'year' integer to bypass missing exact application dates
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    else:
        print("[!] No temporal origin found. Exiting.")
        return
        
    df['is_protested'] = df['is_protested'].fillna(0).astype(int)
    df = df.dropna(subset=['year']).sort_values('year').copy()
    
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 'TCAD ID', 'date', 'application_start_date', 'final_date']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X_raw.select_dtypes(include=[np.number]).fillna(0)
    y = df['is_protested']
    
    print(f"Total Structural Chronology: {len(X)} cases mapped across {X.shape[1]} dimensional vectors.")
    print("----------------------------------------------")
    
    # ---------------------------------------------------------
    # PART A: TEMPORAL DRIFT MULTI-HORIZON (1, 2, 3 Years Out)
    # ---------------------------------------------------------
    print("\nPART A: TEMPORAL DRIFT (MULTI-HORIZON ROT)")
    print("Training on < Anchor, evaluating degradation across T+0, T+1, T+2...")
    drift_results = []
    
    for anchor_year in [2019, 2020, 2021, 2022]:
        train_mask = df['year'] < anchor_year
        if train_mask.sum() < 20:
            continue
            
        X_train, y_train = X[train_mask], y[train_mask]
        
        # Train once per anchor
        cb = CatBoostClassifier(iterations=150, verbose=0)
        cb.fit(X_train, y_train)
        
        for offset in [0, 1, 2, 3]:
            test_year = anchor_year + offset
            if test_year > 2024: continue
            
            test_mask = df['year'] == test_year
            if test_mask.sum() < 2: continue
            
            X_test, y_test = X[test_mask], y[test_mask]
            if y_test.sum() < 1: continue
            
            preds = cb.predict_proba(X_test)[:, 1]
            
            mets = extract_metrics(y_test, preds, f"Anchor <{anchor_year} -> Test {test_year} (+{offset+1}yr)")
            mets['Cases'] = int(test_mask.sum())
            drift_results.append(mets)

    if drift_results:
        print(pd.DataFrame(drift_results).to_string(index=False))
        
    # ---------------------------------------------------------
    # PART B: Explicit Policy-Regime Holdouts (Section 6.B)
    # ---------------------------------------------------------
    print("\nPART B: LEGISLATIVE POLICY REGIME HOLDOUTS (WORST-GROUP OOD)")
    # Train strictly BEFORE legislative year -> Test strictly WITHIN legislative year
    # Relying on 'year' granularity
    
    regimes = [
        {"name": "Pre-2022 Validation", "train_bound": 2021, "test_start": 2021, "test_end": 2021},
        {"name": "2022 Transition", "train_bound": 2022, "test_start": 2022, "test_end": 2022},
        {"name": "HOME Adoption (2024)", "train_bound": 2024, "test_start": 2024, "test_end": 2026}
    ]
    
    regime_results = []
    
    for reg in regimes:
        t_bound = reg['train_bound']
        s_bound = reg['test_start']
        e_bound = reg['test_end']
        
        train_mask = df['year'] < t_bound
        test_mask = (df['year'] >= s_bound) & (df['year'] <= e_bound)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        if test_mask.sum() < 2 or y_test.sum() < 1:
            regime_results.append({'Model': f"CatBoost {reg['name']}", 'PR-AUC': np.nan, 'ROC-AUC': np.nan, 'Brier': np.nan, 'ECE': np.nan, 'Cases': test_mask.sum()})
            continue
            
        cb = CatBoostClassifier(iterations=150, verbose=0)
        cb.fit(X_train, y_train)
        preds = cb.predict_proba(X_test)[:, 1]
        
        mets = extract_metrics(y_test, preds, f"CatBoost {reg['name']}")
        mets['Cases'] = int(test_mask.sum())
        regime_results.append(mets)
        
    if regime_results:
        print(pd.DataFrame(regime_results).to_string(index=False))
    print("==============================================\n")

def run_track1():
    print("Initiating Master Multi-Horizon Structural Engine...")
    
    # Executing for all available multi-modal matrices 
    horizons = {
        'H0 (Filing Baseline)': 'H0_Filing_Master_Enriched.csv',
        'H3 (Pre-Council with NLP)': 'H3_Filing_Master_NLP.csv'
    }
    
    for name, filename in horizons.items():
        path = os.path.join(DATA, filename)
        if os.path.exists(path):
            process_horizon(path, name)
        else:
            print(f"[!] Warning: Data pipeline horizon {filename} not yet constructed.")
            
    print("Evaluation Cycle Exhausted.")

if __name__ == '__main__':
    run_track1()
