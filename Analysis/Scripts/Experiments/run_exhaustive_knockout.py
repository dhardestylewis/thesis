import os
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

print("\n" + "="*70)
print(" EXHAUSTIVE DROP-ONE FEATURE KNOCKOUT AUDIT (CatBoost Baseline)")
print("="*70)

# Path Setup
ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA_FILE = os.path.join(ROOT, 'Data', 'Warehouse_As_Of', 'H0_Filing_Master_Enriched.csv')

OUTPUT_VAR = 'is_protested'

def main():
    print("[1] Loading and cleaning dataset...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df = df.dropna(subset=['year'])
    df['is_protested'] = pd.to_numeric(df['is_protested'], errors='coerce')
    df = df.dropna(subset=['is_protested'])
    df[OUTPUT_VAR] = df['is_protested'].astype(int)

    # Exclude leakage / identifier columns logically
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 
                 'has_audio_record', 'TCAD ID', 'standardized_tcad_id', 
                 'date', 'application_start_date', 'final_date', 
                 'target', 'Target_Opposition_H0']
    
    # Filter text / NLP vectors out for the structured baseline test
    leak_cols = [c for c in df.columns if c.startswith('tfidf_') or c.startswith('speech_') or c in drop_cols]
    
    X = df.drop(columns=[c for c in leak_cols if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
    X = X.loc[:, X.nunique() > 1]  # Drop constants
    X = X.fillna(X.mean())         # Basic imputation structural consistency
    y = df[OUTPUT_VAR]
    
    features = list(X.columns)
    print(f"[+] Operational Shape: {X.shape[0]} Cases, {len(features)} Features")
    
    # 2-Fold Stratified CV
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    
    print("\n[2] Computing Baseline PR-AUC Metrics...")
    baseline_preds = np.zeros(len(y))
    
    for train_idx, test_idx in cv.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        
        baseline_model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
        baseline_model.fit(X_tr, y_tr)
        baseline_preds[test_idx] = baseline_model.predict_proba(X_te)[:, 1]
        
    baseline_prauc = average_precision_score(y, baseline_preds)
    print(f"    Baseline PR-AUC: {baseline_prauc:.4f}")
    
    print("\n[3] Executing Drop-One Feature Ablation Loop...")
    print(f"{'Ablated Feature':<40} | {'Knockout PR-AUC':>15} | {'Delta Vs Baseline':>15}")
    print("-" * 78)
    
    results = []
    
    # Iteratively knock out ONE feature at a time
    for feat in features:
        X_knockout = X.drop(columns=[feat])
        ko_preds = np.zeros(len(y))
        
        for train_idx, test_idx in cv.split(X_knockout, y):
            X_tr, y_tr = X_knockout.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X_knockout.iloc[test_idx], y.iloc[test_idx]
            
            ko_model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
            ko_model.fit(X_tr, y_tr)
            ko_preds[test_idx] = ko_model.predict_proba(X_te)[:, 1]
            
        ko_prauc = average_precision_score(y, ko_preds)
        delta = ko_prauc - baseline_prauc
        results.append({'Feature': feat, 'Knockout_PRAUC': ko_prauc, 'Delta': delta})
        
        # Highlight in Red (or just text) if the model gets BETTER (meaning feature was harmful)
        if delta > 0.001:
            print(f"{feat:<40} | {ko_prauc:>15.4f} | +{delta:>14.4f}  [HARMFUL]", flush=True)
        else:
            print(f"{feat:<40} | {ko_prauc:>15.4f} | {delta:>15.4f}", flush=True)

    print("-" * 78)
    
    # Identify universally harmful features
    harmful_features = sorted([r for r in results if r['Delta'] > 0], key=lambda x: x['Delta'], reverse=True)
    if harmful_features:
        print(f"\n[!] ALERT: Found {len(harmful_features)} explicitly harmful features (removing them improved PR-AUC):")
        for h in harmful_features:
            print(f"    - {h['Feature']}: +{h['Delta']:.4f} PR-AUC when removed")
    else:
        print("\n[+] SUCCESS: No highly harmful features identified. All features structurally contribute monotonically or are noise-neutral.")

if __name__ == "__main__":
    main()
