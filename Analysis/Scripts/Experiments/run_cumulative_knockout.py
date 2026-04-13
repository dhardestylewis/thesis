import os
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

print("\n" + "="*70)
print(" CUMULATIVE KNOCKOUT AUDIT (Testing Interference vs Accumulation)")
print("="*70)

DATA_FILE = r'C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv'
OUTPUT_VAR = 'is_protested'

# The ranked offenders from the single Drop-One run
OFFENDERS = [
    'delta_imprv_sqft_max',
    'land_market_value',
    'latitude',
    'delta_max_far',
    'improvement_market_value',
    'second_most_recent_sale_date',
    'appraised_value',
    'land_acres',
    'most_recent_sale_date',
    'tax_year',
    'abs_subdiv_cd',
    'owner_state'
]

def main():
    print("[1] Loading Data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df = df.dropna(subset=['year'])
    df['is_protested'] = pd.to_numeric(df['is_protested'], errors='coerce')
    df = df.dropna(subset=['is_protested'])
    df[OUTPUT_VAR] = df['is_protested'].astype(int)

    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 
                 'has_audio_record', 'TCAD ID', 'standardized_tcad_id', 
                 'date', 'application_start_date', 'final_date', 
                 'target', 'Target_Opposition_H0']
    
    leak_cols = [c for c in df.columns if c.startswith('tfidf_') or c.startswith('speech_') or c in drop_cols]
    
    X_full = df.drop(columns=[c for c in leak_cols if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
    X_full = X_full.loc[:, X_full.nunique() > 1]
    X_full = X_full.fillna(X_full.mean())
    y = df[OUTPUT_VAR]
    
    # 2-Fold CV for speed
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    
    # Helper to evaluate PR-AUC
    def evaluate_model(X_data):
        preds = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X_data, y):
            X_tr, y_tr = X_data.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X_data.iloc[test_idx], y.iloc[test_idx]
            model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
            model.fit(X_tr, y_tr)
            preds[test_idx] = model.predict_proba(X_te)[:, 1]
        return average_precision_score(y, preds)

    print("\n[2] Computing Baseline...")
    baseline_prauc = evaluate_model(X_full)
    print(f"    Baseline PR-AUC: {baseline_prauc:.4f}")

    print("\n[3] Executing Cumulative Knockout (Dropping progressively)...")
    print(f"{'Dropped Cumulatively':<45} | {'PR-AUC':>10} | {'Delta vs Base':>15}")
    print("-" * 75)

    current_X = X_full.copy()
    
    for i, feature in enumerate(OFFENDERS):
        if feature in current_X.columns:
            current_X = current_X.drop(columns=[feature])
            prauc = evaluate_model(current_X)
            delta = prauc - baseline_prauc
            
            sign = "+" if delta > 0 else ""
            print(f"Step {i+1:<2} (- {feature:<31}) | {prauc:>10.4f} | {sign}{delta:>14.4f}", flush=True)

    print("=" * 75)

if __name__ == "__main__":
    main()
