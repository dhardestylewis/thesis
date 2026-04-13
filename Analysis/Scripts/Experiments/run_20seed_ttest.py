import os
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*75)
print(" 20-SEED PAIRED T-TEST: STATISTICAL SIGNIFICANCE OF FEATURE ABLATION")
print("="*75)

DATA_FILE = r'C:\Users\dhl\data\thesis\thesis\Data\Warehouse_As_Of\H0_Filing_Master_Enriched.csv'
OUTPUT_VAR = 'is_protested'

# The identified highly harmful targets we want to statistically validate
TARGET_OFFENDERS = ['latitude', 'land_market_value', 'delta_imprv_sqft_max']

def main():
    print("[1] Loading Operational Dataset...")
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
    
    # We drop the targets altogether to create the "Ablated" matrix
    X_ablated = X_full.drop(columns=[c for c in TARGET_OFFENDERS if c in X_full.columns])

    print(f"\n[2] Executing 20-Seed Paired Training (Full vs Ablated)...")
    print(f"{'Seed':<6} | {'Baseline PR-AUC':>15} | {'Ablated PR-AUC':>15} | {'Delta':>10}")
    print("-" * 55)

    base_scores = []
    ablated_scores = []
    
    # Run 20 explicit seed iterations (mirroring your thesis methodology)
    for seed in range(42, 62):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        
        base_preds = np.zeros(len(y))
        ab_preds = np.zeros(len(y))
        
        for train_idx, test_idx in cv.split(X_full, y):
            # Baseline
            X_tr_b, y_tr = X_full.iloc[train_idx], y.iloc[train_idx]
            X_te_b = X_full.iloc[test_idx]
            m_base = CatBoostClassifier(iterations=250, depth=6, verbose=0, random_seed=seed)
            m_base.fit(X_tr_b, y_tr)
            base_preds[test_idx] = m_base.predict_proba(X_te_b)[:, 1]
            
            # Ablated
            X_tr_a = X_ablated.iloc[train_idx]
            X_te_a = X_ablated.iloc[test_idx]
            m_ab = CatBoostClassifier(iterations=250, depth=6, verbose=0, random_seed=seed)
            m_ab.fit(X_tr_a, y_tr)
            ab_preds[test_idx] = m_ab.predict_proba(X_te_a)[:, 1]
            
        b_pr = average_precision_score(y, base_preds)
        a_pr = average_precision_score(y, ab_preds)
        
        base_scores.append(b_pr)
        ablated_scores.append(a_pr)
        
        print(f"{seed:<6} | {b_pr:>15.4f} | {a_pr:>15.4f} | {a_pr - b_pr:>10.4f}")

    print("-" * 55)
    delta_array = np.array(ablated_scores) - np.array(base_scores)
    
    # Paired Student's t-test
    t_stat, p_val = stats.ttest_rel(ablated_scores, base_scores)
    
    print("\n[3] Statistical Significance Results:")
    print(f"    Mean PR-AUC Baseline: {np.mean(base_scores):.4f}")
    print(f"    Mean PR-AUC Ablated:  {np.mean(ablated_scores):.4f}")
    print(f"    Mean Delta Jump:      +{np.mean(delta_array):.4f}")
    print(f"    T-Statistic:          {t_stat:.4f}")
    print(f"    P-Value:              {p_val:.5f}")
    
    if p_val < 0.05 and np.mean(delta_array) > 0:
        print("\n[CONCLUSION] SIGNIFICANT STRUCTURAL HARM PROVEN (p < 0.05).")
        print("These features systematically degrade performance across random seeds. Removing them is statistically validated.")
    else:
        print("\n[CONCLUSION] NO SIGNIFICANT HARM. The jump was primarily seed-dependent noise.")

if __name__ == "__main__":
    main()
