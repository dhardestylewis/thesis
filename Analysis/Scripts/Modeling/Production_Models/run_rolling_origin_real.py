import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier

# Environment Setup
ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data", "Warehouse_As_Of")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Track1_Predictive")
os.makedirs(OUT_DIR, exist_ok=True)

def run_rolling_origin_drift():
    print("Executing Real Rolling-Origin Temporal Drift Analysis...")
    master_path = os.path.join(DATA, 'H0_Filing_Master_Enriched.csv')
    df = pd.read_csv(master_path, low_memory=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year', 'is_protested']).sort_values('year')
    
    # Feature list (Consistent with H0 Filing logic)
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
    # Explicitly strip future features to prevent H0 leakage
    future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
    X = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y = df['is_protested'].values
    years = df['year'].values

    anchors = [2018, 2019, 2020, 2021, 2022]
    offsets = [0, 1, 2, 3] # T+0 to T+3
    
    drift_results = []
    
    for anchor in anchors:
        print(f"\n[+] Processing Anchor: Pre-{anchor} Training Window")
        train_mask = years < anchor
        if train_mask.sum() < 50: continue
        
        X_train, y_train = X.values[train_mask], y[train_mask]
        
        # Fit once per anchor
        model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0, auto_class_weights='Balanced', random_seed=42)
        model.fit(X_train, y_train)
        
        for offset in offsets:
            test_year = anchor + offset
            test_mask = years == test_year
            
            if test_mask.sum() < 5 or y[test_mask].sum() < 1:
                prauc = np.nan
            else:
                X_test, y_test = X.values[test_mask], y[test_mask]
                preds = model.predict_proba(X_test)[:, 1]
                prauc = average_precision_score(y_test, preds)
            
            drift_results.append({
                'Anchor': f"Pre-{anchor}",
                'Evaluate_Year': test_year,
                'Offset': f"T+{offset}",
                'PR-AUC': round(prauc, 4) if not np.isnan(prauc) else None
            })
            print(f"    -> Evaluated on {test_year} (T+{offset}): PR-AUC = {prauc:.4f}")

    # Generate Pivot Table for LaTeX
    results_df = pd.DataFrame(drift_results)
    pivot = results_df.pivot(index='Anchor', columns='Offset', values='PR-AUC')
    
    # Save to JSON
    results_df.to_json(os.path.join(OUT_DIR, "rolling_origin_drift.json"), orient='records', indent=2)
    
    # Generate LaTeX Table
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Temporal Predictive Drift: H0 Filing Performance Decay}}",
        r"\label{tab:temporal_drift}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Anchor Training} & \textbf{T+0 (Same Year)} & \textbf{T+1} & \textbf{T+2} & \textbf{T+3} \\",
        r"\midrule"
    ]
    
    for idx, row in pivot.iterrows():
        r = [f"{row[off]:.3f}" if row[off] is not None else "---" for off in ['T+0', 'T+1', 'T+2', 'T+3']]
        tex_lines.append(f"{idx} & {' & '.join(r)} \\\\")
        
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    tex_path = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Tables", "temporal_drift_analysis.tex")
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines))
    print(f"\n[+] Temporal Drift LaTeX Table saved to {tex_path}")

if __name__ == "__main__":
    run_rolling_origin_drift()
