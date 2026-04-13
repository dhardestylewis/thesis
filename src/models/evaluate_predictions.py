import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

# src/models/evaluate_predictions.py
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from data_io.schema import ROOT_DIR, REGISTRY_DIR

def evaluate_predictions():
    print("[+] Executing Formal Evaluation Suite...")
    
    # Load Prediction Registry
    preds_path = REGISTRY_DIR / "prediction_registry.parquet"
    if not preds_path.exists():
        print("    [!] Error: No predictions found.")
        return
        
    df = pd.read_parquet(preds_path)
    
    # We evaluate for the main task: CatBoost on TEMP_OOD_2023_MAIN
    subset = df[(df['model_family'] == 'CatBoost') & (df['split_id'] == 'TEMP_OOD_2023_MAIN')]
    
    if subset.empty:
        print("    [!] No results found for CatBoost/OOD_2023.")
        return
        
    y_true = subset['y_true']
    y_score = subset['y_score_raw']
    
    # Path 1: Ranking
    prauc = average_precision_score(y_true, y_score)
    
    # Path 2: Calibration
    brier = brier_score_loss(y_true, y_score)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred)) # Simplied ECE
    
    # Path 3: Thresholded (at 0.30 and 0.50)
    from sklearn.metrics import precision_score, recall_score
    prec30 = precision_score(y_true, y_score >= 0.30)
    rec30 = recall_score(y_true, y_score >= 0.30)
    prec50 = precision_score(y_true, y_score >= 0.50)
    rec50 = recall_score(y_true, y_score >= 0.50)
    
    results = {
        'ranking': {
            'pr_auc': prauc
        },
        'calibration': {
            'brier': brier,
            'ece': ece
        },
        'thresholded': {
            'precision_30': prec30,
            'recall_30': rec30,
            'precision_50': prec50,
            'recall_50': rec50
        }
    }
    
    # Save Machine-Readable Metrics
    with open(REGISTRY_DIR / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n>>> EVALUATION PATHS (CatBoost OOD) <<<")
    print(f"Path 1 (Ranking):     PR-AUC = {prauc:.3f}")
    print(f"Path 2 (Calibration): Brier = {brier:.3f}, ECE = {ece:.3f}")
    print(f"Path 3 (Threshold):   P@0.30 = {prec30:.2f}, R@0.30 = {rec30:.2f}")

if __name__ == "__main__":
    evaluate_predictions()
