import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve
import json

# Paths
ROOT = Path(r"c:\Users\dhl\data\thesis\thesis")
PIPELINE_DATA = ROOT / "thesis_pipeline" / "data" / "final"

def run_diagnostics():
    print("[+] Running Canonical Stage C Diagnostics...")
    
    # Load predictions
    preds = pd.read_parquet(PIPELINE_DATA / "prediction_registry.parquet")
    
    # We'll evaluate the core CatBoost model
    core_preds = preds[preds['model_family'] == 'CatBoost']
    y_true = core_preds['y_true']
    y_prob = core_preds['y_prob']
    
    # --- PATH 1: RANKING ---
    print("    [~] Path 1: Ranking Evaluation...")
    ranking = {
        'pr_auc': average_precision_score(y_true, y_prob),
        'top_decile_lift': compute_lift(y_true, y_prob, 0.1)
    }
    
    # --- PATH 2: CALIBRATION ---
    print("    [~] Path 2: Calibration Evaluation...")
    calibration = {
        'brier_score': brier_score_loss(y_true, y_prob),
        'calibration_slope': compute_calibration_slope(y_true, y_prob),
        'ece': compute_ece(y_true, y_prob)
    }
    
    # --- PATH 3: THRESHOLDED DIAGNOSTICS ---
    print("    [~] Path 3: Thresholded Evaluation (Budget Analysis)...")
    thresholds = [0.05, 0.1, 0.2] # Top % of cases budgeted for review
    threshold_metrics = []
    
    for k in thresholds:
        metrics = evaluate_at_budget(y_true, y_prob, k)
        threshold_metrics.append({'budget': k, **metrics})
        
    # Combine into a single diagnostic object
    diagnostics = {
        'split_id': 'TEMP_OOD_2023_MAIN',
        'model_family': 'CatBoost',
        'ranking': ranking,
        'calibration': calibration,
        'thresholded_diagnostics': threshold_metrics
    }
    
    with open(PIPELINE_DATA / "canonical_diagnostics.json", 'w') as f:
        json.dump(diagnostics, f, indent=4)
        
    print("\n>>> CANONICAL DIAGNOSTIC SUMMARY <<<")
    print(f"  PR-AUC:           {ranking['pr_auc']:.3f}")
    print(f"  Calibration Slope: {calibration['calibration_slope']:.3f}")
    print(f"  Top-Decile Prec:   {threshold_metrics[1]['precision']:.1%}")

def compute_lift(y_true, y_prob, k):
    df = pd.DataFrame({'y': y_true, 'p': y_prob}).sort_values('p', ascending=False)
    n = int(len(df) * k)
    hit_rate = df.head(n)['y'].mean()
    base_rate = df['y'].mean()
    return hit_rate / base_rate if base_rate > 0 else 0

def compute_calibration_slope(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    if len(prob_true) < 2: return 1.0
    z = np.polyfit(prob_pred, prob_true, 1)
    return z[0]

def compute_ece(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return np.mean(np.abs(prob_true - prob_pred))

def evaluate_at_budget(y_true, y_prob, k):
    df = pd.DataFrame({'y': y_true, 'p': y_prob}).sort_values('p', ascending=False)
    n = int(len(df) * k)
    threshold = df.iloc[n]['p']
    preds = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'threshold': threshold,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
    }

if __name__ == "__main__":
    run_diagnostics()
