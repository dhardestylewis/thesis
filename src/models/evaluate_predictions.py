import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score
from sklearn.calibration import calibration_curve

# src/models/evaluate_predictions.py
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data_io.schema import ROOT_DIR, REGISTRY_DIR

_N_BOOTSTRAP = 2000
_BOOTSTRAP_SEED = 12345


def _bootstrap_prauc_ci(y_true, y_score, n_boot=_N_BOOTSTRAP, seed=_BOOTSTRAP_SEED, alpha=0.05):
    """Compute bootstrap 95% CI for PR-AUC by resampling the evaluation set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() >= 2:
            scores.append(average_precision_score(yt, ys))
    scores = np.asarray(scores)
    return float(np.percentile(scores, 100 * alpha / 2)), float(np.percentile(scores, 100 * (1 - alpha / 2)))


def _ece(y_true, y_prob, n_bins=10):
    """10-bin uniform ECE."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return float(np.mean(np.abs(prob_true - prob_pred)))


def evaluate_predictions():
    print("[+] Executing Formal Evaluation Suite...")

    preds_path = REGISTRY_DIR / "prediction_registry.parquet"
    if not preds_path.exists():
        print("    [!] Error: No predictions found.")
        return

    df = pd.read_parquet(preds_path)

    # Primary task: CatBoost on TEMP_OOD_2023_MAIN
    subset = df[(df['model_family'] == 'CatBoost') & (df['split_id'] == 'TEMP_OOD_2023_MAIN')]
    if subset.empty:
        print("    [!] No results found for CatBoost/OOD_2023.")
        return

    y_true = subset['y_true'].values
    y_score_raw = subset['y_score_raw'].values
    # Use properly calibrated scores when available; fall back to raw
    y_score_cal = (
        subset['y_score_calibrated'].values
        if 'y_score_calibrated' in subset.columns
        and subset['calibration_method'].iloc[0] == 'isotonic_oof'
        else y_score_raw
    )

    # ── Path 1: Ranking (OOD point estimate + bootstrap CI) ─────────────────
    prauc = float(average_precision_score(y_true, y_score_raw))
    ci_low, ci_high = _bootstrap_prauc_ci(y_true, y_score_raw)

    # ── Path 2: Calibration (raw and post-calibration ECE) ──────────────────
    brier = float(brier_score_loss(y_true, y_score_cal))
    ece_raw = _ece(y_true, y_score_raw)
    ece_calibrated = _ece(y_true, y_score_cal)

    # ── Path 3: Thresholded (at 0.30 and 0.50) ──────────────────────────────
    prec30 = float(precision_score(y_true, y_score_raw >= 0.30, zero_division=0))
    rec30  = float(recall_score(y_true, y_score_raw >= 0.30, zero_division=0))
    prec50 = float(precision_score(y_true, y_score_raw >= 0.50, zero_division=0))
    rec50  = float(recall_score(y_true, y_score_raw >= 0.50, zero_division=0))

    results = {
        'ranking': {
            'pr_auc': prauc,
            'pr_auc_ci_low': ci_low,
            'pr_auc_ci_high': ci_high,
            'n_test': int(len(y_true)),
            'n_test_positive': int(y_true.sum()),
        },
        'calibration': {
            'brier': brier,
            'ece_raw': ece_raw,
            'ece': ece_calibrated,         # post-calibration ECE (C1 layer)
            'ece_pre_calibration': ece_raw,
        },
        'thresholded': {
            'precision_30': prec30,
            'recall_30': rec30,
            'precision_50': prec50,
            'recall_50': rec50,
        },
    }

    out_path = REGISTRY_DIR / "evaluation_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n>>> EVALUATION PATHS (CatBoost OOD, n_test={len(y_true)}, n_pos={int(y_true.sum())}) <<<")
    print(f"Path 1 (Ranking):     PR-AUC = {prauc:.3f}  95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"Path 2 (Calibration): Brier = {brier:.3f}, ECE_raw = {ece_raw:.3f}, ECE_calibrated = {ece_calibrated:.3f}")
    print(f"Path 3 (Threshold):   P@0.30 = {prec30:.2f}, R@0.30 = {rec30:.2f}")


if __name__ == "__main__":
    evaluate_predictions()
