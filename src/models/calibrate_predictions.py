import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

# src/models/calibrate_predictions.py
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data_io.schema import ROOT_DIR, REGISTRY_DIR, save_registry

_N_FOLDS = 5
_CALIB_SEED = 42


def calibrate_predictions():
    """Apply out-of-fold isotonic regression calibration to the held-out test predictions.

    Strategy
    --------
    1. Load the training partition from the feature and label registries.
    2. Run ``_N_FOLDS``-fold stratified CV on the **training set** to produce
       out-of-fold predicted probabilities.
    3. Fit an IsotonicRegression calibrator on those OOF training-set predictions.
    4. Apply the fitted calibrator to the raw test-set scores stored in the
       prediction registry, writing ``y_score_calibrated`` and marking
       ``calibration_method = 'isotonic_oof'``.

    This is the standard OOF isotonic calibration procedure: the calibrator
    never sees the test-set labels and is not fitted on the test set itself,
    so the reported calibrated ECE reflects genuine held-out performance.
    """
    print("[+] Running OOF Isotonic Calibration...")

    preds_path = REGISTRY_DIR / "prediction_registry.parquet"
    if not preds_path.exists():
        print("    [!] Error: No predictions to calibrate.")
        return

    df = pd.read_parquet(preds_path)

    # ── Load training data ────────────────────────────────────────────────────
    # Feature data: prefer Data/interim/ (numeric feature matrix); fall back to
    # any file in the repo matching the expected name.
    feat_candidates = [
        ROOT_DIR / "Data" / "interim" / "stage_c_features_raw.parquet",
        ROOT_DIR / "data" / "interim" / "stage_c_features_raw.parquet",
    ]
    feat_path = next((p for p in feat_candidates if p.exists()), None)
    label_path = REGISTRY_DIR / "label_registry.parquet"
    split_path = REGISTRY_DIR / "split_registry.parquet"

    if feat_path is None or not label_path.exists() or not split_path.exists():
        print("    [!] Feature/label/split files not found; skipping real calibration.")
        return

    feat = pd.read_parquet(feat_path)
    labels = pd.read_parquet(label_path)
    splits = pd.read_parquet(split_path)

    lbl = (
        labels[labels['label_version'] == 'label_v1_reconstructed_threshold_crossing']
        .drop_duplicates('case_id')
    )
    feat_view = (
        feat[feat['feature_view'] == 'filing_date_public']
        .drop_duplicates('case_id')
    )
    train_ids = splits[splits['role'] == 'train']['case_id']

    dataset = (
        splits[splits['role'] == 'train']
        .merge(lbl, on='case_id')
        .merge(feat_view, on='case_id')
    )

    meta_cols = {
        'case_id', 'as_of_date', 'feature_view', 'split_id', 'role', 'fold',
        'label_version', 'reconstructed_petition_share', 'threshold_crossed',
        'year', 'filing_date',
    }
    X_train = (
        dataset
        .drop(columns=[c for c in meta_cols if c in dataset.columns], errors='ignore')
        .select_dtypes(include=[np.number])
    )
    y_train = dataset['threshold_crossed'].values
    fill_vals = X_train.median()
    X_train = X_train.fillna(fill_vals)

    print(f"    Training set: n={len(y_train)}, positives={y_train.sum()}")

    # ── Generate OOF training probabilities ──────────────────────────────────
    oof_probs = np.zeros(len(y_train))
    kf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=_CALIB_SEED)
    for fold_train_idx, fold_val_idx in kf.split(X_train, y_train):
        model = CatBoostClassifier(
            iterations=100, depth=4, verbose=0, random_seed=_CALIB_SEED
        )
        model.fit(X_train.iloc[fold_train_idx], y_train[fold_train_idx])
        oof_probs[fold_val_idx] = model.predict_proba(X_train.iloc[fold_val_idx])[:, 1]

    # ── Fit isotonic calibrator on OOF training predictions ──────────────────
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_probs, y_train)
    print(f"    Isotonic calibrator fitted on {len(y_train)} OOF training predictions.")

    # ── Apply calibrator to test-set raw scores ───────────────────────────────
    mask = (df['model_family'] == 'CatBoost') & (df['split_id'] == 'TEMP_OOD_2023_MAIN')
    if not mask.any():
        print("    [!] No CatBoost/TEMP_OOD_2023_MAIN rows found in prediction registry.")
        return

    df.loc[mask, 'y_score_calibrated'] = iso.predict(df.loc[mask, 'y_score_raw'].values)
    df.loc[mask, 'calibration_method'] = 'isotonic_oof'

    save_registry(df, "prediction_registry")
    print("    Calibrated scores written to prediction_registry (calibration_method=isotonic_oof).")


if __name__ == "__main__":
    calibrate_predictions()
