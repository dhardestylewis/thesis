import pandas as pd
import numpy as np
from pathlib import Path
import sys
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# src/models/train_stage_c.py
sys.path.append(str(Path(r"c:\Users\dhl\data\thesis\thesis") / "src"))
from src.data_io.schema import ROOT_DIR, REGISTRY_DIR, save_registry

def train_stage_c(model_family: str = "CatBoost", seed: int = 42):
    print(f"[+] Training Stage C Model: {model_family} (Seed: {seed})...")
    
    # Load registries
    universe = pd.read_parquet(REGISTRY_DIR / "case_universe.parquet")
    labels = pd.read_parquet(REGISTRY_DIR / "label_registry.parquet")
    splits = pd.read_parquet(REGISTRY_DIR / "split_registry.parquet")
    
    # Features from interim
    X_full = pd.read_parquet(ROOT_DIR / "data" / "interim" / "stage_c_features_raw.parquet")
    
    # Task Config Filter (Standard Thesis Task: FILING MAIN)
    lbl = labels[labels['label_version'] == 'label_v1_reconstructed_threshold_crossing'].drop_duplicates('case_id')
    spl = splits[splits['split_id'] == 'TEMP_OOD_2023_MAIN'].drop_duplicates('case_id')
    X_view = X_full[X_full['feature_view'] == 'filing_date_public'].drop_duplicates('case_id')
    
    # Merge Dataset
    dataset = spl.merge(lbl, on='case_id').merge(X_view, on='case_id')
    
    # Optional IPW weighting (Stage A Support)
    weights_path = REGISTRY_DIR / "ipw_weights.parquet"
    if weights_path.exists():
        weights = pd.read_parquet(weights_path)
        dataset = dataset.merge(weights, on='case_id', how='left')
        dataset['ipw_weight'] = dataset['ipw_weight'].fillna(1.0)
    else:
        dataset['ipw_weight'] = 1.0
    
    train = dataset[dataset['role'] == 'train']
    test = dataset[dataset['role'] == 'test']
    
    # Feature columns (numeric only, excluding metadata)
    meta_cols = ['case_id', 'as_of_date', 'feature_view', 'split_id', 'role', 'fold', 
                 'label_version', 'reconstructed_petition_share', 'threshold_crossed', 'year', 'filing_date']
    X_train = train.drop(columns=[c for c in meta_cols if c in train.columns], errors='ignore')
    X_test = test.drop(columns=[c for c in meta_cols if c in test.columns], errors='ignore')
    
    y_train = train['threshold_crossed']
    y_test = test['threshold_crossed']
    
    # Model Selection
    if model_family.lower() == "catboost":
        model = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=seed)
    elif model_family.lower() == "logreg":
        model = LogisticRegression(max_iter=1000, random_state=seed)
    elif model_family.lower() == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        # Placeholder for Deep/Pretrained models if not installed
        print(f"    [!] Warning: {model_family} not fully implemented. Using RF as placeholder.")
        model = RandomForestClassifier(n_estimators=100, random_state=seed)
        
    print(f"    [*] Fitting {model_family} (Weighted)...")
    model.fit(X_train, y_train, sample_weight=train['ipw_weight'] if 'ipw_weight' in train.columns else None)
    
    probs = model.predict_proba(X_test)[:, 1]
    
    # Prepare Prediction Record
    predictions = test[['case_id', 'as_of_date', 'feature_view', 'split_id']].copy()
    predictions['horizon'] = 'filing_date'
    predictions['label_version'] = 'label_v1_reconstructed_threshold_crossing'
    predictions['model_family'] = model_family
    predictions['seed'] = seed
    predictions['calibration_method'] = 'none'
    predictions['y_true'] = y_test
    predictions['y_score_raw'] = probs
    predictions['y_score_calibrated'] = probs
    
    # Append to existing registry if it exists
    existing = pd.read_parquet(REGISTRY_DIR / "prediction_registry.parquet") if (REGISTRY_DIR / "prediction_registry.parquet").exists() else pd.DataFrame()
    
    # Ensure we don't have duplicates for the same RunKey
    if not existing.empty:
        key_cols = ['case_id', 'as_of_date', 'horizon', 'label_version', 'feature_view', 'split_id', 'model_family', 'seed']
        # This is a bit slow for large registries, but safe
        # In practice, we'd probably just append and drop duplicates on save
        full = pd.concat([existing, predictions], ignore_index=True).drop_duplicates(subset=key_cols, keep='last')
    else:
        full = predictions
        
    save_registry(full, "prediction_registry")
    print(f"    Saved {len(predictions)} predictions to prediction_registry.parquet")

    # Step: Extraction Explanations (New Registration Rule)
    from src.interpretation.extract_raw_explanations import extract_explanations
    raw_attr = extract_explanations(model, X_test, model_family)
    
    # Add metadata for RunKey
    raw_attr['model_family'] = model_family
    raw_attr['seed'] = seed
    raw_attr['split_id'] = 'TEMP_OOD_2023_MAIN'
    
    # Save/Append to interpretation registry
    interp_existing = pd.read_parquet(REGISTRY_DIR / "interpretation_registry.parquet") if (REGISTRY_DIR / "interpretation_registry.parquet").exists() else pd.DataFrame()
    interp_full = pd.concat([interp_existing, raw_attr], ignore_index=True)
    save_registry(interp_full, "interpretation_registry")
    print(f"    Registered {len(raw_attr)} feature attributions.")

if __name__ == "__main__":
    # Support command line args for family
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="CatBoost")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_stage_c(model_family=args.model, seed=args.seed)
