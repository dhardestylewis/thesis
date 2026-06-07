import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# Paths
ROOT = Path(r"c:\Users\dhl\data\thesis\thesis")
PIPELINE_DATA = ROOT / "thesis_pipeline" / "data" / "final"

def train_canonical_task():
    print("[+] Executing Canonical Stage C Task...")
    
    # 1. Load Registries
    universe = pd.read_parquet(PIPELINE_DATA / "case_universe.parquet")
    labels = pd.read_parquet(PIPELINE_DATA / "label_registry.parquet")
    features = pd.read_parquet(PIPELINE_DATA / "feature_registry.parquet")
    splits = pd.read_parquet(PIPELINE_DATA / "split_registry.parquet")
    
    # Flatten Registries for current task
    # Task: label_v1, feature_view_admin, split_TEMP_OOD_2023_MAIN
    
    current_label = labels[labels['label_version'] == 'v1_reconstructed_threshold_crossing']
    current_features = features[features['feature_view'] == 'filing_date_public_admin_features']
    current_split = splits[splits['split_id'] == 'TEMP_OOD_2023_MAIN']
    
    # Merge for training
    data = current_split.merge(current_label, on=['case_number', 'as_of_date'])
    data = data.merge(current_features, on=['case_number', 'as_of_date', 'year'])
    
    train_data = data[data['usage'] == 'train']
    test_data = data[data['usage'] == 'test']
    
    X_train = train_data.drop(columns=['case_number', 'year', 'as_of_date', 'split_id', 'usage', 'fold', 'label_version', 'label_value', 'feature_view'], errors='ignore').select_dtypes(include=[np.number])
    y_train = train_data['label_value']
    
    X_test = test_data.drop(columns=['case_number', 'year', 'as_of_date', 'split_id', 'usage', 'fold', 'label_version', 'label_value', 'feature_view'], errors='ignore').select_dtypes(include=[np.number])
    y_test = test_data['label_value']
    
    predictions = []
    
    # 2. Model 1: Logistic Regression
    print("    [~] Training Logistic Regression...")
    lr_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    lr_pipe.fit(X_train, y_train)
    probs_lr = lr_pipe.predict_proba(X_test)[:, 1]
    
    res_lr = test_data[['case_number', 'as_of_date', 'year']].copy()
    res_lr['model_family'] = 'LogisticRegression'
    res_lr['y_prob'] = probs_lr
    res_lr['y_true'] = y_test
    predictions.append(res_lr)
    
    # 3. Model 2: CatBoost
    print("    [~] Training CatBoost...")
    cb = CatBoostClassifier(iterations=200, depth=6, random_seed=42, verbose=0)
    cb.fit(X_train, y_train)
    probs_cb = cb.predict_proba(X_test)[:, 1]
    
    res_cb = test_data[['case_number', 'as_of_date', 'year']].copy()
    res_cb['model_family'] = 'CatBoost'
    res_cb['y_prob'] = probs_cb
    res_cb['y_true'] = y_test
    predictions.append(res_cb)
    
    # Save Prediction Registry
    full_preds = pd.concat(predictions, ignore_index=True)
    full_preds['horizon'] = 'filing_date'
    full_preds['label_version'] = 'v1_reconstructed_threshold_crossing'
    full_preds['feature_view'] = 'filing_date_public_admin_features'
    full_preds['split_id'] = 'TEMP_OOD_2023_MAIN'
    full_preds['seed'] = 42
    
    full_preds.to_parquet(PIPELINE_DATA / "prediction_registry.parquet", index=False)
    
    # Evaluate Headline Metric
    cb_prauc = average_precision_score(y_test, probs_cb)
    print(f"    Canonical CatBoost PR-AUC (OOD 2023-2024): {cb_prauc:.3f}")

if __name__ == "__main__":
    train_canonical_task()
