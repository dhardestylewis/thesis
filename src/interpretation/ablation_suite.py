from pathlib import Path

import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from src.data_io.schema import REGISTRY_DIR, ROOT_DIR, ensure_dirs, save_registry


def run_ablation_suite() -> pd.DataFrame:
    """Run a compact cluster ablation against the canonical Stage C task."""
    import os
    use_gpu = os.environ.get("USE_GPU") == "1"
    cb_task = "GPU" if use_gpu else "CPU"

    ensure_dirs()
    labels = pd.read_parquet(REGISTRY_DIR / "label_registry.parquet")
    splits = pd.read_parquet(REGISTRY_DIR / "split_registry.parquet")
    X_full = pd.read_parquet(ROOT_DIR / "data" / "interim" / "stage_c_features_raw.parquet")

    lbl = labels[labels["label_version"] == "label_v1_reconstructed_threshold_crossing"].drop_duplicates("case_id")
    spl = splits[splits["split_id"] == "TEMP_OOD_2023_MAIN"].drop_duplicates("case_id")
    dataset = spl.merge(lbl, on="case_id").merge(X_full, on="case_id")

    train = dataset[dataset["role"] == "train"]
    test = dataset[dataset["role"] == "test"]

    meta_cols = ["case_id", "as_of_date", "feature_view", "split_id", "role", "fold", "label_version", "reconstructed_petition_share", "threshold_crossed", "year", "filing_date"]
    X_train_full = train.drop(columns=[c for c in meta_cols if c in train.columns], errors="ignore")
    X_test_full = test.drop(columns=[c for c in meta_cols if c in test.columns], errors="ignore")
    numeric_cols = X_train_full.select_dtypes(include=["number"]).columns
    X_train_full = X_train_full[numeric_cols].copy()
    X_test_full = X_test_full[numeric_cols].copy()
    y_train = train["threshold_crossed"]
    y_test = test["threshold_crossed"]

    cb_base = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=42, task_type=cb_task)
    cb_base.fit(X_train_full, y_train)
    base_score = average_precision_score(y_test, cb_base.predict_proba(X_test_full)[:, 1])

    with open(ROOT_DIR / "configs" / "features" / "semantic_clusters.yaml", "r", encoding="utf-8") as f:
        clusters_config = yaml.safe_load(f)["clusters"]

    ablation_results = []
    for cluster_name, features in clusters_config.items():
        present_features = [f for f in features if f in X_train_full.columns]
        if not present_features:
            continue

        X_train_ablated = X_train_full.drop(columns=present_features)
        X_test_ablated = X_test_full.drop(columns=present_features)

        cb_ablated = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=42, task_type=cb_task)
        cb_ablated.fit(X_train_ablated, y_train)
        ablated_score = average_precision_score(y_test, cb_ablated.predict_proba(X_test_ablated)[:, 1])

        ablation_results.append({
            "cluster": cluster_name,
            "base_score": base_score,
            "ablated_score": ablated_score,
            "delta_prauc": base_score - ablated_score,
            "n_features": len(present_features),
        })

    if not ablation_results:
        return pd.DataFrame()

    df_ablation = pd.DataFrame(ablation_results)
    save_registry(df_ablation, "ablation_results")
    return df_ablation

if __name__ == "__main__":
    run_ablation_suite()
