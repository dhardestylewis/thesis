"""Train the canonical Stage C model family and write prediction registry rows."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_io.schema import REGISTRY_DIR, ROOT_DIR, ensure_dirs, save_registry
from src.interpretation.extract_raw_explanations import extract_raw_explanations

PRIMARY_SPLIT_ID = "TEMP_OOD_2023_MAIN"
PRIMARY_LABEL_VERSION = "label_v1_reconstructed_threshold_crossing"
PRIMARY_FEATURE_VIEW = "filing_date_public"
PRIMARY_HORIZON = "filing"


def _load_prediction_registry() -> pd.DataFrame:
    path = REGISTRY_DIR / "prediction_registry.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_weights() -> pd.DataFrame:
    for name in ["ipw_weights.parquet", "selection_weights.parquet"]:
        path = REGISTRY_DIR / name
        if path.exists():
            return pd.read_parquet(path)
    return pd.DataFrame()


def _aggregate_case_weights(weights: pd.DataFrame) -> pd.DataFrame:
    weight_slice = weights[["case_id", "ipw_weight"]].copy()
    weight_slice["case_id"] = weight_slice["case_id"].astype(str)
    grouped = weight_slice.groupby("case_id", as_index=False).agg(ipw_weight=("ipw_weight", "mean"))  # pyright: ignore[reportUnknownMemberType]
    return grouped


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "case_id",
        "as_of_date",
        "feature_view",
        "split_id",
        "role",
        "fold",
        "label_version",
        "reconstructed_petition_share",
        "threshold_crossed",
        "y_true",
        "y_score_raw",
        "y_score_calibrated",
        "sample_weight",
        "prediction_timestamp",
        "horizon",
        "model_family",
        "seed",
        "calibration_method",
    }
    return [c for c in df.columns if c not in excluded]


def _make_model(model_family: str, seed: int) -> Any:
    family = model_family.lower()
    if family == "catboost":
        catboost_module = importlib.import_module("catboost")
        catboost_classifier = getattr(catboost_module, "CatBoostClassifier")
        return catboost_classifier(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=0,
        )
    if family == "logreg":
        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
    if family in {"rf", "randomforest"}:
        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, min_samples_leaf=3)),
            ]
        )
    raise ValueError(f"Unsupported Stage C model family: {model_family}")


def _fit_predict(
    model: Any,
    model_family: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    sample_weight: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    family = model_family.lower()
    if family == "catboost":
        cast_model = model
        cast_model.fit(X_train, y_train, sample_weight=sample_weight)
    elif family == "logreg":
        cast_model = model
        cast_model.fit(X_train, y_train, clf__sample_weight=sample_weight)
    else:
        cast_model = model
        cast_model.fit(X_train, y_train, clf__sample_weight=sample_weight)
    return np.asarray(cast_model.predict_proba(X_test))[:, 1]


def train_stage_c(model_family: str = "CatBoost", seed: int = 42, split_id: str = PRIMARY_SPLIT_ID) -> pd.DataFrame:
    """Train one Stage C model family on the canonical task."""

    ensure_dirs()

    labels = pd.read_parquet(REGISTRY_DIR / "label_registry.parquet")
    splits = pd.read_parquet(REGISTRY_DIR / "split_registry.parquet")
    features = pd.read_parquet(ROOT_DIR / "data" / "interim" / "stage_c_features_raw.parquet")

    label_slice = labels.loc[labels["label_version"] == PRIMARY_LABEL_VERSION, ["case_id", "threshold_crossed", "reconstructed_petition_share"]].copy()
    label_slice["case_id"] = label_slice["case_id"].astype(str)
    label_slice = label_slice.drop_duplicates(subset=["case_id"], keep="first")

    split_slice = splits.loc[splits["split_id"] == split_id, ["case_id", "split_id", "role", "fold"]].copy()
    split_slice["case_id"] = split_slice["case_id"].astype(str)
    split_slice = split_slice.drop_duplicates(subset=["case_id"], keep="first")

    feature_slice = features.loc[features["feature_view"] == PRIMARY_FEATURE_VIEW].copy()
    feature_slice["case_id"] = feature_slice["case_id"].astype(str)
    feature_slice = feature_slice.drop_duplicates(subset=["case_id"], keep="first")

    dataset = split_slice.merge(label_slice, on="case_id", how="inner").merge(feature_slice, on="case_id", how="inner")
    if dataset.empty:
        raise ValueError("Stage C training dataset is empty. Check the registries and feature view.")

    if "role" not in dataset.columns:
        raise ValueError("The split registry must contain a role column.")

    weights = _load_weights()
    if not weights.empty and {"case_id", "ipw_weight"}.issubset(weights.columns):
        weight_slice = _aggregate_case_weights(weights)
        dataset["case_id"] = dataset["case_id"].astype(str)
        dataset = dataset.merge(weight_slice, on="case_id", how="left")
    else:
        dataset["ipw_weight"] = 1.0
    ipw_series = pd.Series(np.asarray(dataset["ipw_weight"], dtype=float), index=dataset.index)
    dataset["ipw_weight"] = cast(Any, ipw_series).fillna(1.0)

    train = dataset[dataset["role"] == "train"].copy()
    test = dataset[dataset["role"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError(f"Primary split {split_id} must contain both train and test rows.")

    feature_cols = _select_feature_columns(dataset)
    if not feature_cols:
        raise ValueError("No model features found after excluding registry columns.")

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_train = train["threshold_crossed"].astype(int)
    y_test = test["threshold_crossed"].astype(int)

    model = _make_model(model_family, seed)
    y_score_raw = _fit_predict(model, model_family, X_train, y_train, X_test, sample_weight=np.asarray(train["ipw_weight"], dtype=float))

    predictions = test[["case_id", "as_of_date", "feature_view", "split_id", "role", "fold"]].copy()
    predictions["horizon"] = PRIMARY_HORIZON
    predictions["label_version"] = PRIMARY_LABEL_VERSION
    predictions["model_family"] = model_family
    predictions["seed"] = seed
    predictions["calibration_method"] = "none"
    predictions["y_true"] = np.asarray(y_test, dtype=int)
    predictions["y_score_raw"] = np.asarray(y_score_raw, dtype=float)
    predictions["y_score_calibrated"] = predictions["y_score_raw"]
    predictions["sample_weight"] = 1.0
    predictions.loc[:, "prediction_timestamp"] = datetime.now(timezone.utc).isoformat()

    existing = _load_prediction_registry()
    combined = pd.concat([existing, predictions], ignore_index=True, sort=False) if not existing.empty else predictions
    if not combined.empty:
        combined = combined.drop_duplicates(
            subset=["case_id", "as_of_date", "horizon", "label_version", "feature_view", "split_id", "model_family", "seed"],
            keep="last",
        ).reset_index(drop=True)
    save_registry(combined, "prediction_registry")

    raw_attr = extract_raw_explanations(model, X_test, model_family)
    raw_attr["case_id"] = "__aggregate__"
    raw_attr["as_of_date"] = predictions["as_of_date"].iloc[0]
    raw_attr["horizon"] = PRIMARY_HORIZON
    raw_attr["label_version"] = PRIMARY_LABEL_VERSION
    raw_attr["feature_view"] = PRIMARY_FEATURE_VIEW
    raw_attr["split_id"] = split_id
    raw_attr["model_family"] = model_family
    raw_attr["seed"] = seed
    raw_attr["calibration_method"] = "none"
    raw_attr["bootstrap_rep"] = 0

    interp_path = REGISTRY_DIR / "interpretation_registry.parquet"
    if interp_path.exists():
        existing_interp = pd.read_parquet(interp_path)
        raw_attr = pd.concat([existing_interp, raw_attr], ignore_index=True, sort=False)
    save_registry(raw_attr, "interpretation_registry")

    artifact_dir = ROOT_DIR / "data" / "final" / "stage_c_models"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{model_family.lower()}_{split_id}_{seed}.json"
    artifact_path.write_text(
        json.dumps(
            {
                "model_family": model_family,
                "seed": seed,
                "split_id": split_id,
                "feature_count": len(feature_cols),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CatBoost")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-id", default=PRIMARY_SPLIT_ID)
    args = parser.parse_args()

    train_stage_c(model_family=args.model, seed=args.seed, split_id=args.split_id)
