"""Calibrate Stage C prediction scores without overwriting raw outputs."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data_io.schema import REGISTRY_DIR, ensure_dirs, save_registry


def calibrate_predictions(
    split_id: str = "TEMP_OOD_2023_MAIN",
    model_family: str = "CatBoost",
    calibration_method: str = "isotonic",
) -> pd.DataFrame:
    """Apply a post-hoc calibration map when a calibration subset is available.

    If no calibration subset exists, the function keeps the raw score and marks
    the calibration method as an explicit no-op.
    """

    ensure_dirs()
    preds_path = REGISTRY_DIR / "prediction_registry.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"Prediction registry not found: {preds_path}")

    df = pd.read_parquet(preds_path).copy()
    mask = (df["split_id"] == split_id) & (df["model_family"] == model_family)
    subset = df.loc[mask].copy()
    if subset.empty:
        raise ValueError(f"No predictions found for split_id={split_id!r} and model_family={model_family!r}.")

    calibration_source = subset[subset["role"].isin(["train", "valid", "calibration"])]
    if calibration_source.empty or calibration_source["y_true"].nunique() < 2:
        df.loc[mask, "y_score_calibrated"] = df.loc[mask, "y_score_raw"]
        df.loc[mask, "calibration_method"] = "identity_noop"
    else:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(calibration_source["y_score_raw"], calibration_source["y_true"])
        df.loc[mask, "y_score_calibrated"] = cast(Any, iso).transform(df.loc[mask, "y_score_raw"])
        df.loc[mask, "calibration_method"] = calibration_method

    save_registry(df, "prediction_registry")
    return df.loc[mask].copy()


if __name__ == "__main__":
    calibrate_predictions()
