"""Extract raw feature-level explanations for downstream semantic aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def extract_explanations(model: Any, X: pd.DataFrame, model_family: str) -> pd.DataFrame:
    """Return raw feature attributions in a model-agnostic format."""

    family = model_family.lower()
    if family == "catboost" and hasattr(model, "get_feature_importance"):
        importances = np.asarray(model.get_feature_importance(), dtype=float)
        explainer = "catboost_feature_importance"
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        explainer = "feature_importances_"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef[0] if coef.ndim > 1 else coef)
        explainer = "absolute_coefficients"
    else:
        importances = np.ones(X.shape[1], dtype=float)
        explainer = "uniform_fallback"

    if importances.sum() == 0:
        importances = np.ones_like(importances, dtype=float)

    shares = importances / importances.sum()
    return pd.DataFrame(
        {
            "feature_name": list(X.columns),
            "attribution_value": shares.astype(float),
            "abs_attribution_value": np.abs(shares.astype(float)),
            "explainer_method": explainer,
        }
    )


def extract_raw_explanations(model: Any, X: pd.DataFrame, model_family: str, output_path: Optional[str] = None) -> pd.DataFrame:
    df = extract_explanations(model, X, model_family)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
    return df


if __name__ == "__main__":
    raise SystemExit("Use this module from a trained model context.")
