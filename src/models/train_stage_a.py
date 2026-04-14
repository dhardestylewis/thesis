import numpy as np
import pandas as pd

from src.data_io.schema import ROOT_DIR, ensure_dirs, save_registry


def train_stage_a_ipw() -> pd.DataFrame:
    """Produce a bounded selection-correction weight registry.

    If no explicit selection target exists, the sidecar returns stabilized unit
    weights so Stage A remains a support layer instead of a parallel model race.
    """

    ensure_dirs()
    features_path = ROOT_DIR / "data" / "interim" / "stage_a_features_raw.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Stage A features not found: {features_path}")

    X = pd.read_parquet(features_path).copy()
    if X.empty:
        raise ValueError("Stage A feature matrix is empty.")

    if "selection_proxy" in X.columns:
        y = np.asarray(X["selection_proxy"].fillna(0).astype(int), dtype=int)  # pyright: ignore[reportUnknownMemberType]
    else:
        y = np.ones(len(X), dtype=int)

    p_a = float(np.clip(y.mean(), 0.05, 0.95))
    propensity = np.full(len(X), p_a, dtype=float)
    weights = np.where(y == 1, p_a / propensity, (1.0 - p_a) / (1.0 - propensity))
    weights = np.clip(weights, 0.5, 2.0)

    weight_df = X[["case_id"]].copy()
    weight_df["ipw_weight"] = weights
    weight_df["propensity_score"] = propensity
    weight_df["selection_target_rate"] = p_a

    save_registry(weight_df, "ipw_weights")
    return weight_df

if __name__ == "__main__":
    train_stage_a_ipw()
