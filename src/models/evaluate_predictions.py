"""Evaluate registered Stage C predictions into a machine-readable metrics object."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.metrics as skm

from src.data_io.schema import PRIMARY_STAGE_C_HORIZON, REGISTRY_DIR, ensure_dirs

PRIMARY_SPLIT_ID = "TEMP_OOD_2023_MAIN"
PRIMARY_MODEL = "CatBoost"
PRIMARY_TASK_ID = "STAGE_C_FILING_MAIN"

average_precision = cast(Callable[..., float], getattr(skm, "average_precision_score"))
brier_score = cast(Callable[..., float], getattr(skm, "brier_score_loss"))
precision_metric = cast(Callable[..., float], getattr(skm, "precision_score"))
recall_metric = cast(Callable[..., float], getattr(skm, "recall_score"))


def _ece(y_true: npt.NDArray[np.int_], y_prob: npt.NDArray[np.float64], n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    total = len(y_true)
    out = 0.0
    for i in range(n_bins):
        idx = bin_ids == i
        if not np.any(idx):
            continue
        acc = float(np.mean(y_true[idx]))
        conf = float(np.mean(y_prob[idx]))
        out += (np.sum(idx) / total) * abs(acc - conf)
    return float(out)


def _ace(y_true: npt.NDArray[np.int_], y_prob: npt.NDArray[np.float64], n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    quantiles = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(quantiles) < 3:
        return _ece(y_true, y_prob, n_bins=n_bins)
    bin_ids = np.digitize(y_prob, quantiles, right=True) - 1
    diffs: list[float] = []
    for i in range(len(quantiles) - 1):
        idx = bin_ids == i
        if not np.any(idx):
            continue
        diffs.append(abs(float(np.mean(y_true[idx])) - float(np.mean(y_prob[idx]))))
    return float(np.mean(diffs)) if diffs else float("nan")


def _bootstrap_pr_auc_ci(
    y_true: npt.NDArray[np.int_],
    y_score: npt.NDArray[np.float64],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for PR-AUC (average precision) on the evaluation rows."""
    if len(y_true) < 2 or int(np.sum(y_true)) < 1:
        return None, None
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    n = len(y_true)
    attempts = 0
    max_attempts = n_boot * 50
    while len(scores) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if int(np.sum(yt)) < 1:
            continue
        ys = y_score[idx]
        scores.append(float(average_precision(yt, ys)))
    if len(scores) < 100:
        return None, None
    lo, hi = float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))
    return lo, hi


def _threshold_metrics(y_true: npt.NDArray[np.int_], y_prob: npt.NDArray[np.float64], threshold: float) -> dict[str, float]:
    preds = (y_prob >= threshold).astype(int)
    return {
        f"precision_at_{threshold:.2f}".replace(".", "_"): float(precision_metric(y_true, preds, zero_division=0)),
        f"recall_at_{threshold:.2f}".replace(".", "_"): float(recall_metric(y_true, preds, zero_division=0)),
    }


def evaluate_predictions(
    split_id: str = PRIMARY_SPLIT_ID,
    model_family: str = PRIMARY_MODEL,
    use_calibrated: bool = True,
    output_path: str | None = None,
) -> dict[str, object]:
    """Evaluate the canonical Stage C prediction registry rows."""

    ensure_dirs()
    preds_path = REGISTRY_DIR / "prediction_registry.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"Prediction registry not found: {preds_path}")

    df = pd.read_parquet(preds_path)
    subset = df[(df["split_id"] == split_id) & (df["model_family"] == model_family)].copy()
    if subset.empty:
        raise ValueError(f"No prediction rows found for split_id={split_id!r}, model_family={model_family!r}.")
    if "horizon" in subset.columns:
        subset = subset.loc[subset["horizon"] == PRIMARY_STAGE_C_HORIZON].copy()
    if subset.empty:
        raise ValueError(
            f"No rows after horizon filter ({PRIMARY_STAGE_C_HORIZON!r}) for split_id={split_id!r}, model_family={model_family!r}."
        )

    y_true = cast(npt.NDArray[np.int_], np.asarray(subset["y_true"], dtype=int))
    score_col = "y_score_calibrated" if use_calibrated and "y_score_calibrated" in subset.columns else "y_score_raw"
    y_score = cast(npt.NDArray[np.float64], np.asarray(subset[score_col], dtype=float))

    top_decile_mask = subset[score_col] >= subset[score_col].quantile(0.9)
    top_decile = subset.loc[top_decile_mask]
    top_decile_precision = float(top_decile["y_true"].mean()) if not top_decile.empty else None
    baseline_rate = float(y_true.mean()) if len(y_true) else None
    top_decile_lift = None
    if top_decile_precision is not None and baseline_rate not in (None, 0.0):
        top_decile_lift = float(top_decile_precision / baseline_rate)

    pr_auc = float(average_precision(y_true, y_score))
    pr_lo, pr_hi = _bootstrap_pr_auc_ci(y_true, y_score)
    ranking: dict[str, Any] = {
        "pr_auc": pr_auc,
        "top_decile_precision": top_decile_precision,
        "top_decile_lift": top_decile_lift,
    }
    if pr_lo is not None and pr_hi is not None:
        ranking["pr_auc_ci_low"] = pr_lo
        ranking["pr_auc_ci_high"] = pr_hi
    calibration: dict[str, Any] = {
        "brier": float(brier_score(y_true, y_score)),
        "ece": _ece(y_true, y_score),
        "ace": _ace(y_true, y_score),
        "calibration_slope": float(np.polyfit(y_score, y_true, deg=1)[0]) if np.unique(y_score).size > 1 else float("nan"),
    }
    thresholded: dict[str, float] = {}
    thresholded.update(_threshold_metrics(y_true, y_score, 0.30))
    thresholded.update(_threshold_metrics(y_true, y_score, 0.50))

    metrics: dict[str, object] = {
        "task_id": PRIMARY_TASK_ID,
        "split_id": split_id,
        "model_family": model_family,
        "score_column": score_col,
        "ranking": ranking,
        "calibration": calibration,
        "thresholded": thresholded,
        "sample_size": int(len(subset)),
        "positive_rate": float(y_true.mean()),
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
    }

    out_path = Path(output_path) if output_path else REGISTRY_DIR / "evaluation_results.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    evaluate_predictions()
