"""
Expanding Window Backtest for Protest Prediction (Rare Class)
=============================================================
- W&B experiment tracking + local appendable JSONL
- Multi-horizon evaluation (1y, 2y, 3y ahead)
- Out-of-sample forecast mode (train all, score future)
- Logistic Regression baseline with class_weight='balanced'
"""
import csv
import sys
import os
import json
import numpy as np
import logging
from collections import defaultdict
from datetime import datetime

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# ---- Config ----
PANEL_PATH = "Data/Panel/Output/Property_Year_Panel_Enriched.csv"
RESULTS_DIR = "Analysis/Results/Backtests"
os.makedirs(RESULTS_DIR, exist_ok=True)

JSONL_PATH = os.path.join(RESULTS_DIR, "experiment_log.jsonl")
LOG_PATH = os.path.join(RESULTS_DIR, "backtest.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

TRAIN_START = 2019
EVAL_START = 2020
EVAL_END = 2024
HORIZONS = [1, 2, 3]  # predict 1, 2, 3 years ahead

NUMERIC_FEATURES = [
    "market_value", "assessed_value", "land_value", "improvement_value",
    "living_area", "deed_acreage", "year_built", "land_acres", "improvement_count",
]
CATEGORICAL_FEATURES = [
    "property_category_code", "lui_general_land_use", "council_district",
]
TARGET = "protest"

# ---- W&B Setup ----
USE_WANDB = False
try:
    import wandb
    USE_WANDB = True
except ImportError:
    log.warning("wandb not installed, logging locally only")


def append_jsonl(record):
    """Append a record to the local JSONL experiment log."""
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def load_panel():
    """Load panel, filter to EARS year-matched rows."""
    log.info("Loading panel from %s", PANEL_PATH)
    rows = []
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["year"])
            if year < TRAIN_START:
                continue
            if row.get("ears_matched") != "1":
                continue
            ears_src = row.get("ears_source", "")
            if "backfill" in ears_src:
                continue
            rows.append(row)

    log.info("Loaded %d rows (years %d+, EARS year-matched)", len(rows), TRAIN_START)

    year_counts = defaultdict(lambda: {"total": 0, "pos": 0})
    for r in rows:
        y = int(r["year"])
        year_counts[y]["total"] += 1
        year_counts[y]["pos"] += int(r[TARGET])
    for y in sorted(year_counts):
        c = year_counts[y]
        rate = 100 * c["pos"] / c["total"] if c["total"] else 0
        log.info("  Year %d: %d rows, %d protests (%.3f%%)", y, c["total"], c["pos"], rate)

    return rows


def featurize(rows):
    """Convert rows to feature matrix + target + years."""
    cat_values = {feat: set() for feat in CATEGORICAL_FEATURES}
    for row in rows:
        for feat in CATEGORICAL_FEATURES:
            val = row.get(feat, "").strip()
            if val:
                cat_values[feat].add(val)

    cat_maps = {}
    for feat in CATEGORICAL_FEATURES:
        vals = sorted(cat_values[feat])
        cat_maps[feat] = {v: i for i, v in enumerate(vals)}

    n_numeric = len(NUMERIC_FEATURES)
    n_cat = sum(len(m) for m in cat_maps.values())
    n_features = n_numeric + n_cat

    feature_names = list(NUMERIC_FEATURES)
    for feat in CATEGORICAL_FEATURES:
        for val in sorted(cat_maps[feat].keys()):
            feature_names.append("%s_%s" % (feat, val))

    log.info("Features: %d numeric + %d categorical = %d total", n_numeric, n_cat, n_features)

    X = np.zeros((len(rows), n_features), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int32)
    years = np.zeros(len(rows), dtype=np.int32)
    ids = []

    for i, row in enumerate(rows):
        for j, feat in enumerate(NUMERIC_FEATURES):
            X[i, j] = safe_float(row.get(feat, ""))
        offset = n_numeric
        for feat in CATEGORICAL_FEATURES:
            val = row.get(feat, "").strip()
            if val and val in cat_maps[feat]:
                X[i, offset + cat_maps[feat][val]] = 1.0
            offset += len(cat_maps[feat])
        y[i] = int(row[TARGET])
        years[i] = int(row["year"])
        ids.append(row.get("standardized_tcad_id", ""))

    return X, y, years, feature_names, ids


def compute_metrics(y_true, y_prob, y_pred):
    """Comprehensive rare-class metrics."""
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, f1_score,
        precision_score, recall_score, brier_score_loss, confusion_matrix,
    )

    n = len(y_true)
    n_pos = int(y_true.sum())
    base_rate = n_pos / n if n else 0

    metrics = {"n_total": n, "n_positive": n_pos, "base_rate": base_rate}

    if n_pos == 0 or n_pos == n:
        return metrics

    metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})

    # Lift at top-K
    for k_frac in [0.001, 0.005, 0.01, 0.05]:
        k = max(1, int(n * k_frac))
        top_k_idx = np.argsort(y_prob)[-k:]
        prec_k = y_true[top_k_idx].sum() / k
        lift = prec_k / base_rate if base_rate > 0 else 0
        recall_k = y_true[top_k_idx].sum() / n_pos if n_pos > 0 else 0
        label = "%.1f%%" % (100 * k_frac)
        metrics["precision@%s" % label] = float(prec_k)
        metrics["lift@%s" % label] = float(lift)
        metrics["recall@%s" % label] = float(recall_k)

    # ECE
    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (y_prob >= edges[b]) & (y_prob < edges[b + 1])
        if mask.sum() > 0:
            ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    metrics["ece"] = float(ece)

    return metrics


def run_backtest():
    """Expanding window backtest with multi-horizon."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("=" * 70)
    log.info("EXPANDING WINDOW BACKTEST — run_id=%s", run_id)
    log.info("=" * 70)
    start = datetime.now()

    # Init W&B
    wb_run = None
    if USE_WANDB:
        wb_run = wandb.init(
            project="zoning-opposition-prediction",
            name="backtest_%s" % run_id,
            config={
                "model": "LogisticRegression",
                "class_weight": "balanced",
                "train_start": TRAIN_START,
                "eval_start": EVAL_START,
                "eval_end": EVAL_END,
                "horizons": HORIZONS,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
            },
            tags=["backtest", "naive", "logreg"],
        )

    rows = load_panel()
    X, y, years, feature_names, ids = featurize(rows)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    all_results = []

    # Multi-horizon expanding window
    for horizon in HORIZONS:
        log.info("=" * 50)
        log.info("HORIZON: %d year(s) ahead", horizon)
        log.info("=" * 50)

        for train_end in range(EVAL_START - 1, EVAL_END - horizon + 1):
            eval_year = train_end + horizon

            if eval_year > EVAL_END:
                continue

            log.info("-" * 50)
            log.info("FOLD: train=[%d-%d], test=%d (h=%d)",
                     TRAIN_START, train_end, eval_year, horizon)

            train_mask = years <= train_end
            test_mask = years == eval_year

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            log.info("  Train: %d rows, %d pos (%.3f%%)",
                     len(y_train), y_train.sum(), 100 * y_train.mean())
            log.info("  Test:  %d rows, %d pos (%.3f%%)",
                     len(y_test), y_test.sum(), 100 * y_test.mean())

            if y_train.sum() < 5 or y_test.sum() == 0:
                log.warning("  Skipping: insufficient positives")
                continue

            # Scale
            scaler = StandardScaler()
            X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0, posinf=0, neginf=0)
            X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0, posinf=0, neginf=0)

            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42
            )
            model.fit(X_train_s, y_train)

            y_prob = model.predict_proba(X_test_s)[:, 1]
            y_pred = model.predict(X_test_s)

            metrics = compute_metrics(y_test, y_prob, y_pred)
            metrics["eval_year"] = eval_year
            metrics["train_end"] = train_end
            metrics["horizon"] = horizon
            metrics["model"] = "LogisticRegression"
            metrics["run_id"] = run_id
            metrics["timestamp"] = datetime.now().isoformat()

            # Top features
            coef = model.coef_[0]
            top_idx = np.argsort(np.abs(coef))[-5:][::-1]
            metrics["top_features"] = [
                {"name": feature_names[i], "coef": float(coef[i])} for i in top_idx
            ]

            log.info("  PR-AUC: %.4f | ROC-AUC: %.4f | F1: %.4f | Lift@1%%: %.1fx",
                     metrics.get("pr_auc", 0), metrics.get("roc_auc", 0),
                     metrics.get("f1", 0), metrics.get("lift@1.0%", 0))

            # Log to W&B
            if wb_run:
                wb_metrics = {
                    "horizon": horizon,
                    "eval_year": eval_year,
                    "pr_auc": metrics.get("pr_auc", 0),
                    "roc_auc": metrics.get("roc_auc", 0),
                    "f1": metrics.get("f1", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "brier_score": metrics.get("brier_score", 0),
                    "ece": metrics.get("ece", 0),
                    "lift_1pct": metrics.get("lift@1.0%", 0),
                    "tp": metrics.get("tp", 0),
                    "fp": metrics.get("fp", 0),
                }
                wandb.log(wb_metrics)

            # Append to local JSONL
            append_jsonl(metrics)

            all_results.append(metrics)

    # ---- Out-of-sample forecast ----
    log.info("=" * 50)
    log.info("OUT-OF-SAMPLE FORECAST (train all, score latest year)")
    log.info("=" * 50)

    # Train on everything up to EVAL_END-1, score EVAL_END
    train_mask = years <= EVAL_END - 1
    score_mask = years == EVAL_END

    X_train, y_train = X[train_mask], y[train_mask]
    X_score = X[score_mask]
    score_ids = [ids[i] for i in range(len(ids)) if score_mask[i]]

    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0, posinf=0, neginf=0)
    X_score_s = np.nan_to_num(scaler.transform(X_score), nan=0, posinf=0, neginf=0)

    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42
    )
    model.fit(X_train_s, y_train)
    scores = model.predict_proba(X_score_s)[:, 1]

    # Save ranked forecast
    forecast_path = os.path.join(RESULTS_DIR, "forecast_scores_%s.csv" % run_id)
    ranked = sorted(zip(score_ids, scores), key=lambda x: -x[1])
    with open(forecast_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["standardized_tcad_id", "protest_probability", "rank"])
        for rank, (pid, score) in enumerate(ranked, 1):
            writer.writerow([pid, "%.6f" % score, rank])

    log.info("Forecast: %d parcels scored, top score=%.4f, saved to %s",
             len(scores), scores.max(), forecast_path)

    # Log top-20 highest risk parcels
    log.info("Top 20 highest-risk parcels:")
    for pid, score in ranked[:20]:
        log.info("  %s: %.4f", pid, score)

    # ---- Summary ----
    log.info("=" * 70)
    log.info("SUMMARY ACROSS ALL FOLDS")
    log.info("=" * 70)

    for horizon in HORIZONS:
        fold_results = [r for r in all_results if r["horizon"] == horizon]
        if not fold_results:
            continue
        log.info("Horizon %d:", horizon)
        for key in ["pr_auc", "roc_auc", "f1", "lift@1.0%"]:
            vals = [r[key] for r in fold_results if key in r]
            if vals:
                log.info("  %s: mean=%.4f std=%.4f [%.4f, %.4f]",
                         key, np.mean(vals), np.std(vals), np.min(vals), np.max(vals))

    if wb_run:
        # Log summary table
        summary_table = wandb.Table(
            columns=["horizon", "eval_year", "pr_auc", "roc_auc", "f1", "lift_1pct"],
            data=[[r["horizon"], r["eval_year"], r.get("pr_auc", 0),
                   r.get("roc_auc", 0), r.get("f1", 0), r.get("lift@1.0%", 0)]
                  for r in all_results]
        )
        wandb.log({"results_table": summary_table})
        wandb.finish()

    elapsed = datetime.now() - start
    log.info("Done. Elapsed: %s", elapsed)
    log.info("Local log: %s", JSONL_PATH)
    log.info("Forecast: %s", forecast_path)


if __name__ == "__main__":
    run_backtest()
