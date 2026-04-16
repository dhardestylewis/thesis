"""
Experiment Runner: Parallel Diffusion v4 Experiments
=====================================================
Runs 6 experiments based on v3 diagnostic findings.
Each experiment logs to W&B and saves artifacts locally.

Usage:
  python experiment_runner.py                 # Run all experiments
  python experiment_runner.py exp01_platt     # Run single experiment
  python experiment_runner.py --wave 1        # Run wave 1 only

Experiments:
  Wave 1 (post-hoc, ~30s each):
    exp01_platt       - Platt scaling on existing diff_scores
    exp02_isotonic    - Isotonic regression on existing diff_scores
    exp06_adaptive    - Per-property-type ensemble weights
  Wave 2 (retrain, ~5min each):
    exp03_focal       - Focal loss classifier
    exp04_larger_clf  - Larger classifier head (256 hidden, 3 layers)
  Wave 3 (retrain+finetune, ~5min):
    exp05_clf_on_gen  - Classifier finetuned on generated features
"""
import csv, json, sys, os, time, math, subprocess, argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# ---- Paths ----
PANEL_PATH = "Data/Panel/Output/Property_Year_Panel_Enriched.csv"
CENTROIDS_PATH = "Data/Panel/Reference/parcel_centroids.csv"
V3_SCORES_PATH = "Analysis/Results/Diffusion_v3/per_parcel_scores.csv"
V3_CHECKPOINT = "Analysis/Results/Diffusion_v3/model_checkpoint.pt"
EXP_BASE_DIR = "Analysis/Results/Experiments"
os.makedirs(EXP_BASE_DIR, exist_ok=True)

# ---- Shared Config (from v3) ----
BASE_CONFIG = {
    "TRAIN_START": 2019,
    "EVAL_YEARS": [2021, 2022, 2023, 2024],
    "NUMERIC_FEATURES": [
        "market_value", "assessed_value", "land_value", "improvement_value",
        "living_area", "deed_acreage", "year_built", "land_acres", "improvement_count",
    ],
    "CATEGORICAL_FEATURES": ["property_category_code", "lui_general_land_use", "council_district"],
    "TARGET": "protest",
    "DIFF_TIMESTEPS": 200,
    "DIFF_HIDDEN": 256,
    "DIFF_LAYERS": 3,
    "DIFF_EPOCHS": 75,
    "DIFF_LR": 3e-4,
    "DIFF_BATCH": 2048,
    "EARLY_STOP_PATIENCE": 10,
    "DDIM_STEPS": 50,
    "N_SCENARIOS": 10,
    "MAX_TRAIN_PAIRS": 100000,
    "MAX_EVAL_PARCELS": 30000,
    "LOOKBACK_YEARS": 2,
    "ENSEMBLE_WEIGHT": 0.5,
    "CLF_HIDDEN": 128,
    "CLF_EPOCHS": 50,
    "CLF_LR": 1e-3,
    "CLF_BATCH": 4096,
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# =============================================
# POST-HOC EXPERIMENTS (Wave 1) — No retraining
# =============================================

def load_v3_scores():
    """Load existing v3 per-parcel scores."""
    scores = []
    with open(V3_SCORES_PATH, "r") as f:
        for row in csv.DictReader(f):
            try:
                scores.append({
                    "pid": row["parcel_id"],
                    "year": int(row["year"]),
                    "lr": float(row["lr_score"]),
                    "diff": float(row["diff_score"]),
                    "ens": float(row["ensemble_score"]),
                    "actual": float(row["actual"]),
                })
            except (ValueError, KeyError):
                continue
    return scores


def compute_metrics(y_true, y_prob, label=""):
    """Compute classification metrics."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  brier_score_loss, f1_score, precision_score, recall_score)
    y_true = np.array(y_true)
    y_prob = np.clip(np.array(y_prob), 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        return {"label": label, "error": "single_class"}

    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # ECE (10 bins)
    bins = np.linspace(0, 1, 11)
    ece = 0
    for i in range(10):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(y_prob[mask].mean() - y_true[mask].mean())
    ece /= len(y_true)

    return {
        "label": label,
        "auc_roc": round(auc_roc, 5),
        "auc_pr": round(auc_pr, 5),
        "brier": round(brier, 5),
        "ece": round(ece, 5),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 5),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 5),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 5),
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
    }


def run_exp01_platt(exp_dir):
    """Platt scaling: fit sigmoid on diff_score using half the data, evaluate on other half."""
    import wandb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    config = {**BASE_CONFIG, "experiment": "exp01_platt", "method": "platt_scaling"}
    wandb.init(project="thesis-diffusion-v4", name="exp01_platt", config=config,
               dir=exp_dir, tags=["wave1", "posthoc", "calibration"])

    scores = load_v3_scores()
    log(f"  Loaded {len(scores)} scores")

    # Group by year for per-year evaluation
    by_year = defaultdict(list)
    for s in scores:
        by_year[s["year"]].append(s)

    all_metrics = {}
    all_calibrated = []

    for year in sorted(by_year):
        rows = by_year[year]
        diff_probs = np.array([s["diff"] for s in rows])
        actuals = np.array([s["actual"] for s in rows])

        # 2-fold: fit on half, predict on other half (then swap)
        calibrated = np.zeros_like(diff_probs)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        for train_idx, test_idx in skf.split(diff_probs, actuals):
            # Platt scaling = logistic regression on log-odds
            X_train = diff_probs[train_idx].reshape(-1, 1)
            X_test = diff_probs[test_idx].reshape(-1, 1)

            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_train, actuals[train_idx])
            calibrated[test_idx] = lr.predict_proba(X_test)[:, 1]

        # Compute metrics for original and calibrated
        orig_metrics = compute_metrics(actuals, diff_probs, f"diff_original_{year}")
        cal_metrics = compute_metrics(actuals, calibrated, f"diff_platt_{year}")

        # Also compute ensemble with LR
        lr_probs = np.array([s["lr"] for s in rows])
        ens_cal = 0.5 * lr_probs + 0.5 * calibrated
        ens_metrics = compute_metrics(actuals, ens_cal, f"ensemble_platt_{year}")

        all_metrics[year] = {
            "original": orig_metrics,
            "platt": cal_metrics,
            "ensemble_platt": ens_metrics,
        }

        for s, cp in zip(rows, calibrated):
            all_calibrated.append({**s, "diff_calibrated": cp, "ens_calibrated": 0.5 * s["lr"] + 0.5 * cp})

        wandb.log({
            f"auc_roc_original/{year}": orig_metrics.get("auc_roc", 0),
            f"auc_roc_platt/{year}": cal_metrics.get("auc_roc", 0),
            f"ece_original/{year}": orig_metrics.get("ece", 0),
            f"ece_platt/{year}": cal_metrics.get("ece", 0),
            f"brier_original/{year}": orig_metrics.get("brier", 0),
            f"brier_platt/{year}": cal_metrics.get("brier", 0),
        })

        log(f"  {year}: ECE {orig_metrics.get('ece',0):.5f} → {cal_metrics.get('ece',0):.5f} | "
            f"Brier {orig_metrics.get('brier',0):.5f} → {cal_metrics.get('brier',0):.5f}")

    # Save artifacts
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save calibrated scores
    csv_path = os.path.join(exp_dir, "per_parcel_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pid", "year", "lr", "diff", "diff_calibrated", "ens_calibrated", "actual"])
        w.writeheader()
        for s in all_calibrated:
            w.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in
                        {"pid": s["pid"], "year": s["year"], "lr": s["lr"], "diff": s["diff"],
                         "diff_calibrated": s["diff_calibrated"], "ens_calibrated": s["ens_calibrated"],
                         "actual": s["actual"]}.items()})

    wandb.save(os.path.join(exp_dir, "metrics.json"))
    wandb.save(csv_path)
    wandb.finish()
    return all_metrics


def run_exp02_isotonic(exp_dir):
    """Isotonic regression: non-parametric calibration."""
    import wandb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    config = {**BASE_CONFIG, "experiment": "exp02_isotonic", "method": "isotonic_regression"}
    wandb.init(project="thesis-diffusion-v4", name="exp02_isotonic", config=config,
               dir=exp_dir, tags=["wave1", "posthoc", "calibration"])

    scores = load_v3_scores()
    by_year = defaultdict(list)
    for s in scores:
        by_year[s["year"]].append(s)

    all_metrics = {}
    all_calibrated = []

    for year in sorted(by_year):
        rows = by_year[year]
        diff_probs = np.array([s["diff"] for s in rows])
        actuals = np.array([s["actual"] for s in rows])

        calibrated = np.zeros_like(diff_probs)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        for train_idx, test_idx in skf.split(diff_probs, actuals):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(diff_probs[train_idx], actuals[train_idx])
            calibrated[test_idx] = iso.predict(diff_probs[test_idx])

        orig_metrics = compute_metrics(actuals, diff_probs, f"diff_original_{year}")
        cal_metrics = compute_metrics(actuals, calibrated, f"diff_isotonic_{year}")

        lr_probs = np.array([s["lr"] for s in rows])
        ens_cal = 0.5 * lr_probs + 0.5 * calibrated
        ens_metrics = compute_metrics(actuals, ens_cal, f"ensemble_isotonic_{year}")

        all_metrics[year] = {
            "original": orig_metrics,
            "isotonic": cal_metrics,
            "ensemble_isotonic": ens_metrics,
        }

        for s, cp in zip(rows, calibrated):
            all_calibrated.append({**s, "diff_calibrated": cp, "ens_calibrated": 0.5 * s["lr"] + 0.5 * cp})

        wandb.log({
            f"auc_roc_isotonic/{year}": cal_metrics.get("auc_roc", 0),
            f"ece_isotonic/{year}": cal_metrics.get("ece", 0),
            f"brier_isotonic/{year}": cal_metrics.get("brier", 0),
        })

        log(f"  {year}: ECE {orig_metrics.get('ece',0):.5f} → {cal_metrics.get('ece',0):.5f} | "
            f"Brier {orig_metrics.get('brier',0):.5f} → {cal_metrics.get('brier',0):.5f}")

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    csv_path = os.path.join(exp_dir, "per_parcel_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pid", "year", "lr", "diff", "diff_calibrated", "ens_calibrated", "actual"])
        w.writeheader()
        for s in all_calibrated:
            w.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in
                        {"pid": s["pid"], "year": s["year"], "lr": s["lr"], "diff": s["diff"],
                         "diff_calibrated": s["diff_calibrated"], "ens_calibrated": s["ens_calibrated"],
                         "actual": s["actual"]}.items()})

    wandb.save(os.path.join(exp_dir, "metrics.json"))
    wandb.save(csv_path)
    wandb.finish()
    return all_metrics


def run_exp06_adaptive(exp_dir):
    """Adaptive ensemble: optimize per-property-type weights."""
    import wandb
    from scipy.optimize import minimize_scalar

    config = {**BASE_CONFIG, "experiment": "exp06_adaptive", "method": "adaptive_ensemble"}
    wandb.init(project="thesis-diffusion-v4", name="exp06_adaptive", config=config,
               dir=exp_dir, tags=["wave1", "posthoc", "ensemble"])

    scores = load_v3_scores()

    # Load panel features for property_category
    panel = {}
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("standardized_tcad_id", "").strip()
            year = int(row["year"])
            if pid:
                panel[(pid, year)] = row.get("property_category_code", "A")

    # Enrich
    for s in scores:
        s["pcat"] = panel.get((s["pid"], s["year"]), "A")

    # Split: use 2021-2023 to learn weights, evaluate on 2024
    train_scores = [s for s in scores if s["year"] in [2021, 2022, 2023]]
    test_scores = [s for s in scores if s["year"] == 2024]

    # Find optimal weight per property category (minimize Brier on training years)
    by_cat = defaultdict(list)
    for s in train_scores:
        by_cat[s["pcat"]].append(s)

    optimal_weights = {}
    for cat, rows in by_cat.items():
        if len(rows) < 20:
            optimal_weights[cat] = 0.5
            continue

        actuals = np.array([s["actual"] for s in rows])
        lr_probs = np.array([s["lr"] for s in rows])
        diff_probs = np.array([s["diff"] for s in rows])

        def brier_for_weight(w):
            ens = w * lr_probs + (1 - w) * diff_probs
            return np.mean((ens - actuals) ** 2)

        result = minimize_scalar(brier_for_weight, bounds=(0, 1), method="bounded")
        optimal_weights[cat] = round(result.x, 3)
        log(f"  {cat}: n={len(rows)}, optimal LR weight={optimal_weights[cat]:.3f} (Brier={result.fun:.6f})")

    # Apply to all years
    all_metrics = {}
    all_scored = []
    by_year = defaultdict(list)
    for s in scores:
        by_year[s["year"]].append(s)

    for year in sorted(by_year):
        rows = by_year[year]
        actuals = np.array([s["actual"] for s in rows])
        lr_probs = np.array([s["lr"] for s in rows])
        diff_probs = np.array([s["diff"] for s in rows])

        # Naive ensemble
        naive_ens = 0.5 * lr_probs + 0.5 * diff_probs
        naive_metrics = compute_metrics(actuals, naive_ens, f"naive_ensemble_{year}")

        # Adaptive ensemble
        adaptive_ens = np.array([
            optimal_weights.get(s["pcat"], 0.5) * s["lr"] +
            (1 - optimal_weights.get(s["pcat"], 0.5)) * s["diff"]
            for s in rows
        ])
        adaptive_metrics = compute_metrics(actuals, adaptive_ens, f"adaptive_ensemble_{year}")

        all_metrics[year] = {
            "naive": naive_metrics,
            "adaptive": adaptive_metrics,
        }

        for s, ae in zip(rows, adaptive_ens):
            all_scored.append({**s, "adaptive_ens": ae})

        wandb.log({
            f"brier_naive/{year}": naive_metrics.get("brier", 0),
            f"brier_adaptive/{year}": adaptive_metrics.get("brier", 0),
            f"auc_roc_naive/{year}": naive_metrics.get("auc_roc", 0),
            f"auc_roc_adaptive/{year}": adaptive_metrics.get("auc_roc", 0),
        })

        log(f"  {year}: Brier naive={naive_metrics.get('brier',0):.5f} → adaptive={adaptive_metrics.get('brier',0):.5f}")

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({**config, "optimal_weights": optimal_weights}, f, indent=2)

    csv_path = os.path.join(exp_dir, "per_parcel_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pid", "year", "pcat", "lr", "diff", "adaptive_ens", "actual"])
        w.writeheader()
        for s in all_scored:
            w.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in
                        {"pid": s["pid"], "year": s["year"], "pcat": s["pcat"],
                         "lr": s["lr"], "diff": s["diff"], "adaptive_ens": s["adaptive_ens"],
                         "actual": s["actual"]}.items()})

    wandb.save(os.path.join(exp_dir, "metrics.json"))
    wandb.save(csv_path)
    wandb.finish()
    return all_metrics


# =============================================
# RETRAINING EXPERIMENTS (Wave 2 & 3)
# =============================================

def run_retrain_experiment(exp_id, exp_dir, config_overrides):
    """
    Run a full retrain experiment by invoking diffusion_v3_diagnostic.py
    with modified config, logging to W&B, and saving all artifacts.
    
    This forks the v3 training pipeline with specific modifications.
    """
    import wandb
    import torch
    import torch.nn as nn

    config = {**BASE_CONFIG, **config_overrides, "experiment": exp_id}
    wandb.init(project="thesis-diffusion-v4", name=exp_id, config=config,
               dir=exp_dir, tags=["retrain", config_overrides.get("wave", "wave2")])

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- Import and run the v3 pipeline with modifications ----
    # We import the core components from the v3 script's module
    sys.path.insert(0, "Analysis/Scripts/Modeling")

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")

    # ---- Load data (same as v3) ----
    log(f"  [{exp_id}] Loading panel data...")
    from diffusion_v3_diagnostic import (
        NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
    )

    # We need to re-implement data loading since the v3 script uses globals
    rows_by_pid = defaultdict(dict)
    all_cats = defaultdict(set)

    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("standardized_tcad_id", "").strip()
            year = int(row["year"])
            if pid and year >= config["TRAIN_START"]:
                rows_by_pid[pid][year] = row
                for cat_feat in CATEGORICAL_FEATURES:
                    all_cats[cat_feat].add(row.get(cat_feat, ""))

    # Build one-hot encoders
    cat_maps = {}
    for feat, vals in all_cats.items():
        sorted_vals = sorted(vals)
        cat_maps[feat] = {v: i for i, v in enumerate(sorted_vals)}
    n_numeric = len(NUMERIC_FEATURES)
    n_cat_total = sum(len(m) for m in cat_maps.values())
    feature_dim = n_numeric + n_cat_total

    def row_to_features(row):
        num = []
        for feat in NUMERIC_FEATURES:
            try:
                num.append(float(row.get(feat, 0) or 0))
            except ValueError:
                num.append(0.0)
        cat = np.zeros(n_cat_total, dtype=np.float32)
        offset = 0
        for feat in CATEGORICAL_FEATURES:
            val = row.get(feat, "")
            if val in cat_maps[feat]:
                cat[offset + cat_maps[feat][val]] = 1.0
            offset += len(cat_maps[feat])
        return np.array(num + cat.tolist(), dtype=np.float32)

    # Build lag features
    n_lag = 2  # protest_lag1, value_change
    lookback = config["LOOKBACK_YEARS"]

    def build_conditioning(pid, year):
        """Build full conditioning vector: [year_features * lookback | lag_features]"""
        feats = []
        for y_offset in range(lookback):
            y = year - y_offset
            if y in rows_by_pid[pid]:
                feats.append(row_to_features(rows_by_pid[pid][y]))
            else:
                feats.append(np.zeros(feature_dim, dtype=np.float32))

        # Lag features
        lag = np.zeros(n_lag, dtype=np.float32)
        if year - 1 in rows_by_pid[pid]:
            lag[0] = float(rows_by_pid[pid][year - 1].get(TARGET, 0) or 0)
            try:
                prev_mv = float(rows_by_pid[pid][year - 1].get("market_value", 0) or 0)
                curr_mv = float(rows_by_pid[pid][year].get("market_value", 0) or 0) if year in rows_by_pid[pid] else 0
                lag[1] = (curr_mv - prev_mv) / max(prev_mv, 1)
            except (ValueError, KeyError):
                pass

        return np.concatenate(feats + [lag])

    cond_dim = lookback * feature_dim + n_lag

    # ---- Build train/eval datasets ----
    log(f"  [{exp_id}] Building datasets...")

    # Collect all training pairs (expanding window like v3)
    train_data = []  # (cond_vector, target_features, protest_label)
    for pid, years in rows_by_pid.items():
        for year in sorted(years):
            if year <= config["TRAIN_START"] + lookback:
                continue
            if year in config["EVAL_YEARS"]:
                continue
            if year - 1 not in years:
                continue
            cond = build_conditioning(pid, year)
            tgt = row_to_features(years[year])
            protest = float(years[year].get(TARGET, 0) or 0)
            train_data.append((cond, tgt, protest))

    if len(train_data) > config["MAX_TRAIN_PAIRS"]:
        np.random.shuffle(train_data)
        train_data = train_data[:config["MAX_TRAIN_PAIRS"]]

    X_cond = np.array([d[0] for d in train_data], dtype=np.float32)
    X_tgt = np.array([d[1] for d in train_data], dtype=np.float32)
    Y_labels = np.array([d[2] for d in train_data], dtype=np.float32)

    log(f"  [{exp_id}] Training data: {len(train_data)} pairs, {int(Y_labels.sum())} positives")

    # Normalize
    cond_numeric_indices = list(range(n_numeric)) + [lookback * feature_dim + i for i in range(n_lag)]
    for idx_set in [cond_numeric_indices]:
        pass  # will normalize after

    cond_mean = X_cond[:, cond_numeric_indices].mean(axis=0)
    cond_std = X_cond[:, cond_numeric_indices].std(axis=0) + 1e-8
    X_cond_norm = X_cond.copy()
    X_cond_norm[:, cond_numeric_indices] = (X_cond[:, cond_numeric_indices] - cond_mean) / cond_std

    tgt_mean = X_tgt[:, :n_numeric].mean(axis=0)
    tgt_std = X_tgt[:, :n_numeric].std(axis=0) + 1e-8
    X_tgt_norm = X_tgt.copy()
    X_tgt_norm[:, :n_numeric] = (X_tgt[:, :n_numeric] - tgt_mean) / tgt_std

    # Train/val split
    n_val = int(len(X_cond) * 0.1)
    perm = np.random.permutation(len(X_cond))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # ---- Build models ----
    log(f"  [{exp_id}] Building models...")

    # Import model classes
    from diffusion_v3_diagnostic import FeatureDDPM, ProtestClassifier

    ddpm = FeatureDDPM(
        feature_dim, n_lag, lookback,
        hidden_dim=config["DIFF_HIDDEN"],
        n_layers=config["DIFF_LAYERS"],
    ).to(device)

    clf_hidden = config.get("CLF_HIDDEN", 128)
    clf_layers = config.get("CLF_LAYERS", 2)

    if clf_layers == 2:
        classifier = ProtestClassifier(
            feature_dim, n_lag, lookback, hidden_dim=clf_hidden
        ).to(device)
    else:
        # Custom larger classifier
        from diffusion_v3_diagnostic import TemporalEncoder
        class LargerClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.temporal_encoder = TemporalEncoder(feature_dim, n_lag, lookback, embed_dim=128)
                self.net = nn.Sequential(
                    nn.Linear(128, clf_hidden), nn.LayerNorm(clf_hidden), nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(clf_hidden, clf_hidden), nn.LayerNorm(clf_hidden), nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(clf_hidden, clf_hidden // 2), nn.LayerNorm(clf_hidden // 2), nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(clf_hidden // 2, 1),
                )
            def forward(self, x_flat):
                emb = self.temporal_encoder(x_flat)
                return self.net(emb)
        classifier = LargerClassifier().to(device)

    n_params_ddpm = sum(p.numel() for p in ddpm.parameters())
    n_params_clf = sum(p.numel() for p in classifier.parameters())
    wandb.log({"n_params_ddpm": n_params_ddpm, "n_params_clf": n_params_clf})
    log(f"  [{exp_id}] DDPM params: {n_params_ddpm:,}, Classifier params: {n_params_clf:,}")

    # ---- Train DDPM (same for all retrain experiments) ----
    log(f"  [{exp_id}] Training DDPM ({config['DIFF_EPOCHS']} epochs)...")

    # Cosine schedule
    T = config["DIFF_TIMESTEPS"]
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=config["DIFF_LR"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["DIFF_EPOCHS"])

    X_tgt_t = torch.tensor(X_tgt_norm, device=device)
    X_cond_t = torch.tensor(X_cond_norm, device=device)

    train_start = time.time()
    diagnostics = []

    for epoch in range(config["DIFF_EPOCHS"]):
        ddpm.train()
        perm_e = torch.randperm(len(train_idx), device=device)
        epoch_losses = []

        for batch_idx in range(max(1, len(train_idx) // config["DIFF_BATCH"])):
            start = batch_idx * config["DIFF_BATCH"]
            end = min(start + config["DIFF_BATCH"], len(train_idx))
            idx = train_idx[perm_e[start:end].cpu().numpy()]

            x0 = X_tgt_t[idx]
            x_cond = X_cond_t[idx]
            batch_size_actual = len(idx)

            t = torch.randint(0, T, (batch_size_actual, 1), device=device)
            noise = torch.randn_like(x0)
            ab_t = alpha_bar[t]
            x_noisy = torch.sqrt(ab_t) * x0 + torch.sqrt(1 - ab_t) * noise

            pred_noise = ddpm(x_noisy, x_cond, t.float())
            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        # Validation
        ddpm.eval()
        with torch.no_grad():
            val_x0 = X_tgt_t[val_idx]
            val_cond = X_cond_t[val_idx]
            t_val = torch.randint(0, T, (len(val_idx), 1), device=device)
            noise_val = torch.randn_like(val_x0)
            ab_val = alpha_bar[t_val]
            x_noisy_val = torch.sqrt(ab_val) * val_x0 + torch.sqrt(1 - ab_val) * noise_val
            pred_val = ddpm(x_noisy_val, val_cond, t_val.float())
            val_loss = nn.functional.mse_loss(pred_val, noise_val).item()

        train_loss = np.mean(epoch_losses)
        diag = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0], "elapsed_s": round(time.time() - train_start, 1)}
        diagnostics.append(diag)
        wandb.log({"ddpm_train_loss": train_loss, "ddpm_val_loss": val_loss, "epoch": epoch})

        if epoch % 10 == 0:
            log(f"    [{exp_id}] Epoch {epoch}: train={train_loss:.5f}, val={val_loss:.5f}")

    ddpm_time = time.time() - train_start

    # ---- Train Classifier ----
    log(f"  [{exp_id}] Training classifier ({config['CLF_EPOCHS']} epochs)...")

    n_pos = int(Y_labels[train_idx].sum())
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    clf_X_train = torch.tensor(X_cond_norm[train_idx], device=device)
    clf_Y_train = torch.tensor(Y_labels[train_idx], device=device).unsqueeze(1)
    clf_X_val = torch.tensor(X_cond_norm[val_idx], device=device)
    clf_Y_val = torch.tensor(Y_labels[val_idx], device=device).unsqueeze(1)

    clf_optimizer = torch.optim.AdamW(classifier.parameters(), lr=config["CLF_LR"], weight_decay=1e-4)
    clf_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(clf_optimizer, T_max=config["CLF_EPOCHS"])

    # Loss function: standard BCE or focal loss
    use_focal = config.get("USE_FOCAL_LOSS", False)
    focal_gamma = config.get("FOCAL_GAMMA", 2.0)
    focal_alpha = config.get("FOCAL_ALPHA", 0.25)

    if use_focal:
        def focal_loss(logits, targets, gamma=focal_gamma, alpha=focal_alpha):
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            probs = torch.sigmoid(logits)
            pt = targets * probs + (1 - targets) * (1 - probs)
            focal_weight = (1 - pt) ** gamma
            alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
            return (alpha_t * focal_weight * bce).mean()
        loss_fn = focal_loss
        log(f"  [{exp_id}] Using focal loss (gamma={focal_gamma}, alpha={focal_alpha})")
    else:
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_fn = bce_loss_fn

    clf_start = time.time()
    for epoch in range(config["CLF_EPOCHS"]):
        classifier.train()
        perm_c = torch.randperm(len(clf_X_train), device=device)
        epoch_losses = []

        for batch_idx in range(max(1, len(clf_X_train) // config["CLF_BATCH"])):
            start = batch_idx * config["CLF_BATCH"]
            end = min(start + config["CLF_BATCH"], len(clf_X_train))
            idx = perm_c[start:end]

            logits = classifier(clf_X_train[idx])
            loss = loss_fn(logits, clf_Y_train[idx])

            clf_optimizer.zero_grad()
            loss.backward()
            clf_optimizer.step()
            epoch_losses.append(loss.item())

        clf_scheduler.step()
        wandb.log({"clf_train_loss": np.mean(epoch_losses), "clf_epoch": epoch})

    clf_time = time.time() - clf_start

    # ---- Optional: Finetune on generated features (exp05) ----
    if config.get("FINETUNE_ON_GENERATED", False):
        log(f"  [{exp_id}] Finetuning classifier on generated features...")
        from diffusion_v3_diagnostic import generate_features_ddim

        # Set up DDIM
        ddim_timesteps = torch.linspace(T - 1, 0, config["DDIM_STEPS"], device=device).long()

        # Generate features for training data
        ddpm.eval()
        gen_batch = 5000
        all_gen = []
        for i in range(0, len(clf_X_train), gen_batch):
            batch = clf_X_train[i:i+gen_batch]
            x = torch.randn(len(batch), feature_dim, device=device)
            for j in range(len(ddim_timesteps)):
                t_idx = ddim_timesteps[j]
                t = torch.full((len(batch), 1), t_idx.item(), device=device)
                pred_noise = ddpm(x, batch, t)
                ab_t = alpha_bar[t_idx]
                x0_pred = (x - torch.sqrt(1 - ab_t) * pred_noise) / torch.sqrt(ab_t)
                if j < len(ddim_timesteps) - 1:
                    ab_next = alpha_bar[ddim_timesteps[j + 1]]
                    x = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * pred_noise
                else:
                    x = x0_pred
            all_gen.append(x)

        gen_features = torch.cat(all_gen, dim=0)

        # Replace most recent features with generated
        clf_X_gen = clf_X_train.clone()
        clf_X_gen[:, :feature_dim] = gen_features

        # Fine-tune for 10 epochs
        ft_optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
        for epoch in range(10):
            classifier.train()
            perm_ft = torch.randperm(len(clf_X_gen), device=device)
            for batch_idx in range(max(1, len(clf_X_gen) // config["CLF_BATCH"])):
                start = batch_idx * config["CLF_BATCH"]
                end = min(start + config["CLF_BATCH"], len(clf_X_gen))
                idx = perm_ft[start:end]
                logits = classifier(clf_X_gen[idx])
                loss = loss_fn(logits, clf_Y_train[idx])
                ft_optimizer.zero_grad()
                loss.backward()
                ft_optimizer.step()

    train_time = ddpm_time + clf_time + (time.time() - clf_start if config.get("FINETUNE_ON_GENERATED") else 0)

    # ---- Save checkpoint ----
    ckpt_path = os.path.join(exp_dir, "model_checkpoint.pt")
    torch.save({
        "ddpm_state_dict": ddpm.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "cond_mean": cond_mean, "cond_std": cond_std,
        "tgt_mean": tgt_mean, "tgt_std": tgt_std,
        "feature_dim": feature_dim, "cond_dim": cond_dim,
        "n_params_ddpm": n_params_ddpm, "n_params_clf": n_params_clf,
        "train_time_s": train_time, "config": config,
    }, ckpt_path)
    wandb.save(ckpt_path)

    # Save training diagnostics
    with open(os.path.join(exp_dir, "training_diagnostics.jsonl"), "w") as f:
        for d in diagnostics:
            f.write(json.dumps(d) + "\n")

    # ---- Inference on eval years ----
    log(f"  [{exp_id}] Running inference on eval years...")
    ddim_timesteps = torch.linspace(T - 1, 0, config["DDIM_STEPS"], device=device).long()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    all_metrics = {}
    all_scores_out = []

    for eval_year in config["EVAL_YEARS"]:
        log(f"  [{exp_id}] Evaluating {eval_year}...")

        # Build test data
        test_pids = []
        test_conds = []
        test_actuals = []

        for pid, years in rows_by_pid.items():
            if eval_year in years:
                test_pids.append(pid)
                cond = build_conditioning(pid, eval_year)
                test_conds.append(cond)
                test_actuals.append(float(years[eval_year].get(TARGET, 0) or 0))

        if len(test_pids) > config["MAX_EVAL_PARCELS"]:
            sel = np.random.choice(len(test_pids), config["MAX_EVAL_PARCELS"], replace=False)
            test_pids = [test_pids[i] for i in sel]
            test_conds = [test_conds[i] for i in sel]
            test_actuals = [test_actuals[i] for i in sel]

        test_conds_arr = np.array(test_conds, dtype=np.float32)
        test_conds_arr[:, cond_numeric_indices] = (test_conds_arr[:, cond_numeric_indices] - cond_mean) / cond_std
        test_actuals_arr = np.array(test_actuals)

        # LogReg baseline
        scaler = StandardScaler()
        lr_X_train = scaler.fit_transform(X_cond[train_idx])
        lr_X_test = scaler.transform(test_conds_arr)
        lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr_model.fit(lr_X_train, Y_labels[train_idx])
        lr_probs = lr_model.predict_proba(lr_X_test)[:, 1]

        # Diffusion inference
        ddpm.eval()
        classifier.eval()
        cond_tensor = torch.tensor(test_conds_arr, device=device)

        batch_gen = 5000
        all_gen = []
        for i in range(0, len(cond_tensor), batch_gen):
            batch = cond_tensor[i:i+batch_gen]
            gen_scenarios = []
            for s in range(config["N_SCENARIOS"]):
                x = torch.randn(len(batch), feature_dim, device=device)
                for j in range(len(ddim_timesteps)):
                    t_idx = ddim_timesteps[j]
                    t = torch.full((len(batch), 1), t_idx.item(), device=device)
                    pred_noise = ddpm(x, batch, t)
                    ab_t = alpha_bar[t_idx]
                    x0_pred = (x - torch.sqrt(1 - ab_t) * pred_noise) / torch.sqrt(ab_t)
                    if j < len(ddim_timesteps) - 1:
                        ab_next = alpha_bar[ddim_timesteps[j + 1]]
                        x = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * pred_noise
                    else:
                        x = x0_pred
                gen_scenarios.append(x)
            all_gen.append(torch.stack(gen_scenarios, dim=1))
        gen_features = torch.cat(all_gen, dim=0)

        # Classify
        with torch.no_grad():
            all_probs = []
            for s in range(config["N_SCENARIOS"]):
                clf_input = cond_tensor.clone()
                clf_input[:, :feature_dim] = gen_features[:, s, :]
                logits = classifier(clf_input)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.append(probs)
            diff_probs = np.mean(all_probs, axis=0)

        # Ensemble
        ens_probs = 0.5 * lr_probs + 0.5 * diff_probs

        # Metrics
        lr_m = compute_metrics(test_actuals_arr, lr_probs, f"lr_{eval_year}")
        diff_m = compute_metrics(test_actuals_arr, diff_probs, f"diff_{eval_year}")
        ens_m = compute_metrics(test_actuals_arr, ens_probs, f"ens_{eval_year}")

        all_metrics[eval_year] = {"lr": lr_m, "diff": diff_m, "ensemble": ens_m}

        wandb.log({
            f"eval/auc_roc_diff/{eval_year}": diff_m.get("auc_roc", 0),
            f"eval/ece_diff/{eval_year}": diff_m.get("ece", 0),
            f"eval/brier_diff/{eval_year}": diff_m.get("brier", 0),
            f"eval/auc_roc_lr/{eval_year}": lr_m.get("auc_roc", 0),
        })

        log(f"    [{exp_id}] {eval_year}: Diff AUC={diff_m.get('auc_roc',0):.4f}, "
            f"ECE={diff_m.get('ece',0):.5f}, Brier={diff_m.get('brier',0):.5f}")

        for i, pid in enumerate(test_pids):
            all_scores_out.append({
                "pid": pid, "year": eval_year,
                "lr": round(lr_probs[i], 6), "diff": round(diff_probs[i], 6),
                "ens": round(ens_probs[i], 6), "actual": int(test_actuals_arr[i]),
            })

    # Save metrics and scores
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    csv_path = os.path.join(exp_dir, "per_parcel_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pid", "year", "lr", "diff", "ens", "actual"])
        w.writeheader()
        for s in all_scores_out:
            w.writerow(s)

    wandb.save(os.path.join(exp_dir, "metrics.json"))
    wandb.save(csv_path)
    wandb.finish()

    return all_metrics


# =============================================
# EXPERIMENT DEFINITIONS
# =============================================

EXPERIMENTS = {
    "exp01_platt": {"func": run_exp01_platt, "wave": 1},
    "exp02_isotonic": {"func": run_exp02_isotonic, "wave": 1},
    "exp06_adaptive": {"func": run_exp06_adaptive, "wave": 1},
    "exp03_focal": {
        "func": lambda d: run_retrain_experiment("exp03_focal", d, {
            "USE_FOCAL_LOSS": True, "FOCAL_GAMMA": 2.0, "FOCAL_ALPHA": 0.25,
            "wave": "wave2",
        }),
        "wave": 2,
    },
    "exp04_larger_clf": {
        "func": lambda d: run_retrain_experiment("exp04_larger_clf", d, {
            "CLF_HIDDEN": 256, "CLF_LAYERS": 3,
            "wave": "wave2",
        }),
        "wave": 2,
    },
    "exp05_clf_on_gen": {
        "func": lambda d: run_retrain_experiment("exp05_clf_on_gen", d, {
            "FINETUNE_ON_GENERATED": True,
            "wave": "wave3",
        }),
        "wave": 3,
    },
}


def run_experiment(exp_id):
    """Run a single experiment with error handling."""
    exp_dir = os.path.join(EXP_BASE_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    start = time.time()
    log(f"\n{'='*60}")
    log(f"STARTING: {exp_id}")
    log(f"{'='*60}")

    try:
        result = EXPERIMENTS[exp_id]["func"](exp_dir)
        elapsed = time.time() - start
        log(f"\n  COMPLETED: {exp_id} in {elapsed:.1f}s")

        # Append to experiment log
        with open(os.path.join(EXP_BASE_DIR, "experiment_log.jsonl"), "a") as f:
            f.write(json.dumps({
                "experiment": exp_id,
                "status": "success",
                "elapsed_s": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            }) + "\n")

        return {"exp_id": exp_id, "status": "success", "elapsed_s": elapsed}

    except Exception as e:
        elapsed = time.time() - start
        log(f"\n  FAILED: {exp_id} after {elapsed:.1f}s: {str(e)}")
        import traceback
        traceback.print_exc()

        with open(os.path.join(EXP_BASE_DIR, "experiment_log.jsonl"), "a") as f:
            f.write(json.dumps({
                "experiment": exp_id,
                "status": "failed",
                "error": str(e),
                "elapsed_s": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            }) + "\n")

        return {"exp_id": exp_id, "status": "failed", "error": str(e)}


def run_all_waves(max_parallel=2):
    """Run all experiments in waves."""
    waves = defaultdict(list)
    for exp_id, info in EXPERIMENTS.items():
        waves[info["wave"]].append(exp_id)

    all_results = []
    for wave_num in sorted(waves):
        exp_ids = waves[wave_num]
        log(f"\n{'#'*60}")
        log(f"WAVE {wave_num}: {', '.join(exp_ids)}")
        log(f"{'#'*60}")

        # Run experiments in this wave in parallel
        with ProcessPoolExecutor(max_workers=min(max_parallel, len(exp_ids))) as executor:
            futures = {executor.submit(run_experiment, eid): eid for eid in exp_ids}
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                log(f"  Wave {wave_num} result: {result['exp_id']} = {result['status']}")

    return all_results


# =============================================
# COMPARISON TABLE
# =============================================

def generate_comparison():
    """Generate a comparison table from all experiment results."""
    results = []
    for exp_id in EXPERIMENTS:
        metrics_path = os.path.join(EXP_BASE_DIR, exp_id, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            results.append({"exp_id": exp_id, "metrics": metrics})

    if not results:
        log("No results to compare.")
        return

    # Build comparison CSV
    csv_path = os.path.join(EXP_BASE_DIR, "comparison_table.csv")
    rows = []
    for r in results:
        for year, year_metrics in r["metrics"].items():
            for model_name, m in year_metrics.items():
                if isinstance(m, dict) and "auc_roc" in m:
                    rows.append({
                        "experiment": r["exp_id"],
                        "year": year,
                        "model": model_name,
                        "auc_roc": m.get("auc_roc", ""),
                        "auc_pr": m.get("auc_pr", ""),
                        "brier": m.get("brier", ""),
                        "ece": m.get("ece", ""),
                        "recall": m.get("recall", ""),
                        "f1": m.get("f1", ""),
                    })

    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    log(f"\nComparison table saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", help="Run specific experiment")
    parser.add_argument("--wave", type=int, help="Run specific wave")
    parser.add_argument("--parallel", type=int, default=2, help="Max parallel runs")
    parser.add_argument("--compare-only", action="store_true", help="Only generate comparison")
    args = parser.parse_args()

    if args.compare_only:
        generate_comparison()
    elif args.experiment:
        if args.experiment in EXPERIMENTS:
            run_experiment(args.experiment)
        else:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            sys.exit(1)
    elif args.wave:
        wave_exps = [eid for eid, info in EXPERIMENTS.items() if info["wave"] == args.wave]
        if not wave_exps:
            print(f"No experiments in wave {args.wave}")
            sys.exit(1)
        for eid in wave_exps:
            run_experiment(eid)
    else:
        results = run_all_waves(max_parallel=args.parallel)
        generate_comparison()
        log("\n" + "="*60)
        log("ALL EXPERIMENTS COMPLETE")
        for r in results:
            log(f"  {r['exp_id']}: {r['status']}")
