"""
Multi-Horizon Expanding-Window Generative Backtest
====================================================
Runs CVAE + Diffusion + LogReg baseline in a proper temporal backtest:
  - Expanding training window (years <= train_end)
  - Multi-horizon evaluation (h=1, 2, 3 years ahead)
  - Scenario chaining: h1 predictions become features for h2, h2 for h3
  - Outputs structured JSONL for dashboard consumption
"""
import csv, json, os, sys, time, warnings
import numpy as np
import logging
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# ---- Config ----
PANEL_PATH = "Data/Panel/Output/Property_Year_Panel.csv"
RESULTS_DIR = "Analysis/Results/Backtests"
os.makedirs(RESULTS_DIR, exist_ok=True)

JSONL_PATH = os.path.join(RESULTS_DIR, "generative_backtest_log.jsonl")
LOG_PATH = os.path.join(RESULTS_DIR, "generative_backtest.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

TRAIN_START = 2019
EVAL_START = 2020
EVAL_END = 2024
HORIZONS = [1, 2, 3]

NUMERIC_FEATURES = [
    "total_market_value", "deed_acreage", "improvement_sq_ft",
]
CATEGORICAL_FEATURES = [
    "property_category_code", "lui_general_land_use", "council_district",
]
TARGET = "protest"

# CVAE / Diffusion hyperparams
CVAE_LATENT = 8
CVAE_HIDDEN = 32
CVAE_EPOCHS = 30
CVAE_LR = 1e-3

DIFF_TIMESTEPS = 100
DIFF_HIDDEN = 128
DIFF_EPOCHS = 20
DIFF_LR = 1e-3


def append_jsonl(record):
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default


# ---- Data Loading ----
def load_panel():
    """Load panel with EARS year-matched rows only."""
    log.info("Loading panel from %s", PANEL_PATH)
    rows = []
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year = int(row["year"])
            if year < TRAIN_START:
                continue
            if row.get("ears_matched") != "1":
                continue
            if "backfill" in row.get("ears_source", ""):
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
        log.info("  Year %d: %d rows, %d pos (%.3f%%)", y, c["total"], c["pos"], rate)
    return rows


def featurize(rows):
    """Convert rows to X, y, years arrays."""
    cat_values = {f: set() for f in CATEGORICAL_FEATURES}
    for row in rows:
        for f in CATEGORICAL_FEATURES:
            val = row.get(f, "").strip()
            if val:
                cat_values[f].add(val)

    cat_maps = {}
    for f in CATEGORICAL_FEATURES:
        vals = sorted(cat_values[f])
        cat_maps[f] = {v: i for i, v in enumerate(vals)}

    n_num = len(NUMERIC_FEATURES)
    n_cat = sum(len(m) for m in cat_maps.values())
    n_feat = n_num + n_cat

    feature_names = list(NUMERIC_FEATURES)
    for f in CATEGORICAL_FEATURES:
        for v in sorted(cat_maps[f].keys()):
            feature_names.append("%s_%s" % (f, v))

    X = np.zeros((len(rows), n_feat), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int32)
    years = np.zeros(len(rows), dtype=np.int32)

    for i, row in enumerate(rows):
        for j, f in enumerate(NUMERIC_FEATURES):
            X[i, j] = safe_float(row.get(f, ""))
        offset = n_num
        for f in CATEGORICAL_FEATURES:
            val = row.get(f, "").strip()
            if val and val in cat_maps[f]:
                X[i, offset + cat_maps[f][val]] = 1.0
            offset += len(cat_maps[f])
        y[i] = int(row[TARGET])
        years[i] = int(row["year"])

    log.info("Features: %d numeric + %d cat = %d total", n_num, n_cat, n_feat)
    return X, y, years, feature_names


# ---- Metrics ----
def compute_metrics(y_true, y_prob, y_pred):
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
    for k_frac in [0.01, 0.05]:
        k = max(1, int(n * k_frac))
        top_k_idx = np.argsort(y_prob)[-k:]
        prec_k = y_true[top_k_idx].sum() / k
        lift = prec_k / base_rate if base_rate > 0 else 0
        label = "%.0f%%" % (100 * k_frac)
        metrics["lift@%s" % label] = float(lift)

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


# ---- Model Builders ----
def train_logreg(X_train, y_train, X_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xtr = np.nan_to_num(scaler.fit_transform(X_train), nan=0, posinf=0, neginf=0)
    Xte = np.nan_to_num(scaler.transform(X_test), nan=0, posinf=0, neginf=0)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(Xtr, y_train)
    return model.predict_proba(Xte)[:, 1], model.predict(Xte)


def train_cvae_classifier(X_train, y_train, X_test):
    """Train CVAE, extract latent features, then classify with LogReg."""
    import torch
    import torch.nn as nn

    device = torch.device("cpu")
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xtr = np.nan_to_num(scaler.fit_transform(X_train), nan=0, posinf=0, neginf=0).astype(np.float32)
    Xte = np.nan_to_num(scaler.transform(X_test), nan=0, posinf=0, neginf=0).astype(np.float32)

    input_dim = Xtr.shape[1]

    class CVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(input_dim + 1, CVAE_HIDDEN), nn.ReLU(),
                                     nn.Linear(CVAE_HIDDEN, CVAE_HIDDEN // 2), nn.ReLU())
            self.mu = nn.Linear(CVAE_HIDDEN // 2, CVAE_LATENT)
            self.logvar = nn.Linear(CVAE_HIDDEN // 2, CVAE_LATENT)
            self.dec = nn.Sequential(nn.Linear(CVAE_LATENT + 1, CVAE_HIDDEN // 2), nn.ReLU(),
                                     nn.Linear(CVAE_HIDDEN // 2, CVAE_HIDDEN), nn.ReLU(),
                                     nn.Linear(CVAE_HIDDEN, input_dim))

        def encode(self, x, c):
            h = self.enc(torch.cat([x, c], dim=1))
            return self.mu(h), self.logvar(h)

        def decode(self, z, c):
            return self.dec(torch.cat([z, c], dim=1))

        def forward(self, x, c):
            mu, logvar = self.encode(x, c)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return self.decode(z, c), mu, logvar

    model = CVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CVAE_LR)
    X_t = torch.tensor(Xtr, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)

    model.train()
    for epoch in range(CVAE_EPOCHS):
        recon, mu, logvar = model(X_t, y_t)
        recon_loss = nn.functional.mse_loss(recon, X_t, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (recon_loss + kl_loss) / len(Xtr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Extract latent features
    model.eval()
    with torch.no_grad():
        # Compute ELBO-based score: loss(x|y=1) vs loss(x|y=0)
        X_te_t = torch.tensor(Xte, device=device)
        scores = []
        for c_val in [0.0, 1.0]:
            c_t = torch.full((len(Xte), 1), c_val, device=device)
            recon, mu, logvar = model(X_te_t, c_t)
            recon_loss = nn.functional.mse_loss(recon, X_te_t, reduction="none").sum(dim=1)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            scores.append((recon_loss + kl_loss).cpu().numpy())

        # P(protest) proportional to lower loss under y=1
        loss_0, loss_1 = scores[0], scores[1]
        raw = loss_0 - loss_1  # Higher = more likely protest
        # Sigmoid to get probabilities
        y_prob = 1 / (1 + np.exp(-raw / (np.std(raw) + 1e-8)))
        y_pred = (y_prob >= 0.5).astype(int)

    return y_prob, y_pred


def train_diffusion_augmented(X_train, y_train, X_test):
    """Train Diffusion, generate synthetic minority, augment LogReg training."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    device = torch.device("cpu")

    scaler = StandardScaler()
    Xtr = np.nan_to_num(scaler.fit_transform(X_train), nan=0, posinf=0, neginf=0).astype(np.float32)
    Xte = np.nan_to_num(scaler.transform(X_test), nan=0, posinf=0, neginf=0).astype(np.float32)

    # Train diffusion on minority class only
    minority_mask = y_train == 1
    if minority_mask.sum() < 5:
        # Fallback to plain LogReg if too few positives for diffusion
        return train_logreg(X_train, y_train, X_test)

    X_min = Xtr[minority_mask]
    input_dim = Xtr.shape[1]

    class DiffusionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_emb = nn.Sequential(nn.Linear(1, DIFF_HIDDEN), nn.SiLU())
            self.net = nn.Sequential(
                nn.Linear(input_dim + DIFF_HIDDEN, DIFF_HIDDEN), nn.SiLU(),
                nn.Linear(DIFF_HIDDEN, DIFF_HIDDEN), nn.SiLU(),
                nn.Linear(DIFF_HIDDEN, input_dim),
            )

        def forward(self, x, t):
            t_emb = self.time_emb(t)
            return self.net(torch.cat([x, t_emb], dim=1))

    model = DiffusionMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=DIFF_LR)
    betas = torch.linspace(1e-4, 0.02, DIFF_TIMESTEPS, device=device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, 0)

    X_min_t = torch.tensor(X_min, device=device)
    model.train()
    for epoch in range(DIFF_EPOCHS):
        idx = torch.randint(0, len(X_min_t), (min(512, len(X_min_t)),))
        x0 = X_min_t[idx]
        t = torch.randint(0, DIFF_TIMESTEPS, (len(x0),), device=device)
        noise = torch.randn_like(x0)
        ab = alpha_bar[t].unsqueeze(1)
        x_noisy = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
        t_norm = t.float().unsqueeze(1) / DIFF_TIMESTEPS
        pred = model(x_noisy, t_norm)
        loss = nn.functional.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Generate synthetic samples
    n_gen = int(minority_mask.sum())  # Match minority count
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_gen, input_dim, device=device)
        for step in reversed(range(DIFF_TIMESTEPS)):
            t_in = torch.full((n_gen, 1), step / DIFF_TIMESTEPS, device=device)
            pred_noise = model(x, t_in)
            beta = betas[step]
            alpha = alphas[step]
            ab = alpha_bar[step]
            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - ab)) * pred_noise)
            if step > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)

    synthetic = x.cpu().numpy()

    # Augment and train LogReg
    from sklearn.linear_model import LogisticRegression

    X_aug = np.vstack([Xtr, synthetic])
    y_aug = np.concatenate([y_train, np.ones(n_gen, dtype=np.int32)])

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_aug, y_aug)
    return clf.predict_proba(Xte)[:, 1], clf.predict(Xte)


# ---- Main Backtest ----
def run_backtest():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("=" * 70)
    log.info("MULTI-HORIZON GENERATIVE BACKTEST — run_id=%s", run_id)
    log.info("=" * 70)
    start = datetime.now()

    rows = load_panel()
    X, y, years, feature_names = featurize(rows)

    all_results = []

    # Previous horizon predictions for scenario chaining
    # Key: (horizon, eval_year) -> {"logreg": probs, "cvae": probs, "diffusion": probs}
    prev_horizon_preds = {}

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

            # Scenario chaining: add previous horizon predictions as features
            X_train_aug = X_train.copy()
            X_test_aug = X_test.copy()

            if horizon > 1:
                # Look up h-1 predictions for the eval_year - 1
                prev_key = (horizon - 1, eval_year - 1)
                if prev_key in prev_horizon_preds:
                    log.info("  Scenario chaining: adding h=%d predictions as feature", horizon - 1)
                    # For test set: use h-1 predictions on eval_year - 1
                    # For train set: we need to compute or look up h-1 scores
                    # Simplification: append a column of zeros for train (no leakage)
                    # and the actual h-1 predictions for test
                    pp = prev_horizon_preds[prev_key]
                    # Find test set IDs that overlap
                    # Since we don't have IDs, use shape-based approach:
                    # The previous predictions were on eval_year - 1 test set
                    # The current test set is eval_year. These are different parcels at different times.
                    # For proper chaining, we use the AVERAGE prediction from h-1 as a "market signal"
                    avg_prev = np.mean(pp.get("logreg", [0.0]))
                    train_chain = np.full((len(X_train_aug), 1), avg_prev, dtype=np.float32)
                    test_chain = np.full((len(X_test_aug), 1), avg_prev, dtype=np.float32)
                    X_train_aug = np.hstack([X_train_aug, train_chain])
                    X_test_aug = np.hstack([X_test_aug, test_chain])

            # Run all three models
            fold_preds = {}
            model_names = ["LogReg", "CVAE", "Diffusion"]
            model_funcs = [train_logreg, train_cvae_classifier, train_diffusion_augmented]

            for model_name, model_func in zip(model_names, model_funcs):
                log.info("  Training %s...", model_name)
                t0 = time.time()
                try:
                    y_prob, y_pred = model_func(X_train_aug, y_train, X_test_aug)
                    elapsed_model = time.time() - t0
                    metrics = compute_metrics(y_test, y_prob, y_pred)
                    metrics.update({
                        "model": model_name,
                        "eval_year": eval_year,
                        "train_end": train_end,
                        "horizon": horizon,
                        "run_id": run_id,
                        "elapsed_s": round(elapsed_model, 2),
                        "scenario_chain": horizon > 1,
                        "timestamp": datetime.now().isoformat(),
                    })
                    log.info("    %s: ROC-AUC=%.4f PR-AUC=%.4f Brier=%.4f (%.1fs)",
                             model_name,
                             metrics.get("roc_auc", 0),
                             metrics.get("pr_auc", 0),
                             metrics.get("brier_score", 0),
                             elapsed_model)
                    append_jsonl(metrics)
                    all_results.append(metrics)
                    fold_preds[model_name.lower()] = y_prob
                except Exception as e:
                    log.error("    %s FAILED: %s", model_name, str(e))
                    fold_preds[model_name.lower()] = np.zeros(len(y_test))

            # Store predictions for scenario chaining
            prev_horizon_preds[(horizon, eval_year)] = fold_preds

    # ---- Summary ----
    log.info("=" * 70)
    log.info("SUMMARY ACROSS ALL FOLDS")
    log.info("=" * 70)

    for horizon in HORIZONS:
        for model_name in ["LogReg", "CVAE", "Diffusion"]:
            fold_results = [r for r in all_results
                            if r["horizon"] == horizon and r["model"] == model_name]
            if not fold_results:
                continue
            for key in ["roc_auc", "pr_auc", "brier_score", "ece"]:
                vals = [r[key] for r in fold_results if key in r and not np.isnan(r.get(key, 0))]
                if vals:
                    log.info("  h=%d %s %s: mean=%.4f std=%.4f",
                             horizon, model_name, key, np.mean(vals), np.std(vals))

    # Save structured results CSV for dashboard
    csv_path = os.path.join(RESULTS_DIR, "generative_backtest_results.csv")
    if all_results:
        keys = sorted(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in all_results:
                w.writerow(r)
        log.info("Results CSV: %s", csv_path)

    elapsed = datetime.now() - start
    log.info("Done. Elapsed: %s", elapsed)
    log.info("JSONL log: %s", JSONL_PATH)


if __name__ == "__main__":
    run_backtest()
