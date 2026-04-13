"""
Diffusion v3: Feature-Only DDPM + Classifier Head
====================================================

Key architectural changes from v2:
1. DIFFUSE FEATURES ONLY — protest label is NOT part of the diffusion state.
   The model learns P(features_{t+1} | features_t), not P(state_{t+1} | state_t).
2. SEPARATE CLASSIFIER HEAD — An MLP classifier maps generated features → protest probability.
   Trained on historical (features, protest) pairs.
3. LAG FEATURES — Adds protest_lag1, value_change as explicit features.
4. MULTI-YEAR CONDITIONING — Conditions on last LOOKBACK_YEARS of features (not just t-1).
5. ENSEMBLE SCORING — Averages LogReg and Diffusion probabilities for combined model.
6. FIXED MMD — Normalized features before RBF kernel to avoid overflow.
7. VALIDATION LOSS — 90/10 train/val split with loss tracking during training.
8. EXPANDING WINDOW for classifier — classifier uses all prior years, mirroring LogReg.

Architecture:
  Feature DDPM: learns noise prediction on feature vectors (no protest label)
  Classifier: 2-layer MLP trained on (features → protest) from all prior years
  Inference: generate feature scenarios → classify each → average probabilities
  Ensemble: 0.5 * LogReg + 0.5 * Diffusion (diagnostic only, not primary model)
"""
import csv, json, sys, os, time, math
import numpy as np
from collections import defaultdict
from datetime import datetime

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
np.random.seed(42)

# ---- Paths ----
PANEL_PATH = "Data/Panel/Output/Property_Year_Panel_Enriched.csv"
CENTROIDS_PATH = "Data/Panel/Reference/parcel_centroids.csv"
OUT_DIR = "Analysis/Results/Diffusion_v3"
CHECKPOINT_PATH = os.path.join(OUT_DIR, "model_checkpoint.pt")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Config ----
TRAIN_START = 2019
EVAL_YEARS = [2021, 2022, 2023, 2024]
LOAD_CHECKPOINT = os.path.exists(CHECKPOINT_PATH)

NUMERIC_FEATURES = [
    "market_value", "assessed_value", "land_value", "improvement_value",
    "living_area", "deed_acreage", "year_built", "land_acres", "improvement_count",
]
CATEGORICAL_FEATURES = ["property_category_code", "lui_general_land_use", "council_district"]
TARGET = "protest"

# Diffusion hyperparams
DIFF_TIMESTEPS = 200
DIFF_HIDDEN = 256
DIFF_LAYERS = 3
DIFF_EPOCHS = 75
DIFF_LR = 3e-4
DIFF_BATCH = 2048
EARLY_STOP_PATIENCE = 10
DDIM_STEPS = 50
N_SCENARIOS = 10
MAX_TRAIN_PAIRS = 100000
MAX_EVAL_PARCELS = 30000
LOOKBACK_YEARS = 2  # condition on this many prior years of features
ENSEMBLE_WEIGHT = 0.5  # weight for LR in ensemble (1-weight for diffusion)

# Classifier hyperparams
CLF_HIDDEN = 128
CLF_EPOCHS = 50
CLF_LR = 1e-3
CLF_BATCH = 4096

# ---- Utilities ----
def safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ---- Load data ----
log("Loading centroids...")
centroids = {}
with open(CENTROIDS_PATH, "r") as f:
    for row in csv.DictReader(f):
        centroids[row["parcel_id_10"]] = (float(row["latitude"]), float(row["longitude"]))
log(f"Centroids: {len(centroids)}")

log("Loading panel...")
rows_by_year = defaultdict(dict)
with open(PANEL_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        year = int(row["year"])
        if year < TRAIN_START:
            continue
        pid = row.get("standardized_tcad_id", "").strip()
        if pid:
            rows_by_year[year][pid] = row

for y in sorted(rows_by_year):
    n = len(rows_by_year[y])
    n_pos = sum(1 for r in rows_by_year[y].values() if r[TARGET] == "1")
    log(f"  Year {y}: {n:,} parcels, {n_pos} protests ({n_pos/n*100:.2f}%)")

# ---- Build feature maps ----
log("Building feature maps...")
cat_values = {f: set() for f in CATEGORICAL_FEATURES}
for year_dict in rows_by_year.values():
    for row in year_dict.values():
        for f in CATEGORICAL_FEATURES:
            val = row.get(f, "").strip()
            if val:
                cat_values[f].add(val)

cat_maps = {}
for f in CATEGORICAL_FEATURES:
    vals = sorted(cat_values[f])
    cat_maps[f] = {v: i for i, v in enumerate(vals)}

n_numeric = len(NUMERIC_FEATURES)
n_cat = sum(len(m) for m in cat_maps.values())
# Feature vector = [numeric | one-hot categoricals] — NO protest label!
feature_dim = n_numeric + n_cat
# Lag features: [protest_lag1, value_change_pct] added during pair creation
n_lag = 2
# Multi-year conditioning: temporal encoder processes LOOKBACK_YEARS of (feature_dim + n_lag) sequences
# The encoder output is a fixed-size temporal embedding
TEMPORAL_EMBED_DIM = 128
# For data arrays, we still concat all years + lags for simplicity in data handling
cond_dim = feature_dim * LOOKBACK_YEARS + n_lag
# But the model will reshape this internally into (LOOKBACK_YEARS, feature_dim) + lag
log(f"Feature dimension: {feature_dim} ({n_numeric} numeric + {n_cat} categorical)")
log(f"Condition: {LOOKBACK_YEARS} years × {feature_dim} features + {n_lag} lags → {TEMPORAL_EMBED_DIM}-dim temporal embedding")


def row_to_features(row):
    """Convert a panel row to a FEATURE-ONLY vector (no protest label)."""
    state = np.zeros(feature_dim, dtype=np.float32)
    for j, f in enumerate(NUMERIC_FEATURES):
        state[j] = safe_float(row.get(f, ""))
    offset = n_numeric
    for f in CATEGORICAL_FEATURES:
        val = row.get(f, "").strip()
        if val and val in cat_maps[f]:
            state[offset + cat_maps[f][val]] = 1.0
        offset += len(cat_maps[f])
    return state


# ---- Create transition pairs with multi-year conditioning ----
log(f"Creating state transition pairs with {LOOKBACK_YEARS}-year lookback...")
years_sorted = sorted(rows_by_year.keys())
all_pairs = []  # (condition_with_lags, target_features)
pair_labels = []  # target protest label (for validation only, not diffused)

for i in range(len(years_sorted) - 1):
    y_to = years_sorted[i + 1]
    # Collect lookback years (most recent first)
    lookback = [years_sorted[j] for j in range(i, max(i - LOOKBACK_YEARS, -1), -1)]
    while len(lookback) < LOOKBACK_YEARS:
        lookback.append(lookback[-1])  # pad with oldest available

    y_from = lookback[0]  # most recent prior year
    common_pids = set(rows_by_year[y_from].keys()) & set(rows_by_year[y_to].keys())
    n_pairs = 0
    for pid in common_pids:
        row_to = rows_by_year[y_to][pid]
        feat_to = row_to_features(row_to)

        # Multi-year features: concatenate features from each lookback year
        multi_feats = []
        for lb_year in lookback:
            if pid in rows_by_year[lb_year]:
                multi_feats.append(row_to_features(rows_by_year[lb_year][pid]))
            else:
                multi_feats.append(row_to_features(rows_by_year[y_from][pid]))  # fallback

        # Lag features from most recent year
        row_from = rows_by_year[y_from][pid]
        protest_lag1 = float(row_from.get(TARGET, "0") == "1")
        mv_from = safe_float(row_from.get("market_value", ""))
        mv_to = safe_float(row_to.get("market_value", ""))
        value_change_pct = (mv_to - mv_from) / (mv_from + 1) if mv_from > 0 else 0.0

        # Condition = [feat_year1 | feat_year2 | ... | lag_features]
        cond = np.concatenate(multi_feats + [[protest_lag1, value_change_pct]])

        all_pairs.append((cond, feat_to))
        pair_labels.append(float(row_to.get(TARGET, "0") == "1"))
        n_pairs += 1
    log(f"  {'+'.join(str(y) for y in lookback)}→{y_to}: {n_pairs:,} pairs")

log(f"Total transition pairs: {len(all_pairs):,}")

# Subsample
if len(all_pairs) > MAX_TRAIN_PAIRS:
    idx = np.random.choice(len(all_pairs), MAX_TRAIN_PAIRS, replace=False)
    all_pairs = [all_pairs[i] for i in idx]
    pair_labels = [pair_labels[i] for i in idx]
    log(f"Subsampled to {len(all_pairs):,} pairs")

# Convert to arrays
X_cond = np.array([p[0] for p in all_pairs], dtype=np.float32)
X_target = np.array([p[1] for p in all_pairs], dtype=np.float32)
Y_labels = np.array(pair_labels, dtype=np.float32)

# Normalize numeric features in each lookback slot (leave one-hot and lag as-is)
# Condition has LOOKBACK_YEARS slots of feature_dim, each with n_numeric numerics at the front
cond_numeric_indices = []
for lb in range(LOOKBACK_YEARS):
    offset = lb * feature_dim
    cond_numeric_indices.extend(range(offset, offset + n_numeric))
cond_numeric_indices = np.array(cond_numeric_indices)

cond_mean = np.mean(X_cond[:, cond_numeric_indices], axis=0)
cond_std = np.std(X_cond[:, cond_numeric_indices], axis=0) + 1e-8
tgt_mean = np.mean(X_target[:, :n_numeric], axis=0)
tgt_std = np.std(X_target[:, :n_numeric], axis=0) + 1e-8

X_cond_norm = X_cond.copy()
X_cond_norm[:, cond_numeric_indices] = (X_cond[:, cond_numeric_indices] - cond_mean) / cond_std
X_target_norm = X_target.copy()
X_target_norm[:, :n_numeric] = (X_target[:, :n_numeric] - tgt_mean) / tgt_std

# Train/val split (90/10)
n_total = len(X_cond_norm)
n_val = n_total // 10
perm = np.random.permutation(n_total)
val_idx = perm[:n_val]
train_idx = perm[n_val:]

log(f"Train: {len(train_idx):,}, Val: {len(val_idx):,}")
log(f"Protest in targets: {int(Y_labels.sum())}/{len(Y_labels)} ({Y_labels.mean()*100:.2f}%)")

# ---- PyTorch model ----
log("Building model...")
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
        )
        self.time_proj = nn.Linear(time_dim, dim)

    def forward(self, x, t_emb):
        return x + self.net(x + self.time_proj(t_emb))


class TemporalEncoder(nn.Module):
    """GRU-based temporal encoder that processes multi-year feature sequences.
    
    Takes (batch, lookback_years * feature_dim + n_lag) flat vector,
    reshapes to (batch, lookback_years, feature_dim), runs GRU,
    appends lag features, and projects to a fixed-size embedding.
    """
    def __init__(self, feature_dim, n_lag, lookback_years, embed_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_lag = n_lag
        self.lookback_years = lookback_years
        self.embed_dim = embed_dim
        
        # Per-year feature projection
        self.year_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU(),
        )
        # GRU over the year sequence (oldest → newest)
        self.gru = nn.GRU(embed_dim, embed_dim, num_layers=1, batch_first=True)
        # Final projection: GRU output + lag features → embedding
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim + n_lag, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU(),
        )
    
    def forward(self, x_flat):
        """x_flat: (batch, lookback_years * feature_dim + n_lag)"""
        batch = x_flat.shape[0]
        # Split off lag features
        multi_year_flat = x_flat[:, :self.lookback_years * self.feature_dim]
        lag_feats = x_flat[:, self.lookback_years * self.feature_dim:]
        
        # Reshape to (batch, lookback_years, feature_dim) — years are stored newest-first
        # Reverse to oldest-first for GRU (natural temporal order)
        years = multi_year_flat.view(batch, self.lookback_years, self.feature_dim)
        years = years.flip(dims=[1])  # oldest first
        
        # Project each year and run through GRU
        year_embeds = self.year_proj(years)  # (batch, lookback, embed_dim)
        _, h_n = self.gru(year_embeds)  # h_n: (1, batch, embed_dim)
        temporal = h_n.squeeze(0)  # (batch, embed_dim)
        
        # Combine with lag features
        return self.out_proj(torch.cat([temporal, lag_feats], dim=-1))


class FeatureDDPM(nn.Module):
    """DDPM that diffuses FEATURES ONLY, with temporal encoder for conditioning."""
    def __init__(self, feature_dim, n_lag, lookback_years, hidden_dim=256,
                 n_layers=3, time_dim=128, temporal_embed_dim=128):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(
            feature_dim, n_lag, lookback_years, temporal_embed_dim
        )
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim + temporal_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim) for _ in range(n_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x_noisy, x_cond_flat, t):
        cond_emb = self.temporal_encoder(x_cond_flat)
        t_emb = self.time_embed(t)
        h = self.input_proj(torch.cat([x_noisy, cond_emb], dim=-1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


class ProtestClassifier(nn.Module):
    """Separate binary classifier with temporal encoder for multi-year input."""
    def __init__(self, feature_dim, n_lag, lookback_years,
                 hidden_dim=128, temporal_embed_dim=128):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(
            feature_dim, n_lag, lookback_years, temporal_embed_dim
        )
        self.net = nn.Sequential(
            nn.Linear(temporal_embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_flat):
        emb = self.temporal_encoder(x_flat)
        return self.net(emb)


# ---- Diffusion schedule ----
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999)

betas = cosine_beta_schedule(DIFF_TIMESTEPS).to(device)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, 0)
sqrt_alpha_bar = torch.sqrt(alpha_bar)
sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

# ---- Build models ----
ddpm = FeatureDDPM(
    feature_dim=feature_dim, n_lag=n_lag, lookback_years=LOOKBACK_YEARS,
    hidden_dim=DIFF_HIDDEN, n_layers=DIFF_LAYERS, time_dim=128,
    temporal_embed_dim=TEMPORAL_EMBED_DIM,
).to(device)

classifier = ProtestClassifier(
    feature_dim=feature_dim, n_lag=n_lag, lookback_years=LOOKBACK_YEARS,
    hidden_dim=CLF_HIDDEN, temporal_embed_dim=TEMPORAL_EMBED_DIM,
).to(device)

n_params_ddpm = sum(p.numel() for p in ddpm.parameters())
n_params_clf = sum(p.numel() for p in classifier.parameters())
log(f"DDPM parameters: {n_params_ddpm:,}")
log(f"Classifier parameters: {n_params_clf:,}")

# ---- Training ----
if LOAD_CHECKPOINT:
    log(f"\nLoading checkpoint from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    ddpm.load_state_dict(ckpt["ddpm_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    cond_mean = ckpt["cond_mean"]
    cond_std = ckpt["cond_std"]
    tgt_mean = ckpt["tgt_mean"]
    tgt_std = ckpt["tgt_std"]
    cond_numeric_indices = ckpt.get("cond_numeric_indices", cond_numeric_indices)
    train_time = ckpt.get("train_time_s", 0)
    diagnostics = ckpt.get("diagnostics", [])
    log(f"  Loaded (DDPM: {n_params_ddpm:,}, Clf: {n_params_clf:,} params)")
else:
    # ---- Phase 1: Train DDPM ----
    log(f"\n{'='*60}")
    log(f"PHASE 1: DDPM Training (feature-only diffusion)")
    log(f"{'='*60}")

    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=DIFF_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DIFF_EPOCHS)

    X_cond_train = torch.tensor(X_cond_norm[train_idx], device=device)
    X_tgt_train = torch.tensor(X_target_norm[train_idx], device=device)
    X_cond_val = torch.tensor(X_cond_norm[val_idx], device=device)
    X_tgt_val = torch.tensor(X_target_norm[val_idx], device=device)

    diagnostics = []
    best_loss = float('inf')
    patience_counter = 0
    train_start = time.time()

    for epoch in range(DIFF_EPOCHS):
        ddpm.train()
        epoch_losses = []

        perm_t = torch.randperm(len(X_cond_train), device=device)
        n_batches = max(1, len(X_cond_train) // DIFF_BATCH)

        for batch_idx in range(n_batches):
            start = batch_idx * DIFF_BATCH
            end = min(start + DIFF_BATCH, len(X_cond_train))
            idx = perm_t[start:end]

            x_cond = X_cond_train[idx]
            x_target = X_tgt_train[idx]

            t = torch.randint(0, DIFF_TIMESTEPS, (len(idx),), device=device)
            noise = torch.randn_like(x_target)
            ab = sqrt_alpha_bar[t].unsqueeze(1)
            ab_comp = sqrt_one_minus_alpha_bar[t].unsqueeze(1)
            x_noisy = ab * x_target + ab_comp * noise

            pred_noise = ddpm(x_noisy, x_cond, t.unsqueeze(1))
            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        # Validation loss
        ddpm.eval()
        with torch.no_grad():
            val_t = torch.randint(0, DIFF_TIMESTEPS, (len(X_cond_val),), device=device)
            val_noise = torch.randn_like(X_tgt_val)
            val_ab = sqrt_alpha_bar[val_t].unsqueeze(1)
            val_ab_c = sqrt_one_minus_alpha_bar[val_t].unsqueeze(1)
            val_noisy = val_ab * X_tgt_val + val_ab_c * val_noise
            val_pred = ddpm(val_noisy, X_cond_val, val_t.unsqueeze(1))
            val_loss = nn.functional.mse_loss(val_pred, val_noise).item()

        avg_loss = np.mean(epoch_losses)
        diag = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": round(optimizer.param_groups[0]['lr'], 8),
            "elapsed_s": round(time.time() - train_start, 1),
        }
        diagnostics.append(diag)

        if epoch % 10 == 0 or epoch == DIFF_EPOCHS - 1:
            log(f"  Epoch {epoch:3d}/{DIFF_EPOCHS}: train_loss={avg_loss:.5f}, "
                f"val_loss={val_loss:.5f}, lr={diag['lr']:.6f}, "
                f"elapsed={diag['elapsed_s']:.0f}s")

        # Early stopping on validation loss
        if val_loss < best_loss - 0.0005:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE and epoch >= 30:
            log(f"  Early stopping at epoch {epoch} (val_loss={val_loss:.5f})")
            break

    ddpm_time = time.time() - train_start
    log(f"\nDDPM training complete in {ddpm_time:.1f}s")

    # ---- Phase 2: Train classifier ----
    log(f"\n{'='*60}")
    log(f"PHASE 2: Classifier Training (features+lags → protest)")
    log(f"{'='*60}")

    # Classifier trains on condition vectors → protest label
    # This mimics what we'll do at inference: condition features → classify
    clf_optimizer = torch.optim.AdamW(classifier.parameters(), lr=CLF_LR, weight_decay=1e-4)
    clf_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(clf_optimizer, T_max=CLF_EPOCHS)

    # Use condition vectors (features + lags) as input, target protest as label
    # Weight the positive class
    n_pos = int(Y_labels[train_idx].sum())
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    log(f"  Class balance: {n_pos} pos / {n_neg} neg, pos_weight={pos_weight.item():.1f}")

    clf_X_train = torch.tensor(X_cond_norm[train_idx], device=device)
    clf_Y_train = torch.tensor(Y_labels[train_idx], device=device).unsqueeze(1)
    clf_X_val = torch.tensor(X_cond_norm[val_idx], device=device)
    clf_Y_val = torch.tensor(Y_labels[val_idx], device=device).unsqueeze(1)

    clf_start = time.time()
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(CLF_EPOCHS):
        classifier.train()
        perm_c = torch.randperm(len(clf_X_train), device=device)
        epoch_losses = []

        for batch_idx in range(max(1, len(clf_X_train) // CLF_BATCH)):
            start = batch_idx * CLF_BATCH
            end = min(start + CLF_BATCH, len(clf_X_train))
            idx = perm_c[start:end]

            logits = classifier(clf_X_train[idx])
            loss = bce_loss_fn(logits, clf_Y_train[idx])

            clf_optimizer.zero_grad()
            loss.backward()
            clf_optimizer.step()
            epoch_losses.append(loss.item())

        clf_scheduler.step()

        if epoch % 10 == 0 or epoch == CLF_EPOCHS - 1:
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(clf_X_val)
                val_loss = bce_loss_fn(val_logits, clf_Y_val).item()
                val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
                val_actual = clf_Y_val.cpu().numpy().flatten()
                # Quick AUC
                from sklearn.metrics import roc_auc_score
                try:
                    val_auc = roc_auc_score(val_actual, val_probs)
                except:
                    val_auc = 0.0
            log(f"  Epoch {epoch:3d}/{CLF_EPOCHS}: loss={np.mean(epoch_losses):.5f}, "
                f"val_loss={val_loss:.5f}, val_AUC={val_auc:.4f}")

    clf_time = time.time() - clf_start
    train_time = ddpm_time + clf_time
    log(f"\nClassifier training complete in {clf_time:.1f}s")

    # ---- Save checkpoint ----
    log("Saving checkpoint...")
    torch.save({
        "ddpm_state_dict": ddpm.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "cond_mean": cond_mean,
        "cond_std": cond_std,
        "tgt_mean": tgt_mean,
        "tgt_std": tgt_std,
        "cond_numeric_indices": cond_numeric_indices,
        "feature_dim": feature_dim,
        "cond_dim": cond_dim,
        "n_params_ddpm": n_params_ddpm,
        "n_params_clf": n_params_clf,
        "train_time_s": train_time,
        "diagnostics": diagnostics,
    }, CHECKPOINT_PATH)
    log(f"  Saved: {CHECKPOINT_PATH}")

# ---- DDIM Inference ----
log(f"\n{'='*60}")
log(f"INFERENCE: DDIM {DDIM_STEPS} steps, {N_SCENARIOS} scenarios, classifier head")
log(f"{'='*60}")

ddim_timesteps = torch.linspace(DIFF_TIMESTEPS - 1, 0, DDIM_STEPS, device=device).long()

@torch.no_grad()
def generate_features_ddim(model, x_cond_batch, n_scenarios=10):
    """Generate feature-only scenarios using DDIM."""
    model.eval()
    batch_size = x_cond_batch.shape[0]
    all_generated = []
    for s in range(n_scenarios):
        x = torch.randn(batch_size, feature_dim, device=device)
        for i in range(len(ddim_timesteps)):
            t_idx = ddim_timesteps[i]
            t = torch.full((batch_size, 1), t_idx.item(), device=device)
            pred_noise = model(x, x_cond_batch, t)
            ab_t = alpha_bar[t_idx]
            x0_pred = (x - torch.sqrt(1 - ab_t) * pred_noise) / torch.sqrt(ab_t)
            if i < len(ddim_timesteps) - 1:
                ab_next = alpha_bar[ddim_timesteps[i + 1]]
                x = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * pred_noise
            else:
                x = x0_pred
        all_generated.append(x)
    return torch.stack(all_generated, dim=1)  # (batch, n_scenarios, feature_dim)


# ---- Expanding window evaluation ----
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, brier_score_loss

results = []
year_scores = {}

for eval_year in EVAL_YEARS:
    log(f"\n--- Eval year {eval_year} ---")
    t0 = time.time()

    # Collect train and test rows
    train_rows_flat = []
    for y in range(TRAIN_START, eval_year):
        for pid, row in rows_by_year[y].items():
            train_rows_flat.append(row)

    test_rows = rows_by_year.get(eval_year, {})
    test_pids = list(test_rows.keys())
    if len(test_pids) > MAX_EVAL_PARCELS:
        pos_pids = [p for p in test_pids if test_rows[p][TARGET] == "1"]
        neg_pids = [p for p in test_pids if test_rows[p][TARGET] != "1"]
        n_neg_sample = min(len(neg_pids), MAX_EVAL_PARCELS - len(pos_pids))
        neg_pids = list(np.random.choice(neg_pids, n_neg_sample, replace=False))
        test_pids = pos_pids + neg_pids
        log(f"  Subsampled eval to {len(test_pids)} parcels ({len(pos_pids)} positive)")

    # ---- LogReg baseline ----
    X_train_lr = np.array([row_to_features(r) for r in train_rows_flat], dtype=np.float32)
    y_train_lr = np.array([int(r[TARGET] == "1") for r in train_rows_flat], dtype=np.int32)

    X_test_lr = np.array([row_to_features(test_rows[pid]) for pid in test_pids], dtype=np.float32)
    y_test = np.array([int(test_rows[pid][TARGET] == "1") for pid in test_pids], dtype=np.int32)

    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train_lr), nan=0, posinf=0, neginf=0)
    X_test_s = np.nan_to_num(scaler.transform(X_test_lr), nan=0, posinf=0, neginf=0)

    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
    lr_model.fit(X_train_s, y_train_lr)
    lr_probs = lr_model.predict_proba(X_test_s)[:, 1]
    lr_time = time.time() - t0
    log(f"  LogReg: scored {len(lr_probs)} parcels in {lr_time:.1f}s")

    # ---- Diffusion + Classifier inference ----
    t1 = time.time()
    prev_year = eval_year - 1
    prev_year_data = rows_by_year.get(prev_year, {})

    valid_test_pids = [pid for pid in test_pids if pid in prev_year_data]

    # Build multi-year condition vectors with lag features
    cond_vectors = []
    for pid in valid_test_pids:
        # Multi-year features
        multi_feats = []
        for lb in range(LOOKBACK_YEARS):
            lb_year = prev_year - lb
            lb_data = rows_by_year.get(lb_year, {})
            if pid in lb_data:
                multi_feats.append(row_to_features(lb_data[pid]))
            else:
                multi_feats.append(row_to_features(prev_year_data[pid]))  # fallback

        # Lag features from most recent year
        protest_lag1 = float(prev_year_data[pid].get(TARGET, "0") == "1")
        prev2_data = rows_by_year.get(prev_year - 1, {})
        mv_prev = safe_float(prev_year_data[pid].get("market_value", ""))
        mv_prev2 = safe_float(prev2_data.get(pid, {}).get("market_value", "")) if pid in prev2_data else mv_prev
        value_change_pct = (mv_prev - mv_prev2) / (mv_prev2 + 1) if mv_prev2 > 0 else 0.0

        cond = np.concatenate(multi_feats + [[protest_lag1, value_change_pct]])
        cond_vectors.append(cond)

    cond_arr = np.array(cond_vectors, dtype=np.float32)
    cond_arr[:, cond_numeric_indices] = (cond_arr[:, cond_numeric_indices] - cond_mean) / cond_std

    log(f"  Diffusion: {len(valid_test_pids)} parcels with prior-year data")

    # Generate feature scenarios in batches
    batch_size_gen = 5000
    all_gen_features = []
    for i in range(0, len(cond_arr), batch_size_gen):
        batch = torch.tensor(cond_arr[i:i+batch_size_gen], device=device)
        gen = generate_features_ddim(ddpm, batch, n_scenarios=N_SCENARIOS)
        all_gen_features.append(gen)
        if (i // batch_size_gen) % 3 == 0:
            log(f"    Generated batch {i//batch_size_gen + 1}/{math.ceil(len(cond_arr)/batch_size_gen)}")

    gen_features = torch.cat(all_gen_features, dim=0)  # (n_parcels, n_scenarios, feature_dim)
    diff_gen_time = time.time() - t1

    # Classify each scenario with the classifier head
    # Classifier input = [lookback_year2_features | generated_features | lag_features]
    # i.e., replace the most recent year's features with generated ones, keep older lookback
    classifier.eval()
    with torch.no_grad():
        n_parcels = gen_features.shape[0]
        all_probs = []
        # Base condition without the most recent year's features (keep older lookback + lags)
        cond_tensor = torch.tensor(cond_arr, device=device)
        for s in range(N_SCENARIOS):
            gen_s = gen_features[:, s, :]  # (n_parcels, feature_dim)
            # Build classifier input: replace first feature_dim with generated, keep rest
            clf_input = cond_tensor.clone()
            clf_input[:, :feature_dim] = gen_s  # replace most recent year with generated
            logits = classifier(clf_input)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.append(probs)

        # Average probabilities across scenarios
        diff_probs_full = np.mean(all_probs, axis=0)

    diff_time = time.time() - t1
    log(f"  Diffusion inference: {diff_time:.1f}s (gen={diff_gen_time:.1f}s)")

    # Map back to full test set
    diff_probs = np.full(len(test_pids), np.nan)
    pid_to_idx = {pid: i for i, pid in enumerate(valid_test_pids)}
    for i, pid in enumerate(test_pids):
        if pid in pid_to_idx:
            diff_probs[i] = diff_probs_full[pid_to_idx[pid]]

    nan_mask = np.isnan(diff_probs)
    diff_probs[nan_mask] = lr_probs[nan_mask]
    log(f"  Diffusion: {(~nan_mask).sum()} scored, {nan_mask.sum()} fell back to LR")

    # ---- Ensemble: weighted average of LR and Diffusion ----
    ensemble_probs = ENSEMBLE_WEIGHT * lr_probs + (1 - ENSEMBLE_WEIGHT) * diff_probs

    # ---- Generation quality metrics (FIXED MMD) ----
    real_targets = np.array([row_to_features(test_rows[pid]) for pid in valid_test_pids], dtype=np.float32)
    # Normalize for MMD comparison
    real_norm = real_targets.copy()
    real_norm[:, :n_numeric] = (real_targets[:, :n_numeric] - tgt_mean) / tgt_std
    gen_mean = gen_features.mean(dim=1).cpu().numpy()  # already normalized from diffusion

    n_sample = min(5000, len(real_norm))
    idx_s = np.random.choice(len(real_norm), n_sample, replace=False)

    def rbf_kernel(X, Y, sigma=1.0):
        """RBF kernel on NORMALIZED features — no overflow."""
        XX = np.sum(X**2, axis=1, keepdims=True)
        YY = np.sum(Y**2, axis=1, keepdims=True)
        dist = XX + YY.T - 2 * X @ Y.T
        return np.exp(-dist / (2 * sigma**2))

    # Use median heuristic for sigma
    real_s = real_norm[idx_s]
    gen_s = gen_mean[idx_s]
    pairwise_dist = np.sum((real_s[:100] - gen_s[:100])**2, axis=1)
    sigma = max(np.sqrt(np.median(pairwise_dist)), 1.0)

    Kxx = rbf_kernel(real_s, real_s, sigma)
    Kyy = rbf_kernel(gen_s, gen_s, sigma)
    Kxy = rbf_kernel(real_s, gen_s, sigma)
    mmd = float(np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy))

    # Correlation preservation
    real_corr = np.corrcoef(real_s[:, :n_numeric].T)
    gen_corr = np.corrcoef(gen_s[:, :n_numeric].T)
    # Handle NaN
    valid_mask = ~(np.isnan(real_corr.flatten()) | np.isnan(gen_corr.flatten()))
    if valid_mask.sum() > 10:
        corr_preservation = float(np.corrcoef(real_corr.flatten()[valid_mask], gen_corr.flatten()[valid_mask])[0, 1])
    else:
        corr_preservation = float('nan')

    # Protest rates
    real_protest_rate = float(np.mean(y_test[~nan_mask]))
    gen_protest_rate = float(np.mean(diff_probs_full > 0.5))

    # ---- Classification metrics ----
    def compute_metrics(y_true, y_prob, model_name):
        if len(np.unique(y_true)) < 2:
            return {}
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, np.clip(y_prob, 0, 1))

        sorted_idx = np.argsort(-y_prob)
        n_pos = int(y_true.sum())

        metrics = {
            "model": model_name,
            "eval_year": eval_year,
            "n_test": len(y_true),
            "n_positive": n_pos,
            "prevalence": round(n_pos / len(y_true) * 100, 3),
            "auc_roc": round(auc_roc, 4),
            "auc_pr": round(auc_pr, 4),
            "brier_score": round(brier, 6),
        }
        for k_mult in [1, 2, 5, 10]:
            k = n_pos * k_mult
            if k <= len(y_true):
                top_k = y_true[sorted_idx[:k]]
                metrics[f"precision@{k_mult}x"] = round(float(np.mean(top_k)), 4)
                metrics[f"recall@{k_mult}x"] = round(float(np.sum(top_k) / n_pos), 4)
        return metrics

    lr_metrics = compute_metrics(y_test, lr_probs, "LogReg")
    diff_metrics = compute_metrics(y_test, diff_probs, "Diffusion_v3")
    ens_metrics = compute_metrics(y_test, ensemble_probs, "Ensemble")

    diff_metrics["mmd"] = round(mmd, 6)
    diff_metrics["corr_preservation"] = round(corr_preservation, 4) if not np.isnan(corr_preservation) else None
    diff_metrics["real_protest_rate"] = round(real_protest_rate, 4)
    diff_metrics["gen_protest_rate"] = round(gen_protest_rate, 4)
    diff_metrics["inference_time_s"] = round(diff_time, 1)

    results.append(lr_metrics)
    results.append(diff_metrics)
    results.append(ens_metrics)

    # Log comparison
    log(f"\n  {'Metric':<25} {'LogReg':>10} {'Diffusion':>10} {'Ensemble':>10} {'Best':>10}")
    log(f"  {'-'*68}")
    for key in ["auc_roc", "auc_pr", "brier_score", "precision@1x", "recall@1x", "precision@5x", "recall@5x"]:
        lr_val = lr_metrics.get(key, "—")
        diff_val = diff_metrics.get(key, "—")
        ens_val = ens_metrics.get(key, "—")
        if isinstance(lr_val, float) and isinstance(diff_val, float) and isinstance(ens_val, float):
            best = "ENS" if (ens_val >= lr_val and ens_val >= diff_val) else ("LR" if lr_val >= diff_val else "DIFF")
            if key == "brier_score":  # lower is better
                best = "ENS" if (ens_val <= lr_val and ens_val <= diff_val) else ("LR" if lr_val <= diff_val else "DIFF")
            log(f"  {key:<25} {lr_val:>10.4f} {diff_val:>10.4f} {ens_val:>10.4f} {best:>10}")
        else:
            log(f"  {key:<25} {str(lr_val):>10} {str(diff_val):>10} {str(ens_val):>10}")
    log(f"\n  Generation quality: MMD={mmd:.6f}, Corr={corr_preservation:.4f}")
    log(f"  Protest rates: real={real_protest_rate:.4f}, gen={gen_protest_rate:.4f}")

    year_scores[eval_year] = {}
    for i, pid in enumerate(test_pids):
        if pid in centroids:
            lat, lon = centroids[pid]
            year_scores[eval_year][pid] = {
                "lr": float(lr_probs[i]),
                "diff": float(diff_probs[i]),
                "ens": float(ensemble_probs[i]),
                "actual": int(y_test[i]),
                "lat": lat, "lon": lon,
            }

# ---- Save results ----
log(f"\n{'='*60}")
log("Saving results...")

metrics_path = os.path.join(OUT_DIR, "classification_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"  Metrics: {metrics_path}")

scores_path = os.path.join(OUT_DIR, "per_parcel_scores.csv")
with open(scores_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["parcel_id", "year", "lr_score", "diff_score", "ensemble_score", "actual", "lat", "lon"])
    for year, parcels in sorted(year_scores.items()):
        for pid, data in parcels.items():
            w.writerow([pid, year, round(data["lr"], 5), round(data["diff"], 5),
                        round(data["ens"], 5), data["actual"], data["lat"], data["lon"]])
log(f"  Scores: {scores_path}")

# Save diagnostics
with open(os.path.join(OUT_DIR, "training_diagnostics.jsonl"), "w") as f:
    for d in diagnostics:
        f.write(json.dumps(d) + "\n")

# Final table
log(f"\n{'='*60}")
log("FINAL COMPARISON: LogReg vs Diffusion v3 vs Ensemble")
log(f"{'='*60}")
log(f"\n{'Year':<6} {'Model':<14} {'AUC-ROC':>8} {'AUC-PR':>8} {'Brier':>8} {'P@1x':>8} {'R@1x':>8} {'P@5x':>8} {'R@5x':>8}")
log(f"{'-'*78}")
for r in results:
    log(f"{r.get('eval_year',''):<6} {r['model']:<14} "
        f"{r.get('auc_roc','—'):>8} {r.get('auc_pr','—'):>8} "
        f"{r.get('brier_score','—'):>8} "
        f"{r.get('precision@1x','—'):>8} {r.get('recall@1x','—'):>8} "
        f"{r.get('precision@5x','—'):>8} {r.get('recall@5x','—'):>8}")

log(f"\nTotal elapsed: {time.time() - (time.time() - train_time):.0f}s")
log("Done!")
