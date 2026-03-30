"""
icp_nonlinear.py — Nonlinear ICP: Residual Invariance Testing
==============================================================
Extends Peters et al. (2016) Method II to nonlinear models.

Instead of testing residuals from linear/logistic regression, we:
1. Train an MLP (V-REx) and CVAE (V-REx) on the full feature set
   (numeric + categorical).
2. Compute prediction residuals R = Y - f_hat(X) from each model.
3. Test whether those residuals satisfy invariance: equal mean and
   equal variance across all 182 zoning environments.

If V-REx has learned a truly invariant representation, its residuals
SHOULD be invariant (not rejected by the ICP test). If the ERM baseline
has memorized environment-specific noise, its residuals WILL NOT be
invariant (rejected).

This closes the loop:
  - Linear ICP → REJECTED (causal mechanism is nonlinear)
  - Logistic ICP → REJECTED (single-layer logistic can't capture it)
  - MLP V-REx ICP → ???  (does the neural invariant representation work?)
  - CVAE V-REx ICP → ???  (does the generative model produce invariant residuals?)

Author: Daniel Hardesty Lewis
Created: 2026-03-09
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import warnings
import sys
sys.stdout = open('icp_results_native.txt', 'w', encoding='utf-8')
import os
import time

warnings.filterwarnings('ignore')

# Config
ALPHA = 0.05
MIN_ENV_SIZE = 10
EPOCHS = 40
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
VREX_PENALTY_WEIGHT = 10.0
LATENT_DIM = 8

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
TENSOR_DIR = os.path.join(PROJECT_DIR, "Analysis", "Data", "Tensors")

NUMERIC_FEATURES = []  # loaded per-fold from metadata


def load_fold(fold_dir):
    """Load pre-built tensors + metadata for one CV fold."""
    import json
    
    X = torch.load(os.path.join(fold_dir, "X_train.pt")).numpy()
    y = torch.load(os.path.join(fold_dir, "y_train.pt")).numpy()
    envs = torch.load(os.path.join(fold_dir, "envs_train.pt")).numpy()
    env_ids = np.load(os.path.join(fold_dir, "env_ids_train.npy"), allow_pickle=True)
    
    with open(os.path.join(fold_dir, "metadata.json")) as f:
        meta = json.load(f)
    
    full_feature_names = meta['full_feature_names']
    unique_envs = meta['unique_envs']
    
    global NUMERIC_FEATURES
    NUMERIC_FEATURES = meta['numeric_features']
    
    print(f"  {len(y):,} rows | {len(full_feature_names)} features | {len(unique_envs)} environments (Council Districts)")
    print(f"  Protest rate: {y.mean():.4f} ({y.sum():.0f} positives)")
    
    return X, y, envs, env_ids, unique_envs, full_feature_names


# ──────────── ICP Test ────────────

def test_residual_invariance(residuals, env_ids, unique_envs, label="Model"):
    """Test whether residuals R have equal mean and equal variance across environments."""
    p_means = []
    p_vars = []

    for e in unique_envs:
        mask = env_ids == e
        if mask.sum() < 5:
            continue
        R_e = residuals[mask]
        R_not_e = residuals[~mask]

        _, p_mean = stats.ttest_ind(R_e, R_not_e, equal_var=False)
        p_means.append(p_mean)

        _, p_var = stats.levene(R_e, R_not_e)
        p_vars.append(p_var)

    if not p_means:
        return None, None, None, False

    raw_min_p_mean = min(p_means)
    raw_min_p_var = min(p_vars)

    bonf_p_mean = raw_min_p_mean * len(p_means)
    bonf_p_var = raw_min_p_var * len(p_vars)
    combined_p = min(2 * min(bonf_p_mean, bonf_p_var), 1.0)
    accepted = combined_p >= ALPHA

    # Count how many individual environment tests are rejected
    n_rejected_mean = sum(1 for p in p_means if p < ALPHA / len(p_means))
    n_rejected_var = sum(1 for p in p_vars if p < ALPHA / len(p_vars))

    return combined_p, n_rejected_mean, n_rejected_var, accepted


# ──────────── Linear/Logistic ICP on Full Features ────────────

def run_linear_icp(X, y, env_ids, unique_envs, feature_names):
    """Run Peters et al. Method II on ALL feature subsets (numeric only, since combinatorial)."""
    print("\n" + "=" * 80)
    print("PART 1: LINEAR ICP (Logistic Regression) - Numeric Features Only")
    print("=" * 80)

    # Only test numeric subsets — exhaustive for sizes 0-2 + full set
    # (ICP's intersection guarantee requires exhaustive enumeration; random sampling is incoherent)
    numeric_indices = list(range(len(NUMERIC_FEATURES)))
    n_num = len(NUMERIC_FEATURES)

    all_subsets = []
    # Sizes 0, 1, 2: exhaustive (1 + 23 + 253 = 277 subsets)
    for size in range(0, min(3, n_num + 1)):
        for combo in combinations(numeric_indices, size):
            all_subsets.append(combo)
    # Full numeric set
    all_subsets.append(tuple(numeric_indices))

    print(f"Testing {len(all_subsets)} numeric subsets for invariance (α = {ALPHA})...")
    for subset in all_subsets:
        names = [NUMERIC_FEATURES[i] for i in subset] if subset else ["∅"]
        label = "{" + ", ".join(names) + "}"

        X_S = X[:, list(subset)] if subset else np.zeros((len(y), 0))

        if X_S.shape[1] == 0:
            f_hat = np.full(len(y), y.mean())
        else:
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf.fit(X_S, y)
            f_hat = clf.predict_proba(X_S)[:, 1]

        residuals = y - f_hat
        p, n_rej_m, n_rej_v, accepted = test_residual_invariance(residuals, env_ids, unique_envs)
        status = "ACCEPTED" if accepted else "REJECTED"
        if p is None:
            print(f"  S = {label:<55s} p_bonf = N/A  envs_rej(mean/var): N/A  {status}")
        else:
            print(f"  S = {label:<55s} p_bonf = {p:.4e}  envs_rej(mean/var): {n_rej_m}/{n_rej_v}  {status}")

    # Also test with ALL features (num + cat)
    print(f"\n  Testing FULL feature set ({X.shape[1]} features)...")
    clf_full = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_full.fit(X, y)
    f_hat_full = clf_full.predict_proba(X)[:, 1]
    res_full = y - f_hat_full
    p, n_rej_m, n_rej_v, accepted = test_residual_invariance(res_full, env_ids, unique_envs)
    status = "ACCEPTED" if accepted else "REJECTED"
    if p is None:
        print(f"  S = {{ALL {X.shape[1]} features}}  p_bonf = N/A  envs_rej(mean/var): N/A  {status}")
    else:
        print(f"  S = {{ALL {X.shape[1]} features}}  p_bonf = {p:.4e}  envs_rej(mean/var): {n_rej_m}/{n_rej_v}  {status}")


# ──────────── MLP V-REx ────────────

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(X, y, envs, method="V-REx"):
    input_dim = X.shape[1]
    model = MLP(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    e_t = torch.LongTensor(envs)
    ds = TensorDataset(X_t, y_t, e_t)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    best_loss = float('inf')
    patience_counter = 0
    PATIENCE = 10
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, eb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')

            unique_e = torch.unique(eb)
            env_risks = [losses[eb == e].mean() for e in unique_e if (eb == e).sum() >= 2]
            if len(env_risks) < 2: continue

            env_risks_stack = torch.stack(env_risks)
            erm_loss = env_risks_stack.mean()

            if method == "V-REx":
                penalty = env_risks_stack.var()
                beta = VREX_PENALTY_WEIGHT if epoch > 20 else VREX_PENALTY_WEIGHT * (epoch / 20.0)
                loss = erm_loss + beta * penalty
            else:
                loss = erm_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    MLP {method} early stop at epoch {epoch+1}/{EPOCHS}")
                break

    return model


# ──────────── CVAE V-REx ────────────

class CVAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim + 1, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.fc_mu = nn.Linear(32, LATENT_DIM)
        self.fc_logvar = nn.Linear(32, LATENT_DIM)
        self.decoder = nn.Sequential(nn.Linear(LATENT_DIM + 1, 32), nn.SiLU(), nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, input_dim))
        # Downstream classifier head (for ICP residual testing)
        self.classifier = nn.Sequential(nn.Linear(LATENT_DIM, 32), nn.SiLU(), nn.Linear(32, 1))

    def encode(self, x, y):
        h = self.encoder(torch.cat([x, y.view(-1, 1)], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z, y):
        return self.decoder(torch.cat([z, y.view(-1, 1)], dim=1))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar, self.classifier(mu).squeeze(-1)


def train_cvae(X, y, envs, method="V-REx"):
    input_dim = X.shape[1]
    model = CVAE(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    e_t = torch.LongTensor(envs)
    ds = TensorDataset(X_t, y_t, e_t)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    best_loss = float('inf')
    patience_counter = 0
    PATIENCE = 10
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, eb in loader:
            optimizer.zero_grad()
            recon, mu, logvar, cls_logits = model(xb, yb)

            # ELBO
            mse = torch.sum((recon - xb)**2, dim=1)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            # Classification loss on the latent mean
            cls_loss = nn.functional.binary_cross_entropy_with_logits(cls_logits, yb, reduction='none')
            per_sample_loss = mse + kld + cls_loss

            unique_e = torch.unique(eb)
            env_risks = [per_sample_loss[eb == e].mean() for e in unique_e if (eb == e).sum() >= 2]
            if len(env_risks) < 2: continue

            env_risks_stack = torch.stack(env_risks)
            erm_loss = env_risks_stack.mean()

            if method == "V-REx":
                penalty = env_risks_stack.var()
                beta = VREX_PENALTY_WEIGHT if epoch > 20 else VREX_PENALTY_WEIGHT * (epoch / 20.0)
                loss = erm_loss + beta * penalty
            else:
                loss = erm_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    CVAE {method} early stop at epoch {epoch+1}/{EPOCHS}")
                break

    return model


# ──────────── Main ────────────

def run_fold(fold_dir, fold_name, fold_meta):
    """Run full ICP pipeline on one CV fold."""
    print(f"\nLoading {fold_name}: Train <= {fold_meta['train_end_year']}, Test = {fold_meta['test_year']}")
    X, y, envs, env_ids, unique_envs, feature_names = load_fold(fold_dir)
    
    results = {'fold': fold_name, 'train_end': fold_meta['train_end_year'], 'test_year': fold_meta['test_year']}
    
    # ── Part 1: Linear ICP ──
    run_linear_icp(X, y, env_ids, unique_envs, feature_names)
    
    # ── Part 2: MLP ICP ──
    print("\n" + "=" * 80)
    print(f"PART 2: MLP ICP ({fold_name})")
    print("=" * 80)
    for method in ["ERM", "V-REx"]:
        print(f"\n  Training MLP ({method})...")
        torch.manual_seed(42); np.random.seed(42)
        model = train_mlp(X, y, envs, method=method)
        model.eval()
        with torch.no_grad():
            logits = model(torch.FloatTensor(X)).numpy()
            f_hat = 1 / (1 + np.exp(-logits))
        residuals = y - f_hat
        p, n_rej_m, n_rej_v, accepted = test_residual_invariance(residuals, env_ids, unique_envs)
        status = "ACCEPTED" if accepted else "REJECTED"
        results[f'MLP_{method}'] = {'p': p, 'accepted': accepted}
        p_str = f"{p:.4e}" if p is not None else "N/A"
        print(f"  MLP {method:<6s} p_bonf = {p_str}  envs_rej: {n_rej_m}/{n_rej_v}  {status}")
    
    # ── Part 3: CVAE ICP ──
    print("\n" + "=" * 80)
    print(f"PART 3: CVAE ICP ({fold_name})")
    print("=" * 80)
    for method in ["ERM", "V-REx"]:
        print(f"\n  Training CVAE ({method})...")
        torch.manual_seed(42); np.random.seed(42)
        model = train_cvae(X, y, envs, method=method)
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X)
            y_t = torch.FloatTensor(y)
            mu, _ = model.encode(X_t, y_t)
            cls_logits = model.classifier(mu).squeeze(-1).numpy()
            f_hat = 1 / (1 + np.exp(-cls_logits))
        residuals = y - f_hat
        p, n_rej_m, n_rej_v, accepted = test_residual_invariance(residuals, env_ids, unique_envs)
        status = "ACCEPTED" if accepted else "REJECTED"
        results[f'CVAE_{method}'] = {'p': p, 'accepted': accepted}
        p_str = f"{p:.4e}" if p is not None else "N/A"
        print(f"  CVAE {method:<6s} p_bonf = {p_str}  envs_rej: {n_rej_m}/{n_rej_v}  {status}")
    
    return results


def main():
    import json
    t0 = time.time()
    
    # Load CV metadata
    cv_meta_path = os.path.join(TENSOR_DIR, 'cv_metadata.json')
    with open(cv_meta_path) as f:
        cv_folds = json.load(f)
    
    print(f"ICP with Expanding-Window CV: {len(cv_folds)} folds")
    print(f"Environments: Austin Council Districts (theory-grounded political subdivisions)")
    print("=" * 80)
    
    all_results = []
    for fold_meta in cv_folds:
        fold_dir = os.path.join(TENSOR_DIR, fold_meta['fold_name'])
        results = run_fold(fold_dir, fold_meta['fold_name'], fold_meta)
        all_results.append(results)
    
    # ── Aggregate results across folds ──
    print("\n" + "#" * 80)
    print("AGGREGATE RESULTS ACROSS ALL FOLDS")
    print("#" * 80)
    for test in ['MLP_ERM', 'MLP_V-REx', 'CVAE_ERM', 'CVAE_V-REx']:
        verdicts = [r[test]['accepted'] for r in all_results if test in r]
        n_accepted = sum(verdicts)
        n_total = len(verdicts)
        print(f"  {test:<12s}: {n_accepted}/{n_total} folds ACCEPTED invariance")
    
    print(f"\nCompleted in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
