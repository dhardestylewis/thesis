"""
shap_vrex_attribution.py — SHAP Feature Attribution on V-REx CVAE
=================================================================
Trains a CVAE with V-REx penalty on 182 zoning environments, then uses
SHAP GradientExplainer to decompose the invariant protest prediction
into per-feature Shapley values.

This names the specific causal drivers that the invariant representation
relies on — completing the pipeline:
  ICP → "no linear subset is invariant"
  V-REx OOD → "the nonlinear representation generalizes"
  SHAP → "these are the features driving the invariant representation"

Author: Daniel Hardesty Lewis
Created: 2026-03-09
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import shap
import warnings
import os
import time

warnings.filterwarnings('ignore')

# Config
EPOCHS = 80
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
VREX_PENALTY_WEIGHT = 10.0
LATENT_DIM = 8
MIN_ENV_SIZE = 5
SAMPLED_SIZE = 33000

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
ENV_PATH = os.path.join(PROJECT_DIR, "Analysis", "Results", "irm_environment_assignments.csv")

NUMERIC_FEATURES = ['total_market_value', 'deed_acreage', 'improvement_sq_ft']
CATEGORICAL_FEATURES = ['property_category_code', 'council_district', 'lui_general_land_use']


# ──────────── CVAE Model ────────────

class CVAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU()
        )
        self.fc_mu = nn.Linear(32, LATENT_DIM)
        self.fc_logvar = nn.Linear(32, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM + 1, 32), nn.SiLU(),
            nn.Linear(32, 64), nn.SiLU(),
            nn.Linear(64, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, 32), nn.SiLU(),
            nn.Linear(32, 1)
        )

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


class EncoderClassifierWrapper(nn.Module):
    """Wraps the CVAE encoder → μ → classifier into a single X → logit function
    for SHAP attribution. Uses the training-set base rate as the conditioning Y."""
    def __init__(self, cvae, base_rate):
        super().__init__()
        self.cvae = cvae
        self.base_rate = base_rate

    def forward(self, x):
        y_dummy = torch.full((x.shape[0],), self.base_rate, device=x.device)
        mu, _ = self.cvae.encode(x, y_dummy)
        return self.cvae.classifier(mu)


# ──────────── Data Loading ────────────

def load_data():
    print("Loading panel data...")
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['year', 'protest', 'standardized_tcad_id']
    panel = pd.read_csv(PANEL_PATH, usecols=cols, low_memory=False)
    panel['improvement_sq_ft'] = pd.to_numeric(panel['improvement_sq_ft'], errors='coerce')
    panel = panel[panel['year'] <= 2024]

    env = pd.read_csv(ENV_PATH).rename(columns={'CASE_NUMBER': 'env_id'})
    df = panel.merge(env, on='standardized_tcad_id', how='left')
    df['env_id'] = df['env_id'].fillna('BACKGROUND')

    env_sizes = df.groupby('env_id').size()
    valid_envs = env_sizes[env_sizes >= MIN_ENV_SIZE].index
    df = df[df['env_id'].isin(valid_envs)]

    env_map = {name: i for i, name in enumerate(df['env_id'].unique())}
    df['env_label'] = df['env_id'].map(env_map)

    positives = df[df['protest'] == 1]
    negatives = df[df['protest'] == 0].sample(n=SAMPLED_SIZE - len(positives), random_state=42)
    df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)

    for col in NUMERIC_FEATURES:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna('Missing').astype(str)

    scaler = StandardScaler()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    X_num = scaler.fit_transform(df[NUMERIC_FEATURES])
    X_cat = ohe.fit_transform(df[CATEGORICAL_FEATURES])
    X = np.hstack([X_num, X_cat]).astype(np.float32)
    y = df['protest'].values.astype(np.float32)
    envs = df['env_label'].values.astype(np.int64)

    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names

    print(f"Dataset: {len(df):,} rows | {len(feature_names)} features | Base rate: {y.mean():.3f}")
    return X, y, envs, feature_names


# ──────────── Training ────────────

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
    for epoch in range(EPOCHS):
        total_loss = 0
        for xb, yb, eb in loader:
            optimizer.zero_grad()
            recon, mu, logvar, cls_logits = model(xb, yb)

            mse = torch.sum((recon - xb)**2, dim=1)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            cls_loss = nn.functional.binary_cross_entropy_with_logits(cls_logits, yb, reduction='none')
            per_sample_loss = mse + kld + cls_loss

            unique_e = torch.unique(eb)
            env_risks = [per_sample_loss[eb == e].mean() for e in unique_e if (eb == e).sum() >= 2]
            if len(env_risks) < 2:
                continue

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
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(loader):.4f}")

    return model


# ──────────── SHAP Attribution ────────────

def run_shap_attribution(model, X, y, feature_names):
    """Run SHAP GradientExplainer on the CVAE encoder→classifier."""
    base_rate = y.mean()
    wrapper = EncoderClassifierWrapper(model, base_rate)
    wrapper.eval()

    X_t = torch.FloatTensor(X)

    # Background: 200 random samples
    np.random.seed(42)
    bg_idx = np.random.choice(len(X), 200, replace=False)
    background = X_t[bg_idx]

    # Explain: 1000 samples (stratified: 500 protest, 500 not)
    protest_idx = np.where(y == 1)[0]
    no_protest_idx = np.where(y == 0)[0]
    explain_idx = np.concatenate([
        np.random.choice(protest_idx, min(500, len(protest_idx)), replace=False),
        np.random.choice(no_protest_idx, 500, replace=False)
    ])
    explain_data = X_t[explain_idx]

    print(f"\nRunning SHAP GradientExplainer...")
    print(f"  Background: {len(background)} samples")
    print(f"  Explaining: {len(explain_data)} samples")

    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(explain_data)

    # shap_values is (n_samples, n_features) for single-output model
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Squeeze if 3D
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    return shap_values, explain_data.numpy(), explain_idx


def print_results(shap_values, feature_names):
    """Print the mean absolute SHAP values as a ranked table."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_abs)[::-1]

    print("\n" + "=" * 80)
    print("SHAP FEATURE ATTRIBUTION — V-REx CVAE Invariant Protest Predictor")
    print("=" * 80)
    print(f"{'Rank':<6} {'Feature':<45} {'Mean |SHAP|':>12} {'% Total':>10}")
    print("-" * 75)

    total = mean_abs.sum()
    for rank, idx in enumerate(sorted_idx):
        pct = 100 * mean_abs[idx] / total if total > 0 else 0
        if pct < 0.5:
            # Stop printing features contributing < 0.5%
            remaining = len(sorted_idx) - rank
            print(f"  ... {remaining} more features each contributing < 0.5%")
            break
        print(f"{rank+1:<6} {feature_names[idx]:<45} {mean_abs[idx]:>12.6f} {pct:>9.1f}%")

    # Top-level summary
    print("\n" + "-" * 75)
    print("TOP CAUSAL DRIVERS (features contributing ≥ 5% of invariant prediction):")
    for idx in sorted_idx:
        pct = 100 * mean_abs[idx] / total if total > 0 else 0
        if pct >= 5:
            direction = "↑ INCREASES" if np.mean(shap_values[:, idx]) > 0 else "↓ DECREASES"
            print(f"  → {feature_names[idx]:<40s} ({pct:.1f}%)  {direction} protest likelihood")


# ──────────── Main ────────────

def main():
    t0 = time.time()
    X, y, envs, feature_names = load_data()

    # Train both ERM and V-REx for comparison
    for method in ["ERM", "V-REx"]:
        print(f"\n{'='*60}")
        print(f"Training CVAE ({method})...")
        print(f"{'='*60}")
        torch.manual_seed(42)
        np.random.seed(42)
        model = train_cvae(X, y, envs, method=method)

        shap_values, explain_X, explain_idx = run_shap_attribution(model, X, y, feature_names)

        print(f"\n--- {method} Model Attribution ---")
        print_results(shap_values, feature_names)

    print(f"\nCompleted in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
