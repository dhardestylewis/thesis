"""
shap_vrex_enriched.py — SHAP on V-REx CVAE with Full Demographic Enrichment
=============================================================================
Retrains the V-REx CVAE using the enriched panel which includes:
  - Property attributes (market value, acreage, sqft)
  - Land use / zoning (LUI, property category, LDB base zone, FAR)
  - ACS Census demographics (race, income, homeownership, rent)
  - Council district

Then runs SHAP GradientExplainer to identify which features the invariant
representation relies on — answering the thesis question:
"Is NIMBYism driven by property characteristics, zoning, or demographics?"

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
EPOCHS = 120
BATCH_SIZE = 512  # Smaller batches for better env sampling per batch
LEARNING_RATE = 1e-3
VREX_PENALTY_WEIGHT = 100.0  # Strong penalty to force invariance
LATENT_DIM = 12  # Bumped from 8 → 12 for richer feature set
MIN_ENV_SIZE = 5
SAMPLED_SIZE = 33000

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
ENV_PATH = os.path.join(PROJECT_DIR, "Analysis", "Results", "irm_environment_assignments.csv")

# Feature groups — organized by causal domain for interpretability
NUMERIC_FEATURES = [
    # Property physical
    'total_market_value', 'deed_acreage', 'land_market_value',
    # Zoning density
    'ldb_far', 'ldb_units',
    # ACS Demographics
    'acs_total_population', 'acs_median_age',
    'acs_race_white', 'acs_race_black', 'acs_race_asian', 'acs_race_hispanic',
    'acs_median_household_income', 'acs_poverty_count',
    'acs_median_home_value',
    'acs_owner_occupied_units', 'acs_renter_occupied_units',
    'acs_median_gross_rent',
]

CATEGORICAL_FEATURES = [
    'property_category_code',
    'council_district',
    'lui_general_land_use',
    'ldb_basezone',
]

# Human-readable group labels for the summary
FEATURE_GROUPS = {
    'total_market_value': 'Property',
    'deed_acreage': 'Property',
    'land_market_value': 'Property',
    'ldb_far': 'Zoning/Density',
    'ldb_units': 'Zoning/Density',
    'acs_total_population': 'Demographics',
    'acs_median_age': 'Demographics',
    'acs_race_white': 'Demographics (Race)',
    'acs_race_black': 'Demographics (Race)',
    'acs_race_asian': 'Demographics (Race)',
    'acs_race_hispanic': 'Demographics (Race)',
    'acs_median_household_income': 'Demographics (Income)',
    'acs_poverty_count': 'Demographics (Income)',
    'acs_median_home_value': 'Demographics',
    'acs_owner_occupied_units': 'Demographics (Tenure)',
    'acs_renter_occupied_units': 'Demographics (Tenure)',
    'acs_median_gross_rent': 'Demographics (Tenure)',
    'property_category_code': 'Land Use',
    'council_district': 'Geography',
    'lui_general_land_use': 'Land Use',
    'ldb_basezone': 'Zoning/Density',
}


class CVAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU()
        )
        self.fc_mu = nn.Linear(64, LATENT_DIM)
        self.fc_logvar = nn.Linear(64, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM + 1, 64), nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.SiLU(),
            nn.Linear(64, 1)
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
    def __init__(self, cvae, base_rate):
        super().__init__()
        self.cvae = cvae
        self.base_rate = base_rate

    def forward(self, x):
        y_dummy = torch.full((x.shape[0],), self.base_rate, device=x.device)
        mu, _ = self.cvae.encode(x, y_dummy)
        return self.cvae.classifier(mu)


def load_data():
    print("Loading enriched panel data...")
    cols_needed = (['standardized_tcad_id', 'year', 'protest'] +
                   NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    # Deduplicate
    cols_needed = list(dict.fromkeys(cols_needed))

    panel = pd.read_csv(PANEL_PATH, usecols=cols_needed, low_memory=False)
    panel = panel[panel['year'] <= 2024]

    env = pd.read_csv(ENV_PATH).rename(columns={'CASE_NUMBER': 'env_id'})
    panel['standardized_tcad_id'] = panel['standardized_tcad_id'].astype(str)
    env['standardized_tcad_id'] = env['standardized_tcad_id'].astype(str)
    df = panel.merge(env, on='standardized_tcad_id', how='left')
    df['env_id'] = df['env_id'].fillna('BACKGROUND')

    env_sizes = df.groupby('env_id').size()
    valid_envs = env_sizes[env_sizes >= MIN_ENV_SIZE].index
    df = df[df['env_id'].isin(valid_envs)]

    env_map = {name: i for i, name in enumerate(df['env_id'].unique())}
    df['env_label'] = df['env_id'].map(env_map)

    # Sample
    positives = df[df['protest'] == 1]
    n_neg = SAMPLED_SIZE - len(positives)
    negatives = df[df['protest'] == 0]
    if len(negatives) > n_neg:
        negatives = negatives.sample(n=n_neg, random_state=42)
    df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Clean numeric
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Clean categorical
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna('Missing').astype(str)

    scaler = StandardScaler()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=15)

    X_num = scaler.fit_transform(df[NUMERIC_FEATURES])
    X_cat = ohe.fit_transform(df[CATEGORICAL_FEATURES])
    X = np.hstack([X_num, X_cat]).astype(np.float32)
    y = df['protest'].values.astype(np.float32)
    envs = df['env_label'].values.astype(np.int64)

    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names

    # Map feature names to groups
    feat_to_group = {}
    for fn in NUMERIC_FEATURES:
        feat_to_group[fn] = FEATURE_GROUPS.get(fn, 'Other')
    for cn in cat_names:
        # e.g. "council_district_3.0" → "council_district" → group
        base = cn.split('_')[0]
        for cat_col in CATEGORICAL_FEATURES:
            if cn.startswith(cat_col):
                feat_to_group[cn] = FEATURE_GROUPS.get(cat_col, 'Other')
                break

    print(f"Dataset: {len(df):,} rows | {X.shape[1]} features "
          f"({len(NUMERIC_FEATURES)} num + {len(cat_names)} cat) | "
          f"Base rate: {y.mean():.3f}")

    return X, y, envs, feature_names, feat_to_group


def train_cvae(X, y, envs, method="V-REx"):
    input_dim = X.shape[1]
    model = CVAE(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y), torch.LongTensor(envs))
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
            per_sample = mse + kld + cls_loss

            unique_e = torch.unique(eb)
            env_risks = [per_sample[eb == e].mean() for e in unique_e if (eb == e).sum() >= 2]
            if len(env_risks) < 2:
                continue

            stack = torch.stack(env_risks)
            erm = stack.mean()

            if method == "V-REx":
                pen = stack.var()
                beta = VREX_PENALTY_WEIGHT if epoch > 20 else VREX_PENALTY_WEIGHT * (epoch / 20.0)
                loss = erm + beta * pen
            else:
                loss = erm

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg = total_loss / max(len(loader), 1)
            print(f"  Epoch {epoch+1:3d} | Loss: {avg:.8f}")

    return model


def run_shap(model, X, y, feature_names, feat_to_group, method_name):
    wrapper = EncoderClassifierWrapper(model, y.mean())
    wrapper.eval()

    X_t = torch.FloatTensor(X)
    np.random.seed(42)
    bg_idx = np.random.choice(len(X), 200, replace=False)

    protest_idx = np.where(y == 1)[0]
    no_protest_idx = np.where(y == 0)[0]
    explain_idx = np.concatenate([
        np.random.choice(protest_idx, min(500, len(protest_idx)), replace=False),
        np.random.choice(no_protest_idx, 500, replace=False)
    ])

    print(f"\n  Running SHAP GradientExplainer ({method_name})...")
    explainer = shap.GradientExplainer(wrapper, X_t[bg_idx])
    shap_values = explainer.shap_values(X_t[explain_idx])

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    total = mean_abs.sum()

    print(f"\n{'='*80}")
    print(f"SHAP ATTRIBUTION — {method_name} CVAE (Enriched with Demographics)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Group':<22} {'Feature':<40} {'|SHAP|':>10} {'%':>7}")
    print("-" * 85)

    for rank, idx in enumerate(sorted_idx):
        pct = 100 * mean_abs[idx] / total if total > 0 else 0
        if pct < 0.3:
            print(f"  ... {len(sorted_idx) - rank} more features < 0.3%")
            break
        group = feat_to_group.get(feature_names[idx], 'Other')
        print(f"{rank+1:<5} {group:<22} {feature_names[idx]:<40} {mean_abs[idx]:>10.5f} {pct:>6.1f}%")

    # Group-level summary
    print(f"\n{'='*80}")
    print(f"GROUP-LEVEL ATTRIBUTION SUMMARY — {method_name}")
    print(f"{'='*80}")
    group_importance = {}
    for idx in range(len(feature_names)):
        g = feat_to_group.get(feature_names[idx], 'Other')
        group_importance[g] = group_importance.get(g, 0) + mean_abs[idx]

    sorted_groups = sorted(group_importance.items(), key=lambda x: -x[1])
    print(f"{'Group':<30} {'Total |SHAP|':>14} {'%':>7}")
    print("-" * 55)
    for g, val in sorted_groups:
        pct = 100 * val / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{g:<30} {val:>14.5f} {pct:>6.1f}%  {bar}")

    # Top drivers
    print(f"\nTOP CAUSAL DRIVERS (≥ 3% of invariant prediction):")
    for idx in sorted_idx:
        pct = 100 * mean_abs[idx] / total if total > 0 else 0
        if pct < 3:
            break
        direction = "↑ INCREASES" if np.mean(shap_values[:, idx]) > 0 else "↓ DECREASES"
        group = feat_to_group.get(feature_names[idx], 'Other')
        print(f"  → [{group}] {feature_names[idx]:<35s} ({pct:.1f}%)  {direction} protest")

    return shap_values


def main():
    t0 = time.time()
    X, y, envs, feature_names, feat_to_group = load_data()

    seeds = {"ERM": 42, "V-REx": 123}  # Different seeds so weights diverge
    for method in ["ERM", "V-REx"]:
        print(f"\n{'='*60}")
        print(f"Training CVAE ({method}) with {X.shape[1]} enriched features...")
        print(f"{'='*60}")
        torch.manual_seed(seeds[method]); np.random.seed(seeds[method])
        model = train_cvae(X, y, envs, method=method)
        run_shap(model, X, y, feature_names, feat_to_group, method)

    print(f"\nCompleted in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
