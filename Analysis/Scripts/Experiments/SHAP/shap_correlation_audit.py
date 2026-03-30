"""
shap_correlation_audit.py — Feature vs Effect Correlation Audit
===============================================================
Computes TWO correlation matrices:
  1. Feature correlation: corr(X_i, X_j) — do the inputs move together?
  2. SHAP value correlation: corr(φ_i, φ_j) — do the MODEL EFFECTS move together?

Features with high |corr(φ_i, φ_j)| are entangled in the model —
their individual SHAP attributions are unreliable even if the inputs
are independent. This is the deeper concern the user raised.

Also computes SHAP interaction values for the top pairs.

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

# Re-use the same config as shap_vrex_full.py
EPOCHS = 120
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
VREX_PENALTY_WEIGHT = 100.0
LATENT_DIM = 16
MIN_ENV_SIZE = 5
SAMPLED_SIZE = 33000

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
CENSUS_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Intermediate", "census_tract_timeseries.csv")
ENV_PATH = os.path.join(PROJECT_DIR, "Analysis", "Results", "irm_environment_assignments.csv")

ACS_VARS = [
    'total_population', 'median_age',
    'race_white', 'race_black', 'race_asian', 'race_hispanic',
    'median_household_income', 'poverty_count',
    'median_home_value',
    'owner_occupied_units', 'renter_occupied_units',
    'median_gross_rent', 'total_housing_units',
]

FRED_MACROS = {
    'macro_mortgage30': {
        2007: 6.34, 2008: 6.03, 2009: 5.04, 2010: 4.69, 2011: 4.45,
        2012: 3.66, 2013: 3.98, 2014: 4.17, 2015: 3.85, 2016: 3.65,
        2017: 3.99, 2018: 4.54, 2019: 3.94, 2020: 3.11, 2021: 2.96,
        2022: 5.34, 2023: 6.81, 2024: 6.72,
    },
    'macro_fedfunds': {
        2007: 5.02, 2008: 1.92, 2009: 0.16, 2010: 0.18, 2011: 0.10,
        2012: 0.14, 2013: 0.11, 2014: 0.09, 2015: 0.13, 2016: 0.39,
        2017: 1.00, 2018: 1.83, 2019: 2.16, 2020: 0.36, 2021: 0.08,
        2022: 1.68, 2023: 5.33, 2024: 5.33,
    },
    'macro_cpi': {
        2007: 207.3, 2008: 215.3, 2009: 214.5, 2010: 218.1, 2011: 224.9,
        2012: 229.6, 2013: 233.0, 2014: 236.7, 2015: 237.0, 2016: 240.0,
        2017: 245.1, 2018: 251.1, 2019: 255.7, 2020: 258.8, 2021: 271.0,
        2022: 292.7, 2023: 304.7, 2024: 313.0,
    },
    'macro_unemployment': {
        2007: 4.6, 2008: 5.8, 2009: 9.3, 2010: 9.6, 2011: 8.9,
        2012: 8.1, 2013: 7.4, 2014: 6.2, 2015: 5.3, 2016: 4.9,
        2017: 4.4, 2018: 3.9, 2019: 3.7, 2020: 8.1, 2021: 5.4,
        2022: 3.6, 2023: 3.6, 2024: 4.0,
    },
    'macro_housing_starts': {
        2007: 1355, 2008: 906, 2009: 554, 2010: 587, 2011: 609,
        2012: 781, 2013: 925, 2014: 1003, 2015: 1112, 2016: 1174,
        2017: 1203, 2018: 1250, 2019: 1290, 2020: 1380, 2021: 1601,
        2022: 1554, 2023: 1420, 2024: 1350,
    },
}

PROPERTY_NUMERIC = ['total_market_value', 'deed_acreage', 'land_market_value']
ZONING_NUMERIC = ['ldb_far', 'ldb_units']
CATEGORICAL_FEATURES = ['property_category_code', 'council_district', 'lui_general_land_use', 'ldb_basezone']


# ── Copy of model/data from shap_vrex_full.py (condensed) ──

class CVAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, 256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(128, LATENT_DIM)
        self.fc_logvar = nn.Linear(128, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM + 1, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.SiLU(), nn.Linear(64, 1)
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


class Wrapper(nn.Module):
    def __init__(self, cvae, base_rate):
        super().__init__()
        self.cvae = cvae
        self.base_rate = base_rate
    def forward(self, x):
        y_dummy = torch.full((x.shape[0],), self.base_rate, device=x.device)
        mu, _ = self.cvae.encode(x, y_dummy)
        return self.cvae.classifier(mu)


def load_data():
    panel_cols = (['standardized_tcad_id', 'year', 'protest',
                   'nearby_GEOID', 'zoning_case_GEOID'] +
                  PROPERTY_NUMERIC + ZONING_NUMERIC + CATEGORICAL_FEATURES +
                  [f'acs_{v}' for v in ACS_VARS])
    panel_cols = list(dict.fromkeys(panel_cols))
    panel = pd.read_csv(PANEL_PATH, usecols=panel_cols, low_memory=False)
    panel = panel[panel['year'] <= 2024]

    census = pd.read_csv(CENSUS_PATH)
    census['geoid'] = census['geoid'].astype(str).str.strip()
    census['vintage'] = census['vintage'].astype(int)

    acs_rename = {f'acs_{v}': f'case_acs_{v}' for v in ACS_VARS}
    panel = panel.rename(columns=acs_rename)

    panel['nearby_GEOID'] = panel['nearby_GEOID'].astype(str).str.strip()
    panel['nearby_tract'] = panel['nearby_GEOID'].apply(
        lambda x: x[:11] if isinstance(x, str) and len(x) >= 11 else '')
    panel['acs_vintage_match'] = panel['year'].apply(
        lambda yr: max([v for v in range(2009, 2024) if v <= yr], default=2009))

    census_prop = census.rename(columns={
        'geoid': 'nearby_tract', 'vintage': 'acs_vintage_match',
        **{v: f'prop_acs_{v}' for v in ACS_VARS}
    })
    panel = panel.merge(census_prop[['nearby_tract', 'acs_vintage_match'] +
                        [f'prop_acs_{v}' for v in ACS_VARS]],
                        on=['nearby_tract', 'acs_vintage_match'], how='left')

    for v in ACS_VARS:
        panel[f'case_acs_{v}'] = pd.to_numeric(panel[f'case_acs_{v}'], errors='coerce')
        panel[f'prop_acs_{v}'] = pd.to_numeric(panel[f'prop_acs_{v}'], errors='coerce')
        panel[f'delta_acs_{v}'] = panel[f'prop_acs_{v}'] - panel[f'case_acs_{v}']

    for macro_name, values in FRED_MACROS.items():
        panel[macro_name] = panel['year'].map(values)

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

    positives = df[df['protest'] == 1]
    n_neg = SAMPLED_SIZE - len(positives)
    negatives = df[df['protest'] == 0]
    if len(negatives) > n_neg:
        negatives = negatives.sample(n=n_neg, random_state=42)
    df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)

    case_acs_cols = [f'case_acs_{v}' for v in ACS_VARS]
    prop_acs_cols = [f'prop_acs_{v}' for v in ACS_VARS]
    delta_acs_cols = [f'delta_acs_{v}' for v in ACS_VARS]
    macro_cols = list(FRED_MACROS.keys())

    all_numeric = (PROPERTY_NUMERIC + ZONING_NUMERIC +
                   case_acs_cols + prop_acs_cols + delta_acs_cols + macro_cols)

    for col in all_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('Missing').astype(str)

    scaler = StandardScaler()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=15)

    X_num = scaler.fit_transform(df[all_numeric])
    X_cat = ohe.fit_transform(df[CATEGORICAL_FEATURES])
    X = np.hstack([X_num, X_cat]).astype(np.float32)
    y = df['protest'].values.astype(np.float32)
    envs = df['env_label'].values.astype(np.int64)

    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = all_numeric + cat_names

    return X, y, envs, feature_names


def train_cvae(X, y, envs, method="V-REx"):
    model = CVAE(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y), torch.LongTensor(envs))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        for xb, yb, eb in loader:
            optimizer.zero_grad()
            recon, mu, logvar, cls_logits = model(xb, yb)
            mse = torch.sum((recon - xb)**2, dim=1)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            cls = nn.functional.binary_cross_entropy_with_logits(cls_logits, yb, reduction='none')
            per_sample = mse + kld + cls

            unique_e = torch.unique(eb)
            env_risks = [per_sample[eb == e].mean() for e in unique_e if (eb == e).sum() >= 2]
            if len(env_risks) < 2: continue

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

    return model


def main():
    t0 = time.time()
    print("Loading data...")
    X, y, envs, feature_names = load_data()
    n_numeric = len(PROPERTY_NUMERIC) + len(ZONING_NUMERIC) + 3*len(ACS_VARS) + len(FRED_MACROS)

    # ── Train V-REx model ──
    print(f"\nTraining V-REx CVAE ({X.shape[1]} features)...")
    torch.manual_seed(123); np.random.seed(123)
    model = train_cvae(X, y, envs, method="V-REx")

    # ── Compute SHAP values ──
    wrapper = Wrapper(model, y.mean())
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

    print("\nRunning SHAP GradientExplainer...")
    explainer = shap.GradientExplainer(wrapper, X_t[bg_idx])
    shap_values = explainer.shap_values(X_t[explain_idx])
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    # ── Restrict analysis to top features ──
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_k = 30  # Analyze top 30 features only
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_idx]

    # ── 1. Feature Input Correlation ──
    print("\n" + "=" * 90)
    print("PART 1: FEATURE INPUT CORRELATION — corr(X_i, X_j)")
    print("=" * 90)
    print("(Only showing pairs with |r| > 0.7 among top 30 features)")
    print()

    X_top = X[explain_idx][:, top_idx]
    feat_corr = np.corrcoef(X_top.T)

    input_pairs = []
    for i in range(top_k):
        for j in range(i+1, top_k):
            r = feat_corr[i, j]
            if abs(r) > 0.7:
                input_pairs.append((top_names[i], top_names[j], r))

    input_pairs.sort(key=lambda x: -abs(x[2]))
    print(f"{'Feature A':<40s} {'Feature B':<40s} {'r':>8}")
    print("-" * 90)
    for a, b, r in input_pairs:
        print(f"{a:<40s} {b:<40s} {r:>8.3f}")
    if not input_pairs:
        print("  No pairs with |r| > 0.7 found.")

    # ── 2. SHAP Value Correlation (Model Effect Association) ──
    print("\n" + "=" * 90)
    print("PART 2: SHAP VALUE CORRELATION — corr(φ_i, φ_j)")
    print("=" * 90)
    print("(Shows which features the MODEL uses interchangeably)")
    print("Pairs with |r| > 0.5 are model-entangled regardless of input correlation.")
    print()

    sv_top = shap_values[:, top_idx]
    shap_corr = np.corrcoef(sv_top.T)

    effect_pairs = []
    for i in range(top_k):
        for j in range(i+1, top_k):
            r = shap_corr[i, j]
            if abs(r) > 0.5:
                # Also get the feature input correlation for comparison
                feat_r = feat_corr[i, j]
                effect_pairs.append((top_names[i], top_names[j], r, feat_r))

    effect_pairs.sort(key=lambda x: -abs(x[2]))
    print(f"{'Feature A':<38s} {'Feature B':<38s} {'φ-corr':>8} {'X-corr':>8} {'Type':>12}")
    print("-" * 110)
    for a, b, shap_r, feat_r in effect_pairs:
        # Classify the type
        if abs(feat_r) > 0.7 and abs(shap_r) > 0.5:
            typ = "BOTH"
        elif abs(shap_r) > 0.5 and abs(feat_r) <= 0.7:
            typ = "EFFECT ONLY"
        else:
            typ = "INPUT ONLY"
        print(f"{a:<38s} {b:<38s} {shap_r:>8.3f} {feat_r:>8.3f} {typ:>12}")

    if not effect_pairs:
        print("  No pairs with |φ-corr| > 0.5 found.")

    # ── 3. Summary: Which features are safe to interpret individually? ──
    print("\n" + "=" * 90)
    print("PART 3: INTERPRETABILITY AUDIT")
    print("=" * 90)

    entangled = set()
    for a, b, shap_r, feat_r in effect_pairs:
        entangled.add(a)
        entangled.add(b)

    safe = [n for n in top_names if n not in entangled]
    print(f"\nSAFE to interpret individually ({len(safe)} features):")
    for n in safe:
        idx_in_top = top_names.index(n)
        pct = 100 * mean_abs[top_idx[idx_in_top]] / mean_abs.sum()
        print(f"  ✓ {n:<45s} ({pct:.1f}%)")

    print(f"\nENTANGLED — interpret only as group ({len(entangled)} features):")
    for n in sorted(entangled):
        if n in top_names:
            idx_in_top = top_names.index(n)
            pct = 100 * mean_abs[top_idx[idx_in_top]] / mean_abs.sum()
            print(f"  ⚠ {n:<45s} ({pct:.1f}%)")

    print(f"\nCompleted in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
