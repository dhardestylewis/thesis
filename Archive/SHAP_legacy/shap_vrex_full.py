"""
shap_vrex_full.py — Full-Feature SHAP on V-REx CVAE
====================================================
Uses the COMPLETE enriched feature set:
  1. Case-area demographics (acs_*) — from zoning_case_GEOID
  2. Property-tract demographics (prop_acs_*) — from nearby_GEOID
  3. Demographic deltas (delta_acs_*) — property tract minus case tract
  4. FRED macroeconomic indicators — by year
  5. Property attributes + LDB zoning + Land Use

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
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
VREX_PENALTY_WEIGHT = 100.0
LATENT_DIM = 16  # Expanded for richer feature set
MIN_ENV_SIZE = 5
SAMPLED_SIZE = 33000

PROJECT_DIR = r"c:\Users\dhl\data\thesis\thesis"
PANEL_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Output", "Property_Year_Panel_Enriched.csv")
CENSUS_PATH = os.path.join(PROJECT_DIR, "Data", "Panel", "Intermediate", "census_tract_timeseries.csv")
ENV_PATH = os.path.join(PROJECT_DIR, "Analysis", "Results", "irm_environment_assignments.csv")

# ACS variable names (matching census_tract_timeseries.csv columns)
ACS_VARS = [
    'total_population', 'median_age',
    'race_white', 'race_black', 'race_asian', 'race_hispanic',
    'median_household_income', 'poverty_count',
    'median_home_value',
    'owner_occupied_units', 'renter_occupied_units',
    'median_gross_rent', 'total_housing_units',
]

# FRED Macro Indicators — Annual Averages (public data, 2007-2024)
# Sources: FRED/St. Louis Fed
FRED_MACROS = {
    'macro_mortgage30': {  # 30-Year Fixed Mortgage Rate (%)
        2007: 6.34, 2008: 6.03, 2009: 5.04, 2010: 4.69, 2011: 4.45,
        2012: 3.66, 2013: 3.98, 2014: 4.17, 2015: 3.85, 2016: 3.65,
        2017: 3.99, 2018: 4.54, 2019: 3.94, 2020: 3.11, 2021: 2.96,
        2022: 5.34, 2023: 6.81, 2024: 6.72,
    },
    'macro_fedfunds': {  # Effective Federal Funds Rate (%)
        2007: 5.02, 2008: 1.92, 2009: 0.16, 2010: 0.18, 2011: 0.10,
        2012: 0.14, 2013: 0.11, 2014: 0.09, 2015: 0.13, 2016: 0.39,
        2017: 1.00, 2018: 1.83, 2019: 2.16, 2020: 0.36, 2021: 0.08,
        2022: 1.68, 2023: 5.33, 2024: 5.33,
    },
    'macro_cpi': {  # CPI-U Index (1982-84=100)
        2007: 207.3, 2008: 215.3, 2009: 214.5, 2010: 218.1, 2011: 224.9,
        2012: 229.6, 2013: 233.0, 2014: 236.7, 2015: 237.0, 2016: 240.0,
        2017: 245.1, 2018: 251.1, 2019: 255.7, 2020: 258.8, 2021: 271.0,
        2022: 292.7, 2023: 304.7, 2024: 313.0,
    },
    'macro_unemployment': {  # US Unemployment Rate (%)
        2007: 4.6, 2008: 5.8, 2009: 9.3, 2010: 9.6, 2011: 8.9,
        2012: 8.1, 2013: 7.4, 2014: 6.2, 2015: 5.3, 2016: 4.9,
        2017: 4.4, 2018: 3.9, 2019: 3.7, 2020: 8.1, 2021: 5.4,
        2022: 3.6, 2023: 3.6, 2024: 4.0,
    },
    'macro_housing_starts': {  # US Housing Starts (thousands, annual)
        2007: 1355, 2008: 906, 2009: 554, 2010: 587, 2011: 609,
        2012: 781, 2013: 925, 2014: 1003, 2015: 1112, 2016: 1174,
        2017: 1203, 2018: 1250, 2019: 1290, 2020: 1380, 2021: 1601,
        2022: 1554, 2023: 1420, 2024: 1350,
    },
}

# Feature organization
PROPERTY_NUMERIC = ['total_market_value', 'deed_acreage', 'land_market_value']
ZONING_NUMERIC = ['ldb_far', 'ldb_units']
CATEGORICAL_FEATURES = ['property_category_code', 'council_district', 'lui_general_land_use', 'ldb_basezone']

FEATURE_GROUPS = {}  # Will be populated dynamically


# ──────────── Data Loading ────────────

def load_data():
    print("=" * 60)
    print("LOADING AND ENRICHING PANEL")
    print("=" * 60)

    # 1. Load enriched panel
    print("Loading enriched panel...")
    panel_cols = (['standardized_tcad_id', 'year', 'protest',
                   'nearby_GEOID', 'zoning_case_GEOID'] +
                  PROPERTY_NUMERIC + ZONING_NUMERIC + CATEGORICAL_FEATURES +
                  [f'acs_{v}' for v in ACS_VARS])
    panel_cols = list(dict.fromkeys(panel_cols))
    panel = pd.read_csv(PANEL_PATH, usecols=panel_cols, low_memory=False)
    panel = panel[panel['year'] <= 2024]
    print(f"  Panel: {len(panel):,} rows")

    # 2. Load census tract timeseries
    print("Loading census tract timeseries...")
    census = pd.read_csv(CENSUS_PATH)
    census['geoid'] = census['geoid'].astype(str).str.strip()
    census['vintage'] = census['vintage'].astype(int)
    print(f"  Census: {len(census):,} tract-year records")

    # 3. Rename existing acs_* columns to case_acs_* (they're on zoning_case_GEOID)
    acs_rename = {f'acs_{v}': f'case_acs_{v}' for v in ACS_VARS}
    panel = panel.rename(columns=acs_rename)

    # 4. Join protester-tract demographics on nearby_GEOID
    print("Joining protester-tract demographics on nearby_GEOID...")
    panel['nearby_GEOID'] = panel['nearby_GEOID'].astype(str).str.strip()
    # Truncate to 11-digit tract GEOID
    panel['nearby_tract'] = panel['nearby_GEOID'].apply(
        lambda x: x[:11] if isinstance(x, str) and len(x) >= 11 else '')

    # Find best ACS vintage for each panel year
    panel['acs_vintage_match'] = panel['year'].apply(
        lambda yr: max([v for v in range(2009, 2024) if v <= yr], default=2009))

    # Merge census data for property tract
    census_prop = census.rename(columns={
        'geoid': 'nearby_tract',
        'vintage': 'acs_vintage_match',
        **{v: f'prop_acs_{v}' for v in ACS_VARS}
    })
    panel = panel.merge(census_prop[['nearby_tract', 'acs_vintage_match'] +
                        [f'prop_acs_{v}' for v in ACS_VARS]],
                        on=['nearby_tract', 'acs_vintage_match'], how='left')
    prop_matched = panel[[f'prop_acs_{ACS_VARS[0]}']].notna().sum().iloc[0]
    print(f"  Property-tract matched: {prop_matched:,} / {len(panel):,} "
          f"({100*prop_matched/len(panel):.1f}%)")

    # 5. Compute demographic deltas (property tract - case tract)
    print("Computing demographic deltas...")
    for v in ACS_VARS:
        case_col = f'case_acs_{v}'
        prop_col = f'prop_acs_{v}'
        delta_col = f'delta_acs_{v}'
        panel[case_col] = pd.to_numeric(panel[case_col], errors='coerce')
        panel[prop_col] = pd.to_numeric(panel[prop_col], errors='coerce')
        panel[delta_col] = panel[prop_col] - panel[case_col]

    # 6. Add FRED macros by year
    print("Adding FRED macro indicators...")
    for macro_name, values in FRED_MACROS.items():
        panel[macro_name] = panel['year'].map(values)

    # 7. Load environment assignments
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

    # 8. Sample
    positives = df[df['protest'] == 1]
    n_neg = SAMPLED_SIZE - len(positives)
    negatives = df[df['protest'] == 0]
    if len(negatives) > n_neg:
        negatives = negatives.sample(n=n_neg, random_state=42)
    df = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 9. Build feature matrix
    # Numeric features: property + zoning + case_acs + prop_acs + delta_acs + macros
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

    # Build feature-to-group mapping
    feat_to_group = {}
    for f in PROPERTY_NUMERIC:
        feat_to_group[f] = 'Property'
    for f in ZONING_NUMERIC:
        feat_to_group[f] = 'Zoning/Density'
    for f in case_acs_cols:
        base = f.replace('case_acs_', '')
        if 'race' in base:
            feat_to_group[f] = 'Case Tract: Race'
        elif 'income' in base or 'poverty' in base:
            feat_to_group[f] = 'Case Tract: Income'
        elif 'owner' in base or 'renter' in base or 'rent' in base or 'housing' in base:
            feat_to_group[f] = 'Case Tract: Tenure'
        else:
            feat_to_group[f] = 'Case Tract: Demo'
    for f in prop_acs_cols:
        base = f.replace('prop_acs_', '')
        if 'race' in base:
            feat_to_group[f] = 'Prop Tract: Race'
        elif 'income' in base or 'poverty' in base:
            feat_to_group[f] = 'Prop Tract: Income'
        elif 'owner' in base or 'renter' in base or 'rent' in base or 'housing' in base:
            feat_to_group[f] = 'Prop Tract: Tenure'
        else:
            feat_to_group[f] = 'Prop Tract: Demo'
    for f in delta_acs_cols:
        base = f.replace('delta_acs_', '')
        if 'race' in base:
            feat_to_group[f] = 'Δ Demo: Race'
        elif 'income' in base or 'poverty' in base:
            feat_to_group[f] = 'Δ Demo: Income'
        elif 'owner' in base or 'renter' in base or 'rent' in base or 'housing' in base:
            feat_to_group[f] = 'Δ Demo: Tenure'
        else:
            feat_to_group[f] = 'Δ Demo: General'
    for f in macro_cols:
        feat_to_group[f] = 'Macro/FRED'
    for cn in cat_names:
        for cat_col in CATEGORICAL_FEATURES:
            if cn.startswith(cat_col):
                grp = {'property_category_code': 'Land Use',
                       'council_district': 'Geography',
                       'lui_general_land_use': 'Land Use',
                       'ldb_basezone': 'Zoning/Density'}
                feat_to_group[cn] = grp.get(cat_col, 'Other')
                break

    print(f"\n{'='*60}")
    print(f"FINAL DATASET")
    print(f"{'='*60}")
    print(f"  Rows: {len(df):,} | Features: {X.shape[1]}")
    print(f"  Numeric: {len(all_numeric)} ({len(PROPERTY_NUMERIC)} property + "
          f"{len(ZONING_NUMERIC)} zoning + {len(case_acs_cols)} case ACS + "
          f"{len(prop_acs_cols)} prop ACS + {len(delta_acs_cols)} delta + "
          f"{len(macro_cols)} macro)")
    print(f"  Categorical (OHE): {len(cat_names)}")
    print(f"  Base rate: {y.mean():.3f}")

    return X, y, envs, feature_names, feat_to_group


# ──────────── CVAE Model ────────────

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


# ──────────── Training ────────────

def train_cvae(X, y, envs, method="V-REx"):
    model = CVAE(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y), torch.LongTensor(envs))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        n_batches = 0
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
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            avg = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1:3d} | Loss: {avg:.6f}")

    return model


# ──────────── SHAP ────────────

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

    print(f"\n{'='*90}")
    print(f"SHAP ATTRIBUTION — {method_name} CVAE (Full Enrichment)")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Group':<22} {'Feature':<45} {'|SHAP|':>10} {'%':>7}")
    print("-" * 90)

    for rank, idx in enumerate(sorted_idx):
        pct = 100 * mean_abs[idx] / total if total > 0 else 0
        if pct < 0.3:
            print(f"  ... {len(sorted_idx) - rank} more features < 0.3%")
            break
        group = feat_to_group.get(feature_names[idx], 'Other')
        print(f"{rank+1:<5} {group:<22} {feature_names[idx]:<45} {mean_abs[idx]:>10.6f} {pct:>6.1f}%")

    # Group-level summary
    print(f"\n{'='*90}")
    print(f"GROUP-LEVEL ATTRIBUTION — {method_name}")
    print(f"{'='*90}")
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
        print(f"{g:<30} {val:>14.6f} {pct:>6.1f}%  {bar}")

    # Top drivers with directions
    print(f"\nTOP INVARIANT DRIVERS ({method_name}, ≥ 2%):")
    for idx in sorted_idx:
        pct = 100 * mean_abs[idx] / total if total > 0 else 0
        if pct < 2:
            break
        direction = "↑ INCREASES" if np.mean(shap_values[:, idx]) > 0 else "↓ DECREASES"
        group = feat_to_group.get(feature_names[idx], 'Other')
        print(f"  → [{group}] {feature_names[idx]:<40s} ({pct:.1f}%)  {direction} protest")

    return shap_values


# ──────────── Main ────────────

def main():
    t0 = time.time()
    X, y, envs, feature_names, feat_to_group = load_data()

    seeds = {"ERM": 42, "V-REx": 123}
    for method in ["ERM", "V-REx"]:
        print(f"\n{'='*60}")
        print(f"Training CVAE ({method}) with {X.shape[1]} features...")
        print(f"{'='*60}")
        torch.manual_seed(seeds[method]); np.random.seed(seeds[method])
        model = train_cvae(X, y, envs, method=method)
        run_shap(model, X, y, feature_names, feat_to_group, method)

    print(f"\nCompleted in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
