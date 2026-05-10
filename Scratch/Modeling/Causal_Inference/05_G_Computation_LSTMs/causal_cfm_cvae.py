"""
causal_cfm_cvae.py
==================
Causal Seq2Seq CVAE with Conditional Flow Matching (CFM) Decoder.

Architecture:
  - BiLSTM Encoder: X_obs → (μ, σ) → z  [case-level confounder latent]
  - LSTM Transition: autoregressive hidden state across biweekly periods
  - CFM Decoder: velocity field v_θ(y_τ, τ, z, h_t, dose) that transports
    noise → outcome distribution. Replaces the Gaussian linear head.

Causal identification (unchanged from Seq2SeqCVAE):
  - Treatment intervention: petition_pct overridden directly (do-operator)
  - Counterfactuals: hold z fixed, vary dose
  - Confounding: KL forces z to encode residual confounders
  - ITE = E[Y(d)] - E[Y(d')] from the same z

Training objective:
  L = L_CFM + β * KL(q(z|X) || N(0,I))
  L_CFM = E_τ[||v_θ(y_τ, τ, z, h_t, d) - (y_true - ε)||²]
  where y_τ = τ*y_true + (1-τ)*ε, τ ~ Uniform(0,1)

Reference:
  Lipman et al. 2022 "Flow Matching for Generative Modeling"
  Tong et al. 2023 "Improving and Generalizing Flow-Matching" (torchcfm)
"""

import os, time, math

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
PANEL_PATH = os.environ.get("PANEL_PATH", "aws_deploy/biweekly_panel_aws.csv")
OUT_DIR    = os.environ.get("OUT_DIR",    "aws_deploy")
S3_BUCKET  = os.environ.get("S3_BUCKET",  "s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline")

HIDDEN_DIM  = 256
LATENT_DIM  = 64
CFM_HIDDEN  = 512   # velocity field hidden dim (larger = more expressive)
CFM_LAYERS  = 5     # depth of velocity MLP
N_LAYERS    = 3     # LSTM layers
BATCH_SIZE  = 64
EPOCHS      = 80
LR          = 3e-4
BETA        = 0.5   # KL weight
CFM_STEPS   = 20    # ODE steps at inference (Euler)
MC_SAMPLES  = 10    # Monte Carlo draws per case at val time

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "petition_pct_this_period", "cumulative_petition_pct",
    "cumulative_council_hearings_lag1", "cumulative_commission_hearings_lag1",
    "cumulative_council_nlp_lag1", "cumulative_yea_votes", "cumulative_nay_votes",
    "net_vote_margin", "council_yea_pct", "council_nay_pct", "council_abstain_pct",
    "knn_petition_rate_1km",
    "active_cases_100m", "active_cases_250m", "active_cases_500m",
    "active_cases_1km", "active_cases_2km", "active_gravity_index_t",
    "council_nlp_document_count", "council_nlp_density_hits",
    "council_nlp_oppose_hits", "council_nlp_traffic_hits",
    "Aggregate_Sentiment", "pdf_requested_height_ft",
    "appraised_value", "building_age", "affordability_proxy",
    "fed_funds_rate", "fed_funds_rate_momentum", "fed_funds_rate_1yr_lag",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_1yr_lag",
    "local_unemployment_rate", "local_unemployment_rate_momentum",
    "total_population", "median_household_income", "renter_share", "rent_burden",
    "race_white", "race_black", "race_hispanic", "median_age",
    "commission_hearings_this_period", "council_hearings_this_period",
    "filing_event", "hearing_frequency", "hearing_velocity_3p",
    "cumulative_commission_hearings", "cumulative_council_hearings",
    "bw_sin", "bw_cos", "census_tract", "council_district",
    "cumulative_petition_count", "cumulative_petition_events",
    "cumulative_petition_count_lag1", "cumulative_petition_events_lag1",
    "cumulative_petition_pct", "dist_petition_rate_lag1",
    "petition_intensity_per_ft", "petition_velocity_3p", "petition_velocity_3p_lag1",
    "cumulative_petition_pct_lag1",   # explicit shift(1).cumsum() — safe for propensity
    "pdf_requested_max_far", "pdf_proposed_height_ft",
    "pdf_story_count", "pdf_story_height_ft", "pdf_compatibility_height_ft"
]

# Targets: cumulative lagged states (not per-period increments)
# These are exactly what the autoregressive state propagates, and they're
# interpretable: "by month 12, 4.2 commission hearings accumulated"
TARGETS = [
    "resolved",                          # [0] binary → sigmoid
    "height_concession_pct",             # [1] hurdle: gate + conditional CFM
    "cumulative_nlp_total_tokens_lag1",  # [2] cumulative total paperwork tokens
    "cumulative_commission_hearings_lag1",# [3] cumulative commission hearings
    "cumulative_council_hearings_lag1",  # [4] cumulative council hearings
]
Y_DIM = len(TARGETS)
HEIGHT_IDX = 1  # index of height_concession_pct in TARGETS — handled by hurdle decoder

# Feature indices for autoregressive state update
CUM_TOK_COL   = "cumulative_council_nlp_lag1"
CUM_COMM_COL  = "cumulative_commission_hearings_lag1"
CUM_COUN_COL  = "cumulative_council_hearings_lag1"


def load_data():
    print(f"Loading data from {PANEL_PATH}...", flush=True)
    df = pd.read_csv(PANEL_PATH)

    df = df.sort_values(['case_number', 'period_seq'])

    def _fraction_01(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
        # Robustly handle either 0–1 or 0–100 source panels.
        if x.quantile(0.99) > 1.0:
            x = x / 100.0
        return x.clip(lower=0.0, upper=1.0)

    if "petition_pct_this_period" not in df.columns:
        df["petition_pct_this_period"] = 0.0
    df["petition_pct_this_period"] = _fraction_01(df["petition_pct_this_period"])

    # Recompute cumulative petition state from the normalized current-period dose.
    df["cumulative_petition_pct"] = (
        df.groupby("case_number")["petition_pct_this_period"]
          .transform(lambda s: s.fillna(0.0).cumsum())
          .clip(lower=0.0, upper=1.0)
    )
    df["cumulative_petition_pct_lag1"] = (
        df.groupby("case_number")["petition_pct_this_period"]
          .transform(lambda s: s.shift(1).fillna(0.0).cumsum())
          .clip(lower=0.0, upper=1.0)
    )

    # Explicit lagged petition history for PropNet only.
    if "petition_count_this_period" in df.columns:
        pet_count_t = pd.to_numeric(df["petition_count_this_period"], errors="coerce").fillna(0.0)
    else:
        pet_count_t = (df["petition_pct_this_period"] > 0).astype(float)
    df["_petition_event_t"] = pet_count_t
    df["cumulative_petition_count_lag1"] = (
        df.groupby("case_number")["_petition_event_t"]
          .transform(lambda s: s.shift(1).fillna(0.0).cumsum())
    )
    df["cumulative_petition_events_lag1"] = df["cumulative_petition_count_lag1"]
    df["petition_velocity_3p_lag1"] = (
        df.groupby("case_number")["petition_pct_this_period"]
          .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
          .fillna(0.0)
    )

    # ─── Feature Engineering for Targets ──────────────────────────────────────────
    # Lagged cumulative features
    if 'council_hearings_this_period' in df.columns:
        df['cumulative_council_hearings_lag1'] = df.groupby('case_number')['council_hearings_this_period'].apply(lambda x: x.shift(1).fillna(0).cumsum()).reset_index(level=0, drop=True)
    if 'commission_hearings_this_period' in df.columns:
        df['cumulative_commission_hearings_lag1'] = df.groupby('case_number')['commission_hearings_this_period'].apply(lambda x: x.shift(1).fillna(0).cumsum()).reset_index(level=0, drop=True)
    if 'nlp_total_tokens' in df.columns:
        df['cumulative_nlp_total_tokens_lag1'] = df.groupby('case_number')['nlp_total_tokens'].apply(lambda x: x.shift(1).fillna(0).cumsum()).reset_index(level=0, drop=True)

    # ── Vote margin columns ───────────────────────────────────────────────────
    # Pre-engineered by enrich_biweekly_panel_votes.py into the panel CSV.
    # If columns are absent (legacy panels), fall back to zero-fill.
    for col in ["yea_votes_this_period", "nay_votes_this_period",
                "cumulative_yea_votes", "cumulative_nay_votes", "net_vote_margin"]:
        if col not in df.columns:
            df[col] = 0.0

    
    # Net height change reframed as a concession percentage to normalize across building scales
    if "pdf_requested_height_ft" in df.columns:
        initial_req = df.groupby("case_number")["pdf_requested_height_ft"].transform("max")
        current_constraint = df[["pdf_requested_height_ft", "pdf_staff_recommends_ht"]].min(axis=1) if "pdf_staff_recommends_ht" in df.columns else df["pdf_requested_height_ft"]
        current_constraint = current_constraint.fillna(initial_req)
        final_ht = df["pdf_reduced_to_ft"].fillna(current_constraint).fillna(0) if "pdf_reduced_to_ft" in df.columns else current_constraint.fillna(0)
        
        concession = (initial_req - final_ht) / initial_req.replace(0, np.nan)
        df["height_concession_pct"] = concession.clip(lower=0.0, upper=1.0).fillna(0.0)
    else:
        df["height_concession_pct"] = 0.0

    # Petition percentage columns were normalized/recomputed above. Do not divide again.

    # Log-transform fat-tailed variables before z-scoring to preserve extremes without blowing up variance
    for col in ["cumulative_commission_hearings_lag1", "cumulative_council_hearings_lag1", "cumulative_nlp_total_tokens_lag1"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])



    # De-duplicate features list against what's actually in df
    available_features = [f for f in dict.fromkeys(FEATURES) if f in df.columns]
    available_targets  = [t for t in TARGETS if t in df.columns]
    print(f"Features: {len(available_features)}  Targets: {len(available_targets)}", flush=True)

    # Z-score continuous features
    # Excluded: binary flags, strict percentages, zero-inflated outcomes with point mass at 0
    global ZSCORE_EXCLUDE
    ZSCORE_EXCLUDE = {
        "resolved",
        "petition_pct_this_period",
        "cumulative_petition_pct",
        "height_concession_pct",   # 85% zeros — z-scoring destroys the spike structure
    }
    norm_dict = {}
    for col in available_features + available_targets:
        if col in ZSCORE_EXCLUDE:
            continue
        vals = df[col].fillna(0).values
        m, s = vals.mean(), vals.std()
        s = max(s, 1e-8)
        df[col] = (vals - m) / s
        norm_dict[col] = (m, s)

    # ── Spatio-Covariate Clustering ──
    from sklearn.cluster import KMeans
    first_periods = df.groupby("case_number").first()
    cluster_features = first_periods[["latitude", "longitude"]].copy()
    if "proposed_max_far" in first_periods.columns:
        cluster_features["far"] = first_periods["proposed_max_far"].fillna(0)
    if "pdf_requested_height_ft" in first_periods.columns:
        cluster_features["height"] = first_periods["pdf_requested_height_ft"].fillna(0)
    
    cluster_features["dose"] = df.groupby("case_number")["petition_pct_this_period"].max().fillna(0)
    
    scaled_coords = StandardScaler().fit_transform(cluster_features.fillna(0).values)
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
    
    cell_assignments = pd.DataFrame(index=first_periods.index)
    cell_assignments["cell_id"] = kmeans.fit_predict(scaled_coords)


    # Build per-case tensors of shape (T, F) and (T, Y)
    cases = df["case_number"].unique()
    T_MAX = 55  # biweekly periods 0-54

    X_list, Y_list, lengths = [], [], []
    for case in cases:
        sub = df[df["case_number"] == case].sort_values("period_seq")
        T = min(len(sub), T_MAX)
        x = np.zeros((T_MAX, len(available_features)))
        y = np.zeros((T_MAX, len(available_targets)))
        x[:T] = sub[available_features].fillna(0).values[:T]
        y[:T] = sub[available_targets].fillna(0).values[:T]
        X_list.append(x)
        Y_list.append(y)
        lengths.append(T)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y_list), dtype=torch.float32)
    L = torch.tensor(lengths, dtype=torch.long)

    treat_idx = [available_features.index("petition_pct_this_period"),
                 available_features.index("cumulative_petition_pct")]

    filing_years = pd.to_datetime(first_periods["period_start"]).dt.year
    return X, Y, L, available_features, available_targets, norm_dict, treat_idx, cases, cell_assignments, filing_years


# ──────────────────────────────────────────────────────────────────────────────
# Model Components
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal embedding for flow time τ ∈ [0,1]."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tau):
        # tau: (B,)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=tau.device) / half)
        args  = tau[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


def focal_bce_raw(logit, target, gamma=2.0, alpha=0.85):
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p = torch.sigmoid(logit)
    pt = target * p + (1 - target) * (1 - p)
    w = target * alpha + (1 - target) * (1 - alpha)
    return ((1 - pt).clamp_min(1e-6) ** gamma) * bce * w


class PropensityNetwork(nn.Module):
    """Predicts Expected Dose given Confounders & Trajectory for Orthogonalization."""
    def __init__(self, z_dim, h_dim, raw_dim=0):
        super().__init__()
        self.z_dim = z_dim
        in_dim = (z_dim if z_dim > 0 else 0) + h_dim + raw_dim
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.SiLU()
        )
        self.head_cls = nn.Linear(256, 1)   # logit for nonzero
        self.head_reg = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )  # expected magnitude if nonzero

    def forward(self, z, h, raw=None):
        parts = []
        if self.z_dim > 0 and z is not None:
            parts.append(z)
        parts.append(h)
        if raw is not None:
            parts.append(raw)
        feat  = self.shared(torch.cat(parts, dim=-1))
        logit = self.head_cls(feat)
        mag   = self.head_reg(feat)
        return logit, mag

class VelocityField(nn.Module):
    """
    CFM velocity field v_θ(y_τ, τ, z, h, dose_enc) → dy/dτ

    This is the learned transport from noise → outcome distribution,
    conditioned on all causal context.
    """
    def __init__(self, y_dim, z_dim, h_dim, time_emb_dim=32, hidden=512, n_layers=5):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        in_dim = y_dim + time_emb_dim + z_dim + h_dim + 2  # +2 for Hurdle vector (is_nonzero, residual)
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, y_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y_t, tau, z, h, dose_enc):
        """
        y_t:      (B, Y)  noisy sample at flow time τ
        tau:      (B,)    flow time scalar
        z:        (B, Z)  causal confounder latent
        h:        (B, H)  LSTM hidden state (trajectory context)
        dose_enc: (B, 2)  hurdle dose encoding [is_nonzero, residual]
        """
        if self.training:
            # Targeted confounder dropout
            mask = (torch.rand(z.shape[0], 1, device=z.device) > 0.3).float()
            z = z * mask
            
        t_emb = self.time_emb(tau)                         # (B, time_emb_dim)
        x = torch.cat([y_t, t_emb, z, h, dose_enc], dim=-1)
        return self.net(x)


class CausalEncoder(nn.Module):
    """BiLSTM encoder: X_obs → (μ, σ) → z  [confounders]"""
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers,
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.mu_head    = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.rnn(x)
        ctx = out[:, -1, :]  # last timestep of bidirectional
        mu     = self.mu_head(ctx)
        logvar = self.logvar_head(ctx)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return z, mu, logvar


class CausalTransition(nn.Module):
    """LSTM that maintains autoregressive hidden state across timesteps."""
    def __init__(self, input_dim, y_dim, hidden_dim, n_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_dim + y_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=0.1)
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

    def step(self, x_t, y_prev, state):
        """Single-step transition.  x_t: (B, F)  y_prev: (B, Y)"""
        inp = torch.cat([x_t, y_prev], dim=-1).unsqueeze(1)  # (B, 1, F+Y)
        out, state = self.rnn(inp, state)
        return out.squeeze(1), state  # h_t: (B, hidden_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Full Causal CFM Model
# ──────────────────────────────────────────────────────────────────────────────

class CausalSeq2SeqCFM(nn.Module):
    def __init__(self, input_dim, y_dim, hidden_dim, latent_dim,
                 cfm_hidden, cfm_layers, n_layers, treat_idx,
                 f_cum_tok=None, f_cum_comm=None, f_cum_coun=None,
                 t_idx_tok=None, t_idx_comm=None, t_idx_coun=None,
                 skip_confounder_idx=None, prop_z_dim=None,
                 prop_petition_idx=None,
                 treatment_derived_idx=None):
        super().__init__()
        self.treat_idx          = treat_idx
        self.y_dim              = y_dim
        self.hidden_dim         = hidden_dim
        self.latent_dim         = latent_dim
        self.f_cum_tok          = f_cum_tok
        self.f_cum_comm         = f_cum_comm
        self.f_cum_coun         = f_cum_coun
        self.t_idx_tok          = t_idx_tok
        self.t_idx_comm         = t_idx_comm
        self.t_idx_coun         = t_idx_coun
        self.skip_confounder_idx = skip_confounder_idx
        self.prop_petition_idx  = prop_petition_idx  # lagged petition history features
        self.treatment_derived_idx = treatment_derived_idx or []
        self.mask_from_transition_idx = sorted(set((treat_idx or []) + self.treatment_derived_idx))

        # prop_z_dim defaults to latent_dim — z encodes petition history, essential for propensity
        _prop_z_dim = latent_dim if prop_z_dim is None else prop_z_dim

        self.encoder    = CausalEncoder(input_dim, hidden_dim, latent_dim, n_layers)
        self.transition = CausalTransition(input_dim, y_dim, hidden_dim, n_layers)

        raw_dim = (len(skip_confounder_idx) if skip_confounder_idx else 0) + \
                  (len(prop_petition_idx)   if prop_petition_idx   else 0)
        self.prop_net = PropensityNetwork(_prop_z_dim, hidden_dim, raw_dim=raw_dim)

        # Joint velocity field: all targets EXCEPT height (height handled by hurdle decoder)
        self.non_height_idx = [i for i in range(y_dim) if i != HEIGHT_IDX]

        self.cfm = VelocityField(
            y_dim=len(self.non_height_idx), z_dim=latent_dim, h_dim=hidden_dim,
            hidden=cfm_hidden, n_layers=cfm_layers
        )

        # ── Hurdle Height Decoder ─────────────────────────────────────────────
        # Gate: P(height_concession > 0 | z, h, dose_enc)
        self.height_gate = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim + 2, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        # Conditional CFM: p(height | height > 0, z, h, dose_enc)
        self.height_cfm = VelocityField(
            y_dim=1, z_dim=latent_dim, h_dim=hidden_dim,
            time_emb_dim=32, hidden=256, n_layers=3
        )

    # ── Training forward: compute CFM loss at all timesteps ──────────────────
    def forward_train(self, X, Y, L=None):
        """
        X: (B, T, F)   observed feature sequences
        Y: (B, T, Y)   target sequences
        L: (B,)        true sequence lengths (used to mask padded timesteps)
        Returns: cfm_loss (scalar), kl_loss (scalar), prop_loss (scalar)
        """
        B, T, _ = X.shape
        device = X.device

        # 1. Encode confounders from first 4 observed periods (pre-filing context)
        z, mu, logvar = self.encoder(X[:, :4, :])

        # 2. Vectorized Autoregressive Rollout (Teacher-Forcing)
        state = self.transition.init_hidden(B, device)

        # Sequence inputs for t=4..T
        x_seq = X[:, 4:, :]  # (B, T-4, F)
        y_true_seq = Y[:, 4:, :]  # (B, T-4, Y)
        # Shifted Y for previous step inputs. At t=4, y_prev is zeros.
        y_prev_seq = torch.cat([torch.zeros(B, 1, self.y_dim, device=device), Y[:, 4:-1, :]], dim=1)
        
        # TARGET LEAK FIX: Mask the petition dose out of the sequence before RNN steps
        x_seq_unbiased = x_seq.clone()
        for idx in self.mask_from_transition_idx:
            x_seq_unbiased[:, :, idx] = 0.0
            
        inp_seq = torch.cat([x_seq_unbiased, y_prev_seq], dim=-1)  # (B, T-4, F+Y)
        out_seq, _ = self.transition.rnn(inp_seq, state)  # (B, T-4, H) - Unbiased context!

        # Flatten sequences for CFM operations
        B_seq = B * (T - 4)
        y_true_flat = y_true_seq.reshape(B_seq, self.y_dim)
        dose_flat = x_seq[:, :, self.treat_idx[0]].unsqueeze(-1).reshape(B_seq, 1)
        z_flat = z.unsqueeze(1).repeat(1, T - 4, 1).reshape(B_seq, -1)
        h_flat = out_seq.reshape(B_seq, self.hidden_dim)
        
        # raw_confounders: static demographics + lagged cumulative petition history
        raw_parts = []
        if self.skip_confounder_idx is not None:
            raw_parts.append(x_seq[:, :, self.skip_confounder_idx].reshape(B_seq, -1))
        if self.prop_petition_idx is not None:
            # Use lagged petition features (cumulative up to t-1) — causal, no leakage
            raw_parts.append(x_seq[:, :, self.prop_petition_idx].reshape(B_seq, -1))
        raw_confounders = torch.cat(raw_parts, dim=-1) if raw_parts else None

        tau = torch.rand(B_seq, device=device)

        # 4. Padding mask — exclude padded zero timesteps from all losses
        if L is not None:
            valid_t  = torch.arange(T - 4, device=device).unsqueeze(0)   # (1, T-4)
            pad_mask = (valid_t < (L.unsqueeze(1) - 4).clamp(min=0))     # (B, T-4)
            pad_flat = pad_mask.reshape(B_seq, 1).float()                 # (B_seq, 1)
        else:
            pad_flat = torch.ones(B_seq, 1, device=device)

        valid_denom = pad_flat.sum().clamp_min(1.0)

        # 3. Double Machine Learning: Residual Dose & Hurdle Encoding
        prop_logit, prop_mag = self.prop_net(
            z_flat.detach(),
            h_flat.detach(),
            raw_confounders.detach() if raw_confounders is not None else None
        )

        # Hurdle Propensity Loss — focal loss handles 99:1 class imbalance
        is_nonzero = (dose_flat > 0).float()
        loss_cls = (focal_bce_raw(prop_logit, is_nonzero) * pad_flat).sum() / valid_denom
        pos_reg = (is_nonzero.squeeze(-1) > 0) & (pad_flat.squeeze(-1) > 0)
        loss_reg = (
            F.mse_loss(prop_mag[pos_reg], dose_flat[pos_reg])
            if pos_reg.sum() > 0 else torch.tensor(0.0, device=device)
        )
        prop_loss  = loss_cls + loss_reg

        # Combined predicted expected dose
        prop_pred     = torch.sigmoid(prop_logit) * prop_mag
        dose_residual = dose_flat - prop_pred.detach()
        dose_enc      = torch.cat([is_nonzero, dose_residual], dim=-1)
        
        # 5. Joint CFM loss — EXCLUDE height dimension (handled by hurdle decoder below)
        y_joint_true = y_true_flat[:, self.non_height_idx]
        eps_joint = torch.randn_like(y_joint_true)
        y_tau = tau.unsqueeze(-1) * y_joint_true + (1 - tau.unsqueeze(-1)) * eps_joint

        v_pred = self.cfm(y_tau, tau, z_flat, h_flat, dose_enc)
        v_target = y_joint_true - eps_joint

        w = torch.where(dose_flat > 0, 6.0, 1.0) * pad_flat
        cfm_raw = F.l1_loss(v_pred, v_target, reduction="none").mean(dim=-1, keepdim=True)
        cfm_loss = (cfm_raw * w).sum() / valid_denom

        # 6. Hurdle Height Decoder
        height_true = y_true_flat[:, HEIGHT_IDX:HEIGHT_IDX+1]          # (B_seq, 1)
        gate_feat   = torch.cat([z_flat.detach(), h_flat.detach(), dose_enc.detach()], dim=-1)
        gate_logit  = self.height_gate(gate_feat)
        has_concession = (height_true > 0).float()                      # 1 if any concession

        # Gate loss: focal BCE on all valid (non-padded) timesteps
        gate_loss = (
            focal_bce_raw(gate_logit, has_concession, gamma=2.0, alpha=0.90)
            * pad_flat
        ).sum() / valid_denom

        # Conditional height CFM: train only on rows with positive concession
        pos_mask = (height_true.squeeze(-1) > 0) & (pad_flat.squeeze(-1) > 0)
        if pos_mask.sum() > 0:
            ht_true_pos = height_true[pos_mask]                         # (N+, 1)
            ht_tau      = tau[pos_mask]
            ht_eps      = torch.randn_like(ht_true_pos)
            ht_y_tau    = ht_tau.unsqueeze(-1) * ht_true_pos + (1 - ht_tau.unsqueeze(-1)) * ht_eps
            ht_v_pred   = self.height_cfm(ht_y_tau, ht_tau,
                                          z_flat[pos_mask], h_flat[pos_mask],
                                          dose_enc[pos_mask])
            ht_cfm_loss = F.l1_loss(ht_v_pred, ht_true_pos - ht_eps)
        else:
            ht_cfm_loss = torch.tensor(0.0, device=device)

        # Combine: cfm_loss covers [resolved, tokens, comm, council]; hurdle covers height
        cfm_loss = cfm_loss + gate_loss + ht_cfm_loss

        w_b     = w.view(B, T - 4).mean(dim=1, keepdim=True)
        kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1, keepdim=True) * w_b).mean()
        return cfm_loss, kl_loss, prop_loss

    # ── Inference: Euler ODE solve for counterfactual samples ────────────────
    @torch.no_grad()
    def sample(self, X_pre, X_t, dose_val=None, n_steps=CFM_STEPS):
        """
        Generate counterfactual trajectory under do(D = dose_val).
        X_pre: (B, 4, F)  pre-filing context
        X_t:   (B, T, F)  full sequence (treatment overridden externally)
        dose_val: float   the do-intervention value
        Returns: preds (B, T, Y)
        """
        B, T, _ = X_t.shape
        device  = X_t.device

        z, _, _ = self.encoder(X_pre)
        state   = self.transition.init_hidden(B, device)
        y_prev  = torch.zeros(B, self.y_dim, device=device)
        preds   = torch.zeros(B, T, self.y_dim, device=device)

        if dose_val is None:
            # Factual reconstruction: use each row's observed petition path.
            dose_seq = X_t[:, :, self.treat_idx[0]].clone()
        elif isinstance(dose_val, float) or isinstance(dose_val, int):
            dose_seq = torch.full((B, T), dose_val, device=device)
        else:
            dose_seq = dose_val.to(device)

        for t in range(4, T):
            x_t = X_t[:, t, :].clone()
            
            # TARGET LEAK FIX: Step the RNN with unbiased context
            x_t_unbiased = x_t.clone()
            for idx in self.mask_from_transition_idx:
                x_t_unbiased[:, idx] = 0.0

            h_t, state = self.transition.step(x_t_unbiased, y_prev, state)

            # Main Euler ODE: all targets EXCEPT height
            y_joint = torch.randn(B, len(self.non_height_idx), device=device)
            dt = 1.0 / n_steps
            for step in range(n_steps):
                tau = torch.full((B,), step / n_steps, device=device)

                # Encode counterfactual dose using Propensity network
                raw_cf_parts = []
                if self.skip_confounder_idx is not None:
                    raw_cf_parts.append(x_t_unbiased[:, self.skip_confounder_idx])
                if self.prop_petition_idx is not None:
                    raw_cf_parts.append(x_t_unbiased[:, self.prop_petition_idx])
                raw_cf_confounders = torch.cat(raw_cf_parts, dim=-1) if raw_cf_parts else None

                prop_logit, prop_mag = self.prop_net(z, h_t, raw_cf_confounders)
                prop_pred     = torch.sigmoid(prop_logit) * prop_mag
                
                # Use current timestep from sequence dose
                current_dose = dose_seq[:, t - 4].unsqueeze(1) if dose_seq.shape[-1] == (T - 4) else dose_seq[:, t].unsqueeze(1)
                
                dose_residual = current_dose - prop_pred
                dose_enc = torch.cat([(current_dose > 0).float(), dose_residual], dim=-1)

                v = self.cfm(y_joint, tau, z, h_t, dose_enc)
                y_joint = y_joint + v * dt

            y = torch.zeros(B, self.y_dim, device=device)
            y[:, self.non_height_idx] = y_joint

            # Hurdle height: gate × conditional CFM magnitude
            gate_feat  = torch.cat([z, h_t, dose_enc], dim=-1)
            gate_prob  = torch.sigmoid(self.height_gate(gate_feat))    # P(height > 0)

            ht_sample  = torch.randn(B, 1, device=device)
            for step in range(n_steps):
                tau_s = torch.full((B,), step / n_steps, device=device)
                ht_v  = self.height_cfm(ht_sample, tau_s, z, h_t, dose_enc)
                ht_sample = ht_sample + ht_v * dt

            # E[height] = P(gate=1) × predicted magnitude, clipped to [0, 1]
            y[:, HEIGHT_IDX:HEIGHT_IDX+1] = (gate_prob * ht_sample.clamp(0, 1))

            preds[:, t, :] = y

            # Write predicted cumulative states back into X_t for next step
            # (autoregressive: model sees its own predictions going forward)
            if t < T - 1:
                if self.f_cum_tok  is not None and self.t_idx_tok is not None: X_t[:, t+1, self.f_cum_tok]  = y[:, self.t_idx_tok]
                if self.f_cum_comm is not None and self.t_idx_comm is not None: X_t[:, t+1, self.f_cum_comm] = y[:, self.t_idx_comm]
                if self.f_cum_coun is not None and self.t_idx_coun is not None: X_t[:, t+1, self.f_cum_coun] = y[:, self.t_idx_coun]

            y_prev = y

        return preds


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='Fold index to train (0 to k_folds-1)')
    parser.add_argument('--k_folds', type=int, default=9, help='Number of cross-validation folds')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    return parser.parse_args()

def train():
    args = parse_args()
    X, Y, L, features, targets, norm_dict, treat_idx, cases, cell_assignments, filing_years = load_data()
    n_cases = len(X)
    input_dim = X.shape[-1]
    y_dim_actual = Y.shape[-1]

    print(f"Dataset: {n_cases} cases | input_dim={input_dim} | y_dim={y_dim_actual}", flush=True)

    # ── Spatio-Temporal Diagonal CV ──
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    
    cutoffs = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    if args.fold >= len(cutoffs):
        raise ValueError(f"Fold {args.fold} is out of bounds for the {len(cutoffs)} cutoffs.")
        
    cutoff_year = cutoffs[args.fold]
    
    gkf = GroupKFold(n_splits=args.k_folds)
    all_groups = cell_assignments.loc[cases, "cell_id"].values
    
    # We use GKF just to grab the spatial splits for this fold
    splits = list(gkf.split(cases, groups=all_groups))
    in_dist_case_idx, ood_case_idx = splits[args.fold]
    
    train_cells = cell_assignments.loc[cases[in_dist_case_idx], "cell_id"].unique()
    test_cells = cell_assignments.loc[cases[ood_case_idx], "cell_id"].unique()
    
    # Apply strict Temporal cutoffs
    # Train: Cases in the 40 cells filed <= cutoff_year
    train_mask = (filing_years.loc[cases] <= cutoff_year) & (cell_assignments.loc[cases, "cell_id"].isin(train_cells))
    train_cases_selected = filing_years.loc[cases][train_mask].index.values
    
    if len(train_cases_selected) < 50:
        print(f"  [SKIP] Fold {args.fold} (Cutoff {cutoff_year}): Not enough historical training data ({len(train_cases_selected)} < 50).", flush=True)
        return
        
    # Split the temporal 'Train' set further into a random Train/Val for early stopping
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_sub_idx, va_sub_idx = next(gss.split(train_cases_selected, groups=train_cases_selected))
    
    final_tr_cases = train_cases_selected[tr_sub_idx]
    final_va_cases = train_cases_selected[va_sub_idx]
    
    # Find the indices in the massive X tensor
    tr_idx = np.where(np.isin(cases, final_tr_cases))[0]
    va_idx = np.where(np.isin(cases, final_va_cases))[0]
    
    print(f"Executing Fold {args.fold+1}/{args.k_folds} (Cutoff {cutoff_year}) | Train: {len(tr_idx)} | Val: {len(va_idx)}", flush=True)

    train_ds = TensorDataset(X[tr_idx], Y[tr_idx], L[tr_idx])
    val_ds   = TensorDataset(X[va_idx], Y[va_idx], L[va_idx])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    f_cum_tok  = features.index(CUM_TOK_COL)  if CUM_TOK_COL  in features else None
    f_cum_comm = features.index(CUM_COMM_COL) if CUM_COMM_COL in features else None
    f_cum_coun = features.index(CUM_COUN_COL) if CUM_COUN_COL in features else None
    
    t_idx_tok  = targets.index(CUM_TOK_COL)  if CUM_TOK_COL  in targets else None
    t_idx_comm = targets.index(CUM_COMM_COL) if CUM_COMM_COL in targets else None
    t_idx_coun = targets.index(CUM_COUN_COL) if CUM_COUN_COL in targets else None

    # Static demographic confounders (skip into propensity directly)
    skip_cols = [
        "dist_petition_rate_lag1", "knn_petition_rate_1km",
        "race_white", "renter_share", "median_household_income",
        "mortgage_rate_30yr_momentum", "fed_funds_rate_momentum"
    ]
    skip_idx = [features.index(c) for c in skip_cols if c in features]

    # Lagged petition history — explicit shift(1).cumsum(), no current-period leakage
    petition_lag_cols = [
        "cumulative_petition_pct_lag1",
        "cumulative_petition_count_lag1",
        "cumulative_petition_events_lag1",
        "petition_velocity_3p_lag1",
    ]
    petition_lag_idx = [features.index(c) for c in petition_lag_cols if c in features]

    treatment_derived_cols = [
        "petition_pct_this_period",
        "cumulative_petition_pct",
        "cumulative_petition_count",
        "cumulative_petition_events",
        "petition_velocity_3p",
        "petition_intensity_per_ft",
    ]
    treatment_derived_idx = [features.index(c) for c in treatment_derived_cols if c in features]

    model = CausalSeq2SeqCFM(
        input_dim=input_dim, y_dim=y_dim_actual,
        hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
        cfm_hidden=CFM_HIDDEN, cfm_layers=CFM_LAYERS,
        n_layers=N_LAYERS, treat_idx=treat_idx,
        f_cum_tok=f_cum_tok, f_cum_comm=f_cum_comm, f_cum_coun=f_cum_coun,
        t_idx_tok=t_idx_tok, t_idx_comm=t_idx_comm, t_idx_coun=t_idx_coun,
        skip_confounder_idx=skip_idx,
        prop_petition_idx=petition_lag_idx,
        treatment_derived_idx=treatment_derived_idx,
        # prop_z_dim defaults to latent_dim (64) — z encodes full petition trajectory
    ).to(device)

    print("cumulative_petition_pct_lag1 in features:",
          "cumulative_petition_pct_lag1" in features, flush=True)
    print("petition features:", [f for f in features if "petition" in f], flush=True)
    print("treat_idx:", treat_idx, "prop_petition_idx:", petition_lag_idx,
          "masked_treatment_idx:", treatment_derived_idx, flush=True)


    if args.compile:
        print("Compiling model via torch.compile...", flush=True)
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    best_cfm_loss = float("inf")
    best_height_score = float("inf")
    patience = 10
    epochs_without_improvement = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_cfm, tr_kl = 0., 0.
        for Xb, Yb, Lb in train_dl:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            
            with torch.autocast("cuda", dtype=torch.float16):
                cfm_loss, kl_loss, prop_loss = model.forward_train(Xb, Yb, Lb.to(device))
                loss = cfm_loss + BETA * kl_loss + prop_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            tr_cfm += cfm_loss.item()
            tr_kl  += kl_loss.item()
        scheduler.step()
        tr_cfm /= len(train_dl)
        tr_kl  /= len(train_dl)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        va_cfm, va_kl, va_prop = 0., 0., 0.
        all_surv_true, all_surv_pred = [], []
        all_ht_true,   all_ht_pred   = [], []

        with torch.no_grad():
            for Xb, Yb, Lb in val_dl:
                Xb, Yb = Xb.to(device), Yb.to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    cfm_loss, kl_loss, prop_loss = model.forward_train(Xb, Yb, Lb.to(device))
                va_cfm += cfm_loss.item()
                va_kl  += kl_loss.item()
                va_prop += prop_loss.item()

                # Sample predictions for metrics (use factual observed dose sequence)
                dose_obs = Xb[:, 4:, treat_idx[0]]
                preds = model.sample(Xb[:, :4, :], Xb, dose_obs)
                
                # Mask out padding tokens based on Lb (sequence length)
                B, T, _ = Yb.shape
                valid_t = torch.arange(T - 4, device=device).unsqueeze(0)
                pad_mask = (valid_t < (Lb.to(device).unsqueeze(1) - 4).clamp(min=0)).reshape(-1).cpu().numpy()
                valid_rows = pad_mask > 0

                # [0] resolved: sigmoid
                surv_pred = torch.sigmoid(preds[:, 4:, 0]).cpu().numpy().flatten()[valid_rows]
                surv_true = Yb[:, 4:, 0].cpu().numpy().flatten()[valid_rows]
                ht_pred   = preds[:, 4:, 1].cpu().numpy().flatten()[valid_rows]
                ht_true   = Yb[:, 4:, 1].cpu().numpy().flatten()[valid_rows]
                
                all_surv_true.append(surv_true)
                all_surv_pred.append(surv_pred)
                all_ht_true.append(ht_true)
                all_ht_pred.append(ht_pred)

        va_cfm /= len(val_dl)
        va_kl  /= len(val_dl)
        va_prop /= len(val_dl)

        surv_true_np = np.concatenate(all_surv_true)
        surv_pred_np = np.concatenate(all_surv_pred)
        ht_true_np   = np.concatenate(all_ht_true)
        ht_pred_np   = np.concatenate(all_ht_pred)

        try:
            auc = roc_auc_score(surv_true_np.round(), (surv_pred_np > 0.5).astype(int))
        except Exception:
            auc = float("nan")
            
        pos_mask = ht_true_np > 0
        zero_mask = ~pos_mask
        
        ht_mae_pos = np.abs(ht_true_np[pos_mask] - ht_pred_np[pos_mask]).mean() if pos_mask.sum() > 0 else 0.0
        ht_mae_0 = np.abs(ht_true_np[zero_mask] - ht_pred_np[zero_mask]).mean() if zero_mask.sum() > 0 else 0.0
        elapsed = (time.time() - t0) / 60

        print(
            f"[Epoch {epoch:03d}/{EPOCHS}] "
            f"TR cfm={tr_cfm:.4f} kl={tr_kl:.4f} | "
            f"VA cfm={va_cfm:.4f} kl={va_kl:.4f} prop={va_prop:.4f} | "
            f"AUC={auc:.3f} MAE_Pos={ht_mae_pos:.3f} MAE_0={ht_mae_0:.3f} | "
            f"{elapsed:.1f}min elapsed",
            flush=True
        )

        # Save best
        score_cfm = va_cfm + BETA * va_kl
        score_height = va_cfm + 0.5 * ht_mae_pos + 0.2 * ht_mae_0

        if score_cfm < best_cfm_loss:
            best_cfm_loss = score_cfm
            ckpt = os.path.join(OUT_DIR, f"causal_cfm_weights_fold_{args.fold}_best_cfm.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Saved best-CFM model (score={best_cfm_loss:.4f})", flush=True)

        if score_height < best_height_score:
            best_height_score = score_height
            epochs_without_improvement = 0
            ckpt = os.path.join(OUT_DIR, f"causal_cfm_weights_fold_{args.fold}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Saved best-height model (score={best_height_score:.4f})", flush=True)

            # ── Feature Manifest ─────────────────────────────────────────────────
            # Write alongside checkpoint so local eval can reconstruct the exact
            # biweekly panel without reverse-engineering weight shapes.
            import json as _json
            manifest = {
                "fold": args.fold,
                "input_dim": int(X.shape[2]),
                "y_dim": int(Y.shape[2]),
                "feature_names": features,
                "target_names": targets,
                "treat_idx": treat_idx if isinstance(treat_idx, list) else [int(treat_idx)],
                "skip_confounder_idx": model.skip_confounder_idx if hasattr(model, 'skip_confounder_idx') else None,
                "prop_petition_idx": model.prop_petition_idx,
                "treatment_derived_idx": model.treatment_derived_idx,
                "non_height_idx": model.non_height_idx,
                "height_idx": HEIGHT_IDX,
                "zscore_exclude": sorted(list(ZSCORE_EXCLUDE)),
                "norm_dict": norm_dict,
                "hidden_dim": 256,
                "latent_dim": 64,
                "cfm_hidden": 512,
                "cfm_layers": 5,
                "n_layers": 3,
                "T_MAX": 55,
                "panel_path": PANEL_PATH,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            manifest_path = os.path.join(OUT_DIR, f"feature_manifest_fold_{args.fold}.json")
            with open(manifest_path, "w") as _mf:
                _json.dump(manifest, _mf, indent=2)
            print(f"  ✓ Saved feature manifest → {manifest_path}", flush=True)
            if S3_BUCKET:
                os.system(f"aws s3 cp {manifest_path} {S3_BUCKET}/output/feature_manifest_fold_{args.fold}.json > /dev/null 2>&1")

            # Upload to S3 every time we improve
            if S3_BUCKET:
                os.system(f"aws s3 cp {ckpt} {S3_BUCKET}/output/causal_cfm_weights_fold_{args.fold}.pt > /dev/null 2>&1")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).", flush=True)
                break

    print("\n=== Training complete ===", flush=True)
    print(f"Best CFM score: {best_cfm_loss:.4f} | Best height score: {best_height_score:.4f}", flush=True)


if __name__ == "__main__":
    train()
