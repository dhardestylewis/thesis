"""
causal_cfm_cvae.py -- Zero-Inflated Causal Hurdle CFM

Architecture:
  - LSTM encoder over pre-intervention periods (VAE)
  - LSTM transition network for autoregressive dynamics
  - DML propensity network for dose residualization
  - Resolved outcome: BCE head
  - Height concession pct: focal gate + positive-only 1D CFM
  - Remaining continuous targets: joint CFM

Fixes over the previous causal_seq2seq_cvae.py:
  1. L masks prevent padded zeros from entering any loss
  2. height_concession_pct is NOT z-scored; treated as zero-inflated bounded [0,1]
  3. Height is split into event gate + positive magnitude -- zeros never enter the flow
  4. Focal / positive-weighted BCE for the height gate
  5. Petition dose left in raw [0,1] units; only structural features z-scored
  6. G-computation uses expected-value output, not sampled Bernoulli / median

Usage:
  python causal_cfm_cvae.py --fold 0
"""

import argparse
import gc
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Environment ──────────────────────────────────────────────────────────────
OUT_DIR = os.environ.get("OUT_DIR", ".")
PANEL_PATH = os.environ.get(
    "PANEL_PATH",
    os.path.join(OUT_DIR, "biweekly_panel.csv"),
)

# ── Constants ────────────────────────────────────────────────────────────────
T_MAX = 55
PRE_PERIODS = 4
CFM_STEPS = 20
HEIGHT_EPS = 1e-4
HEIGHT_TARGET = "height_concession_pct"

LATENT_DIM = 48
HIDDEN_DIM = 256
CFM_HIDDEN = 256
CFM_LAYERS = 3
N_ENC_LAYERS = 2
N_TRANS_LAYERS = 2
DOSE_DIM = 2

BATCH_SIZE = 64
EPOCHS = 60
LR = 3e-4
KL_BETA_MAX = 1e-3
PATIENCE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helper functions ─────────────────────────────────────────────────────────

def logit_clip(y: torch.Tensor, eps: float = HEIGHT_EPS) -> torch.Tensor:
    y = y.clamp(eps, 1.0 - eps)
    return torch.log(y) - torch.log1p(-y)


def inv_logit(u: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(u).clamp(0.0, 1.0)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: "torch.Tensor | None" = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    p = torch.sigmoid(logits)
    pt = p * targets + (1.0 - p) * (1.0 - targets)
    focal = (1.0 - pt).clamp_min(1e-4).pow(gamma)
    return focal * bce


# ── Network Components ────────────────────────────────────────────────────────

class VelocityField(nn.Module):
    """Continuous-Flow-Matching velocity network."""

    def __init__(
        self,
        y_dim: int,
        z_dim: int,
        h_dim: int,
        hidden: int = CFM_HIDDEN,
        n_layers: int = CFM_LAYERS,
        dose_dim: int = DOSE_DIM,
    ):
        super().__init__()
        self.tau_embed = nn.Sequential(nn.Linear(1, 16), nn.SiLU())
        inp = y_dim + 16 + z_dim + h_dim + dose_dim
        layers: list[nn.Module] = []
        for i in range(n_layers):
            layers += [nn.Linear(inp if i == 0 else hidden, hidden), nn.SiLU(), nn.Dropout(0.1)]
        layers.append(nn.Linear(hidden, y_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y_tau, tau, z, h, dose_enc):
        tau_emb = self.tau_embed(tau.unsqueeze(-1))
        return self.net(torch.cat([y_tau, tau_emb, z, h, dose_enc], dim=-1))


class ContextHead(nn.Module):
    """MLP head conditioned on (z, h, dose_enc) for binary outcomes."""

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        dose_dim: int = DOSE_DIM,
        hidden: int = 256,
        out_dim: int = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + h_dim + dose_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z, h, dose_enc):
        return self.net(torch.cat([z, h, dose_enc], dim=-1))


class EncoderLSTM(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int, latent_dim: int, n_layers: int = 2):
        super().__init__()
        drop = 0.1 if n_layers > 1 else 0.0
        self.rnn = nn.LSTM(x_dim, hidden_dim, n_layers, batch_first=True, dropout=drop)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_pre):
        _, (h, _) = self.rnn(x_pre)
        h_last = h[-1]
        mu = self.mu(h_last)
        lv = self.logvar(h_last).clamp(-10.0, 4.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return z, mu, lv


class TransitionRNN(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        drop = 0.1 if n_layers > 1 else 0.0
        self.rnn = nn.LSTM(x_dim + y_dim, hidden_dim, n_layers, batch_first=True, dropout=drop)

    def init_hidden(self, B: int, dev):
        z = torch.zeros
        return (z(self.n_layers, B, self.hidden_dim, device=dev),
                z(self.n_layers, B, self.hidden_dim, device=dev))

    def step(self, x_t, y_prev, state):
        inp = torch.cat([x_t, y_prev], dim=-1).unsqueeze(1)
        out, new_state = self.rnn(inp, state)
        return out.squeeze(1), new_state


class PropensityNet(nn.Module):
    """
    Estimates P(dose > 0 | confounders) and E[dose | dose > 0, confounders].
    Used for DML partial residualization of the treatment signal.
    """

    def __init__(self, z_dim: int, h_dim: int, confounder_dim: int = 0, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(z_dim + h_dim + confounder_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.cls_head = nn.Linear(hidden, 1)
        self.mag_head = nn.Linear(hidden, 1)

    def forward(self, z, h, confounders=None):
        parts = [z, h]
        if confounders is not None:
            parts.append(confounders)
        shared = self.shared(torch.cat(parts, dim=-1))
        return self.cls_head(shared), torch.sigmoid(self.mag_head(shared))


# ── Main Model ────────────────────────────────────────────────────────────────

class CausalSeq2SeqCFM(nn.Module):
    """
    Zero-Inflated Causal Hurdle CFM.

    Target layout (y_dim columns):
      idx 0 : resolved              -> BCE head
      idx 1 : height_concession_pct -> focal gate + positive CFM
      idx 2+ : cumulative continuous targets -> joint CFM
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        treat_idx: list,
        skip_confounder_idx: "list | None" = None,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        cfm_hidden: int = CFM_HIDDEN,
        cfm_layers: int = CFM_LAYERS,
        resolved_idx: int = 0,
        height_idx: int = 1,
        # Autoregressive state update slots (feature_idx, target_idx pairs)
        f_cum_tok: "int | None" = None,
        t_idx_tok: "int | None" = None,
        f_cum_comm: "int | None" = None,
        t_idx_comm: "int | None" = None,
        f_cum_coun: "int | None" = None,
        t_idx_coun: "int | None" = None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.treat_idx = treat_idx
        self.skip_confounder_idx = skip_confounder_idx
        self.resolved_idx = resolved_idx
        self.height_idx = height_idx
        self.cont_idx = [i for i in range(y_dim) if i not in (resolved_idx, height_idx)]

        self.f_cum_tok = f_cum_tok
        self.t_idx_tok = t_idx_tok
        self.f_cum_comm = f_cum_comm
        self.t_idx_comm = t_idx_comm
        self.f_cum_coun = f_cum_coun
        self.t_idx_coun = t_idx_coun

        self.encoder = EncoderLSTM(x_dim, hidden_dim, latent_dim, N_ENC_LAYERS)
        self.transition = TransitionRNN(x_dim, y_dim, hidden_dim, N_TRANS_LAYERS)

        cfd = len(skip_confounder_idx) if skip_confounder_idx else 0
        self.prop_net = PropensityNet(latent_dim, hidden_dim, cfd)

        self.resolved_head = ContextHead(latent_dim, hidden_dim, out_dim=1)
        self.height_gate = ContextHead(latent_dim, hidden_dim, out_dim=1)

        self.height_pos_cfm = VelocityField(
            y_dim=1, z_dim=latent_dim, h_dim=hidden_dim,
            hidden=cfm_hidden, n_layers=cfm_layers,
        )
        self.cont_cfm = VelocityField(
            y_dim=len(self.cont_idx), z_dim=latent_dim, h_dim=hidden_dim,
            hidden=cfm_hidden, n_layers=cfm_layers,
        )

        # Filled from norm_dict after model construction
        self.register_buffer("height_u_mean", torch.tensor(0.0))
        self.register_buffer("height_u_std", torch.tensor(1.0))
        self.register_buffer("height_pos_weight", torch.tensor(10.0))

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward_train(self, X, Y, L=None):
        """
        X : (B, T, F)
        Y : (B, T, Y_DIM)
        L : (B,) true sequence lengths -- required to mask padded zeros
        """
        B, T, _ = X.shape
        dev = X.device
        n_pred = T - PRE_PERIODS

        z, mu, logvar = self.encoder(X[:, :PRE_PERIODS, :])

        x_seq = X[:, PRE_PERIODS:, :]          # (B, n_pred, F)
        y_true_seq = Y[:, PRE_PERIODS:, :]     # (B, n_pred, Y)
        y_prev_seq = torch.cat(
            [torch.zeros(B, 1, self.y_dim, device=dev), Y[:, PRE_PERIODS:-1, :]],
            dim=1,
        )

        # Zero out treatment in transition context (no treatment leakage)
        x_seq_unb = x_seq.clone()
        for idx in self.treat_idx:
            x_seq_unb[:, :, idx] = 0.0

        state = self.transition.init_hidden(B, dev)
        inp_seq = torch.cat([x_seq_unb, y_prev_seq], dim=-1)
        out_seq, _ = self.transition.rnn(inp_seq, state)   # (B, n_pred, H)

        # Flatten across batch × time
        B_seq = B * n_pred
        y_true_flat = y_true_seq.reshape(B_seq, self.y_dim)
        dose_flat = x_seq[:, :, self.treat_idx[0]].unsqueeze(-1).reshape(B_seq, 1)
        z_flat = z.unsqueeze(1).repeat(1, n_pred, 1).reshape(B_seq, -1)
        h_flat = out_seq.reshape(B_seq, self.hidden_dim)

        # Pad mask: do not supervise on periods beyond true sequence length
        if L is None:
            valid = torch.ones(B, n_pred, device=dev, dtype=torch.bool)
        else:
            steps = torch.arange(PRE_PERIODS, T, device=dev).unsqueeze(0)
            valid = steps < L.to(dev).unsqueeze(1)      # (B, n_pred)
        valid_flat = valid.reshape(B_seq, 1).float()
        valid_denom = valid_flat.sum().clamp_min(1.0)

        # Confounders for propensity net
        raw_conf = None
        if self.skip_confounder_idx is not None:
            raw_conf = x_seq_unb[:, :, self.skip_confounder_idx].reshape(B_seq, -1)

        # ── Propensity (DML) ──────────────────────────────────────────────────
        prop_logit, prop_mag = self.prop_net(
            z_flat.detach(),
            h_flat.detach(),
            raw_conf.detach() if raw_conf is not None else None,
        )

        dose_nz = (dose_flat > 0).float()

        prop_cls_raw = F.binary_cross_entropy_with_logits(
            prop_logit, dose_nz, reduction="none"
        )
        loss_prop_cls = (prop_cls_raw * valid_flat).sum() / valid_denom

        dose_pos_mask = ((dose_nz > 0) & (valid_flat > 0)).squeeze(-1)
        if dose_pos_mask.any():
            loss_prop_reg = F.mse_loss(prop_mag[dose_pos_mask], dose_flat[dose_pos_mask])
        else:
            loss_prop_reg = prop_mag.new_tensor(0.0)

        prop_loss = loss_prop_cls + loss_prop_reg

        prop_pred = torch.sigmoid(prop_logit) * prop_mag
        dose_residual = dose_flat - prop_pred.detach()
        dose_enc = torch.cat([dose_nz, dose_residual], dim=-1)   # (B_seq, 2)

        # ── 1. Resolved: BCE ──────────────────────────────────────────────────
        resolved_true = y_true_flat[:, [self.resolved_idx]].clamp(0.0, 1.0)
        resolved_logit = self.resolved_head(z_flat, h_flat, dose_enc)
        res_raw = F.binary_cross_entropy_with_logits(
            resolved_logit, resolved_true, reduction="none"
        )
        resolved_loss = (res_raw * valid_flat).sum() / valid_denom

        # ── 2. Height gate: focal BCE ─────────────────────────────────────────
        height_true = y_true_flat[:, [self.height_idx]].clamp(0.0, 1.0)
        height_pos_label = (height_true > HEIGHT_EPS).float()
        gate_logit = self.height_gate(z_flat, h_flat, dose_enc)
        gate_raw = focal_bce_with_logits(
            gate_logit, height_pos_label,
            pos_weight=self.height_pos_weight.view(1),
            gamma=2.0,
        )
        height_gate_loss = (gate_raw * valid_flat).sum() / valid_denom

        # ── 3. Height positive magnitude: CFM on logit scale ─────────────────
        height_pos_mask = (
            (height_true[:, 0] > HEIGHT_EPS) & (valid_flat[:, 0] > 0)
        )

        if height_pos_mask.any():
            hpos = height_true[height_pos_mask].clamp(HEIGHT_EPS, 1.0 - HEIGHT_EPS)
            u_true = (logit_clip(hpos) - self.height_u_mean) / self.height_u_std.clamp_min(1e-6)

            tau_h = torch.rand(u_true.shape[0], device=dev)
            eps_h = torch.randn_like(u_true)
            u_tau = tau_h.unsqueeze(-1) * u_true + (1.0 - tau_h.unsqueeze(-1)) * eps_h

            v_pred_h = self.height_pos_cfm(
                u_tau, tau_h,
                z_flat[height_pos_mask],
                h_flat[height_pos_mask],
                dose_enc[height_pos_mask],
            )
            v_target_h = u_true - eps_h

            # Up-weight petition-period rows without contaminating zero-concession rows
            w_h = 1.0 + 4.0 * dose_nz[height_pos_mask]
            height_flow_loss = (
                F.smooth_l1_loss(v_pred_h, v_target_h, reduction="none") * w_h
            ).mean()
        else:
            height_flow_loss = y_true_flat.new_tensor(0.0)

        # ── 4. Continuous cumulative targets: joint CFM ───────────────────────
        cont_true = y_true_flat[:, self.cont_idx]
        tau_c = torch.rand(B_seq, device=dev)
        eps_c = torch.randn_like(cont_true)
        cont_tau = tau_c.unsqueeze(-1) * cont_true + (1.0 - tau_c.unsqueeze(-1)) * eps_c

        v_pred_c = self.cont_cfm(cont_tau, tau_c, z_flat, h_flat, dose_enc)
        v_target_c = cont_true - eps_c
        cont_raw = F.smooth_l1_loss(v_pred_c, v_target_c, reduction="none").mean(
            dim=-1, keepdim=True
        )
        cont_loss = (cont_raw * valid_flat).sum() / valid_denom

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        decoder_loss = (
            cont_loss
            + 0.5 * resolved_loss
            + 3.0 * height_gate_loss
            + 1.5 * height_flow_loss
        )
        return decoder_loss, kl_loss, prop_loss

    # ── Counterfactual sampling ───────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        X_pre,
        X_t,
        dose_val=None,
        n_steps: int = CFM_STEPS,
        height_mc_samples: int = 8,
        sample_height_atom: bool = False,
    ):
        """
        Generate counterfactual trajectory.

        dose_val : if None, reads dose from X_t[:, t, treat_idx[0]] at each t.
                   Pass a scalar to impose a fixed do(d) policy.
        Returns preds : (B, T, Y_DIM)
          - resolved    : raw logit (apply sigmoid to get probability)
          - height      : expected concession % in [0, 1]  (P(>0) * E[mag | >0])
          - cont targets: ODE-integrated samples
        """
        B, T, _ = X_t.shape
        dev = X_t.device

        z, _, _ = self.encoder(X_pre)
        state = self.transition.init_hidden(B, dev)

        y_prev = torch.zeros(B, self.y_dim, device=dev)
        preds = torch.zeros(B, T, self.y_dim, device=dev)
        dt = 1.0 / n_steps

        for t in range(PRE_PERIODS, T):
            x_t = X_t[:, t, :].clone()

            if dose_val is None:
                dose_t = x_t[:, self.treat_idx[0]].unsqueeze(-1)
            else:
                dose_t = torch.full((B, 1), float(dose_val), device=dev)

            x_t_unb = x_t.clone()
            for idx in self.treat_idx:
                x_t_unb[:, idx] = 0.0

            h_t, state = self.transition.step(x_t_unb, y_prev, state)

            raw_cf = None
            if self.skip_confounder_idx is not None:
                raw_cf = x_t_unb[:, self.skip_confounder_idx]

            prop_logit, prop_mag = self.prop_net(z, h_t, raw_cf)
            dose_residual = dose_t - (torch.sigmoid(prop_logit) * prop_mag)
            dose_enc = torch.cat([(dose_t > 0).float(), dose_residual], dim=-1)

            y_t = torch.zeros(B, self.y_dim, device=dev)

            # Resolved
            y_t[:, self.resolved_idx] = self.resolved_head(z, h_t, dose_enc).squeeze(-1)

            # Continuous targets via ODE
            cont_y = torch.randn(B, len(self.cont_idx), device=dev)
            for step in range(n_steps):
                tau = torch.full((B,), step / n_steps, device=dev)
                v = self.cont_cfm(cont_y, tau, z, h_t, dose_enc)
                cont_y = cont_y + v * dt
            y_t[:, self.cont_idx] = cont_y

            # Height hurdle: E[Y] = P(Y>0) * E[Y | Y>0]
            p_pos = torch.sigmoid(self.height_gate(z, h_t, dose_enc))  # (B, 1)

            z_rep = z.repeat_interleave(height_mc_samples, dim=0)
            h_rep = h_t.repeat_interleave(height_mc_samples, dim=0)
            d_rep = dose_enc.repeat_interleave(height_mc_samples, dim=0)

            u = torch.randn(B * height_mc_samples, 1, device=dev)
            for step in range(n_steps):
                tau = torch.full((B * height_mc_samples,), step / n_steps, device=dev)
                v = self.height_pos_cfm(u, tau, z_rep, h_rep, d_rep)
                u = u + v * dt

            u_raw = u.view(B, height_mc_samples, 1) * self.height_u_std + self.height_u_mean
            height_pos_mag = inv_logit(u_raw).mean(dim=1)   # (B, 1)

            if sample_height_atom:
                height_out = torch.bernoulli(p_pos) * height_pos_mag
            else:
                height_out = p_pos * height_pos_mag           # expected value

            y_t[:, self.height_idx] = height_out.squeeze(-1)
            preds[:, t, :] = y_t

            # Autoregressive state feedback
            if t < T - 1:
                if self.f_cum_tok is not None and self.t_idx_tok is not None:
                    X_t[:, t + 1, self.f_cum_tok] = y_t[:, self.t_idx_tok]
                if self.f_cum_comm is not None and self.t_idx_comm is not None:
                    X_t[:, t + 1, self.f_cum_comm] = y_t[:, self.t_idx_comm]
                if self.f_cum_coun is not None and self.t_idx_coun is not None:
                    X_t[:, t + 1, self.f_cum_coun] = y_t[:, self.t_idx_coun]

            y_prev = y_t

        return preds


# ── Data loading ──────────────────────────────────────────────────────────────

# Features excluded from the model regardless
_ALWAYS_EXCLUDE = {
    "case_number", "period_start", "period_start_dt", "year", "quarter",
    "petition_year", "petition_quarter", "latitude", "longitude",
    "shape_area", "council_district", "census_tract", "land_use_code",
    "label_petition_total_pct", "label_valid_protest",
    "label_real_days_in_pipeline", "label_valid_petition_pct",
    "label_exact_geometric_petition_pct",
    "vote_event", "vote_friction", "yea_this_year", "nay_this_year",
    "censored",
    "cumulative_council_hearings", "cumulative_commission_hearings",
    "Aggregate_Sentiment", "max_opponent_experience",
    # Prevent height leakage from informative missingness
    "pdf_final_approved_ht", "approved_height_ft",
}

# Features that must NOT be z-scored (kept in native units)
_NO_ZSCORE = {
    "resolved",
    "petition_pct_this_period",
    "cumulative_petition_pct",
    HEIGHT_TARGET,
    # Binary flags
    "commission_hearings_this_period",
    "council_hearings_this_period",
    "petition_event",
}

# Right-skewed features to log-transform before z-scoring
_LOG1P = {"land_acres", "market_value", "appraised_value",
          "council_nlp_total_tokens", "cumulative_council_nlp_lag1"}

AVAILABLE_TARGETS = [
    "resolved",
    HEIGHT_TARGET,
    "cumulative_vote_friction",
    "council_nlp_total_tokens",
    "commission_hearings_this_period",
    "council_hearings_this_period",
]


def _engineer_height_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Compute height_concession_pct in [0, 1]."""
    if "pdf_requested_height_ft" not in df.columns:
        df[HEIGHT_TARGET] = 0.0
        return df

    initial_req = (
        df.groupby("case_number")["pdf_requested_height_ft"]
        .transform("max")
        .clip(lower=1.0)
    )

    if "pdf_staff_recommends_ht" in df.columns:
        current_constraint = df[["pdf_requested_height_ft", "pdf_staff_recommends_ht"]].min(axis=1)
    else:
        current_constraint = df["pdf_requested_height_ft"]

    current_constraint = current_constraint.fillna(initial_req)

    if "pdf_reduced_to_ft" in df.columns:
        final_ht = df["pdf_reduced_to_ft"].fillna(current_constraint)
    else:
        final_ht = current_constraint

    pct = ((initial_req - final_ht) / initial_req).clip(0.0, 1.0).fillna(0.0)
    df[HEIGHT_TARGET] = pct.astype(np.float32)
    return df


def load_data():
    """
    Load the biweekly panel, engineer targets, normalize features.

    Returns
    -------
    df            : pd.DataFrame with all engineered + normalized columns
    features      : list[str]  feature column names
    targets       : list[str]  target column names (AVAILABLE_TARGETS order)
    norm_dict     : dict       normalization stats + height logit stats
    """
    if os.path.exists("/home/ubuntu/biweekly_panel.csv"):
        df = pd.read_csv("/home/ubuntu/biweekly_panel.csv", low_memory=False)
    else:
        df = pd.read_csv(PANEL_PATH, low_memory=False)

    df["period_start_dt"] = pd.to_datetime(df["period_start"], errors="coerce")
    df = df.sort_values(["case_number", "period_seq"]).reset_index(drop=True)

    # ── Vote features ────────────────────────────────────────────────────────
    for col in ["cumulative_yea_votes", "cumulative_nay_votes", "net_vote_margin",
                "yea_this_year", "nay_this_year"]:
        if col not in df.columns:
            df[col] = 0.0

    df["vote_friction"] = df.get("vote_event", pd.Series(0, index=df.index)) * (
        1 + df["cumulative_nay_votes"].clip(upper=10)
    )
    df["cumulative_vote_friction"] = df.groupby("case_number")["vote_friction"].cumsum()

    # ── NLP token leakage guard ───────────────────────────────────────────────
    if "council_hearings_this_period" in df.columns and "council_nlp_total_tokens" in df.columns:
        df["council_nlp_total_tokens"] = df["council_nlp_total_tokens"].where(
            df["council_hearings_this_period"] > 0, other=0
        )

    # ── Lagged cumulative features (no leakage) ───────────────────────────────
    for src, dst in [
        ("council_hearings_this_period", "cumulative_council_hearings_lag1"),
        ("commission_hearings_this_period", "cumulative_commission_hearings_lag1"),
        ("council_nlp_total_tokens", "cumulative_council_nlp_lag1"),
    ]:
        if src in df.columns:
            df[dst] = (
                df.groupby("case_number")[src]
                .apply(lambda x: x.shift(1).fillna(0).cumsum())
                .reset_index(level=0, drop=True)
            )
        else:
            df[dst] = 0.0

    # ── Petition percentages ─────────────────────────────────────────────────
    if "petition_pct_this_period" in df.columns:
        df["petition_pct_this_period"] = df["petition_pct_this_period"].clip(upper=100.0)
        df["cumulative_petition_pct"] = (
            df.groupby("case_number")["petition_pct_this_period"]
            .cumsum()
            .clip(upper=100.0)
        )
    else:
        df["petition_pct_this_period"] = 0.0
        df["cumulative_petition_pct"] = 0.0

    # ── Height concession target ──────────────────────────────────────────────
    df = _engineer_height_pct(df)

    # ── Outlier cleaning ──────────────────────────────────────────────────────
    if "building_age" in df.columns:
        valid_age = df.loc[df["building_age"] <= 250, "building_age"]
        df.loc[df["building_age"] > 250, "building_age"] = valid_age.mean()
    if "yr_built" in df.columns:
        good = df["yr_built"] >= 1850
        df.loc[(df["yr_built"] < 1850) & (df["yr_built"] > 0), "yr_built"] = df.loc[good, "yr_built"].mean()

    # ── Ensure all targets exist ──────────────────────────────────────────────
    for t in AVAILABLE_TARGETS:
        if t not in df.columns:
            df[t] = 0.0
        df[t] = pd.to_numeric(df[t], errors="coerce").fillna(0.0)

    # ── Hearing event columns: clip to binary ─────────────────────────────────
    for col in ["commission_hearings_this_period", "council_hearings_this_period"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)

    # ── Log-transform right-skewed columns before z-scoring ──────────────────
    for col in _LOG1P:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # ── Determine feature list ────────────────────────────────────────────────
    exclude = _ALWAYS_EXCLUDE | set(AVAILABLE_TARGETS)
    features = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Ensure all feature columns are numeric
    for f in features:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0).astype(np.float32)

    # ── Normalization ─────────────────────────────────────────────────────────
    # Skip: petition percentages (raw dose), height target (handled separately),
    # resolved (binary label), and binary hearing flags.
    norm_dict: dict = {}
    for col in features:
        if col in _NO_ZSCORE:
            continue
        vals = df[col].values.astype(np.float32)
        m, s = float(vals.mean()), float(vals.std())
        s = max(s, 1e-8)
        df[col] = ((vals - m) / s).astype(np.float32)
        norm_dict[col] = (m, s)

    # ── Height logit stats (for the positive-magnitude CFM) ───────────────────
    height_raw = df[HEIGHT_TARGET].clip(0.0, 1.0).values.astype(np.float32)
    height_pos = height_raw[height_raw > HEIGHT_EPS]

    if len(height_pos) > 0:
        hp_clipped = np.clip(height_pos, HEIGHT_EPS, 1.0 - HEIGHT_EPS)
        u_vals = np.log(hp_clipped) - np.log1p(-hp_clipped)
        norm_dict["_height_pos_logit_mean"] = float(u_vals.mean())
        norm_dict["_height_pos_logit_std"] = float(max(u_vals.std(), 1e-6))
        pos_rate = float((height_raw > HEIGHT_EPS).mean())
        norm_dict["_height_pos_rate"] = pos_rate
        norm_dict["_height_pos_weight"] = float(min((1.0 - pos_rate) / max(pos_rate, 1e-6), 50.0))
    else:
        norm_dict["_height_pos_logit_mean"] = 0.0
        norm_dict["_height_pos_logit_std"] = 1.0
        norm_dict["_height_pos_rate"] = 0.0
        norm_dict["_height_pos_weight"] = 1.0

    print(
        f"  [load_data] {len(df):,} rows | {df['case_number'].nunique():,} cases | "
        f"{len(features)} features | y_dim={len(AVAILABLE_TARGETS)}",
        flush=True,
    )
    print(
        f"  [load_data] height pos rate={norm_dict['_height_pos_rate']:.3f} | "
        f"pos_weight={norm_dict['_height_pos_weight']:.1f}",
        flush=True,
    )

    return df, features, AVAILABLE_TARGETS, norm_dict


def build_tensors(df, features, targets, cases, max_seq=T_MAX):
    """Build padded (B, T, F) / (B, T, Y) tensors and true length vector L."""
    sub = df[df["case_number"].isin(cases)].sort_values(["case_number", "period_seq"])
    case_sizes = sub.groupby("case_number").size()
    c_list = case_sizes.index.values

    n = len(c_list)
    X_out = np.zeros((n, max_seq, len(features)), dtype=np.float32)
    Y_out = np.zeros((n, max_seq, len(targets)), dtype=np.float32)
    L_out = np.zeros(n, dtype=np.int64)

    feat_arr = sub[features].values.astype(np.float32)
    targ_arr = sub[targets].fillna(0.0).values.astype(np.float32)

    idx = 0
    for i, c in enumerate(c_list):
        size = case_sizes[c]
        length = min(size, max_seq)
        X_out[i, :length, :] = feat_arr[idx: idx + length]
        Y_out[i, :length, :] = targ_arr[idx: idx + length]
        L_out[i] = length
        idx += size

    return (
        torch.from_numpy(X_out),
        torch.from_numpy(Y_out),
        torch.from_numpy(L_out),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(fold: int, df, features, targets, norm_dict):
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    print(f"\n{'='*60}", flush=True)
    print(f"FOLD {fold}", flush=True)

    first_dt = df.groupby("case_number")["period_start_dt"].min()

    # Spatial micro-cell clustering for group-aware CV
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler as SKScaler

    first_periods = df.groupby("case_number").first()
    clust_feats = first_periods[["latitude", "longitude"]].copy()
    if "proposed_max_far" in first_periods.columns:
        clust_feats["far"] = first_periods["proposed_max_far"].fillna(0)
    clust_feats["dose"] = df.groupby("case_number")["petition_pct_this_period"].max().fillna(0)
    scaled_coords = SKScaler().fit_transform(clust_feats.fillna(0).values)
    cells = KMeans(n_clusters=50, random_state=42, n_init=10).fit_predict(scaled_coords)
    cell_map = dict(zip(first_periods.index, cells))

    all_cases = first_dt.index.values
    all_groups = np.array([cell_map.get(c, 0) for c in all_cases])

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(all_cases, groups=all_groups))
    if fold >= len(splits):
        print(f"  [SKIP] fold {fold} out of range", flush=True)
        return

    train_idx, test_idx = splits[fold]

    cutoffs = [2017, 2018, 2019, 2020, 2021]
    cutoff = cutoffs[fold % len(cutoffs)]
    cutoff_date = pd.to_datetime(f"{cutoff}-12-31")
    end_test_date = pd.to_datetime(f"{cutoff + 3}-12-31")

    train_cells = set(all_groups[train_idx])
    test_cells = set(all_groups[test_idx])

    in_dist = (first_dt <= cutoff_date) & first_dt.index.map(
        lambda c: cell_map.get(c, -1) in train_cells
    )
    all_train_cases = first_dt[in_dist].index.values

    if len(all_train_cases) < 50:
        print(f"  [SKIP] Not enough training cases: {len(all_train_cases)}", flush=True)
        return

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_i, va_i = next(gss.split(all_train_cases, groups=all_train_cases))
    train_cases = all_train_cases[tr_i]
    val_cases = all_train_cases[va_i]

    test_cases = first_dt[
        (first_dt > cutoff_date)
        & (first_dt <= end_test_date)
        & first_dt.index.map(lambda c: cell_map.get(c, -1) in test_cells)
    ].index.values

    if len(test_cases) < 5:
        print(f"  [SKIP] Not enough test cases: {len(test_cases)}", flush=True)
        return

    X_tr, Y_tr, L_tr = build_tensors(df, features, targets, train_cases)
    X_va, Y_va, L_va = build_tensors(df, features, targets, val_cases)
    X_te, Y_te, L_te = build_tensors(df, features, targets, test_cases)

    print(
        f"  Train: {len(train_cases)} | Val: {len(val_cases)} | "
        f"Test(OOD/{cutoff+1}-{cutoff+3}): {len(test_cases)}",
        flush=True,
    )

    treat_idx = [
        features.index("petition_pct_this_period"),
        features.index("cumulative_petition_pct"),
    ]

    skip_cf_idx = []
    for col in ["proposed_max_far", "pdf_requested_height_ft", "land_acres"]:
        if col in features:
            skip_cf_idx.append(features.index(col))

    # Autoregressive update slots
    f_cum_tok = features.index("cumulative_council_nlp_lag1") if "cumulative_council_nlp_lag1" in features else None
    f_cum_comm = features.index("cumulative_commission_hearings_lag1") if "cumulative_commission_hearings_lag1" in features else None
    f_cum_coun = features.index("cumulative_council_hearings_lag1") if "cumulative_council_hearings_lag1" in features else None
    t_idx_tok = targets.index("council_nlp_total_tokens") if "council_nlp_total_tokens" in targets else None
    t_idx_comm = targets.index("commission_hearings_this_period") if "commission_hearings_this_period" in targets else None
    t_idx_coun = targets.index("council_hearings_this_period") if "council_hearings_this_period" in targets else None

    model = CausalSeq2SeqCFM(
        x_dim=len(features),
        y_dim=len(targets),
        treat_idx=treat_idx,
        skip_confounder_idx=skip_cf_idx or None,
        f_cum_tok=f_cum_tok, t_idx_tok=t_idx_tok,
        f_cum_comm=f_cum_comm, t_idx_comm=t_idx_comm,
        f_cum_coun=f_cum_coun, t_idx_coun=t_idx_coun,
    ).to(device)

    # Inject height stats from norm_dict
    model.height_u_mean.fill_(norm_dict.get("_height_pos_logit_mean", 0.0))
    model.height_u_std.fill_(norm_dict.get("_height_pos_logit_std", 1.0))
    model.height_pos_weight.fill_(norm_dict.get("_height_pos_weight", 10.0))

    # Separate prop_net parameters for lower LR (DML stability)
    prop_params = list(model.prop_net.parameters())
    prop_ids = {id(p) for p in prop_params}
    other_params = [p for p in model.parameters() if id(p) not in prop_ids]

    opt = torch.optim.AdamW(
        [{"params": other_params, "lr": LR},
         {"params": prop_params, "lr": LR * PROP_LR_SCALE}],
        weight_decay=1e-5,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.05)

    train_ds = TensorDataset(X_tr, Y_tr, L_tr)
    val_ds = TensorDataset(X_va, Y_va, L_va)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val = float("inf")
    patience_ctr = 0
    total_steps = max(1, EPOCHS * len(train_dl))
    step = 0

    for ep in range(EPOCHS):
        kl_beta = min(KL_BETA_MAX, KL_BETA_MAX * (step / (0.3 * total_steps + 1)))
        model.train()
        ep_dec, ep_kl, ep_prop, n_batches = 0.0, 0.0, 0.0, 0

        for Xb, Yb, Lb in train_dl:
            Xb, Yb, Lb = Xb.to(device), Yb.to(device), Lb.to(device)
            opt.zero_grad()
            dec_loss, kl_loss, prop_loss = model.forward_train(Xb, Yb, Lb)
            loss = dec_loss + kl_beta * kl_loss + 0.1 * prop_loss
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_dec += dec_loss.item()
                ep_kl += kl_loss.item()
                ep_prop += prop_loss.item()
                n_batches += 1
            else:
                opt.zero_grad()
            step += 1

        sched.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb, Lb in val_dl:
                Xb, Yb, Lb = Xb.to(device), Yb.to(device), Lb.to(device)
                dec, kl, prop = model.forward_train(Xb, Yb, Lb)
                val_loss += (dec + kl_beta * kl + 0.1 * prop).item()
        val_loss /= max(1, len(val_dl))

        if n_batches > 0:
            print(
                f"  Ep {ep:3d} | dec={ep_dec/n_batches:.4f} kl={ep_kl/n_batches:.4f} "
                f"prop={ep_prop/n_batches:.4f} | val={val_loss:.4f}",
                flush=True,
            )

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"_best_fold_{fold}.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {ep}", flush=True)
                break

    # Restore best checkpoint
    best_ckpt = os.path.join(OUT_DIR, f"_best_fold_{fold}.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Evaluation
    _evaluate_fold(fold, model, X_va, Y_va, L_va, X_te, Y_te, L_te, targets)

    # Save model and manifest
    ckpt_path = os.path.join(OUT_DIR, f"cfm_weights_fold_{fold}.pt")
    torch.save(model.state_dict(), ckpt_path)

    manifest = {
        "fold": fold,
        "features": features,
        "targets": targets,
        "treat_idx": treat_idx,
        "skip_confounder_idx": skip_cf_idx,
        "resolved_idx": model.resolved_idx,
        "height_idx": model.height_idx,
        "cont_idx": model.cont_idx,
        "f_cum_tok": f_cum_tok, "t_idx_tok": t_idx_tok,
        "f_cum_comm": f_cum_comm, "t_idx_comm": t_idx_comm,
        "f_cum_coun": f_cum_coun, "t_idx_coun": t_idx_coun,
        "norm_dict": {
            k: ([float(v[0]), float(v[1])] if isinstance(v, tuple) else float(v))
            for k, v in norm_dict.items()
        },
        "x_dim": len(features),
        "y_dim": len(targets),
    }
    mf_path = os.path.join(OUT_DIR, f"feature_manifest_fold_{fold}.json")
    with open(mf_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"  Saved: {ckpt_path} | {mf_path}", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate_fold(fold, model, X_va, Y_va, L_va, X_te, Y_te, L_te, targets):
    """Quick teacher-forced evaluation metrics per fold."""
    from sklearn.metrics import roc_auc_score, mean_absolute_error

    model.eval()
    resolved_idx = model.resolved_idx
    height_idx = model.height_idx

    for split_name, X, Y, L in [("VAL", X_va, Y_va, L_va), ("TEST", X_te, Y_te, L_te)]:
        all_preds, all_true, all_valid = [], [], []

        with torch.no_grad():
            for i in range(0, len(X), BATCH_SIZE):
                Xb = X[i: i + BATCH_SIZE].to(device)
                Yb = Y[i: i + BATCH_SIZE].to(device)
                Lb = L[i: i + BATCH_SIZE].to(device)

                B, T, _ = Xb.shape
                n_pred = T - PRE_PERIODS
                steps = torch.arange(PRE_PERIODS, T, device=device).unsqueeze(0)
                valid = steps < Lb.unsqueeze(1)          # (B, n_pred)

                z, mu, lv = model.encoder(Xb[:, :PRE_PERIODS, :])
                y_prev_seq = torch.cat([
                    torch.zeros(B, 1, model.y_dim, device=device),
                    Yb[:, PRE_PERIODS:-1, :],
                ], dim=1)
                x_seq_unb = Xb[:, PRE_PERIODS:, :].clone()
                for idx in model.treat_idx:
                    x_seq_unb[:, :, idx] = 0.0
                state = model.transition.init_hidden(B, device)
                inp = torch.cat([x_seq_unb, y_prev_seq], dim=-1)
                out_seq, _ = model.transition.rnn(inp, state)

                B_seq = B * n_pred
                dose_flat = Xb[:, PRE_PERIODS:, model.treat_idx[0]].unsqueeze(-1).reshape(B_seq, 1)
                z_flat = z.unsqueeze(1).repeat(1, n_pred, 1).reshape(B_seq, -1)
                h_flat = out_seq.reshape(B_seq, model.hidden_dim)

                raw_cf = None
                if model.skip_confounder_idx:
                    raw_cf = x_seq_unb[:, :, model.skip_confounder_idx].reshape(B_seq, -1)

                prop_logit, prop_mag = model.prop_net(z_flat, h_flat, raw_cf)
                dose_nz = (dose_flat > 0).float()
                dose_residual = dose_flat - (torch.sigmoid(prop_logit) * prop_mag)
                dose_enc = torch.cat([dose_nz, dose_residual], dim=-1)

                gate_logit = model.height_gate(z_flat, h_flat, dose_enc)
                p_pos = torch.sigmoid(gate_logit).reshape(B, n_pred)
                height_true = Yb[:, PRE_PERIODS:, height_idx]
                resolved_logit = model.resolved_head(z_flat, h_flat, dose_enc).reshape(B, n_pred)
                resolved_true = Yb[:, PRE_PERIODS:, resolved_idx]

                all_preds.append({
                    "height_p_pos": p_pos.cpu(),
                    "resolved_logit": resolved_logit.cpu(),
                    "height_true": height_true.cpu(),
                    "resolved_true": resolved_true.cpu(),
                    "valid": valid.cpu(),
                })

        # Aggregate
        hp = torch.cat([d["height_p_pos"] for d in all_preds], dim=0)
        hl = torch.cat([d["resolved_logit"] for d in all_preds], dim=0)
        ht = torch.cat([d["height_true"] for d in all_preds], dim=0)
        rt = torch.cat([d["resolved_true"] for d in all_preds], dim=0)
        vm = torch.cat([d["valid"] for d in all_preds], dim=0)

        vm_flat = vm.reshape(-1)
        hp_flat = hp.reshape(-1)[vm_flat].numpy()
        ht_flat = ht.reshape(-1)[vm_flat].numpy()
        rt_flat = rt.reshape(-1)[vm_flat].numpy()
        rl_flat = hl.reshape(-1)[vm_flat].numpy()

        ht_binary = (ht_flat > HEIGHT_EPS).astype(float)
        mae_gate = np.abs(hp_flat - ht_binary).mean()

        if ht_binary.sum() > 5:
            roc_gate = roc_auc_score(ht_binary, hp_flat)
        else:
            roc_gate = float("nan")

        if len(np.unique(rt_flat)) > 1:
            resolved_prob = 1.0 / (1.0 + np.exp(-rl_flat))
            roc_resolved = roc_auc_score(rt_flat, resolved_prob)
        else:
            roc_resolved = float("nan")

        pos_rate_pred = float((hp_flat > 0.5).mean())
        pos_rate_true = float(ht_binary.mean())

        print(
            f"  [{split_name}] Height gate ROC={roc_gate:.3f} | MAE={mae_gate:.4f} | "
            f"pred_pos_rate={pos_rate_pred:.3f} vs true={pos_rate_true:.3f} | "
            f"Resolved ROC={roc_resolved:.3f}",
            flush=True,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Fold index 0-4")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print(f"Zero-Inflated Causal Hurdle CFM  |  device={device}", flush=True)
    print("=" * 60, flush=True)

    df, features, targets, norm_dict = load_data()
    train_fold(args.fold, df, features, targets, norm_dict)

    print("\n[SUCCESS] Fold complete.", flush=True)


if __name__ == "__main__":
    main()
