"""
vae_gcomp_lstm_hybrid.py
=========================
Hybrid Causal Pipeline: VAE generates counterfactual covariate trajectories,
LSTM produces precise outcome predictions from each generated trajectory.

Pipeline:
  1. Train Conditional VAE on training data
  2. Train Multi-Task LSTM on training data
  3. For each test case, create two intervention regimes:
     - Control: petition_pct = 0 (no organized opposition)
     - Treated: petition_pct = observed value
  4. Sample N=500 latent z values per case per regime from VAE
  5. Decode each z into a full 30-step covariate trajectory
  6. Feed each trajectory through the LSTM for precise outcome predictions
  7. Compute distributional ATE with credible intervals
"""

import os, time, math, re
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"
N_SAMPLES = 500

# ============================================================
# DATA
# ============================================================
def load_data():
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
    
    features = [
        "land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km",
        "local_unemployment_rate", "mortgage_rate_30yr", "period_seq", "petition_pct_this_period",
        "cumulative_petition_pct", "bw_sin", "bw_cos"
    ]
    targets = ["resolved", "vote_event", "proposed_max_height_ft", "council_nlp_total_tokens", "commission_hearings_this_period", "council_hearings_this_period"]
    
    for f in features + targets:
        df[f] = df[f].fillna(0)
    
    norm_dict = {}
    for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km",
              "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        mean_v = df[f].mean()
        std_v = df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)
    
    return df, features, targets, norm_dict


def build_tensors(df, features, targets, max_seq=30):
    df = df.sort_values(["case_number", "period_seq"])
    case_sizes = df.groupby("case_number").size()
    cases = case_sizes.index.values
    n_cases = len(cases)

    X_all = np.zeros((n_cases, max_seq, len(features)), dtype=np.float32)
    Y_all = np.zeros((n_cases, max_seq, len(targets)), dtype=np.float32)
    lengths_all = np.zeros(n_cases, dtype=np.int64)

    feat_vals = df[features].values.astype(np.float32)
    targ_vals = df[targets].values.astype(np.float32)

    idx = 0
    for i, case in enumerate(cases):
        size = case_sizes[case]
        length = min(size, max_seq)
        X_all[i, :length, :] = feat_vals[idx:idx+length]
        Y_all[i, :length, :] = targ_vals[idx:idx+length]
        lengths_all[i] = length
        idx += size

    # Random 80/20 case-level split for proportional treated-case distribution
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_cases)
    split = int(0.8 * n_cases)
    train_idx = perm[:split]
    test_idx  = perm[split:]

    return (
        torch.from_numpy(X_all[train_idx]),
        torch.from_numpy(Y_all[train_idx]),
        torch.from_numpy(lengths_all[train_idx]),
        torch.from_numpy(X_all[test_idx]),
        torch.from_numpy(Y_all[test_idx]),
        torch.from_numpy(lengths_all[test_idx]),
    )

# ============================================================
# MODELS
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ConditionalVAE(nn.Module):
    """VAE that encodes full trajectories and decodes covariate sequences."""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # Encoder
        self.enc_proj = nn.Linear(input_dim, d_model)
        self.enc_pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                              dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        
        # Decoder: reconstructs the FULL covariate sequence from z + positional
        self.dec_proj = nn.Linear(latent_dim, d_model)
        self.dec_pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                              dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)  # Reconstruct covariates
    
    def encode(self, x):
        h = self.enc_pos(self.enc_proj(x))
        h = self.encoder(h)
        h_pool = h.mean(dim=1)
        return self.fc_mu(h_pool), self.fc_logvar(h_pool)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len=30):
        B = z.size(0)
        # Broadcast z into a sequence
        z_seq = z.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, latent_dim]
        h = self.dec_pos(self.dec_proj(z_seq))
        h = self.decoder(h)
        return self.output_proj(h)  # [B, T, input_dim]
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar


class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=0.1 if num_layers > 1 else 0)
        self.head_surv = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_vote = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_ht   = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_tok  = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_comm = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_coun = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    
    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            h, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            h, _ = self.lstm(x)
        surv  = self.head_surv(h)
        vote  = self.head_vote(h)
        ht    = self.head_ht(h)
        tok   = self.head_tok(h)
        comm  = self.head_comm(h)
        coun  = self.head_coun(h)
        return torch.cat([surv, vote, ht, tok, comm, coun], dim=-1)


# ============================================================
# TRAINING
# ============================================================
def train_vae(model, X_train, epochs=40, lr=0.001, batch_size=256, kl_weight=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_recon, total_kl = 0, 0
        for (bx,) in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(bx)
            
            mask = (bx[:, :, -1] != 0).float().unsqueeze(-1)  # [B, T, 1]
            
            recon_loss = (((recon - bx) ** 2) * mask).sum() / mask.sum()
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        if epoch % 10 == 0:
            print(f"  VAE Epoch {epoch}: Recon={total_recon/len(loader):.4f} KL={total_kl/len(loader):.4f}", flush=True)


def train_lstm(model, X_train, Y_train, L_train, X_val, Y_val, L_val, epochs=20, lr=0.001, batch_size=256):
    # Dynamically compute pos_weight from actual class imbalance in training set
    # Y_train[:, :, 0] is the survival (resolved) binary target — the primary causal outcome
    y_flat = Y_train[:, :, 0].flatten()
    n_pos = y_flat.sum().item()
    n_neg = (y_flat == 0).sum().item()
    computed_pos_weight = n_neg / (n_pos + 1e-8)
    print(f"  [LSTM] Computed pos_weight: {computed_pos_weight:.1f} (n_pos={int(n_pos)}, n_neg={int(n_neg)})", flush=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([computed_pos_weight]), reduction='none')
    crit_mse = nn.MSELoss(reduction='none')
    
    dataset = TensorDataset(X_train, Y_train, L_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, Y_val, L_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    import copy
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    grad_norms = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by, bl in loader:
            optimizer.zero_grad()
            preds = model(bx, bl)
            mask = (bx[:, :, -1] != 0).float()
            
            loss_surv = (crit_bce(preds[:, :, 0], by[:, :, 0]) * mask).sum() / mask.sum()
            loss_vote = (crit_bce(preds[:, :, 1], by[:, :, 1]) * mask).sum() / mask.sum()
            loss_ht   = (crit_mse(preds[:, :, 2], by[:, :, 2]) * mask).sum() / mask.sum()
            loss_tok  = (crit_mse(preds[:, :, 3], by[:, :, 3]) * mask).sum() / mask.sum()
            loss_comm = (crit_mse(preds[:, :, 4], by[:, :, 4]) * mask).sum() / mask.sum()
            loss_coun = (crit_mse(preds[:, :, 5], by[:, :, 5]) * mask).sum() / mask.sum()
            
            loss = loss_surv + loss_vote + loss_ht + loss_tok + loss_comm + loss_coun
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            grad_norms.append(grad_norm)
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Loop
        model.eval()
        # Per-head tracking across val batches
        head_losses = {"surv": 0, "vote": 0, "ht": 0, "tok": 0, "comm": 0, "coun": 0}
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, bl in val_loader:
                preds = model(bx, bl)
                mask = (bx[:, :, -1] != 0).float()
                l_s = (crit_bce(preds[:, :, 0], by[:, :, 0]) * mask).sum() / mask.sum()
                l_v = (crit_bce(preds[:, :, 1], by[:, :, 1]) * mask).sum() / mask.sum()
                l_h = (crit_mse(preds[:, :, 2], by[:, :, 2]) * mask).sum() / mask.sum()
                l_t = (crit_mse(preds[:, :, 3], by[:, :, 3]) * mask).sum() / mask.sum()
                l_m = (crit_mse(preds[:, :, 4], by[:, :, 4]) * mask).sum() / mask.sum()
                l_c = (crit_mse(preds[:, :, 5], by[:, :, 5]) * mask).sum() / mask.sum()
                v_loss = l_s + l_v + l_h + l_t + l_m + l_c
                val_loss += v_loss.item()
                head_losses["surv"] += l_s.item()
                head_losses["vote"] += l_v.item()
                head_losses["ht"]   += l_h.item()
                head_losses["tok"]  += l_t.item()
                head_losses["comm"] += l_m.item()
                head_losses["coun"] += l_c.item()
                
        val_loss /= len(val_loader)
        for k in head_losses:
            head_losses[k] /= len(val_loader)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            
        if (epoch + 1) % 5 == 0:
            avg_grad = np.mean(grad_norms[-len(loader):]) if grad_norms else 0
            print(f"  LSTM Epoch {epoch+1:02d}/{epochs} | Train: {train_loss/len(loader):.4f} | Val: {val_loss:.4f} | GradNorm: {avg_grad:.4f}", flush=True)
            print(f"    Per-Head Val -> surv: {head_losses['surv']:.4f} | vote: {head_losses['vote']:.4f} | ht: {head_losses['ht']:.4f} | tok: {head_losses['tok']:.4f} | comm: {head_losses['comm']:.4f} | coun: {head_losses['coun']:.4f}", flush=True)
            
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [Early Stop] Best checkpoint: Epoch {best_epoch}/{epochs} | Best Val Loss: {best_val_loss:.4f} | Avg GradNorm: {np.mean(grad_norms):.4f}", flush=True)


# ============================================================
# COUNTERFACTUAL PIPELINE
# ============================================================
def run_counterfactual_pipeline(vae, lstm, X_test, features, norm_dict, n_samples=N_SAMPLES):
    """
    For each test case:
      1. Encode into latent z
      2. Sample N latent vectors around the posterior
      3. Decode each into a covariate trajectory
      4. Create CONTROL (petition=0) and TREATED (petition=observed) variants
      5. Feed each through the LSTM
      6. Collect distributional outcomes
    """
    vae.eval()
    lstm.eval()
    
    pet_idx = features.index("petition_pct_this_period")
    cum_pet_idx = features.index("cumulative_petition_pct")
    
    B = X_test.shape[0]
    
    # Encode all test cases
    with torch.no_grad():
        mu, logvar = vae.encode(X_test)
    
    print(f"\n[*] Running {n_samples} counterfactual samples per case across {B} test cases...", flush=True)
    
    # Storage for final-step predictions
    control_surv = np.zeros((B, n_samples))
    treated_surv = np.zeros((B, n_samples))
    control_vote = np.zeros((B, n_samples))
    treated_vote = np.zeros((B, n_samples))
    control_ht = np.zeros((B, n_samples))
    treated_ht = np.zeros((B, n_samples))
    control_tok = np.zeros((B, n_samples))
    treated_tok = np.zeros((B, n_samples))
    
    batch_size = 64  # Process samples in batches to avoid OOM
    
    with torch.no_grad():
        for sample_start in range(0, n_samples, batch_size):
            sample_end = min(sample_start + batch_size, n_samples)
            n_batch = sample_end - sample_start
            
            # Sample z for ALL test cases at once
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(B, n_batch, mu.size(1))  # [B, n_batch, latent_dim]
            z_samples = mu.unsqueeze(1) + eps * std.unsqueeze(1)  # [B, n_batch, latent_dim]
            
            # Process each sample in the batch
            for s_idx in range(n_batch):
                z = z_samples[:, s_idx, :]  # [B, latent_dim]
                
                # Decode into covariate trajectory
                gen_traj = vae.decode(z, seq_len=30)  # [B, 30, n_features]
                
                # CONTROL: zero out petition features
                control_traj = gen_traj.clone()
                control_traj[:, :, pet_idx] = 0.0
                control_traj[:, :, cum_pet_idx] = 0.0
                
                # TREATED: keep generated petition values (or use observed)
                treated_traj = gen_traj.clone()
                # Use OBSERVED petition values from the real data
                treated_traj[:, :, pet_idx] = X_test[:, :, pet_idx]
                treated_traj[:, :, cum_pet_idx] = X_test[:, :, cum_pet_idx]
                
                # Feed through LSTM
                ctrl_pred = lstm(control_traj)   # [B, 30, 6]
                trt_pred = lstm(treated_traj)    # [B, 30, 6]
                
                # Extract final-step predictions
                s_global = sample_start + s_idx
                control_surv[:, s_global] = torch.sigmoid(ctrl_pred[:, -1, 0]).numpy()
                treated_surv[:, s_global] = torch.sigmoid(trt_pred[:, -1, 0]).numpy()
                control_vote[:, s_global] = torch.sigmoid(ctrl_pred[:, -1, 1]).numpy()
                treated_vote[:, s_global] = torch.sigmoid(trt_pred[:, -1, 1]).numpy()
                control_ht[:, s_global] = ctrl_pred[:, -1, 2].numpy()
                treated_ht[:, s_global] = trt_pred[:, -1, 2].numpy()
                control_tok[:, s_global] = ctrl_pred[:, -1, 3].numpy()
                treated_tok[:, s_global] = trt_pred[:, -1, 3].numpy()
            
            if (sample_start + batch_size) % 100 == 0:
                print(f"  Processed {min(sample_start + batch_size, n_samples)}/{n_samples} samples...", flush=True)
    
    return control_surv, treated_surv, control_vote, treated_vote, control_ht, treated_ht, control_tok, treated_tok


def main():
    print("=" * 60, flush=True)
    print("VAE -> G-Computation LSTM Hybrid Pipeline", flush=True)
    print("=" * 60, flush=True)
    
    print("\n[1/5] Loading data...", flush=True)
    df, features, targets, norm_dict = load_data()
    
    print("[2/5] Building tensors...", flush=True)
    X_train, Y_train, L_train, X_test, Y_test, L_test = build_tensors(df, features, targets)
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}", flush=True)
    
    print("\n[3/5] Training Conditional VAE (world model)...", flush=True)
    vae = ConditionalVAE(len(features))
    
    print("Training Causal VAE...")
    train_vae(vae, X_train, epochs=20, lr=0.001)
    
    print("Training Outcome Surrogate LSTM...")
    lstm = MultiTaskLSTM(len(features))
    train_lstm(lstm, X_train, Y_train, L_train, X_test, Y_test, L_test, epochs=20, lr=0.001)
    
    print("\n[5/5] Running Counterfactual G-Computation...", flush=True)
    ctrl_surv, trt_surv, ctrl_vote, trt_vote, ctrl_ht, trt_ht, ctrl_tok, trt_tok = run_counterfactual_pipeline(
        vae, lstm, X_test, features, norm_dict, n_samples=N_SAMPLES
    )
    
    # Denormalize height
    mean_ht, std_ht = norm_dict["proposed_max_height_ft"]
    ctrl_ht_ft = ctrl_ht * std_ht + mean_ht
    trt_ht_ft = trt_ht * std_ht + mean_ht
    
    # ---- AGGREGATE RESULTS ----
    print("\n" + "=" * 60, flush=True)
    print("DISTRIBUTIONAL CAUSAL ESTIMATES", flush=True)
    print("=" * 60, flush=True)
    
    # ATE for Survival (probability of case being resolved/killed)
    ate_surv = (trt_surv - ctrl_surv).mean(axis=1)  # Per-case ATE
    print(f"\n--- SURVIVAL (Case Resolution) ---", flush=True)
    print(f"  Control P(Resolved) [No Opposition]:     {ctrl_surv.mean():.4f}", flush=True)
    print(f"  Treated P(Resolved) [With Opposition]:   {trt_surv.mean():.4f}", flush=True)
    print(f"  ATE (mean):  {ate_surv.mean():.4f}", flush=True)
    print(f"  ATE (2.5%):  {np.percentile(ate_surv, 2.5):.4f}", flush=True)
    print(f"  ATE (97.5%): {np.percentile(ate_surv, 97.5):.4f}", flush=True)
    
    # ATE for Height
    ate_ht = (trt_ht_ft - ctrl_ht_ft).mean(axis=1)  # Per-case ATE in feet
    print(f"\n--- HEIGHT REDUCTION (Vertical Feet) ---", flush=True)
    print(f"  Control Mean Height [No Opposition]:     {ctrl_ht_ft.mean():.2f} ft", flush=True)
    print(f"  Treated Mean Height [With Opposition]:   {trt_ht_ft.mean():.2f} ft", flush=True)
    print(f"  ATE (mean):  {ate_ht.mean():.2f} ft", flush=True)
    print(f"  ATE (2.5%):  {np.percentile(ate_ht, 2.5):.2f} ft", flush=True)
    print(f"  ATE (97.5%): {np.percentile(ate_ht, 97.5):.2f} ft", flush=True)
    
    # ATE for Probability of Approval (vote_event)
    ate_vote = (trt_vote - ctrl_vote).mean(axis=1)
    print(f"\n--- PROBABILITY OF APPROVAL (Vote Event) ---", flush=True)
    print(f"  Control P(Approval) [No Opposition]:     {ctrl_vote.mean():.4f}", flush=True)
    print(f"  Treated P(Approval) [With Opposition]:   {trt_vote.mean():.4f}", flush=True)
    print(f"  ATE (mean):  {ate_vote.mean():.4f}", flush=True)
    print(f"  ATE (2.5%):  {np.percentile(ate_vote, 2.5):.4f}", flush=True)
    print(f"  ATE (97.5%): {np.percentile(ate_vote, 97.5):.4f}", flush=True)
    
    # ATE for Cumulative Meeting Tokens
    ate_tok = (trt_tok - ctrl_tok).mean(axis=1)
    print(f"\n--- CUMULATIVE MEETING TOKENS ---", flush=True)
    print(f"  Control Tokens [No Opposition]:     {ctrl_tok.mean():.2f}", flush=True)
    print(f"  Treated Tokens [With Opposition]:   {trt_tok.mean():.2f}", flush=True)
    print(f"  ATE (mean):  {ate_tok.mean():.2f}", flush=True)
    print(f"  ATE (2.5%):  {np.percentile(ate_tok, 2.5):.2f}", flush=True)
    print(f"  ATE (97.5%): {np.percentile(ate_tok, 97.5):.2f}", flush=True)
    
    # Heterogeneous Treatment Effects: Cases with actual opposition
    has_petition = X_test[:, :, features.index("petition_pct_this_period")].sum(dim=1).numpy() > 0
    print(f"\n--- SUBGROUP: Cases WITH Observed Opposition ({has_petition.sum()} cases) ---", flush=True)
    if has_petition.sum() > 0:
        sub_ate_surv = ate_surv[has_petition]
        sub_ate_ht = ate_ht[has_petition]
        print(f"  Survival ATE (mean):   {sub_ate_surv.mean():.4f} [{np.percentile(sub_ate_surv, 2.5):.4f}, {np.percentile(sub_ate_surv, 97.5):.4f}]", flush=True)
        print(f"  Height ATE (mean):     {sub_ate_ht.mean():.2f} ft [{np.percentile(sub_ate_ht, 2.5):.2f}, {np.percentile(sub_ate_ht, 97.5):.2f}]", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    
    # Save raw distributions for plotting
    results = pd.DataFrame({
        "control_surv_mean": ctrl_surv.mean(axis=1),
        "treated_surv_mean": trt_surv.mean(axis=1),
        "control_surv_std": ctrl_surv.std(axis=1),
        "treated_surv_std": trt_surv.std(axis=1),
        "control_ht_mean": ctrl_ht_ft.mean(axis=1),
        "treated_ht_mean": trt_ht_ft.mean(axis=1),
        "control_ht_std": ctrl_ht_ft.std(axis=1),
        "treated_ht_std": trt_ht_ft.std(axis=1),
        "ate_surv": ate_surv,
        "ate_ht": ate_ht,
        "has_petition": has_petition
    })
    out_path = os.path.join(OUT_DIR, "vae_gcomp_counterfactuals.csv")
    results.to_csv(out_path, index=False)
    print(f"\nSaved distributional counterfactuals to {out_path}", flush=True)


if __name__ == "__main__":
    main()
