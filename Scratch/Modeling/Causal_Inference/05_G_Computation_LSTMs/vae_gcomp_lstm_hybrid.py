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
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"
VAE_CHECKPOINT = rf"{OUT_DIR}\causal_vae_weights.pt"
LSTM_CHECKPOINT = rf"{OUT_DIR}\causal_lstm_weights.pt"
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
    for f in ["land_acres", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km",
              "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        mean_v = df[f].mean()
        std_v = df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)
        
    # Log1p transforms for highly right-skewed variables
    for f in ["proposed_max_height_ft", "council_nlp_total_tokens"]:
        df[f] = np.log1p(df[f].clip(lower=0))
        
    # Binary occurrence for sparse count variables
    for f in ["commission_hearings_this_period", "council_hearings_this_period"]:
        df[f] = df[f].clip(lower=0, upper=1)
    
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
def train_vae(model, X_train, X_val, features, epochs=50, lr=0.001, batch_size=256, kl_weight=0.0005, smooth_weight=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=(device.type=='cuda'))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    dataset = TensorDataset(X_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    val_dataset = TensorDataset(X_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    scaler = torch.amp.GradScaler('cuda')
    pet_idx = features.index("petition_pct_this_period")
    
    import copy
    best_val_recon = float('inf')
    best_state = None
    patience_limit = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_recon, total_kl = 0, 0
        for (bx,) in loader:
            bx = bx.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                recon, mu, logvar = model(bx)
                mask = (bx[:, :, -1] != 0).float().unsqueeze(-1)  # [B, T, 1]
                recon_loss = (((recon - bx) ** 2) * mask).sum() / mask.sum()
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                smooth_loss = (((recon[:, 1:, :] - recon[:, :-1, :]) ** 2) * mask[:, 1:, :]).sum() / mask[:, 1:, :].sum()
                loss = recon_loss + kl_weight * kl_loss + smooth_weight * smooth_loss
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
        model.eval()
        val_recon = 0
        val_pet_recon = 0
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                with torch.amp.autocast('cuda'):
                    recon, mu, logvar = model(bx)
                    mask = (bx[:, :, -1] != 0).float().unsqueeze(-1)
                    v_loss = (((recon - bx) ** 2) * mask).sum() / mask.sum()
                    v_pet = (((recon[:, :, pet_idx] - bx[:, :, pet_idx]) ** 2) * mask.squeeze(-1)).sum() / mask.squeeze(-1).sum()
                val_recon += v_loss.item()
                val_pet_recon += v_pet.item()
        val_recon /= len(val_loader)
        val_pet_recon /= len(val_loader)
        
        scheduler.step(val_recon)
        
        if val_recon < best_val_recon:
            best_val_recon = val_recon
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  VAE Epoch {epoch:02d}: Train Recon={total_recon/len(loader):.4f} | Train KL={total_kl/len(loader):.4f} | Val Recon={val_recon:.4f} | Val PetMSE={val_pet_recon:.4f}", flush=True)

        if patience_counter >= patience_limit:
            print(f"  [VAE Early Stop] Halted at Epoch {epoch}. Best Val Recon: {best_val_recon:.4f}", flush=True)
            break

    print(f"  [VAE] Loading best state and saving checkpoint to {VAE_CHECKPOINT}...", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), VAE_CHECKPOINT)


def train_lstm(model, X_train, Y_train, L_train, X_val, Y_val, L_val, epochs=20, lr=0.001, batch_size=256):
    y_flat = Y_train[:, :, 0].flatten()
    n_pos = y_flat.sum().item()
    n_neg = (y_flat == 0).sum().item()
    computed_pos_weight = n_neg / (n_pos + 1e-8)
    print(f"  [LSTM] Computed pos_weight: {computed_pos_weight:.1f} (n_pos={int(n_pos)}, n_neg={int(n_neg)})", flush=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, fused=(device.type=='cuda'))
    crit_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([computed_pos_weight], device=device), reduction='none')
    crit_mse = nn.MSELoss(reduction='none')
    
    dataset = TensorDataset(X_train, Y_train, L_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    val_dataset = TensorDataset(X_val, Y_val, L_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    scaler = torch.amp.GradScaler('cuda')
    
    import copy
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    patience_limit = 10
    patience_counter = 0
    grad_norms = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by, bl in loader:
            bx, by, bl = bx.to(device), by.to(device), bl.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                preds = model(bx, bl)
                mask = (bx[:, :, -1] != 0).float()
                
                loss_surv = (crit_bce(preds[:, :, 0], by[:, :, 0]) * mask).sum() / mask.sum()
                loss_vote = (crit_bce(preds[:, :, 1], by[:, :, 1]) * mask).sum() / mask.sum()
                loss_ht   = (crit_mse(preds[:, :, 2], by[:, :, 2]) * mask).sum() / mask.sum()
                loss_tok  = (crit_mse(preds[:, :, 3], by[:, :, 3]) * mask).sum() / mask.sum()
                loss_comm = (crit_bce(preds[:, :, 4], by[:, :, 4]) * mask).sum() / mask.sum()
                loss_coun = (crit_bce(preds[:, :, 5], by[:, :, 5]) * mask).sum() / mask.sum()
                
                loss = loss_surv + loss_vote + loss_ht + loss_tok + loss_comm + loss_coun
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            grad_norms.append(grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        model.eval()
        head_losses = {"surv": 0, "vote": 0, "ht": 0, "tok": 0, "comm": 0, "coun": 0}
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, bl in val_loader:
                bx, by, bl = bx.to(device), by.to(device), bl.to(device)
                with torch.amp.autocast('cuda'):
                    preds = model(bx, bl)
                    mask = (bx[:, :, -1] != 0).float()
                    l_s = (crit_bce(preds[:, :, 0], by[:, :, 0]) * mask).sum() / mask.sum()
                    l_v = (crit_bce(preds[:, :, 1], by[:, :, 1]) * mask).sum() / mask.sum()
                    l_h = (crit_mse(preds[:, :, 2], by[:, :, 2]) * mask).sum() / mask.sum()
                    l_t = (crit_mse(preds[:, :, 3], by[:, :, 3]) * mask).sum() / mask.sum()
                    l_m = (crit_bce(preds[:, :, 4], by[:, :, 4]) * mask).sum() / mask.sum()
                    l_c = (crit_bce(preds[:, :, 5], by[:, :, 5]) * mask).sum() / mask.sum()
                v_loss = l_s + l_v + l_h + l_t + l_m + l_c
                val_loss += v_loss.item()
                head_losses["surv"] += l_s.item()
                head_losses["vote"] += l_v.item()
                head_losses["ht"]   += l_h.item()
                head_losses["tok"]  += l_t.item()
                head_losses["comm"] += l_m.item()
                head_losses["coun"] += l_c.item()
                
        val_loss /= len(val_loader)
        for k in head_losses: head_losses[k] /= len(val_loader)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  LSTM Epoch {epoch:02d}/{epochs} | Train: {train_loss/len(loader):.4f} | Val: {val_loss:.4f} | GradNorm: {np.mean(grad_norms):.4f}", flush=True)
            print(f"    Per-Head Val -> surv: {head_losses['surv']:.4f} | vote: {head_losses['vote']:.4f} | ht: {head_losses['ht']:.4f} | tok: {head_losses['tok']:.4f} | comm: {head_losses['comm']:.4f} | coun: {head_losses['coun']:.4f}", flush=True)
            
        if patience_counter >= patience_limit:
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), LSTM_CHECKPOINT)
    print(f"  [LSTM] Best checkpoint saved to {LSTM_CHECKPOINT}. Best Val Loss: {best_val_loss:.4f}", flush=True)


# ============================================================
# COUNTERFACTUAL PIPELINE
# ============================================================
def run_counterfactual_pipeline(vae, lstm, X_test, features):
    vae.eval()
    lstm.eval()
    
    pet_idx = features.index("petition_pct_this_period")
    cum_pet_idx = features.index("cumulative_petition_pct")
    
    B = X_test.shape[0]
    n_samples = 500
    
    with torch.no_grad():
        mu, logvar = vae.encode(X_test)
    
    print(f"\n[*] Running {n_samples} counterfactual samples per case across {B} test cases...", flush=True)
    
    doses = np.linspace(0.0, 1.0, 11).tolist()
    results = {
        d: {
            "surv": np.zeros((B, n_samples)),
            "vote": np.zeros((B, n_samples)),
            "ht": np.zeros((B, n_samples)),
            "tok": np.zeros((B, n_samples)),
        } for d in doses
    }
    
    batch_size = 32
    
    with torch.no_grad():
        for sample_start in range(0, n_samples, batch_size):
            sample_end = min(sample_start + batch_size, n_samples)
            n_batch_samples = sample_end - sample_start
            
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(B, n_batch_samples, mu.size(1), device=device)
            z_samples = mu.unsqueeze(1) + eps * std.unsqueeze(1)
            
            for s_idx in range(n_batch_samples):
                z = z_samples[:, s_idx, :]
                gen_traj = vae.decode(z, seq_len=30)
                
                # Expand gen_traj for 11 doses: [B, 11, 30, F]
                traj_stack = gen_traj.unsqueeze(1).repeat(1, 11, 1, 1)
                
                for d_idx, d in enumerate(doses):
                    traj_stack[:, d_idx, :, pet_idx] = 0.0
                    traj_stack[:, d_idx, 4, pet_idx] = d # Intervention at period 5
                    traj_stack[:, d_idx, 4:, cum_pet_idx] = d # Propagate cumulative
                
                # Flatten to [B*11, 30, F] for the LSTM
                lstm_input = traj_stack.view(B * 11, 30, -1)
                preds = lstm(lstm_input) # Returns [B*11, 30, 6]
                
                # Reshape back to [B, 11, 30, 6]
                preds = preds.view(B, 11, 30, 6)
                
                s_global = sample_start + s_idx
                
                for d_idx, d in enumerate(doses):
                    # 1. P(Resolved) = 1 - S(T)
                    h_surv = torch.sigmoid(preds[:, d_idx, :, 0]) # [B, 30]
                    P_surv = 1.0 - torch.prod(1.0 - h_surv, dim=1)
                    results[d]["surv"][:, s_global] = P_surv.cpu().numpy()
                    
                    # 2. P(Vote)
                    h_vote = torch.sigmoid(preds[:, d_idx, :, 1])
                    P_vote = 1.0 - torch.prod(1.0 - h_vote, dim=1)
                    results[d]["vote"][:, s_global] = P_vote.cpu().numpy()
                    
                    # 3. Height (Terminal)
                    ht_raw = torch.expm1(preds[:, d_idx, -1, 2])
                    results[d]["ht"][:, s_global] = ht_raw.cpu().numpy()
                    
                    # 4. Tokens (Sum)
                    tok_raw = torch.expm1(preds[:, d_idx, :, 3]).sum(dim=1)
                    results[d]["tok"][:, s_global] = tok_raw.cpu().numpy()
            
            if (sample_start + n_batch_samples) % 64 == 0 or (sample_start + n_batch_samples) == n_samples:
                print(f"  Processed {sample_start + n_batch_samples}/{n_samples} samples...", flush=True)
                
    return results


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
    vae = ConditionalVAE(len(features)).to(device)
    
    if os.path.exists(VAE_CHECKPOINT):
        print(f"  Found existing VAE checkpoint at {VAE_CHECKPOINT}. Loading weights...", flush=True)
        vae.load_state_dict(torch.load(VAE_CHECKPOINT, weights_only=True))
    else:
        print("Training Causal VAE...")
        train_vae(vae, X_train, X_test, features, epochs=50, lr=0.001)
    
    print("Training Outcome Surrogate LSTM...")
    lstm = MultiTaskLSTM(len(features)).to(device)
    
    if os.path.exists(LSTM_CHECKPOINT):
        print(f"  Found existing LSTM checkpoint at {LSTM_CHECKPOINT}. Loading weights...", flush=True)
        lstm.load_state_dict(torch.load(LSTM_CHECKPOINT, weights_only=True))
    else:
        train_lstm(lstm, X_train, Y_train, L_train, X_test, Y_test, L_test, epochs=20, lr=0.001)
    
    print("\n[5/5] Running Counterfactual Dose-Response Matrix...", flush=True)
    results = run_counterfactual_pipeline(vae, lstm, X_test, features)
    
    # ---- AGGREGATE RESULTS ----
    print("\n" + "=" * 60, flush=True)
    print("DOSE-RESPONSE CAUSAL SURFACE (N=1317 cases)", flush=True)
    print("=" * 60, flush=True)
    
    ctrl_d = 0.0
    doses = np.linspace(0.0, 1.0, 11).tolist()
    
    for dose in doses[1:]:
        print(f"\n>>> DOSE: {dose*100:.0f}% Petition Severity vs Control (0%) <<<", flush=True)
        
        # Calculate true Epistemic Uncertainty by evaluating ATEs across the 500 parallel universes
        surv_universes = (results[dose]["surv"] - results[ctrl_d]["surv"]).mean(axis=0)  # [500]
        vote_universes = (results[dose]["vote"] - results[ctrl_d]["vote"]).mean(axis=0)
        ht_universes = (results[dose]["ht"] - results[ctrl_d]["ht"]).mean(axis=0)
        tok_universes = (results[dose]["tok"] - results[ctrl_d]["tok"]).mean(axis=0)
        
        print(f"  P(Resolved):  Control={results[ctrl_d]['surv'].mean():.4f} | Treated={results[dose]['surv'].mean():.4f} | ATE: {surv_universes.mean():+.4f} [{np.percentile(surv_universes, 2.5):+.4f}, {np.percentile(surv_universes, 97.5):+.4f}]")
        print(f"  P(Approval):  Control={results[ctrl_d]['vote'].mean():.4f} | Treated={results[dose]['vote'].mean():.4f} | ATE: {vote_universes.mean():+.4f} [{np.percentile(vote_universes, 2.5):+.4f}, {np.percentile(vote_universes, 97.5):+.4f}]")
        print(f"  Height (ft):  Control={results[ctrl_d]['ht'].mean():.2f} | Treated={results[dose]['ht'].mean():.2f} | ATE: {ht_universes.mean():+.2f} ft [{np.percentile(ht_universes, 2.5):+.2f}, {np.percentile(ht_universes, 97.5):+.2f}]")
        print(f"  Total Tokens: Control={results[ctrl_d]['tok'].mean():.1f} | Treated={results[dose]['tok'].mean():.1f} | ATE: {tok_universes.mean():+.1f} toks [{np.percentile(tok_universes, 2.5):+.1f}, {np.percentile(tok_universes, 97.5):+.1f}]")
    
    print("\n" + "=" * 60, flush=True)
    
    # We will save the dose-response matrix directly to a new CSV
    flattened_data = []
    for d in results.keys():
        flattened_data.append(pd.DataFrame({
            "dose": d,
            "surv_mean": results[d]["surv"].mean(axis=1),
            "vote_mean": results[d]["vote"].mean(axis=1),
            "ht_mean": results[d]["ht"].mean(axis=1),
            "tok_mean": results[d]["tok"].mean(axis=1)
        }))
    df_results = pd.concat(flattened_data)
    
    out_csv = rf"{OUT_DIR}\vae_dose_response_surface.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved dose-response matrix to {out_csv}", flush=True)

if __name__ == "__main__":
    main()
