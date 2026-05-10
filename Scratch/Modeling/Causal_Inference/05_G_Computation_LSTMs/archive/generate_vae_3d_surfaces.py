"""
generate_vae_3d_surfaces.py
===========================
Generates 3D interactive HTML surfaces representing the causal effect of
neighborhood opposition on development project outcomes, using a
Conditional VAE -> LSTM hybrid architecture.

Outputs:
- causal_vae_3d_survival.html
- causal_vae_3d_height.html
"""

import os, math, re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from torch.utils.data import TensorDataset, DataLoader
import warnings
import functools

print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"

# Models save paths
VAE_PATH = os.path.join(OUT_DIR, "vae_model_3d.pt")
LSTM_PATH = os.path.join(OUT_DIR, "lstm_model_3d.pt")

N_SAMPLES = 200  # Latent samples per grid point

# ============================================================
# DATA PIPELINE
# ============================================================
def load_data():
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
    
    features = [
        "land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km",
        "local_unemployment_rate", "mortgage_rate_30yr", "period_seq", "petition_pct_this_period",
        "cumulative_petition_pct", "bw_sin", "bw_cos"
    ]
    targets = ["resolved", "proposed_max_height_ft", "commission_hearings_this_period", "council_hearings_this_period"]
    
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
    
    feat_vals = df[features].values.astype(np.float32)
    targ_vals = df[targets].values.astype(np.float32)
    
    idx = 0
    for i, case in enumerate(cases):
        size = case_sizes[case]
        length = min(size, max_seq)
        X_all[i, :length, :] = feat_vals[idx:idx+length]
        Y_all[i, :length, :] = targ_vals[idx:idx+length]
        idx += size
    
    return torch.from_numpy(X_all), torch.from_numpy(Y_all)

# ============================================================
# ARCHITECTURE
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
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_proj = nn.Linear(input_dim, d_model)
        self.enc_pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                              dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        
        self.dec_proj = nn.Linear(latent_dim, d_model)
        self.dec_pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                              dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)
    
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
        z_seq = z.unsqueeze(1).expand(-1, seq_len, -1)
        h = self.dec_pos(self.dec_proj(z_seq))
        h = self.decoder(h)
        return self.output_proj(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x.size(1)), mu, logvar

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, 1, batch_first=True)
        self.head_surv = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_ht   = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_comm = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_coun = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x):
        h, _ = self.lstm(x)
        return torch.cat([self.head_surv(h), self.head_ht(h), self.head_comm(h), self.head_coun(h)], dim=-1)

# ============================================================
# TRAINING
# ============================================================
def train_models(X, Y, features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    Y = Y.to(device)
    
    vae = ConditionalVAE(len(features)).to(device)
    lstm = MultiTaskLSTM(len(features)).to(device)
    
    # Train VAE
    print("[*] Training VAE (40 epochs)...")
    opt_vae = torch.optim.Adam(vae.parameters(), lr=0.001)
    loader_vae = DataLoader(TensorDataset(X), batch_size=256, shuffle=True)
    
    vae.train()
    for epoch in range(40):
        for (bx,) in loader_vae:
            opt_vae.zero_grad()
            recon, mu, logvar = vae(bx)
            mask = (bx[:, :, -1] != 0).float().unsqueeze(-1)
            recon_loss = (((recon - bx) ** 2) * mask).sum() / mask.sum()
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            (recon_loss + 0.0005 * kl_loss).backward()
            opt_vae.step()
    torch.save(vae.state_dict(), VAE_PATH)
    
    # Train LSTM
    print("[*] Training LSTM (30 epochs)...")
    opt_lstm = torch.optim.Adam(lstm.parameters(), lr=0.005)
    loader_lstm = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=True)
    crit_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0], device=device), reduction='none')
    crit_mse = nn.MSELoss(reduction='none')
    
    lstm.train()
    for epoch in range(30):
        for bx, by in loader_lstm:
            opt_lstm.zero_grad()
            preds = lstm(bx)
            mask = (bx[:, :, -1] != 0).float()
            l1 = (crit_bce(preds[:, :, 0], by[:, :, 0]) * mask).sum() / mask.sum()
            l2 = (crit_mse(preds[:, :, 1], by[:, :, 1]) * mask).sum() / mask.sum()
            l3 = (crit_mse(preds[:, :, 2], by[:, :, 2]) * mask).sum() / mask.sum()
            l4 = (crit_mse(preds[:, :, 3], by[:, :, 3]) * mask).sum() / mask.sum()
            (l1 + l2 + l3 + l4).backward()
            opt_lstm.step()
    torch.save(lstm.state_dict(), LSTM_PATH)
    
    return vae.cpu(), lstm.cpu()

# ============================================================
# CAUSAL SURFACE GENERATION
# ============================================================
def generate_surfaces(vae, lstm, archetype_tensor, features, norm_dict):
    vae.eval()
    lstm.eval()
    
    pet_idx = features.index("petition_pct_this_period")
    cum_pet_idx = features.index("cumulative_petition_pct")
    
    # Generate latent base
    with torch.no_grad():
        mu, logvar = vae.encode(archetype_tensor)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(N_SAMPLES, mu.size(1))
        z_samples = mu + eps * std  # [N, latent_dim]
        base_trajs = vae.decode(z_samples, seq_len=30)  # [N, 30, F]
    
    # Grid definition
    eras = np.arange(1, 16)  # Intervention timing: Step 1 to 15
    intensities = np.linspace(0.0, 1.0, 20)  # Petition %
    
    grid_surv_mean = np.zeros((len(eras), len(intensities)))
    grid_surv_025  = np.zeros((len(eras), len(intensities)))
    grid_surv_975  = np.zeros((len(eras), len(intensities)))
    
    grid_ht_mean = np.zeros((len(eras), len(intensities)))
    grid_ht_025  = np.zeros((len(eras), len(intensities)))
    grid_ht_975  = np.zeros((len(eras), len(intensities)))
    
    print("[*] Generating 3D Surface Data...")
    
    for i, t in enumerate(eras):
        for j, p in enumerate(intensities):
            # Apply intervention to all N base trajectories
            treated_trajs = base_trajs.clone()
            
            # Intervention: From time t onward, set petition
            treated_trajs[:, t:, pet_idx] = p
            # Calculate cumulative properly
            cum_pet = torch.cumsum(treated_trajs[:, :, pet_idx], dim=1)
            treated_trajs[:, :, cum_pet_idx] = cum_pet
            
            # Predict
            with torch.no_grad():
                preds = lstm(treated_trajs)  # [N, 30, 4]
                
            # Extract final step outcomes
            surv_probs = torch.sigmoid(preds[:, -1, 0]).numpy()
            ht_preds = preds[:, -1, 1].numpy()
            
            # Denormalize height
            mean_ht, std_ht = norm_dict["proposed_max_height_ft"]
            ht_preds_ft = ht_preds * std_ht + mean_ht
            
            # Calculate net change from initially requested height (t=0)
            initial_ht_norm = treated_trajs[:, 0, features.index("proposed_max_height_ft")].numpy()
            initial_ht_ft = initial_ht_norm * std_ht + mean_ht
            net_change_ft = ht_preds_ft - initial_ht_ft
            
            # Statistics
            grid_surv_mean[i, j] = np.mean(surv_probs)
            grid_surv_025[i, j] = np.percentile(surv_probs, 2.5)
            grid_surv_975[i, j] = np.percentile(surv_probs, 97.5)
            
            grid_ht_mean[i, j] = np.mean(net_change_ft)
            grid_ht_025[i, j] = np.percentile(net_change_ft, 2.5)
            grid_ht_975[i, j] = np.percentile(net_change_ft, 97.5)
            
    return eras, intensities, grid_surv_mean, grid_surv_025, grid_surv_975, grid_ht_mean, grid_ht_025, grid_ht_975

def plot_3d_surface(x, y, z_mean, z_lower, z_upper, title, z_label, filename, colorscale):
    X, Y = np.meshgrid(y, x)  # x=eras, y=intensities
    
    fig = go.Figure()
    
    # Upper Bound (Translucent)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=z_upper,
        showscale=False, opacity=0.3, colorscale='Greys',
        name="97.5% Credible Interval"
    ))
    
    # Lower Bound (Translucent)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=z_lower,
        showscale=False, opacity=0.3, colorscale='Greys',
        name="2.5% Credible Interval"
    ))
    
    # Mean Surface (Solid)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=z_mean,
        colorscale=colorscale, opacity=1.0,
        name="Mean Outcome",
        colorbar=dict(title=z_label)
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Intensity (% Petitions)',
            yaxis_title='Timing (Bi-Weekly Periods)',
            zaxis_title=z_label,
            camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    out_path = os.path.join(OUT_DIR, filename)
    fig.write_html(out_path)
    print(f"Saved {filename}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("[1/4] Loading and preparing data...")
    df, features, targets, norm_dict = load_data()
    X_all, Y_all = build_tensors(df, features, targets)
    
    vae = ConditionalVAE(len(features))
    lstm = MultiTaskLSTM(len(features))
    
    if os.path.exists(VAE_PATH) and os.path.exists(LSTM_PATH):
        print("[2/4] Loading pre-trained models from disk...")
        vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
        lstm.load_state_dict(torch.load(LSTM_PATH, map_location='cpu'))
    else:
        print("[2/4] Training models from scratch...")
        vae, lstm = train_models(X_all, Y_all, features)
    
    print("[3/4] Establishing Archetype Case...")
    # Find a median case: ~2 acres, near mean gravity
    median_acre = df["land_acres"].median()
    df["acre_dist"] = (df["land_acres"] - median_acre).abs()
    archetype_case = df.sort_values("acre_dist").iloc[0]["case_number"]
    arch_df = df[df["case_number"] == archetype_case].sort_values("period_seq")
    arch_seq = arch_df[features].values
    if len(arch_seq) < 30:
        pad = np.zeros((30 - len(arch_seq), len(features)))
        arch_seq = np.vstack([arch_seq, pad])
    arch_tensor = torch.tensor(arch_seq, dtype=torch.float32).unsqueeze(0)  # [1, 30, F]
    
    print("[4/4] Generating Distributional Grids...")
    e, i, s_mean, s_025, s_975, h_mean, h_025, h_975 = generate_surfaces(
        vae, lstm, arch_tensor, features, norm_dict
    )
    
    print("[*] Plotting Surfaces...")
    plot_3d_surface(
        e, i, s_mean, s_025, s_975,
        title="Distributional Causal Surface: P(Survival) vs NIMBY Friction",
        z_label="Probability of Survival",
        filename="causal_vae_3d_survival.html",
        colorscale="Viridis"
    )
    
    plot_3d_surface(
        e, i, h_mean, h_025, h_975,
        title="Distributional Causal Surface: Net Height Reduction vs NIMBY Friction",
        z_label="Net Height Change (Feet)",
        filename="causal_vae_3d_height.html",
        colorscale="Magma"
    )

if __name__ == "__main__":
    main()
