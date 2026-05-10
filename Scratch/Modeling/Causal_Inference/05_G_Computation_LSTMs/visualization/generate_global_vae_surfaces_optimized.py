"""
generate_global_vae_surfaces_optimized.py
=========================================
CUDA-optimized script to generate the Unbiased Global ATE Causal Surface
across all 1,290 test cases in under 60 seconds.
"""

import os, math, time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import functools

print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"

VAE_PATH = os.path.join(OUT_DIR, "vae_model_3d.pt")
LSTM_PATH = os.path.join(OUT_DIR, "lstm_model_3d.pt")

N_SAMPLES = 50  # 50 samples is plenty for a 1290-case global ensemble
BATCH_SIZE = 128  # Number of cases to process concurrently on GPU

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

def build_test_tensors(df, features, max_seq=30):
    df = df.sort_values(["case_number", "period_seq"])
    case_years = df.groupby("case_number")["year"].min()
    
    test_cases = case_years[case_years >= 2019].index.values
    
    test_df = df[df["case_number"].isin(test_cases)]
    case_sizes = test_df.groupby("case_number").size()
    n_cases = len(test_cases)
    
    X_test = np.zeros((n_cases, max_seq, len(features)), dtype=np.float32)
    feat_vals = test_df[features].values.astype(np.float32)
    
    idx = 0
    for i, case in enumerate(test_cases):
        size = case_sizes[case]
        length = min(size, max_seq)
        X_test[i, :length, :] = feat_vals[idx:idx+length]
        idx += size
        
    return torch.from_numpy(X_test)

# ============================================================
# ARCHITECTURE (Copied precisely to load weights)
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
    
    def decode(self, z, seq_len=30):
        z_seq = z.unsqueeze(1).expand(-1, seq_len, -1)
        h = self.dec_pos(self.dec_proj(z_seq))
        h = self.decoder(h)
        return self.output_proj(h)

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
# CUDA BATCHED GENERATOR
# ============================================================
def generate_global_surfaces(vae, lstm, X_test, features, norm_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running heavily vectorized inference on {device}...")
    
    vae = vae.to(device)
    lstm = lstm.to(device)
    vae.eval()
    lstm.eval()
    
    pet_idx = features.index("petition_pct_this_period")
    cum_pet_idx = features.index("cumulative_petition_pct")
    ht_idx = features.index("proposed_max_height_ft")
    
    eras = np.arange(1, 16)
    intensities = np.linspace(0.0, 1.0, 20)
    
    n_eras = len(eras)
    n_ints = len(intensities)
    
    global_surv_mean = np.zeros((n_eras, n_ints))
    global_surv_sq = np.zeros((n_eras, n_ints))  # For variance across cases
    
    global_ht_mean = np.zeros((n_eras, n_ints))
    global_ht_sq = np.zeros((n_eras, n_ints))
    
    total_cases = X_test.size(0)
    
    # Pre-build the grid coordinates for fast indexing
    grid_e = torch.tensor(eras, device=device)
    grid_i = torch.tensor(intensities, device=device, dtype=torch.float32)
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_start in range(0, total_cases, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_cases)
            bx = X_test[batch_start:batch_end].to(device)
            B = bx.size(0)
            
            # 1. Encode the test cases
            mu, logvar = vae.encode(bx)
            std = torch.exp(0.5 * logvar)
            
            # 2. Sample N latents
            eps = torch.randn(B, N_SAMPLES, mu.size(1), device=device)
            z_samples = mu.unsqueeze(1) + eps * std.unsqueeze(1)  # [B, N, latent_dim]
            
            # Decode all samples in a flat batch
            z_flat = z_samples.view(B * N_SAMPLES, -1)
            base_trajs_flat = vae.decode(z_flat, seq_len=30)  # [B*N, 30, F]
            base_trajs = base_trajs_flat.view(B, N_SAMPLES, 30, len(features))  # [B, N, 30, F]
            
            # Store initial heights for calculating net reduction
            initial_ht_norm = base_trajs[:, :, 0, ht_idx]  # [B, N]
            
            # We want to evaluate all 300 grid points. 
            # Doing [B, 300, N, 30, F] would OOM the GPU.
            # So we loop over the 300 grid points, but vectorizing across the B cases and N samples.
            for e_idx, t in enumerate(eras):
                for i_idx, p in enumerate(intensities):
                    # Clone base trajectories
                    treated_trajs = base_trajs.clone()
                    
                    # Apply intervention
                    treated_trajs[:, :, t:, pet_idx] = p
                    cum_pet = torch.cumsum(treated_trajs[:, :, :, pet_idx], dim=2)
                    treated_trajs[:, :, :, cum_pet_idx] = cum_pet
                    
                    # Flat forward pass through LSTM
                    treated_flat = treated_trajs.view(B * N_SAMPLES, 30, len(features))
                    preds_flat = lstm(treated_flat)
                    preds = preds_flat.view(B, N_SAMPLES, 30, 4)
                    
                    # Final step predictions
                    surv_probs = torch.sigmoid(preds[:, :, -1, 0])  # [B, N]
                    ht_preds = preds[:, :, -1, 1]  # [B, N]
                    
                    # Calculate net height change in normalized space, then denormalize
                    net_change_norm = ht_preds - initial_ht_norm
                    mean_ht, std_ht = norm_dict["proposed_max_height_ft"]
                    net_change_ft = net_change_norm * std_ht
                    
                    # Average over the N samples to get the expected outcome per case
                    case_surv_mean = surv_probs.mean(dim=1).cpu().numpy()  # [B]
                    case_ht_mean = net_change_ft.mean(dim=1).cpu().numpy()  # [B]
                    
                    # Accumulate global statistics (we compute variance later using E[X^2] - E[X]^2)
                    global_surv_mean[e_idx, i_idx] += np.sum(case_surv_mean)
                    global_surv_sq[e_idx, i_idx] += np.sum(case_surv_mean ** 2)
                    
                    global_ht_mean[e_idx, i_idx] += np.sum(case_ht_mean)
                    global_ht_sq[e_idx, i_idx] += np.sum(case_ht_mean ** 2)
            
            elapsed = time.time() - start_time
            print(f"  Processed {batch_end}/{total_cases} cases... ({elapsed:.1f}s)")
            
    # Finalize Global ATE and Standard Deviations
    global_surv_mean /= total_cases
    global_surv_var = (global_surv_sq / total_cases) - (global_surv_mean ** 2)
    global_surv_std = np.sqrt(np.maximum(global_surv_var, 0))
    
    global_ht_mean /= total_cases
    global_ht_var = (global_ht_sq / total_cases) - (global_ht_mean ** 2)
    global_ht_std = np.sqrt(np.maximum(global_ht_var, 0))
    
    # 95% Confidence Intervals for the Mean (roughly +/- 1.96 * std)
    s_025 = global_surv_mean - 1.96 * global_surv_std
    s_975 = global_surv_mean + 1.96 * global_surv_std
    
    h_025 = global_ht_mean - 1.96 * global_ht_std
    h_975 = global_ht_mean + 1.96 * global_ht_std

    return eras, intensities, global_surv_mean, s_025, s_975, global_ht_mean, h_025, h_975

def plot_3d_surface(x, y, z_mean, z_lower, z_upper, title, z_label, filename, colorscale):
    X, Y = np.meshgrid(y, x)
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=z_upper,
        showscale=False, opacity=0.3, colorscale='Greys',
        name="97.5% Confidence"
    ))
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=z_lower,
        showscale=False, opacity=0.3, colorscale='Greys',
        name="2.5% Confidence"
    ))
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=z_mean,
        colorscale=colorscale, opacity=1.0,
        name="Mean ATE",
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

def main():
    print("[1/3] Loading Data and Pre-Trained Models...")
    df, features, targets, norm_dict = load_data()
    X_test = build_test_tensors(df, features)
    
    vae = ConditionalVAE(len(features))
    lstm = MultiTaskLSTM(len(features))
    vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    lstm.load_state_dict(torch.load(LSTM_PATH, map_location='cpu'))
    
    print(f"[2/3] Generating Global ATE Surface (1,290 cases)...")
    e, i, s_mean, s_025, s_975, h_mean, h_025, h_975 = generate_global_surfaces(
        vae, lstm, X_test, features, norm_dict
    )
    
    print("[3/3] Plotting Global Surfaces...")
    plot_3d_surface(
        e, i, s_mean, s_025, s_975,
        title="Unbiased Global ATE: P(Survival) vs NIMBY Friction",
        z_label="Global Probability of Survival",
        filename="causal_vae_3d_survival_GLOBAL.html",
        colorscale="Viridis"
    )
    
    plot_3d_surface(
        e, i, h_mean, h_025, h_975,
        title="Unbiased Global ATE: Net Height Reduction vs NIMBY Friction",
        z_label="Global Net Height Change (Feet)",
        filename="causal_vae_3d_height_GLOBAL.html",
        colorscale="Magma"
    )

if __name__ == "__main__":
    main()
