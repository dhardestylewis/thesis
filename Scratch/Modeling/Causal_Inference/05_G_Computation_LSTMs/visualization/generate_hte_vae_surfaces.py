"""
generate_hte_vae_surfaces.py
============================
Generates Heterogeneous Treatment Effect (HTE) 3D Causal Surfaces
by empirically binning test cases across key structural covariates.
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

N_SAMPLES = 50
INTERVENTION_TIMING = 5  # Fix intervention at biweekly period 5

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
        
    # Store raw unnormalized values for the X-axis labels
    raw_dict = {
        "delta_max_height_ft": {},   # filled below from enriched delta CSV
        "mortgage_rate_30yr": df.groupby("case_number")["mortgage_rate_30yr"].first().to_dict(),
        "knn_petition_rate_1km": df.groupby("case_number")["knn_petition_rate_1km"].first().to_dict()
    }
    
    # Load enriched upzone delta (temporally-routed LDC join)
    delta_path = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\upzone_delta_enriched.csv"
    delta_df = pd.read_csv(delta_path)
    raw_dict["delta_max_height_ft"] = delta_df.set_index("case_number")["delta_max_height_ft"].to_dict()
    
    norm_dict = {}
    for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km",
              "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        mean_v = df[f].mean()
        std_v = df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)
    
    return df, features, targets, norm_dict, raw_dict

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
        
    return torch.from_numpy(X_test), test_cases

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
# HTE GENERATOR
# ============================================================
def generate_hte_surfaces(vae, lstm, X_test, test_cases, features, norm_dict, raw_dict, target_feat, n_bins=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)
    lstm = lstm.to(device)
    vae.eval()
    lstm.eval()
    
    print(f"\n[*] Generating HTE Surface for: {target_feat}")
    
    # 1. Empirical Binning — filter to cases with a valid, non-NaN value
    raw_vals_all = np.array([raw_dict[target_feat].get(c, np.nan) for c in test_cases])
    valid_mask = np.isfinite(raw_vals_all)
    
    # For delta-style features (many zeros), also restrict to non-zero to get meaningful bins
    if target_feat == "delta_max_height_ft":
        valid_mask = valid_mask & (raw_vals_all != 0)
    
    raw_vals = raw_vals_all[valid_mask]
    X_test_valid = X_test[valid_mask]
    test_cases_valid = test_cases[valid_mask]
    
    print(f"  Valid cases for binning: {len(raw_vals)} / {len(test_cases)}")
    
    if len(raw_vals) < n_bins * 2:
        print(f"  WARNING: Not enough cases ({len(raw_vals)}) for {n_bins} bins. Skipping.")
        return None, None, None, None, None, None, None, None
    
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(raw_vals, quantiles)
    bin_edges = np.unique(bin_edges)
    actual_bins = len(bin_edges) - 1
    bin_indices = np.digitize(raw_vals, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, actual_bins - 1)
    
    bin_centers = np.array([(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(actual_bins)])
    intensities = np.linspace(0.0, 1.0, 20)
    
    grid_surv_mean = np.zeros((actual_bins, len(intensities)))
    grid_surv_sq = np.zeros((actual_bins, len(intensities)))
    
    grid_ht_mean = np.zeros((actual_bins, len(intensities)))
    grid_ht_sq = np.zeros((actual_bins, len(intensities)))
    
    bin_counts = np.zeros(actual_bins)
    
    pet_idx = features.index("petition_pct_this_period")
    cum_pet_idx = features.index("cumulative_petition_pct")
    ht_idx = features.index("proposed_max_height_ft")
    
    with torch.no_grad():
        for b_idx in range(actual_bins):
            case_mask = (bin_indices == b_idx)
            bx = X_test_valid[case_mask].to(device)
            B = bx.size(0)
            
            if B == 0: continue
            bin_counts[b_idx] = B
            
            mu, logvar = vae.encode(bx)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(B, N_SAMPLES, mu.size(1), device=device)
            z_samples = mu.unsqueeze(1) + eps * std.unsqueeze(1)
            
            z_flat = z_samples.view(B * N_SAMPLES, -1)
            base_trajs_flat = vae.decode(z_flat, seq_len=30)
            base_trajs = base_trajs_flat.view(B, N_SAMPLES, 30, len(features))
            
            initial_ht_norm = base_trajs[:, :, 0, ht_idx]
            
            for i_idx, p in enumerate(intensities):
                treated_trajs = base_trajs.clone()
                # Intervention at Fixed Timing (t=5)
                treated_trajs[:, :, INTERVENTION_TIMING:, pet_idx] = p
                cum_pet = torch.cumsum(treated_trajs[:, :, :, pet_idx], dim=2)
                treated_trajs[:, :, :, cum_pet_idx] = cum_pet
                
                treated_flat = treated_trajs.view(B * N_SAMPLES, 30, len(features))
                preds_flat = lstm(treated_flat)
                preds = preds_flat.view(B, N_SAMPLES, 30, 4)
                
                surv_probs = torch.sigmoid(preds[:, :, -1, 0])
                ht_preds = preds[:, :, -1, 1]
                
                net_change_norm = ht_preds - initial_ht_norm
                mean_ht, std_ht = norm_dict["proposed_max_height_ft"]
                net_change_ft = net_change_norm * std_ht
                
                case_surv_mean = surv_probs.mean(dim=1).cpu().numpy()
                case_ht_mean = net_change_ft.mean(dim=1).cpu().numpy()
                
                grid_surv_mean[b_idx, i_idx] = np.sum(case_surv_mean)
                grid_surv_sq[b_idx, i_idx] = np.sum(case_surv_mean ** 2)
                
                grid_ht_mean[b_idx, i_idx] = np.sum(case_ht_mean)
                grid_ht_sq[b_idx, i_idx] = np.sum(case_ht_mean ** 2)
                
    # Finalize stats
    valid_bins = bin_counts > 0
    
    # Filter arrays to only valid bins
    grid_surv_mean = grid_surv_mean[valid_bins] / bin_counts[valid_bins, None]
    grid_surv_sq = grid_surv_sq[valid_bins] / bin_counts[valid_bins, None]
    grid_surv_var = grid_surv_sq - (grid_surv_mean ** 2)
    global_surv_std = np.sqrt(np.maximum(grid_surv_var, 0))
    s_025 = grid_surv_mean - 1.96 * global_surv_std
    s_975 = grid_surv_mean + 1.96 * global_surv_std

    grid_ht_mean = grid_ht_mean[valid_bins] / bin_counts[valid_bins, None]
    grid_ht_sq = grid_ht_sq[valid_bins] / bin_counts[valid_bins, None]
    grid_ht_var = grid_ht_sq - (grid_ht_mean ** 2)
    global_ht_std = np.sqrt(np.maximum(grid_ht_var, 0))
    h_025 = grid_ht_mean - 1.96 * global_ht_std
    h_975 = grid_ht_mean + 1.96 * global_ht_std
    
    actual_bin_centers = bin_centers[valid_bins]
    
    return actual_bin_centers, intensities, grid_surv_mean, s_025, s_975, grid_ht_mean, h_025, h_975

def plot_3d_surface(x, y, z_mean, z_lower, z_upper, title, x_label, z_label, filename, colorscale):
    X, Y = np.meshgrid(y, x)  # x=bins, y=intensities
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
            yaxis_title=x_label,
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
    df, features, targets, norm_dict, raw_dict = load_data()
    X_test, test_cases = build_test_tensors(df, features)
    
    vae = ConditionalVAE(len(features))
    lstm = MultiTaskLSTM(len(features))
    vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    lstm.load_state_dict(torch.load(LSTM_PATH, map_location='cpu'))
    
    # Target Features for HTE
    targets_config = [
        ("delta_max_height_ft", "Upzone Ambition: Proposed Height Above Existing Zoning Entitlement (Feet)", "upzone_delta"),
        ("mortgage_rate_30yr", "Macroeconomic Climate at Time of Filing (30-Year Mortgage Rate)", "mortgage"),
        ("knn_petition_rate_1km", "Historical Neighborhood Opposition Rate Within 1km", "knn")
    ]
    
    for feat_id, feat_name, short_name in targets_config:
        result = generate_hte_surfaces(
            vae, lstm, X_test, test_cases, features, norm_dict, raw_dict, feat_id
        )
        if result[0] is None:
            print(f"  Skipping {feat_id} - insufficient valid cases.")
            continue
        x_bins, i, s_mean, s_025, s_975, h_mean, h_025, h_975 = result
        
        # Free VRAM between surface generations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        plot_3d_surface(
            x_bins, i, s_mean, s_025, s_975,
            title=f"HTE Surface: P(Survival) by {feat_name}",
            x_label=feat_name,
            z_label="Probability of Survival",
            filename=f"hte_{short_name}_survival.html",
            colorscale="Viridis"
        )
        
        plot_3d_surface(
            x_bins, i, h_mean, h_025, h_975,
            title=f"HTE Surface: Net Height Reduction by {feat_name}",
            x_label=feat_name,
            z_label="Net Height Change (Feet)",
            filename=f"hte_{short_name}_height.html",
            colorscale="Magma"
        )

if __name__ == "__main__":
    main()
