import os, math, time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = os.path.join(OUT_DIR, "biweekly_panel.csv")
MODEL_DIR = os.path.join(OUT_DIR, "saved_models")

SEEDS = [42, 123]
N_SAMPLES_PER_SEED = 100
YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

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
        self.enc_proj = nn.Linear(input_dim, d_model)
        self.enc_pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.dec_proj = nn.Linear(latent_dim, d_model)
        self.dec_pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
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
        self.head_with = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_aff  = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x):
        h, _ = self.lstm(x)
        return torch.cat([self.head_surv(h), self.head_ht(h), self.head_comm(h), self.head_coun(h), self.head_with(h), self.head_aff(h)], dim=-1)

def get_normalization_stats():
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    features = [
        "land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km",
        "local_unemployment_rate", "mortgage_rate_30yr", "period_seq", "petition_pct_this_period",
        "cumulative_petition_pct", "bw_sin", "bw_cos"
    ]
    norm_dict = {}
    for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km",
              "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        norm_dict[f] = (df[f].fillna(0).mean(), df[f].fillna(0).std())
    return features, norm_dict

def get_year_data(year, features, norm_dict, X_grid, Y_grid, ht_range, dose_range):
    n_features = len(features)
    pet_idx = features.index("petition_pct_this_period")
    cum_pet_idx = features.index("cumulative_petition_pct")
    ht_idx = features.index("proposed_max_height_ft")
    mean_ht, std_ht = norm_dict["proposed_max_height_ft"]
    mean_aff, std_aff = norm_dict.get("affordability_proxy", (0, 1))
    
    Z_ht_samples = np.zeros((len(SEEDS) * N_SAMPLES_PER_SEED, len(dose_range), len(ht_range)))
    Z_surv_samples = np.zeros((len(SEEDS) * N_SAMPLES_PER_SEED, len(dose_range), len(ht_range)))
    Z_comm_samples = np.zeros((len(SEEDS) * N_SAMPLES_PER_SEED, len(dose_range), len(ht_range)))
    Z_coun_samples = np.zeros((len(SEEDS) * N_SAMPLES_PER_SEED, len(dose_range), len(ht_range)))
    Z_with_samples = np.zeros((len(SEEDS) * N_SAMPLES_PER_SEED, len(dose_range), len(ht_range)))
    Z_aff_samples = np.zeros((len(SEEDS) * N_SAMPLES_PER_SEED, len(dose_range), len(ht_range)))
    
    seq_len = 30
    base_traj = torch.zeros((1, seq_len, n_features), device=DEVICE)
    for f, idx in zip(features, range(n_features)):
        if f in norm_dict:
            base_traj[0, :, idx] = (0.0 - norm_dict[f][0]) / (norm_dict[f][1] + 1e-8)
    base_traj[0, :, features.index("period_seq")] = torch.arange(seq_len) / 10.0
    
    sample_idx = 0
    for seed in SEEDS:
        vae = ConditionalVAE(n_features).to(DEVICE)
        lstm = MultiTaskLSTM(n_features).to(DEVICE)
        vae.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"vae_{year}_seed{seed}.pt"), map_location=DEVICE))
        lstm.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"lstm_{year}_seed{seed}.pt"), map_location=DEVICE))
        vae.eval()
        lstm.eval()
        
        with torch.no_grad():
            mu, logvar = vae.encode(base_traj)
            std = torch.exp(0.5 * logvar)
            
            for _ in range(N_SAMPLES_PER_SEED):
                eps = torch.randn_like(std)
                z = mu + eps * std
                decoded_base = vae.decode(z, seq_len)
                
                grid_size = len(dose_range) * len(ht_range)
                batch_traj = decoded_base.expand(grid_size, -1, -1).clone()
                
                ht_flat = X_grid.flatten()
                dose_flat = Y_grid.flatten()
                ht_norm_flat = (ht_flat - mean_ht) / (std_ht + 1e-8)
                
                batch_traj[:, :, ht_idx] = torch.tensor(ht_norm_flat, device=DEVICE).unsqueeze(1)
                batch_traj[:, 5:, pet_idx] = torch.tensor(dose_flat, device=DEVICE).unsqueeze(1)
                batch_traj[:, :, cum_pet_idx] = torch.cumsum(batch_traj[:, :, pet_idx], dim=1)
                
                preds = lstm(batch_traj)
                
                surv_probs = torch.sigmoid(preds[:, -1, 0]).cpu().numpy()
                final_hts_norm = preds[:, -1, 1].cpu().numpy()
                final_hts_ft = final_hts_norm * std_ht + mean_ht
                ates_ht = final_hts_ft - ht_flat
                
                comm_preds = preds[:, -1, 2].cpu().numpy()
                coun_preds = preds[:, -1, 3].cpu().numpy()
                with_probs = torch.sigmoid(preds[:, -1, 4]).cpu().numpy()
                aff_preds = preds[:, -1, 5].cpu().numpy() * std_aff + mean_aff
                
                Z_ht_samples[sample_idx] = ates_ht.reshape(len(dose_range), len(ht_range))
                Z_surv_samples[sample_idx] = surv_probs.reshape(len(dose_range), len(ht_range))
                Z_comm_samples[sample_idx] = comm_preds.reshape(len(dose_range), len(ht_range))
                Z_coun_samples[sample_idx] = coun_preds.reshape(len(dose_range), len(ht_range))
                Z_with_samples[sample_idx] = with_probs.reshape(len(dose_range), len(ht_range))
                Z_aff_samples[sample_idx] = aff_preds.reshape(len(dose_range), len(ht_range))
                
                sample_idx += 1
                
        del vae, lstm
        torch.cuda.empty_cache()
        
    def make_dict(arr):
        return {"median": np.median(arr, axis=0), "p25": np.percentile(arr, 25, axis=0), "p75": np.percentile(arr, 75, axis=0)}
        
    return make_dict(Z_ht_samples), make_dict(Z_surv_samples), make_dict(Z_comm_samples), \
           make_dict(Z_coun_samples), make_dict(Z_with_samples), make_dict(Z_aff_samples)

def create_animated_plot(X, Y, data_dicts, title, z_label, z_range, filename):
    fig = go.Figure()

    # Initial Traces (2015)
    first_year = YEARS[0]
    initial_data = data_dicts[first_year]
    
    fig.add_trace(go.Surface(z=initial_data["median"], x=X, y=Y, colorscale='Viridis', name='Median', opacity=1.0))
    fig.add_trace(go.Surface(z=initial_data["p75"], x=X, y=Y, colorscale=[[0, 'lightgray'], [1, 'white']], showscale=False, opacity=0.3, name='p75'))
    fig.add_trace(go.Surface(z=initial_data["p25"], x=X, y=Y, colorscale=[[0, 'black'], [1, 'darkgray']], showscale=False, opacity=0.3, name='p25'))

    # Frames
    frames = []
    for year in YEARS:
        d = data_dicts[year]
        frame = go.Frame(
            data=[
                go.Surface(z=d["median"]),
                go.Surface(z=d["p75"]),
                go.Surface(z=d["p25"])
            ],
            name=str(year)
        )
        frames.append(frame)
    
    fig.frames = frames

    # Animation controls
    sliders = [dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(font=dict(size=20), prefix="Evaluation Year: ", visible=True, xanchor="right"),
        transition=dict(duration=500, easing="cubic-in-out"),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.1,
        y=0,
        steps=[dict(args=[[str(y)], dict(frame=dict(duration=500, redraw=True), mode="immediate", transition=dict(duration=500))],
                    label=str(y), method="animate") for y in YEARS]
    )]

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Initial Requested Height (ft)',
            yaxis_title='Neighborhood Petition Dose (%)',
            zaxis_title=z_label,
            zaxis=dict(range=z_range),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, transition=dict(duration=500))]),
                dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
            ],
            direction="left", pad=dict(r=10, t=87), showactive=False, x=0.1, xanchor="right", y=0, yanchor="top"
        )],
        sliders=sliders,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    filepath = os.path.join(OUT_DIR, filename)
    fig.write_html(filepath, auto_play=False)
    print(f"Saved animation to {filepath}")

def main():
    print("Loading data...")
    features, norm_dict = get_normalization_stats()
    
    ht_range = np.linspace(40, 400, 20)
    dose_range = np.linspace(0.0, 1.0, 20)
    X_grid, Y_grid = np.meshgrid(ht_range, dose_range)
    
    ht_dicts, surv_dicts, comm_dicts, coun_dicts, with_dicts, aff_dicts = {}, {}, {}, {}, {}, {}
    
    ht_global_min, ht_global_max = float('inf'), float('-inf')
    surv_global_min, surv_global_max = float('inf'), float('-inf')
    comm_global_min, comm_global_max = float('inf'), float('-inf')
    coun_global_min, coun_global_max = float('inf'), float('-inf')
    with_global_min, with_global_max = float('inf'), float('-inf')
    aff_global_min, aff_global_max = float('inf'), float('-inf')
    
    for year in YEARS:
        t0 = time.time()
        print(f"Computing grid for {year}...")
        ht_d, surv_d, comm_d, coun_d, with_d, aff_d = get_year_data(year, features, norm_dict, X_grid, Y_grid, ht_range, dose_range)
        ht_dicts[year] = ht_d
        surv_dicts[year] = surv_d
        comm_dicts[year] = comm_d
        coun_dicts[year] = coun_d
        with_dicts[year] = with_d
        aff_dicts[year] = aff_d
        
        # Track global min/max for tighter dynamic Z-axis scaling
        ht_global_min = min(ht_global_min, np.min(ht_d["p25"]))
        ht_global_max = max(ht_global_max, np.max(ht_d["p75"]))
        surv_global_min = min(surv_global_min, np.min(surv_d["p25"]))
        surv_global_max = max(surv_global_max, np.max(surv_d["p75"]))
        comm_global_min = min(comm_global_min, np.min(comm_d["p25"]))
        comm_global_max = max(comm_global_max, np.max(comm_d["p75"]))
        coun_global_min = min(coun_global_min, np.min(coun_d["p25"]))
        coun_global_max = max(coun_global_max, np.max(coun_d["p75"]))
        with_global_min = min(with_global_min, np.min(with_d["p25"]))
        with_global_max = max(with_global_max, np.max(with_d["p75"]))
        aff_global_min = min(aff_global_min, np.min(aff_d["p25"]))
        aff_global_max = max(aff_global_max, np.max(aff_d["p75"]))
        
        print(f"  -> Done in {time.time()-t0:.1f}s")
        
    print("Rendering HTML animations...")
    create_animated_plot(X_grid, Y_grid, ht_dicts, 
                         "Animated Density Bonus Paradox: Height ATE (2015-2022)", 
                         "Net Change in Height (ft)", [ht_global_min, ht_global_max],
                         "animated_causal_height_surface.html")
                         
    create_animated_plot(X_grid, Y_grid, surv_dicts, 
                         "Animated Survival Probability (2015-2022)", 
                         "Probability of Survival", [surv_global_min, surv_global_max],
                         "animated_causal_survival_surface.html")
                         
    create_animated_plot(X_grid, Y_grid, comm_dicts, 
                         "Animated Commission Hearings ATE (2015-2022)", 
                         "Additional Commission Hearings", [comm_global_min, comm_global_max],
                         "animated_causal_commission_surface.html")
                         
    create_animated_plot(X_grid, Y_grid, coun_dicts, 
                         "Animated Council Hearings ATE (2015-2022)", 
                         "Additional Council Hearings", [coun_global_min, coun_global_max],
                         "animated_causal_council_surface.html")
                         
    create_animated_plot(X_grid, Y_grid, with_dicts, 
                         "Animated Withdrawal Probability (2015-2022)", 
                         "Probability of Withdrawal", [with_global_min, with_global_max],
                         "animated_causal_withdrawal_surface.html")
                         
    create_animated_plot(X_grid, Y_grid, aff_dicts, 
                         "Animated Affordability Proxy (2015-2022)", 
                         "Affordability Proxy Score", [aff_global_min, aff_global_max],
                         "animated_causal_affordability_surface.html")

if __name__ == "__main__":
    main()
