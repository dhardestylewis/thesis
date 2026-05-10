import os
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch.nn as nn

# Constants & Paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.environ.get("OUT_DIR", r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs")
PANEL_PATH = os.environ.get("PANEL_PATH", rf"{OUT_DIR}\biweekly_panel.csv")
VAE_PATH = rf"{OUT_DIR}\causal_vae_weights.pt"
LSTM_PATH = rf"{OUT_DIR}\causal_lstm_weights.pt"
N_SAMPLES = 500

# ============================================================
# ARCHITECTURES (Must match exactly to load weights)
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        import math
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
        return self.fc_mu(h.mean(dim=1)), self.fc_logvar(h.mean(dim=1))
        
    def decode(self, z, seq_len=30):
        z_seq = z.unsqueeze(1).expand(-1, seq_len, -1)
        h = self.dec_pos(self.dec_proj(z_seq))
        return self.output_proj(self.decoder(h))

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.head_surv = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_vote = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_ht   = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_tok  = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_comm = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_coun = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        
    def forward(self, x):
        h, _ = self.lstm(x)
        return torch.cat([self.head_surv(h), self.head_vote(h), self.head_ht(h), 
                          self.head_tok(h), self.head_comm(h), self.head_coun(h)], dim=-1)

# ============================================================
# MAIN
# ============================================================
def generate_3d_surfaces():
    print("[1/4] Loading models and baseline data...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    features = ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km",
                "local_unemployment_rate", "mortgage_rate_30yr", "period_seq", "petition_pct_this_period",
                "cumulative_petition_pct", "bw_sin", "bw_cos"]
                
    norm_dict = {}
    for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km",
              "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        norm_dict[f] = (df[f].mean(), df[f].std())

    if not os.path.exists(VAE_PATH) or not os.path.exists(LSTM_PATH):
        print(f"ERROR: Model weights not found at {VAE_PATH}. Awaiting S3 download.")
        return

    vae = ConditionalVAE(len(features)).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE, weights_only=True))
    lstm = MultiTaskLSTM(len(features)).to(DEVICE)
    lstm.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE, weights_only=True))
    vae.eval(); lstm.eval()

    print("[2/4] Constructing Synthetic Meshgrid Baseline...")
    heights = np.linspace(40, 400, 20)
    doses = np.linspace(0.0, 1.0, 20)
    
    # Base sequence: all features at their global mean
    base_seq = np.zeros((30, len(features)), dtype=np.float32)
    for i, f in enumerate(features):
        if f in norm_dict:
            base_seq[:, i] = (df[f].mean() - norm_dict[f][0]) / norm_dict[f][1]
    
    ps_idx = features.index("period_seq")
    for t in range(30):
        base_seq[t, ps_idx] = ((t+1) - norm_dict["period_seq"][0]) / norm_dict["period_seq"][1]

    base_tensor = torch.tensor(base_seq, device=DEVICE).unsqueeze(0) # [1, 30, F]
    
    pet_idx = features.index("petition_pct_this_period")
    cum_pet_idx = features.index("cumulative_petition_pct")
    ht_idx = features.index("proposed_max_height_ft")

    # Arrays to hold percentiles
    surv_p50 = np.zeros((20, 20))
    surv_p10 = np.zeros((20, 20))
    surv_p90 = np.zeros((20, 20))

    print("[3/4] Running MC-Dropout Inference across 400 Grid Coordinates...")
    with torch.no_grad():
        mu, logvar = vae.encode(base_tensor)
        std = torch.exp(0.5 * logvar)
        
        for i, ht in enumerate(heights):
            for j, dose in enumerate(doses):
                # 500 Samples for this grid coordinate
                eps = torch.randn(N_SAMPLES, mu.size(1), device=DEVICE)
                z_samples = mu + eps * std # [N, latent_dim]
                
                gen_trajs = vae.decode(z_samples, seq_len=30) # [N, 30, F]
                
                # Force inject the Meshgrid coordinate
                ht_norm = (ht - norm_dict["proposed_max_height_ft"][0]) / norm_dict["proposed_max_height_ft"][1]
                gen_trajs[:, :, ht_idx] = ht_norm
                
                gen_trajs[:, 4, pet_idx] = dose # Dose hits at period 5
                gen_trajs[:, 4:, cum_pet_idx] = dose
                
                preds = lstm(gen_trajs) # [N, 30, 6]
                
                # Calc survival prob
                h_surv = torch.sigmoid(preds[:, :, 0]) # [N, 30]
                P_surv = 1.0 - torch.prod(1.0 - h_surv, dim=1).cpu().numpy() # [N]
                
                surv_p50[i, j] = np.percentile(P_surv, 50)
                surv_p10[i, j] = np.percentile(P_surv, 10)
                surv_p90[i, j] = np.percentile(P_surv, 90)
                
            print(f"  > Processed Height Step {i+1}/20...")

    print("[4/4] Rendering 3D Plotly Surface...")
    import plotly.graph_objects as go
    import webbrowser
    
    X, Y = np.meshgrid(doses*100, heights)
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(x=X, y=Y, z=surv_p90, opacity=0.3, colorscale='Greys', showscale=False, name='90th Percentile'))
    fig.add_trace(go.Surface(x=X, y=Y, z=surv_p10, opacity=0.3, colorscale='Greys', showscale=False, name='10th Percentile'))
    fig.add_trace(go.Surface(x=X, y=Y, z=surv_p50, opacity=1.0, colorscale='Viridis', colorbar=dict(title="Probability"), name='Median Response'))

    fig.update_layout(
        title="OOD Generalization: P(Resolution) vs. Height & Petition Dose",
        scene=dict(
            xaxis_title='Petition Dose (%)',
            yaxis_title='Initial Requested Height (ft)',
            zaxis_title='P(Resolution)',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    out_path = os.path.join(OUT_DIR, "causal_3d_meshgrid_surface.html")
    fig.write_html(out_path)
    webbrowser.open_new_tab(f"file://{out_path}")
    print(f"Done! Saved and opened {out_path}")

if __name__ == "__main__":
    generate_3d_surfaces()
