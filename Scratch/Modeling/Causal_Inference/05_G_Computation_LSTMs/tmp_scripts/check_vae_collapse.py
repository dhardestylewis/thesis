import torch, torch.nn as nn, pandas as pd, numpy as np

# Path to weights
OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
VAE_PATH = rf"{OUT_DIR}\causal_vae_weights.pt"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"

# Model Arch (Must match exactly)
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

def check_latent_diversity():
    print("--- VAE Latent Diagnostic ---")
    df = pd.read_csv(PANEL_PATH)
    features = [
        "land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km",
        "local_unemployment_rate", "mortgage_rate_30yr", "period_seq", "petition_pct_this_period",
        "cumulative_petition_pct", "bw_sin", "bw_cos"
    ]
    
    # Calculate normalization (same as training)
    norm_dict = {}
    for f in ["land_acres", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km",
              "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        norm_dict[f] = (df[f].mean(), df[f].std())

    # Load model
    vae = ConditionalVAE(len(features))
    vae.load_state_dict(torch.load(VAE_PATH, weights_only=True))
    vae.eval()
    
    # Process a small sample of cases
    cases = df["case_number"].unique()[:100]
    mu_list = []
    
    for case in cases:
        group = df[df["case_number"] == case]
        data = group[features].fillna(0).values.copy()
        
        # Apply normalization
        for i, f in enumerate(features):
            if f in norm_dict:
                data[:, i] = (data[:, i] - norm_dict[f][0]) / (norm_dict[f][1] + 1e-8)
        
        if len(data) > 30: data = data[:30]
        if len(data) < 30:
            pad = np.zeros((30 - len(data), len(features)))
            data = np.vstack([data, pad])
        
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu, _ = vae.encode(x)
            mu_list.append(mu.numpy())
    
    mus = np.vstack(mu_list) # [100, 32]
    variances = mus.var(axis=0)
    active_units = (variances > 0.001).sum() # Lower threshold for "Active"
    
    print(f"Total Latent Units: {mus.shape[1]}")
    print(f"Active Latent Units (Var > 0.001): {active_units}")
    print(f"Mean Latent Variance: {variances.mean():.6f}")
    
    if active_units < 2:
        print("CRITICAL WARNING: Posterior Collapse Detected. The VAE is acting as a point-estimate.")
    else:
        print(f"Success: VAE Latent Space has {active_units} active components.")

if __name__ == "__main__":
    check_latent_diversity()
