import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from torch.utils.data import TensorDataset, DataLoader
from run_causal_lstm_era_slider import precompute_tensors, get_train_tensors_from_cache

# --- DATA PREP ---
print("[*] Loading Dataset...")
df = pd.read_csv(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv", low_memory=False)
master = pd.read_csv(r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv", low_memory=False)
OVERLAY_STRIP = re.compile(r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)")
INTENSITY = {"W":1,"RR":1,"AG":1,"DR":1,"SF-1":2,"SF-2":2,"SF-3":2,"SF-4A":3,"SF-4B":3,"SF-5":3,"SF-6":3,"TF":3,"MF-1":4,"MF-2":4,"MF-3":5,"MF-4":5,"MF-5":6,"MF-6":6,"LO":5,"GO":6,"NO":5,"LR":6,"GR":7,"CS":7,"CS-1":7,"CR":7,"CH":8,"LI":8,"MI":9,"HI":9,"CBD":9,"DMU":8,"TOD":7,"MU":7,"PUD":7,"P":6}
def get_int(z): return INTENSITY.get(OVERLAY_STRIP.sub("", str(z).strip().upper()).strip("-"), np.nan)
master["case_number"] = master["case_number"].astype(str).str.strip()
master["req_int"] = master["Requested_Zoning"].apply(get_int)
master["fin_int"] = master["Final_Zoning"].apply(get_int)
master["z_changed"] = master["Requested_Zoning"].str.strip() != master["Final_Zoning"].str.strip()
master["t_downgrade"] = ((master["fin_int"] < master["req_int"]) & master["z_changed"]).astype(float)
df["case_number"] = df["case_number"].astype(str).str.strip()
df = df.merge(master[["case_number", "t_downgrade"]].drop_duplicates("case_number"), on="case_number", how="left")
df["t_downgrade"] = df["t_downgrade"].fillna(0)

df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
features = ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq", "petition_pct_this_period", "cumulative_petition_pct", "bw_sin", "bw_cos"]
for f in features: df[f] = df[f].fillna(0)

for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
    mean_v = df[f].mean()
    std_v = df[f].std()
    df[f] = (df[f] - mean_v) / (std_v + 1e-8)

cache = precompute_tensors(df, features, 30)
era_dt = pd.to_datetime("2020-01-01")
dataset, _ = get_train_tensors_from_cache(cache, era_dt)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Truncate sequences to T=15 for generative benchmark
X_all = torch.stack([x[0][:15] for x in dataset]).to(device)
dataloader = DataLoader(TensorDataset(X_all), batch_size=256, shuffle=True)

N, T, F = X_all.shape
Z_DIM = 8
EPOCHS = 5

print(f"[*] Benchmark Dataset: N={N}, T={T}, F={F}")
print("="*50)

# --- 1. CAUSAL SEQUENCE GAN ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(Z_DIM, 64, batch_first=True)
        self.fc = nn.Linear(64, F)
    def forward(self, z):
        out, _ = self.lstm(z)
        return self.fc(out)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(F, 64, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

G = Generator().to(device)
D = Discriminator().to(device)
opt_G = torch.optim.Adam(G.parameters(), lr=1e-3)
opt_D = torch.optim.Adam(D.parameters(), lr=1e-3)
bce = nn.BCELoss()

t0 = time.time()
for epoch in range(EPOCHS):
    for batch in dataloader:
        real_x = batch[0]
        bs = real_x.size(0)
        
        # Train D
        opt_D.zero_grad()
        z = torch.randn(bs, T, Z_DIM).to(device)
        fake_x = G(z)
        
        real_loss = bce(D(real_x), torch.ones(bs, 1).to(device))
        fake_loss = bce(D(fake_x.detach()), torch.zeros(bs, 1).to(device))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        opt_D.step()
        
        # Train G
        opt_G.zero_grad()
        g_loss = bce(D(fake_x), torch.ones(bs, 1).to(device))
        g_loss.backward()
        opt_G.step()
gan_time = time.time() - t0
print(f"[BENCHMARK] Causal GAN (5 Epochs): {gan_time:.2f} seconds")

# --- 2. SEQUENTIAL VAE ---
class SeqVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.LSTM(F, 64, batch_first=True)
        self.fc_mu = nn.Linear(64, Z_DIM)
        self.fc_logvar = nn.Linear(64, Z_DIM)
        
        self.dec = nn.LSTM(Z_DIM, 64, batch_first=True)
        self.fc_out = nn.Linear(64, F)
        
    def forward(self, x):
        out, (h, c) = self.enc(x)
        h_last = h[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        z_seq = z.unsqueeze(1).repeat(1, T, 1)
        dec_out, _ = self.dec(z_seq)
        return self.fc_out(dec_out), mu, logvar

vae = SeqVAE().to(device)
opt_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)

t0 = time.time()
for epoch in range(EPOCHS):
    for batch in dataloader:
        x = batch[0]
        opt_vae.zero_grad()
        recon, mu, logvar = vae(x)
        
        recon_loss = nn.MSELoss()(recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        loss = recon_loss + 0.01 * kl_loss
        loss.backward()
        opt_vae.step()
vae_time = time.time() - t0
print(f"[BENCHMARK] Sequential VAE (5 Epochs): {vae_time:.2f} seconds")

# --- 3. CAUSAL FLOW MATCHING ---
class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input is x_t (F) + t (1)
        self.lstm = nn.LSTM(F + 1, 64, batch_first=True)
        self.fc = nn.Linear(64, F)
    def forward(self, x_t, t_val):
        t_seq = t_val.view(-1, 1, 1).repeat(1, T, 1)
        inp = torch.cat([x_t, t_seq], dim=-1)
        out, _ = self.lstm(inp)
        return self.fc(out)

flow_net = FlowNet().to(device)
opt_flow = torch.optim.Adam(flow_net.parameters(), lr=1e-3)

t0 = time.time()
for epoch in range(EPOCHS):
    for batch in dataloader:
        x_1 = batch[0]
        bs = x_1.size(0)
        
        opt_flow.zero_grad()
        
        # Sample base distribution (noise)
        x_0 = torch.randn_like(x_1)
        
        # Sample random time uniformly [0, 1]
        t = torch.rand(bs, device=device)
        
        # Conditional Flow Matching interpolation
        t_expanded = t.view(bs, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # True optimal transport vector field
        v_target = x_1 - x_0
        
        # Predict vector field
        v_pred = flow_net(x_t, t)
        
        loss = nn.MSELoss()(v_pred, v_target)
        loss.backward()
        opt_flow.step()
flow_time = time.time() - t0
print(f"[BENCHMARK] Causal Flow Matching (5 Epochs): {flow_time:.2f} seconds")
print("="*50)

# --- 4. ACCURACY / GENERATIVE QUALITY EVALUATION ---
print("[*] Evaluating Generative Accuracy...")

# Get a real batch for baseline comparison
real_batch = next(iter(dataloader))[0]
real_sample = real_batch[0].cpu().detach().numpy()

# GAN Generation
with torch.no_grad():
    z_gan = torch.randn(1, T, Z_DIM).to(device)
    fake_gan = G(z_gan)[0].cpu().numpy()

# VAE Generation
with torch.no_grad():
    # Sample from standard normal prior
    z_vae = torch.randn(1, Z_DIM).to(device)
    z_seq = z_vae.unsqueeze(1).repeat(1, T, 1)
    fake_vae = vae.dec(z_seq)[0]
    fake_vae = vae.fc_out(fake_vae)[0].cpu().numpy()

# Flow Matching Generation (Requires ODE Integration via Euler method)
with torch.no_grad():
    # Start at noise
    x_t = torch.randn(1, T, F).to(device)
    steps = 10
    dt = 1.0 / steps
    for step in range(steps):
        t_val = torch.tensor([step * dt]).to(device)
        v = flow_net(x_t, t_val)
        x_t = x_t + v * dt  # Euler step
    fake_flow = x_t[0].cpu().numpy()

# Metric: Measure structural coherence via Auto-Correlation and Variance
def eval_sequence(fake_seq, name):
    mse_to_real_mean = np.mean((fake_seq - np.mean(real_batch.cpu().numpy(), axis=0))**2)
    seq_variance = np.var(fake_seq)
    print(f"  > {name:22} | Variance: {seq_variance:.4f} | MSE vs Real Mean: {mse_to_real_mean:.4f}")

eval_sequence(real_sample, "Real Data Sample")
eval_sequence(fake_gan, "Causal GAN Generated")
eval_sequence(fake_vae, "Sequential VAE Generated")
eval_sequence(fake_flow, "Flow Match Generated")

print("="*50)
print("Benchmark Complete!")
