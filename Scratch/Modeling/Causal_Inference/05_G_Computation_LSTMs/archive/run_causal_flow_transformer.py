import os
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import re
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go

# --- 1. DATA PREP ---
def load_and_prep_data():
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
    
    # NEW: We append the targets directly to the input sequence!
    features = [
        "land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", 
        "local_unemployment_rate", "mortgage_rate_30yr", "period_seq", "petition_pct_this_period", 
        "cumulative_petition_pct", "bw_sin", "bw_cos", 
        "resolved", "t_downgrade", "commission_hearings_this_period", "council_hearings_this_period"
    ]
    for f in features: df[f] = df[f].fillna(0)
    
    norm_dict = {}
    for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        mean_v = df[f].mean()
        std_v = df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)
        
    return df, features, norm_dict

def extract_tensors(df, features, max_len=30):
    cases = df['case_number'].unique()
    X = []
    groups = df.groupby('case_number')
    for case in cases:
        group = groups.get_group(case).sort_values('period_seq')
        tensor = np.zeros((max_len, len(features)))
        length = min(len(group), max_len)
        tensor[:length, :] = group[features].values[:length]
        
        # We need the target (downgrade) to exist at ALL steps after resolution for the generative model to learn it smoothly
        if length > 0:
            is_dg = tensor[length-1, features.index("t_downgrade")]
            tensor[length-1:, features.index("t_downgrade")] = is_dg
            tensor[length-1:, features.index("resolved")] = 1.0

        X.append(tensor)
    return torch.tensor(np.stack(X), dtype=torch.float32)

# --- 2. CFM-TRANSFORMER ARCHITECTURE ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CausalFlowTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=128, nhead=4, num_layers=4, max_seq_len=30):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.out_proj = nn.Linear(d_model, feature_dim)

    def forward(self, x, t):
        # x: [B, T, F], t: [B]
        B, Seq_len, F = x.shape
        device = x.device
        
        x_emb = self.feature_proj(x) # [B, T, D]
        t_emb = self.time_mlp(t).unsqueeze(1).expand(-1, Seq_len, -1) # [B, T, D]
        
        positions = torch.arange(Seq_len, device=device).unsqueeze(0).expand(B, -1)
        p_emb = self.pos_emb(positions) # [B, T, D]
        
        # Add sequence positions and continuous diffusion time
        h = x_emb + t_emb + p_emb
        
        out = self.transformer(h)
        v = self.out_proj(out)
        return v

# --- 3. RK4 INPAINTING SOLVER ---
def rk4_inpaint_solver(model, true_x, intervention_p, intervention_pct, features, steps=20):
    device = true_x.device
    B, Seq_len, F = true_x.shape
    dt = 1.0 / steps
    
    # Start at noise
    x_t = torch.randn_like(true_x).to(device)
    
    idx_pct = features.index("petition_pct_this_period")
    idx_cum = features.index("cumulative_petition_pct")
    
    for i in range(steps):
        t_val = torch.ones(B, device=device) * (i * dt)
        
        # CAUSAL INPAINTING: Overwrite history
        # We know the true sequence up to intervention_p - 1.
        # We also know the true exact shock at intervention_p.
        if intervention_p > 0:
            x_t[:, :intervention_p, :] = true_x[:, :intervention_p, :]
            
        # Apply the hard intervention condition
        x_t[:, intervention_p, idx_pct] = intervention_pct
        x_t[:, intervention_p:, idx_cum] = intervention_pct # simplified step
        
        # RK4 Integration
        t0 = t_val
        t1 = t_val + dt/2
        t2 = t_val + dt
        
        k1 = model(x_t, t0)
        k2 = model(x_t + k1 * dt/2, t1)
        k3 = model(x_t + k2 * dt/2, t1)
        k4 = model(x_t + k3 * dt, t2)
        
        x_t = x_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return x_t

# --- 4. EXECUTION ---
def main():
    print("[*] Initializing CFM-T Pipeline...")
    df, features, norm_dict = load_and_prep_data()
    
    # Filter to 2020 era for speed
    df_era = df[df['period_start'] <= '2020-01-01']
    X_train = extract_tensors(df_era, features, 30)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = X_train.to(device)
    
    model = CausalFlowTransformer(feature_dim=len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dataloader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)
    
    print("[*] Training Conditional Flow Matching...")
    EPOCHS = 20
    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            x_1 = batch[0]
            B = x_1.size(0)
            optimizer.zero_grad()
            
            x_0 = torch.randn_like(x_1)
            t = torch.rand(B, device=device)
            t_expand = t.view(B, 1, 1)
            
            x_t = (1 - t_expand) * x_0 + t_expand * x_1
            v_target = x_1 - x_0
            
            v_pred = model(x_t, t)
            loss = nn.MSELoss()(v_pred, v_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(dataloader):.4f}")
            
    print(f"[*] Training Complete in {time.time()-t0:.1f}s")
    
    print("[*] Generating Causal Surface via RK4 Inpainting...")
    model.eval()
    
    # We take the first 500 cases from our empirical pool to run counterfactuals on
    true_x_pool = X_train[:500] 
    
    periods = list(range(1, 16))
    pcts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    
    Z_downgrade = np.zeros((len(pcts), len(periods)))
    
    idx_dg = features.index("t_downgrade")
    
    with torch.no_grad():
        for i_p, pct in enumerate(pcts):
            for i_per, p in enumerate(periods):
                # Run the ODE solver to generate the counterfactuals
                # intervention_p is p-1
                generated_x = rk4_inpaint_solver(model, true_x_pool, p-1, pct, features, steps=10)
                
                # The model generated the full sequence. The outcome t_downgrade is read at the final timestep.
                # Because the model outputs continuous floats for the binary target, we apply sigmoid to interpret as probability
                # Wait, the real data was strictly 0 or 1, not logits. 
                # Flow matching models continuous data, so the output is literally continuous in [0, 1].
                final_dg_vals = generated_x[:, -1, idx_dg].cpu().numpy()
                
                # Clip to probability bounds
                probs = np.clip(final_dg_vals, 0, 1)
                Z_downgrade[i_p, i_per] = np.mean(probs)
                
    # --- 5. PLOTLY VISUALIZATION ---
    fig = go.Figure(data=[go.Surface(z=Z_downgrade, x=periods, y=pcts, colorscale='Magma')])
    fig.update_layout(title='CFM-T Generated Counterfactual Surface: Downgrade Probability',
                      scene=dict(xaxis_title='Intervention Period', yaxis_title='Petition %', zaxis_title='P(Downgrade)'))
    
    out_path = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\cfm_causal_surface.html"
    fig.write_html(out_path)
    print(f"[*] Surface exported to {out_path}")

if __name__ == "__main__":
    main()
