import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import re
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go
from sklearn.metrics import average_precision_score

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
    
    features = [
        "land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", 
        "local_unemployment_rate", "mortgage_rate_30yr", "period_seq", "petition_pct_this_period", 
        "cumulative_petition_pct", "bw_sin", "bw_cos", 
        "resolved", "t_downgrade", "commission_hearings_this_period", "council_hearings_this_period"
    ]
    for f in features: df[f] = df[f].fillna(0)
    
    for f in ["land_acres", "proposed_max_height_ft", "archetype_pct_Spatial_Gravity", "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr", "period_seq"]:
        mean_v = df[f].mean()
        std_v = df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        
    return df, features

def extract_tensors(df, features, max_len=30):
    cases = df['case_number'].unique()
    X = []
    groups = df.groupby('case_number')
    for case in cases:
        group = groups.get_group(case).sort_values('period_seq')
        tensor = np.zeros((max_len, len(features)))
        length = min(len(group), max_len)
        tensor[:length, :] = group[features].values[:length]
        
        if length > 0:
            is_dg = tensor[length-1, features.index("t_downgrade")]
            tensor[length-1:, features.index("t_downgrade")] = is_dg
            tensor[length-1:, features.index("resolved")] = 1.0

        X.append(tensor)
    return torch.tensor(np.stack(X), dtype=torch.float32)

# --- 2. ACT ARCHITECTURE ---
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

class AutoRegressiveTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=60)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.out_proj = nn.Linear(d_model, feature_dim)

    def forward(self, x):
        # x: [B, T, F]
        Seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(Seq_len).to(x.device)
        
        x_emb = self.feature_proj(x)
        x_emb = self.pos_encoder(x_emb)
        
        out = self.transformer(x_emb, mask=mask, is_causal=True)
        return self.out_proj(out)

# --- 3. HYBRID LOSS ---
def hybrid_loss(pred, target, features):
    # Pred and target are [B, T, F]
    idx_res = features.index("resolved")
    idx_dg = features.index("t_downgrade")
    
    # Continuous features
    cont_mask = torch.ones(len(features), dtype=torch.bool)
    cont_mask[idx_res] = False
    cont_mask[idx_dg] = False
    
    mse_loss = nn.MSELoss()(pred[:, :, cont_mask], target[:, :, cont_mask])
    
    # Binary features (BCE with Logits)
    bce = nn.BCEWithLogitsLoss()
    bce_res = bce(pred[:, :, idx_res], target[:, :, idx_res])
    
    # Massive weight on downgrade
    pos_weight = torch.tensor([50.0], device=pred.device)
    bce_dg = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred[:, :, idx_dg], target[:, :, idx_dg])
    
    return mse_loss + bce_res + 2.0 * bce_dg

# --- 4. AUTOREGRESSIVE GENERATION ---
def generate_autoregressive(model, true_x, intervention_p, intervention_pct, features, max_len=30):
    device = true_x.device
    B, _, F = true_x.shape
    
    # We start with the known history up to intervention_p-1
    # Actually intervention_p is the 0-indexed period. If p=5, index=4.
    # So history is indices 0 to intervention_p-1
    seq = true_x[:, :intervention_p, :].clone()
    
    idx_pct = features.index("petition_pct_this_period")
    idx_cum = features.index("cumulative_petition_pct")
    idx_res = features.index("resolved")
    idx_dg = features.index("t_downgrade")
    
    for t in range(intervention_p, max_len):
        # Apply intervention condition for the NEXT step to be predicted
        # Wait, the transformer takes the sequence up to t-1 and predicts step t.
        # If we have sequence [0 ... t-1], we predict step t.
        pred_logits = model(seq) # [B, T, F]
        next_step = pred_logits[:, -1:, :].clone() # [B, 1, F]
        
        # Apply Sigmoid to binary channels for the generated sequence
        next_step[:, 0, idx_res] = torch.sigmoid(next_step[:, 0, idx_res])
        next_step[:, 0, idx_dg]  = torch.sigmoid(next_step[:, 0, idx_dg])
        
        # Inject the intervention manually into the generated step if it is the intervention period
        if t == intervention_p:
            next_step[:, 0, idx_pct] = intervention_pct
        else:
            next_step[:, 0, idx_pct] = 0.0 # Protests usually drop after the spike
            
        if t >= intervention_p:
            # Accumulate
            prev_cum = seq[:, -1, idx_cum]
            next_step[:, 0, idx_cum] = torch.clamp(prev_cum + next_step[:, 0, idx_pct], 0, 1)
        
        # Append
        seq = torch.cat([seq, next_step], dim=1)
        
        # We could terminate early if all batch items resolved, but for vectorization we run to max_len
        
    return seq

# --- 5. EXECUTION ---
def main():
    print("[*] Initializing Autoregressive Causal Transformer Pipeline...")
    df, features = load_and_prep_data()
    
    cases = df['case_number'].unique()
    np.random.seed(42)
    np.random.shuffle(cases)
    
    train_cases = cases[:int(len(cases)*0.8)]
    test_cases = cases[int(len(cases)*0.8):]
    
    df_train = df[df['case_number'].isin(train_cases)]
    df_test = df[df['case_number'].isin(test_cases)]
    
    X_train = extract_tensors(df_train, features, max_len=30)
    X_test = extract_tensors(df_test, features, max_len=30)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    
    model = AutoRegressiveTransformer(feature_dim=len(features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Next step prediction setup
    # Input is X[:, :-1, :], Target is X[:, 1:, :]
    dataset = TensorDataset(X_train[:, :-1, :], X_train[:, 1:, :])
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    print("[*] Training ACT with Hybrid Weighted Loss...")
    EPOCHS = 200
    model.train()
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, steps_per_epoch=len(dataloader), epochs=EPOCHS)
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = hybrid_loss(pred, batch_y, features)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        if epoch % 25 == 0 or epoch == EPOCHS-1:
            print(f"Epoch {epoch} | Loss: {total_loss/len(dataloader):.4f}")
            
    print(f"[*] Training Complete in {time.time()-t0:.1f}s")
    
    print("[*] Evaluating Holdout PRAUC on Generated Downzoning Outcomes...")
    model.eval()
    with torch.no_grad():
        # Metric 1: 1-Step Ahead (Month 1 Out)
        test_pred = model(X_test[:, :-1, :])
        idx_dg = features.index("t_downgrade")
        
        # test_pred is [B, 29, F], targets are X_test[:, 1:, F]
        # We check the final valid step of the sequence for the 1-step prediction
        # To be fair, let's just flatten the whole 29-step tensor and check accuracy of every single 1-step prediction!
        probs_1step = torch.sigmoid(test_pred[:, :, idx_dg]).cpu().numpy().flatten()
        true_1step = X_test[:, 1:, idx_dg].cpu().numpy().flatten()
        prauc_1step = average_precision_score(true_1step, probs_1step)
        
        # Metric 2: Full Sequence Autoregression (Month 14 Out)
        intervention_p = 5
        generated_seqs = generate_autoregressive(model, X_test, intervention_p, 0.0, features, max_len=30)
        
        final_gen = generated_seqs[:, -1, idx_dg].cpu().numpy()
        y_probs = np.clip(final_gen, 0, 1)
        y_true = X_test[:, -1, idx_dg].cpu().numpy()
        prauc_full = average_precision_score(y_true, y_probs)
        
        baseline = np.mean(y_true)
        print(f"  > ACT Holdout PRAUC (1-Step Ahead): {prauc_1step:.4f}")
        print(f"  > ACT Holdout PRAUC (Full Seq Gen): {prauc_full:.4f}")
        print(f"  > Random Baseline: {baseline:.4f}")
        
    print("[*] Skipping Causal Surface Generation for rapid PRAUC evaluation...")

if __name__ == "__main__":
    main()
