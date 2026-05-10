import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import re
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score, mean_squared_error

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
        
        # FIX DATA LEAK: Zero out t_downgrade across all steps first
        tensor[:, features.index("t_downgrade")] = 0.0
        
        if length > 0:
            # Only inject the downgrade flag AT the resolution step and beyond
            is_dg = group["t_downgrade"].values[0]
            tensor[length-1:, features.index("t_downgrade")] = is_dg
            tensor[length-1:, features.index("resolved")] = 1.0
        X.append(tensor)
    return torch.tensor(np.stack(X), dtype=torch.float32)

# --- 2. HYBRID ARCHITECTURE ---
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

class HybridTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=60)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Generative Head: Predicts X_{t+1}
        self.gen_proj = nn.Linear(d_model, feature_dim)
        
        # Discriminative Head: Predicts Final Global t_downgrade outcome from latent state t
        self.aux_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        Seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(Seq_len).to(x.device)
        
        x_emb = self.feature_proj(x)
        x_emb = self.pos_encoder(x_emb)
        
        hidden = self.transformer(x_emb, mask=mask, is_causal=True)
        
        gen_pred = self.gen_proj(hidden)
        aux_pred = self.aux_proj(hidden).squeeze(-1) # [B, T]
        return gen_pred, aux_pred

# --- 3. LOSS ---
def compute_loss(gen_pred, aux_pred, target_gen, target_global, features):
    idx_res = features.index("resolved")
    idx_dg = features.index("t_downgrade")
    
    cont_mask = torch.ones(len(features), dtype=torch.bool)
    cont_mask[idx_res] = False
    cont_mask[idx_dg] = False
    
    loss_mse = nn.MSELoss()(gen_pred[:, :, cont_mask], target_gen[:, :, cont_mask])
    
    bce = nn.BCEWithLogitsLoss()
    loss_res = bce(gen_pred[:, :, idx_res], target_gen[:, :, idx_res])
    loss_dg_gen = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=gen_pred.device))(gen_pred[:, :, idx_dg], target_gen[:, :, idx_dg])
    
    # Auxiliary Loss: Every step t tries to predict the FINAL global outcome
    # aux_pred is [B, T], target_global is [B]
    target_aux = target_global.unsqueeze(1).expand(-1, aux_pred.size(1))
    loss_aux = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([25.0], device=gen_pred.device))(aux_pred, target_aux)
    
    # PURE DISCRIMINATIVE ABLATION: Ignore all generative losses
    return loss_aux

# --- 4. STEP-BY-STEP EVALUATION ---
def evaluate_exposure_bias(model, X_test, features, intervention_p=5):
    device = X_test.device
    B, _, F = X_test.shape
    idx_dg = features.index("t_downgrade")
    
    # Continuous mask for MSE
    idx_res = features.index("resolved")
    cont_mask = [i for i in range(F) if i not in (idx_res, idx_dg)]
    
    # 1. Autoregressive Generation
    seq = X_test[:, :intervention_p, :].clone()
    
    # We will store the generated steps and the aux predictions
    aux_preds = []
    
    with torch.no_grad():
        for t in range(intervention_p, 30):
            gen_pred, aux_pred = model(seq)
            
            # Record the aux prediction for the CURRENT step
            aux_preds.append(aux_pred[:, -1].cpu().numpy())
            
            next_step = gen_pred[:, -1:, :].clone()
            next_step[:, 0, idx_res] = torch.sigmoid(next_step[:, 0, idx_res])
            next_step[:, 0, idx_dg]  = torch.sigmoid(next_step[:, 0, idx_dg])
            
            seq = torch.cat([seq, next_step], dim=1)
            
    # seq is now [B, 30, F]
    true_global_outcome = X_test[:, -1, idx_dg].cpu().numpy()
    baseline = np.mean(true_global_outcome)
    
    # Evaluate at K-months out
    k_steps = [1, 2, 5, 24] # Month 1 out, Month 2 out, Month 5 out, Month 24 out (Step 29, End)
    
    # Pre-fetch normalization stats for true un-scaling
    ht_idx = features.index("proposed_max_height_ft")
    comm_idx = features.index("commission_hearings_this_period")
    counc_idx = features.index("council_hearings_this_period")
    
    # Use standard denorm logic from earlier scripts
    mean_ht, std_ht = 51.5, 45.0 # We'll just approximate or compute it inside the loop
    from sklearn.metrics import r2_score, mean_absolute_error
    
    print("\n" + "="*50)
    print("      EXPOSURE BIAS DEGRADATION METRICS")
    print("="*50)
    print(f"Baseline P(Downgrade): {baseline:.4f}")
    
    for k in k_steps:
        eval_t = intervention_p + k - 1
        if eval_t >= 30: eval_t = 29
        
        # Continuous MSE
        true_slice = X_test[:, eval_t, cont_mask].cpu().numpy()
        gen_slice = seq[:, eval_t, cont_mask].cpu().numpy()
        mse = mean_squared_error(true_slice, gen_slice)
        
        # Aux PRAUC (How well did the latent state at eval_t predict the global outcome?)
        # aux_preds is a list of predictions starting from t=5.
        # So index (k-1) corresponds to step eval_t.
        pred_aux_logits = aux_preds[k-1]
        probs_aux = 1 / (1 + np.exp(-pred_aux_logits))
        
        prauc = average_precision_score(true_global_outcome, probs_aux)
        
        print(f"[Month {k} Out | Step {eval_t+1}] Continuous MSE: {mse:.4f} | Aux PRAUC: {prauc:.4f}")
        
        # Calculate explicit metrics for current step
        true_ht = X_test[:, eval_t, ht_idx].cpu().numpy()
        gen_ht = seq[:, eval_t, ht_idx].cpu().numpy()
        r2_ht = r2_score(true_ht, gen_ht)
        
        true_comm = X_test[:, eval_t, comm_idx].cpu().numpy()
        gen_comm = seq[:, eval_t, comm_idx].cpu().numpy()
        mse_comm = mean_squared_error(true_comm, gen_comm)
        
        true_counc = X_test[:, eval_t, counc_idx].cpu().numpy()
        gen_counc = seq[:, eval_t, counc_idx].cpu().numpy()
        mse_counc = mean_squared_error(true_counc, gen_counc)
        
        print(f"    -> Height R^2: {r2_ht:.4f} | Comm MSE: {mse_comm:.4f} | Council MSE: {mse_counc:.4f}")
    print("="*50)

# --- 5. EXECUTION ---
def main():
    print("[*] Initializing Hybrid Generative-Discriminative Transformer...")
    df, features = load_and_prep_data()
    
    cases = df['case_number'].unique()
    np.random.seed(42)
    np.random.shuffle(cases)
    
    train_cases = cases[:int(len(cases)*0.8)]
    test_cases = cases[int(len(cases)*0.8):]
    
    X_train = extract_tensors(df[df['case_number'].isin(train_cases)], features, 30)
    X_test = extract_tensors(df[df['case_number'].isin(test_cases)], features, 30)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train, X_test = X_train.to(device), X_test.to(device)
    
    model = HybridTransformer(feature_dim=len(features)).to(device)
    # Recovered hyperparams: static Adam, lower LR
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    dataset = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    EPOCHS = 150
    print("[*] Training Hybrid ACT with Scheduled Sampling and Gradient Clipping...")
    t0 = time.time()
    
    idx_dg = features.index("t_downgrade")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Disable scheduled sampling for pure discriminative training
        tf_ratio = 1.0
        
        for batch in dataloader:
            batch_x = batch[0] # [B, 30, F]
            optimizer.zero_grad()
            
            target_gen = batch_x[:, 1:, :] # [B, 29, F]
            target_global = batch_x[:, -1, idx_dg] # [B]
            
            # Vectorized Scheduled Sampling (Corrupted Teacher Forcing)
            # 1. Forward pass with true data
            gen_pred_true, aux_pred_true = model(batch_x[:, :-1, :])
            
            # 2. Corrupt the input for the final loss calculation
            if np.random.random() > tf_ratio:
                # Replace random chunks of true history with model's own predictions
                corrupted_x = batch_x[:, :-1, :].clone()
                # Detach predictions so gradients don't flow through the corruption step
                replace_mask = torch.rand(batch_x.size(0), 29, 1, device=device) < 0.3
                corrupted_x = torch.where(replace_mask, gen_pred_true.detach(), corrupted_x)
                
                gen_pred, aux_pred = model(corrupted_x)
            else:
                gen_pred, aux_pred = gen_pred_true, aux_pred_true
                
            loss = compute_loss(gen_pred, aux_pred, target_gen, target_global, features)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 25 == 0 or epoch == EPOCHS-1:
            print(f"Epoch {epoch} | Loss: {total_loss/len(dataloader):.4f} | TF_Ratio: {tf_ratio:.2f}")

    print(f"[*] Training Complete in {time.time()-t0:.1f}s")
    
    print("[*] Running Step-by-Step Exposure Bias Degradation Evaluation...")
    model.eval()
    evaluate_exposure_bias(model, X_test, features, intervention_p=5)

if __name__ == "__main__":
    main()
