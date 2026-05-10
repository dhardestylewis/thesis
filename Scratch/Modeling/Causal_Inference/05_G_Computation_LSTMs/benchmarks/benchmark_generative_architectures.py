"""
benchmark_generative_architectures.py
======================================
Head-to-head comparison of three generative Transformer variants against the
discriminative Multi-Task LSTM baseline, all evaluated on the same temporal
split using the continuous height / commission / council targets.

Architecture A: Existing Hybrid Transformer evaluated DISCRIMINATIVELY (true inputs)
Architecture B: Non-Autoregressive (BERT-style) Bidirectional Transformer
Architecture C: Conditional VAE Transformer
Baseline:       Multi-Task LSTM (discriminative)
"""

import os, time, math, re, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    average_precision_score, mean_squared_error,
    r2_score, mean_absolute_error, precision_recall_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
PANEL_PATH = rf"{OUT_DIR}\biweekly_panel.csv"

# ============================================================
# SHARED DATA PIPELINE
# ============================================================
def load_data():
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    master = pd.read_csv(r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv", low_memory=False)
    
    OVERLAY_STRIP = re.compile(r"(-NP|-CO|-H|-V|-CURE|-NCCD|-MU|-L|-SH|-DB90|-DB110|-ETOD|-PDA|-IA|-UC|-CU|-ICG|-W|-LEED|-SR|-PO|-DT|-NO|-OLD)")
    INTENSITY = {"W":1,"RR":1,"AG":1,"DR":1,"SF-1":2,"SF-2":2,"SF-3":2,"SF-4A":3,"SF-4B":3,"SF-5":3,"SF-6":3,"TF":3,
                 "MF-1":4,"MF-2":4,"MF-3":5,"MF-4":5,"MF-5":6,"MF-6":6,"LO":5,"GO":6,"NO":5,"LR":6,"GR":7,
                 "CS":7,"CS-1":7,"CR":7,"CH":8,"LI":8,"MI":9,"HI":9,"CBD":9,"DMU":8,"TOD":7,"MU":7,"PUD":7,"P":6}
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
    print("  Sorting and grouping...", flush=True)
    df = df.sort_values(["case_number", "period_seq"])
    
    # Get case-level metadata
    case_years = df.groupby("case_number")["year"].min()
    case_sizes = df.groupby("case_number").size()
    cases = case_years.index.values
    n_cases = len(cases)
    n_feat = len(features)
    n_targ = len(targets)
    
    # Pre-allocate
    X_all = np.zeros((n_cases, max_seq, n_feat), dtype=np.float32)
    Y_all = np.zeros((n_cases, max_seq, n_targ), dtype=np.float32)
    years = np.zeros(n_cases, dtype=np.int32)
    
    print(f"  Filling {n_cases} case tensors...", flush=True)
    feat_vals = df[features].values.astype(np.float32)
    targ_vals = df[targets].values.astype(np.float32)
    
    # Build index into the sorted df
    idx = 0
    for i, case in enumerate(cases):
        size = case_sizes[case]
        length = min(size, max_seq)
        X_all[i, :length, :] = feat_vals[idx:idx+length]
        Y_all[i, :length, :] = targ_vals[idx:idx+length]
        years[i] = case_years[case]
        idx += size
    
    train_mask = years < 2019
    test_mask = ~train_mask
    
    print(f"  Train: {train_mask.sum()} | Test: {test_mask.sum()}", flush=True)
    return (
        torch.from_numpy(X_all[train_mask]),
        torch.from_numpy(Y_all[train_mask]),
        torch.from_numpy(X_all[test_mask]),
        torch.from_numpy(Y_all[test_mask]),
    )


def evaluate_model(name, preds, Y_test, mask, norm_dict):
    """Evaluate a model's predictions against ground truth."""
    # preds: [B, T, 4] -> resolved, height, commission, council
    # Y_test: [B, T, 4] -> same order
    
    # Survival PRAUC
    surv_pred = preds[:, :, 0][mask].numpy()
    surv_true = Y_test[:, :, 0][mask].numpy()
    surv_prob = torch.sigmoid(torch.tensor(surv_pred)).numpy() if surv_pred.max() > 1 or surv_pred.min() < 0 else surv_pred
    prec, rec, _ = precision_recall_curve(surv_true, surv_prob)
    prauc = auc(rec, prec)
    
    # Height R^2 
    ht_pred = preds[:, :, 1][mask].numpy()
    ht_true = Y_test[:, :, 1][mask].numpy()
    r2_ht = r2_score(ht_true, ht_pred)
    # Denormalize for MAE in feet
    mean_ht, std_ht = norm_dict["proposed_max_height_ft"]
    mae_ht = mean_absolute_error(
        ht_true * std_ht + mean_ht,
        ht_pred * std_ht + mean_ht
    )
    
    # Commission MSE
    comm_pred = preds[:, :, 2][mask].numpy()
    comm_true = Y_test[:, :, 2][mask].numpy()
    mse_comm = mean_squared_error(comm_true, comm_pred)
    
    # Council MSE
    counc_pred = preds[:, :, 3][mask].numpy()
    counc_true = Y_test[:, :, 3][mask].numpy()
    mse_counc = mean_squared_error(counc_true, counc_pred)
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Survival PR-AUC:        {prauc:.4f}")
    print(f"  Height R^2:             {r2_ht:.4f}")
    print(f"  Height MAE (feet):      {mae_ht:.2f}")
    print(f"  Commission MSE:         {mse_comm:.4f}")
    print(f"  Commission RMSE:        {np.sqrt(mse_comm):.4f} hearings")
    print(f"  Council MSE:            {mse_counc:.4f}")
    print(f"  Council RMSE:           {np.sqrt(mse_counc):.4f} hearings")
    print(f"{'='*60}")
    
    return {
        "name": name,
        "prauc": prauc,
        "r2_ht": r2_ht,
        "mae_ht": mae_ht,
        "mse_comm": mse_comm,
        "mse_counc": mse_counc
    }


# ============================================================
# ARCHITECTURE A: Causal Transformer (Discriminative Evaluation)
# Uses causal masking during training, but feeds TRUE inputs at
# inference instead of autoregressive rollout.
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


class CausalTransformerMTL(nn.Module):
    """Architecture A: Causal (masked) Transformer with 4 task heads."""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                          dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        self.head_surv = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_ht   = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_comm = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_coun = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x):
        T = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.pos(self.proj(x))
        h = self.transformer(h, mask=mask, is_causal=True)
        return torch.cat([
            self.head_surv(h),
            self.head_ht(h),
            self.head_comm(h),
            self.head_coun(h)
        ], dim=-1)  # [B, T, 4]


# ============================================================
# ARCHITECTURE B: Non-Autoregressive (BERT-style) Bidirectional
# ============================================================
class BidirectionalTransformerMTL(nn.Module):
    """Architecture B: Full bidirectional attention, no causal mask."""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                          dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        self.head_surv = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_ht   = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_comm = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_coun = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x):
        h = self.pos(self.proj(x))
        h = self.transformer(h)  # NO MASK — full bidirectional attention
        return torch.cat([
            self.head_surv(h),
            self.head_ht(h),
            self.head_comm(h),
            self.head_coun(h)
        ], dim=-1)


# ============================================================
# ARCHITECTURE C: Conditional VAE Transformer
# ============================================================
class ConditionalVAETransformer(nn.Module):
    """Architecture C: Encodes sequence into latent z, decodes targets."""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: bidirectional transformer -> latent
        self.enc_proj = nn.Linear(input_dim, d_model)
        self.enc_pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                              dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        
        # Decoder: latent + positional -> targets per step
        self.dec_proj = nn.Linear(input_dim + latent_dim, d_model)
        self.dec_pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                              dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        
        self.head_surv = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_ht   = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_comm = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_coun = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def encode(self, x):
        h = self.enc_pos(self.enc_proj(x))
        h = self.encoder(h)
        # Pool over time dimension
        h_pool = h.mean(dim=1)  # [B, d_model]
        return self.fc_mu(h_pool), self.fc_logvar(h_pool)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, x, z):
        B, T, _ = x.shape
        # Broadcast z across time
        z_expand = z.unsqueeze(1).expand(-1, T, -1)  # [B, T, latent_dim]
        dec_input = torch.cat([x, z_expand], dim=-1)  # [B, T, input_dim + latent_dim]
        h = self.dec_pos(self.dec_proj(dec_input))
        h = self.decoder(h)
        return torch.cat([
            self.head_surv(h),
            self.head_ht(h),
            self.head_comm(h),
            self.head_coun(h)
        ], dim=-1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        preds = self.decode(x, z)
        return preds, mu, logvar


# ============================================================
# BASELINE: Multi-Task LSTM
# ============================================================
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
        return torch.cat([
            self.head_surv(h),
            self.head_ht(h),
            self.head_comm(h),
            self.head_coun(h)
        ], dim=-1)


# ============================================================
# TRAINING LOOP (shared for A, B, Baseline)
# ============================================================
def train_standard(model, X_train, Y_train, epochs=30, lr=0.003, batch_size=256):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    crit_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]), reduction='none')
    crit_mse = nn.MSELoss(reduction='none')
    
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total = 0
        for bx, by in loader:
            optimizer.zero_grad()
            preds = model(bx)  # [B, T, 4]
            
            # Mask out padding
            mask = (bx[:, :, -1] != 0).float()  # Use last feature (bw_cos) as proxy
            
            loss_surv = (crit_bce(preds[:, :, 0], by[:, :, 0]) * mask).sum() / mask.sum()
            loss_ht   = (crit_mse(preds[:, :, 1], by[:, :, 1]) * mask).sum() / mask.sum()
            loss_comm = (crit_mse(preds[:, :, 2], by[:, :, 2]) * mask).sum() / mask.sum()
            loss_coun = (crit_mse(preds[:, :, 3], by[:, :, 3]) * mask).sum() / mask.sum()
            
            loss = loss_surv + loss_ht + loss_comm + loss_coun
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()


def train_vae(model, X_train, Y_train, epochs=30, lr=0.003, batch_size=256, kl_weight=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    crit_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]), reduction='none')
    crit_mse = nn.MSELoss(reduction='none')
    
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            preds, mu, logvar = model(bx)
            
            mask = (bx[:, :, -1] != 0).float()
            
            loss_surv = (crit_bce(preds[:, :, 0], by[:, :, 0]) * mask).sum() / mask.sum()
            loss_ht   = (crit_mse(preds[:, :, 1], by[:, :, 1]) * mask).sum() / mask.sum()
            loss_comm = (crit_mse(preds[:, :, 2], by[:, :, 2]) * mask).sum() / mask.sum()
            loss_coun = (crit_mse(preds[:, :, 3], by[:, :, 3]) * mask).sum() / mask.sum()
            
            recon_loss = loss_surv + loss_ht + loss_comm + loss_coun
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("GENERATIVE ARCHITECTURE BENCHMARK")
    print("=" * 60)
    
    print("\n[1/6] Loading data...")
    df, features, targets, norm_dict = load_data()
    
    print("[2/6] Building tensors (temporal split: train < 2019, test >= 2019)...")
    X_train, Y_train, X_test, Y_test = build_tensors(df, features, targets)
    print(f"  Train: {X_train.shape[0]} cases | Test: {X_test.shape[0]} cases")
    
    # Padding mask for evaluation
    mask = (X_test[:, :, -1] != 0)  # bw_cos != 0
    
    results = []
    
    # --- BASELINE: Multi-Task LSTM ---
    print("\n[3/6] Training BASELINE: Multi-Task LSTM...")
    t0 = time.time()
    lstm = MultiTaskLSTM(len(features))
    train_standard(lstm, X_train, Y_train, epochs=30, lr=0.005)
    lstm.eval()
    with torch.no_grad():
        preds_lstm = lstm(X_test)
    print(f"  Trained in {time.time()-t0:.1f}s")
    results.append(evaluate_model("BASELINE: Multi-Task LSTM (Discriminative)", preds_lstm, Y_test, mask, norm_dict))
    
    # --- ARCH A: Causal Transformer (Discriminative Eval) ---
    print("\n[4/6] Training ARCH A: Causal Transformer (Discriminative Eval)...")
    t0 = time.time()
    arch_a = CausalTransformerMTL(len(features))
    train_standard(arch_a, X_train, Y_train, epochs=30, lr=0.001)
    arch_a.eval()
    with torch.no_grad():
        preds_a = arch_a(X_test)  # Feed TRUE inputs, no autoregressive rollout
    print(f"  Trained in {time.time()-t0:.1f}s")
    results.append(evaluate_model("ARCH A: Causal Transformer (Discriminative Eval)", preds_a, Y_test, mask, norm_dict))
    
    # --- ARCH B: Bidirectional Transformer ---
    print("\n[5/6] Training ARCH B: Bidirectional (BERT-style) Transformer...")
    t0 = time.time()
    arch_b = BidirectionalTransformerMTL(len(features))
    train_standard(arch_b, X_train, Y_train, epochs=30, lr=0.001)
    arch_b.eval()
    with torch.no_grad():
        preds_b = arch_b(X_test)
    print(f"  Trained in {time.time()-t0:.1f}s")
    results.append(evaluate_model("ARCH B: Bidirectional (BERT-style) Transformer", preds_b, Y_test, mask, norm_dict))
    
    # --- ARCH C: Conditional VAE ---
    print("\n[6/6] Training ARCH C: Conditional VAE Transformer...")
    t0 = time.time()
    arch_c = ConditionalVAETransformer(len(features))
    train_vae(arch_c, X_train, Y_train, epochs=30, lr=0.001)
    arch_c.eval()
    with torch.no_grad():
        preds_c, _, _ = arch_c(X_test)
    print(f"  Trained in {time.time()-t0:.1f}s")
    results.append(evaluate_model("ARCH C: Conditional VAE Transformer", preds_c, Y_test, mask, norm_dict))
    
    # --- SUMMARY TABLE ---
    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Architecture':<50} {'PRAUC':>8} {'Ht R^2':>8} {'Ht MAE':>8} {'Comm MSE':>10} {'Coun MSE':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<50} {r['prauc']:>8.4f} {r['r2_ht']:>8.4f} {r['mae_ht']:>8.2f} {r['mse_comm']:>10.4f} {r['mse_counc']:>10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
