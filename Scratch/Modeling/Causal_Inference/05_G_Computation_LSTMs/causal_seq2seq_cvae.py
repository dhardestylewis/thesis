import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, KFold, GroupKFold
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.environ.get("OUT_DIR", r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756")
PANEL_PATH = os.environ.get("PANEL_PATH", os.path.join(OUT_DIR, "biweekly_panel.csv"))
CAUSAL_SURFACE_PATH = os.environ.get("CAUSAL_SURFACE_PATH", os.path.join(OUT_DIR, "vae_dose_response_surface_expanded.csv"))
VAE_PATH = os.path.join(OUT_DIR, "causal_vae_weights.pt")
LSTM_PATH = os.path.join(OUT_DIR, "causal_lstm_weights.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 500

# Spatial Grid Definitions (Quantile Macro-Cells)
# Heights and Doses are now dynamically bucketed via pd.qcut

# ============================================================
# DATA & SPATIAL SPLIT
# ============================================================
def load_data_and_cells():
    import os
    if os.path.exists("/home/ubuntu/biweekly_panel.csv"):
        df = pd.read_csv("/home/ubuntu/biweekly_panel.csv", low_memory=False)
    else:
        df = pd.read_csv(PANEL_PATH, low_memory=False)
    df['period_start_dt'] = pd.to_datetime(df['period_start'])
    
    # === TIME-VARYING VOTE FEATURES ===
    # Extract year from PDF filename (format: YYYY_meetingid_Minutes.pdf)
    # and build cumulative yea/nay counts per case, forward-filled into panel
    vote_margins_path = os.path.join(r"C:\Users\dhl\data\Thesis\thesis\Data\interim", "engineered_vote_margins.csv")
    if os.path.exists(vote_margins_path):
        votes_df = pd.read_csv(vote_margins_path)
        votes_df['year'] = votes_df['source_file'].str.extract(r'^(\d{4})').astype(int)
        # Aggregate by case + year: total yea and nay in that year's meetings
        yearly_votes = votes_df.groupby(['case_number', 'year']).agg(
            yea_this_year=('yea_votes', 'sum'),
            nay_this_year=('nay_votes', 'sum')
        ).reset_index()
        
        df['year'] = df['period_start_dt'].dt.year
        df = df.merge(yearly_votes, on=['case_number', 'year'], how='left')
        df['yea_this_year'] = df['yea_this_year'].fillna(0)
        df['nay_this_year'] = df['nay_this_year'].fillna(0)
        
        # Forward-fill: cumulative votes up to this period
        df = df.sort_values(['case_number', 'period_seq'])
        df['cumulative_yea_votes'] = df.groupby('case_number')['yea_this_year'].cumsum()
        df['cumulative_nay_votes'] = df.groupby('case_number')['nay_this_year'].cumsum()
        df['net_vote_margin'] = df['cumulative_yea_votes'] - df['cumulative_nay_votes']
    else:
        df['cumulative_yea_votes'] = 0
        df['cumulative_nay_votes'] = 0
        df['net_vote_margin'] = 0
    
    # === LEAKAGE FIX: Zero out NLP tokens where no council hearing occurred ===
    df['council_nlp_total_tokens'] = df['council_nlp_total_tokens'].where(
        df['council_hearings_this_period'] > 0, other=0
    )
    
    # === V22: LAGGED CUMULATIVE FEATURES (all leakage-free: use t-1 history only) ===
    df = df.sort_values(['case_number', 'period_seq'])
    
    df['cumulative_council_hearings_lag1'] = (
        df.groupby('case_number')['council_hearings_this_period']
        .apply(lambda x: x.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True)
    )
    df['cumulative_commission_hearings_lag1'] = (
        df.groupby('case_number')['commission_hearings_this_period']
        .apply(lambda x: x.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True)
    )
    df['cumulative_council_nlp_lag1'] = (
        df.groupby('case_number')['council_nlp_total_tokens']
        .apply(lambda x: x.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True)
    )
    
    # === STANDARD FEATURE ENGINEERING ===
    df['vote_friction'] = df['vote_event'] * (1 + df['cumulative_nay_votes'].clip(upper=10))
    df['cumulative_vote_friction'] = df.groupby('case_number')['vote_friction'].cumsum()
    df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
    
    targets = ["resolved", "cumulative_vote_friction", "net_height_change",
               "council_nlp_total_tokens", "commission_hearings_this_period", "council_hearings_this_period",
               "yea_votes_this_period", "nay_votes_this_period"]
               
    exclude = set(targets + [
        "case_number", "period_start", "period_start_dt", "year", "quarter", "petition_year", "petition_quarter",
        "latitude", "longitude", "shape_area", "council_district", "census_tract", "land_use_code",
        "label_petition_total_pct", "label_valid_protest", "label_real_days_in_pipeline", 
        "label_valid_petition_pct", "label_exact_geometric_petition_pct",
        "vote_event", "vote_friction", "yea_this_year", "nay_this_year", "censored",
        "cumulative_council_hearings", "cumulative_commission_hearings" # leaky versions
    ])
    
    features = [c for c in df.columns if c not in exclude]
    
    for f in features + targets:
        if f not in df.columns: df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    
    norm_dict = {}
    for f in features + ["net_height_change"]:
        if f in ["proposed_max_height_ft", "council_nlp_total_tokens", "net_vote_margin", "cumulative_yea_votes", "cumulative_nay_votes", "petition_pct_this_period", "cumulative_petition_pct"]:
            continue # Skip specific raw values that need custom handling below
        mean_v, std_v = df[f].mean(), df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)
        
    for f in ["proposed_max_height_ft", "council_nlp_total_tokens"]:
        df[f] = np.log1p(df[f].clip(lower=0))
    for f in ["commission_hearings_this_period", "council_hearings_this_period"]:
        df[f] = df[f].clip(lower=0, upper=1)
        
    # Build Cell Assignments
    first_periods = df.groupby("case_number").first()
    max_doses = df.groupby("case_number")["petition_pct_this_period"].max()
    
    cell_assignments = pd.DataFrame(index=first_periods.index)
    cell_assignments["height_bin"] = pd.qcut(first_periods["proposed_max_height_ft"], q=4, labels=False, duplicates='drop').fillna(0).astype(int)
    cell_assignments["dose_bin"] = pd.qcut(max_doses, q=4, labels=False, duplicates='drop').fillna(0).astype(int)
    cell_assignments["cell_id"] = cell_assignments["height_bin"].astype(str) + "_" + cell_assignments["dose_bin"].astype(str)
    
    unique_cells = cell_assignments["cell_id"].unique()
    
    return df, features, targets, norm_dict, cell_assignments, unique_cells

def build_tensors(df, features, targets, cases, max_seq=55):
    sub_df = df[df["case_number"].isin(cases)].sort_values(["case_number", "period_seq"])
    case_sizes = sub_df.groupby("case_number").size()
    c_list = case_sizes.index.values
    
    n_c = len(c_list)
    X_out = np.zeros((n_c, max_seq, len(features)), dtype=np.float32)
    Y_out = np.zeros((n_c, max_seq, len(targets)), dtype=np.float32)
    L_out = np.zeros(n_c, dtype=np.int64)
    
    feat_vals = sub_df[features].values.astype(np.float32)
    targ_vals = sub_df[targets].values.astype(np.float32)
    
    idx = 0
    for i, c in enumerate(c_list):
        size = case_sizes[c]
        length = min(size, max_seq)
        X_out[i, :length, :] = feat_vals[idx:idx+length]
        Y_out[i, :length, :] = targ_vals[idx:idx+length]
        L_out[i] = length
        idx += size
        
    return torch.from_numpy(X_out), torch.from_numpy(Y_out), torch.from_numpy(L_out)

# ============================================================
# MODELS
# ============================================================
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # Softmax attention weights over the input dimension
        self.grn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        weights = self.grn(x)
        return x * weights, weights

class Seq2SeqCVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, num_layers=2):
        super().__init__()
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim)
        
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_dim + hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        self.head_surv = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_vote = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_ht   = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_tok  = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_comm = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_coun = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_yea  = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_nay  = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, x):
        x_filtered, _ = self.vsn(x)
        _, (h, _) = self.encoder_lstm(x_filtered)
        return self.fc_mu(h[-1]), self.fc_logvar(h[-1])
        
    def decode(self, x, z):
        x_filtered, weights = self.vsn(x)
        z_expanded = self.z_proj(z).unsqueeze(1).expand(-1, x.size(1), -1)
        dec_in = torch.cat([x_filtered, z_expanded], dim=-1)
        h, _ = self.decoder_lstm(dec_in)
        out = torch.cat([
            self.head_surv(h), self.head_vote(h), self.head_ht(h), self.head_tok(h), 
            self.head_comm(h), self.head_coun(h), self.head_yea(h), self.head_nay(h)
        ], dim=-1)
        return out, weights
        
    def forward(self, x_pre, x_full):
        # Encode pre-intervention sequence (e.g. t=0 to 4)
        mu, logvar = self.encode(x_pre)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        # Decode the full sequence conditioned on Z
        preds, weights = self.decode(x_full, z)
        return preds, mu, logvar, weights

# ============================================================
# TRAINING LOGIC WITH EARLY STOPPING
# ============================================================
def train_seq2seq_cvae(model, X_train, Y_train, X_val, Y_val, epochs=50, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    PS_IDX = 6  # period_seq
    
    bce = nn.BCEWithLogitsLoss(reduction='none')
    mse = nn.MSELoss(reduction='none')
    best_loss = float('inf')
    patience = 7
    patience_counter = 0
    total_steps = max(1, epochs * (len(X_train) // 256))
    step = 0
    
    for ep in range(epochs):
        kl_beta = min(0.001, 0.001 * (step / (0.3 * total_steps)))
        model.train()
        for i in range(0, len(X_train), 256):
            bx, by = X_train[i:i+256].to(device), Y_train[i:i+256].to(device)
            # Pre-intervention sequence is the first 4 timesteps (t=0,1,2,3)
            bx_pre = bx[:, :4, :]
            
            opt.zero_grad()
            preds, mu, logvar, weights = model(bx_pre, bx)
            logvar = logvar.clamp(-10, 4)
            mask = (bx[:, :, PS_IDX] != 0).float()
            
            # Reconstruction Loss (Targets)
            l_surv = (bce(preds[:, :, 0], by[:, :, 0]) * mask).sum() / mask.sum()
            l_vote = (mse(preds[:, :, 1], by[:, :, 1]) * mask).sum() / mask.sum()
            l_ht   = (mse(preds[:, :, 2], by[:, :, 2]) * mask).sum() / mask.sum()
            l_tok  = (mse(preds[:, :, 3], by[:, :, 3]) * mask).sum() / mask.sum()
            l_comm = (mse(preds[:, :, 4], by[:, :, 4]) * mask).sum() / mask.sum()
            l_coun = (mse(preds[:, :, 5], by[:, :, 5]) * mask).sum() / mask.sum()
            l_yea  = (mse(preds[:, :, 6], by[:, :, 6]) * mask).sum() / mask.sum()
            l_nay  = (mse(preds[:, :, 7], by[:, :, 7]) * mask).sum() / mask.sum()
            
            # Sparsity penalty on attention weights to encourage aggressive feature selection
            l_sparse = 0.01 * torch.mean(torch.abs(weights))
            
            recon_loss = l_surv + l_vote + l_ht + l_tok + l_comm + l_coun + l_yea + l_nay
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kl_beta * kl_loss + l_sparse
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            else:
                opt.zero_grad()
                print(f"  [Seq2Seq] NaN loss at ep={ep} i={i}, skipping batch", flush=True)
            step += 1
            
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i in range(0, len(X_val), 256):
                bx, by = X_val[i:i+256].to(device), Y_val[i:i+256].to(device)
                bx_pre = bx[:, :4, :]
                preds, mu, logvar, _ = model(bx_pre, bx)
                mask = (bx[:, :, PS_IDX] != 0).float()
                
                l_surv = (bce(preds[:, :, 0], by[:, :, 0]) * mask).sum() / mask.sum()
                l_vote = (mse(preds[:, :, 1], by[:, :, 1]) * mask).sum() / mask.sum()
                l_ht   = (mse(preds[:, :, 2], by[:, :, 2]) * mask).sum() / mask.sum()
                l_tok  = (mse(preds[:, :, 3], by[:, :, 3]) * mask).sum() / mask.sum()
                l_comm = (mse(preds[:, :, 4], by[:, :, 4]) * mask).sum() / mask.sum()
                l_coun = (mse(preds[:, :, 5], by[:, :, 5]) * mask).sum() / mask.sum()
                l_yea  = (mse(preds[:, :, 6], by[:, :, 6]) * mask).sum() / mask.sum()
                l_nay  = (mse(preds[:, :, 7], by[:, :, 7]) * mask).sum() / mask.sum()
                
                recon_loss = l_surv + l_vote + l_ht + l_tok + l_comm + l_coun + l_yea + l_nay
                val_loss += recon_loss.item()
                
            val_loss /= max(1, len(X_val) // 256)
            
        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return model

def compute_and_print_metrics(split_name, model, X, Y, L, batch_size=256):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_absolute_error, r2_score
    from scipy.stats import spearmanr
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = X[i:i+batch_size].to(device)
            bx_pre = bx[:, :4, :]
            preds, _, _, _ = model(bx_pre, bx)
            all_preds.append(preds.cpu())
    
    preds = torch.cat(all_preds, dim=0)
    # Use period_seq (index 6) as mask: zero-padded steps have period_seq == 0
    PS_IDX = 6
    mask = (X[:, :, PS_IDX] != 0).cpu()
    
    y_true = Y.cpu()[mask].numpy()
    y_pred = preds[mask].numpy()
    
    if len(y_true) == 0:
        print(f"  > [{split_name} METRICS] Skipped — empty after mask.", flush=True)
        return
    
    def binary_metrics(y_t, y_p):
        if len(np.unique(y_t)) > 1:
            return roc_auc_score(y_t, y_p), average_precision_score(y_t, y_p), brier_score_loss(y_t, y_p)
        return 0.0, 0.0, 0.0
        
    # Binary Tasks (Sigmoid applied to logits)
    surv_roc, surv_pr, surv_brier = binary_metrics(y_true[:, 0], 1.0 / (1.0 + np.exp(-y_pred[:, 0])))
    
    # Continuous Tasks — MAE + R² + mean pred vs mean actual
    vote_mae  = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    vote_r2   = r2_score(y_true[:, 1], y_pred[:, 1])
    vote_corr = spearmanr(y_true[:, 1], y_pred[:, 1]).statistic
    vote_mean_actual = y_true[:, 1].mean()
    vote_mean_pred   = y_pred[:, 1].mean()
    
    ht_mae  = mean_absolute_error(y_true[:, 2], y_pred[:, 2])
    ht_r2   = r2_score(y_true[:, 2], y_pred[:, 2])
    tok_mae = mean_absolute_error(np.expm1(y_true[:, 3]), np.expm1(y_pred[:, 3]))
    comm_mae = mean_absolute_error(y_true[:, 4], y_pred[:, 4])
    coun_mae = mean_absolute_error(y_true[:, 5], y_pred[:, 5])
    
    print(f"  > [{split_name} METRICS] Surv: ROC {surv_roc:.3f} | PR {surv_pr:.3f} | Brier {surv_brier:.3f}", flush=True)
    print(f"  > [{split_name} METRICS] MAE | Vote: {vote_mae:.2f} | Height: {ht_mae:.1f}ft | Tokens: {tok_mae:.1f} | Comm: {comm_mae:.2f} | Coun: {coun_mae:.2f}", flush=True)
    print(f"  > [{split_name} METRICS] Vote R2: {vote_r2:.3f} | Vote Spearman: {vote_corr:.3f} | Vote mean(pred)={vote_mean_pred:.3f} vs mean(actual)={vote_mean_actual:.3f} | Ht R2: {ht_r2:.3f}", flush=True)

# ============================================================
# INFERENCE
# ============================================================
def run_counterfactual_inference(model, X_test, features, norm_dict):
    model.eval()
    pet_idx = features.index("petition_pct_this_period")
    cum_pet_idx = features.index("cumulative_petition_pct")
    
    # Feature indices for the autoregressive updates
    f_coun = features.index("cumulative_council_hearings_lag1")
    f_comm = features.index("cumulative_commission_hearings_lag1")
    f_tok = features.index("cumulative_council_nlp_lag1")
    f_yea = features.index("cumulative_yea_votes")
    f_nay = features.index("cumulative_nay_votes")
    f_margin = features.index("net_vote_margin")
    
    mean_coun, std_coun = norm_dict["cumulative_council_hearings_lag1"]
    mean_comm, std_comm = norm_dict["cumulative_commission_hearings_lag1"]
    mean_tok, std_tok = norm_dict["cumulative_council_nlp_lag1"]
    
    doses = np.linspace(0.0, 1.0, 11).tolist()
    results = {d: {"surv": np.zeros((X_test.size(0), 1)),
                   "vote": np.zeros((X_test.size(0), 1)),
                   "ht":   np.zeros((X_test.size(0), 1)),
                   "tok":  np.zeros((X_test.size(0), 1))} for d in doses}
                   
    with torch.no_grad():
        for d in doses:
            # 1. Clone the factual sequence
            X_t = X_test.clone().to(device)
            X_pre = X_t[:, :4, :]
            
            # 2. Inject Intervention at t=4 (period 5)
            X_t[:, 4, pet_idx] = d
            X_t[:, 4:, cum_pet_idx] = d
            
            # 3. Autoregressive Rollout from t=4 to 54
            for t in range(4, 54):
                # Predict current step
                preds, _, _, _ = model(X_pre, X_t)
                preds_t = preds[:, t, :] # (N, 8)
                
                # Extract predicted endogenous additions
                # targets: [resolved, vote_friction, net_height_change, tokens, comm, coun, yea, nay]
                pred_tok = torch.expm1(preds_t[:, 3]) # log1p was used, so expm1 to get linear
                pred_comm = torch.sigmoid(preds_t[:, 4]) # bounded 0-1
                pred_coun = torch.sigmoid(preds_t[:, 5])
                pred_yea = torch.relu(preds_t[:, 6]) # prevent negative votes
                pred_nay = torch.relu(preds_t[:, 7])
                
                # Unnormalize current state
                curr_coun = X_t[:, t, f_coun] * (std_coun + 1e-8) + mean_coun
                curr_comm = X_t[:, t, f_comm] * (std_comm + 1e-8) + mean_comm
                curr_tok = X_t[:, t, f_tok] * (std_tok + 1e-8) + mean_tok
                curr_yea = X_t[:, t, f_yea] # unscaled
                curr_nay = X_t[:, t, f_nay] # unscaled
                
                # Update state
                next_coun = curr_coun + pred_coun
                next_comm = curr_comm + pred_comm
                next_tok = curr_tok + pred_tok
                next_yea = curr_yea + pred_yea
                next_nay = curr_nay + pred_nay
                
                # Renormalize and assign to t+1
                X_t[:, t+1, f_coun] = (next_coun - mean_coun) / (std_coun + 1e-8)
                X_t[:, t+1, f_comm] = (next_comm - mean_comm) / (std_comm + 1e-8)
                X_t[:, t+1, f_tok] = (next_tok - mean_tok) / (std_tok + 1e-8)
                X_t[:, t+1, f_yea] = next_yea
                X_t[:, t+1, f_nay] = next_nay
                X_t[:, t+1, f_margin] = next_yea - next_nay
                
            # 4. Final Inference on Rolled-out Trajectory
            preds, _, _, _ = model(X_pre, X_t)
            
            # Extract cumulative outcomes
            results[d]["surv"][:, 0] = torch.sigmoid(preds[:, :, 0]).mean(dim=1).cpu().numpy()
            results[d]["vote"][:, 0] = preds[:, -1, 1].cpu().numpy()
            # net_height_change is native scale (linear difference)
            results[d]["ht"][:, 0] = preds[:, -1, 2].cpu().numpy()
            results[d]["tok"][:, 0] = torch.expm1(preds[:, :, 3]).sum(dim=1).cpu().numpy()

    return results

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60, flush=True)
    print("SPATIO-TEMPORAL G-COMPUTATION (DIAGONAL CV)", flush=True)
    print("=" * 60, flush=True)
    
    df, features, targets, norm_dict, cell_assignments, unique_cells = load_data_and_cells()
    first_periods_dt = df.groupby("case_number")["period_start_dt"].min()
    
    cutoffs = [2017, 2018, 2019, 2020, 2021]
    doses = np.linspace(0.0, 1.0, 11).tolist()
    
    pooled_results = {d: {"surv": [], "vote": [], "ht": [], "tok": []} for d in doses}
    
    gkf = GroupKFold(n_splits=5)
    
    # Generate the 5 volume-weighted cell splits
    all_cases = first_periods_dt.index.values
    all_groups = cell_assignments.loc[all_cases, "cell_id"].values
    
    for cutoff, (train_case_idx, test_case_idx) in zip(cutoffs, gkf.split(all_cases, groups=all_groups)):
        print("\n" + "=" * 50, flush=True)
        print(f"--- SPATIO-TEMPORAL FOLD: CUTOFF {cutoff} ---", flush=True)
        
        cutoff_date = pd.to_datetime(f"{cutoff}-12-31")
        end_test_date = pd.to_datetime(f"{cutoff+3}-12-31")
        
        # Identify the assigned OOD cells for this fold based on the cases
        train_cells = cell_assignments.loc[all_cases[train_case_idx], "cell_id"].unique()
        test_cells = cell_assignments.loc[all_cases[test_case_idx], "cell_id"].unique()
            
        # SPATIO-TEMPORAL SPLIT
        in_dist_train_mask = (first_periods_dt <= cutoff_date) & (cell_assignments["cell_id"].isin(train_cells))
        all_train_cases = first_periods_dt[in_dist_train_mask].index.values
        
        # IN-DISTRIBUTION VALIDATION SPLIT (Purely Random for Early Stopping)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(all_train_cases, groups=all_train_cases))
        train_cases = all_train_cases[train_idx]
        val_cases = all_train_cases[val_idx]
        
        test_cases = first_periods_dt[
            (first_periods_dt > cutoff_date) & 
            (first_periods_dt <= end_test_date) & 
            (cell_assignments["cell_id"].isin(test_cells))
        ].index.values
        
        if len(test_cases) == 0: continue
        
        # Guard against degenerate folds (Fold 3 had 96 train / 630 test)
        MIN_TRAIN = 200
        MIN_TEST  = 5
        if len(train_cases) < MIN_TRAIN or len(test_cases) < MIN_TEST:
            print(f"  [SKIP] Fold {cutoff}: train={len(train_cases)} < {MIN_TRAIN} or test={len(test_cases)} < {MIN_TEST} — degenerate split.", flush=True)
            continue

            
        X_train, Y_train, L_train = build_tensors(df, features, targets, train_cases)
        X_val, Y_val, L_val = build_tensors(df, features, targets, val_cases)
        X_test, Y_test, L_test = build_tensors(df, features, targets, test_cases)
        
        fold_idx = cutoffs.index(cutoff) + 1
        print(f"  [Model {fold_idx}/5] Train Cases: {len(train_cases)} | Val Cases: {len(val_cases)} | Test Cases (OOD/{cutoff+1}-{cutoff+3}): {len(test_cases)}", flush=True)
        
        print(f"  > Training Seq2Seq CVAE (Early Stopping)...", flush=True)
        model = Seq2SeqCVAE(len(features)).to(device)
        model = train_seq2seq_cvae(model, X_train, Y_train, X_val, Y_val, epochs=50, lr=1e-3)
        
        # OOD QUANTITATIVE EVALUATION (THE EXTRAPOLATION PROOF)
        print(f"\n  === EVALUATION PHASE ===", flush=True)
        # Note: We replaced the dual model with the unified Seq2SeqCVAE
        
        compute_and_print_metrics("TRAIN", model, X_train, Y_train, L_train)
        compute_and_print_metrics("VAL", model, X_val, Y_val, L_val)
        if len(X_test) > 0:
            compute_and_print_metrics("TEST", model, X_test, Y_test, L_test)
        
        # Counterfactual Inference strictly on the OOD & OOT Test Cases
        print(f"  > Running Inference on {len(test_cases)} OOD cases...", flush=True)
        cutoff_fold_res = run_counterfactual_inference(model, X_test, features, norm_dict)
            
        for d in doses:
            pooled_results[d]["surv"].append(cutoff_fold_res[d]["surv"])
            pooled_results[d]["vote"].append(cutoff_fold_res[d]["vote"])
            pooled_results[d]["ht"].append(cutoff_fold_res[d]["ht"])
            pooled_results[d]["tok"].append(cutoff_fold_res[d]["tok"])
            
        # Save final fold model weights for 3D analysis
        if cutoff == 2021:
            seq2seq_path = os.environ.get("SEQ2SEQ_CHECKPOINT", os.path.join(OUT_DIR, "causal_seq2seq_weights.pt"))
            torch.save(model.state_dict(), seq2seq_path)

    print("\n" + "=" * 60, flush=True)
    print("AGGREGATING ALL SPATIO-TEMPORAL FOLDS", flush=True)
    
    summary = []
    flattened_data = []
    
    for d in doses:
        if len(pooled_results[d]["surv"]) == 0: continue
        surv_all = np.concatenate(pooled_results[d]["surv"], axis=0)
        vote_all = np.concatenate(pooled_results[d]["vote"], axis=0)
        ht_all   = np.concatenate(pooled_results[d]["ht"], axis=0)
        tok_all  = np.concatenate(pooled_results[d]["tok"], axis=0)
        
        summary.append({
            "dose": d,
            "surv_p50": np.percentile(surv_all, 50),
            "surv_p10": np.percentile(surv_all, 10),
            "surv_p90": np.percentile(surv_all, 90),
            "ht_p50": np.percentile(ht_all, 50),
            "ht_p10": np.percentile(ht_all, 10),
            "ht_p90": np.percentile(ht_all, 90),
            "tok_p50": np.percentile(tok_all, 50),
            "tok_p10": np.percentile(tok_all, 10),
            "tok_p90": np.percentile(tok_all, 90),
        })
        
        flattened_data.append(pd.DataFrame({
            "dose": d,
            "surv_mean": surv_all.mean(axis=1),
            "vote_mean": vote_all.mean(axis=1),
            "ht_mean": ht_all.mean(axis=1),
            "tok_mean": tok_all.mean(axis=1)
        }))
        
    sum_df = pd.DataFrame(summary)
    df_results = pd.concat(flattened_data)
    
    df_results.to_csv(CAUSAL_SURFACE_PATH, index=False)
    
    # Save the final trained weights for local plotting
    seq2seq_path = os.environ.get("SEQ2SEQ_CHECKPOINT", os.path.join(OUT_DIR, "causal_seq2seq_weights.pt"))
    torch.save(model.state_dict(), seq2seq_path)
    
    print(f"\n[√] Final God Table materialized at: {CAUSAL_SURFACE_PATH}", flush=True)
    print("\n--- POOLED CAUSAL FRICTION SURFACE (P10 - P50 - P90) ---", flush=True)
    print(sum_df[["dose", "surv_p50", "ht_p50", "tok_p50"]].to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
