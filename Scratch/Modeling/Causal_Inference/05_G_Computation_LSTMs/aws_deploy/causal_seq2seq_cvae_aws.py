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

OUT_DIR = os.environ.get("OUT_DIR", ".")
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
    if 'petition_pct_this_period' in df.columns:
        df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
    
    # === DYNAMIC TARGET ENGINEERING (Time-Varying Mediators) ===
    # Compute the dynamic running concession at time t (how much the developer has conceded SO FAR).
    # 1. Anchor: Initial maximum height requested by the developer
    if "pdf_requested_height_ft" in df.columns:
        initial_req = df.groupby("case_number")["pdf_requested_height_ft"].transform("max")
        # 2. Current Constraint at time t: Minimum of current request and staff recommendation.
        current_constraint = df[["pdf_requested_height_ft", "pdf_staff_recommends_ht"]].min(axis=1) if "pdf_staff_recommends_ht" in df.columns else df["pdf_requested_height_ft"]
        # Default to initial request if no constraint exists yet (concession = 0)
        current_constraint = current_constraint.fillna(initial_req)
        
        final_ht = df["pdf_reduced_to_ft"].fillna(current_constraint).fillna(0) if "pdf_reduced_to_ft" in df.columns else current_constraint.fillna(0)
        # 3. Time-varying dynamic target
        df["net_height_change"] = (initial_req - final_ht).clip(lower=0).fillna(0)
    else:
        df["net_height_change"] = 0

    targets = ["resolved", "cumulative_vote_friction", "net_height_change",
               "council_nlp_total_tokens", "commission_hearings_this_period", "council_hearings_this_period",
               "yea_votes_this_period", "nay_votes_this_period"]
               
    exclude = set(targets + [
        "case_number", "period_start", "period_start_dt", "year", "quarter", "petition_year", "petition_quarter",
        "latitude", "longitude", "shape_area", "council_district", "census_tract", "land_use_code",
        "label_petition_total_pct", "label_valid_protest", "label_real_days_in_pipeline", 
        "label_valid_petition_pct", "label_exact_geometric_petition_pct",
        "vote_event", "vote_friction", "yea_this_year", "nay_this_year", "censored",
        "cumulative_council_hearings", "cumulative_commission_hearings", # leaky versions
        "Aggregate_Sentiment", # Leaky and zero-inflated target, replaced by nlp_oppose_hits
        "max_opponent_experience" # Degenerate (pure zeros) due to missing bipartite graph
    ])
    
    features = [c for c in df.columns if c not in exclude]
    
    for f in features + targets:
        if f not in df.columns: df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    
    # === OUTLIER / TYPO CLEANING ===
    if "building_age" in df.columns:
        # Fix 0 AD year built typoes (resulting in 2019 year building_age)
        df.loc[df["building_age"] > 250, "building_age"] = df.loc[df["building_age"] <= 250, "building_age"].mean()
    if "yr_built" in df.columns:
        # Fix the underlying yr_built as well
        df.loc[(df["yr_built"] < 1850) & (df["yr_built"] > 0), "yr_built"] = df.loc[df["yr_built"] >= 1850, "yr_built"].mean()
    if "petition_pct_this_period" in df.columns:
        df["petition_pct_this_period"] = df["petition_pct_this_period"].clip(upper=100.0)
    if "cumulative_petition_pct" in df.columns:
        df["cumulative_petition_pct"] = df["cumulative_petition_pct"].clip(upper=100.0)
        
    norm_dict = {}
    for f in features + ["net_height_change"]:
        if f in ["pdf_requested_height_ft", "council_nlp_total_tokens", "net_vote_margin", "cumulative_yea_votes", "cumulative_nay_votes", "petition_pct_this_period", "cumulative_petition_pct"]:
            continue # Skip specific raw values that need custom handling below
        
        # Apply Log Transform to massively right-skewed real estate and physical constraints
        if f in ["land_acres", "market_value", "appraised_value"]:
            df[f] = np.log1p(df[f].clip(lower=0))
            
        mean_v, std_v = df[f].mean(), df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)
        
    for f in ["pdf_requested_height_ft", "council_nlp_total_tokens"]:
        df[f] = np.log1p(df[f].clip(lower=0))
    for f in ["commission_hearings_this_period", "council_hearings_this_period"]:
        df[f] = df[f].clip(lower=0, upper=1)
        
    # Build Cell Assignments (Distributed Spatial Micro-Cells)
    first_periods = df.groupby("case_number").first()
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Extract Spatial + Feature variables for 'Spatio-Structural' clustering!
    features_for_clustering = first_periods[["latitude", "longitude"]].copy()
    if "proposed_max_far" in first_periods.columns:
        features_for_clustering["far"] = first_periods["proposed_max_far"].fillna(0)
    features_for_clustering["dose"] = max_doses = df.groupby("case_number")["petition_pct_this_period"].max().fillna(0)
    
    # Scale them so Latitude/Longitude degrees don't completely overpower FAR/Dose magnitudes
    scaled_coords = StandardScaler().fit_transform(features_for_clustering.fillna(0).values)
    
    # Cluster into 50 Spatio-Structural micro-neighborhoods (Automatically determines best ratios)
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_coords)
    
    cell_assignments = pd.DataFrame(index=first_periods.index)
    cell_assignments["cell_id"] = cluster_labels
    
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
    targ_vals = sub_df[targets].fillna(0).values.astype(np.float32)
    
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
        # Scale by feature dimension to preserve original variance (TFT standard practice)
        # Otherwise Softmax shrinks all structural features by 1/N, causing treatments to dominate
        return x * weights * x.shape[-1], weights

class Seq2SeqCVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, num_layers=2, treat_idx=None, confounder_idx=None):
        super().__init__()
        self.treat_idx = treat_idx if treat_idx is not None else []
        self.confounder_idx = confounder_idx if confounder_idx is not None else []
        self.struct_dim = input_dim - len(self.treat_idx)
        
        # VSN now only operates on structural features
        self.vsn = VariableSelectionNetwork(self.struct_dim, hidden_dim)
        
        # The LSTM receives structural features (from VSN) + treatment features directly concatenated
        self.encoder_lstm = nn.LSTM(self.struct_dim + len(self.treat_idx), hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(self.struct_dim + len(self.treat_idx) + hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        self.head_surv = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_vote = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_ht   = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_tok  = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_comm = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_coun = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_yea  = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_nay  = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

        # Direct residual injection from treatment features to output heads
        if len(self.treat_idx) > 0:
            self.treatment_residual = nn.Linear(len(self.treat_idx), 8)
        else:
            self.treatment_residual = None

    def _split_features(self, x):
        if self.training and len(self.confounder_idx) > 0:
            # Apply 20% targeted dropout to confounders to prevent OOD rejection manifold constraints
            if torch.rand(1).item() < 0.2:
                x = x.clone() # avoid inplace mutation
                x[:, :, self.confounder_idx] = 0.0

        if not self.treat_idx:
            return x, None
        
        # Keep structural features and treatment features separate
        struct_mask = torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
        struct_mask[self.treat_idx] = False
        return x[:, :, struct_mask], x[:, :, self.treat_idx]

    def encode(self, x):
        x_struct, x_treat = self._split_features(x)
        x_filtered, _ = self.vsn(x_struct)
        
        if x_treat is not None:
            lstm_in = torch.cat([x_filtered, x_treat], dim=-1)
        else:
            lstm_in = x_filtered
            
        _, (h, _) = self.encoder_lstm(lstm_in)
        return self.fc_mu(h[-1]), self.fc_logvar(h[-1])
        
    def decode(self, x, z):
        x_struct, x_treat = self._split_features(x)
        x_filtered, weights = self.vsn(x_struct)
        
        z_expanded = self.z_proj(z).unsqueeze(1).expand(-1, x.size(1), -1)
        
        if x_treat is not None:
            dec_in = torch.cat([x_filtered, x_treat, z_expanded], dim=-1)
        else:
            dec_in = torch.cat([x_filtered, z_expanded], dim=-1)
            
        h, _ = self.decoder_lstm(dec_in)
        
        out = torch.cat([
            self.head_surv(h), self.head_vote(h), self.head_ht(h), self.head_tok(h), 
            self.head_comm(h), self.head_coun(h), self.head_yea(h), self.head_nay(h)
        ], dim=-1)
        
        # Inject the treatment signal directly into the outputs, bypassing LSTM and VAE
        if self.treatment_residual is not None and x_treat is not None:
            out = out + self.treatment_residual(x_treat)
            
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
def train_seq2seq_cvae(model, X_train, Y_train, X_val, Y_val, features, epochs=50, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    PS_IDX = features.index("period_seq")
    
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
        step_count = 0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        for i in range(0, len(X_train), 256):
            bx, by = X_train[i:i+256].to(device), Y_train[i:i+256].to(device)
            # Pre-intervention sequence is the first 4 timesteps (t=0,1,2,3)
            bx_pre = bx[:, :4, :]
            
            opt.zero_grad()
            preds, mu, logvar, weights = model(bx_pre, bx)
            logvar = logvar.clamp(-10, 4)
            mask = (bx[:, :, PS_IDX] != 0).float()
            
            # Reconstruction Loss (Targets)
            # Use .clamp(min=1.0) on the denominator to prevent division by zero NaNs if a batch has no active sequences.
            l_surv = (bce(preds[:, :, 0], by[:, :, 0]) * mask).sum() / mask.sum().clamp(min=1.0)
            l_vote = (mse(preds[:, :, 1], by[:, :, 1]) * mask).sum() / mask.sum().clamp(min=1.0)
            l_ht   = (mse(preds[:, :, 2], by[:, :, 2]) * mask).sum() / mask.sum().clamp(min=1.0)
            l_tok  = (mse(preds[:, :, 3], by[:, :, 3]) * mask).sum() / mask.sum().clamp(min=1.0)
            l_comm = (mse(preds[:, :, 4], by[:, :, 4]) * mask).sum() / mask.sum().clamp(min=1.0)
            l_coun = (mse(preds[:, :, 5], by[:, :, 5]) * mask).sum() / mask.sum().clamp(min=1.0)
            l_yea  = (mse(preds[:, :, 6], by[:, :, 6]) * mask).sum() / mask.sum().clamp(min=1.0)
            l_nay  = (mse(preds[:, :, 7], by[:, :, 7]) * mask).sum() / mask.sum().clamp(min=1.0)
            
            # Sparsity penalty on attention weights to encourage aggressive feature selection
            l_sparse = 0.01 * torch.mean(torch.abs(weights))
            
            recon_loss = l_surv + l_vote + l_ht + l_tok + l_comm + l_coun + l_yea + l_nay
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kl_beta * kl_loss + l_sparse
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                step_count += 1
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
                
                l_surv = (bce(preds[:, :, 0], by[:, :, 0]) * mask).sum() / mask.sum().clamp(min=1.0)
                l_vote = (mse(preds[:, :, 1], by[:, :, 1]) * mask).sum() / mask.sum().clamp(min=1.0)
                l_ht   = (mse(preds[:, :, 2], by[:, :, 2]) * mask).sum() / mask.sum().clamp(min=1.0)
                l_tok  = (mse(preds[:, :, 3], by[:, :, 3]) * mask).sum() / mask.sum().clamp(min=1.0)
                l_comm = (mse(preds[:, :, 4], by[:, :, 4]) * mask).sum() / mask.sum().clamp(min=1.0)
                l_coun = (mse(preds[:, :, 5], by[:, :, 5]) * mask).sum() / mask.sum().clamp(min=1.0)
                l_yea  = (mse(preds[:, :, 6], by[:, :, 6]) * mask).sum() / mask.sum().clamp(min=1.0)
                l_nay  = (mse(preds[:, :, 7], by[:, :, 7]) * mask).sum() / mask.sum().clamp(min=1.0)
                
                recon_loss = l_surv + l_vote + l_ht + l_tok + l_comm + l_coun + l_yea + l_nay
                val_loss += recon_loss.item()
                
            val_loss /= max(1, len(X_val) // 256)
            
        train_recon_avg = epoch_recon_loss / max(1, step_count)
        train_kl_avg = epoch_kl_loss / max(1, step_count)
        print(f"  Epoch [{ep}/{epochs}] | Train Recon: {train_recon_avg:.4f} | KL: {train_kl_avg:.4f} (beta={kl_beta:.4f}) | Val Recon: {val_loss:.4f}", flush=True)
            
        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  > Early stopping triggered at epoch {ep}", flush=True)
                break
    return model

def compute_and_print_metrics(split_name, model, X, Y, L, features, batch_size=256):
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
    PS_IDX = features.index("period_seq")
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

def compute_autoregressive_metrics(split_name, model, X, Y, L, features, norm_dict):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_absolute_error, r2_score
    from scipy.stats import spearmanr
    model.eval()
    
    # Feature indices
    f_coun = features.index("cumulative_council_hearings_lag1")
    f_comm = features.index("cumulative_commission_hearings_lag1")
    f_tok = features.index("cumulative_council_nlp_lag1")
    f_yea = features.index("cumulative_yea_votes")
    f_nay = features.index("cumulative_nay_votes")
    f_margin = features.index("net_vote_margin")
    
    mean_coun, std_coun = norm_dict["cumulative_council_hearings_lag1"]
    mean_comm, std_comm = norm_dict["cumulative_commission_hearings_lag1"]
    mean_tok, std_tok = norm_dict["cumulative_council_nlp_lag1"]
    mean_ht, std_ht = norm_dict["net_height_change"]
    
    y_true = np.zeros((len(X), 8))
    y_pred = np.zeros((len(X), 8))
    
    with torch.no_grad():
        X_t = X.clone().to(device)
        X_pre = X_t[:, :4, :]
        
        # Autoregressive Rollout from t=4 to 54
        for t in range(4, 54):
            preds, _, _, _ = model(X_pre, X_t)
            preds_t = preds[:, t, :]
            
            pred_tok = torch.expm1(preds_t[:, 3])
            pred_comm = torch.clamp(preds_t[:, 4], 0, 1)
            pred_coun = torch.clamp(preds_t[:, 5], 0, 1)
            pred_yea = torch.relu(preds_t[:, 6])
            pred_nay = torch.relu(preds_t[:, 7])
            
            curr_coun = X_t[:, t, f_coun] * (std_coun + 1e-8) + mean_coun
            curr_comm = X_t[:, t, f_comm] * (std_comm + 1e-8) + mean_comm
            curr_yea = X_t[:, t, f_yea]
            curr_nay = X_t[:, t, f_nay]
            
            next_coun = curr_coun + pred_coun
            next_comm = curr_comm + pred_comm
            next_tok = pred_tok
            next_yea = curr_yea + pred_yea
            next_nay = curr_nay + pred_nay
            
            X_t[:, t+1, f_coun] = (next_coun - mean_coun) / (std_coun + 1e-8)
            X_t[:, t+1, f_comm] = (next_comm - mean_comm) / (std_comm + 1e-8)
            X_t[:, t+1, f_tok] = (next_tok - mean_tok) / (std_tok + 1e-8)
            X_t[:, t+1, f_yea] = next_yea
            X_t[:, t+1, f_nay] = next_nay
            X_t[:, t+1, f_margin] = next_yea - next_nay
            
        # Final pass
        preds, _, _, _ = model(X_pre, X_t)
        
        for i in range(len(X)):
            valid_len = (X_t[i, :, features.index("period_seq")] != 0).sum()
            idx = max(0, valid_len - 1)
            y_true[i] = Y[i, idx, :].cpu().numpy()
            y_pred[i] = preds[i, idx, :].cpu().numpy()
            
            # Correct specific units in prediction
            y_pred[i, 0] = 1.0 / (1.0 + np.exp(-y_pred[i, 0])) # Surv sigmoid
            y_pred[i, 2] = (y_pred[i, 2] * std_ht) + mean_ht # Height un-normalize
            y_pred[i, 3] = np.expm1(y_pred[i, 3]) # Token linear
            
            # Correct targets
            y_true[i, 3] = np.expm1(y_true[i, 3])
            
    # Binary
    if len(np.unique(y_true[:, 0])) > 1:
        surv_roc = roc_auc_score(y_true[:, 0], y_pred[:, 0])
        surv_pr = average_precision_score(y_true[:, 0], y_pred[:, 0])
    else:
        surv_roc = surv_pr = 0.0
        
    vote_mae = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    vote_corr = spearmanr(y_true[:, 1], y_pred[:, 1]).statistic
    ht_mae = mean_absolute_error(y_true[:, 2], y_pred[:, 2])
    tok_mae = mean_absolute_error(y_true[:, 3], y_pred[:, 3])
    
    print(f"  > [{split_name} METRICS] Surv: ROC {surv_roc:.3f} | PR {surv_pr:.3f}", flush=True)
    print(f"  > [{split_name} METRICS] MAE | Vote: {vote_mae:.2f} | Height: {ht_mae:.1f}ft | Tokens: {tok_mae:.1f}", flush=True)
    print(f"  > [{split_name} METRICS] Vote Spearman: {vote_corr:.3f}", flush=True)

# ============================================================
# INFERENCE
# ============================================================

def run_counterfactual_inference(model, X_test, features, norm_dict, mc_samples=25):
    model.eval()
    if len(X_test) == 0:
        return {d: {"surv": np.array([]), "vote": np.array([]), "ht": np.array([]), "tok": np.array([]), "time_series_surv": [], "time_series_vote": [], "time_series_ht": []} for d in np.linspace(0.0, 1.0, 11).tolist()}
        
    pet_idx = features.index("petition_pct_this_period")
    cum_pet_idx = features.index("cumulative_petition_pct")
    
    # Feature indices for the autoregressive updates
    f_coun = features.index("cumulative_council_hearings_lag1")
    f_comm = features.index("cumulative_commission_hearings_lag1")
    f_tok = features.index("cumulative_council_nlp_lag1")
    f_yea = features.index("cumulative_yea_votes")
    f_nay = features.index("cumulative_nay_votes")
    f_margin = features.index("net_vote_margin")
    f_req = features.index("pdf_requested_height_ft")
    
    mean_coun, std_coun = norm_dict["cumulative_council_hearings_lag1"]
    mean_comm, std_comm = norm_dict["cumulative_commission_hearings_lag1"]
    mean_tok, std_tok = norm_dict["cumulative_council_nlp_lag1"]
    mean_ht, std_ht = norm_dict["net_height_change"]
    
    # Calculate native requested height
    native_req_ht = np.expm1(X_test[:, 0, f_req].cpu().numpy())
    # Identify cases actively seeking height
    upzone_mask = native_req_ht > 35.0  # Assumes standard base height is ~35ft
    
    # Vectorize Monte Carlo Clones across the batch dimension
    N_cases = X_test.size(0)
    X_test_mc = X_test.unsqueeze(1).repeat(1, mc_samples, 1, 1).view(N_cases * mc_samples, X_test.size(1), X_test.size(2))
    
    doses = np.linspace(0.0, 1.0, 11).tolist()
    results = {d: {"surv": np.zeros((N_cases, 1)),
                   "vote": np.zeros((N_cases, 1)),
                   "ht":   np.zeros((N_cases, 1)),
                   "ht_pct": np.zeros((N_cases, 1)),
                   "tok":  np.zeros((N_cases, 1)),
                   "time_series_surv": np.zeros((N_cases, 55)),
                   "time_series_vote": np.zeros((N_cases, 55)),
                   "time_series_ht": np.zeros((N_cases, 55)),
                   "time_series_ht_pct": np.zeros((N_cases, 55))} for d in doses}
                   
    with torch.no_grad():
        for d in doses:
            # 1. Clone the factual sequence across the MC batch
            X_t = X_test_mc.clone().to(device)
            X_pre = X_t[:, :4, :]
            
            # 2. Inject Intervention at t=4 (period 5)
            X_t[:, 4, pet_idx] = d
            X_t[:, 4:, cum_pet_idx] = d
            
            # Setup time-series loggers
            ts_surv = np.zeros((N_cases * mc_samples, 55))
            ts_vote = np.zeros((N_cases * mc_samples, 55))
            ts_ht   = np.zeros((N_cases * mc_samples, 55))
            ts_ht_pct = np.zeros((N_cases * mc_samples, 55))
            
            # Repeat native_req_ht across MC dimension to calculate intermediate percentages
            native_req_ht_mc = np.repeat(native_req_ht, mc_samples)
            
            # 3. Autoregressive Rollout from t=4 to 54
            for t in range(4, 54):
                # Predict current step
                preds, _, _, _ = model(X_pre, X_t)
                preds_t = preds[:, t, :] # (N*MC, 8)
                
                # Log intermediate metrics for time-series trajectory extraction
                ts_surv[:, t] = torch.sigmoid(preds_t[:, 0]).cpu().numpy()
                ts_vote[:, t] = preds_t[:, 1].cpu().numpy()
                ts_ht_val     = (preds_t[:, 2].cpu().numpy() * std_ht) + mean_ht
                ts_ht[:, t]   = ts_ht_val
                
                # Height % Achieved (Avoid division by zero with np.where)
                ts_ht_pct[:, t] = np.where(native_req_ht_mc > 0, (native_req_ht_mc + ts_ht_val) / native_req_ht_mc, 1.0)
                
                # Extract predicted endogenous additions
                pred_tok = torch.expm1(preds_t[:, 3])
                pred_comm = torch.sigmoid(preds_t[:, 4])
                pred_coun = torch.sigmoid(preds_t[:, 5])
                pred_yea = torch.relu(preds_t[:, 6])
                pred_nay = torch.relu(preds_t[:, 7])
                
                # Unnormalize current state
                curr_coun = X_t[:, t, f_coun] * (std_coun + 1e-8) + mean_coun
                curr_comm = X_t[:, t, f_comm] * (std_comm + 1e-8) + mean_comm
                curr_yea = X_t[:, t, f_yea]
                curr_nay = X_t[:, t, f_nay]
                
                # Update state
                next_coun = curr_coun + pred_coun
                next_comm = curr_comm + pred_comm
                next_tok = pred_tok
                next_yea = curr_yea + pred_yea
                next_nay = curr_nay + pred_nay
                
                # Renormalize and assign to t+1
                X_t[:, t+1, f_coun] = (next_coun - mean_coun) / (std_coun + 1e-8)
                X_t[:, t+1, f_comm] = (next_comm - mean_comm) / (std_comm + 1e-8)
                X_t[:, t+1, f_tok]  = (next_tok - mean_tok) / (std_tok + 1e-8)
                X_t[:, t+1, f_yea]  = next_yea
                X_t[:, t+1, f_nay]  = next_nay
                X_t[:, t+1, f_margin] = next_yea - next_nay
                
            # 4. Final Inference on Rolled-out Trajectory
            preds, _, _, _ = model(X_pre, X_t)
            
            # Extract cumulative outcomes for (N*MC)
            mc_surv = torch.sigmoid(preds[:, :, 0]).mean(dim=1).cpu().numpy()
            mc_vote = preds[:, -1, 1].cpu().numpy()
            mc_ht   = (preds[:, -1, 2].cpu().numpy() * std_ht) + mean_ht
            mc_tok  = torch.expm1(preds[:, -1, 3]).cpu().numpy()
            mc_ht_pct = np.where(native_req_ht_mc > 0, (native_req_ht_mc + mc_ht) / native_req_ht_mc, 1.0)
            
            # Aggregate across the Monte Carlo clones via mean()
            results[d]["surv"][:, 0] = mc_surv.reshape(N_cases, mc_samples).mean(axis=1)
            results[d]["vote"][:, 0] = mc_vote.reshape(N_cases, mc_samples).mean(axis=1)
            results[d]["ht"][:, 0]   = mc_ht.reshape(N_cases, mc_samples).mean(axis=1)
            results[d]["tok"][:, 0]  = mc_tok.reshape(N_cases, mc_samples).mean(axis=1)
            
            # Condition Height Percentage to ONLY Upzone Cases
            ht_pct_mean = mc_ht_pct.reshape(N_cases, mc_samples).mean(axis=1)
            results[d]["ht_pct"][:, 0] = np.where(upzone_mask, ht_pct_mean, np.nan)
            
            # Aggregate the time-series logs (N_cases, 55)
            results[d]["time_series_surv"] = ts_surv.reshape(N_cases, mc_samples, 55).mean(axis=1)
            results[d]["time_series_vote"] = ts_vote.reshape(N_cases, mc_samples, 55).mean(axis=1)
            results[d]["time_series_ht"]   = ts_ht.reshape(N_cases, mc_samples, 55).mean(axis=1)
            
            # Condition the time-series percentage
            ts_ht_pct_mean = ts_ht_pct.reshape(N_cases, mc_samples, 55).mean(axis=1)
            results[d]["time_series_ht_pct"] = np.where(upzone_mask[:, None], ts_ht_pct_mean, np.nan)

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
    
    pooled_results = {d: {"surv": [], "vote": [], "ht": [], "tok": [], "ht_pct": [],
                          "time_series_surv": [], "time_series_vote": [], "time_series_ht": [], "time_series_ht_pct": []} for d in doses}
    
    gkf = GroupKFold(n_splits=5)
    
    all_cases = first_periods_dt.index.values
    all_groups = cell_assignments.loc[all_cases, "cell_id"].values
    
    print("DEBUG ALL GROUPS LEN:", len(all_groups), "UNIQUE:", len(np.unique(all_groups)), flush=True)
    
    for fold_idx, (cutoff, (train_case_idx, test_case_idx)) in enumerate(zip(cutoffs, gkf.split(all_cases, groups=all_groups))):
        print("\n" + "=" * 50, flush=True)
        print(f"--- SPATIO-STRUCTURAL FOLD: CUTOFF {cutoff} ---", flush=True)
        
        cutoff_date = pd.to_datetime(f"{cutoff}-12-31")
        end_test_date = pd.to_datetime(f"{cutoff+3}-12-31")
        
        # Identify the assigned OOD micro-cells for this fold
        train_cells = cell_assignments.loc[all_cases[train_case_idx], "cell_id"].unique()
        test_cells = cell_assignments.loc[all_cases[test_case_idx], "cell_id"].unique()
            
        # SPATIO-TEMPORAL SPLIT (Train on 40 spatial cells <= cutoff)
        in_dist_train_mask = (first_periods_dt <= cutoff_date) & (cell_assignments["cell_id"].isin(train_cells))
        all_train_cases = first_periods_dt[in_dist_train_mask].index.values
        
        if len(all_train_cases) < 50:
            print(f"  [SKIP] Fold {cutoff}: Not enough historical training data ({len(all_train_cases)} < 50).", flush=True)
            continue
            
        # IN-DISTRIBUTION VALIDATION SPLIT (Purely Random for Early Stopping)
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(all_train_cases, groups=all_train_cases))
        train_cases = all_train_cases[train_idx]
        val_cases = all_train_cases[val_idx]
        
        # SPATIO-TEMPORAL OOD TEST (Test on 10 OOD spatial cells > cutoff)
        test_cases = first_periods_dt[
            (first_periods_dt > cutoff_date) & 
            (first_periods_dt <= end_test_date) &
            (cell_assignments["cell_id"].isin(test_cells))
        ].index.values
        
        if len(test_cases) < 5:
            print(f"  [SKIP] Fold {cutoff}: Not enough future test cases ({len(test_cases)} < 5).", flush=True)
            continue

            
        X_train, Y_train, L_train = build_tensors(df, features, targets, train_cases)
        X_val, Y_val, L_val = build_tensors(df, features, targets, val_cases)
        X_test, Y_test, L_test = build_tensors(df, features, targets, test_cases)
        
        print(f"  [Model {fold_idx + 1}/5] Train Cases: {len(train_cases)} | Val Cases: {len(val_cases)} | Test Cases (OOD/{cutoff+1}-{cutoff+3}): {len(test_cases)}", flush=True)
        
        treat_idx = [
            features.index("petition_pct_this_period"), 
            features.index("cumulative_petition_pct")
        ]
        
        confounder_idx = []
        if "proposed_max_far" in features:
            confounder_idx.append(features.index("proposed_max_far"))
        if "pdf_requested_height_ft" in features:
            confounder_idx.append(features.index("pdf_requested_height_ft"))
        if "land_acres" in features:
            confounder_idx.append(features.index("land_acres"))
            
        print(f"  > Training Seq2Seq CVAE (Early Stopping)...", flush=True)
        model = Seq2SeqCVAE(
            len(features), 
            treat_idx=treat_idx, 
            confounder_idx=confounder_idx
        ).to(device)
        model = train_seq2seq_cvae(model, X_train, Y_train, X_val, Y_val, features, epochs=50, lr=1e-3)
        
        # OOD QUANTITATIVE EVALUATION (THE EXTRAPOLATION PROOF)
        print(f"\n  === EVALUATION PHASE ===", flush=True)
        # Note: We replaced the dual model with the unified Seq2SeqCVAE
        
        # Output 1-step Teacher Forced Metrics
        compute_and_print_metrics(f"VAL", model, X_val, Y_val, L_val, features)
        compute_and_print_metrics(f"TEST", model, X_test, Y_test, L_test, features)
        
        # Output Full Autoregressive Rollout Metrics
        compute_autoregressive_metrics(f"AUTOREGRESSIVE TEST", model, X_test, Y_test, L_test, features, norm_dict)
        
        # Counterfactual Inference strictly on the OOD & OOT Test Cases
        print(f"  > Running Inference on {len(test_cases)} OOD cases...", flush=True)
        cutoff_fold_res = run_counterfactual_inference(model, X_test, features, norm_dict)
            
        for d in doses:
            pooled_results[d]["surv"].append(cutoff_fold_res[d]["surv"])
            pooled_results[d]["vote"].append(cutoff_fold_res[d]["vote"])
            pooled_results[d]["ht"].append(cutoff_fold_res[d]["ht"])
            pooled_results[d]["ht_pct"].append(cutoff_fold_res[d]["ht_pct"])
            pooled_results[d]["tok"].append(cutoff_fold_res[d]["tok"])
            
            # Append Time-Series Deltas if they exist
            if "time_series_surv" in cutoff_fold_res[d] and len(cutoff_fold_res[d]["time_series_surv"]) > 0:
                pooled_results[d]["time_series_surv"].append(cutoff_fold_res[d]["time_series_surv"])
                pooled_results[d]["time_series_vote"].append(cutoff_fold_res[d]["time_series_vote"])
                pooled_results[d]["time_series_ht"].append(cutoff_fold_res[d]["time_series_ht"])
                pooled_results[d]["time_series_ht_pct"].append(cutoff_fold_res[d]["time_series_ht_pct"])
            
        # Save final fold model weights for 3D analysis
        if cutoff == 2021:
            seq2seq_path = os.environ.get("SEQ2SEQ_CHECKPOINT", os.path.join(OUT_DIR, "causal_seq2seq_weights.pt"))
            torch.save(model.state_dict(), seq2seq_path)

    print("\n" + "=" * 60, flush=True)
    print("AGGREGATING ALL SPATIO-TEMPORAL FOLDS", flush=True)
    
    summary = []
    flattened_data = []
    ts_summary = []
    
    for d in doses:
        if len(pooled_results[d]["surv"]) == 0: continue
        surv_all = np.concatenate(pooled_results[d]["surv"], axis=0)
        vote_all = np.concatenate(pooled_results[d]["vote"], axis=0)
        ht_all   = np.concatenate(pooled_results[d]["ht"], axis=0)
        tok_all  = np.concatenate(pooled_results[d]["tok"], axis=0)
        
        # Strip NaNs for the conditional Height Percentage
        ht_pct_all = np.concatenate(pooled_results[d]["ht_pct"], axis=0)
        ht_pct_valid = ht_pct_all[~np.isnan(ht_pct_all)]
        
        summary.append({
            "dose": d,
            "surv_p50": np.percentile(surv_all, 50),
            "surv_p10": np.percentile(surv_all, 10),
            "surv_p90": np.percentile(surv_all, 90),
            "ht_p50": np.percentile(ht_all, 50),
            "ht_p10": np.percentile(ht_all, 10),
            "ht_p90": np.percentile(ht_all, 90),
            "ht_pct_p50": np.percentile(ht_pct_valid, 50) if len(ht_pct_valid) > 0 else np.nan,
            "tok_p50": np.percentile(tok_all, 50),
        })
        
        flattened_data.append(pd.DataFrame({
            "dose": d,
            "surv_mean": surv_all.mean(axis=1),
            "vote_mean": vote_all.mean(axis=1),
            "ht_mean": ht_all.mean(axis=1),
            "tok_mean": tok_all.mean(axis=1)
        }))
        
        # Pool the Time-Series data to calculate average trajectory per dose
        if len(pooled_results[d]["time_series_surv"]) > 0:
            ts_surv_avg = np.concatenate(pooled_results[d]["time_series_surv"], axis=0).mean(axis=0) # shape: (55,)
            ts_vote_avg = np.concatenate(pooled_results[d]["time_series_vote"], axis=0).mean(axis=0)
            ts_ht_avg   = np.concatenate(pooled_results[d]["time_series_ht"], axis=0).mean(axis=0)
            
            ts_ht_pct_all = np.concatenate(pooled_results[d]["time_series_ht_pct"], axis=0)
            ts_ht_pct_avg = np.nanmean(ts_ht_pct_all, axis=0) # nanmean safely handles the NaN upzone masks
            
            for t in range(55):
                ts_summary.append({
                    "dose": d,
                    "t": t,
                    "surv": ts_surv_avg[t],
                    "vote": ts_vote_avg[t],
                    "ht": ts_ht_avg[t],
                    "ht_pct": ts_ht_pct_avg[t]
                })
        
    sum_df = pd.DataFrame(summary)
    df_results = pd.concat(flattened_data)
    
    df_results.to_csv(CAUSAL_SURFACE_PATH, index=False)
    print(f"\n[√] Final God Table materialized at: {CAUSAL_SURFACE_PATH}", flush=True)
    
    if len(ts_summary) > 0:
        ts_df = pd.DataFrame(ts_summary)
        ts_path = CAUSAL_SURFACE_PATH.replace(".csv", "_time_series.csv")
        ts_df.to_csv(ts_path, index=False)
        print(f"Saved Time-Series Trajectories to {ts_path}", flush=True)
    
    # Save the final trained weights for local plotting
    seq2seq_path = os.environ.get("SEQ2SEQ_CHECKPOINT", os.path.join(OUT_DIR, "causal_seq2seq_weights.pt"))
    torch.save(model.state_dict(), seq2seq_path)
    
    print("\n[SUCCESS] Spatio-Temporal G-Computation Complete.")

if __name__ == "__main__":
    main()
