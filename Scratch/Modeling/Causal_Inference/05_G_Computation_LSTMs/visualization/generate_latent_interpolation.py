import os
import torch
import numpy as np
import pandas as pd
from causal_seq2seq_cvae import Seq2SeqCVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    PANEL_PATH = r"biweekly_panel.csv"
    if not os.path.exists(PANEL_PATH): PANEL_PATH = r"/data/biweekly_panel.csv"
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    df['vote_friction'] = df['vote_event'] * (1 + df['cumulative_nay_votes'].clip(upper=10))
    df['cumulative_vote_friction'] = df.groupby('case_number')['vote_friction'].cumsum()
    if 'petition_pct_this_period' in df.columns:
        df['cumulative_petition_pct'] = df.groupby('case_number')['petition_pct_this_period'].cumsum()
        
    if "pdf_requested_height_ft" in df.columns:
        initial_req = df.groupby("case_number")["pdf_requested_height_ft"].transform("max")
        current_constraint = df[["pdf_requested_height_ft", "pdf_staff_recommends_ht"]].min(axis=1) if "pdf_staff_recommends_ht" in df.columns else df["pdf_requested_height_ft"]
        current_constraint = current_constraint.fillna(initial_req)
        final_ht = df["pdf_reduced_to_ft"].fillna(current_constraint).fillna(0) if "pdf_reduced_to_ft" in df.columns else current_constraint.fillna(0)
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
        "pdf_council_date", "pdf_council_agenda_url", "pdf_council_transcripts", 
        "pdf_commission_date", "pdf_commission_agenda_url", "pdf_commission_transcripts",
        "pdf_requested_zoning", "Final_Zoning", "Initial_Zoning_Base", "Final_Zoning_Base",
        "status_date", "update_date", "status", "ordinance", "T0", "T_vote", "T_end", 
        "censored", "building_age"
    ])

    features = [c for c in df.columns if c not in exclude]
    for f in features + targets:
        if f not in df.columns: df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    norm_dict = {}
    for f in features + ["net_height_change"]:
        if f in ["pdf_requested_height_ft", "council_nlp_total_tokens", "net_vote_margin", "cumulative_yea_votes", "cumulative_nay_votes", "petition_pct_this_period", "cumulative_petition_pct"]:
            continue
        if f in ["land_acres", "market_value", "appraised_value"]:
            df[f] = np.log1p(df[f].clip(lower=0))
        mean_v, std_v = df[f].mean(), df[f].std()
        df[f] = (df[f] - mean_v) / (std_v + 1e-8)
        norm_dict[f] = (mean_v, std_v)

    for f in ["pdf_requested_height_ft", "council_nlp_total_tokens"]:
        df[f] = np.log1p(df[f].clip(lower=0))
    for f in ["commission_hearings_this_period", "council_hearings_this_period"]:
        df[f] = df[f].clip(lower=0, upper=1)

    return df, features, targets, norm_dict

def main():
    print("Loading data for Latent Interpolation...")
    df, features, targets, norm_dict = load_data()
    
    ckpt_path = "causal_seq2seq_weights.pt"
    if not os.path.exists(ckpt_path): ckpt_path = "/data/output/causal_seq2seq_weights.pt"
    if not os.path.exists(ckpt_path):
        print(f"Waiting for model weights...")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    
    treat_idx = [features.index("petition_pct_this_period"), features.index("cumulative_petition_pct")]
    confounder_idx = []
    if "proposed_max_far" in features: confounder_idx.append(features.index("proposed_max_far"))
    if "pdf_requested_height_ft" in features: confounder_idx.append(features.index("pdf_requested_height_ft"))
    if "land_acres" in features: confounder_idx.append(features.index("land_acres"))

    model = Seq2SeqCVAE(len(features), treat_idx=treat_idx, confounder_idx=confounder_idx).to(device)
    model.load_state_dict(ckpt)
    model.eval()

    # 1. Isolate the "Safe" centroid (Cases that resolved quickly with zero hearings)
    safe_cases = df[(df['period_seq'] <= 5) & (df['resolved'] == 1)]['case_number'].unique()
    # 2. Isolate the "Danger" centroid (Cases that took > 30 periods and had multiple hearings)
    danger_cases = df[(df['period_seq'] > 30) & (df['cumulative_council_hearings_lag1'] > 2)]['case_number'].unique()
    
    print(f"Found {len(safe_cases)} Safe cases and {len(danger_cases)} Danger cases.")
    
    def extract_tensor(case_list, max_cases=100):
        sub = df[df['case_number'].isin(case_list[:max_cases])].copy()
        for f in ["land_acres", "market_value", "appraised_value"]:
            if f in sub.columns: sub[f] = np.log1p(sub[f].clip(lower=0))
        for f in norm_dict:
            if f in sub.columns: sub[f] = (sub[f] - norm_dict[f][0]) / (norm_dict[f][1] + 1e-8)
        if 'pdf_requested_height_ft' in sub.columns:
            sub['pdf_requested_height_ft'] = np.log1p(sub['pdf_requested_height_ft'].clip(lower=0))
        
        MAX_SEQ = 55
        feat_arr = sub[features].values.astype(np.float32)
        case_sizes = sub.groupby('case_number').size()
        
        X = np.zeros((len(case_sizes), MAX_SEQ, len(features)), dtype=np.float32)
        idx = 0
        for k, c in enumerate(case_sizes.index):
            size = case_sizes[c]
            length = min(size, MAX_SEQ)
            X[k, :length, :] = feat_arr[idx:idx+length]
            idx += size
        return torch.tensor(X, dtype=torch.float32).to(device)

    X_safe = extract_tensor(safe_cases)
    X_danger = extract_tensor(danger_cases)
    
    with torch.no_grad():
        # Encode into latent space to get mu
        mu_safe, _ = model.encode(X_safe)
        mu_danger, _ = model.encode(X_danger)
        
        # Calculate centroids (mean across the cohort dimension)
        Z_safe = mu_safe.mean(dim=0, keepdim=True)
        Z_danger = mu_danger.mean(dim=0, keepdim=True)
        
        # Interpolate 10 steps between Safe and Danger
        steps = 10
        alphas = torch.linspace(0, 1, steps).to(device)
        
        Z_interp = []
        for a in alphas:
            Z_interp.append((1 - a) * Z_safe + a * Z_danger)
        Z_interp = torch.cat(Z_interp, dim=0) # Shape: (10, latent_dim)
        
        # We need a dummy structural input to decode. We'll use the Safe centroid's raw features
        X_dummy = X_safe[0:1].expand(steps, -1, -1).clone()
        
        # Decode the interpolated latent representations
        preds, _ = model.decode(X_dummy, Z_interp)
        
        # Extract survival hazard and height concession trajectories
        surv_traj = torch.sigmoid(preds[:, :, 0]).cpu().numpy()
        mean_ht, std_ht = norm_dict["net_height_change"]
        ht_traj = (preds[:, :, 2].cpu().numpy() * (std_ht + 1e-8)) + mean_ht
        
    results = []
    for step in range(steps):
        for t in range(55):
            results.append({
                "interpolation_step": step,
                "danger_weight": float(alphas[step].cpu().numpy()),
                "period_seq": t + 1,
                "hazard_surv": surv_traj[step, t],
                "concession_ht": ht_traj[step, t]
            })
            
    pd.DataFrame(results).to_csv("latent_tipping_point_gradient.csv", index=False)
    print("Saved latent_tipping_point_gradient.csv!")

if __name__ == "__main__":
    main()
