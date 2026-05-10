import os
import torch
import numpy as np
import pandas as pd
from causal_seq2seq_cvae import Seq2SeqCVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    PANEL_PATH = r"biweekly_panel.csv"
    if not os.path.exists(PANEL_PATH):
        # Fallback for AWS
        PANEL_PATH = r"/data/biweekly_panel.csv"
    
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    # Same feature extraction as causal_seq2seq_cvae.py
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
    print("Loading data...")
    df, features, targets, norm_dict = load_data()
    
    ckpt_path = "causal_seq2seq_weights.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/data/output/causal_seq2seq_weights.pt"
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

    MAX_SEQ = 55
    pet_idx = features.index("petition_pct_this_period")
    cum_idx = features.index("cumulative_petition_pct")
    
    # Isolate a highly vulnerable cohort (e.g. High Height, High Density)
    sub = df[df['pdf_requested_height_ft'] > np.log1p(60)].sort_values(['case_number', 'period_seq'])
    cases_in_sub = sub['case_number'].unique()[:100]
    sub = sub[sub['case_number'].isin(cases_in_sub)]
    
    sub_norm = sub[features].copy()
    for f in ["land_acres", "market_value", "appraised_value"]:
        if f in sub_norm.columns: sub_norm[f] = np.log1p(sub_norm[f].clip(lower=0))
    for f in norm_dict:
        if f in sub_norm.columns: sub_norm[f] = (sub_norm[f] - norm_dict[f][0]) / (norm_dict[f][1] + 1e-8)
    if 'pdf_requested_height_ft' in sub_norm.columns:
        sub_norm['pdf_requested_height_ft'] = np.log1p(sub['pdf_requested_height_ft'].clip(lower=0))

    feat_arr = sub_norm.values.astype(np.float32)
    case_sizes = sub.groupby('case_number').size()

    n_cases = len(cases_in_sub)
    X_cohort = np.zeros((n_cases, MAX_SEQ, len(features)), dtype=np.float32)
    idx = 0
    for k, c in enumerate(cases_in_sub):
        size = case_sizes[c]
        length = min(size, MAX_SEQ)
        X_cohort[k, :length, :] = feat_arr[idx:idx+length]
        idx += size
    X_t = torch.tensor(X_cohort, dtype=torch.float32)

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

    results = []
    
    # We want to see bands for different injection timings and intensities
    timings = [5, 10, 15, 20]
    intensities = [0, 20, 50, 100]

    for timing in timings:
        for intensity in intensities:
            X_cf = X_t.clone().to(device)
            # Inject intervention
            X_cf[:, timing-1, pet_idx] = float(intensity)
            X_cf[:, timing-1:, cum_idx] = float(intensity)

            with torch.no_grad():
                X_pre = X_cf[:, :4, :]
                for t in range(timing-1, 54):
                    preds, _, _, _ = model(X_pre, X_cf)
                    preds_t = preds[:, t, :]
                    
                    pred_tok = torch.expm1(preds_t[:, 3])
                    pred_comm = torch.clamp(preds_t[:, 4], 0, 1)
                    pred_coun = torch.clamp(preds_t[:, 5], 0, 1)
                    pred_yea = torch.relu(preds_t[:, 6])
                    pred_nay = torch.relu(preds_t[:, 7])

                    curr_coun = X_cf[:, t, f_coun] * (std_coun + 1e-8) + mean_coun
                    curr_comm = X_cf[:, t, f_comm] * (std_comm + 1e-8) + mean_comm
                    curr_tok = X_cf[:, t, f_tok] * (std_tok + 1e-8) + mean_tok
                    curr_yea = X_cf[:, t, f_yea]
                    curr_nay = X_cf[:, t, f_nay]

                    next_coun = curr_coun + pred_coun
                    next_comm = curr_comm + pred_comm
                    next_tok = curr_tok + pred_tok
                    next_yea = curr_yea + pred_yea
                    next_nay = curr_nay + pred_nay

                    X_cf[:, t+1, f_coun] = (next_coun - mean_coun) / (std_coun + 1e-8)
                    X_cf[:, t+1, f_comm] = (next_comm - mean_comm) / (std_comm + 1e-8)
                    X_cf[:, t+1, f_tok] = (next_tok - mean_tok) / (std_tok + 1e-8)
                    X_cf[:, t+1, f_yea] = next_yea
                    X_cf[:, t+1, f_nay] = next_nay
                    X_cf[:, t+1, f_margin] = next_yea - next_nay

                # Final inference pass over the rolled-out trajectories
                preds, _, _, _ = model(X_pre, X_cf)
                
                # Extract the step-by-step trajectories
                surv_traj = torch.sigmoid(preds[:, :, 0]).mean(dim=0).cpu().numpy() # Shape (55)
                ht_traj = (preds[:, :, 2].cpu().numpy() * (std_ht + 1e-8)) + mean_ht
                ht_traj = ht_traj.mean(axis=0) # Average across the 100 cases
                
                for t_step in range(MAX_SEQ):
                    results.append({
                        "timing_injected": timing,
                        "intensity_pct": intensity,
                        "period_seq": t_step + 1,
                        "hazard_surv": surv_traj[t_step],
                        "concession_ht": ht_traj[t_step]
                    })
                    
    pd.DataFrame(results).to_csv("counterfactual_trajectory_bands.csv", index=False)
    print("Saved counterfactual_trajectory_bands.csv!")

if __name__ == "__main__":
    main()
