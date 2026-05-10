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
    print("Loading data for VSN Attention Map...")
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

    # Get structural feature names (excluding the treatment features which bypass VSN)
    struct_features = [f for i, f in enumerate(features) if i not in treat_idx]
    
    # Grab a random sample of 500 cases to calculate average attention
    sample_cases = df['case_number'].unique()[:500]
    sub = df[df['case_number'].isin(sample_cases)].copy()
    
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
    X = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Pass the input through the Variable Selection Network directly
        x_struct, _ = model._split_features(X)
        _, weights = model.vsn(x_struct) # weights shape: (batch, seq, num_struct_features)
        
        # Average the attention weights across all 500 cases to get the global attention over time
        avg_weights = weights.mean(dim=0).cpu().numpy() # shape: (seq, num_struct_features)
        
    # Convert to a DataFrame
    results = []
    for t in range(MAX_SEQ):
        for f_idx, feat_name in enumerate(struct_features):
            results.append({
                "period_seq": t + 1,
                "feature": feat_name,
                "attention_weight": avg_weights[t, f_idx]
            })
            
    df_att = pd.DataFrame(results)
    df_att.to_csv("vsn_attention_map_over_time.csv", index=False)
    print("Saved vsn_attention_map_over_time.csv!")
    
    # Also save the top 10 most important features globally (averaged over time)
    global_importance = df_att.groupby("feature")["attention_weight"].mean().sort_values(ascending=False)
    global_importance.to_csv("vsn_global_feature_importance.csv")
    print("Saved vsn_global_feature_importance.csv!")

if __name__ == "__main__":
    main()
