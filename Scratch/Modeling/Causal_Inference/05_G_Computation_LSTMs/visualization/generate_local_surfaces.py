import os
import torch
import pandas as pd
import numpy as np
import pickle
from causal_seq2seq_cvae import load_data_and_cells, Seq2SeqCVAE, run_counterfactual_inference, build_tensors

def generate_local_surfaces():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading data...")
    df, features, targets, norm_dict, cell_assignments, unique_cells = load_data_and_cells()
    
    first_periods_dt = df.groupby("case_number")["period_start_dt"].min()
    
    cutoff_date = pd.to_datetime("2021-12-31")
    end_test_date = pd.to_datetime("2024-12-31")
    
    test_cases = first_periods_dt[
        (first_periods_dt > cutoff_date) & 
        (first_periods_dt <= end_test_date)
    ].index.values
    
    X_test, Y_test, L_test = build_tensors(df, features, targets, test_cases)
    X_test = X_test.to(device)
    
    print(f"Extracted {len(X_test)} OOD test cases.")
    
    # 2. Load Weights
    treat_idx = [
        features.index("petition_pct_this_period"), 
        features.index("cumulative_petition_pct")
    ]
    
    model = Seq2SeqCVAE(
        input_dim=len(features),
        latent_dim=16,
        hidden_dim=128,
        num_layers=2,
        treat_idx=treat_idx,
        confounder_idx=[]
    ).to(device)
    
    weights_path = "aws_deploy/causal_seq2seq_weights.pt"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found!")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 3. Run Inference
    print("Running 25x Monte Carlo Inference... (This takes 5 seconds locally)")
    results = run_counterfactual_inference(model, X_test, features, norm_dict, mc_samples=25)
    
    # 4. Extract outputs exactly like AWS did
    doses = np.linspace(0.0, 1.0, 11).tolist()
    summary = []
    ts_summary = []
    
    for d in doses:
        surv_all = results[d]["surv"]
        vote_all = results[d]["vote"]
        ht_all   = results[d]["ht"]
        tok_all  = results[d]["tok"]
        ht_pct_all = results[d]["ht_pct"]
        
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
        
        if len(results[d]["time_series_surv"]) > 0:
            ts_surv_avg = results[d]["time_series_surv"].mean(axis=0)
            ts_vote_avg = results[d]["time_series_vote"].mean(axis=0)
            ts_ht_avg   = results[d]["time_series_ht"].mean(axis=0)
            ts_ht_pct_avg = np.nanmean(results[d]["time_series_ht_pct"], axis=0)
            
            for t in range(55):
                ts_summary.append({
                    "dose": d,
                    "t": t,
                    "surv": ts_surv_avg[t],
                    "vote": ts_vote_avg[t],
                    "ht": ts_ht_avg[t],
                    "ht_pct": ts_ht_pct_avg[t]
                })
                
    # Save the proper CSVs locally
    pd.DataFrame(summary).to_csv("aws_deploy/causal_friction_surface.csv", index=False)
    pd.DataFrame(ts_summary).to_csv("aws_deploy/causal_time_series_surface.csv", index=False)
    print("\n[SUCCESS] Generated causal_friction_surface.csv and causal_time_series_surface.csv locally!")

if __name__ == "__main__":
    generate_local_surfaces()
