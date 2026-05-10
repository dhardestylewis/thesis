import os
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import causal_seq2seq_cvae
causal_seq2seq_cvae.PANEL_PATH = "aws_deploy/biweekly_panel_aws.csv"
from causal_seq2seq_cvae import load_data_and_cells, Seq2SeqCVAE, build_tensors

def generate_surfaces():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading exactly matched AWS dataset...")
    df, features, targets, norm_dict, cell_assignments, unique_cells = load_data_and_cells()
    
    first_periods_dt = df.groupby("case_number")["period_start_dt"].min()
    
    cutoff_date = pd.to_datetime("2021-12-31")
    end_test_date = pd.to_datetime("2024-12-31")
    
    test_cases = first_periods_dt[
        (first_periods_dt > cutoff_date) & 
        (first_periods_dt <= end_test_date)
    ].index.values
    
    initial_req = df.groupby("case_number")["pdf_requested_height_ft"].max()
    high_rise_cases = initial_req[initial_req >= 60].index.values
    test_cases = np.intersect1d(test_cases, high_rise_cases)
    
    print(f"Isolated {len(test_cases)} High-Rise cases in the OOD test set.")
    
    X_test, Y_test, L_test = build_tensors(df, features, targets, test_cases)
    X_test = X_test.to(device)
    
    print(f"Loaded {len(X_test)} OOD test cases.")
    
    # 2. Load Model
    treat_idx = [features.index("petition_pct_this_period"), features.index("cumulative_petition_pct")]
    
    model = Seq2SeqCVAE(
        input_dim=len(features),
        latent_dim=32,
        hidden_dim=128,
        num_layers=2,
        treat_idx=treat_idx,
        confounder_idx=[]
    ).to(device)
    
    weights_path = "aws_deploy/causal_seq2seq_weights.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 3. Autoregressive Inference (Over multiple doses)
    PS_IDX = features.index("period_seq")
    f_coun = features.index("cumulative_council_hearings_lag1")
    f_comm = features.index("cumulative_commission_hearings_lag1")
    f_tok = features.index("cumulative_council_nlp_lag1")
    f_yea = features.index("cumulative_yea_votes")
    f_nay = features.index("cumulative_nay_votes")
    f_margin = features.index("net_vote_margin")
    
    pet_idx = features.index("petition_pct_this_period")
    cum_pet_idx = features.index("cumulative_petition_pct")
    
    mean_coun, std_coun = norm_dict["cumulative_council_hearings_lag1"]
    mean_comm, std_comm = norm_dict["cumulative_commission_hearings_lag1"]
    mean_tok, std_tok = norm_dict["cumulative_council_nlp_lag1"]
    mean_ht, std_ht = norm_dict["net_height_change"]
    
    doses = np.linspace(0.0, 1.0, 11).tolist()
    mc_samples = 25
    N_cases = len(X_test)
    
    # Pre-allocate grids for 3D surfaces: shape (len(doses), 55)
    grid_surv = np.zeros((len(doses), 55))
    grid_ht = np.zeros((len(doses), 55))
    grid_tok = np.zeros((len(doses), 55))
    
    print("Executing 55-step Autoregressive Rollouts for 11 Doses (MC=25)...")
    
    # Sub-sample cases to prevent CPU memory crash with MC=25
    if N_cases > 150:
        # We only need enough cases to get a stable average
        torch.manual_seed(42)
        indices = torch.randperm(N_cases)[:150]
        X_test = X_test[indices]
        N_cases = 150
        print(f"Sub-sampled to {N_cases} cases to fit in CPU memory.")
        
    X_test_mc = X_test.unsqueeze(1).repeat(1, mc_samples, 1, 1).view(N_cases * mc_samples, X_test.size(1), X_test.size(2))
    
    with torch.no_grad():
        for d_idx, d in enumerate(doses):
            print(f"  > Processing Dose {d:.1f}...")
            
            X_t = X_test_mc.clone()
            X_pre = X_t[:, :4, :]
            
            # Inject Intervention at t=4
            X_t[:, 4, pet_idx] = d
            X_t[:, 4:, cum_pet_idx] = d
            
            ts_surv = np.zeros((N_cases * mc_samples, 55))
            ts_ht = np.zeros((N_cases * mc_samples, 55))
            ts_tok = np.zeros((N_cases * mc_samples, 55))
            
            for t in range(4, 55):
                preds, _, _, _ = model(X_pre, X_t)
                preds_t = preds[:, t, :]
                
                # Log metrics
                ts_surv[:, t] = torch.sigmoid(preds_t[:, 0]).cpu().numpy()
                ts_ht[:, t] = (preds_t[:, 2].cpu().numpy() * std_ht) + mean_ht
                ts_tok[:, t] = torch.expm1(preds_t[:, 3]).cpu().numpy()
                
                if t < 54:
                    pred_tok = torch.expm1(preds_t[:, 3])
                    pred_comm = torch.sigmoid(preds_t[:, 4])
                    pred_coun = torch.sigmoid(preds_t[:, 5])
                    pred_yea = torch.relu(preds_t[:, 6])
                    pred_nay = torch.relu(preds_t[:, 7])
                    
                    cur_comm = (X_t[:, t, f_comm] * std_comm) + mean_comm
                    cur_coun = (X_t[:, t, f_coun] * std_coun) + mean_coun
                    cur_tok = (X_t[:, t, f_tok] * std_tok) + mean_tok
                    
                    next_comm = cur_comm + pred_comm
                    next_coun = cur_coun + pred_coun
                    next_tok = cur_tok + pred_tok
                    
                    X_t[:, t+1, f_comm] = (next_comm - mean_comm) / std_comm
                    X_t[:, t+1, f_coun] = (next_coun - mean_coun) / std_coun
                    X_t[:, t+1, f_tok] = (next_tok - mean_tok) / std_tok
                    X_t[:, t+1, f_yea] = X_t[:, t, f_yea] + pred_yea
                    X_t[:, t+1, f_nay] = X_t[:, t, f_nay] + pred_nay
                    X_t[:, t+1, f_margin] = X_t[:, t+1, f_yea] - X_t[:, t+1, f_nay]
                    
            # Average across N_cases * mc_samples to get the mean trajectory
            grid_surv[d_idx, :] = ts_surv.mean(axis=0)
            grid_ht[d_idx, :] = ts_ht.mean(axis=0)
            grid_tok[d_idx, :] = ts_tok.mean(axis=0)

    # 4. Generate Plotly 3D Dashboards
    print("Generating 3D Surfaces...")
    times = np.arange(55)
    
    def make_3d_plot(grid, z_title, plot_title, filename, colorscale='Viridis'):
        fig = go.Figure(data=[go.Surface(z=grid, x=times, y=doses, colorscale=colorscale)])
        fig.update_layout(title=plot_title,
                          scene=dict(xaxis_title='Time (Bi-weekly Periods)',
                                     yaxis_title='Petition Dose (0 to 1)',
                                     zaxis_title=z_title),
                          template="plotly_dark", height=800)
        
        anti_path = os.path.join(r"C:\Users\dhl\.gemini\antigravity\brain\52e35f87-22e1-4135-9cf3-329ccde9b487", filename)
        fig.write_html(anti_path)
        print(f"Saved {filename}")

    make_3d_plot(grid_surv, 'Survival Probability', '3D Causal Trajectory: Survival', 'surface_3d_survival.html', 'RdBu')
    make_3d_plot(grid_ht, 'Height Concession (ft)', '3D Causal Trajectory: Height Concession', 'surface_3d_height.html', 'Magma')
    make_3d_plot(grid_tok, 'NLP Tokens (Outrage)', '3D Causal Trajectory: Public Outrage', 'surface_3d_tokens.html', 'Inferno')
    
    print("All 3D dashboards generated successfully.")

if __name__ == "__main__":
    generate_surfaces()
