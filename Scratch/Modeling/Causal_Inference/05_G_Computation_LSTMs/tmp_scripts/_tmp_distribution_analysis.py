import sqlite3
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

import causal_cfm_cvae
causal_cfm_cvae.PANEL_PATH = "biweekly_panel.csv"
from causal_cfm_cvae import load_data

def main():
    print("Loading Ground Truth Data via load_data()...")
    X_all, y_true_all, _, features, _, norm_dict, _, _, _, _ = load_data()
    
    # y_true_all shape is (N, T, Y) which is (1462, 55, 5)
    # Target order:
    # 0: label_real_days_in_pipeline (actually surv probability)
    # 1: net_height_change
    # 2: cumulative_council_nlp_lag1
    # 3: cumulative_commission_hearings_lag1
    # 4: cumulative_council_hearings_lag1
    
    # Flatten the true targets
    y_true = y_true_all.reshape(-1, 5).cpu().numpy()
    
    # Remove rows where all values are exactly 0 (padding) or NaN
    # Since we pad sequences, we should only look at valid timesteps.
    # A simple proxy is keeping rows where surv probability is not 0 (or at least where height isn't exactly mean normalized 0 if we care, but let's just use all non-zero rows)
    # Actually, let's just un-normalize them directly.
    
    mean_coun, std_coun = norm_dict["cumulative_council_hearings_lag1"]
    mean_comm, std_comm = norm_dict["cumulative_commission_hearings_lag1"]
    mean_tok,  std_tok  = norm_dict["cumulative_council_nlp_lag1"]
    mean_ht,   std_ht   = norm_dict["net_height_change"]
    
    surv_true = y_true[:, 0]
    ht_true   = (y_true[:, 1] * std_ht) + mean_ht
    tok_true  = (y_true[:, 2] * std_tok) + mean_tok
    comm_true = (y_true[:, 3] * std_comm) + mean_comm
    coun_true = (y_true[:, 4] * std_coun) + mean_coun
    
    df_true = pd.DataFrame({
        'surv': surv_true,
        'ht_delta': ht_true,
        'cum_tok': tok_true,
        'cum_comm': comm_true,
        'cum_coun': coun_true
    })
    
    print("\nLoading Neural ODE Outputs from SQLite...")
    try:
        con = sqlite3.connect('dashboard_cache/surfaces.db')
        df_pred = pd.read_sql('SELECT surv, ht_delta, cum_tok, cum_comm, cum_coun FROM case_preds', con)
        con.close()
    except Exception as e:
        print(f"Error loading DB: {e}")
        return
        
    print("\n--- GROUND TRUTH (Historical Data) ---")
    print(df_true.describe(percentiles=[.10, .50, .90, .99]))
    print("\n--- ODE COUNTERFACTUALS (Generated Data) ---")
    print(df_pred.describe(percentiles=[.10, .50, .90, .99]))
    
if __name__ == "__main__":
    main()
