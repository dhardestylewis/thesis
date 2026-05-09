"""
08e_run_multihorizon_oot.py

Walk-Forward Multi-Horizon Out-Of-Time evaluation across the full
pre-registered benchmark roster (matching the thesis architectural families):
  - Tree Ensembles  : CatBoost, Random Forest
  - Regularized Linear: Logistic L2, Logistic L1 (ElasticNet)
  - Deep (proxy)    : MLP (sklearn)
  - Distributionally Robust: Logistic with sample-reweighted V-REx proxy

Biweekly panel. Horizons: 14-day, 3-month, 6-month, 1-year, 2-year.
Walk-forward cutoffs: 2018-2024.
Outputs: artifacts/multihorizon_multicutoff_all_models.csv
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime
import shutil
import gc

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

ROOT       = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/biweekly_panel.csv"
OUT_CSV    = ROOT / "artifacts/multihorizon_multicutoff_all_models.csv"

FEATS = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings_lag1",
    "commission_hearings_this_period", "cumulative_commission_hearings_lag1",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct",
    "Remand_Count",
    "cumulative_min_signer_dist", "cumulative_max_signer_dist", "cumulative_median_signer_dist",
    "cumulative_signers_within_200ft", "cumulative_signers_outside_200ft",
    "cumulative_unofficial_protest_intensity",
    "cumulative_protester_embed_dim1", "cumulative_protester_embed_dim2",
    "cumulative_protester_embed_dim3", "cumulative_protester_embed_dim4",
    "cumulative_temporal_protesting_pct_sf", "cumulative_temporal_silent_pct_sf",
    "cumulative_temporal_protesting_pct_com", "cumulative_temporal_silent_pct_com",
    "cumulative_temporal_protesting_pct_mf", "cumulative_temporal_silent_pct_mf",
    "cumulative_delta_protesting_friction", "cumulative_delta_silent_friction",
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income",
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
    "treasury_10yr_yield", "treasury_10yr_yield_filing_delta",
    "fed_funds_rate", "fed_funds_rate_filing_delta",
    "local_unemployment_rate", "local_unemployment_rate_filing_delta",
    "knn_petition_rate_1km", "dist_petition_rate_lag1",
]

HORIZONS = {
    "14_Days":  1,
    "3_Months": 6,
    "6_Months": 13,
    "1_Year":   26,
    "2_Years":  52,
}

TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def get_models(scale_pos_weight: float):
    """Return the pre-registered benchmark roster."""
    return {
        # Tree Ensembles
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="AUC", random_seed=42, verbose=False, task_type="GPU"
        ),
        "RandomForest": "XGB_RF_Placeholder",
        # Regularized Linear
        "LogisticL2": "PyTorch_LogisticL2_Placeholder",
        "LogisticL1": "PyTorch_LogisticL1_Placeholder",
        "Linear": "PyTorch_Linear_Placeholder",
        # Deep (MLP proxy)
        "MLP": "PyTorch_MLP_Placeholder"
    }
    
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        models["TabNet"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", TabNetClassifier(n_d=8, n_a=8, n_steps=3, gamma=1.3, seed=42, verbose=0))
        ])
    except ImportError:
        print("WARNING: pytorch-tabnet not installed. Skipping TabNet.")
        
    return models

def build_and_train_pytorch_mlp(X_tr, y_tr, X_te, hidden_dims=[], epochs=20, lr=0.01, l1_reg=0.0, l2_reg=0.0, batch_size=16):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    layers = []
    in_dim = X_tr.shape[1]
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    
    model = nn.Sequential(*layers).to(device)
    
    pos_weight = max(1.0, (len(y_tr) - sum(y_tr)) / max(1, sum(y_tr)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    
    X_t = torch.tensor(X_tr_sc, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    
    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            if l1_reg > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_reg * l1_norm
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
        logits = model(X_te_t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
    del X_t, y_t, dl, model, X_te_t, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
        
    return probs


def build_and_train_lstm(df_tr, df_te, y_tr, y_te, feats, seq_len=6, epochs=10, lr=0.01):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler

    # Scale features
    scaler = StandardScaler()
    X_tr_flat = scaler.fit_transform(df_tr[feats].values)
    X_te_flat = scaler.transform(df_te[feats].values)

    def make_3d(X_vals, cases, y, seq_len):
        X_3d = np.zeros((len(X_vals), seq_len, len(feats)), dtype=np.float32)
        for i in range(len(X_vals)):
            start_idx = max(0, i - seq_len + 1)
            while start_idx < i and cases[start_idx] != cases[i]:
                start_idx += 1
            valid_len = i - start_idx + 1
            X_3d[i, -valid_len:, :] = X_vals[start_idx:i+1, :]
        return torch.tensor(X_3d), torch.tensor(y, dtype=np.float32).unsqueeze(1)

    X_tr_3d, y_tr_t = make_3d(X_tr_flat, df_tr['case_number'].values, y_tr, seq_len)
    X_te_3d, _ = make_3d(X_te_flat, df_te['case_number'].values, y_te, seq_len)

    class SimpleLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(len(feats), 64).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, (len(y_tr)-sum(y_tr))/max(1, sum(y_tr)))]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dl = DataLoader(TensorDataset(X_tr_3d, y_tr_t), batch_size=16, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_te_3d.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        
    del X_tr_3d, y_tr_t, X_te_3d, dl, model, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
        
    return probs.flatten()


def build_target(df: pd.DataFrame, window: int) -> pd.Series:
    if window == 1:
        return df["petition_event"].astype(int)
    target = df.groupby("case_number")["petition_event"].transform(
        lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1].shift(-1)
    )
    return target.fillna(0).astype(int)


def run():
    print("1. Loading biweekly panel...")
    df_raw = pd.read_csv(PANEL_PATH, low_memory=False)
    df_raw = df_raw.sort_values(["case_number", "period_seq"]).reset_index(drop=True)
    
    print("2. Truncating post-petition rows (removing Target Leakage)...")
    # Identify the first petition event for each case
    first_petition = df_raw[df_raw['petition_event'] == 1].groupby('case_number')['period_seq'].min()
    df_raw['first_petition_seq'] = df_raw['case_number'].map(first_petition)
    # Keep only rows where period_seq <= first_petition_seq (or where petition never happens)
    df_raw = df_raw[(df_raw['first_petition_seq'].isna()) | (df_raw['period_seq'] <= df_raw['first_petition_seq'])]
    df_raw = df_raw.drop(columns=['first_petition_seq']).reset_index(drop=True)

    feats  = [f for f in FEATS if f in df_raw.columns]
    print(f"   {len(df_raw):,} rows | {df_raw['case_number'].nunique():,} cases | {len(feats)} features")

    # ── Precompute all horizon targets once ──────────────────────────────────
    print("3. Precomputing horizon targets (once)...")
    target_cols = {}
    for h_name, window in HORIZONS.items():
        target_cols[h_name] = build_target(df_raw, window).values
        print(f"   [{h_name}] done")

    # Cache feature matrix and year array to avoid repeated extraction
    X_all     = df_raw[feats].fillna(0).values
    year_arr  = df_raw["year"].values

    # Free massive dataframe
    del df_raw
    gc.collect()

    results = []

    for year_cutoff in TEST_YEARS:
        print(f"\n=== Walk-Forward Cutoff: {year_cutoff} ===", flush=True)
        train_mask = year_arr < year_cutoff
        test_mask  = year_arr == year_cutoff

        if test_mask.sum() == 0:
            continue

        X_tr_all = X_all[train_mask]
        X_te      = X_all[test_mask]

        for h_name, window in HORIZONS.items():
            y_all = target_cols[h_name]
            y_tr  = y_all[train_mask]
            y_te  = y_all[test_mask]

            if y_tr.sum() == 0 or y_te.sum() == 0:
                print(f"  [{h_name}] Skipped — no positives", flush=True)
                continue

            spw      = max(1.0, (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
            naive_pr = float(y_tr.mean())
            models   = get_models(spw)
            # Add to sequence if LSTM is supported
            # models["LSTM"] = "PyTorch_LSTM_Placeholder"

            for m_name, clf in models.items():
                try:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    import gc
                    gc.collect()
                    
                    if m_name == "LSTM":
                        try:
                            # Pass full df slices so we can build 3D temporal tensors grouped by case
                            df_tr = df_raw[train_mask]
                            df_te = df_raw[test_mask]
                            y_pred = build_and_train_lstm(df_tr, df_te, y_tr, y_te, feats)
                        except ImportError:
                            print(f"  [{h_name}] LSTM skipped (torch not installed)")
                            continue
                    elif m_name == "CatBoost":
                        clf.fit(X_tr_all, y_tr, verbose=False)
                        y_pred = clf.predict_proba(X_te)[:, 1]
                    elif m_name == "RandomForest":
                        from xgboost import XGBRFClassifier
                        rf = XGBRFClassifier(n_estimators=300, max_depth=8, scale_pos_weight=spw, tree_method='hist', device='cuda', random_state=42)
                        rf.fit(X_tr_all, y_tr)
                        y_pred = rf.predict_proba(X_te)[:, 1]
                    elif m_name == "LogisticL2":
                        y_pred = build_and_train_pytorch_mlp(X_tr_all, y_tr, X_te, hidden_dims=[], epochs=20, lr=0.01, l2_reg=1e-2)
                    elif m_name == "LogisticL1":
                        y_pred = build_and_train_pytorch_mlp(X_tr_all, y_tr, X_te, hidden_dims=[], epochs=20, lr=0.01, l1_reg=1e-3)
                    elif m_name == "Linear":
                        y_pred = build_and_train_pytorch_mlp(X_tr_all, y_tr, X_te, hidden_dims=[], epochs=20, lr=0.01)
                    elif m_name == "MLP":
                        y_pred = build_and_train_pytorch_mlp(X_tr_all, y_tr, X_te, hidden_dims=[128, 64, 32], epochs=30, lr=1e-3)
                    else:
                        clf.fit(X_tr_all, y_tr)
                        y_pred = clf.predict_proba(X_te)[:, 1]
                    roc = roc_auc_score(y_te, y_pred)
                    pr  = average_precision_score(y_te, y_pred)

                    print(f"  [{h_name:<10}] {m_name:<15} ROC: {roc:.4f} | PR: {pr:.4f}", flush=True)

                    results.append({
                        "Test_Year":     year_cutoff,
                        "Horizon":       h_name,
                        "Model":         m_name,
                        "Model_Family":  ("Tree" if m_name in ("CatBoost", "RandomForest")
                                          else "Linear" if "Logistic" in m_name
                                          else "Deep"),
                        "ROC_AUC":       roc,
                        "PR_AUC":        pr,
                        "Naive_PR_AUC":  naive_pr,
                        "Train_Samples": int(train_mask.sum()),
                        "Test_Samples":  int(test_mask.sum()),
                    })
                    
                    # Incremental save to prevent data loss on crash
                    os.makedirs(OUT_CSV.parent, exist_ok=True)
                    pd.DataFrame(results).to_csv(OUT_CSV, index=False)

                except Exception as e:
                    print(f"  [{h_name}] {m_name} FAILED: {e}", flush=True)

    res_df = pd.DataFrame(results)
    
    # MLOps Run Tracking
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "artifacts" / "runs" / run_id
    os.makedirs(run_dir, exist_ok=True)
    
    # Save isolated copy
    run_csv = run_dir / OUT_CSV.name
    res_df.to_csv(run_csv, index=False)
    
    # Save metadata
    meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "script": Path(__file__).name,
        "rows_processed": len(res_df),
        "features": FEATS
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
        
    # Copy back to main artifacts path for latex/downstream scripts
    os.makedirs(OUT_CSV.parent, exist_ok=True)
    shutil.copy2(run_csv, OUT_CSV)
    
    print(f"\n[+] Done. {len(res_df)} rows saved to tracked run directory: {run_dir}")
    print(f"[+] Output synchronized to downstream dependency: {OUT_CSV}")
    print(res_df.groupby(["Model", "Horizon"])[["ROC_AUC", "PR_AUC"]].mean().round(4).to_string())


if __name__ == "__main__":
    run()
