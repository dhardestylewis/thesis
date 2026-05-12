"""
08e_run_annualized_oot.py

Annualizes the biweekly panel (last period per case-year, no leakage) then
runs the same walk-forward multi-horizon evaluation as the biweekly version
across the full pre-registered benchmark roster:
  - Tree: CatBoost, Random Forest
  - Linear: Logistic L2, Logistic L1
  - Deep: MLP

Horizon targets (1, 2, 3 years) are built by looking for petition events
in FUTURE calendar years relative to each (case, year) snapshot.

Outputs: artifacts/annualized_multihorizon_multicutoff_all_models.csv
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime
import shutil

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

ROOT       = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "Data/Panel/biweekly_panel.csv"
OUT_CSV    = ROOT / "artifacts/annualized_multihorizon_multicutoff_all_models.csv"

FEATS = [
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct",
    "cumulative_council_hearings_lag1", "cumulative_commission_hearings_lag1",
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
    
    # Restored Spatial/Temporal/PDF Features
    "active_cases_100m", "active_cases_250m", "active_cases_500m", "active_cases_1km", "active_cases_2km", "active_gravity_index_t",
    "hearing_frequency", "petition_intensity_per_ft",
    "hearing_velocity_3p", "petition_velocity_3p",
    "pdf_requested_height_ft", "pdf_requested_max_far", "pdf_proposed_height_ft",
    "pdf_story_count", "pdf_story_height_ft", "pdf_compatibility_height_ft"
]

HORIZONS  = {"1_Year": 1, "2_Years": 2, "3_Years": 3}
TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]


def annualize(df: pd.DataFrame) -> pd.DataFrame:
    """Last biweekly period per (case, year) — richest in-year state, no leakage."""
    ann = df.sort_values(["case_number", "period_seq"]).groupby(["case_number", "year"]).last().reset_index()
    
    # Recover the true target event which may have happened earlier in the year
    events = df.groupby(["case_number", "year"])["petition_event"].max().reset_index()
    ann = ann.drop(columns=["petition_event"]).merge(events, on=["case_number", "year"], how="left")
    
    return ann


def build_horizon_target(annual: pd.DataFrame, horizon_years: int) -> pd.Series:
    """1 if any petition_event fires in the next horizon_years calendar years."""
    evt = annual[["case_number", "year", "petition_event"]].copy()
    out = []
    for _, row in annual.iterrows():
        fut = evt[
            (evt["case_number"] == row["case_number"]) &
            (evt["year"] > row["year"]) &
            (evt["year"] <= row["year"] + horizon_years)
        ]
        out.append(1 if fut["petition_event"].sum() > 0 else 0)
    return pd.Series(out, index=annual.index)


def get_models(spw: float) -> dict:
    return {
        "CatBoost": CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            scale_pos_weight=spw,
            eval_metric="AUC", random_seed=42, verbose=False, thread_count=-1
        ),
        "RandomForest": "XGB_RF_Placeholder",
        "LogisticL2": "PyTorch_LogisticL2_Placeholder",
        "LogisticL1": "PyTorch_LogisticL1_Placeholder",
        "Linear": "PyTorch_Linear_Placeholder",
        "MLP": "PyTorch_MLP_Placeholder",
    }

def build_and_train_pytorch_mlp(X_tr, y_tr, X_te, hidden_dims=[], epochs=20, lr=0.01, l1_reg=0.0, l2_reg=0.0, batch_size=2048):
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
        
    return probs


def run():
    print("1. Loading biweekly panel...")
    bw = pd.read_csv(PANEL_PATH, low_memory=False)
    bw = bw.sort_values(["case_number", "period_seq"])

    print("2. Annualizing (last period per case-year)...")
    annual = annualize(bw)
    feats  = [f for f in FEATS if f in annual.columns]
    print(f"   {len(annual):,} annual rows | {annual['case_number'].nunique():,} cases | {len(feats)} features")

    # ── Precompute all horizon targets once ──────────────────────────────────
    print("3. Precomputing horizon targets (once)...")
    target_cols = {}
    for h_name, h_years in HORIZONS.items():
        print(f"   [{h_name}] Precomputing...")
        target_cols[h_name] = build_horizon_target(annual, h_years)

    results = []

    for test_year in TEST_YEARS:
        print(f"\n=== Walk-Forward Cutoff: {test_year} ===")

        train_mask = annual["year"] < test_year
        test_mask  = annual["year"] == test_year

        if test_mask.sum() == 0:
            continue

        for h_name, h_years in HORIZONS.items():
            y_all = target_cols[h_name]
            y_tr  = y_all[train_mask].values
            y_te  = y_all[test_mask].values

            if y_tr.sum() == 0 or y_te.sum() == 0:
                print(f"  [{h_name}] Skipped — no positives")
                continue

            X_tr = annual[feats][train_mask].fillna(0).values
            X_te = annual[feats][test_mask].fillna(0).values

            spw      = max(1.0, (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
            naive_pr = float(y_tr.mean())
            models   = get_models(spw)

            for m_name, clf in models.items():
                try:
                    if m_name == "CatBoost":
                        clf.fit(X_tr, y_tr,
                                eval_set=(X_te, y_te),
                                early_stopping_rounds=50,
                                verbose=False)
                        y_pred = clf.predict_proba(X_te)[:, 1]
                    elif m_name == "RandomForest":
                        from xgboost import XGBRFClassifier
                        rf = XGBRFClassifier(n_estimators=300, max_depth=8, scale_pos_weight=spw, tree_method='hist', n_jobs=-1, random_state=42)
                        rf.fit(X_tr, y_tr)
                        y_pred = rf.predict_proba(X_te)[:, 1]
                    elif m_name == "LogisticL2":
                        y_pred = build_and_train_pytorch_mlp(X_tr, y_tr, X_te, hidden_dims=[], epochs=20, lr=0.01, l2_reg=1e-2)
                    elif m_name == "LogisticL1":
                        y_pred = build_and_train_pytorch_mlp(X_tr, y_tr, X_te, hidden_dims=[], epochs=20, lr=0.01, l1_reg=1e-3)
                    elif m_name == "Linear":
                        y_pred = build_and_train_pytorch_mlp(X_tr, y_tr, X_te, hidden_dims=[], epochs=20, lr=0.01)
                    elif m_name == "MLP":
                        y_pred = build_and_train_pytorch_mlp(X_tr, y_tr, X_te, hidden_dims=[128, 64, 32], epochs=30, lr=1e-3)
                    else:
                        clf.fit(X_tr, y_tr)
                        y_pred = clf.predict_proba(X_te)[:, 1]
                        
                    roc = roc_auc_score(y_te, y_pred)
                    pr  = average_precision_score(y_te, y_pred)

                    print(f"  [{test_year}] {m_name:<15} ROC: {roc:.4f} | PR: {pr:.4f}", flush=True)

                    results.append({
                        "Test_Year":     test_year,
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
                    
                    # Incremental save
                    os.makedirs(OUT_CSV.parent, exist_ok=True)
                    pd.DataFrame(results).to_csv(OUT_CSV, index=False)

                except Exception as e:
                    print(f"  [{h_name}] {m_name} FAILED: {e}")

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
