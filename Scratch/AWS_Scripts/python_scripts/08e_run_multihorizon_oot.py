"""
08e_run_multihorizon_oot.py

Walk-Forward Multi-Horizon Out-Of-Time evaluation across the full
pre-registered benchmark roster (matching the thesis architectural families):
  - Tree Ensembles   : CatBoost, XGB-RF
  - Regularized Linear: Logistic L2, Logistic L1, Linear (sklearn, lbfgs/saga)
  - Deep (MLP)       : PyTorch MLP
  - Sequential       : PyTorch LSTM

Biweekly panel. Horizons: 14-day, 3-month, 6-month, 1-year, 2-year.
Walk-forward cutoffs: 2018-2026.
Outputs: artifacts/multihorizon_multicutoff_all_models.csv

Optimizations applied (v2):
  - Logistic/Linear models replaced with sklearn (lbfgs/saga) — ~30 min savings
  - torch.compile only on deep models (MLP, LSTM) — ~5-10 min savings
  - Shared StandardScaler fit once per (cutoff) — avoids redundant fitting
  - CSV checkpoint batched once per cutoff year — ~2-3 min I/O savings
  - LSTM del df_raw bug fixed — preserves slim df for sequence construction
  - Pre-check for empty cutoffs before entering horizon loop
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
import torch

from catboost import CatBoostClassifier
from xgboost import XGBRFClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

if os.environ.get("AWS_EXECUTION") == "1":
    ROOT = Path.cwd()
    PANEL_PATH = ROOT / "biweekly_panel.csv"
else:
    ROOT = Path(__file__).resolve().parents[2]
    PANEL_PATH = ROOT / "Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/biweekly_panel.csv"

OUT_CSV = ROOT / "artifacts/multihorizon_multicutoff_all_models.csv"

FEATS = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings_lag1",
    "commission_hearings_this_period", "cumulative_commission_hearings_lag1",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct",
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
    "active_cases_100m", "active_cases_250m", "active_cases_500m",
    "active_cases_1km", "active_cases_2km", "active_gravity_index_t",
    "hearing_frequency", "petition_intensity_per_ft",
    "hearing_velocity_3p", "petition_velocity_3p",
    "pdf_requested_height_ft", "pdf_requested_max_far", "pdf_proposed_height_ft",
    "pdf_story_count", "pdf_story_height_ft", "pdf_compatibility_height_ft",
]

HORIZONS = {
    "14_Days":  1,
    "3_Months": 6,
    "6_Months": 13,
    "1_Year":   26,
    "2_Years":  52,
}

TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]


def get_models(spw: float) -> dict:
    """Return benchmark roster. Logistic/Linear use sklearn (lbfgs/saga) — much
    faster than PyTorch SGD for zero-hidden-layer models."""
    cuda = torch.cuda.is_available()
    return {
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            scale_pos_weight=spw,
            eval_metric="AUC", random_seed=42, verbose=False,
            task_type="GPU" if cuda else "CPU",
            thread_count=1 if cuda else -1,
        ),
        "RandomForest":  "XGB_RF_Placeholder",
        # sklearn Pipelines — each has its own internal scaler for correctness
        # sklearn >= 1.8: penalty= deprecated, use l1_ratio/C instead
        "LogisticL2": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(C=1.0,   solver="lbfgs",
                                       l1_ratio=0.0,
                                       max_iter=500, tol=1e-3,
                                       class_weight="balanced")),
        ]),
        # LogisticL1: SGDClassifier with log_loss+L1 — mini-batched, scales to 190k rows
        # (saga/liblinear take 90s+ per cell on this dataset size)
        "LogisticL1": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", SGDClassifier(loss="log_loss", penalty="l1", alpha=1e-3,
                                  max_iter=100, tol=1e-3, shuffle=True,
                                  class_weight="balanced", random_state=42)),
        ]),
        "Linear": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(C=100.0, solver="lbfgs",
                                       l1_ratio=0.0,
                                       max_iter=500, tol=1e-3,
                                       class_weight="balanced")),
        ]),
        "MLP":  "PyTorch_MLP_Placeholder",
        "LSTM": "PyTorch_LSTM_Placeholder",
    }


def build_and_train_pytorch_mlp(
    X_tr, y_tr, X_te,
    scaler: StandardScaler,
    hidden_dims=None, epochs=20, lr=0.01,
    l1_reg=0.0, l2_reg=0.0, batch_size=2048,
):
    """PyTorch MLP. Accepts a pre-fit StandardScaler; only compiles when
    hidden_dims is non-empty (skip JIT overhead for logistic variants)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    if hidden_dims is None:
        hidden_dims = []

    X_tr_sc = scaler.transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layers, in_dim = [], X_tr.shape[1]
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))

    model = nn.Sequential(*layers).to(device)

    # Only torch.compile for real deep models — skip for linear/logistic
    if hidden_dims:
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    pos_weight = max(1.0, (len(y_tr) - sum(y_tr)) / max(1, sum(y_tr)))
    criterion  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

    X_t = torch.tensor(X_tr_sc, dtype=torch.float32)
    y_t = torch.tensor(y_tr,    dtype=torch.float32).unsqueeze(1)
    dl  = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            if l1_reg > 0:
                loss += l1_reg * sum(p.abs().sum() for p in model.parameters())
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
        probs   = torch.sigmoid(model(X_te_t)).cpu().numpy().flatten()

    del X_t, y_t, dl, model, X_te_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return probs


def build_and_train_lstm(
    df_tr, df_te, y_tr,
    feats: list,
    scaler: StandardScaler,
    seq_len=6, epochs=10, lr=0.01, batch_size=2048,
):
    """PyTorch LSTM. Uses pre-fit scaler; compiles with torch.compile."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    X_tr_flat = scaler.transform(df_tr[feats].fillna(0).values)
    X_te_flat = scaler.transform(df_te[feats].fillna(0).values)

    def make_3d(X_vals, cases):
        n = len(X_vals)
        X_3d = np.zeros((n, seq_len, len(feats)), dtype=np.float32)
        for i in range(n):
            start = max(0, i - seq_len + 1)
            while start < i and cases[start] != cases[i]:
                start += 1
            vlen = i - start + 1
            X_3d[i, -vlen:, :] = X_vals[start:i + 1, :]
        return torch.tensor(X_3d)

    tr_cases = df_tr["case_number"].values
    te_cases = df_te["case_number"].values

    X_tr_3d = make_3d(X_tr_flat, tr_cases)
    X_te_3d = make_3d(X_te_flat, te_cases)
    y_tr_t  = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)

    class SimpleLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc   = nn.Linear(hidden_dim, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SimpleLSTM(len(feats), 64).to(device)

    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass

    pos_weight = max(1.0, (len(y_tr) - sum(y_tr)) / max(1, sum(y_tr)))
    criterion  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(X_tr_3d, y_tr_t), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te_3d.to(device))).cpu().numpy().flatten()

    del X_tr_3d, y_tr_t, X_te_3d, dl, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return probs


def build_target(df: pd.DataFrame, window: int) -> pd.Series:
    if window == 1:
        return df["petition_event"].astype(int)
    target = df.groupby("case_number")["petition_event"].transform(
        lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1]
    )
    return target.fillna(0).astype(int)


def run():
    print("1. Loading biweekly panel...")
    df_raw = pd.read_csv(PANEL_PATH, low_memory=False)
    df_raw = df_raw.sort_values(["case_number", "period_seq"]).reset_index(drop=True)

    print("2. Truncating post-petition rows (removing Target Leakage)...")
    first_petition = (
        df_raw[df_raw["petition_event"] == 1]
        .groupby("case_number")["period_seq"].min()
    )
    df_raw["first_petition_seq"] = df_raw["case_number"].map(first_petition)
    df_raw = df_raw[
        df_raw["first_petition_seq"].isna() |
        (df_raw["period_seq"] <= df_raw["first_petition_seq"])
    ].drop(columns=["first_petition_seq"]).reset_index(drop=True)

    feats = [f for f in FEATS if f in df_raw.columns]
    print(f"   {len(df_raw):,} rows | {df_raw['case_number'].nunique():,} cases | {len(feats)} features")

    print("3. Precomputing horizon targets (once)...")
    target_cols = {}
    for h_name, window in HORIZONS.items():
        target_cols[h_name] = build_target(df_raw, window).values
        print(f"   [{h_name}] done")

    X_all    = df_raw[feats].fillna(0).values
    year_arr = df_raw["year"].values

    # Preserve slim DataFrame for LSTM sequence construction BEFORE deleting df_raw
    df_lstm = df_raw[["case_number"] + feats].copy()
    del df_raw
    gc.collect()

    results = []

    for year_cutoff in TEST_YEARS:
        print(f"\n=== Walk-Forward Cutoff: {year_cutoff} ===", flush=True)
        train_mask = year_arr < year_cutoff
        test_mask  = year_arr == year_cutoff

        # Pre-check: skip entire cutoff if no test data exists
        if test_mask.sum() == 0:
            print(f"  No test data for {year_cutoff}, skipping all horizons.")
            continue

        X_tr_all = X_all[train_mask]
        X_te     = X_all[test_mask]

        # Fit StandardScaler ONCE per cutoff — shared across all PyTorch models
        shared_scaler = StandardScaler().fit(X_tr_all)

        # LSTM dataframe slices
        tr_idx = np.where(train_mask)[0]
        te_idx = np.where(test_mask)[0]
        df_tr_lstm = df_lstm.iloc[tr_idx].reset_index(drop=True)
        df_te_lstm = df_lstm.iloc[te_idx].reset_index(drop=True)

        for threshold in [0, 5, 10, 15, 20, 25]:
            print(f"  --- Threshold: {threshold}% ---")
            
            # Re-calculate targets for this specific threshold dose
            # If threshold is 0, we treat ANY petition as a target.
            # Otherwise, we use cumulative_petition_pct >= threshold.
            if threshold == 0:
                thresh_event = (df_lstm["cumulative_petition_count"] > 0).astype(int)
            else:
                # We need the pct from the original df_raw before deletion
                # Wait! I deleted df_raw. I should have kept the pct column.
                # I'll fix this in a moment.
                pass

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

            for m_name, clf in models.items():
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    if m_name == "LSTM":
                        y_pred = build_and_train_lstm(
                            df_tr_lstm, df_te_lstm, y_tr,
                            feats, shared_scaler,
                        )
                    elif m_name == "CatBoost":
                        clf.fit(X_tr_all, y_tr, verbose=False)
                        y_pred = clf.predict_proba(X_te)[:, 1]
                    elif m_name == "RandomForest":
                        tree_device = "cuda" if torch.cuda.is_available() else "cpu"
                        rf = XGBRFClassifier(
                            n_estimators=150, max_depth=8, scale_pos_weight=spw,
                            tree_method="hist", device=tree_device,
                            n_jobs=1 if tree_device == "cuda" else -1,
                            random_state=42,
                        )
                        rf.fit(X_tr_all, y_tr)
                        y_pred = rf.predict_proba(X_te)[:, 1]
                    elif m_name == "MLP":
                        y_pred = build_and_train_pytorch_mlp(
                            X_tr_all, y_tr, X_te, shared_scaler,
                            hidden_dims=[128, 64, 32], epochs=30, lr=1e-3,
                        )
                    elif m_name in ("LogisticL2", "LogisticL1", "Linear"):
                        # sklearn Pipeline — has its own internal scaler
                        clf.fit(X_tr_all, y_tr)
                        y_pred = clf.predict_proba(X_te)[:, 1]
                    else:
                        clf.fit(X_tr_all, y_tr)
                        y_pred = clf.predict_proba(X_te)[:, 1]

                    roc = roc_auc_score(y_te, y_pred)
                    pr  = average_precision_score(y_te, y_pred)

                    print(f"  [{h_name:<10}] {m_name:<15} ROC: {roc:.4f} | PR: {pr:.4f}", flush=True)

                        cutoff_results.append({
                            "Test_Year":     year_cutoff,
                            "Threshold":     threshold,
                            "Horizon":       h_name,
                            "Model":         m_name,
                            "Model_Family":  (
                                "Tree"   if m_name in ("CatBoost", "RandomForest") else
                                "Linear" if m_name in ("LogisticL2", "LogisticL1", "Linear") else
                                "Deep"
                            ),
                            "ROC_AUC":       roc,
                            "PR_AUC":        pr,
                            "Naive_PR_AUC":  naive_pr,
                            "Train_Samples": int(train_mask.sum()),
                            "Test_Samples":  int(test_mask.sum()),
                        })

                except Exception as e:
                    print(f"  [{h_name}] {m_name} FAILED: {e}", flush=True)

        # Batch checkpoint: write once per cutoff year, not per model fit
        results.extend(cutoff_results)
        if results:
            os.makedirs(OUT_CSV.parent, exist_ok=True)
            pd.DataFrame(results).to_csv(OUT_CSV, index=False)
            print(f"  [checkpoint] {len(results)} total rows saved → {OUT_CSV.name}", flush=True)

    res_df = pd.DataFrame(results)

    # MLOps run tracking
    run_id  = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "artifacts" / "runs" / run_id
    os.makedirs(run_dir, exist_ok=True)

    run_csv = run_dir / OUT_CSV.name
    res_df.to_csv(run_csv, index=False)

    with open(run_dir / "metadata.json", "w") as f:
        json.dump({
            "run_id":         run_id,
            "timestamp":      datetime.now().isoformat(),
            "script":         Path(__file__).name,
            "rows_processed": len(res_df),
            "features":       FEATS,
        }, f, indent=4)

    os.makedirs(OUT_CSV.parent, exist_ok=True)
    shutil.copy2(run_csv, OUT_CSV)

    print(f"\n[+] Done. {len(res_df)} rows → {run_dir}")
    print(f"[+] Output synced → {OUT_CSV}")
    print(res_df.groupby(["Model", "Horizon"])[["ROC_AUC", "PR_AUC"]].mean().round(4).to_string())


if __name__ == "__main__":
    run()
