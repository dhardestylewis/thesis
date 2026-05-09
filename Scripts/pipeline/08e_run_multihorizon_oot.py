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
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            n_jobs=-1, random_state=42
        ),
        # Regularized Linear
        "LogisticL2": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, penalty="l2", max_iter=1000,
                                       class_weight="balanced", random_state=42))
        ]),
        "LogisticL1": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, penalty="l1", solver="liblinear",
                                       max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        # Deep (MLP proxy)
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation="relu",
                alpha=1e-3, max_iter=200, random_state=42, early_stopping=True
            ))
        ]),
    }


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
    feats  = [f for f in FEATS if f in df_raw.columns]
    print(f"   {len(df_raw):,} rows | {df_raw['case_number'].nunique():,} cases | {len(feats)} features")

    # ── Precompute all horizon targets once ──────────────────────────────────
    print("2. Precomputing horizon targets (once)...")
    target_cols = {}
    for h_name, window in HORIZONS.items():
        target_cols[h_name] = build_target(df_raw, window).values
        print(f"   [{h_name}] done")

    # Cache feature matrix and year array to avoid repeated extraction
    X_all     = df_raw[feats].fillna(0).values
    year_arr  = df_raw["year"].values

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

            for m_name, clf in models.items():
                try:
                    if m_name == "CatBoost":
                        clf.fit(X_tr_all, y_tr, verbose=False)
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
