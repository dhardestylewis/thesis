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
OUT_CSV    = ROOT / "artifacts/annualized_multihorizon_multicutoff_all_models.csv"

FEATS = [
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct",
    "cumulative_council_hearings_lag1", "cumulative_commission_hearings_lag1",
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
    "cumulative_protesting_pct_single_family", "cumulative_silent_pct_single_family",
    "cumulative_protesting_pct_commercial", "cumulative_silent_pct_commercial",
    "cumulative_protesting_pct_multifamily", "cumulative_silent_pct_multifamily",
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income",
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "race_black", "race_hispanic", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
    "treasury_10yr_yield", "treasury_10yr_yield_filing_delta",
    "fed_funds_rate", "fed_funds_rate_filing_delta",
    "local_unemployment_rate", "local_unemployment_rate_filing_delta",
    "knn_petition_rate_1km", "dist_petition_rate_lag1",
    "label_real_days_in_pipeline", "Aggregate_Sentiment", "net_height_change",
]

HORIZONS  = {"1_Year": 1, "2_Years": 2, "3_Years": 3}
TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def annualize(df: pd.DataFrame) -> pd.DataFrame:
    """Last biweekly period per (case, year) — richest in-year state, no leakage."""
    return (df.sort_values(["case_number", "period_seq"])
              .groupby(["case_number", "year"])
              .last()
              .reset_index())


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
            eval_metric="AUC", random_seed=42, verbose=False, task_type="GPU"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            n_jobs=-1, random_state=42
        ),
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
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), alpha=1e-3,
                max_iter=200, random_state=42, early_stopping=True
            ))
        ]),
    }


def run():
    print("1. Loading biweekly panel...")
    bw = pd.read_csv(PANEL_PATH, low_memory=False)
    bw = bw.sort_values(["case_number", "period_seq"])

    print("2. Annualizing (last period per case-year)...")
    annual = annualize(bw)
    feats  = [f for f in FEATS if f in annual.columns]
    print(f"   {len(annual):,} annual rows | {annual['case_number'].nunique():,} cases | {len(feats)} features")

    results = []

    for test_year in TEST_YEARS:
        print(f"\n=== Walk-Forward Cutoff: {test_year} ===")

        train_df = annual[annual["year"] < test_year].copy()
        test_df  = annual[annual["year"] == test_year].copy()

        if len(test_df) == 0:
            continue

        for h_name, h_years in HORIZONS.items():
            print(f"  [{h_name}] Building targets...")
            train_df["target"] = build_horizon_target(train_df, h_years)
            test_df["target"]  = build_horizon_target(test_df,  h_years)

            if train_df["target"].sum() == 0 or test_df["target"].sum() == 0:
                print(f"  [{h_name}] Skipped — no positives")
                continue

            X_tr = train_df[feats].fillna(0).values
            y_tr = train_df["target"].values
            X_te = test_df[feats].fillna(0).values
            y_te = test_df["target"].values

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
                    else:
                        clf.fit(X_tr, y_tr)

                    y_pred = clf.predict_proba(X_te)[:, 1]
                    roc = roc_auc_score(y_te, y_pred)
                    pr  = average_precision_score(y_te, y_pred)

                    print(f"  [{h_name:<10}] {m_name:<15} ROC: {roc:.4f} | PR: {pr:.4f}")

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
                        "Train_Cases":   len(X_tr),
                        "Test_Cases":    len(X_te),
                    })

                except Exception as e:
                    print(f"  [{h_name}] {m_name} FAILED: {e}")

    res_df = pd.DataFrame(results)
    os.makedirs(OUT_CSV.parent, exist_ok=True)
    res_df.to_csv(OUT_CSV, index=False)
    print(f"\n[+] Done. {len(res_df)} rows saved to {OUT_CSV}")
    print(res_df.groupby(["Model", "Horizon"])[["ROC_AUC", "PR_AUC"]].mean().round(4).to_string())


if __name__ == "__main__":
    run()
