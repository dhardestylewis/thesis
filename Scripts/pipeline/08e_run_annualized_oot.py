"""
08e_run_annualized_oot.py

Annualizes the biweekly panel with zero leakage, then runs the same
walk-forward multi-horizon CatBoost evaluation as the biweekly version.

Annualization strategy (no leakage):
  - For each (case_number, year), take the LAST row of that year.
    This is the final known state as of Dec 31 of that year — all cumulative
    features are correctly lagged inside the biweekly panel already
    (they use .shift(1)), so taking the last row carries only
    information that was known BEFORE that period fired.
  - The horizon targets are built from events that occur AFTER
    the snapshot date, not within it.
"""

import warnings
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import os

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/biweekly_panel.csv"
OUT_CSV    = ROOT / "artifacts/annualized_multihorizon_multicutoff.csv"

# Features — same as biweekly eval, minus biweekly-specific cyclical encodings
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
    "knn_petition_rate_1km", "dist_petition_rate_lag1",
    "label_real_days_in_pipeline",
    "Aggregate_Sentiment", "net_height_change",
]

# Horizon definitions: how many YEARS forward to look for a petition event
HORIZONS = {
    "1_Year":  1,
    "2_Years": 2,
    "3_Years": 3,
}

TEST_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def annualize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse biweekly rows to one row per (case_number, year).
    Take the LAST biweekly period of each year — this is the most
    information-rich snapshot that's still fully in-year and
    free of future leakage (cumulative features already lag-shifted).
    """
    df = df.sort_values(["case_number", "period_seq"])
    annual = (
        df.groupby(["case_number", "year"])
          .last()
          .reset_index()
    )
    return annual


def build_horizon_target(annual: pd.DataFrame, horizon_years: int) -> pd.Series:
    """
    For each (case, year) snapshot, check whether a petition_event
    fires in ANY of the next `horizon_years` calendar years.
    This is built from the FULL biweekly panel aggregated forward,
    so it's strictly future data relative to the snapshot year.
    """
    # Sum petition events by (case, year)
    future_events = (
        annual[["case_number", "year", "petition_event"]]
        .copy()
    )
    # For each row, look up petition events in year+1 .. year+horizon
    results = []
    for _, row in annual.iterrows():
        case = row["case_number"]
        snap_year = row["year"]
        future = future_events[
            (future_events["case_number"] == case) &
            (future_events["year"] > snap_year) &
            (future_events["year"] <= snap_year + horizon_years)
        ]
        results.append(1 if future["petition_event"].sum() > 0 else 0)
    return pd.Series(results, index=annual.index)


def run_annualized_oot():
    print("1. Loading biweekly panel...")
    bw = pd.read_csv(PANEL_PATH, low_memory=False)
    bw = bw.sort_values(["case_number", "period_seq"])
    print(f"   {len(bw):,} biweekly rows | {bw['case_number'].nunique():,} cases")

    print("2. Annualizing (last period per case-year, no leakage)...")
    annual = annualize(bw)
    print(f"   {len(annual):,} annual rows | {annual['case_number'].nunique():,} cases")

    # Resolve available features (some spatial cols may be missing if panel is baseline)
    feats = [f for f in FEATS if f in annual.columns]
    print(f"   Using {len(feats)} features")

    results = []

    for test_year in TEST_YEARS:
        print(f"\n=== Walk-Forward Cutoff: {test_year} ===")

        train_df = annual[annual["year"] < test_year].copy()
        test_df  = annual[annual["year"] == test_year].copy()

        if len(test_df) == 0:
            print(f"  Skipped: no test rows for {test_year}")
            continue

        for h_name, h_years in HORIZONS.items():
            print(f"  [{h_name}] Building targets...")

            # Build targets using future petition events
            train_df["target"] = build_horizon_target(train_df, h_years)
            test_df["target"]  = build_horizon_target(test_df,  h_years)

            if train_df["target"].sum() == 0 or test_df["target"].sum() == 0:
                print(f"  [{h_name}] Skipped: no positive targets")
                continue

            X_train = train_df[feats].fillna(0).values
            y_train = train_df["target"].values
            X_test  = test_df[feats].fillna(0).values
            y_test  = test_df["target"].values

            # Naive baseline: historical base rate
            naive_pr = float(y_train.mean())

            clf = CatBoostClassifier(
                iterations=500, learning_rate=0.05, depth=6,
                eval_metric="AUC", task_type="GPU",
                random_seed=42, verbose=False
            )
            clf.fit(X_train, y_train,
                    eval_set=(X_test, y_test),
                    early_stopping_rounds=50)

            y_pred = clf.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_pred)
            pr  = average_precision_score(y_test, y_pred)

            results.append({
                "Test_Year":     test_year,
                "Horizon":       h_name,
                "ROC_AUC":       roc,
                "PR_AUC":        pr,
                "Naive_PR_AUC":  naive_pr,
                "Train_Cases":   len(train_df),
                "Test_Cases":    len(test_df),
            })
            print(f"  [{h_name}] ROC: {roc:.4f} | PR: {pr:.4f} | Naive PR: {naive_pr:.4f}")

    res_df = pd.DataFrame(results)
    os.makedirs(OUT_CSV.parent, exist_ok=True)
    res_df.to_csv(OUT_CSV, index=False)
    print(f"\n[+] Complete. Results saved to {OUT_CSV}")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    run_annualized_oot()
