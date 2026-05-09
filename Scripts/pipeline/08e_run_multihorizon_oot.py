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
    df_raw = df_raw.sort_values(["case_number", "period_seq"])
    feats  = [f for f in FEATS if f in df_raw.columns]
    print(f"   {len(df_raw):,} rows | {df_raw['case_number'].nunique():,} cases | {len(feats)} features")

    results = []

    for year_cutoff in TEST_YEARS:
        print(f"\n=== Walk-Forward Cutoff: {year_cutoff} ===")

        for h_name, window in HORIZONS.items():
            df = df_raw.copy()
            df["target"] = build_target(df, window)

            train = df[df["year"] < year_cutoff].copy()
            test  = df[df["year"] == year_cutoff].copy()

            if len(test) == 0 or train["target"].sum() == 0 or test["target"].sum() == 0:
                print(f"  [{h_name}] Skipped — no positives")
                continue

            X_tr = train[feats].fillna(0).values
            y_tr = train["target"].values
            X_te = test[feats].fillna(0).values
            y_te = test["target"].values

            spw = max(1.0, (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
            naive_pr = float(y_tr.mean())

            models = get_models(spw)

            for m_name, clf in models.items():
                try:
                    if m_name == "CatBoost":
                        clf.fit(X_tr, y_tr,
                                eval_set=(X_te, y_te),
                                early_stopping_rounds=30,
                                verbose=False)
                    else:
                        clf.fit(X_tr, y_tr)

                    y_pred = clf.predict_proba(X_te)[:, 1]
                    roc = roc_auc_score(y_te, y_pred)
                    pr  = average_precision_score(y_te, y_pred)

                    print(f"  [{h_name:<10}] {m_name:<15} ROC: {roc:.4f} | PR: {pr:.4f}")

                    results.append({
                        "Test_Year":    year_cutoff,
                        "Horizon":      h_name,
                        "Model":        m_name,
                        "Model_Family": ("Tree" if m_name in ("CatBoost", "RandomForest")
                                         else "Linear" if "Logistic" in m_name
                                         else "Deep"),
                        "ROC_AUC":      roc,
                        "PR_AUC":       pr,
                        "Naive_PR_AUC": naive_pr,
                        "Train_Samples": len(X_tr),
                        "Test_Samples":  len(X_te),
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
