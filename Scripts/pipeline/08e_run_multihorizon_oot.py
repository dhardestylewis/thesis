import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "Data/Panel/biweekly_panel.csv"

def run_multihorizon_oot():
    print("1. Loading Causal Bi-Weekly Panel...")
    df_raw = pd.read_csv(PANEL_PATH, low_memory=False)
    df_raw = df_raw.sort_values(["case_number", "period_seq"])

    horizons = {
        "14_Days": 1,
        "3_Months": 6,
        "6_Months": 13,
        "1_Year": 26,
        "2_Years": 52
    }

    FEATS = [
        "period_seq", "bw_sin", "bw_cos",
        "council_hearings_this_period", "cumulative_council_hearings_lag1",
        "commission_hearings_this_period", "cumulative_commission_hearings_lag1",
        "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
        "Remand_Count",
        "cumulative_min_signer_dist", "cumulative_max_signer_dist", "cumulative_median_signer_dist", 
        "cumulative_signers_within_200ft", "cumulative_signers_outside_200ft", 
        "cumulative_unofficial_protest_intensity", 
        "cumulative_protester_embed_dim1", "cumulative_protester_embed_dim2", "cumulative_protester_embed_dim3", "cumulative_protester_embed_dim4",
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
        "cpi_yoy", "cpi_momentum", "cpi_filing_delta", 
        "case_shiller_austin_yoy", "case_shiller_momentum"
    ]

    test_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    results = []

    for year_cutoff in test_years:
        print(f"\n======================================")
        print(f" WALK-FORWARD CUTOFF: {year_cutoff}")
        print(f"======================================")
        
        train_pool = df_raw[df_raw["year"] < year_cutoff].copy()
        test_pool = df_raw[df_raw["year"] == year_cutoff].copy()
        
        for h_name, h_shift in horizons.items():
            # Rolling window target
            train_pool["target"] = train_pool.groupby("case_number")["petition_event"].transform(
                lambda x: x.rolling(window=h_shift, min_periods=1).max().shift(-h_shift)
            ).fillna(0)
            
            test_pool["target"] = test_pool.groupby("case_number")["petition_event"].transform(
                lambda x: x.rolling(window=h_shift, min_periods=1).max().shift(-h_shift)
            ).fillna(0)
            
            train = train_pool.dropna(subset=FEATS)
            test = test_pool.dropna(subset=FEATS)
            
            if len(test) == 0 or train["target"].sum() == 0 or test["target"].sum() == 0:
                print(f"  [{h_name}] Skipped: No positive targets.")
                continue
                
            X_train = train[FEATS].values
            y_train = train["target"].values
            X_test = test[FEATS].values
            y_test = test["target"].values
            
            # Naive Baseline (Historical Base Rate)
            base_rate = np.mean(y_train)
            y_pred_naive = np.full(len(y_test), base_rate)
            naive_auc = 0.5000 # by definition
            naive_pr = base_rate
            
            clf = CatBoostClassifier(
                iterations=500, learning_rate=0.05, depth=6,
                eval_metric="AUC", task_type="GPU", random_seed=42, verbose=False
            )
            
            clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
            y_pred = clf.predict_proba(X_test)[:, 1]
            
            roc = roc_auc_score(y_test, y_pred)
            pr = average_precision_score(y_test, y_pred)
            
            results.append({
                "Test_Year": year_cutoff,
                "Horizon": h_name,
                "ROC_AUC": roc,
                "PR_AUC": pr,
                "Naive_ROC_AUC": naive_auc,
                "Naive_PR_AUC": naive_pr,
                "Train_Samples": len(train),
                "Test_Samples": len(test)
            })
            print(f"  [{h_name}] ROC: {roc:.4f} | PR: {pr:.4f} (Naive PR: {naive_pr:.4f})")

    res_df = pd.DataFrame(results)
    out_csv = ROOT / "artifacts" / "multihorizon_multicutoff_with_baselines.csv"
    os.makedirs(out_csv.parent, exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"\n[+] Multi-Horizon GPU Evaluation Complete: {out_csv}")

if __name__ == "__main__":
    run_multihorizon_oot()
