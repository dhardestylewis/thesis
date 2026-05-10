import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"

print("1. Loading Bi-Weekly Panel...")
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
    # Process & Bureaucracy
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings_lag1",
    "commission_hearings_this_period", "cumulative_commission_hearings_lag1",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
    "Remand_Count",
    
    # Advanced Causal Spatial & Temporal Features
    "cumulative_min_signer_dist", "cumulative_max_signer_dist", "cumulative_median_signer_dist", 
    "cumulative_signers_within_200ft", "cumulative_signers_outside_200ft", 
    "cumulative_unofficial_protest_intensity", 
    "cumulative_protester_embed_dim1", "cumulative_protester_embed_dim2", "cumulative_protester_embed_dim3", "cumulative_protester_embed_dim4",
    "cumulative_temporal_protesting_pct_sf", "cumulative_temporal_silent_pct_sf",
    "cumulative_temporal_protesting_pct_com", "cumulative_temporal_silent_pct_com",
    "cumulative_temporal_protesting_pct_mf", "cumulative_temporal_silent_pct_mf",
    "cumulative_delta_protesting_friction", "cumulative_delta_silent_friction",
    
    # Economics & Demographics
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    
    # Macro Shocks
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
    "treasury_10yr_yield", "treasury_10yr_yield_filing_delta", 
    "fed_funds_rate", "fed_funds_rate_filing_delta", 
    "local_unemployment_rate", "local_unemployment_rate_filing_delta",
    
    # Spatial Gravity
    "knn_petition_rate_1km", "dist_petition_rate_lag1"
]

results = []
test_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

for year_cutoff in test_years:
    print(f"\n======================================")
    print(f"WALK-FORWARD CUTOFF: Testing Year {year_cutoff}")
    print(f"======================================")
    
    for h_name, window in horizons.items():
        df = df_raw.copy()
        
        if window == 1:
            df["target"] = df["petition_event"]
        else:
            df["target"] = df.groupby("case_number")["petition_event"].transform(
                lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1].shift(-1)
            )
            df["target"] = df["target"].fillna(0)
        
        df["target"] = df["target"].astype(int)
        
        # Temporal Split based on the exact year of the biweekly period
        train = df[df["year"] < year_cutoff].copy()
        test = df[df["year"] == year_cutoff].copy()
        
        if len(test) == 0 or train["target"].sum() == 0 or test["target"].sum() == 0:
            print(f"  [{h_name}] Skipped: No positive targets in train or test for {year_cutoff}.")
            continue
            
        model_train = train[["case_number", "target"] + FEATS].copy()
        model_test = test[["case_number", "target"] + FEATS].copy()
        
        model_train[FEATS] = model_train[FEATS].fillna(0)
        model_test[FEATS] = model_test[FEATS].fillna(0)
        
        X_train, y_train = model_train[FEATS], model_train["target"]
        X_test, y_test   = model_test[FEATS], model_test["target"]
        
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        clf = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='AUC',
            random_seed=42,
            verbose=False,
            task_type="GPU"
        )
        
        clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30, verbose=False)
        
        y_pred = clf.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_pred)
        pr = average_precision_score(y_test, y_pred)
        
        print(f"  [{h_name:<10}] Test PR AUC: {pr:.4f} | Test ROC AUC: {roc:.4f}")
        
        results.append({
            "Test_Year": year_cutoff,
            "Horizon": h_name,
            "ROC_AUC": roc,
            "PR_AUC": pr,
            "Train_Samples": len(X_train),
            "Test_Samples": len(X_test)
        })

res_df = pd.DataFrame(results)
res_df.to_csv(r"C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts\multihorizon_multicutoff.csv", index=False)
print("\nComplete. Results saved to multihorizon_multicutoff.csv")
