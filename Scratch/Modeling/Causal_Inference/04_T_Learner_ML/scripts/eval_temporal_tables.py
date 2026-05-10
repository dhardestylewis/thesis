import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import sys
import os
import warnings

warnings.filterwarnings('ignore')
sys.path.append(r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML")
from run_causal_ml_sweep import load_fully_hydrated_data

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts"

def eval_annualized_by_year():
    print("1. Generating Annualized Year-by-Year Table...")
    df, features, categorical_features = load_fully_hydrated_data()
    df["filing_year"] = pd.to_datetime(df["application_start_date"]).dt.year
    df = df.dropna(subset=["exact_geometric_petition_pct", "filing_year"])
    df["target"] = (df["exact_geometric_petition_pct"] >= 20).astype(int)
    
    model_df = df[["case_number", "target", "filing_year"] + features].copy().dropna()
    years_to_evaluate = sorted([y for y in model_df["filing_year"].unique() if y >= 2018])
    
    out_of_fold_preds = np.full(len(model_df), np.nan)
    
    model = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, verbose=0, random_seed=42, auto_class_weights='Balanced')
    
    results = []
    
    for yr in years_to_evaluate:
        train_mask = model_df["filing_year"] < yr
        test_mask = model_df["filing_year"] == yr
        
        train = model_df[train_mask]
        test = model_df[test_mask]
        
        # We need at least 1 positive sample in train to fit a classifier
        if train["target"].sum() < 2 or test["target"].sum() == 0:
            continue
            
        model.fit(Pool(train[features], train["target"], cat_features=categorical_features))
        preds = model.predict_proba(test[features])[:, 1]
        
        # Store for possible later use
        model_df.loc[test_mask, "oof_pred"] = preds
        
        pos = test["target"].sum()
        total = len(test)
        
        roc = roc_auc_score(test["target"], preds)
        pr = average_precision_score(test["target"], preds)
        brier = brier_score_loss(test["target"], preds)
        lift = pr / (pos / total)
            
        results.append({
            "Test Year": int(yr),
            "Training Cutoff": f"< {int(yr)}",
            "Total Cases": total,
            "Petitions": pos,
            "Petition Rate": f"{(pos/total)*100:.1f}%",
            "OOT ROC AUC": roc,
            "OOT PR AUC": pr,
            "OOT PR Lift": lift,
            "OOT Brier Score": brier
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "table_annual_performance_by_year.csv"), index=False)
    print("Saved table_annual_performance_by_year.csv")


def eval_biweekly_by_horizon():
    print("\n2. Generating Biweekly Horizon Table...")
    PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"
    
    FEATS_DYNAMIC = [
        "period_seq", "bw_sin", "bw_cos",
        "council_hearings_this_period", "cumulative_council_hearings",
        "commission_hearings_this_period", "cumulative_commission_hearings",
        "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
        "Remand_Count", "market_value", "building_age", "land_acres",
        "total_population", "median_household_income", "renter_share", "rent_burden", "affordability_proxy",
        "race_white", "median_age", "mortgage_rate_30yr", "mortgage_rate_30yr_momentum",
        "treasury_10yr_yield", "fed_funds_rate", "local_unemployment_rate",
        "knn_petition_rate_1km", "dist_petition_rate_lag1"
    ]
    
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    df_hazard = df[df["period_seq"] > 0].copy()
    target = "petition_event"
    
    model_df = df_hazard[["case_number", target] + FEATS_DYNAMIC].copy().dropna()
    
    # 5-fold CV to get out-of-fold predictions for ALL periods
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    out_of_fold_preds = np.zeros(len(model_df))
    
    model = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, verbose=0, random_seed=42, auto_class_weights='Balanced')
    
    for train_idx, test_idx in gkf.split(model_df, groups=model_df["case_number"]):
        train, test = model_df.iloc[train_idx], model_df.iloc[test_idx]
        model.fit(Pool(train[FEATS_DYNAMIC], train[target]))
        preds = model.predict_proba(test[FEATS_DYNAMIC])[:, 1]
        out_of_fold_preds[test_idx] = preds
        
    model_df["oof_pred"] = out_of_fold_preds
    
    # Map period_seq to horizon buckets (2 weeks per period)
    def map_horizon(seq):
        if seq <= 2: return "1 Month"
        if seq <= 6: return "3 Months"
        if seq <= 12: return "6 Months"
        if seq <= 24: return "1 Year"
        return "1+ Years"
        
    model_df["horizon"] = model_df["period_seq"].apply(map_horizon)
    horizons = ["1 Month", "3 Months", "6 Months", "1 Year", "1+ Years"]
    
    results = []
    for h in horizons:
        sub = model_df[model_df["horizon"] == h]
        pos = sub[target].sum()
        total = len(sub)
        
        if pos > 0 and (total - pos) > 0:
            roc = roc_auc_score(sub[target], sub["oof_pred"])
            pr = average_precision_score(sub[target], sub["oof_pred"])
            brier = brier_score_loss(sub[target], sub["oof_pred"])
            lift = pr / (pos / total)
        else:
            roc, pr, brier, lift = np.nan, np.nan, np.nan, np.nan
            
        results.append({
            "Forecasting Horizon": h,
            "Total Biweekly Observations": total,
            "Petition Events": pos,
            "Hazard Rate": f"{(pos/total)*100:.2f}%",
            "ROC AUC": roc,
            "PR AUC": pr,
            "PR Lift": lift,
            "Brier Score": brier
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "table_hazard_performance_by_horizon.csv"), index=False)
    print("Saved table_hazard_performance_by_horizon.csv")

if __name__ == "__main__":
    eval_annualized_by_year()
    eval_biweekly_by_horizon()
