import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, mean_absolute_error, r2_score

print("Loading supplemented zoning panel...")
# We use the supplemented zoning cases (6,700 cases)
df = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML\model_ready_zoning_supplemented.csv", low_memory=False)

# But wait, earlier we found out model_ready_zoning_supplemented.csv had a corrupted petition percentage!
# So we need to merge the true petition percentages!
p = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\petition_signers_backfilled.csv")
valid_petitions = p[p["signed"] == 1].groupby("case_number")["area_pct"].sum().reset_index()

df = df.merge(valid_petitions, on="case_number", how="left")
df["Valid_Petition_Pct"] = df["area_pct"].fillna(0.0)

# Create Hurdle Targets
# Hurdle 1: Will a protest occur at all? (Petition > 0)
df["is_protested"] = (df["Valid_Petition_Pct"] > 0).astype(int)

# Hurdle 2: Severity (given protest > 0)
# We will use Valid_Petition_Pct directly for Hurdle 2 target

# Features — full thesis deployment-eligible set (leakage-free at filing date)
FEATS = [
    # Site geometry / administrative
    "general_land_use", "shape_area", "council_district",
    # Parcel economics (TCAD EARS)
    "market_value", "land_market_value", "improvement_market_value",
    "land_acres", "building_age", "improvement_sq_ft", "improvement_ratio",
    # Zoning intensity (PDF-extracted, filing-date)
    "pdf_requested_height_ft", "pdf_proposed_height_ft",
    "pdf_requested_max_far", "pdf_compatibility_height_ft",
    # Demographics (ACS)
    "total_population", "median_household_income",
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "race_hispanic", "median_age",
    # Macro drivers
    "mortgage_rate_30yr", "local_unemployment_rate",
    "fed_funds_rate", "treasury_10yr_yield",
    "mortgage_rate_30yr_momentum", "local_unemployment_rate_momentum",
    "fed_funds_rate_momentum", "treasury_10yr_yield_momentum",
    # Spatial contagion / gravity
    "knn_petition_rate_1km", "dist_petition_rate_lag1",
    "active_cases_1km", "active_cases_2km",
    # NLP filing-day signals (leakage-safe: period_seq==1 only)
    "nlp_document_count", "nlp_oppose_hits", "nlp_traffic_hits", "nlp_density_hits",
    # Developer / process velocity
    "hearing_velocity_3p", "max_opponent_experience",
]
df["general_land_use"] = df["general_land_use"].astype(str).fillna("UNKNOWN")

# Ensure numeric types and fill NaNs — CatBoost handles missing natively but be explicit
numeric_feats = [f for f in FEATS if f not in ["general_land_use", "council_district"]]
for f in numeric_feats:
    df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)

# Wait, df does not have 'year', it has 'App_Date'
df["year"] = pd.to_datetime(df["App_Date"], errors="coerce").dt.year
df = df.dropna(subset=["year"])

cat_features = ["general_land_use"]

print("============================================================")
print(" STAGE 1: PROTEST PROBABILITY CLASSIFIER - MULTI-HORIZON MATRIX ")
print("============================================================")

results = []

for anchor in range(2016, 2024):
    for horizon in [1, 2, 3, 4, 5]:
        target_year = anchor + horizon
        if target_year > 2024:
            continue
            
        train_mask = df["year"] <= anchor
        test_mask = df["year"] == target_year
        
        X_train = df[train_mask][FEATS]
        y1_train = df[train_mask]["is_protested"]
        
        X_test = df[test_mask][FEATS]
        y1_test = df[test_mask]["is_protested"]
        
        if len(X_train) == 0 or len(X_test) == 0 or y1_train.sum() == 0 or y1_test.sum() == 0:
            continue
        
        clf = CatBoostClassifier(iterations=250, learning_rate=0.05, depth=6, task_type="GPU", verbose=0)
        clf.fit(X_train, y1_train, cat_features=cat_features, eval_set=(X_test, y1_test), early_stopping_rounds=30)
        
        y1_pred_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y1_test, y1_pred_prob)
        pr = average_precision_score(y1_test, y1_pred_prob)
        
        # Hurdle 2 (Severity)
        h2_train_mask = train_mask & (df["is_protested"] == 1)
        h2_test_mask = test_mask & (df["is_protested"] == 1)
        
        X2_train = df[h2_train_mask][FEATS]
        y2_train = df[h2_train_mask]["Valid_Petition_Pct"]
        
        X2_test = df[h2_test_mask][FEATS]
        y2_test = df[h2_test_mask]["Valid_Petition_Pct"]
        
        mae = np.nan
        if len(X2_train) > 0 and len(X2_test) > 0:
            reg = CatBoostRegressor(iterations=250, learning_rate=0.05, depth=6, task_type="GPU", verbose=0, loss_function="MAE")
            reg.fit(X2_train, y2_train, cat_features=cat_features, eval_set=(X2_test, y2_test), early_stopping_rounds=30)
            y2_pred = reg.predict(X2_test)
            mae = mean_absolute_error(y2_test, y2_pred)
        
        results.append({
            "Anchor": anchor,
            "Horizon": f"T+{horizon}",
            "Test_Year": target_year,
            "Test_N": len(X_test),
            "Test_Base_Rate": f"{y1_test.mean()*100:.2f}%",
            "H1_PR_AUC": pr,
            "H1_ROC_AUC": auc,
            "H2_MAE": mae
        })
        
        print(f"Anchor {anchor} -> Horizon T+{horizon} ({target_year}): PR={pr:.4f} | AUC={auc:.4f} | MAE={mae:.2f}% | N={len(X_test)}")

print("\n--- Horizon Attrition Summary (Averaged across Anchors) ---")
res_df = pd.DataFrame(results)
agg = res_df.groupby("Horizon")[["H1_PR_AUC", "H1_ROC_AUC", "H2_MAE"]].mean().reset_index()
print(agg.to_string(index=False))

print("\n--- Full Multi-Horizon Matrix ---")
matrix = res_df.pivot(index="Anchor", columns="Horizon", values="H1_PR_AUC")
print(matrix.round(4).fillna("---"))
