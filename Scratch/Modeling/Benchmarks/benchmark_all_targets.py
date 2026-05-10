import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

PANEL_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"
OUT_MD = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\artifacts\catboost_benchmarks.md"

print("Loading Bi-Weekly Panel...")
df = pd.read_csv(PANEL_PATH, low_memory=False)

FEATS_DYNAMIC = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings",
    "commission_hearings_this_period", "cumulative_commission_hearings",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
    "Remand_Count",
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum",
    "treasury_10yr_yield", "fed_funds_rate", "local_unemployment_rate",
    "knn_petition_rate_1km", "dist_petition_rate_lag1",
    "proposed_max_height_ft","proposed_max_far","proposed_max_bldg_cov_pct",
    "existing_max_height_ft","existing_max_far","existing_max_bldg_cov_pct",
]

FEATS_STATIC = [
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum",
    "treasury_10yr_yield", "fed_funds_rate", "local_unemployment_rate",
    "knn_petition_rate_1km", "dist_petition_rate_lag1",
    "proposed_max_height_ft","proposed_max_far","proposed_max_bldg_cov_pct",
    "existing_max_height_ft","existing_max_far","existing_max_bldg_cov_pct",
]

def run_model(model_name, data, target, feats):
    print(f"\n{'='*50}\nTraining {model_name} (Target: {target})\n{'='*50}")
    
    # Preprocess
    model_df = data[["case_number", target] + feats].copy()
    
    # Split by case to prevent leakage
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(model_df, groups=model_df["case_number"]))
    
    train = model_df.iloc[train_idx]
    test  = model_df.iloc[test_idx]
    
    X_train, y_train = train[feats], train[target]
    X_test, y_test   = test[feats], test[target]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")
    
    pos_count = y_train.sum()
    if pos_count == 0:
        return f"## {model_name}\nError: No positive samples in training set."
        
    scale_pos_weight = (len(y_train) - pos_count) / pos_count
    
    clf = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric='PRAUC',
        random_seed=42,
        verbose=100
    )
    
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    cr = classification_report(y_test, y_pred)
    
    imp = pd.DataFrame({"Feature": feats, "Importance": clf.feature_importances_})
    imp = imp.sort_values("Importance", ascending=False).head(10)
    
    res = f"""## {model_name}
**Target:** `{target}`  
**Training Size:** {len(X_train)} | **Test Size:** {len(X_test)}

### Metrics
* **ROC AUC:** {roc_auc:.4f}
* **PR AUC:** {pr_auc:.4f}

### Top 10 Features
```text
{imp.to_string(index=False)}
```

### Classification Report
```text
{cr}
```
"""
    return res

results = []
results.append("# Multi-Target CatBoost Benchmarks\n\nThis artifact documents the performance of the Thesis CatBoost Model (`learning_rate=0.05`, `max_depth=6`) across three different econometric paradigms.\n")

# Model 1: Administrative Hazard (vote_event)
df_hazard = df[df["period_seq"] > 0]
results.append(run_model("1. Administrative Hazard Model (Longitudinal)", df_hazard, "vote_event", FEATS_DYNAMIC))

# Model 2: NIMBY Mobilization Hazard (petition_event)
results.append(run_model("2. NIMBY Mobilization Hazard Model (Longitudinal)", df_hazard, "petition_event", FEATS_DYNAMIC))

# Model 3: Day-1 Static Classifier (label_valid_protest)
df_static = df[df["period_seq"] == 1].copy()
df_static["is_remanded"] = (df_static["Remand_Count"] > 0).astype(int)

results.append(run_model("3. Day-1 Thesis Model (Static Cross-Sectional: label_valid_protest)", df_static, "label_valid_protest", FEATS_STATIC))

# Model 4: Day-1 Static Classifier (resolved)
results.append(run_model("4. Day-1 Classifier (Static Cross-Sectional: resolved)", df_static, "resolved", FEATS_STATIC))

# Model 5: Day-1 Static Classifier (is_remanded)
results.append(run_model("5. Day-1 Classifier (Static Cross-Sectional: is_remanded)", df_static, "is_remanded", FEATS_STATIC))

with open(OUT_MD, "w") as f:
    f.write("\n---\n".join(results))

print(f"\nSaved benchmarks to {OUT_MD}")
