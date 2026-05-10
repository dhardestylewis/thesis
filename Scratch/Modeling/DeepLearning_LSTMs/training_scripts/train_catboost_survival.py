import os
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"

print("1. Loading Bi-Weekly Panel...")
df = pd.read_csv(PANEL_PATH, low_memory=False)

TARGET = sys.argv[1] if len(sys.argv) > 1 else "vote_event"
print(f"TARGET: {TARGET}")

df = df[df["period_seq"] > 0].copy()

FEATS = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings_lag1",
    "commission_hearings_this_period", "cumulative_commission_hearings_lag1",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
    "cumulative_min_signer_dist", "cumulative_max_signer_dist", "cumulative_median_signer_dist", 
    "cumulative_signers_within_200ft", "cumulative_signers_outside_200ft", 
    "cumulative_unofficial_protest_intensity", 
    "cumulative_protester_embed_dim1", "cumulative_protester_embed_dim2", "cumulative_protester_embed_dim3", "cumulative_protester_embed_dim4",
    "cumulative_temporal_protesting_pct_sf", "cumulative_temporal_silent_pct_sf",
    "cumulative_temporal_protesting_pct_com", "cumulative_temporal_silent_pct_com",
    "cumulative_temporal_protesting_pct_mf", "cumulative_temporal_silent_pct_mf",
    "cumulative_delta_protesting_friction", "cumulative_delta_silent_friction",
    "Remand_Count",
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
    "treasury_10yr_yield", "treasury_10yr_yield_filing_delta", 
    "fed_funds_rate", "fed_funds_rate_filing_delta", 
    "local_unemployment_rate", "local_unemployment_rate_filing_delta",
    "knn_petition_rate_1km", "dist_petition_rate_lag1"
]

print("\n2. Preprocessing...")
model_df = df[["case_number", TARGET] + FEATS].copy()
model_df[FEATS] = model_df[FEATS].fillna(0)

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(model_df, groups=model_df["case_number"]))

train = model_df.iloc[train_idx]
test  = model_df.iloc[test_idx]

X_train, y_train = train[FEATS], train[TARGET]
X_test, y_test   = test[FEATS], test[TARGET]

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

print("\n3. Training CatBoost Hazard Model...")
clf = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='AUC',
    random_seed=42,
    verbose=False,
    task_type="GPU"
)

clf.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
    verbose=100
)

print("\n4. Evaluation...")
y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

print(f"   ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"   PR AUC:  {average_precision_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))
