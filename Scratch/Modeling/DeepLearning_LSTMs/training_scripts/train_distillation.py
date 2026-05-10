import os
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

PANEL_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"
TARGET = "petition_event" # We are specifically testing mobilization KD

print("1. Loading Bi-Weekly Panel and Teacher Probabilities...")
df = pd.read_csv(PANEL_PATH, low_memory=False)
probs = pd.read_csv(r"C:\Users\dhl\data\Thesis\thesis\Data\Panel\gru_probs_petition_event.csv")

probs["period_seq"] = probs["period_seq"].astype(int)
df = df.merge(probs, on=["case_number", "period_seq"], how="inner")
df = df[df["period_seq"] > 0].copy()

FEATS = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings",
    "commission_hearings_this_period", "cumulative_commission_hearings",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
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
model_df = df[["case_number", TARGET, "gru_prob"] + FEATS].copy()
model_df[FEATS] = model_df[FEATS].fillna(0)

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(model_df, groups=model_df["case_number"]))

train = model_df.iloc[train_idx]
test  = model_df.iloc[test_idx]

X_train, y_train_soft = train[FEATS], train["gru_prob"]
X_test, y_test_hard   = test[FEATS], test[TARGET]

print("\n3. Training CatBoost Student Model (Regressor on Soft Targets)...")
clf = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    eval_metric='RMSE',
    random_seed=42,
    verbose=False
)

# We train on the soft targets (the GRU's continuous probability output)
clf.fit(
    X_train, y_train_soft,
    verbose=100
)

print("\n4. Evaluating Student on Hard Labels...")
# The output of the regressor is already the probability estimate
y_pred_proba = clf.predict(X_test)
# Clip to [0,1] just in case of regression overshoot
y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)
y_pred = (y_pred_proba > 0.5).astype(int)

print(f"   ROC AUC: {roc_auc_score(y_test_hard, y_pred_proba):.4f}")
print(f"   PR AUC:  {average_precision_score(y_test_hard, y_pred_proba):.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test_hard, y_pred))
