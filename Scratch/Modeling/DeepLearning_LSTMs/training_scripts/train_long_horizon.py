import os
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

PANEL_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"

print("1. Loading Bi-Weekly Panel...")
df = pd.read_csv(PANEL_PATH, low_memory=False)

print("2. Engineering 3-Month Forward Target...")
# Sort explicitly by case and period to ensure rolling logic is perfect
df = df.sort_values(["case_number", "period_seq"])

# We want the max value of petition_event in the NEXT 6 periods (t+1 to t+6)
# Reverse, rolling max, reverse back, shift by -1 (to exclude current period t)
df["petition_event_3mo"] = df.groupby("case_number")["petition_event"].transform(
    lambda x: x.iloc[::-1].rolling(window=6, min_periods=1).max().iloc[::-1].shift(-1)
)

# Fill NAs with 0 (if case ends before 6 periods, no future protests happened)
df["petition_event_3mo"] = df["petition_event_3mo"].fillna(0)

# Ensure it's int
df["petition_event_3mo"] = df["petition_event_3mo"].astype(int)

TARGET = "petition_event_3mo"

# Filter strictly to valid periods (period_seq > 0)
df = df[df["period_seq"] > 0].copy()

FEATS = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "lag1_cumulative_council_hearings",
    "commission_hearings_this_period", "lag1_cumulative_commission_hearings",
    "lag1_cumulative_petition_events", "lag1_cumulative_petition_count", "lag1_cumulative_petition_pct", 
    "lag1_Remand_Count",
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

print(f"   Event rate (3-mo horizon): {(df[TARGET].sum() / len(df))*100:.2f}%")

print("\n3. Preprocessing...")
model_df = df[["case_number", TARGET] + FEATS].copy()
model_df[FEATS] = model_df[FEATS].fillna(0)

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(model_df, groups=model_df["case_number"]))

train = model_df.iloc[train_idx]
test  = model_df.iloc[test_idx]

X_train, y_train = train[FEATS], train[TARGET]
X_test, y_test   = test[FEATS], test[TARGET]

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

print("\n4. Training CatBoost Forward-Horizon Model...")
clf = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='AUC',
    random_seed=42,
    verbose=False
)

clf.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
    verbose=100
)

print("\n5. Evaluation...")
y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

print(f"   ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"   PR AUC:  {average_precision_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))

print("\n6. Feature Importance...")
# Simple weight importance
imp = pd.DataFrame({"Feature": FEATS, "Importance": clf.feature_importances_})
imp = imp.sort_values("Importance", ascending=False).head(10)
print(imp.to_string(index=False))
