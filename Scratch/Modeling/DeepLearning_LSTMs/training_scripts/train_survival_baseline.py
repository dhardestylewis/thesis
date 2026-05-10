import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import shap
import sys

PANEL_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"

print("1. Loading Bi-Weekly Panel...")
df = pd.read_csv(PANEL_PATH, low_memory=False)
print(f"   Loaded shape: {df.shape}")

# Define Target for Discrete-Time Hazard Model
TARGET = sys.argv[1] if len(sys.argv) > 1 else "vote_event"
print(f"TARGET: {TARGET}")

# Filter out periods after the vote event happens (in a strict hazard model, 
# a case is removed from the risk set once the event occurs).
# Our skeleton build already stops the sequence at T_end (which is T_vote for resolved cases),
# so the last row for resolved cases is the event period.
df = df[df["period_seq"] > 0] # ensure basic validity

# Select Features
FEATS = [
    # Temporal & Cyclical
    "period_seq", "bw_sin", "bw_cos",
    
    # Milestone Flags & Counts
    "council_hearings_this_period", "cumulative_council_hearings",
    "commission_hearings_this_period", "cumulative_commission_hearings",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
    "Remand_Count",
    
    # Parcel Data
    "market_value", "building_age", "land_acres",
    
    # Demographics (ACS)
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    
    # Macroeconomic (FRED)
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
    "treasury_10yr_yield", "treasury_10yr_yield_filing_delta", 
    "fed_funds_rate", "fed_funds_rate_filing_delta", 
    "local_unemployment_rate", "local_unemployment_rate_filing_delta",
    
    # Spatial Lags
    "knn_petition_rate_1km", "dist_petition_rate_lag1"
]

print("\n2. Preprocessing...")
# Keep only features we selected
model_df = df[["case_number", TARGET] + FEATS].copy()

# Handle missingness for XGBoost (XGB handles NaNs natively, but we ensure no complete column drops)
print(f"   Features to train on: {len(FEATS)}")

# Group train/test split by case_number so no leakage occurs across time periods!
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(model_df, groups=model_df["case_number"]))

train = model_df.iloc[train_idx]
test  = model_df.iloc[test_idx]

X_train, y_train = train[FEATS], train[TARGET]
X_test, y_test   = test[FEATS], test[TARGET]

print(f"   Train samples: {len(X_train)} (Cases: {train['case_number'].nunique()})")
print(f"   Test samples:  {len(X_test)} (Cases: {test['case_number'].nunique()})")
print(f"   Event rate: {(y_train.sum() / len(y_train))*100:.2f}% (highly imbalanced risk set)")

print("\n3. Training Baseline XGBoost Hazard Model...")
# Scale pos weight for imbalanced target (rare event per period)
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    random_state=42,
    n_jobs=-1
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("\n4. Evaluation...")
y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

print(f"   ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"   PR AUC:  {average_precision_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))

print("\n5. Feature Importance...")
# Simple weight importance
imp = pd.DataFrame({"Feature": FEATS, "Importance": clf.feature_importances_})
imp = imp.sort_values("Importance", ascending=False).head(10)
print(imp.to_string(index=False))

print("\n6. Generating SHAP Plot...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
out_file = f"C:\\Users\\dhl\\.gemini\\antigravity\\brain\\1c4648c0-f36a-4614-a8f1-c9e2e5621756\\artifacts\\shap_summary_{TARGET}.png"
plt.savefig(out_file, bbox_inches='tight', dpi=300)
plt.close()
print(f"\nSaved {out_file}")
