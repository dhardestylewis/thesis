import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, mean_absolute_error, r2_score

PARQUET_PATH = r"c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\04_T_Learner_ML\annualized_all_parcel_panel.parquet"

print("1. Loading Annualized All-Parcel Panel (2.8M rows)...")
df = pd.read_parquet(PARQUET_PATH)

# TARGETS
TARGET_H1 = "is_filed_this_year"
TARGET_H2 = "Valid_Petition_Pct"

# FEATURES
FEATS = [
    "lui_general_land_use", "lui_shape_area", "council_district",
    "total_population", "median_household_income", "median_home_value",
    "median_gross_rent", "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "race_black", "race_hispanic", "median_age",
    "market_value", "appraised_value", "land_acres", "building_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", 
    "fed_funds_rate", "fed_funds_rate_momentum",
    "local_unemployment_rate", "local_unemployment_rate_momentum"
]

print("\n2. Preprocessing & Temporal Splitting...")
model_df = df[["year", TARGET_H1, TARGET_H2] + FEATS].copy()
model_df["lui_general_land_use"] = model_df["lui_general_land_use"].astype(str)

# Train: 2007-2021 | Test: 2022-2025
train_mask = model_df["year"] <= 2021
test_mask = model_df["year"] > 2021

X_train = model_df[train_mask][FEATS]
X_test = model_df[test_mask][FEATS]

# ── HURDLE 1: FILING PROBABILITY (All Parcels) ───────────────────────────────
print("\n" + "="*60)
print(" HURDLE 1: PROSPECTIVITY (FILING CLASSIFIER)")
print("="*60)

y1_train = model_df[train_mask][TARGET_H1]
y1_test = model_df[test_mask][TARGET_H1]

print(f"   Train samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"   Train Event rate: {(y1_train.sum() / len(y1_train))*100:.3f}%")

pos_count = y1_train.sum()
scale_pos_weight = (len(y1_train) - pos_count) / pos_count if pos_count > 0 else 1

cat_features = ["lui_general_land_use"]

clf = CatBoostClassifier(
    iterations=250,
    learning_rate=0.1,
    depth=6,
    scale_pos_weight=scale_pos_weight,
    cat_features=cat_features,
    task_type="GPU",
    random_seed=42,
    verbose=50
)

print("   Training Hurdle 1...")
clf.fit(X_train, y1_train, eval_set=(X_test, y1_test), use_best_model=True)

y1_pred_proba = clf.predict_proba(X_test)[:, 1]
print(f"\n   [Hurdle 1 Test Metrics]")
print(f"   ROC AUC: {roc_auc_score(y1_test, y1_pred_proba):.4f}")
print(f"   PR AUC:  {average_precision_score(y1_test, y1_pred_proba):.4f}")


# ── HURDLE 2: PROTEST SEVERITY (Conditional on Filing) ───────────────────────
print("\n" + "="*60)
print(" HURDLE 2: PROTEST SEVERITY (CONDITIONAL REGRESSOR)")
print("="*60)

# Filter for the cases where a filing actually occurred (Ground Truth)
h2_train_mask = train_mask & (model_df[TARGET_H1] == 1)
h2_test_mask = test_mask & (model_df[TARGET_H1] == 1)

X2_train = model_df[h2_train_mask][FEATS]
y2_train = model_df[h2_train_mask][TARGET_H2]

X2_test = model_df[h2_test_mask][FEATS]
y2_test = model_df[h2_test_mask][TARGET_H2]

print(f"   Train cases: {len(X2_train)} | Test cases: {len(X2_test)}")
print(f"   Average train petition %: {y2_train.mean():.1f}%")

reg = CatBoostRegressor(
    iterations=250,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    task_type="GPU",
    random_seed=42,
    verbose=0
)

print("   Training Hurdle 2...")
reg.fit(X2_train, y2_train, eval_set=(X2_test, y2_test), use_best_model=True)

y2_pred = reg.predict(X2_test)

print(f"\n   [Hurdle 2 Test Metrics (Conditional)]")
print(f"   MAE: {mean_absolute_error(y2_test, y2_pred):.2f}%")
print(f"   R2:  {r2_score(y2_test, y2_pred):.4f}")

# Convert continuous petition severity to binary Protested (>= 20%)
y2_test_binary = (y2_test >= 20).astype(int)
y2_pred_binary = (y2_pred >= 20).astype(int)
print("\n   [Hurdle 2 Binary Classification (Threshold >= 20%)]")
print(classification_report(y2_test_binary, y2_pred_binary, zero_division=0))


# ── SYNTHESIS: UNCONDITIONAL EXPECTATION ────────────────────────────────────
print("\n" + "="*60)
print(" SYNTHESIS: UNCONDITIONAL PROTEST SEVERITY")
print("="*60)

# The unconditional predicted severity across the entire test map
# E[Petition] = P(Filing) * E[Petition | Filing]
E_petition_given_filing = reg.predict(X_test)
E_petition_given_filing = np.clip(E_petition_given_filing, 0, 100)  # constrain %

unconditional_expected_severity = y1_pred_proba * E_petition_given_filing

print(f"   Total test parcels evaluated: {len(unconditional_expected_severity):,}")
print(f"   City-wide max expected severity: {unconditional_expected_severity.max():.2f}%")
print(f"   City-wide mean expected severity: {unconditional_expected_severity.mean():.4f}%")

print("\nDone!")
