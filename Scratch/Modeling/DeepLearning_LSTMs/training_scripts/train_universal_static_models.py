import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, mean_squared_error, r2_score

BASE = r"C:\Users\dhl\data\Thesis\thesis\Data\Panel"
ALLOCATION_CSV = os.path.join(BASE, "spatial_allocation_panel.csv")
PANEL_CSV = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"

print("1. Loading Datasets...")
# Load non-zoning parcels (Control Group)
alloc_df = pd.read_csv(ALLOCATION_CSV, low_memory=False)
control_df = alloc_df[alloc_df["is_rezoned"] == 0].copy()

# Ensure matching baseline features
BASE_FEATS = [
    "total_population", "median_household_income",
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age"
]

# Zoning specific features to pad with 0
ZONING_FEATS_TO_PAD = [
    "proposed_max_height_ft", "proposed_max_far", "existing_max_far",
    "proposed_max_bldg_cov_pct", "existing_max_bldg_cov_pct",
    "dist_petition_rate_lag1", "knn_petition_rate_1km"
]

FEATS = BASE_FEATS + ZONING_FEATS_TO_PAD + [
    "existing_max_height_ft_pdf", "existing_max_height_ft_ldb", 
    "height_delta_pdf", "height_delta_ldb", "far_delta"
]

# Create targets and padding for control group
for target in ["label_valid_protest", "resolved", "existing_max_height_ft_pdf", "height_delta_pdf", "height_delta_ldb", "far_delta"] + ZONING_FEATS_TO_PAD:
    control_df[target] = 0
control_df["label_petition_total_pct"] = 0.0

# Assign the mapped statutory LDB height for the control group
control_df["existing_max_height_ft_ldb"] = alloc_df[alloc_df["is_rezoned"] == 0]["existing_max_height_ft_mapped"].fillna(0)

# Load zoning cases (Treatment Group)
panel_df = pd.read_csv(PANEL_CSV, low_memory=False)
treated_df = panel_df[panel_df["period_seq"] == 1].copy()

# Preserve PDF-extracted height
treated_df["existing_max_height_ft_pdf"] = treated_df["existing_max_height_ft"]

# Pull in Temporal Statutory LDB height
temp_heights = pd.read_csv(os.path.join(BASE, "temporal_case_heights.csv"))
treated_df = treated_df.merge(temp_heights, on="case_number", how="left")
treated_df["existing_max_height_ft_ldb"] = treated_df["existing_max_height_ft_statutory"].fillna(0)

# Fill missing target/zoning values in treatment
treated_df["label_petition_total_pct"] = treated_df["label_petition_total_pct"].fillna(0)
for z_feat in ZONING_FEATS_TO_PAD + ["existing_max_height_ft_pdf"]:
    treated_df[z_feat] = treated_df[z_feat].fillna(0)

# Engineer explicit deltas for the treatment group
treated_df["height_delta_pdf"] = treated_df["proposed_max_height_ft"] - treated_df["existing_max_height_ft_pdf"]
treated_df["height_delta_ldb"] = treated_df["proposed_max_height_ft"] - treated_df["existing_max_height_ft_ldb"]
treated_df["far_delta"] = treated_df["proposed_max_far"] - treated_df["existing_max_far"]

# Align columns
control_df = control_df[FEATS + ["label_valid_protest", "resolved", "label_petition_total_pct"]]
treated_df = treated_df[FEATS + ["label_valid_protest", "resolved", "label_petition_total_pct"]]

# Concatenate into the Universal Panel
universe = pd.concat([control_df, treated_df], ignore_index=True)
print(f"   Constructed Universal Panel: {universe.shape[0]} total parcels ({len(treated_df)} zoning, {len(control_df)} non-zoning)")

# Fill missing ACS features
for col in BASE_FEATS:
    universe[col] = universe[col].fillna(universe[col].median())

# ==========================================
# MODEL 1: THE "WHAT" (label_valid_protest)
# ==========================================
print("\n" + "="*50)
print("Training Model 1: Universal Valid Protest Classifier")
print("="*50)
TARGET = "label_valid_protest"

X_train, X_test, y_train, y_test = train_test_split(
    universe[FEATS], universe[TARGET], test_size=0.2, random_state=42, stratify=universe[TARGET]
)

pos_count = y_train.sum()
print(f"Base Rate: {(pos_count / len(y_train))*100:.3f}%")
scale_pos_weight = (len(y_train) - pos_count) / pos_count

clf = CatBoostClassifier(
    iterations=300, learning_rate=0.05, depth=6,
    scale_pos_weight=scale_pos_weight, eval_metric='PRAUC',
    random_seed=42, verbose=0
)
clf.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"PR AUC:  {average_precision_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))

# ==========================================
# MODEL 2: THE "HOW MUCH" (label_petition_total_pct)
# ==========================================
print("\n" + "="*50)
print("Training Model 2: Universal Petition Magnitude Regressor")
print("="*50)
TARGET = "label_petition_total_pct"

X_train, X_test, y_train, y_test = train_test_split(
    universe[FEATS], universe[TARGET], test_size=0.2, random_state=42
)

reg = CatBoostRegressor(
    iterations=300, learning_rate=0.05, depth=6,
    random_seed=42, verbose=0
)
reg.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

y_pred = reg.predict(X_test)
y_pred = np.clip(y_pred, 0, 100) # clip to physical pct bounds

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R-squared: {r2_score(y_test, y_pred):.4f}")
