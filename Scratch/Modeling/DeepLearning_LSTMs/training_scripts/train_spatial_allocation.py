import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\Panel\spatial_allocation_panel.csv"

print("1. Loading Spatial Allocation Panel...")
df = pd.read_csv(PANEL_PATH, low_memory=False)
print(f"   Loaded shape: {df.shape}")

TARGET = "is_rezoned"

FEATS = [
    "lui_general_land_use", "lui_shape_area", "council_district",
    "total_population", "median_household_income", "median_home_value",
    "median_gross_rent", "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "race_black", "race_hispanic", "median_age"
]

print("\n2. Preprocessing...")
model_df = df[[TARGET] + FEATS].copy()

# Fill NaNs natively or pass explicitly
model_df["lui_general_land_use"] = model_df["lui_general_land_use"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    model_df[FEATS], model_df[TARGET], test_size=0.2, random_state=42, stratify=model_df[TARGET]
)

print(f"   Train samples: {len(X_train)}")
print(f"   Test samples:  {len(X_test)}")
print(f"   Event rate: {(y_train.sum() / len(y_train))*100:.2f}% (Highly Imbalanced Spatial Target)")

print("\n3. Training CatBoostClassifier (Stage 1 Allocation)...")
pos_count = y_train.sum()
scale_pos_weight = (len(y_train) - pos_count) / pos_count

cat_features = ["lui_general_land_use"]

clf = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    scale_pos_weight=scale_pos_weight,
    eval_metric='PRAUC',
    cat_features=cat_features,
    random_seed=42,
    verbose=100
)

clf.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

print("\n4. Evaluation...")
y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

print(f"   ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"   PR AUC:  {average_precision_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))

print("\n5. Feature Importance...")
imp = pd.DataFrame({"Feature": FEATS, "Importance": clf.feature_importances_})
imp = imp.sort_values("Importance", ascending=False).head(10)
print(imp.to_string(index=False))
