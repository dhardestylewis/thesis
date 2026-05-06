"""
run_spatial_ablation.py
========================
Evaluates the marginal contribution of each advanced feature group
by training a simple CatBoost model with and without each group.
Outputs a ranked feature importance table and ablation delta CSV.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from catboost import CatBoostClassifier

PANEL_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"
OUT_DIR    = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

print("Loading panel...")
df = pd.read_csv(PANEL_PATH, low_memory=False)
df = df.sort_values(["case_number", "period_seq"])

TARGET = "label_petition_total_pct"  # adjust to your actual binary target column

# Feature groups
BASELINE = [
    "proposed_max_height_ft", "land_acres", "period_seq",
    "cumulative_council_hearings", "cumulative_commission_hearings",
    "cumulative_petition_events", "cumulative_petition_pct",
    "knn_petition_rate_1km", "local_unemployment_rate", "mortgage_rate_30yr",
]

VELOCITY = ["hearing_velocity_3p", "petition_velocity_3p", "hearing_frequency",
            "petition_intensity_per_ft", "staff_concession_ratio"]

GRAPH    = ["max_opponent_experience"]

SPATIAL_RINGS = ["active_cases_100m", "active_cases_250m", "active_cases_500m",
                 "active_cases_1km", "active_cases_2km"]

GRAVITY  = ["active_gravity_index_t"]

def evaluate(feature_set, label):
    cols = [c for c in feature_set if c in df.columns]
    if not cols:
        print(f"  Skipping {label} — no columns found.")
        return None
    
    sub = df[cols + [TARGET, "case_number"]].dropna()
    
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(sub, groups=sub["case_number"]))
    
    X_train = sub.iloc[train_idx][cols]
    y_train = sub.iloc[train_idx][TARGET]
    X_test  = sub.iloc[test_idx][cols]
    y_test  = sub.iloc[test_idx][TARGET]
    
    if y_train.sum() == 0 or y_test.sum() == 0:
        print(f"  Skipping {label} — no positive cases in split.")
        return None
    
    model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4,
                                eval_metric="AUC", random_seed=42, verbose=0,
                                class_weights={0: 1, 1: int(len(y_train) / (y_train.sum() + 1))})
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    
    pr  = average_precision_score(y_test, preds)
    roc = roc_auc_score(y_test, preds)
    bs  = brier_score_loss(y_test, preds)
    
    print(f"  {label:40s} PR AUC: {pr:.4f} | ROC AUC: {roc:.4f} | Brier: {bs:.4f}")
    return {"Feature_Group": label, "N_Features": len(cols), "PR_AUC": pr, "ROC_AUC": roc, "Brier": bs}

results = []
print("\n--- Ablation Study ---")
r = evaluate(BASELINE, "Baseline Only")
if r: results.append(r)
r = evaluate(BASELINE + VELOCITY, "Baseline + Velocity")
if r: results.append(r)
r = evaluate(BASELINE + GRAPH, "Baseline + Graph Centrality")
if r: results.append(r)
r = evaluate(BASELINE + SPATIAL_RINGS, "Baseline + Spatial Rings")
if r: results.append(r)
r = evaluate(BASELINE + GRAVITY, "Baseline + Gravity Index")
if r: results.append(r)
r = evaluate(BASELINE + VELOCITY + GRAPH + SPATIAL_RINGS + GRAVITY, "Full Feature Set (All Groups)")
if r: results.append(r)

out = pd.DataFrame(results)
if len(out) > 0:
    # Delta vs baseline
    baseline_pr = out[out["Feature_Group"] == "Baseline Only"]["PR_AUC"].values
    if len(baseline_pr) > 0:
        out["Delta_PR_AUC_vs_Baseline"] = out["PR_AUC"] - baseline_pr[0]
    
    path = os.path.join(OUT_DIR, "spatial_ablation_results.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved ablation results to {path}")
    print(out.to_string(index=False))
