import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score

PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"

print("1. Loading Bi-Weekly Panel...")
df_raw = pd.read_csv(PANEL_PATH, low_memory=False)
df_raw = df_raw.sort_values(["case_number", "period_seq"])

horizons = {
    "14_Days": 1,
    "3_Months": 6,
    "6_Months": 13,
    "1_Year": 26,
    "1.5_Years": 39,
    "2_Years": 52,
    "3_Years": 78,
    "4_Years": 104
}

FEATS = [
    # Process & Bureaucracy
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings_lag1",
    "commission_hearings_this_period", "cumulative_commission_hearings_lag1",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
    "Remand_Count",
    
    # Advanced Causal Spatial & Temporal Features
    "cumulative_min_signer_dist", "cumulative_max_signer_dist", "cumulative_median_signer_dist", 
    "cumulative_signers_within_200ft", "cumulative_signers_outside_200ft", 
    "cumulative_unofficial_protest_intensity", 
    "cumulative_protester_embed_dim1", "cumulative_protester_embed_dim2", "cumulative_protester_embed_dim3", "cumulative_protester_embed_dim4",
    "cumulative_temporal_protesting_pct_sf", "cumulative_temporal_silent_pct_sf",
    "cumulative_temporal_protesting_pct_com", "cumulative_temporal_silent_pct_com",
    "cumulative_temporal_protesting_pct_mf", "cumulative_temporal_silent_pct_mf",
    "cumulative_delta_protesting_friction", "cumulative_delta_silent_friction",
    
    # Economics & Demographics
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    
    # Macro Shocks
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
    "treasury_10yr_yield", "treasury_10yr_yield_filing_delta", 
    "fed_funds_rate", "fed_funds_rate_filing_delta", 
    "local_unemployment_rate", "local_unemployment_rate_filing_delta",
    
    # Spatial Gravity
    "knn_petition_rate_1km", "dist_petition_rate_lag1"
]

results = []
feature_importances = {}

for name, window in horizons.items():
    print(f"\n======================================")
    print(f"Executing Horizon: {name} ({window} periods)")
    
    df = df_raw.copy()
    
    if window == 1:
        df["target"] = df["petition_event"]
    else:
        # Reverse, rolling max, reverse back, shift by -1
        df["target"] = df.groupby("case_number")["petition_event"].transform(
            lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1].shift(-1)
        )
        df["target"] = df["target"].fillna(0)
    
    df["target"] = df["target"].astype(int)
    
    # Filter to valid periods
    df = df[df["period_seq"] > 0]
    
    event_rate = df["target"].sum() / len(df)
    print(f"Event Rate: {event_rate*100:.2f}%")
    
    model_df = df[["case_number", "target"] + FEATS].copy()
    model_df[FEATS] = model_df[FEATS].fillna(0)
    
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(model_df, groups=model_df["case_number"]))
    
    train = model_df.iloc[train_idx]
    test  = model_df.iloc[test_idx]
    
    X_train, y_train = train[FEATS], train["target"]
    X_test, y_test   = test[FEATS], test["target"]
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
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
    
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=False)
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"PR AUC: {pr_auc:.4f} | ROC AUC: {roc_auc:.4f}")
    
    results.append({
        "Horizon": name,
        "Window": window,
        "Event_Rate": event_rate,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc
    })
    
    # Save importance for 6 months
    if name == "6_Months":
        imp = pd.DataFrame({"Feature": FEATS, "Importance": clf.feature_importances_})
        imp = imp.sort_values("Importance", ascending=False).head(10)
        print("\nTop Features for 6-Month Horizon:")
        print(imp.to_string(index=False))

print("\nSaving Gravity Curve...")
res_df = pd.DataFrame(results)

sns.set_theme(style="whitegrid", palette="rocket")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary Axis: PR AUC
sns.lineplot(data=res_df, x="Horizon", y="PR_AUC", marker="o", linewidth=3, ax=ax1, color="#b30000", label="PR AUC")
ax1.set_ylabel("Precision-Recall AUC (Predictive Power)", fontsize=12, fontweight='bold', color="#b30000")
ax1.set_ylim(0, max(res_df["PR_AUC"].max() * 1.5, 0.1))

# Secondary Axis: Event Rate
ax2 = ax1.twinx()
sns.barplot(data=res_df, x="Horizon", y="Event_Rate", alpha=0.3, ax=ax2, color="gray")
ax2.set_ylabel("Baseline Event Rate", fontsize=12, color="gray")
ax2.grid(False)

plt.title("The Gravity Curve: Predictive Power Scales with Macro Horizon", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\horizon_gravity_curve_with_arch.png", dpi=300, bbox_inches='tight')
print("Complete.")
