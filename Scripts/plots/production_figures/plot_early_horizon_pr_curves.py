import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from catboost import CatBoostClassifier
from pathlib import Path
import os
import sys

# Styling
try:
    sys.path.append(os.path.abspath('Scripts'))
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "Data/Panel/biweekly_panel.csv"
FIG_DIR = ROOT / "Thesis_Draft/Draft_v1/Figures/exhibits"
os.makedirs(FIG_DIR, exist_ok=True)

FEATS = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings_lag1",
    "commission_hearings_this_period", "cumulative_commission_hearings_lag1",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct",
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income",
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum", "mortgage_rate_30yr_filing_delta",
    "treasury_10yr_yield", "treasury_10yr_yield_filing_delta",
    "fed_funds_rate", "fed_funds_rate_filing_delta",
    "local_unemployment_rate", "local_unemployment_rate_filing_delta",
    "knn_petition_rate_1km", "dist_petition_rate_lag1",
    "active_cases_100m", "active_cases_250m", "active_cases_500m",
    "active_cases_1km", "active_cases_2km", "active_gravity_index_t",
    "hearing_frequency", "petition_intensity_per_ft",
    "hearing_velocity_3p", "petition_velocity_3p",
    "pdf_requested_height_ft", "pdf_requested_max_far", "pdf_proposed_height_ft",
    "pdf_story_count", "pdf_story_height_ft", "pdf_compatibility_height_ft",
]

def build_target(df: pd.DataFrame, window: int) -> pd.Series:
    if window == 1:
        return df["petition_event"].astype(int)
    target = df.groupby("case_number")["petition_event"].transform(
        lambda x: x.iloc[::-1].rolling(window=window, min_periods=1).max().iloc[::-1]
    )
    return target.fillna(0).astype(int)

print("Loading data...")
df_raw = pd.read_csv(PANEL_PATH, low_memory=False)
df_raw = df_raw.sort_values(["case_number", "period_seq"]).reset_index(drop=True)

first_petition = (
    df_raw[df_raw["petition_event"] == 1]
    .groupby("case_number")["period_seq"].min()
)
df_raw["first_petition_seq"] = df_raw["case_number"].map(first_petition)
df_raw = df_raw[
    df_raw["first_petition_seq"].isna() |
    (df_raw["period_seq"] <= df_raw["first_petition_seq"])
].drop(columns=["first_petition_seq"]).reset_index(drop=True)

feats = [f for f in FEATS if f in df_raw.columns]
X_all = df_raw[feats].fillna(0).values
year_arr = df_raw["year"].values

# Test cutoff: train < 2023, test >= 2023 (to get enough samples for smooth PR curves)
train_mask = year_arr < 2023
test_mask = year_arr >= 2023

X_tr = X_all[train_mask]
X_te = X_all[test_mask]

# Define Early Horizons
horizons = {
    "14 Days": 1,
    "1 Month": 2,
    "3 Months": 6
}

colors = {
    "14 Days": "navy",
    "1 Month": "teal",
    "3 Months": "coral"
}

plt.figure(figsize=(7, 6))

for h_name, window in horizons.items():
    print(f"Evaluating {h_name}...")
    y_all = build_target(df_raw, window).values
    y_tr = y_all[train_mask]
    y_te = y_all[test_mask]
    
    spw = max(1.0, (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
    
    clf = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="AUC", 
        random_seed=42, verbose=False, task_type="CPU", thread_count=-1
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict_proba(X_te)[:, 1]
    
    p, r, _ = precision_recall_curve(y_te, y_pred)
    auc = average_precision_score(y_te, y_pred)
    
    plt.plot(r, p, '-', label=f"CatBoost ({h_name}) (AUC={auc:.3f})", color=colors[h_name], linewidth=2)
    
    if h_name == "14 Days":
        base_rate = float(y_te.mean())
        plt.axhline(base_rate, color='black', linestyle=':', label=f"Random Chance 14d (AUC={base_rate:.3f})")

plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.title("Early-Horizon Precision-Recall Decay (CatBoost)", pad=15)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_file = os.path.join(FIG_DIR, "fig_pr_curves_early_horizon.pdf")
plt.savefig(out_file)
print(f"Saved: {out_file}")
