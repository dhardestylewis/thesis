import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupShuffleSplit
import shap
import warnings
import sys
warnings.filterwarnings('ignore')

PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"
OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts"

FEATS_DYNAMIC = [
    "period_seq", "bw_sin", "bw_cos",
    "council_hearings_this_period", "cumulative_council_hearings",
    "commission_hearings_this_period", "cumulative_commission_hearings",
    "cumulative_petition_events", "cumulative_petition_count", "cumulative_petition_pct", 
    "Remand_Count",
    "market_value", "building_age", "land_acres",
    "total_population", "median_household_income", 
    "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "median_age",
    "mortgage_rate_30yr", "mortgage_rate_30yr_momentum",
    "treasury_10yr_yield", "fed_funds_rate", "local_unemployment_rate",
    "knn_petition_rate_1km", "dist_petition_rate_lag1"
]

def main():
    print("1. Loading Bi-Weekly Panel...")
    df = pd.read_csv(PANEL_PATH, low_memory=False)
    
    # Filter to longitudinal (period_seq > 0)
    df_hazard = df[df["period_seq"] > 0].copy()
    
    target = "petition_event"
    print(f"\n2. Training NIMBY Mobilization Hazard Model (Target: {target})...")
    
    model_df = df_hazard[["case_number", target] + FEATS_DYNAMIC].copy()
    model_df = model_df.dropna()
    
    # Train-Test Split by case to prevent leakage (though we'll explain the whole set for SHAP to maximize sample)
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(model_df, groups=model_df["case_number"]))
    
    train = model_df.iloc[train_idx]
    
    X_train, y_train = train[FEATS_DYNAMIC], train[target]
    
    pos_count = y_train.sum()
    scale_pos_weight = (len(y_train) - pos_count) / pos_count if pos_count > 0 else 1.0
    
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric='PRAUC',
        random_seed=42,
        verbose=100,
        task_type="GPU"
    )
    
    pool_train = Pool(X_train, y_train)
    model.fit(pool_train)
    
    print("\n3. Executing SHAP TreeExplainer (Interactions) on full dataset...")
    # Use the full dataset for SHAP to get a complete feature attribution map
    X_all = model_df[FEATS_DYNAMIC]
    y_all = model_df[target]
    pool_all = Pool(X_all, y_all)
    
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(pool_all)
    
    print("4. Rendering SHAP Interaction Summary Plot...")
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_interaction_values, X_all, max_display=12, show=False)
    
    plt.title(f"SHAP Feature Interactions: Longitudinal NIMBY Hazard\n(CatBoost predicting period petition event)", fontsize=14, weight="bold", y=1.05)
    plt.tight_layout()
    out_path_summary = rf"{OUT_DIR}\causal_shap_interaction_summary_biweekly.png"
    plt.savefig(out_path_summary, bbox_inches="tight")
    print(f"Interaction summary artifact saved to {out_path_summary}")
    plt.close()

if __name__ == "__main__":
    main()
