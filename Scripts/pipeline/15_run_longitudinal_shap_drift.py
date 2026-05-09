import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timezone
import warnings

warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier
import shap

# --- Configuration ---
ROOT_DIR = Path(r"c:\Users\dhl\data\Thesis\thesis")
REGISTRY_DIR = ROOT_DIR / "registries"
ARTIFACTS_DIR = Path(r"c:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical Training Configuration for Stage C
model_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'Logloss',
    'verbose': 0,
    'random_seed': 42,
    'task_type': 'GPU',
    'devices': '0'
}

# Features to exclude from training
def get_excluded_features():
    return [
        "case_id", "as_of_date", "feature_view", "split_id", "role", "fold",
        "threshold_crossed", "horizon", "label_version", "weight", "ipw_weight",
        "index", "latitude", "longitude"
    ]

# Human readable mapping
feature_mapping = {
    "acs2_median_household_income": "Median Household Income",
    "acs2_race_white": "% White Population",
    "acs2_race_hispanic": "% Hispanic Population",
    "acs2_median_home_value": "Median Home Value",
    "district_n_parcels_lag_6yr": "Density (Parcels)",
    "district_median_structure_age_lag_6yr": "Median Structure Age",
    "district_renter_share_lag_6yr": "Renter Share",
    "district_gross_site_area_acres_lag_6yr": "Site Area (Acres)",
    "district_delta_max_height_ft_lag_6yr": "Max Height Increase (ft)",
    "district_delta_max_far_lag_6yr": "Max FAR Increase",
    "district_protest_rate_lag_6yr": "Historical Protest Rate",
    "district_owner_occupancy_share_lag_6yr": "Owner Occupied Share",
    "district_median_household_income_lag_6yr": "Local Median Income",
    "district_bisg_white_nbr_lag_6yr": "Neighborhood % White",
    "district_bisg_hispanic_nbr_lag_6yr": "Neighborhood % Hispanic",
    "acs_median_household_income": "Median Household Income",
    "acs_total_population": "Total Population",
}

def clean_feat_name(f):
    return feature_mapping.get(f, f.replace("district_", "").replace("_lag_6yr", "").replace("_", " ").title())

def main():
    print("1. Loading Registries for Longitudinal Drift Analysis...")
    features = pd.read_parquet(ROOT_DIR / "data" / "interim" / "stage_c_features_raw.parquet")
    labels = pd.read_parquet(REGISTRY_DIR / "label_registry.parquet")
    
    # Filter to main horizon and reconstruct target
    labels = labels[labels["label_version"] == "label_v1_reconstructed_threshold_crossing"]
    
    df = features.merge(labels[["case_id", "threshold_crossed"]], on="case_id", how="inner")
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    
    # Exclusions
    excluded = get_excluded_features()
    feature_cols = [c for c in df.columns if c not in excluded]

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c in numeric_cols]

    years = [2020, 2021, 2022, 2023]
    
    drift_records = []
    
    print("\n2. Iterating over Temporal Cutoffs...")
    for year in years:
        print(f"   ---> Evaluating Cutoff Year: {year}")
        train_df = df[df["as_of_date"] < pd.Timestamp(f"{year}-01-01")]
        test_df = df[(df["as_of_date"] >= pd.Timestamp(f"{year}-01-01")) & (df["as_of_date"] < pd.Timestamp(f"{year+1}-01-01"))]
        
        X_train = train_df[feature_cols]
        y_train = train_df["threshold_crossed"].astype(int)
        
        X_test = test_df[feature_cols]
        
        # Fill NAs with 0 (since CatBoost can handle it, but for SHAP it's easier)
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        print(f"        Train: {len(X_train)} | Test: {len(X_test)}")
        
        model = CatBoostClassifier(**model_params)
        model.fit(X_train, y_train, verbose=0)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        
        # Mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        for idx, feat in enumerate(feature_cols):
            drift_records.append({
                "Year": year,
                "Feature": clean_feat_name(feat),
                "RawFeature": feat,
                "MeanAbsSHAP": mean_abs_shap[idx]
            })

    print("\n3. Processing Drift Data and Generating Charts...")
    drift_df = pd.DataFrame(drift_records)
    
    # Identify the Top N features overall to plot
    top_overall = drift_df.groupby("Feature")["MeanAbsSHAP"].mean().nlargest(10).index.tolist()
    
    plot_df = drift_df[drift_df["Feature"].isin(top_overall)]
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=plot_df, 
        x="Year", 
        y="MeanAbsSHAP", 
        hue="Feature", 
        marker="o", 
        linewidth=2.5,
        markersize=8
    )
    plt.title("Longitudinal Causal Drift: Feature Importance Over Time (Filing Horizon)", fontsize=16)
    plt.ylabel("Mean Absolute SHAP Value (Impact on Protest Risk)")
    plt.xlabel("Temporal Cutoff (Evaluation Year)")
    plt.xticks(years)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "causal_shap_temporal_drift_line.png", dpi=300)
    plt.close()
    
    # Generate a bump chart (Rank drift)
    drift_df['Rank'] = drift_df.groupby("Year")["MeanAbsSHAP"].rank(method="dense", ascending=False)
    rank_df = drift_df[drift_df["Feature"].isin(top_overall)]
    
    plt.figure(figsize=(12, 8))
    for feat in top_overall:
        feat_data = rank_df[rank_df["Feature"] == feat]
        plt.plot(feat_data["Year"], feat_data["Rank"], marker='o', linewidth=2.5, label=feat)
    
    plt.gca().invert_yaxis()
    plt.title("Rank Drift: Shift in Feature Dominance Over Time", fontsize=16)
    plt.ylabel("Feature Importance Rank (1 = Most Important)")
    plt.xlabel("Temporal Cutoff")
    plt.xticks(years)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "causal_shap_temporal_drift_bump.png", dpi=300)
    plt.close()

    print(f"Done! Drift analysis plots saved to {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()
