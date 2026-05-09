"""
14_run_spatial_and_interaction_shap.py
Generate Interaction SHAP and Geography of Attribution for the main thesis CatBoost model.
"""

import sys
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as cx
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path to import pipeline schema
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data_io.schema import ROOT_DIR, REGISTRY_DIR, WAREHOUSE_MASTER

ARTIFACTS_DIR = Path(r"c:\Users\dhl\.gemini\antigravity\brain\d3ab3523-14f9-4766-904c-a53779e8e0c8\artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

PRIMARY_SPLIT_ID = "TEMP_OOD_2023_MAIN"
PRIMARY_LABEL_VERSION = "label_v1_reconstructed_threshold_crossing"

def get_excluded_features() -> list[str]:
    return [
        "raw_proposed_land_use",
        "raw_existing_land_use",
        "filing_year",
        "council_district",
        "total_buffer_sqft",
        "signer_sqft",
        "exact_geometric_petition_pct"
    ]

def main():
    print("1. Loading Pipeline Registries and Features...")
    labels = pd.read_parquet(REGISTRY_DIR / "label_registry.parquet")
    splits = pd.read_parquet(REGISTRY_DIR / "split_registry.parquet")
    features = pd.read_parquet(ROOT_DIR / "data" / "interim" / "stage_c_features_raw.parquet")

    label_slice = labels.loc[labels["label_version"] == PRIMARY_LABEL_VERSION, ["case_id", "threshold_crossed"]].copy()
    label_slice["case_id"] = label_slice["case_id"].astype(str)

    split_slice = splits.loc[splits["split_id"] == PRIMARY_SPLIT_ID, ["case_id", "role"]].copy()
    split_slice["case_id"] = split_slice["case_id"].astype(str)

    features["case_id"] = features["case_id"].astype(str)

    df = label_slice.merge(split_slice, on="case_id", how="inner").merge(features, on="case_id", how="inner")

    # Exclusions
    excluded = get_excluded_features()
    feature_cols = [c for c in features.columns if c != "case_id" and c not in excluded]

    # Explicitly remove object/string columns that cannot be passed as floats
    for c in list(feature_cols):
        if features[c].dtype == "object" or pd.api.types.is_string_dtype(features[c]):
            feature_cols.remove(c)

    train_df = df[df["role"] == "train"]
    test_df = df[df["role"] == "test"]

    X_train = train_df[feature_cols]
    y_train = train_df["threshold_crossed"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["threshold_crossed"].astype(int)

    print(f"   Train cases: {len(X_train)} | Test OOD cases: {len(X_test)}")

    print("\n2. Training Canonical CatBoost Model on GPU...")
    import catboost
    model = catboost.CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=0,
        task_type="GPU",
    )
    model.fit(X_train, y_train)

    print("\n3. Computing SHAP Interaction Values on OOD Test Set...")
    # Interactions are expensive, computing on test set (OOD)
    explainer = shap.TreeExplainer(model)
    # SHAP interaction values scale O(N * M^2), so we just use test set
    interaction_values = explainer.shap_interaction_values(X_test)
    
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
    
    # Apply mapping to X_test for the summary plot
    X_test_renamed = X_test.rename(columns=lambda x: feature_mapping.get(x, x.replace("district_", "").replace("_lag_6yr", "").replace("_", " ").title()))
    
    # Save a basic SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(explainer.shap_values(X_test), X_test_renamed, max_display=15, show=False)
    plt.title("OOD SHAP Attribution (Stage C Baseline)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "causal_shap_summary_baseline.png", dpi=300)
    plt.close()

    print("\n4. Generating Semantic Interaction Heatmap...")
    # Simplify interaction matrix for top features
    mean_abs_interactions = np.abs(interaction_values).mean(axis=0)
    
    # Get top 15 interacting features
    np.fill_diagonal(mean_abs_interactions, 0) # ignore main effects for sorting interactions
    top_indices = np.argsort(mean_abs_interactions.sum(axis=0))[-15:]
    top_features = [feature_cols[i] for i in top_indices]
    
    clean_top_features = [feature_mapping.get(f, f.replace("district_", "").replace("_lag_6yr", "").replace("_", " ").title()) for f in top_features]
    
    top_interaction_matrix = mean_abs_interactions[np.ix_(top_indices, top_indices)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(top_interaction_matrix, xticklabels=clean_top_features, yticklabels=clean_top_features, cmap="magma", annot=False)
    plt.title("SHAP Interaction Intensity (Mean Absolute Value, Top 15 OOD Features)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "causal_shap_interaction_heatmap.png", dpi=300)
    plt.close()

    print("\n5. Computing Geography of Attribution...")
    print("   Loading Master Enriched Warehouse Data for coordinates...")
    master = pd.read_csv(WAREHOUSE_MASTER)
    master["case_id"] = master["case_number"].astype(str)
    
    # Calculate main SHAP values for all OOD cases
    shap_vals = explainer.shap_values(X_test)
    
    # Sum of absolute SHAP per case (total predictive friction/push)
    test_df["total_abs_shap"] = np.abs(shap_vals).sum(axis=1)
    # Get directional sum (is the model pushing probability up or down?)
    test_df["directional_shap"] = shap_vals.sum(axis=1)
    
    geo_df = test_df[["case_id", "total_abs_shap", "directional_shap"]].merge(
        master[["case_id", "latitude", "longitude", "acs2_median_household_income"]], 
        on="case_id", 
        how="inner"
    )
    
    geo_df = geo_df.dropna(subset=["latitude", "longitude"])
    
    gdf = gpd.GeoDataFrame(
        geo_df, 
        geometry=gpd.points_from_xy(geo_df.longitude, geo_df.latitude),
        crs="EPSG:4326"
    )
    
    # Project to Web Mercator for Contextily
    gdf = gdf.to_crs(epsg=3857)

    print("   Downloading Travis County Census Tract Geometries...")
    try:
        tracts = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2021/TRACT/tl_2021_48_tract.zip")
        travis_tracts = tracts[tracts['COUNTYFP'] == '453']
        travis_tracts = travis_tracts.to_crs(epsg=3857)
        
        # Spatial join to aggregate income onto the tracts for cases in our dataset
        joined = gpd.sjoin(travis_tracts, gdf, how='inner', predicate='contains')
        tract_income = joined.groupby('GEOID')['acs2_median_household_income'].median().reset_index()
        travis_tracts = travis_tracts.merge(tract_income, on='GEOID', how='left')
        
    except Exception as e:
        print(f"   Warning: Failed to load Census tracts online ({e}). Proceeding without solid boundaries.")
        travis_tracts = None

    print("   Plotting Geographical SHAP and Demographic Comparisons...")
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # Left Panel: Single Map with Solid Income Tracts overlaid by SHAP dots
    ax1 = axes[0]
    if travis_tracts is not None:
        travis_tracts.plot(
            ax=ax1, 
            column="acs2_median_household_income", 
            cmap="Blues", 
            alpha=0.5, 
            edgecolor="white", 
            linewidth=0.5,
            missing_kwds={'color': 'lightgrey'}
        )
        
    # Overlay SHAP dots colored by Directional SHAP and sized by Total Magnitude
    scatter = ax1.scatter(
        gdf.geometry.x, 
        gdf.geometry.y, 
        c=gdf["directional_shap"], 
        cmap="coolwarm", 
        s=gdf["total_abs_shap"] * 120, 
        alpha=0.8, 
        edgecolors="black", 
        linewidth=0.5
    )
    
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.5)
    cbar.set_label("Directional SHAP (Blue = Decrease Risk, Red = Increase Risk)")
    
    cx.add_basemap(ax1, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
    ax1.set_title("Overlay: ACS Median Income Boundaries + SHAP Risk Intensity", fontsize=16)
    ax1.axis('off')

    # Right Panel: SHAP Heatmap (Kernel Density of SHAP Attribution Magnitude)
    ax2 = axes[1]
    
    sns.kdeplot(
        x=gdf.geometry.x, 
        y=gdf.geometry.y, 
        weights=gdf["total_abs_shap"], 
        fill=True, 
        cmap="YlOrRd", 
        alpha=0.6, 
        ax=ax2,
        levels=15
    )
    gdf.plot(
        ax=ax2, 
        color="black", 
        markersize=10, 
        alpha=0.3
    )
    
    cx.add_basemap(ax2, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.DarkMatter)
    ax2.set_title("Spatial KDE Heatmap of Absolute SHAP Signal Strength", fontsize=16)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "causal_spatial_intensity_2023.png", dpi=300)
    plt.close()

    print("\n6. Generating SHAP Dependence Plots (Attribution vs Raw Value)...")
    # SHAP Dependence plots show the raw feature value on the x-axis, 
    # the SHAP attribution on the y-axis, and color by an interacting feature.
    
    features_to_plot = [
        ("district_protest_rate_lag_6yr", "acs2_median_household_income"),
        ("district_delta_max_height_ft_lag_6yr", "district_bisg_white_nbr_lag_6yr"),
        ("acs2_median_household_income", "district_renter_share_lag_6yr")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (feat, interaction_feat) in enumerate(features_to_plot):
        if feat in X_test.columns and interaction_feat in X_test.columns:
            # We use the raw explainer.shap_values(X_test) for this, which returns a matrix
            shap.dependence_plot(
                feat, 
                shap_vals, 
                X_test, 
                display_features=X_test_renamed,
                interaction_index=interaction_feat,
                ax=axes[i],
                show=False
            )
            # Add a title
            human_feat = feature_mapping.get(feat, feat)
            human_int = feature_mapping.get(interaction_feat, interaction_feat)
            axes[i].set_title(f"Dependence: {human_feat}\ncolored by {human_int}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "causal_shap_dependence_demographics.png", dpi=300)
    plt.close()

    # Also rename the other artifacts to bust cache
    import os
    if os.path.exists(ARTIFACTS_DIR / "pipeline_shap_summary.png"):
        os.rename(ARTIFACTS_DIR / "pipeline_shap_summary.png", ARTIFACTS_DIR / "causal_shap_summary_baseline.png")
    if os.path.exists(ARTIFACTS_DIR / "pipeline_shap_interaction_heatmap.png"):
        os.rename(ARTIFACTS_DIR / "pipeline_shap_interaction_heatmap.png", ARTIFACTS_DIR / "causal_shap_interaction_heatmap.png")

    print(f"\nAll artifacts generated in {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()
