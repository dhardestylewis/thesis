
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from pathlib import Path

# --- Configuration ---
ROOT = Path(r"c:\Users\dhl\data\Thesis\thesis")
FIG_DIR = ROOT / "Thesis_Draft" / "GSAPP_Final_Submission" / "Figures" / "exhibits"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 18-Cluster Semantic Taxonomy
SEMANTIC_CLUSTERS = {
    # Property Value
    "market_value": "Property Value", "land_market_value": "Property Value", 
    "improvement_market_value": "Property Value", "appraised_value": "Property Value",
    "total_market_value": "Property Value", "ldb_appraised_val": "Property Value",
    
    # Parcel Size
    "land_acres": "Parcel Size", "shape_area": "Parcel Size", "improvement_sq_ft": "Parcel Size", 
    "land_to_building_ratio": "Parcel Size", "improvement_ratio": "Parcel Size",
    "ldb_land_acres": "Parcel Size", "ldb_lotsize": "Parcel Size", "deed_acreage": "Parcel Size",
    "gross_site_area_acres": "Parcel Size",
    
    # Building Age
    "yr_built": "Building Age", "building_age": "Building Age",
    "ldb_yr_built": "Building Age", "year_built": "Building Age",
    
    # Land Use Type
    "land_use_code": "Land Use Type", "exemption_flag_hs": "Land Use Type", "homesite_flag": "Land Use Type",
    
    # Time in Review
    "bw_sin": "Time in Review", "bw_cos": "Time in Review", "period_seq": "Time in Review",
    
    # Racial Composition
    "race_white": "Racial Composition", "race_black": "Racial Composition", 
    "race_hispanic": "Racial Composition", "total_population": "Racial Composition", 
    "acs_race_white": "Racial Composition", "acs_race_hispanic": "Racial Composition",
    "acs_race_black": "Racial Composition", "acs_race_asian": "Racial Composition",
    
    # Housing Tenure
    "owner_share": "Housing Tenure", "renter_share": "Housing Tenure",
    "acs_owner_occupied_units": "Housing Tenure", "acs_renter_occupied_units": "Housing Tenure",
    
    # Income & Rent
    "median_household_income": "Income & Rent", "median_gross_rent": "Income & Rent", 
    "rent_burden": "Income & Rent", "affordability_proxy": "Income & Rent",
    "acs_median_household_income": "Income & Rent", "acs_poverty_count": "Income & Rent",
    
    # Prior Petition Activity
    "knn_petition_rate_1km": "Prior Petition Activity", "dist_petition_rate_lag1": "Prior Petition Activity",
    
    # Signer Proximity
    "cumulative_min_signer_dist": "Signer Proximity", "cumulative_max_signer_dist": "Signer Proximity", 
    "cumulative_median_signer_dist": "Signer Proximity", "cumulative_signers_within_200ft": "Signer Proximity", 
    "cumulative_signers_outside_200ft": "Signer Proximity",
    
    # Protest Intensity
    "cumulative_unofficial_protest_intensity": "Protest Intensity", 
    "cumulative_delta_protesting_friction": "Protest Intensity", "cumulative_delta_silent_friction": "Protest Intensity",
    
    # Opposition by Land Use
    "cumulative_protesting_pct_single_family": "Opposition by Land Use", "cumulative_silent_pct_single_family": "Opposition by Land Use",
    "cumulative_protesting_pct_commercial": "Opposition by Land Use", "cumulative_silent_pct_commercial": "Opposition by Land Use",
    "cumulative_protesting_pct_multifamily": "Opposition by Land Use", "cumulative_silent_pct_multifamily": "Opposition by Land Use",
    
    # Zoning Density
    "ldb_far": "Zoning Density", "ldb_units": "Zoning Density", "pdf_requested_max_far": "Zoning Density",
    
    # Improvement Scale
    "ldb_imprv_sqft": "Improvement Scale", "pdf_proposed_height_ft": "Improvement Scale",
    
    # Mortgage Rate
    "mortgage_rate_30yr": "Mortgage Rate", "mortgage_rate_30yr_momentum": "Mortgage Rate",
    
    # Capital Markets
    "treasury_10yr_yield": "Capital Markets", "fed_funds_rate": "Capital Markets",
    
    # Local Labor Market
    "local_unemployment_rate": "Local Labor Market"
}

def plot_beeswarm_main_effects(shap_matrix, X_display, title, filename):
    plt.figure(figsize=(10, 8))
    # Note: shap_matrix should be (N, M)
    shap.summary_plot(shap_matrix, X_display, max_display=15, show=False, plot_size=None)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved {filename}")

def run_forecasting_interaction():
    print("--- Generating Forecasting Interaction SHAP (CatBoost) ---")
    model_path = ROOT / "Analysis" / "Output" / "Track1_Predictive" / "Models" / "stage_c_model_H0.joblib"
    data_path = ROOT / "Data" / "Warehouse_As_Of" / "canonical" / "H0_Filing_Master_Enriched_v2.csv"
    
    if not model_path.exists():
        print(f"  [!] Model not found: {model_path}")
        return

    model = joblib.load(model_path)
    df = pd.read_csv(data_path, low_memory=False)
    
    # Data Cleaning (matching drift script)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    X_raw = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    drop_cols = ['year', 'is_protested', 'case_number', 'reconstructed_petition_share', 'area_pct']
    X = X_raw.drop(columns=[c for c in drop_cols if c in X_raw.columns])
    X = X.fillna(0)
    
    # Subsample for speed
    X_sample = X.sample(n=min(600, len(X)), random_state=42)
    
    explainer = shap.TreeExplainer(model)
    print("  [*] Computing Interaction Values...")
    # This returns (N, M, M)
    interaction_values = explainer.shap_interaction_values(X_sample)
    
    if isinstance(interaction_values, list):
        interaction_values = interaction_values[1] # Positive class
        
    # Extract Main Effects (diagonal)
    main_effects = np.diagonal(interaction_values, axis1=1, axis2=2)
    
    # Aggregate to clusters
    features = X_sample.columns
    cluster_data = {}
    cluster_shap = {}
    
    for i, feat in enumerate(features):
        cluster = SEMANTIC_CLUSTERS.get(feat, "Other")
        if cluster not in cluster_data:
            cluster_data[cluster] = []
            cluster_shap[cluster] = []
        cluster_data[cluster].append(X_sample[feat])
        cluster_shap[cluster].append(main_effects[:, i])
        
    final_X = pd.DataFrame({k: pd.concat(v, axis=1).mean(axis=1) for k, v in cluster_data.items()})
    final_shap = np.column_stack([np.sum(v, axis=0) for v in cluster_shap.values()])
    final_X.columns = list(cluster_data.keys())
    
    plot_beeswarm_main_effects(
        final_shap, final_X, 
        "Interaction TreeSHAP: Forecasting Main Effects (Filing Date)",
        "fig_ch4_14_forecasting_interaction_shap.pdf"
    )

def run_causal_interaction():
    print("--- Generating Causal DML Interaction SHAP (CausalForestDML) ---")
    model_path = ROOT / "Data" / "Zoning_Cases" / "causal_models_production.pkl"
    if not model_path.exists():
        print(f"  [!] Causal model not found: {model_path}")
        return
        
    m_dict = joblib.load(model_path)
    cf = m_dict['cf_joint']
    hurdle = m_dict['hurdle_model']
    features = m_dict['features'] # Includes 'P_withdraw'
    ex_ante = [f for f in features if f != 'P_withdraw']
    
    # Load the specific DML panel
    panel_path = ROOT / "Data" / "Panel" / "cross_sectional_dml_panel.csv"
    df = pd.read_csv(panel_path, low_memory=False)
    
    # Generate P_withdraw using the hurdle model
    X_ex_ante = df[ex_ante].fillna(0).values
    df['P_withdraw'] = hurdle.predict_proba(X_ex_ante)[:, 1]
    
    X = df[features].fillna(0)
    X_sample = X.sample(n=min(50, len(X)), random_state=42)
    
    # Explaining the constant_marginal_effect (the 'treatment effect' surface)
    print("  [*] Computing Causal SHAP Values (Constant Marginal Effect, N=50)...")
    
    def effect_fn(X_in):
        return cf.const_marginal_effect(X_in)[:, 0]

    explainer = shap.Explainer(effect_fn, X_sample, max_evals=100)
    shap_values = explainer(X_sample)
    
    # Rename features to clusters
    X_renamed = X_sample.copy()
    X_renamed.columns = [SEMANTIC_CLUSTERS.get(c, c) for c in X_renamed.columns]
    
    # Aggregate SHAP by cluster
    unique_clusters = list(set(X_renamed.columns))
    agg_shap = np.zeros((len(X_sample), len(unique_clusters)))
    agg_X = pd.DataFrame(index=X_sample.index, columns=unique_clusters)
    
    for i, cluster in enumerate(unique_clusters):
        cols = [j for j, c in enumerate(X_renamed.columns) if c == cluster]
        agg_shap[:, i] = shap_values.values[:, cols].sum(axis=1)
        agg_X[cluster] = X_sample.iloc[:, cols].mean(axis=1)
        
    plot_beeswarm_main_effects(
        agg_shap, agg_X,
        "Causal TreeSHAP: Treatment Effect Heterogeneity (DML)",
        "fig_ch5_14_causal_dml_interaction_shap.pdf"
    )

if __name__ == "__main__":
    run_causal_interaction()
