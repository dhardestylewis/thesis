"""
forecasting_model.py
====================
Priority 4: Forecasting / Modeling Extensions + Local Micro-Variables + SHAP
Constructs baseline prediction models (Random Forest) augmented with 
neighborhood economic/idiosyncratic variables from Panel V3.

Model 1: Predicts development approval 
Model 2: Predicts NIMBY protest occurrence 

Outputs:
  - Analysis/Output/Forecasting/fig12_shap_summary_approval.png
  - Analysis/Output/Forecasting/fig13_shap_summary_protest.png
  - Analysis/Output/Forecasting/fig14_shap_interactions_protest.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import BallTree

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT, "Data")
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Forecasting")
os.makedirs(OUT_DIR, exist_ok=True)

ZONING_CSV = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "enriched_zoning_data_updated.csv")
PET_CSV = os.path.join(DATA, "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv")
PANEL_CSV = os.path.join(DATA, "Panel", "Output", "Property_Year_Panel_v3.csv")

def classify_use(use):
    if pd.isna(use): return 'Unknown'
    use_lower = str(use).lower()
    if 'family' in use_lower or 'residential' in use_lower or 'condo' in use_lower:
        return 'Residential'
    if 'commercial' in use_lower or 'retail' in use_lower or 'office' in use_lower or 'mu' in use_lower:
        return 'Commercial'
    if 'industrial' in use_lower or 'warehouse' in use_lower:
        return 'Industrial'
    return 'Other'

def train_and_explain(X, y, target_name, prefix):
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train Model
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Eval
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)
    
    # SHAP Explainer
    print(f"Generating SHAP plots for {target_name}...")
    explainer = shap.TreeExplainer(rf)
    
    # 1. Main SHAP Values
    shap_values = explainer.shap_values(X_test)
    
    # Random Forest shap_values is a list for multi-class [class_0, class_1]
    if isinstance(shap_values, list):
        shap_vals_target = shap_values[1]
    else:
        shap_vals_target = shap_values
        
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals_target, X_test, show=False)
    plt.title(f"SHAP Values: Drivers of {target_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{prefix}_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. SHAP Interactions (only for protest model, to save time/complexity)
    if "Protest" in target_name:
        try:
            print("Computing SHAP Interaction Tensors... this may take a moment.")
            shap_interaction_values = explainer.shap_interaction_values(X_test)
            if isinstance(shap_interaction_values, list):
                shap_int_target = shap_interaction_values[1]
            else:
                shap_int_target = shap_interaction_values
                
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_int_target, X_test, show=False)
            plt.title(f"SHAP Interactions: Drivers of {target_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{prefix}_interactions.png"), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Skipping exact interactions due to constraint: {e}")
            
    return auc, report

def run_forecasting():
    print("=== PRIORITY 4: FORECASTING MODEL & SHAP ENRICHMENT ===")
    
    zoning = pd.read_csv(ZONING_CSV, low_memory=False)
    pet = pd.read_csv(PET_CSV)
    
    zoning['case_number'] = zoning['Case Number'].fillna(zoning['case_number']).astype(str).str.strip()
    pet['case_number'] = pet['case_number'].astype(str).str.strip()
    
    zoning_dedup = zoning.drop_duplicates(subset=['case_number']).copy()
    zoning_dedup['is_approved'] = zoning_dedup['approval_date'].notna().astype(int)
    zoning_dedup['is_protested'] = zoning_dedup['case_number'].isin(set(pet['case_number'])).astype(int)
    
    zoning_dedup['year'] = zoning_dedup['case_number'].str.extract(r'C\d+[A-Z]*-(\d{4})')[0].astype(float)
    zoning_dedup['gross_site_area_acres'] = pd.to_numeric(zoning_dedup['gross_site_area_acres'], errors='coerce').fillna(0)
    zoning_dedup['group'] = zoning_dedup['proposed_land_use'].apply(classify_use)
    zoning_dedup['council_district'] = zoning_dedup['council_district'].fillna(-1).astype(str)
    
    zoning_dedup = zoning_dedup.dropna(subset=['latitude', 'longitude', 'year'])
    zoning_dedup['lat_rad'] = np.radians(zoning_dedup['latitude'])
    zoning_dedup['lon_rad'] = np.radians(zoning_dedup['longitude'])
    
    # ---------------------------------------------------------
    # FEATURE: SPATIAL CONTAGION (Nearest prior protest)
    # ---------------------------------------------------------
    print("Computing Contagion (Distance to recent protests)...")
    distances = []
    protest_df = zoning_dedup[zoning_dedup['is_protested'] == 1].dropna(subset=['lat_rad', 'lon_rad'])
    
    for idx, row in zoning_dedup.iterrows():
        yr = row['year']
        lat, lon = row['lat_rad'], row['lon_rad']
        
        recent_protests = protest_df[(protest_df['year'] >= yr - 3) & (protest_df['year'] < yr)]
        if len(recent_protests) > 0:
            tree = BallTree(recent_protests[['lat_rad', 'lon_rad']].values, metric='haversine')
            dist, _ = tree.query([[lat, lon]], k=1)
            dist_miles = dist[0][0] * 3958.8
            distances.append(dist_miles)
        else:
            distances.append(15.0) # max city dist
    zoning_dedup['dist_recent_protest_miles'] = distances
    
    # ---------------------------------------------------------
    # FEATURE: MICRO-VARIABLES (Y-1 Panel Buffering)
    # ---------------------------------------------------------
    print("Loading Panel V3 to compute Localized Micro-Variables (Y-1)...")
    panel_cols = ['standardized_tcad_id', 'latitude', 'longitude', 'year', 'appraised_value', 'lui_general_land_use']
    panel = pd.read_csv(PANEL_CSV, usecols=panel_cols, low_memory=False)
    panel = panel.dropna(subset=['latitude', 'longitude', 'year'])
    
    # Create unique TCAD points for spatial lookup
    tcad_unique = panel.drop_duplicates(subset=['standardized_tcad_id']).copy()
    tcad_unique['lat_rad'] = np.radians(tcad_unique['latitude'])
    tcad_unique['lon_rad'] = np.radians(tcad_unique['longitude'])
    
    print("Building TCAD Spatial BallTree...")
    tcad_tree = BallTree(tcad_unique[['lat_rad', 'lon_rad']].values, metric='haversine')
    
    # radius in rads = miles / 3958.8 (using 0.1 miles roughly ~500 ft for neighborhood)
    radius_rad = 0.1 / 3958.8
    points = zoning_dedup[['lat_rad', 'lon_rad']].values
    
    print("Querying neighborhood clusters...")
    indices_list = tcad_tree.query_radius(points, r=radius_rad)
    
    # Extract IDs corresponding to indices
    all_tcads = tcad_unique['standardized_tcad_id'].values
    
    # Dictionaries to hold new features
    med_values = []
    sf_pcts = []
    num_neighbors = []
    
    print("Aggregating local Y-1 panel data...")
    # Pre-index Panel by strictly [TCAD, Year] to make lookups fast
    panel['year'] = panel['year'].astype(int)
    panel_keyed = panel.set_index(['standardized_tcad_id', 'year'])
    
    for i, (idx, row) in enumerate(zoning_dedup.iterrows()):
        yr = int(row['year'])
        y_minus_1 = yr - 1
        
        # Get TCADs in radius
        neighbors_idx = indices_list[i]
        tcads_in_radius = all_tcads[neighbors_idx]
        
        # We need the values of these TCADs at year Y-1
        keys = [(t, y_minus_1) for t in tcads_in_radius]
        
        # Filter for keys genuinely in the panel
        valid_keys = [k for k in keys if k in panel_keyed.index]
        if not valid_keys:
            med_values.append(np.nan)
            sf_pcts.append(np.nan)
            num_neighbors.append(0)
            continue
            
        subset = panel_keyed.loc[valid_keys]
        
        # Metrics
        vals = subset['appraised_value'].dropna()
        med_val = vals.median() if len(vals) > 0 else np.nan
        
        # Single Family density (LUI code 100.0)
        lucs = subset['lui_general_land_use'].dropna()
        is_sf = (lucs == 100.0).sum()
        sf_pct = (is_sf / len(lucs) * 100) if len(lucs) > 0 else 0
        
        med_values.append(med_val)
        sf_pcts.append(sf_pct)
        num_neighbors.append(len(subset))
        
    zoning_dedup['median_neighbor_wealth_Y1'] = med_values
    zoning_dedup['pct_neighbors_single_family_Y1'] = sf_pcts
    zoning_dedup['neighbor_density'] = num_neighbors
    
    # Fill missing Y-1 values with city-wide medians (robust imputation)
    zoning_dedup['median_neighbor_wealth_Y1'] = zoning_dedup['median_neighbor_wealth_Y1'].fillna(zoning_dedup['median_neighbor_wealth_Y1'].median())
    zoning_dedup['pct_neighbors_single_family_Y1'] = zoning_dedup['pct_neighbors_single_family_Y1'].fillna(zoning_dedup['pct_neighbors_single_family_Y1'].median())
    
    # ---------------------------------------------------------
    # MODELING & SHAP
    # ---------------------------------------------------------
    features = [
        'gross_site_area_acres', 'year', 'dist_recent_protest_miles',
        'median_neighbor_wealth_Y1', 'pct_neighbors_single_family_Y1', 'neighbor_density'
    ]
    cat_features = ['group', 'council_district']
    
    model_df = zoning_dedup[features + cat_features + ['is_approved', 'is_protested']].dropna()
    model_df = pd.get_dummies(model_df, columns=cat_features, drop_first=True)
    
    # To prevent special characters in column names destroying tree models
    model_df.columns = [str(c).replace('[', '').replace(']', '').replace('<', '') for c in model_df.columns]
    
    with open(os.path.join(OUT_DIR, "model_metrics.txt"), "w") as f:
        f.write("=== ZONING APPROVAL MODEL (SHAP) ===\n")
        
        X_app = model_df.drop(['is_approved'], axis=1) # is_protested is a feature
        y_app = model_df['is_approved']
        auc_app, rep_app = train_and_explain(X_app, y_app, "Zoning Approval", "fig12_shap_approval")
        
        f.write(f"Random Forest AUC: {auc_app:.3f}\n")
        f.write(rep_app + "\n\n")
        
        f.write("=== NIMBY PROTEST MODEL (SHAP) ===\n")
        X_pro = model_df.drop(['is_protested', 'is_approved'], axis=1) # is_approved removed
        y_pro = model_df['is_protested']
        auc_pro, rep_pro = train_and_explain(X_pro, y_pro, "NIMBY Protest Occurrence", "fig13_shap_protest")
        
        f.write(f"Random Forest AUC: {auc_pro:.3f}\n")
        f.write(rep_pro + "\n")
        
    print("Forecasting suite with SHAP interactions complete.")

if __name__ == "__main__":
    run_forecasting()
