import os, sys, pandas as pd, numpy as np, shap
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

# Path Setup
ROOT = r"C:\Users\dhl\data\thesis\thesis"
_scripts_dir = os.path.join(ROOT, 'Analysis', 'Scripts')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from artifact_registry import DATA_WAREHOUSE_DIR

DATA = str(DATA_WAREHOUSE_DIR)
OUTPUT_VAR = 'is_protested'

SEMANTIC_CLUSTERS = {
    'acs_owner_occupied_units': 'Housing Tenure',
    'acs_renter_occupied_units': 'Housing Tenure',
    'acs_total_housing_units': 'Housing Tenure',
    'acs_race_white': 'Demographic Composition',
    'acs_race_hispanic': 'Demographic Composition',
    'acs_race_black': 'Demographic Composition',
    'acs_race_asian': 'Demographic Composition',
    'acs_median_gross_rent': 'Neighborhood Income & Rent',
    'acs_median_household_income': 'Neighborhood Income & Rent',
    'acs_poverty_count': 'Neighborhood Income & Rent',
    'acs_median_home_value': 'Neighborhood Income & Rent',
    'ldb_appraised_val': 'Property Valuation',
    'ldb_market_val': 'Property Valuation',
    'land_market_value': 'Property Valuation',
    'total_market_value': 'Property Valuation',
    'ldb_yr_built': 'Structure Age',
    'year_built': 'Structure Age',
    'year': 'Filing Timeline',
    'ldb_land_acres': 'Parcel Scale',
    'gross_site_area_acres': 'Parcel Scale',
    'deed_acreage': 'Parcel Scale',
    'ldb_lotsize': 'Parcel Scale',
    'ldb_land_use': 'Land Use Classification',
    'lui_land_use': 'Land Use Classification',
    'lui_general_land_use': 'Land Use Classification',
    'protest': 'Historical Protest Activity',
    'spatial_contagion_3yr': 'Historical Protest Activity',
    'spatial_contagion_1yr': 'Historical Protest Activity',
    'ldb_far': 'Zoning Density',
    'ldb_units': 'Zoning Density',
    'ldb_imprv_sqft': 'Improvement Scale',
}

def main():
    print("\n" + "="*70)
    print(" SEMANTIC ABLATION VALIDATION (PRE vs POST CLUSTERING)")
    print("="*70)

    df = pd.read_csv(os.path.join(DATA, "H0_Filing_Master_Enriched.csv"), low_memory=False)
    df = df.dropna(subset=['year']).copy()
    df[OUTPUT_VAR] = df['is_protested'].fillna(0).astype(int)

    # 1. Prepare Full Stack Data
    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'TCAD ID', 'standardized_tcad_id', 'date', 'application_start_date', 'final_date']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
    features_raw = list(X_raw.columns)
    y = df[OUTPUT_VAR]

    # 2. Prepare Conceptual Stack Data (Ablation)
    print("[+] Building Conceptual Stack (Pre-Clustered)...")
    X_concept = pd.DataFrame(index=X_raw.index)
    
    # Invert semantic map
    groups = {}
    for f, g in SEMANTIC_CLUSTERS.items():
        if f in features_raw:
            groups.setdefault(g, []).append(f)
    
    scaler = StandardScaler()
    for gname, gfeats in groups.items():
        # Clean features first
        valid_feats = X_raw[gfeats].select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        if valid_feats.empty: continue
        
        # Fill NaNs with mean before scaling
        valid_feats = valid_feats.fillna(valid_feats.mean())
        
        # Scale and aggregate
        try:
            scaled = scaler.fit_transform(valid_feats)
            X_concept[gname] = np.mean(scaled, axis=1)
        except Exception as e:
            print(f"      [!] Skipping {gname}: {e}")
            continue
    
    # Catch any leftovers as individual features
    leftovers = [f for f in features_raw if f not in SEMANTIC_CLUSTERS]
    for f in leftovers:
        if X_raw[f].nunique() > 1:
            X_concept[f] = X_raw[f].fillna(X_raw[f].mean())
            if len(X_concept.columns) > 50: break # limit to avoid noise

    # Remove constant columns in final set
    X_concept = X_concept.loc[:, X_concept.nunique() > 1]
    X_raw = X_raw.loc[:, X_raw.nunique() > 1]
    features_raw = list(X_raw.columns)

    print(f"[+] Final Feature Counts: Raw={len(X_raw.columns)}, Concept={len(X_concept.columns)}")

    # 3. Train Both Models
    print("[+] Training Full Stack Model (Raw Features)...")
    model_raw = CatBoostClassifier(iterations=200, depth=6, verbose=0, random_seed=42).fit(X_raw, y)
    
    print("[+] Training Conceptual Model (Ablation)...")
    model_concept = CatBoostClassifier(iterations=200, depth=6, verbose=0, random_seed=42).fit(X_concept, y)

    # 4. Extract SHAP
    print("[+] Extracting SHAP Attributions...")
    explainer_raw = shap.TreeExplainer(model_raw)
    sv_raw = explainer_raw.shap_values(X_raw)
    if isinstance(sv_raw, list): sv_raw = sv_raw[1] if len(sv_raw)>1 else sv_raw[0]
    
    explainer_concept = shap.TreeExplainer(model_concept)
    sv_concept = explainer_concept.shap_values(X_concept)
    if isinstance(sv_concept, list): sv_concept = sv_concept[1] if len(sv_concept)>1 else sv_concept[0]

    # 5. Aggregate Raw SVs into Concept Buckets (Post-Hoc)
    raw_sums = {}
    total_abs_raw = np.abs(sv_raw).sum()
    for gname, gfeats in groups.items():
        indices = [features_raw.index(f) for f in gfeats if f in features_raw]
        if not indices: continue
        raw_sums[gname] = (np.abs(sv_raw[:, indices]).sum() / total_abs_raw) * 100
    
    # 6. Extract Concept SVs (Pre-Clustered)
    concept_sums = {}
    total_abs_concept = np.abs(sv_concept).sum()
    for i, gname in enumerate(X_concept.columns):
        if gname in groups:
            concept_sums[gname] = (np.abs(sv_concept[:, i]).sum() / total_abs_concept) * 100

    # 7. Comparison Table
    print("\n" + "-"*65)
    print(f"{'Semantic Concept':<30} {'Full Stack %':>15} {'Ablation %':>15}")
    print("-"*65)
    
    v1_list, v2_list = [], []
    for g in sorted(groups.keys()):
        v1 = raw_sums.get(g, 0)
        v2 = concept_sums.get(g, 0)
        print(f"{g:<30} {v1:>15.1f}% {v2:>15.1f}%")
        v1_list.append(v1)
        v2_list.append(v2)

    rho, p = spearmanr(v1_list, v2_list)
    print("-"*65)
    print(f"ALIGNMENT SCORE (Spearman rho): {rho:.3f} (p={p:.4f})")
    
    accuracy_raw = model_raw.score(X_raw, y)
    accuracy_concept = model_concept.score(X_concept, y)
    print(f"Accuracy Loss (Raw vs Concept): {accuracy_raw:.3f} -> {accuracy_concept:.3f}")
    print("="*70)

if __name__ == "__main__":
    main()
