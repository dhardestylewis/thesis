import os, sys, pandas as pd, numpy as np, shap
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

# Path Setup
ROOT = r"C:\Users\dhl\data\thesis\thesis"
_scripts_dir = os.path.join(ROOT, 'Analysis', 'Scripts')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from artifact_registry import DATA_WAREHOUSE_DIR

DATA = str(DATA_WAREHOUSE_DIR)

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

def _rename_feature(name):
    LABELS = {'gross_site_area_acres': 'Site Area', 'ldb_land_acres': 'Land Area',
              'deed_acreage': 'Deed Acreage', 'ldb_yr_built': 'Year Built',
              'ldb_ilr': 'Improvement Ratio', 'ldb_far': 'Floor Area Ratio',
              'ldb_units': 'Unit Count', 'ldb_appraised_val': 'Appraised Value',
              'ldb_market_val': 'Market Value', 'acs_median_household_income': 'Median Income',
              'acs_owner_occupied_units': 'Owner-Occupied Units', 'acs_race_white': 'White Population',
              'protest': 'Historical Protest', 'spatial_contagion_3yr': 'Nearby Protests (3yr)'}
    if name in LABELS: return LABELS[name]
    cleaned = name
    for prefix in ('acs_', 'ldb_', 'lui_', 'delta_'):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return cleaned.replace('_', ' ').title()

def _cluster_features(X, features, threshold=0.30):
    corr = X[features].corr(method='spearman').abs().clip(0, 1).fillna(0)
    corr_vals = corr.values.copy()
    np.fill_diagonal(corr_vals, 1.0)
    dist = np.clip((1.0 - corr_vals + (1.0 - corr_vals).T) / 2, 0, None)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=threshold, criterion='distance')
    return labels

def _get_cluster_names(features, labels, shap_matrix):
    clusters = {}
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        feats = [features[i] for i in idx]
        cluster_shap = np.abs(shap_matrix[:, idx]).mean(axis=0)
        top_feat = feats[np.argmax(cluster_shap)]
        name = SEMANTIC_CLUSTERS.get(top_feat, _rename_feature(top_feat) if len(feats)==1 else f"{_rename_feature(top_feat)} Cluster")
        if name in clusters: clusters[name].extend(idx.tolist())
        else: clusters[name] = idx.tolist()
    return clusters, labels

def main():
    print("\n" + "="*70)
    print(" MULTI-HORIZON IDENTICAL-TARGET ATTRIBUTION TEST (Target=2023)")
    print("="*70)

    df = pd.read_csv(os.path.join(DATA, "H0_Filing_Master_Enriched.csv"), low_memory=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year']).copy()
    df['is_protested'] = df['is_protested'].fillna(0).astype(int)

    drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'has_audio_record', 
                 'TCAD ID', 'date', 'application_start_date', 'final_date', 'standardized_tcad_id', 
                 'Prob_H=4', 'Prob_LGBM_H=4', 'Prob_CB_H=4', 'Prob_Optimal_H=4', 'ipw', 
                 'council_district', 'council_district_x']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    leak_cols = [c for c in df_clean.columns if c.startswith('tfidf_') or c.startswith('speech_')]
    if leak_cols: df_clean = df_clean.drop(columns=leak_cols)

    X_all = df_clean.select_dtypes(include=[np.number])
    y_all = df['is_protested']
    features = list(X_all.columns)

    print("  [+] Computing Reference Structure (Leakage-Free Pre-2019)...")
    pre_2019_mask = df['year'] < 2019
    ref_labels = _cluster_features(X_all[pre_2019_mask], features)
    ref_cb = CatBoostClassifier(iterations=50, depth=4, verbose=0, random_seed=42).fit(X_all[pre_2019_mask], y_all[pre_2019_mask])
    ref_sv = shap.TreeExplainer(ref_cb).shap_values(X_all[pre_2019_mask])
    if isinstance(ref_sv, list): ref_sv = ref_sv[1] if len(ref_sv)>1 else ref_sv[0]
    global_cluster_map, _ = _get_cluster_names(features, ref_labels, ref_sv)

    TARGET_YEAR = 2023
    te_mask = df['year'] == TARGET_YEAR
    X_test = X_all[te_mask]
    n_shap = min(1000, len(X_test))
    X_shap = X_test.sample(n=n_shap, random_state=42) if len(X_test) > n_shap else X_test

    base_model = CalibratedClassifierCV(
        estimator=CatBoostClassifier(iterations=150, depth=6, l2_leaf_reg=3, learning_rate=0.03, verbose=0, random_seed=42, auto_class_weights='Balanced'),
        method='sigmoid', cv=5
    )

    origins = [2018, 2019, 2020, 2021, 2022]
    all_shares = {}
    
    for origin in origins:
        horizon = TARGET_YEAR - origin
        tr_mask = df['year'] <= origin
        print(f"\n  Training Origin: <= {origin} (Horizon to 2023: {horizon} years) | N={tr_mask.sum()}")
        
        cb = CalibratedClassifierCV(
            estimator=CatBoostClassifier(iterations=150, depth=6, l2_leaf_reg=3, learning_rate=0.03, verbose=0, random_seed=42, auto_class_weights='Balanced'),
            method='sigmoid', cv=5
        )
        cb.fit(X_all[tr_mask], y_all[tr_mask])
        base_cb = cb.calibrated_classifiers_[0].estimator
        
        explainer = shap.TreeExplainer(base_cb)
        sv = explainer.shap_values(X_shap)
        if isinstance(sv, list): sv = sv[1] if len(sv)>1 else sv[0]
        if hasattr(sv, 'values'): sv = sv.values
        
        total_abs_shap = np.abs(sv).sum()
        shares = {}
        for gname, gidx in global_cluster_map.items():
            shares[gname] = (np.abs(sv[:, gidx]).sum() / total_abs_shap) * 100
            
        all_shares[origin] = shares
        
        # Print top 5
        for g, s in sorted(shares.items(), key=lambda x: -x[1])[:5]:
            print(f"    {g:<30} {s:>5.1f}%")

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    print("\n" + "="*70)
    print(" SUMMARY: RANK STABILITY OF 2023 PREDICTORS ACROSS DIFFERENT ORIGINS")
    print("="*70)
    ref_shares = all_shares[2022] # 1-year horizon is the "ground truth"
    
    results = []
    for origin in origins[:-1]:
        shares = all_shares[origin]
        common = sorted(ref_shares.keys())
        v1 = [ref_shares[k] for k in common]
        v2 = [shares[k] for k in common]
        rho, p = spearmanr(v1, v2)
        horizon = TARGET_YEAR - origin
        print(f"  Horizon={horizon}yr (Train<={origin}) vs Horizon=1yr (Train<=2022): rho = {rho:.3f}")
        results.append({'Horizon': horizon, 'Origin': origin, 'Spearman_rho': rho})
    
    # 1yr self-comparison
    results.append({'Horizon': 1, 'Origin': 2022, 'Spearman_rho': 1.0})
    
    df_res = pd.DataFrame(results).sort_values('Horizon', ascending=False)
    
    # Generate Figure
    fig, ax1 = plt.subplots(figsize=(7, 4))
    
    # Plot 1: Rank Correlation Line
    ax1.plot(df_res['Horizon'], df_res['Spearman_rho'], marker='o', linewidth=2, color='#2c3e50', label='Rank Correlation (vs 1yr)')
    ax1.set_xlabel('Forecasting Horizon (Years)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Spearman Rank Correlation ($\\rho$)', fontsize=11, color='#2c3e50', fontweight='bold')
    ax1.set_ylim(0.7, 1.02)
    ax1.tick_params(axis='y', colors='#2c3e50')
    ax1.set_xticks(df_res['Horizon'])
    ax1.invert_xaxis() # 5yr on left, 1yr on right
    
    # Plot 2: Parcel Scale % Bar
    ax2 = ax1.twinx()
    parcel_shares = [all_shares[row['Origin']].get('Parcel Scale', 0) for _, row in df_res.iterrows()]
    bars = ax2.bar(df_res['Horizon'], parcel_shares, width=0.4, alpha=0.3, color='#e74c3c', label='Parcel Scale Share')
    ax2.set_ylabel('Parcel Scale Attribution Share (%)', fontsize=11, color='#c0392b', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', colors='#c0392b')
    ax2.grid(False)
    
    plt.title('Multi-Horizon Attribution Stability (Target Year: 2023)', fontweight='bold', fontsize=12)
    
    # Save Outputs
    output_dir = os.path.join(ROOT, 'Thesis_Draft', 'Draft_v1', 'Figures', 'Track1_Exhibits')
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'fig_multi_horizon_attribution.pdf')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n  [+] Saved Figure: {fig_path}")
    
    metrics_path = os.path.join(ROOT, 'Analysis', 'Output', 'Track1_Predictive', 'Metrics', 'multi_horizon_attribution.csv')
    df_res.to_csv(metrics_path, index=False)
    print(f"  [+] Saved Metrics: {metrics_path}")

if __name__ == "__main__":
    main()
